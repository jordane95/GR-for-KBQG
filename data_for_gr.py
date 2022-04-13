import os
import json
import re
import string
import numpy as np
from tqdm import tqdm
import sys
import copy
import random
import time

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from eval_webnlg.pycocotools.coco import COCO
from eval_webnlg.pycocoevalcap.eval import COCOEvalCap


def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out


# Pre-training dataset (wikidata)
class WikidataDataset(Dataset):
    def __init__(self, logger, args, data_path, knowledge_file, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer
        # Load data
        with open(self.data_path + '.json', 'r') as f:
            self.data = json.load(f)

        self.knowledge = knowledge_file

        print("Total samples = {}; Total entities = {}".format(len(self.data), len(self.knowledge)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode
        self.metric = "BLEU"
        self.forbid_duplicate_relation = True
        self.max_fact = 8
        self.max_entity = 12
        self.mask_prob = eval(args.mask_prob)

        # Get the ids for special tokens
        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' [head]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [relation]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [tail]', add_special_tokens=False)
        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)

        if self.args.model_name == "bart":
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token = self.tokenizer.additional_special_tokens[0]
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        if self.args.model_name == "bart":
            if self.args.append_another_bos:
                self.add_bos_id = [self.tokenizer.bos_token_id] * 2
            else:
                self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []

    def __len__(self):
        return len(self.data)

    def rel_mask(self, rel, text_relation, relation_change, p=0.15):
        # Mask relations
        # rel_input: relation ids in the encoder input (corrupted)
        # rel_input_token: relation tokens for rel_input
        # rel_label: relation ids in the encoder output (complete)
        # rel_label_token: relation tokens for rel_label

        rel_label = relation_change[rel][1]
        rel_label_token = copy.deepcopy(rel)

        if random.random() > p:
            rel_input = copy.deepcopy(rel_label)
            rel_input_token = copy.deepcopy(rel_label_token)
        else:
            # Similar to BERT, 80% mask, 10% replace, 10% unchanged
            step_prob = random.random()
            if step_prob < 0.8:
                rel_input = [self.mask_token_id] * len(rel_label)
                rel_input_token = ' '.join([self.mask_token] * len(rel_label))
            else:
                if step_prob < 0.9:
                    replace_rel = random.choice(text_relation)
                    replace_rel_ids = relation_change[replace_rel][1]
                    if len(rel_label) >= len(replace_rel_ids):
                        rel_input = replace_rel_ids + [self.mask_token_id] * (len(rel_label) - len(replace_rel_ids))
                        rel_input_token = ' '.join(
                            [replace_rel] + [self.mask_token] * (len(rel_label) - len(replace_rel_ids)))
                    else:
                        rel_input = replace_rel_ids[:len(rel_label)]
                        rel_input_token = ' '.join(self.tokenizer.convert_ids_to_tokens(rel_input))
                else:
                    rel_input = copy.deepcopy(rel_label)
                    rel_input_token = copy.deepcopy(rel_label_token)

        return rel_label, rel_label_token, rel_input, rel_input_token

    def linearize_v2(self, entity, entity_change, text_relation, head_ids, rel_ids, tail_ids,
                     relation_change, cnt_edge, adj_matrix):
        # Linearize knowledge graphs into sequences, modified based on
        # https://github.com/wenhuchen/KGPT/blob/main/code/DataLoader.py

        # string: encoder input ids (corrupted)
        # string_label: encoder label ids (complete)
        # string_tokens: encoder input tokens
        # string_label_tokens: encoder label tokens
        # nodes: node ids for each token
        # edges: edge ids for each token

        if len(entity[0]) == 0:
            return [], [], '', [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids)
        string = copy.deepcopy(string_label)
        string_tokens = ' [head]'
        string_label_tokens = ' [head]'
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))

        string_label += entity_change[entity[0]][3]
        string_label_tokens += ' {}'.format(entity[0])
        string += entity_change[entity[0]][1]
        string_tokens += ' {}'.format(entity_change[entity[0]][0])
        nodes.extend([entity_change[entity[0]][4]] * len(entity_change[entity[0]][3]))
        edges.extend([-1] * len(entity_change[entity[0]][3]))

        triple_id = [1] * len(string)

        # Deal with the description relation in kgtext
        if len(entity[1]) != 0:
            rel_label, rel_label_token, rel_input, rel_input_token = self.rel_mask('description', text_relation,
                                                                                   relation_change, p=self.mask_prob[2])

            words = rel_ids + rel_input + tail_ids + entity_change[entity[1]][1]
            words_tokens = ' [relation] ' + rel_input_token + ' [tail] ' + entity_change[entity[1]][0]
            words_label = rel_ids + rel_label + tail_ids + entity_change[entity[1]][3]
            words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, entity[1])
            nodes.extend(
                    [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[entity[1]][4]] * len(
                        entity_change[entity[1]][3]))
            edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                        len(tail_ids) + len(entity_change[entity[1]][3])))
            if entity_change[entity[0]][4] < len(adj_matrix) and entity_change[entity[1]][4] < len(adj_matrix):
                adj_matrix[entity_change[entity[0]][4]][entity_change[entity[1]][4]] = cnt_edge

            cnt_edge += 1

            string += words
            string_tokens += words_tokens
            string_label += words_label
            string_label_tokens += words_label_tokens
            triple_id += [triple_id[-1] + 1] * len(words)

        added = set()
        # Deal with other triples in knowledge graphs
        for rel in entity[2]:
            if self.forbid_duplicate_relation and rel[0] in added:
                pass
            else:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    rel_label, rel_label_token, rel_input, rel_input_token = self.rel_mask(rel[0], text_relation,
                                                                                relation_change, p=self.mask_prob[2])

                    words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][3]
                    words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, rel[1])
                    words = rel_ids + rel_input + tail_ids + entity_change[rel[1]][1]
                    words_tokens = ' [relation] ' + rel_input_token + ' [tail] ' + entity_change[rel[1]][0]
                    nodes.extend(
                            [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[rel[1]][4]] * len(
                                entity_change[rel[1]][3]))
                    edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                                len(tail_ids) + len(entity_change[rel[1]][3])))
                    if entity_change[entity[0]][4] < len(adj_matrix) and entity_change[rel[1]][4] < len(adj_matrix):
                        adj_matrix[entity_change[entity[0]][4]][entity_change[rel[1]][4]] = cnt_edge

                    cnt_edge += 1

                    string += words
                    string_tokens += words_tokens
                    string_label += words_label
                    string_label_tokens += words_label_tokens
                    triple_id += [triple_id[-1] + 1] * len(words)
                    added.add(rel[0])

            if len(added) >= self.max_fact:
                break

        assert len(string) == len(string_label) == len(nodes) == len(edges)

        return string, triple_id, string_tokens, string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    def get_all_entities_per_sample(self, mark_entity_number, mark_entity):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = self.knowledge[entity_id]
            if len(entity[0]) == 0:
                continue
            if len(entity[1]) != 0:
                text_entity.add(entity[1])
                text_relation.add('description')

            added = set()
            for rel in entity[2]:
                if self.forbid_duplicate_relation and rel[0] in added:
                    pass
                else:
                    if len(rel[0]) != 0 and len(rel[1]) != 0:
                        text_relation.add(rel[0])
                        text_entity.add(rel[1])
                        added.add(rel[0])

                if len(added) >= self.max_fact:
                    break

        text_entity_list = list(text_entity)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    # Acquire the result of each entity after perturbation
    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        ent_change = {}
        total_entity = mark_entity + text_entity
        # mark entity prob = 0.4 (mask_prob[0])
        mark_entity_mask = random.choices([0.0, 1.0], weights=[self.mask_prob[0], 1.0 - self.mask_prob[0]], k=len(mark_entity))
        # text entity prob = 0.2 (mask_prob[1])
        text_entity_mask = random.choices([0.0, 1.0], weights=[self.mask_prob[1], 1.0 - self.mask_prob[1]], k=len(text_entity))
        total_entity_mask = np.concatenate((mark_entity_mask, text_entity_mask))

        assert len(total_entity_mask) == len(total_entity)

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            if total_entity_mask[ent_id] == 0:
                tmp_prob = random.random()
                if tmp_prob < 0.8:
                    token_str = " ".join([self.mask_token] * len(entity_toks))
                    token_id = [self.mask_token_id] * len(entity_toks)
                    ent_change[total_entity[ent_id]] = [token_str, token_id, total_entity[ent_id], entity_toks, ent_id]
                else:
                    if tmp_prob < 0.9:
                        replace_entity = random.choice(total_entity)
                        replace_entity_toks = self.tokenizer.encode(" {}".format(replace_entity),
                                                                    add_special_tokens=False)
                        if len(entity_toks) > len(replace_entity_toks):
                            token_str = "{} {}".format(replace_entity, " ".join(
                                [self.mask_token] * (len(entity_toks) - len(replace_entity_toks))))
                            token_id = replace_entity_toks + [self.mask_token_id] * (
                                    len(entity_toks) - len(replace_entity_toks))
                            ent_change[total_entity[ent_id]] = [token_str, token_id, total_entity[ent_id], entity_toks,
                                                                ent_id]
                        else:
                            token_id = replace_entity_toks[:len(entity_toks)]
                            token_str = " ".join(self.tokenizer.convert_ids_to_tokens(token_id))
                            ent_change[total_entity[ent_id]] = [token_str, token_id, total_entity[ent_id], entity_toks,
                                                                ent_id]
                    else:
                        ent_change[total_entity[ent_id]] = [total_entity[ent_id], entity_toks, total_entity[ent_id],
                                                            entity_toks, ent_id]
            else:
                ent_change[total_entity[ent_id]] = [total_entity[ent_id], entity_toks, total_entity[ent_id],
                                                    entity_toks, ent_id]

        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = [text_relation[rel_id],
                                                 self.tokenizer.encode(' {}'.format(text_relation[rel_id]), \
                                                                       add_special_tokens=False)]

        return ent_change, rel_change

    # Acquire the masked text
    def text_mask(self, word, word_cnt, p=0.15):
        """Word level masking

        Args:
            word (str): word
            word_cnt (int): word index
            p (float): mask probability
        Returns:
            word_label (List[int]): list of token ids for word
            word_laebl_token (str): same as `word`
            word_input (List[int]): masked token ids
            word_input_token (str): string representation for `word_input`
            word_whole (List[int]): [word_cnt] * len(word_label)
        """
        if len(word) == 0:
            return [], '', [], '', []

        word_label = self.tokenizer.encode(" {}".format(word), add_special_tokens=False) # List[int], token ids
        word_whole = [word_cnt] * len(word_label)
        word_label_token = copy.deepcopy(word)

        if random.random() > p: # no mask with probability of 1-p
            word_input = copy.deepcopy(word_label)
            word_input_token = copy.deepcopy(word_label_token)
        else:
            step_prob = random.random()
            if step_prob < 0.8:
                # p = 0.8
                # mask the whole word, replace it with [MASK] token
                word_input = [self.mask_token_id] * len(word_label)
                word_input_token = ' '.join([self.mask_token] * len(word_label))
            else:
                if step_prob < 0.9:
                    # p = 0.1
                    # replace the first token with a random token in vocab, the following remain [MASK]
                    replace_word = [random.randint(0, self.tokenizer.vocab_size - 1)]
                    replace_word_token = [self.tokenizer.convert_ids_to_tokens(replace_word[0])]
                    # len(replace_word) == 1
                    if len(word_label) >= len(replace_word):
                        word_input = replace_word + [self.mask_token_id] * (len(word_label) - len(replace_word))
                        word_input_token = ' '.join(
                            replace_word_token + [self.mask_token] * (len(word_label) - len(replace_word)))
                    else:
                        word_input = replace_word[:len(word_label)]
                        word_input_token = ' '.join(replace_word_token[:len(word_label)])
                else:
                    # p = 0.1
                    # do not mask
                    word_input = copy.deepcopy(word_label)
                    word_input_token = copy.deepcopy(word_label_token)

        return word_label, word_label_token, word_input, word_input_token, word_whole

    def truncate_pair_ar(self, a, b, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        """

        Args:
            a (List[int]): input linearized graph token ids
            b (List[int]): masked text token ids, as input corrputed text, after postprocessing by removing sucessive [MASK] tokens
            add_bos_ids (List[int]): 0/1/2 bos token ids
            graph_ids (List[int]): token ids for [graph]
            text_ids (List[int]): token ids for [text]
            node_ids (List[int]): node index of the linearized graph
            edge_ids (List[int]): edge index of the linearized graph
        Returns:
            input_ids (List[int]): add_bos_id + graph_ids + a + text_ids + b + eos_token_id
            attn_mask (List[int]):
            input_node_ids (List[int]):
            input_edge_ids (List[int]):
        """
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1 # 1 for eos
        if len(a) + len(b) > length_a_b: # trunction
            a = a[:(length_a_b - len(b))]
            node_ids = node_ids[:(length_a_b - len(b))]
            edge_ids = edge_ids[:(length_a_b - len(b))]
        input_ids = add_bos_id + graph_ids + a + text_ids + b + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + len(b) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + len(b) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    def truncate_pair_ae(self, a, b, a_ori, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) + len(b) > length_a_b:
            a = a[:(length_a_b - len(b))]
            node_ids = node_ids[:(length_a_b - len(b))]
            edge_ids = edge_ids[:(length_a_b - len(b))]
            a_ori = a_ori[:(length_a_b - len(b))]
        input_ids = add_bos_id + graph_ids + a + text_ids + b + [self.tokenizer.eos_token_id]
        input_ids_ori = add_bos_id + graph_ids + a_ori + text_ids + b + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + len(b) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + len(b) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_ids_ori += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids_ori))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == len(input_node_ids) == len(input_edge_ids)
        assert len(input_ids) == len(input_ids_ori) == self.args.max_input_length
        return input_ids, attn_mask, input_ids_ori, input_node_ids, input_edge_ids

    def truncate_pair_ot(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        # empty text
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    # TODO: adapt this for retrieval enhanced generation
    # Prepare data for text reconstruction
    def ar_prep_data(self, answers_input, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        """

        Args:
            answers_input (List[int]): masked text token ids, as input corrputed text
            answers (List[int]): original whole text token ids, as target labels
            questions (List[int]): input linearized graph token ids
            add_bos_ids (List[int]): 0/1/2 bos token ids
            graph_ids (List[int]): token ids for [graph]
            text_ids (List[int]): token ids for [text]
            node_ids (List[int]): node index of the linearized graph
            edge_ids (List[int]): edge index of the linearized graph
        Returns:
            input_ids
            input_attn_mask
            decoder_label_ids
            decoder_attn_mask
            input_node_ids
            input_edge_ids
        """
        # merge mask in answers_input
        text_pertubed_input = []
        for data_id in range(len(answers_input)):
            if len(text_pertubed_input) == 0:
                text_pertubed_input.append(answers_input[data_id])
            else:
                # avoid consecutive [MASK] tokens
                # if encoutered 2 [MASK] tokens, skip to the next
                if answers_input[data_id] != self.mask_token_id:
                    text_pertubed_input.append(answers_input[data_id])
                else:
                    if text_pertubed_input[-1] != self.mask_token_id:
                        text_pertubed_input.append(answers_input[data_id])

        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1: # trunction
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id] # add bos and eos
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids)) # mask
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids)) # padding
        assert len(decoder_label_ids) == self.args.max_output_length # assure that all examples are padded to the same length
        assert len(decoder_label_ids) == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ar(questions,
                                                                                           text_pertubed_input,
                                                                                           add_bos_id, graph_ids,
                                                                                           text_ids, node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids

    # Prepare data for graph reconstruction
    def ae_prep_data(self, questions_input, questions, answers, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):

        input_ids, attn_mask, input_ids_ori, input_node_ids, input_edge_ids = self.truncate_pair_ae(questions_input,
                                                                                                    answers, questions,
                                                                                                    add_bos_id,
                                                                                                    graph_ids, text_ids,
                                                                                                    node_ids, edge_ids)

        encoder_label_ids = [-1 if input_ids[idx] == input_ids_ori[idx] else input_ids_ori[idx] for idx in
                             range(len(input_ids))]

        assert len(encoder_label_ids) == len(input_ids)

        return input_ids, attn_mask, encoder_label_ids, input_node_ids, input_edge_ids

    # Prepare data for embedding alignment
    def ot_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids, word_whole_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
            word_whole_ids = word_whole_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_word_whole_ids = [-1] * len(add_bos_id) + word_whole_ids + [-1]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_word_whole_ids += [-1] * (self.args.max_output_length - len(decoder_word_whole_ids))
        assert len(decoder_label_ids) == self.args.max_output_length
        assert len(decoder_label_ids) == len(decoder_attn_mask) == len(decoder_word_whole_ids)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ot(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids, decoder_word_whole_ids

    def __getitem__(self, idx):

        entry = self.data[idx]

        entities = []
        for _ in entry['kblinks']:
            if _ is not None and _ in self.knowledge and _ not in entities:
                entities.append(_)

        # strings / string tokens: corrupted linearized graph (ids / tokens)
        # strings_label / string_label_tokens: complete linearized graph (ids / tokens)
        # entity_ids: entity index of the corrupted linearized graph
        # node_ids / edge_ids: node / edge index of the corrupted linearized graph

        strings = []
        strings_label = []
        entity_ids = []
        triple_ids = []
        node_ids = []
        edge_ids = []
        strings_tokens = ''
        strings_label_tokens = ''

        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [self.knowledge[ele_entity][0] for ele_entity in entities] + [
            self.knowledge[entry['title_kb_id']][0]]
        mark_entity_number = entities + [entry['title_kb_id']]
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity)
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        total_entity = mark_entity + text_entity
        adj_matrix = [[-1] * (self.args.max_node_length + 1) for _ in range(self.args.max_node_length + 1)]

        cnt_edge = 0

        if 'title' in entry:
            entity = self.knowledge[entry['title_kb_id']]

            string, triple_id, string_tokens, string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = \
                self.linearize_v2(
                entity,
                entity_change,
                text_relation,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings += string
            strings_tokens += string_tokens
            strings_label += string_label
            strings_label_tokens += string_label_tokens
            entity_ids += [0] * len(string)
            triple_ids += triple_id
            node_ids += nodes
            edge_ids += edges

        for i, entity_id in enumerate(entities):
            if i + 1 >= self.max_entity:
                break

            entity = self.knowledge[entity_id]

            string, triple_id, string_tokens, string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                text_relation,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings += string
            strings_tokens += string_tokens
            strings_label += string_label
            strings_label_tokens += string_label_tokens
            entity_ids += [i + 1] * len(string)
            triple_ids += triple_id
            node_ids += nodes
            edge_ids += edges

        position_ids = list(range(len(strings)))
        assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)
        assert len(strings) == len(strings_label) == len(node_ids) == len(edge_ids)

        # words_input_ids / words_input_tokens: corrupted texts (ids / tokens)
        # words_label_ids / words_label_tokens: complete texts (ids / tokens)
        # words_whole_ids: word index

        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens, words_whole_ids = [], '', [], '', []
        # sentence level

        word_cnt = 0
        for word in entry['text']:
            if word not in total_entity:
                # word level
                word_label_ids, word_label_tokens, word_input_ids, word_input_tokens, word_whole_ids = \
                    self.text_mask(word, word_cnt, p=self.mask_prob[4]) # for 'text'
            else:
                word_label_ids, word_label_tokens, word_input_ids, word_input_tokens, word_whole_ids = \
                    self.text_mask(word, word_cnt, p=self.mask_prob[3]) # for 'text(entity)'

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens
            words_input_ids += word_input_ids
            words_input_tokens += ' ' + word_input_tokens
            words_whole_ids += word_whole_ids
            if len(word_label_ids) > 0:
                word_cnt += 1

        assert len(words_input_ids) == len(words_label_ids) == len(words_whole_ids)

        # We focus on ar
        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, input_edge_ids_ar = \
            self.ar_prep_data(words_input_ids, words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)
        
        
        input_ids_ae, attn_mask_ae, encoder_label_ids, input_node_ids_ae, input_edge_ids_ae = \
            self.ae_prep_data(strings, strings_label, words_label_ids, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)
        input_ids_ot, attn_mask_ot, decoder_label_ids_ot, decoder_attn_mask_ot, input_node_ids_ot, input_edge_ids_ot, decoder_whole_ids_ot = \
            self.ot_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids, words_whole_ids)

        node_length_ar, node_length_ae, node_length_ot = max(input_node_ids_ar) + 1, max(input_node_ids_ae) + 1, max(
            input_node_ids_ot) + 1
        edge_length_ar, edge_length_ae, edge_length_ot = max(input_edge_ids_ar) + 1, max(input_edge_ids_ae) + 1, max(
            input_edge_ids_ot) + 1
        word_length_ot = max(decoder_whole_ids_ot) + 1

        def masked_fill(src, masked_value, fill_value):
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar, input_edge_ids_ar = masked_fill(input_node_ids_ar, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ar, -1, self.args.max_edge_length)
        input_node_ids_ae, input_edge_ids_ae = masked_fill(input_node_ids_ae, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ae, -1, self.args.max_edge_length)
        input_node_ids_ot, input_edge_ids_ot = masked_fill(input_node_ids_ot, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ot, -1, self.args.max_edge_length)
        decoder_whole_ids_ot = masked_fill(decoder_whole_ids_ot, -1, self.args.max_output_length)

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)
        adj_matrix_ae = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)
        adj_matrix_ot = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)

        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length == len(input_node_ids_ar) == len(
            input_edge_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length
        assert len(input_ids_ae) == len(attn_mask_ae) == len(encoder_label_ids) == self.args.max_input_length == len(
            input_node_ids_ae) == len(input_edge_ids_ae)
        assert len(input_ids_ot) == len(attn_mask_ot) == self.args.max_input_length == len(input_node_ids_ot) == len(
            input_edge_ids_ot)
        assert len(decoder_label_ids_ot) == len(decoder_attn_mask_ot) == self.args.max_output_length == len(
            decoder_whole_ids_ot)

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_ids_ae = torch.LongTensor(input_ids_ae)
        attn_mask_ae = torch.LongTensor(attn_mask_ae)
        encoder_label_ids = torch.LongTensor(encoder_label_ids)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        input_edge_ids_ar = torch.LongTensor(input_edge_ids_ar)
        input_node_ids_ae = torch.LongTensor(input_node_ids_ae)
        input_edge_ids_ae = torch.LongTensor(input_edge_ids_ae)
        node_length_ar = torch.LongTensor([node_length_ar])
        node_length_ae = torch.LongTensor([node_length_ae])
        edge_length_ar = torch.LongTensor([edge_length_ar])
        edge_length_ae = torch.LongTensor([edge_length_ae])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)
        adj_matrix_ae = torch.LongTensor(adj_matrix_ae)

        input_ids_ot = torch.LongTensor(input_ids_ot)
        attn_mask_ot = torch.LongTensor(attn_mask_ot)
        input_node_ids_ot = torch.LongTensor(input_node_ids_ot)
        input_edge_ids_ot = torch.LongTensor(input_edge_ids_ot)
        decoder_label_ids_ot = torch.LongTensor(decoder_label_ids_ot)
        decoder_attn_mask_ot = torch.LongTensor(decoder_attn_mask_ot)
        decoder_whole_ids_ot = torch.LongTensor(decoder_whole_ids_ot)
        adj_matrix_ot = torch.LongTensor(adj_matrix_ot)
        word_length_ot = torch.LongTensor([word_length_ot])
        node_length_ot = torch.LongTensor([node_length_ot])
        edge_length_ot = torch.LongTensor([edge_length_ot])

        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, \
               input_edge_ids_ar, node_length_ar, edge_length_ar, adj_matrix_ar, \
               input_ids_ae, attn_mask_ae, encoder_label_ids, input_node_ids_ae, input_edge_ids_ae, \
               node_length_ae, edge_length_ae, adj_matrix_ae, \
               input_ids_ot, attn_mask_ot, decoder_label_ids_ot, decoder_attn_mask_ot, decoder_whole_ids_ot, \
               input_node_ids_ot, input_edge_ids_ot, node_length_ot, edge_length_ot, word_length_ot, adj_matrix_ot


class WikidataDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(WikidataDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                                 num_workers=args.num_workers)


class WebNLGDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(WebNLGDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                               num_workers=args.num_workers)


# Downstream dataset (webnlg, webquestions, pathquestions)
# Most parts are similar to WikidataDataset
class WebNLGDataset(Dataset):
    def __init__(self, logger, args, data_path, tokenizer, mode):
        self.data_path = data_path # path to raw data file
        self.tokenizer = tokenizer # tokenizer to tokenize textual data
        with open(self.data_path + '.json', 'r') as f:
            self.data = json.load(f) # json style dataset, List[Dict[str, ...]]

        print("Total samples = {}".format(len(self.data)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list # ensure self.data is a list of dicts
        assert all(["id" in d for d in self.data]), self.data[0].keys() # ensure each data sample contains a 'id' key
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"]) # convert "id" key from int to string

        self.args = args
        self.data_type = mode # 'train' or 'test'
        self.metric = "BLEU"

        # make special tokens
        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' [head]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [relation]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [tail]', add_special_tokens=False)
        
        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)

        # mask token and mask_token_id use different APIs for BART and other pretrained models
        if self.args.model_name == "bart":
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token = self.tokenizer.additional_special_tokens[0]
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])
        
        # same for bos token
        if self.args.model_name == "bart":
            if self.args.append_another_bos:
                self.add_bos_id = [self.tokenizer.bos_token_id] * 2
            else:
                self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []

    def __len__(self):
        return len(self.data)

    def linearize_v2(self, entity, entity_change, head_ids, rel_ids, tail_ids,
                        relation_change, cnt_edge, adj_matrix):
        """

        Args:
            entity (List[str, str, List[List[str, str]]]): contain a mark entity, and all its relation linking to tail entities
            ent_change (Dict[str, List[List[int], int]]): mapping all entity names to its token ids and ent_id
            head_ids (List[int]): token ids for " [head]"
            rel_ids (List[int]): token ids for " [relation]"
            tail_ids (List[int]): token ids for " [tail]"
            relation_change (Dict[str, List[int]]): mapping each relation name to its token ids
            cnt_edge (int): current edge counts
            adj_matrix (List[List[int]]): current adjency matrix
        Returns:

        """
        """linearize the knowledge graph to sequence"""

        # string_label: encoder ids
        # string_label_tokens: encoder tokens

        """ entity = 
            [
                "none",
                "none",
                [
                    [
                        "office position or title",
                        "Minister of Food"
                    ]
                ]
            ]
        """

        # ent_change (Dict[str, List[List[int], int]]): mapping all entity names to its token ids and ent_id

        if len(entity[0]) == 0: # cnt_edge and adj_matrix remain unchanged
            return [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids) # List[int]
        string_label_tokens = ' [head]' # str
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))

        string_label += entity_change[entity[0]][0] # List[int], token ids for mark entity in entity
        string_label_tokens += ' {}'.format(entity[0]) # " [head] none"
        nodes.extend([entity_change[entity[0]][1]] * len(entity_change[entity[0]][0])) # extend nodes with node id for the first mark entity
        edges.extend([-1] * len(entity_change[entity[0]][0]))

        for rel in entity[2]: # rel = ["office position or title", "Minister of Food"]
            if len(rel[0]) != 0 and len(rel[1]) != 0:
                rel_label = relation_change[rel[0]] # relation token ids, List[int]
                rel_label_token = copy.deepcopy(rel[0]) # relation name, str
                words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][0] # just the tokenized ids of the following str
                words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, rel[1]) # " [relation] office position or title [tail] Minister of Food"
                nodes.extend(
                        [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[rel[1]][1]] * len(
                            entity_change[rel[1]][0]))
                edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                            len(tail_ids) + len(entity_change[rel[1]][0])))
                if entity_change[entity[0]][1] < len(adj_matrix) and entity_change[rel[1]][1] < len(adj_matrix):
                    # if the id of head entity and tail entity are all inside max_node_length
                    adj_matrix[entity_change[entity[0]][1]][entity_change[rel[1]][1]] = cnt_edge # adj_matrix[head_id][tail_id] = edge_id

                cnt_edge += 1
                string_label += words_label
                string_label_tokens += words_label_tokens

        assert len(string_label) == len(nodes) == len(edges)

        # string_label = tokenizer.encode(string_label_tokens)
        # string_label_tokens = " [head] none  [relation] office position or title [tail] Minister of Food"
        # nodes is a list of indicators, -1 for non entity token id, the corresponding entity id for entity token id
        # edges do the same thing
        # cnt_edge is an integer recording the current edges so far
        # adj_matrix records the edge between head and tail entities
        return string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    def get_all_entities_per_sample(self, mark_entity_number, mark_entity, entry):
        """Get all textual entities and relations in the entry sample, excluding mark_entity

        Args:
            mark_entity_number (List[str]): list of entity number
            mark_entity (List[str]): list of mark entities
            entry (Dict[str, ...]): a data sample for graph -> question
        Returns:
            text_entity_list (List[str]): all entity names in entry sample, excluding mark_entity
            text_relation_list (List[str]): all relation names in entry sample
        """
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = entry['kbs'][entity_id] # a list
            """ entity = 
            [
                "none",
                "none",
                [
                    [
                        "office position or title",
                        "Minister of Food"
                    ]
                ]
            ]
            """
            if len(entity[0]) == 0: # no mark entity in this sample
                continue
            for rel in entity[2]: # a list of list
                """ entity[2] = 
                [
                    [
                        "office position or title",
                        "Minister of Food"
                    ]
                ]
                """
                """ rel = [
                        "office position or title",
                        "Minister of Food"
                ] """
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    text_relation.add(rel[0]) # containing all relations
                    text_entity.add(rel[1]) # containing all tail entities

        text_entity_list = list(text_entity) # ["Minister of Food"]
        text_relation_list = list(text_relation) # ["office position or title"]

        # removing all mark entity in text entity
        for entity_ele in mark_entity: # mark_entity = ["none", "River Thames", "England"]
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        """Get mapping of entity -> (entity token id, entity id) and relation -> relation token ids for the sample
        Assuming no duplicate relation exists in one sample

        Args:
            mark_entity (List[str]): all head entities
            text_entity (List[str]): all tail entities, exclusive of mark_entity
            text_relation (List[str]): all relations
        Returns:
            ent_change (Dict[str, List[List[int], int]]): mapping entity name to its token ids and ent_id
            rel_change (Dict[str, List[int]]): mapping relation name -> token ids
        """

        # during fine-tuning, we don't mask entities or relations
        ent_change = {} # Dict[str, List[List[int], int]], mapping entity name to its token ids and ent_id
        total_entity = mark_entity + text_entity # all entities

        # mark_entity = ["none", "River Thames", "England"]
        # text_entity = ["Minister of Food"]
        # total_entity = ["none", "River Thames", "England", "Minister of Food"]

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False) # List[int]
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]

        # relation change only includes the relation tokens and ids
        rel_change = {} # Dict[str, List[int]], mapping relation name -> token ids
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                      add_special_tokens=False)

        return ent_change, rel_change

    def truncate_pair_ar(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    def ar_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        """

        Args:
            answers (List[int]): token ids for every word in target sequence
            questions (List[int]): token ids for every entity in source graph
            add_bos_id (List[int]): 0/1/2 bos token ids
            graph_ids (List[int]): token ids for " [graph]"
            text_ids (List[int]): token ids for " [text]"
            node_ids (List[int]): node ids or -1, same length as questions
            edge_ids (List[int]): edge ids or -1, same length as questions
        Returns:

        """
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)

        # trunction
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        
        # add bos end eos
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ar(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids

    def __getitem__(self, idx):
        """fetch a single data sample in the dataset"""

        entry = self.data[idx] # Dict[str, ...]
        # entry is a data sample in json file

        entities = []

        # entry['kbs']: Dict[str, ...]
        for _ in entry['kbs']:
            entities.append(_) # _ is an integer index number starting from 0, in type string
        # entities = ['0', '1', '2']

        strings_label = []
        node_ids = []
        edge_ids = []
        strings_label_tokens = ''

        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [entry['kbs'][ele_entity][0] for ele_entity in entities] # ["none", "River Thames", "England"]
        mark_entity_number = entities # ['0', '1', '2']
        
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity, entry) # list of str
        # text_entity = ["Minister of Food"]
        # text_relation = ["office position or title"]
        
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        """
        ent_change (Dict[str, List[List[int], int]]): mapping entity name to its token ids and ent_id
        rel_change (Dict[str, List[int]]): mapping relation name -> token ids
        """

        total_entity = mark_entity + text_entity
        adj_matrix = [[-1] * (self.args.max_node_length + 1) for _ in range(self.args.max_node_length + 1)] # matrix of shape [max_node_len+1, max_node_len+1], full of -1 for now

        cnt_edge = 0

        if 'title' in entry: # usually no
            entity = self.knowledge[entry['title_kb_id']]

            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings_label += string_label
            strings_label_tokens += string_label_tokens
        
        # construct the knowledge graph
        for i, entity_id in enumerate(entities): # entities = ['0', '1', '2']
            entity = entry['kbs'][entity_id]
            """ entity = 
            [
                "none",
                "none",
                [
                    [
                        "office position or title",
                        "Minister of Food"
                    ]
                ]
            ]
            """
            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings_label += string_label
            strings_label_tokens += string_label_tokens
            node_ids += nodes
            edge_ids += edges

        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens = [], '', [], ''
        current_text = random.choice(entry['text']) # entry['text'] is a list, but in wq and pq, it's of length 1, only in webnlg it's multiple sentences

        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, input_edge_ids_ar = \
            self.ar_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)

        node_length_ar = max(input_node_ids_ar) + 1
        edge_length_ar = max(input_edge_ids_ar) + 1

        def masked_fill(src, masked_value, fill_value):
            """Replace masked_value and beyond fill_value with fill_value"""
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar, input_edge_ids_ar = masked_fill(input_node_ids_ar, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ar, -1, self.args.max_edge_length)

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)

        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length == len(input_node_ids_ar) == len(
            input_edge_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        input_edge_ids_ar = torch.LongTensor(input_edge_ids_ar)
        node_length_ar = torch.LongTensor([node_length_ar])
        edge_length_ar = torch.LongTensor([edge_length_ar])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)

        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, \
               input_node_ids_ar, input_edge_ids_ar, node_length_ar, edge_length_ar, adj_matrix_ar, entry['id']


def evaluate_bleu(data_ref, data_sys):
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}
    return scores["Bleu_4"]
