""" Apply the graph retriever on dev and test set """
import os
import logging
import argparse

from graph_encoder import GraphEncoder
from data_for_index import WebNLGDataset, WebNLGDataLoader

from faiss_retriever import BaseFaissIPRetriever

import pickle

import numpy as np
import glob

from itertools import chain
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_args():
    parser = argparse.ArgumentParser(description="Apply the graph retriever on dev and test set.")

    # Basic parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/wq",
        help="Path to the dataset."
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        help="Path to save the retrieved data."
    )

    # Model parameters
    parser.add_argument(
        "--graph_model_name_or_path",
        type=str,
        default="pretrain_model/jointgt_bart",
        help="Path to the pretrained graph encoder model."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="pretrain_model/jointgt_bart",
        help="Path to the pretrained tokenizer."
    )
    parser.add_argument("--model_name", type=str, default="bart")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=32)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument("--append_another_bos", action='store_true', default=False)
    parser.add_argument('--max_node_length', type=int, default=50)
    parser.add_argument('--max_edge_length', type=int, default=60)

    # Retrieval related parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="Batch size of query searching."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Number of graphs to be retrieved."
    )

    # Other parameters
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="Number of workers for dataloaders")
    args = parser.parse_args()
    return args


def encode(model, dataloader):
    graph_embeddings = []
    indices = []

    with torch.no_grad():
        for batch in train_dataloader:
            batch_graph_embeddings = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                input_node_ids=batch[4],
                input_edge_ids=batch[5],
                node_length=batch[6],
                edge_length=batch[7],
                adj_matrix=batch[8],
            ) # [batch_size, graph_emb_dim]

            indices.append(batch[9].cpu())

            graph_embeddings.append(batch_graph_embeddings.cpu())
    
    graph_embeddings = torch.cat(graph_embeddings).numpy()
    indices = torch.cat(indices).numpy()
    return graph_embeddings, indices


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
    return all_scores, psg_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(os.path.join(ranking_save_file, "retrieved_test.json"), 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def main():
    args = get_args()

    train_dataset = WebNLGDataset(logger, args, os.path.join(args.data_path, "sbert_train_for_graph_retrieval"), tokenizer, "train")
    test_dataset = WebNLGDataset(logger, args, os.path.join(args.data_path, "sbert_test_for_graph_retrieval"), tokenizer, "test")

    train_dataloader = WebNLGDataLoader(args, train_dataset, "test")
    test_dataloader = WebNLGDataLoader(args, dev_dataset, "test")

    # Load model parameters
    model = GraphEncoder.from_pretrained(args.graph_model_name_or_path)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    
    model.eval()

    logger.info("Encoding corpus...")
    p_reps, p_lookup = encode(model, train_dataloader)
    
    logger.info("Encoding queries...")
    q_reps, q_lookup = encode(model, test_dataloader)

    retriever = BaseFaissIPRetriever(p_reps)

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, p_lookup, args)
    logger.info('Index Search Finished')

    test_data = test_dataset.data

    test_labels = {} # Dict[int, List[int]], qid -> refs
    
    for entry in test_data:
        true_idxs = entry['refs'] # List[int]
        qid = entry['id']
        test_labels[qid] = true_idxs
    

    recalls = []
    for qid, pred_pids in zip(q_lookup, psg_indices):
        pred_pids: List[int]
        qid: int
        true_pids: List[int] = test_labels[qid]
        p = len(set(pred_pids) & set(true_pids)) / len(pred_pids)
        r = len(set(pred_pids) & set(true_pids)) / len(true_pids)
        recalls.append(r)
    recall = sum(recalls) / len(recalls)

    logger.info(f"Recall@{args.depth}: ", recall)

    # write retrieved data to disk

    args.save_path = args.save_path or args.data_path
    

if __name__ == "__main__":
    main()
