import sys, os, lucene, threading, time
import math
from multiprocessing import Pool
import shutil

from datetime import datetime

from org.apache.lucene import analysis, document, index, queryparser, search, store, util
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory, MMapDirectory
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.search.similarities import BM25Similarity, TFIDFSimilarity
import random

import json
import string
import glob
import bz2
import gzip
import sys
from tqdm import tqdm
from nltk import sent_tokenize
from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from datasets import Dataset

stops_en = set(stopwords.words('english'))
exclude = set(string.punctuation)

def remove_punc(text):
    """Remove punctuations in text

    Args:
        text (str):
    Return:
        str
    """
    return ''.join(ch for ch in text if ch not in exclude)

def word_tokenize(text, lowercase=True):
    """Split a sentence to words, removing stop words and puncutations

    Args:
        text (str): a sentence
    Return:
        the first 600 words in `text`, seperated by white space
    """
    words = tokenize(text)
    outputs = []
    for token in words:
        if token not in stops_en and token not in exclude:
            outputs.append( remove_punc(token) )

    return ' '.join(outputs[:600])

class MyMemLucene():

    def __init__(self):

        lucene.initVM()
        # # # lucene # # #
        self.t1 = FieldType()
        self.t1.setStored(True)
        self.t1.setTokenized(False)
        self.t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        self.t2 = FieldType()
        self.t2.setStored(True)
        self.t2.setTokenized(True)
        self.t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        self.t3 = FieldType()
        self.t3.setStored(True)

        self.analyzer = StandardAnalyzer()


    def built_RAM(self, data, key, value):
        """

        Args:
            data (Dataset): the index dataset
            key (str): key field as query
            value (str): value field for target
        Returns:
        """
        self.index_directory = RAMDirectory()
        config = IndexWriterConfig( self.analyzer )
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        iwriter = IndexWriter(self.index_directory, config)

        print('Building REINA index ...')
        qbar = tqdm(total=len(data[key]))

        for instance_key, instance_value in zip(data[key], data[value]):
            doc = Document()
            doc.add(Field(key, instance_key, self.t2))
            doc.add(Field(value, instance_value, self.t2))

            try:
                iwriter.addDocument(doc)
            except:
                print(instance_value)
                continue
            qbar.update(1)
        qbar.close()
        iwriter.close()

    def retrieve_RAM(self, lines, docs_num, key, value):
        """

        Args:
            lines (List[int]): queries
            docs_num (int):
            key (str):
            value (str):
        Returns:
            output_all (List[str]): query + retrieved targets
        """

        ireader = DirectoryReader.open(self.index_directory)
        isearcher = search.IndexSearcher(ireader)
        isearcher.setSimilarity(BM25Similarity())

        parser = queryparser.classic.QueryParser( key, self.analyzer)

        output_all = []
        for question in lines:
            try:
                query = parser.parse(question)
            except:
                try:
                    query = parser.parse(word_tokenize(question))
                except:
                    output_all.append(question)
                    continue

    
            hits = isearcher.search(query, max(20, docs_num) ).scoreDocs
            output = []
            for hit in hits:
                hitDoc = isearcher.doc(hit.doc)
                try:
                    if hitDoc[key] == question: continue # avoid retrieving itself
                    output.append( hitDoc[value] )
                    
                except:
                    continue

            instance = ' '.join(output[:docs_num])
            output_all.append(instance)

        return output_all

class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global mylc
        mylc = MyMemLucene()
        mylc.built_RAM( self.args['index_data'] , self.args['key'], self.args['value'] )


    def retrieve_lines(self, lines):
        """

        Args:
            lines (List[str]): queries
        Returns:
            output ():
        """
        output = mylc.retrieve_RAM( lines, 5, self.args['key'], self.args['value'] )
        return output


def reina_apply(raw_datasets, key, value, num_proc):
    """Bulid a renia dataset
    
    Args:
        raw_datasets (Dict[str, Dataset]): train, dev and test set
        key (str):
        value (str):
        num_proc (int):
    Returns:
        datasets_new (Dict[str, Dataset]): reina dataset, where the key is retrieved from corpus, value is ground truth
    """
    
    train_set = raw_datasets['train'] # only using train set as index
    
    # adapt to the APIs provided for huggingface Dataset object
    index_data_list = dict()
    index_data_list[key] = [entry[key] for entry in train_set]

    # entry['text'] is a list of str
    index_data_list[value] = [entry[value][0] for entry in train_set]

    query_data_dict = {k:v for k, v in raw_datasets.items()} # size of 3
    datasets_new = defaultdict(dict)

    retriever = MultiprocessingEncoder({'index_data': index_data_list, 'key': key, 'value': value})
    pool = Pool(num_proc, initializer=retriever.initializer)
    

    for set_name, query_data in query_data_dict.items():
        # set_name = 'train'
        # query_data = train_set
        print(set_name)
        # query_data is the dataset, key is the input field, such as, article in summarization
        lines = [  q[key]  for q in query_data ] # queries

        # imap just like map
        encoded_lines = pool.imap(retriever.retrieve_lines, zip(*[lines]), 100)
        print('REINA start ...')
        lines_reina = []
        qbar = tqdm(total=len(query_data))
        key_id = 0
        for line_id, lines_ir in enumerate(encoded_lines):
            for line in lines_ir:
                lines_reina.append(line)
                key_id += 1
            qbar.update(len(lines_ir))

        qbar.close()
        datasets_new[set_name] = [entry.update({"references": ref}) for entry, ref in zip(query_data, lines_reina)]
    return datasets_new

def reina(raw_datasets, key, value, use_cache, num_proc=10):
    """Build reina dataset with retrieved exemplars, from raw datasets

    Args:
        raw_datasets (Dataset): huggingface Dataset object
        key (str): 
        value (str): 
        use_cache (bool): 
        num_proc (int): maybe the number of multiprocessing workers
    """

    import torch
    import pickle
    
    reina_path = os.getenv("HF_DATASETS_CACHE",os.path.join(os.path.expanduser('~'), '.cache/huggingface/datasets/'))
    reina_path = os.path.join(reina_path, 'reina')
    reina_dataset_path = os.path.join(reina_path, 'reina_dataset.pkl')
    
    if torch.cuda.current_device() == 0:
        print('REINA path for cache: ' + reina_dataset_path)
        print('Please remove it if data modified!')

    if not use_cache and torch.cuda.current_device() == 0:
        # build a new reina dataset
        datasets_new = reina_apply(raw_datasets, key, value, num_proc)

        # save the reina dataset to disk
        if not os.path.isdir(reina_path):
            os.makedirs(reina_path)
        with open(reina_dataset_path, 'wb') as fpw:
            pickle.dump(datasets_new, fpw)
     
    torch.distributed.barrier()
    with open(reina_dataset_path, 'rb') as fpr:
        datasets_new = pickle.load(fpr)

    return datasets_new

def reina_offline(data_path, key, value, num_proc):
    datasets = load_g2s_dataset(data_path)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    datasets_new = reina_apply(datasets, key, value, num_proc)
    for set_name in ['validation', 'test', 'train']:
        if set_name not in datasets_new: continue

        print('REINA for ' + set_name)
        with open(os.path.join(data_path, 'reina' + set_name + '.json'), 'w') as fpw:
            json.dump(datasets_new[set_name], fpw)


def load_g2s_dataset(data_path):
    """Load and process raw graph2text dataset from disk

    Args:
        data_path (str): path to the datasets
    Returns:
        g2s_datasets (Dict): train, validation and test set
    """

    def get_all_entities_per_sample(mark_entity_number, mark_entity, entry):
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


    def linearize_graph(entry):
        entities = []

        # entry['kbs']: Dict[str, ...]
        for _ in entry['kbs']:
            entities.append(_) # _ is an integer index number starting from 0, in type string
        # entities = ['0', '1', '2']

        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [entry['kbs'][ele_entity][0] for ele_entity in entities] # ["none", "River Thames", "England"]
        mark_entity_number = entities # ['0', '1', '2']
        
        text_entity, text_relation = get_all_entities_per_sample(mark_entity_number, mark_entity, entry) # list of str
        # text_entity = ["Minister of Food"]
        # text_relation = ["office position or title"]

        total_entity = mark_entity + text_entity
        
        entry['kg'] = " ".join(total_entity + text_relation)

        return entry

    # TODO: convert 'kbs' field to 'graph' field
    g2s_datasets = {}
    with open(os.path.join(data_path, "train.json"), 'r') as f:
        train_set = json.load(f)
    with open(os.path.join(data_path, "dev.json"), 'r') as f:
        dev_set = json.load(f)
    with open(os.path.join(data_path, "test.json"), 'r') as f:
        test_set = json.load(f)

    g2s_datasets['train'] = list(map(linearize_graph, train_set))
    g2s_datasets['validation'] = list(map(linearize_graph, dev_set))
    g2s_datasets['test'] = list(map(linearize_graph, test_set))
    
    return g2s_datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str, default='data/wq',
                        help='Path to the dataset')
    parser.add_argument('--key_column', type=str, default='kg',
                        help='REINA key')
    parser.add_argument('--value_column', type=str, default='text',
                        help='REINA value')
    parser.add_argument('--reina_workers', type=int, default=10,
                        help='REINA workers')

    args = parser.parse_args()
    
    reina_offline(args.data_path, args.key_column, args.value_column, args.reina_workers)

