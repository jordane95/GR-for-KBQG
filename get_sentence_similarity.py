""" Add sentence embeddings to the text of graph2text dataset, as supervision signal """
import os
import json
import random
import argparse

from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Add sentence embeddings to the text of graph2text dataset.")
parser.add_argument("--data_path", type=str, default="data/wq", help="Path to the dataset to be processed.")
args = parser.parse_args()

# loading data
with open(os.path.join(args.data_path, "train.json"), 'r') as f:
    train_set = json.load(f)
with open(os.path.join(args.data_path, "dev.json"), 'r') as f:
    dev_set = json.load(f)
with open(os.path.join(args.data_path, "test.json"), 'r') as f:
    test_set = json.load(f)


# get sentence embeddings from s-bert
train_sentences = [entry['text'][0] for entry in train_set]
dev_sentences = [entry['text'][0] for entry in dev_set]
test_sentences = [entry['text'][0] for entry in test_set]

model = SentenceTransformer("bert-base-nli-mean-tokens")

train_sentence_embeddings = model.encode(train_sentences)
dev_sentence_embeddings = model.encode(dev_sentences)
test_sentence_embeddings = model.encode(test_sentences)

