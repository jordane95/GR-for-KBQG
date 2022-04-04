""" Retrieve similar questions with sentence-bert """
import os
import json
import random
import argparse

import faiss

from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Retrieve similar questions with s-bert.")
parser.add_argument("--data_path", type=str, default="data/wq", help="Path to the dataset to be processed.")
parser.add_argument("--k", type=int, default=5, help="Number of retrieved questions.")

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


# build index with questions from train set
d = train_sentence_embeddings.shape[1]

index = faiss.IndexFlatL2(d)
index.add(train_sentence_embeddings)

N = index.ntotal

# retrieving process
k = args.k
D_train, I_train = index.search(train_sentence_embeddings, k+1)
D_dev, I_dev = index.search(dev_sentence_embeddings, k)
D_test, I_test = index.search(test_sentence_embeddings, k)


refs_train = [[train_sentences[i] for i in idxs[1:]] for idxs in I_train] # elimating itself for train set
refs_dev = [[train_sentences[i] for i in idxs] for idxs in I_dev]
refs_test = [[train_sentences[i] for i in idxs] for idxs in I_test]

new_train_set = [{**entry, 'refs': refs} for entry, refs in zip(train_set, refs_train)]
new_dev_set = [{**entry, 'refs': refs} for entry, refs in zip(dev_set, refs_dev)]
new_test_set = [{**entry, 'refs': refs} for entry, refs in zip(test_set, refs_test)]

# save the retrieval augmented dataset to disk
with open(os.path.join(args.data_path, 'sbert_train.json'), 'w') as fpw:
    json.dump(new_train_set, fpw)
with open(os.path.join(args.data_path, 'sbert_dev.json'), 'w') as fpw:
    json.dump(new_dev_set, fpw)
with open(os.path.join(args.data_path, 'sbert_test.json'), 'w') as fpw:
    json.dump(new_test_set, fpw)
