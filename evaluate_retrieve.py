""" Script to evaluate the pure retrieval performance """
import os
import argparse

import json
import nltk

from nltk.translate.bleu_score import corpus_bleu


parser = argparse.ArgumentParser(description="Script to evaluate the performance of purely retrieval method.")
parser.add_argument("--data_path", type=str, default="data/wq", help="Path to the dataset containing retrieved texts.")


args = parser.parse_args()


def postprocess_text(preds, labels):
    """Batch-level postprocessing
    """

    # post processing to adapt to the BLEU score calculation
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # BLEU expects sentence in form of words
    preds = [nltk.word_tokenize(pred) for pred in preds]
    labels = [[nltk.word_tokenize(label)] for label in labels]
    return preds, labels


with open(os.path.join(args.data_path, "reinatest.json"), 'r') as f:
    data = json.load(f)


references = []
hypotheses = []

for entry in data:
    target_text = entry['text']
    retrieved_text = [entry['ref'][0] if len(entry['ref']) != 0 else ""]
    hyp, ref = postprocess_text(retrieved_text, target_text)

    hypotheses.extend(hyp)
    references.extend(ref)



result = corpus_bleu(
    list_of_references=references,
    hypotheses=hypotheses,
    weights=[
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25,  0.25, 0.25)
    ]
)

print(f"BLEU1: {result[0]}, BLUE2: {result[1]}, BLEU3: {result[2]}, BLEU4: {result[3]}")
