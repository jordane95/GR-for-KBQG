""" Train a bi graph encoder with MSE distillation from a pretrained sentence encoder """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random

from tqdm import tqdm, trange
import numpy as np
import torch

from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import BartTokenizer

from graph_encoder import NGRGraphEncoder

from data_for_gr import WebNLGDatasetForGR, WebNLGDataLoader

from tensorboardX import SummaryWriter

train_writer = SummaryWriter("log/train")
dev_writer = SummaryWriter("log/dev")


def cos_sim(a: Tensor, b: Tensor):
    # [a, d] * [b, d] -> [a, b]
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


class BiGraphEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph_model = NGRGraphEncoder.from_pretrained(args.graph_model_name_or_path)

        self.sim_fn = cos_sim
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        graph_embeddings = self.graph_model(
            input_ids=batch[0],
            attention_mask=batch[1],
            input_node_ids=batch[2],
            input_edge_ids=batch[3],
            node_length=batch[4],
            edge_length=batch[5],
            adj_matrix=batch[6],
        )
        # [batch_size, graph_emb_size]

        pos_graph_embeddings = self.graph_model(
            input_ids=batch[7],
            attention_mask=batch[8],
            input_node_ids=batch[9],
            input_edge_ids=batch[10],
            node_length=batch[11],
            edge_length=batch[12],
            adj_matrix=batch[13],
        )
        # [batch_size, graph_emb_size]

        graph_similarity = self.sim_fn(graph_embeddings, pos_graph_embeddings) # [batch_size, batch_size]
        
        batch_size = graph_similarity.size(0)
        labels = torch.tensor(range(batch_size), dtype=torch.long, device=graph_similarity.device)

        loss = self.loss_fn(graph_similarity, labels)
        return loss


def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    train_dataset = WebNLGDatasetForGR(logger, args, os.path.join(args.data_path, "sbert_train_for_graph_retrieval"), tokenizer, "train")
    dev_dataset = WebNLGDatasetForGR(logger, args, os.path.join(args.data_path, "sbert_dev_for_graph_retrieval"), tokenizer, "val")

    train_dataloader = WebNLGDataLoader(args, train_dataset, "train")
    dev_dataloader = WebNLGDataLoader(args, dev_dataset, "val")

    # Load model parameters
    model = BiGraphEncoder(args)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.no_lr_decay:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=1000000)

    train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer)


def train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer):
    model.train()
    global_step = 0
    wait_step = 0
    train_losses = []
    best_accuracy = -99999999
    stop_training = False

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting training!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            if global_step == 1:
                for tmp_id in range(9):
                    print(batch[tmp_id])

            loss = model(batch)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            
            train_writer.add_scalar("loss", loss.item(), global_step)
            
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # Gradient accumulation
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            # Print loss and evaluate on the valid set
            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = evaluate(model if args.n_gpu == 1 else model.module, dev_dataloader)
                dev_writer.add_scalar("loss", -curr_em, global_step)
                logger.info("Step %d Train loss %.2f Learning rate %.2e %s %.2f on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    scheduler.get_lr()[0],
                    "NCE loss",
                    curr_em,
                    epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    model_to_save = model.graph_model.module if hasattr(model.graph_model, 'module') else model.graph_model
                    model_to_save.save_pretrained(args.output_dir)
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                (dev_dataloader.dataset.metric, best_accuracy * 100.0, curr_em * 100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break

@torch.no_grad()
def evaluate(model, dev_dataloader):
    model.eval()
    neg_loss = 0
    # Inference on the test set
    count = 0
    for i, batch in enumerate(dev_dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        loss = model(batch)
        neg_loss -= loss.item()
        count += 1
    model.train()
    return neg_loss / count


def main():
    parser = argparse.ArgumentParser(description="Arguments to train a graph encoder from a sentence encoder.")

    # Basic parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/wq",
        help="Path to the dataset."
    )
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature in contrastive loss."
    )
    # Model parameters
    parser.add_argument(
        "--graph_model_name_or_path",
        type=str,
        default="pretrain_model/jointgt_bart",
        help="Path to the pretrained graph encoder model."
    )
    parser.add_argument(
        "--sentence_model_name_or_path",
        type=str,
        default="bert-base-nli-mean-tokens",
        help="Name on huggingface or path to pretrained sentence encoder."
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

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)
    parser.add_argument("--no_lr_decay", action='store_true', default=False)

    # Other parameters
    parser.add_argument('--eval_period', type=int, default=1000,
                        help="Evaluate  model")
    parser.add_argument('--save_period', type=int, default=1000,
                        help="Save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="Number of workers for dataloaders")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Start writing logs

    log_filename = "{}log.txt".format("")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Using {} gpus".format(args.n_gpu))
    run(args, logger)


if __name__ == '__main__':
    main()
