#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python train_graph_retriever.py \
    --data_path data/wq \
    --output_dir retriever/bart_wq \
    --train_batch_size 16 \
    --predict_batch_size 16 \
    --max_input_length 256 \
    --max_output_length 128 \
    --append_another_bos \
    --learning_rate 1e-3 \
    --num_train_epochs 30 \
    --warmup_steps 1000 \
    --eval_period 600 \
    --wait_step 15