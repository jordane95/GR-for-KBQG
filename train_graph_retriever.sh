#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_graph_retriever.py \
    --output_dir retriever/bart_wq \
    --train_batch_size 16 \
    --predict_batch_size 32 \
    --max_input_length 256 \
    --max_output_length 128 \
    --append_another_bos \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --warmup_steps 3400 \
    --eval_period 600 \
    --wait_step 15