#!/bin/bash

## The standard limited context Transformer
## --attn_type 2: indicates the stadard transformer 

export CUDA_VISIBLE_DEVICES=0,1

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 128 \
        --n_head 8 \
        --d_head 16 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 2000 \
        --max_step 500000 \
        --attn_type 2 \
        --tgt_len 256 \
        --mem_len 0 \
        --eval_tgt_len 256 \
        --batch_size 96 \
        --multi_gpu \
        --use_wandb \
        --project_name '2021-01--lm-2048-128' \
        ${@:2}

## Set batch size to 1 for evaluation 
elif [[ $1 == 'valid' ]]; then
    echo 'Run validation...'
    python eval_sliding_window.py \
        --cuda \
        --batch_size 1 \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 256 \
        --mem_len 0 \
        --clamp_len 256 \
        --split valid \
        ${@:2}

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval_sliding_window.py \
        --cuda \
        --batch_size 1 \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 256 \
        --mem_len 0 \
        --clamp_len 256 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
