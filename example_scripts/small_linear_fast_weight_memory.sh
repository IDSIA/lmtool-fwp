#!/bin/bash

## Linear Transformer with our fast weight memory update rule
## --attn_type 24: indicates the combination: linear transformer + our update rule.

## other options are:
## Standard Transformer: 2

## Standard models (Fast weights with sum update:
##   * Linear Transformer: 4 (pure pytorch) or 34 (custom cuda kernel)
##   * Performer: 5 (pure pytorch) or 35 (custom cuda kernel)
##   * DPFP: 6 (pure pytorch) or 36 (custom cuda kernel)

## Fast weights with our update rule:
##   * Linear Transformer: 24 (no attn normalisation to retrieve with key) or 44 (with attn normalisation)
##   * Performer: 25 (no attn normalisation to retrieve with key) or 45 (with attn normalisation)
##   * DPFP: 26 (no attn normalisation to retrieve with key) or 46 (with attn normalisation)

## for Performers, `m` can be specified via --performer_proj_dim
## for DPFP, `\nu` can be specified via --dpfp_n_roll 

## --skip_attn_normalization: disable the attn normalisation everywhere.
## (this should give the best performance)

export CUDA_VISIBLE_DEVICES=0,1
# requires 2 GPUs with 16 GB memory

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
        --attn_type 24 \
        --tgt_len 256 \
        --mem_len 0 \
        --eval_tgt_len 256 \
        --batch_size 96 \
        --multi_gpu \
        --use_wandb \
        --project_name '2021-01--lm-2048-128' \
        ${@:2}

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
