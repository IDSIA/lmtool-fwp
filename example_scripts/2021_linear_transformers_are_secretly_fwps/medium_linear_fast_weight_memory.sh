#!/bin/bash

## Linear Transformer with our fast weight memory update rule
## --attn_type 24: indicates the combination: linear transformer + our update rule.

## other options are:

## - Standard Transformer: 2

## - Fast weights with the sum update rule:
##    * Linear Transformer: 4 (pure pytorch) or 34 (cuda kernel)
##    * Performer: 5 (pure pytorch) or 35 (cuda kernel)
##    * DPFP: 6 (pure pytorch) or 36 (cuda kernel)

## - Fast weights with our update rule:
##    * Linear Transformer: 24 (no attn normalization) or 44 (with attn normalization)
##    * Performer: 25 (no attn normalization) or 45 (with attn normalisation)
##    * DPFP: 26 (no attn normalization) or 46 (with attn normalisation)

## for Performers, `m` can be specified via --performer_proj_dim
## for DPFP, `\nu` can be specified via --dpfp_n_roll 

## For attn_type 44, 45, 46
## --skip_attn_normalization: disable the attn normalization everywhere.
## (this should give the best performance)
## For attn_type 24, 25, 26, skip_attn_normalization is set to True by default
## if

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
        --d_model 256 \
        --n_head 8 \
        --d_head 32 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 2000 \
        --max_step 400000 \
        --attn_type 24 \
        --tgt_len 384 \
        --mem_len 0 \
        --eval_tgt_len 384 \
        --batch_size 56 \
        --multi_gpu \
        --use_wandb \
        --project_name '2021-01--lm-2048-256' \
        ${@:2}

elif [[ $1 == 'valid' ]]; then
    echo 'Run validation...'
    python eval_sliding_window.py \
        --cuda \
        --batch_size 1 \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 384 \
        --mem_len 0 \
        --clamp_len 384 \
        --split valid \
        ${@:2}

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval_sliding_window.py \
        --cuda \
        --batch_size 1 \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 384 \
        --mem_len 0 \
        --clamp_len 384 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
