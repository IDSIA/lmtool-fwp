#!/bin/bash

## In order to train and evaluate models without truncating context,
## --carry_over_fast_weight flag should be added.

export CUDA_VISIBLE_DEVICES=0,1

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
        --carry_over_fast_weight \
        --no_pos \
        --multi_gpu \
        --use_wandb \
        --project_name '2021-01--lm-2048-256-cco' \
        ${@:2}

elif [[ $1 == 'valid' ]]; then
    echo 'Run validation...'
    python eval.py \
        --cuda \
        --batch_size 1 \
        --carry_over_fast_weight \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 384 \
        --mem_len 0 \
        --clamp_len 384 \
        --split valid \
        ${@:2}

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --batch_size 1 \
        --carry_over_fast_weight \
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
