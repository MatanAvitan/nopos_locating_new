#!/bin/bash
# R2 Attention-Only 1-Head Training Command
#
# Prerequisites:
#   - OpenWebText data prepared in nanoGPT/data/openwebtext/
#   - CUDA available

cd "$(dirname "$0")/../../nanoGPT"

python train_2layer_mechanism.py \
    --regime R2 \
    --n_head 1 \
    --max_iters 40000 \
    --r2_attn_head_only \
    --wandb
