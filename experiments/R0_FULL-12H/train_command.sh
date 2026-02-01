#!/bin/bash
# R0 Full 12-Head Training Command
#
# Prerequisites:
#   - OpenWebText data prepared in nanoGPT/data/openwebtext/
#   - CUDA available

cd "$(dirname "$0")/../../nanoGPT"

python train_2layer_mechanism.py \
    --regime R0 \
    --wandb
