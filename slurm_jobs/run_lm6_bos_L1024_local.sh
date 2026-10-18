#!/bin/bash
# The 6-layer BOS LM at L = 1024 on one local GPU (dsinlp01). Usage: run_lm6_bos_L1024_local.sh <gpu> <seed>
# Same settings as run_lm6_bos_L1024.sh, with 32 x 4 x 1024 = 131072 tokens per step. Resumes from ckpt.pt.
set -euo pipefail
GPU=$1; SEED=$2
cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
OUT=out-lm-6layer-fulltrain-bos-L1024-seed$SEED
RESUME=""
if [ -f "$OUT/ckpt.pt" ]; then RESUME="--init_from=resume"; fi
CUDA_VISIBLE_DEVICES=$GPU /home/nlp/matan_avitan/.conda/envs/py312/bin/python -u train_nope.py config/train_lm_6layer_fulltrain_bos_ddp.py \
    --seed=$SEED --out_dir=$OUT --block_size=1024 --batch_size=32 --gradient_accumulation_steps=4 \
    --wandb_log=False --dtype=bfloat16 $RESUME >> ../logs/lm_bos_L1024_6l_seed${SEED}_local.log 2>&1
