#!/bin/bash
# BOS-trained NoPE LM at L = 2048 on one local GPU (dsinlp01). Usage: run_lm_bos_L2048_local.sh <gpu> <n_layer> <seed>
# Same config as config/train_lm_6layer_fulltrain_bos_ddp.py, with the same tokens per optimizer step:
# 16 x 4 x 2048 = 131072. Resumes from ckpt.pt if the run stopped.
# Attention-stat logging is off, so the fused attention kernel is used (same attention, faster at L = 2048).
set -euo pipefail
GPU=$1; NL=$2; SEED=$3
cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
OUT=out-lm-${NL}layer-fulltrain-bos-L2048-seed$SEED
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
RESUME=""
if [ -f "$OUT/ckpt.pt" ]; then RESUME="--init_from=resume"; fi
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_nope.py config/train_lm_6layer_fulltrain_bos_ddp.py \
    --seed=$SEED --out_dir=$OUT --n_layer=$NL --block_size=2048 --batch_size=16 --gradient_accumulation_steps=4 \
    --wandb_log=False --dtype=bfloat16 --log_attention_stats=False $RESUME \
    >> ../logs/lm_bos_L2048_${NL}l_seed$SEED.log 2>&1
