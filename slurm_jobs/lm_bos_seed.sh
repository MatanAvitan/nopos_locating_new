#!/bin/bash
# One seed of the BOS-trained 6-layer NoPE LM (config/train_lm_6layer_fulltrain_bos_ddp.py),
# single GPU, same tokens per optimizer step (batch * accum * 128 = 131072).
# Usage: lm_bos_seed.sh <gpu> <seed> <dtype> <batch> <accum>
GPU=$1; SEED=$2; DTYPE=$3; BATCH=$4; ACCUM=$5
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
OUT=out-lm-6layer-fulltrain-bos-seed$SEED
RESUME=""; [ -f $OUT/ckpt.pt ] && RESUME="--init_from=resume"
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_nope.py config/train_lm_6layer_fulltrain_bos_ddp.py \
  --seed=$SEED --out_dir=$OUT --wandb_log=False --dtype=$DTYPE \
  --batch_size=$BATCH --gradient_accumulation_steps=$ACCUM $RESUME \
  > ../logs/lm_bos_seed$SEED.log 2>&1
