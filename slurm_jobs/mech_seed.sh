#!/bin/bash
# One more seed of a paper position model at L = 1024 with its original recipe.
# Usage: mech_seed.sh <gpu> <dtype> <n_head> <lr> <seed>
#   1H (ATTN2-FULL): n_head 1, lr 3e-4.   12H (FULL-12H): n_head 12, lr 6e-4.
GPU=$1; DTYPE=$2; NH=$3; LR=$4; SEED=$5
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
OUT=out-mechanism-${NH}H-1024-seed$SEED
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_2layer_mechanism.py --regime R0 --block_size 1024 --use_flash \
  --batch_size 32 --gradient_accumulation_steps 4 --n_head $NH --seed $SEED --max_iters 20000 \
  --max_iters_override 20000 --eval_iters 50 --eval_interval 500 --no_compile --learning_rate $LR \
  --dtype $DTYPE --out_dir $OUT --resume_dir $OUT/run > ../logs/mech_${NH}H_seed$SEED.log 2>&1
