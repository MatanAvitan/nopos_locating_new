#!/bin/bash
# Worker for the 1H length sweep and Layer-1 controls (beta spec, sweep).
# Usage: sweep_1h_worker.sh <gpu> <dtype> <job> [<job> ...]
# A job is "L:seed:variant" with variant in {std, unifL1, noMLP1}.
# Every run keeps 131072 tokens per optimizer step (batch * accum * L), lr 3e-4,
# 20000 iterations, the ATTN2-FULL recipe otherwise. Runs resume from
# <out_dir>/run/last_ckpt.pt, so a restarted worker continues where it stopped.
GPU=$1; DTYPE=$2; shift 2
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
for job in "$@"; do
  IFS=: read L SEED VAR <<< "$job"
  BATCH=$((32768 / L)); ACCUM=4
  EXTRA=""; TAG=""
  [ "$VAR" = "unifL1" ] && EXTRA="--l1_uniform" && TAG="-unifL1"
  [ "$VAR" = "noMLP1" ] && EXTRA="--l1_no_mlp" && TAG="-noMLP1"
  EVAL_ITERS=50; [ "$L" -ge 8192 ] && EVAL_ITERS=20
  OUT=out-sweep-1H-L${L}${TAG}-s${SEED}
  if [ -f "$OUT/run/final_ckpt.pt" ]; then echo "skip $OUT"; continue; fi
  echo "$(date) start $OUT on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_2layer_mechanism.py --regime R0 \
    --block_size $L --use_flash --batch_size $BATCH --gradient_accumulation_steps $ACCUM \
    --n_head 1 --seed $SEED --max_iters 20000 --max_iters_override 20000 \
    --eval_iters $EVAL_ITERS --eval_interval 500 --no_compile --learning_rate 3e-4 \
    --dtype $DTYPE $EXTRA --out_dir $OUT --resume_dir $OUT/run \
    > ../logs/sweep_$(basename $OUT).log 2>&1
  echo "$(date) done $OUT exit $?"
done
