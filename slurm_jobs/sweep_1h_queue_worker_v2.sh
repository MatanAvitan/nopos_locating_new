#!/bin/bash
# Lock-aware worker for the 1H length sweep and Layer-1 controls.
# Usage: sweep_1h_queue_worker.sh <gpu> <dtype> <queue_file> [<wait_pid>]
# Each queue line is "L:seed:variant" (variant in {std, unifL1, noMLP1}).
# A worker claims a job by creating logs/locks/<out_dir> (mkdir is atomic), so
# any number of workers can share one queue. If <wait_pid> is given, the
# worker first waits for that process (a run already on this GPU) to exit.
# Recipe: 131072 tokens per optimizer step (batch * 4 * L), lr 3e-4, 20000 iters.
GPU=$1; DTYPE=$2; QUEUE=$3; WAIT=$4
ROOT=/home/nlp/matan_avitan/git/nopos_locating_new
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
mkdir -p $ROOT/logs/locks
if [ -n "$WAIT" ]; then while kill -0 $WAIT 2>/dev/null; do sleep 60; done; fi
cd $ROOT/nanoGPT
while read -r job; do
  [ -z "$job" ] && continue
  IFS=: read L SEED VAR <<< "$job"
  EXTRA=""; TAG=""
  [ "$VAR" = "unifL1" ] && EXTRA="--l1_uniform" && TAG="-unifL1"
  [ "$VAR" = "noMLP1" ] && EXTRA="--l1_no_mlp" && TAG="-noMLP1"
  [ "$VAR" = "relu" ] && EXTRA="--activation relu" && TAG="-relu"
  OUT=out-sweep-1H-L${L}${TAG}-s${SEED}
  mkdir $ROOT/logs/locks/$OUT 2>/dev/null || continue
  EVAL_ITERS=50; [ "$L" -ge 8192 ] && EVAL_ITERS=20
  echo "$(date) start $OUT on $(hostname) GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_2layer_mechanism.py --regime R0 \
    --block_size $L --use_flash --batch_size $((32768 / L)) --gradient_accumulation_steps 4 \
    --n_head 1 --seed $SEED --max_iters 20000 --max_iters_override 20000 \
    --eval_iters $EVAL_ITERS --eval_interval 500 --no_compile --learning_rate 3e-4 \
    --dtype $DTYPE $EXTRA --out_dir $OUT --resume_dir $OUT/run \
    > $ROOT/logs/sweep_$OUT.log 2>&1
  echo "$(date) done $OUT exit $?"
done < $QUEUE
echo "$(date) queue empty"
