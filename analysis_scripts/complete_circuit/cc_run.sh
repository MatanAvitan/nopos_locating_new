#!/bin/bash
# Run one complete-circuit script over several models on one GPU.
# A model is skipped when its output marker exists, so a rerun is cheap.
# Usage: [EXTRA="--split test"] [TAG=_test] cc_run.sh <gpu> <script> <marker-file-name> <model> [<model> ...]
GPU=$1; SCRIPT=$2; MARK=$3; shift 3
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
ROOT=/home/nlp/matan_avitan/git/nopos_locating_new
cd $ROOT/analysis_scripts/complete_circuit
STAGE=${SCRIPT%.py}; STAGE=${STAGE#cc_}
for k in "$@"; do
  if [ -f "$ROOT/results/mechanism/complete_circuit/$STAGE/$k/$MARK" ]; then echo "skip $k"; continue; fi
  echo "$(date +%T) start $k"
  BS=4; case $k in 12h_*) BS=2 ;; lm*) BS=16 ;; esac
  CUDA_VISIBLE_DEVICES=$GPU $PY $SCRIPT --model $k --device cuda:0 --batch-size $BS $EXTRA > $ROOT/logs/cc_${STAGE}_$k${TAG}.log 2>&1
  echo "$(date +%T) done $k exit $?"
done
echo "all done"
