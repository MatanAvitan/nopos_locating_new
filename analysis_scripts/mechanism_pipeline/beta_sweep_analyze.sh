#!/bin/bash
# Analyse every sweep / 1-layer checkpoint whose training finished and whose
# beta_v2 summary is missing. Loops until all are done.
# Usage: beta_sweep_analyze.sh <gpu>
GPU=$1
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
ROOT=/home/nlp/matan_avitan/git/nopos_locating_new
cd $ROOT/analysis_scripts/mechanism_pipeline
KEYS=$($PY -c "from beta_common import MODELS; print(' '.join(k for k in MODELS if k.startswith(('sw_','h1_'))))")
while true; do
  left=0
  for k in $KEYS; do
    read ckpt L run <<< $($PY -c "from beta_common import MODELS, CHECKPOINTS; print(CHECKPOINTS['$k'], MODELS['$k'][2], MODELS['$k'][3])")
    final=$(dirname $ckpt)/final_ckpt.pt
    [ -f "$ROOT/results/mechanism/$run/beta_v2/summary.json" ] && continue
    left=$((left + 1))
    [ -f "$final" ] || continue
    mkdir -p $ROOT/logs/locks
    mkdir $ROOT/logs/locks/ana_$k 2>/dev/null || continue   # another runner has it
    BS=8; [ "$L" -le 256 ] && BS=32; [ "$L" -ge 2048 ] && BS=2; [ "$L" -ge 8192 ] && BS=1
    echo "$(date) analyse $k (L=$L)"
    CUDA_VISIBLE_DEVICES=$GPU $PY beta_extract.py --model $k --batch-size $BS --device cuda:0 > $ROOT/logs/ana_$k.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$GPU $PY beta_analyze.py --model $k >> $ROOT/logs/ana_$k.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$GPU $PY beta_v2.py --model $k --batch-size $BS --device cuda:0 >> $ROOT/logs/ana_$k.log 2>&1
    echo "$(date) done $k exit $?"
  done
  [ $left -eq 0 ] && break
  sleep 300
done
echo "all analysed"
