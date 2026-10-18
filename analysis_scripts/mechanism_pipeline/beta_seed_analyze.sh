#!/bin/bash
# Analyse the extra seeds (1H / 12H at L = 1024, BOS-trained LMs) when their
# training has finished: beta_extract -> beta_analyze -> beta_v2.
# A model is claimed with an atomic mkdir lock, so several runners can share work.
# Usage: beta_seed_analyze.sh <gpu>
GPU=$1
PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
ROOT=/home/nlp/matan_avitan/git/nopos_locating_new
cd $ROOT/analysis_scripts/mechanism_pipeline
mkdir -p $ROOT/logs/locks
KEYS="lmbos_s42 seed_1h_s3 seed_12h_s1 seed_12h_s2 seed_12h_s3 lmbos_s1 lmbos_s2 lmbos_s3 lmbos_s4 lmbos_s5 lmbos_s6 lmbos_s7 lmbos_s8 lmbos_s9"
done_training() {   # toy: final_ckpt.pt exists; LM: its log reached iter 20000 (seed 42 is long done)
  case $1 in
    lmbos_s42) return 0 ;;
    lmbos_s*) grep -qa "^iter 20000:" $ROOT/logs/lm_bos_seed${1#lmbos_s}.log 2>/dev/null ;;
    seed_12h_s*) [ -f $ROOT/nanoGPT/out-mechanism-12H-1024-seed${1#seed_12h_s}/run/final_ckpt.pt ] ;;
    seed_1h_s*) [ -f $ROOT/nanoGPT/out-mechanism-1H-1024-seed${1#seed_1h_s}/run/final_ckpt.pt ] ;;
  esac
}
while true; do
  left=0
  for k in $KEYS; do
    run=$($PY -c "from beta_common import MODELS; print(MODELS['$k'][3])" 2>/dev/null)
    [ -f "$ROOT/results/mechanism/$run/beta_v2/summary.json" ] && continue
    left=$((left + 1))
    done_training $k || continue
    mkdir $ROOT/logs/locks/ana_$k 2>/dev/null || continue
    BS=8; case $k in lmbos_*) BS=32 ;; seed_12h_*) BS=2 ;; esac
    echo "$(date) analyse $k"
    CUDA_VISIBLE_DEVICES=$GPU $PY beta_extract.py --model $k --batch-size $BS --device cuda:0 > $ROOT/logs/ana_$k.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$GPU $PY beta_analyze.py --model $k >> $ROOT/logs/ana_$k.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$GPU $PY beta_v2.py --model $k --batch-size $BS --device cuda:0 >> $ROOT/logs/ana_$k.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$GPU $PY story_evidence.py --model $k --batch-size $BS --device cuda:0 >> $ROOT/logs/ana_$k.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$GPU $PY softmax_beta_check.py --model $k --batch-size $BS --device cuda:0 >> $ROOT/logs/ana_$k.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$GPU $PY causal_story.py --model $k --batch-size $BS --device cuda:0 >> $ROOT/logs/ana_$k.log 2>&1
    echo "$(date) done $k exit $?"
  done
  [ $left -eq 0 ] && break
  sleep 300
done
echo "all analysed"
