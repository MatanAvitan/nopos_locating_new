#!/bin/bash
# Start the four L = 2048 runs on dsinlp01 GPUs 0, 1, 3, 4 (at most 4 jobs on one server).
S=/home/nlp/matan_avitan/git/nopos_locating_new/slurm_jobs/run_lm_bos_L2048_local.sh
LOG=/home/nlp/matan_avitan/git/nopos_locating_new/logs
for a in "0 6 1" "1 6 2" "3 2 1" "4 2 2"; do
    set -- $a
    echo "=== START $(date) with log_attention_stats=False" >> $LOG/lm_bos_L2048_${2}l_seed$3.log
    setsid nohup bash $S $1 $2 $3 > /dev/null 2>&1 < /dev/null &
done
