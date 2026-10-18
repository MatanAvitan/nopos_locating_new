#!/bin/bash
# On dsinlp01: finish 6-layer L = 1024 seed 1 on GPU 3 and start seed 3 on GPU 4 now.
# Start seed 4 on GPU 0 after the 6-layer L = 2048 seed 1 run there ends (at most 4 jobs on this server).
S=/home/nlp/matan_avitan/git/nopos_locating_new/slurm_jobs/run_lm6_bos_L1024_local.sh
setsid nohup bash $S 3 1 > /dev/null 2>&1 < /dev/null &
setsid nohup bash $S 4 3 > /dev/null 2>&1 < /dev/null &
setsid nohup bash -c "while ps -u \$USER -o cmd | grep -q '[o]ut-lm-6layer-fulltrain-bos-L2048-seed1'; do sleep 60; done; bash $S 0 4" > /dev/null 2>&1 < /dev/null &
