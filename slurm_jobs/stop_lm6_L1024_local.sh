#!/bin/bash
# On dsinlp01: stop the local 6-layer L = 1024 runs for the given seeds, and their launcher shells.
for SEED in "$@"; do
    PIDS=$(ps -u $USER -o pid=,args= | grep "out_dir=out-lm-6layer-fulltrain-bos-L1024-seed$SEED " | grep -v grep | awk '{print $1}')
    SH=$(ps -u $USER -o pid=,args= | grep "run_lm6_bos_L1024_local.sh [0-9]* $SEED$" | grep -v grep | awk '{print $1}')
    echo "seed $SEED: train $PIDS, launcher $SH"
    kill $SH $PIDS 2>/dev/null
done
sleep 15
ps -u $USER -o args= | grep "[t]rain_nope" | grep -o "out_dir=[^ ]*"
