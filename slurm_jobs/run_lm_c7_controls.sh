#!/bin/bash
#SBATCH --job-name=LM-C7ctrl
#SBATCH --output=logs/slurm_lm_c7_controls_%j.out
#SBATCH --error=logs/slurm_lm_c7_controls_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# P8 (exps.md): the C7 replacement controls pre-registered in exps.md C7 but
# never implemented -- fixed-random / nonBOS-direction / norm-only replacement
# of the block-0 BOS OV image, all holding the block-1 attention weights at
# their realized values. Reruns the full P2 intervention suite (the shared
# interventions are deterministic and must reproduce their recorded values,
# which doubles as a reproducibility check).
#
# Resume: each model's run overwrites its own run dir atomically at the end,
# so a requeue reruns whichever model has not yet recorded C7_ctrl_random.
# Steps are ~10 min each.
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new

run_one () {
    local ckpt=$1 run_id=$2
    if python -c "
import json,sys
p='results/mechanism/$run_id/summary.json'
try: s=json.load(open(p))['lm_interventions']
except Exception: sys.exit(1)
sys.exit(0 if 'C7_ctrl_random' in s else 1)
"; then
        echo "[skip] $run_id already has the C7 controls"
        return 0
    fi
    echo "[run ] $run_id"
    python analysis_scripts/mechanism_pipeline/lm_interventions.py \
        --ckpt "$ckpt" --run-id "$run_id" --context-length 128
}

run_one nanoGPT/out-lm-6layer-fulltrain-ddp/ckpt.pt lm6_L128
run_one nanoGPT/out-lm-6layer-fulltrain-bos/ckpt.pt  lm6bos_L128
