#!/bin/bash
#SBATCH --job-name=LM6-BOS
#SBATCH --output=logs/slurm_lm6_fulltrain_bos_%j.out
#SBATCH --error=logs/slurm_lm6_fulltrain_bos_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# P4 (exps.md): 6-layer NoPE LM trained with a fixed BOS (50256) at position 0
# of every training window — same data/optimizer/seed as
# out-lm-6layer-fulltrain-ddp, matching the toys' anchored protocol.
# Resumes from out-lm-6layer-fulltrain-bos/ckpt.pt if requeued.
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
RESUME=""
if [ -f out-lm-6layer-fulltrain-bos/ckpt.pt ]; then
    RESUME="--init_from=resume"
fi
python train_nope.py config/train_lm_6layer_fulltrain_bos_ddp.py $RESUME
