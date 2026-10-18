#!/bin/bash
#SBATCH --job-name=LM6BOS-ana
#SBATCH --output=logs/slurm_lm6bos_analysis_%j.out
#SBATCH --error=logs/slurm_lm6bos_analysis_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# P4 (exps.md): full LM analysis on the BOS-trained checkpoint —
# mechanism measurements (BOS + no-BOS control) and the P2 intervention
# suite, same protocol as lm6_L128. Each step overwrites its own run dir,
# so a requeue simply reruns from the top (steps are ~minutes each).
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new
CKPT=nanoGPT/out-lm-6layer-fulltrain-bos/ckpt.pt
python analysis_scripts/mechanism_pipeline/lm_mechanism.py \
    --ckpt $CKPT --run-id lm6bos_L128 --context-length 128
python analysis_scripts/mechanism_pipeline/lm_mechanism.py \
    --ckpt $CKPT --run-id lm6bos_L128_nobos --context-length 128 --no-bos
python analysis_scripts/mechanism_pipeline/lm_interventions.py \
    --ckpt $CKPT --run-id lm6bos_L128 --context-length 128
