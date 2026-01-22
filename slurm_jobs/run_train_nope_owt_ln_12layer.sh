#!/bin/bash
#SBATCH --job-name=nope-ln-12l
#SBATCH --output=logs/slurm_train_ln_12layer_%j.out
#SBATCH --error=logs/slurm_train_ln_12layer_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# 12-layer NoPE transformer with LayerNorm on OpenWebText

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_nope.py config/train_nope_owt_ln_12layer.py
