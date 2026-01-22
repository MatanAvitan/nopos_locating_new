#!/bin/bash
#SBATCH --job-name=baseline-pe-12l
#SBATCH --output=logs/slurm_train_pe_12layer_%j.out
#SBATCH --error=logs/slurm_train_pe_12layer_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# 12-layer baseline transformer WITH positional embeddings on OpenWebText

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_nope.py config/train_baseline_owt_pe_12layer.py
