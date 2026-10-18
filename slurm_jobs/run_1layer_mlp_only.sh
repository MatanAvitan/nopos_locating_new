#!/bin/bash
#SBATCH --job-name=nope-1l-mlp
#SBATCH --output=logs/slurm_1layer_mlp_only_%j.out
#SBATCH --error=logs/slurm_1layer_mlp_only_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# 1-layer NoPE MLP-only position decoding on dgx-b200-01

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_1layer_mlp_only.py --wandb --max_iters 80000
