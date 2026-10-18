#!/bin/bash
#SBATCH --job-name=nope-owt-bn2-h200
#SBATCH --output=logs/slurm_nope_bn2_%j.out
#SBATCH --error=logs/slurm_nope_bn2_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_nope.py config/train_nope_owt_bn2.py