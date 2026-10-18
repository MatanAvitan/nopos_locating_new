#!/bin/bash
#SBATCH --job-name=nope-owt-ln-h200
#SBATCH --output=logs/slurm_nope_ln_%j.out
#SBATCH --error=logs/slurm_nope_ln_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_nope.py config/train_nope_owt_ln.py