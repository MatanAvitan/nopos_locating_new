#!/bin/bash
#SBATCH --job-name=R0-32k
#SBATCH --output=logs/slurm_mechanism_R0_32k_%j.out
#SBATCH --error=logs/slurm_mechanism_R0_32k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# FULL-12H (R0) with 32K context — geometric clock at long sequences
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R0 \
    --block_size 32768 \
    --use_flash \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --n_head 12 \
    --max_iters 5000 \
    --max_iters_override 5000 \
    --wandb \
    --out_dir out-mechanism-R0-32k
