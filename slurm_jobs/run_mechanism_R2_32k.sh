#!/bin/bash
#SBATCH --job-name=R2-32k
#SBATCH --output=logs/slurm_mechanism_R2_32k_%j.out
#SBATCH --error=logs/slurm_mechanism_R2_32k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# ATTN2-1H (R2) with 32K context — geometric clock at long sequences
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R2 \
    --r2_attn_head_only \
    --block_size 32768 \
    --use_flash \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --n_head 1 \
    --max_iters 10000 \
    --max_iters_override 10000 \
    --wandb \
    --out_dir out-mechanism-R2-32k
