#!/bin/bash
#SBATCH --job-name=A2F-64k-lr15
#SBATCH --output=logs/slurm_mechanism_ATTN2FULL_65536_lr15_%j.out
#SBATCH --error=logs/slurm_mechanism_ATTN2FULL_65536_lr15_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# P5 criterion (l) retry: L=65536 failed to converge at LR 3e-4 (val R2 0.25);
# retry once at LR 1.5e-4, same recipe otherwise.
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R0 \
    --block_size 65536 \
    --use_flash \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --n_head 1 \
    --max_iters 20000 \
    --max_iters_override 20000 \
    --eval_iters 20 \
    --eval_interval 500 \
    --wandb \
    --no_compile \
    --learning_rate 1.5e-4 \
    --out_dir out-mechanism-ATTN2FULL-65536-lr15 \
    --resume_dir out-mechanism-ATTN2FULL-65536-lr15/R0/run1
