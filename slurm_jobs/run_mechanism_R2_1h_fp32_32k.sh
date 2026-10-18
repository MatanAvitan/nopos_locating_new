#!/bin/bash
#SBATCH --job-name=R2-1h-fp32-32k
#SBATCH --output=logs/slurm_mechanism_R2_1h_fp32_32k_%j.out
#SBATCH --error=logs/slurm_mechanism_R2_1h_fp32_32k_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# ATTN2-1H (R2) with 32K context in float32 — test if bf16 precision limits position resolution
# At L=32K, adjacent positions differ by ~1/32K in mixing coefficient;
# bf16 mantissa (10 bits) may not resolve this.
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R2 \
    --r2_attn_head_only \
    --block_size 32768 \
    --use_flash \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --n_head 1 \
    --max_iters 10000 \
    --max_iters_override 10000 \
    --eval_iters 50 \
    --eval_interval 500 \
    --dtype float32 \
    --wandb \
    --no_compile \
    --out_dir out-mechanism-R2-1h-fp32-32k
