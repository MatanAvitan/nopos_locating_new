#!/bin/bash
#SBATCH --job-name=R2-12h-32k
#SBATCH --output=logs/slurm_mechanism_R2_12h_32k_%j.out
#SBATCH --error=logs/slurm_mechanism_R2_12h_32k_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# ATTN2-12H (R2) with 32K context — test if multi-head fixes scaling degradation
# Same trainable params as 1-head (2.36M), only the attention structure differs
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R2 \
    --r2_attn_head_only \
    --block_size 32768 \
    --use_flash \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --n_head 12 \
    --max_iters 5000 \
    --max_iters_override 5000 \
    --eval_iters 50 \
    --eval_interval 500 \
    --wandb \
    --no_compile \
    --out_dir out-mechanism-R2-12h-32k
