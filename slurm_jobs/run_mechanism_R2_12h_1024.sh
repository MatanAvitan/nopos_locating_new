#!/bin/bash
#SBATCH --job-name=R2-12h-1024
#SBATCH --output=logs/slurm_mechanism_R2_12h_1024_%j.out
#SBATCH --error=logs/slurm_mechanism_R2_12h_1024_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ATTN2-12H (R2) with 1024 context — test multi-head noise-averaging hypothesis
# Same architecture as ATTN2-1H but with 12 heads
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R2 \
    --r2_attn_head_only \
    --block_size 1024 \
    --use_flash \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --n_head 12 \
    --max_iters 40000 \
    --max_iters_override 40000 \
    --eval_iters 50 \
    --eval_interval 500 \
    --wandb \
    --no_compile \
    --out_dir out-mechanism-R2-12h-1024
