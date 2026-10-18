#!/bin/bash
#SBATCH --job-name=A2F-1024-lr3
#SBATCH --output=logs/slurm_mechanism_ATTN2FULL_1024_lr3_%j.out
#SBATCH --error=logs/slurm_mechanism_ATTN2FULL_1024_lr3_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ATTN2-FULL: fully trained (R0) 2-layer SINGLE-HEAD model, L=1024.
# Missing cell of the (head count x training freedom) design; pre-registered
# in overleaf/nopos_nips/exps.md "P1: ATTN2-FULL". Matches
# run_mechanism_R0_1024.sh exactly except --n_head 1.
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R0 \
    --block_size 1024 \
    --use_flash \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --n_head 1 \
    --max_iters 20000 \
    --max_iters_override 20000 \
    --eval_iters 50 \
    --eval_interval 500 \
    --wandb \
    --no_compile \
    --learning_rate 3e-4 --out_dir out-mechanism-ATTN2FULL-1024-lr3e4
