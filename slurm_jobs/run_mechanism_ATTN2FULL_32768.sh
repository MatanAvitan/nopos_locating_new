#!/bin/bash
#SBATCH --job-name=A2F-32k
#SBATCH --output=logs/slurm_mechanism_ATTN2FULL_32768_%j.out
#SBATCH --error=logs/slurm_mechanism_ATTN2FULL_32768_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# P5 (exps.md): ATTN2-FULL at L=32768 — context-length scaling test.
# Same recipe as the canonical L=8192 run (LR 3e-4, 131k tokens/iter);
# --resume_dir makes requeues/resubmissions continue from last_ckpt.pt.
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R0 \
    --block_size 32768 \
    --use_flash \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --n_head 1 \
    --max_iters 20000 \
    --max_iters_override 20000 \
    --eval_iters 20 \
    --eval_interval 500 \
    --wandb \
    --no_compile \
    --learning_rate 3e-4 \
    --out_dir out-mechanism-ATTN2FULL-32768 \
    --resume_dir out-mechanism-ATTN2FULL-32768/R0/run1
