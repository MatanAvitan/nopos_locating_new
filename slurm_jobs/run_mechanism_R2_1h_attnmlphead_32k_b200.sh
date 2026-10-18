#!/bin/bash
#SBATCH --job-name=R2-1h-32k-smlp
#SBATCH --output=logs/slurm_R2_1h_32k_smlp_%j.out
#SBATCH --error=logs/slurm_R2_1h_32k_smlp_%j.err
#SBATCH --partition=B200-8h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00

eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R2 \
    --r2_attn_mlp_head_only \
    --block_size 32768 \
    --n_head 1 \
    --use_flash \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_iters 10000 \
    --max_iters_override 10000 \
    --eval_iters 20 \
    --eval_interval 250 \
    --no_compile \
    --out_dir out-mechanism-R2-32k-attnmlphead-strict-b200
