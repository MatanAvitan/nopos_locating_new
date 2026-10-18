#!/bin/bash
#SBATCH --job-name=R2-1h-8192-smlp
#SBATCH --output=logs/slurm_R2_1h_8192_smlp_%j.out
#SBATCH --error=logs/slurm_R2_1h_8192_smlp_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
python train_2layer_mechanism.py \
    --regime R2 \
    --r2_attn_mlp_head_only \
    --block_size 8192 \
    --n_head 1 \
    --use_flash \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_iters 40000 \
    --max_iters_override 40000 \
    --eval_iters 50 \
    --eval_interval 500 \
    --no_compile \
    --out_dir out-mechanism-R2-8192-attnmlphead-strict-b200
