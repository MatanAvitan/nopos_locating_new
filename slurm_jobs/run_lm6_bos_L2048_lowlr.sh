#!/bin/bash
#SBATCH --job-name=LM6-BOS-L2048-lowlr
#SBATCH --output=/home/nlp/matan_avitan/git/nopos_locating_new/logs/slurm_lm6_bos_L2048_lowlr_%j.out
#SBATCH --error=/home/nlp/matan_avitan/git/nopos_locating_new/logs/slurm_lm6_bos_L2048_lowlr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue

# Rerun of the 6-layer BOS LM at L = 2048, seed 1. The first run became unstable after step 5500.
# Changes: explicit attention (log_attention_stats stays True, as in every stable run) and a lower
# peak learning rate, 1.5e-4 (min_lr 1.5e-5). Tokens per step: 16 x 4 x 2048 = 131072.
# Runs on SLURM (b200 env) or locally: run_lm6_bos_L2048_lowlr.sh [gpu]. Resumes from ckpt.pt.
if [ -n "$1" ]; then export CUDA_VISIBLE_DEVICES=$1; PY=/home/nlp/matan_avitan/.conda/envs/py312/bin/python
else eval "$(conda shell.bash hook)"; conda activate b200; PY=python; fi
cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
OUT=out-lm-6layer-fulltrain-bos-L2048-lr1.5e-4-seed1
RESUME=""
if [ -f "$OUT/ckpt.pt" ]; then
    ITER=$($PY -c "import torch; print(torch.load('$OUT/ckpt.pt', map_location='cpu', weights_only=False, mmap=True)['iter_num'])")
    if [ "$ITER" -ge 20000 ]; then echo "already at iteration $ITER"; exit 0; fi
    RESUME="--init_from=resume"
fi
$PY -u train_nope.py config/train_lm_6layer_fulltrain_bos_ddp.py \
    --seed=1 --out_dir=$OUT --block_size=2048 --batch_size=16 --gradient_accumulation_steps=4 \
    --learning_rate=1.5e-4 --min_lr=1.5e-5 --wandb_log=False --dtype=bfloat16 $RESUME
