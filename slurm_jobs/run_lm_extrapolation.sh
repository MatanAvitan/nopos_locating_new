#!/bin/bash
#SBATCH --job-name=lm6bos-extrap
#SBATCH --output=logs/slurm_lm_extrap_%j.out
#SBATCH --error=logs/slurm_lm_extrap_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# P6: length extrapolation of lm6bos (train L=128 -> eval up to 2048).
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new
python analysis_scripts/mechanism_pipeline/lm_extrapolation.py \
    --lengths 128,256,512,1024,2048
