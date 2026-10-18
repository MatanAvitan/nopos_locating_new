#!/bin/bash
#SBATCH --job-name=eval-3x4
#SBATCH --output=logs/slurm_eval_3x4_%j.out
#SBATCH --error=logs/slurm_eval_3x4_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

eval "$(conda shell.bash hook)"
conda activate b200
cd /home/nlp/matan_avitan/git/nopos_locating_new
python analysis_scripts/experiments/eval_three_models_four_contexts.py
