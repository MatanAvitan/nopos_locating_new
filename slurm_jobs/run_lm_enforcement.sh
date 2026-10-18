#!/bin/bash
#SBATCH --job-name=lm6bos-enforce
#SBATCH --output=logs/slurm_lm_enforce_%j.out
#SBATCH --error=logs/slurm_lm_enforce_%j.err
#SBATCH --partition=B200-4h,A100-4h,H200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# P7: enforce the trained alpha schedule on lm6bos at unseen positions.
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new
python analysis_scripts/mechanism_pipeline/lm_schedule_enforcement.py \
    --lengths 512,1024,2048
