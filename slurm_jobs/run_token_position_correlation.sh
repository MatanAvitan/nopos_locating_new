#!/bin/bash
#SBATCH --job-name=nope_token_corr
#SBATCH --output=logs/slurm_token_position_correlation_%j.out
#SBATCH --error=logs/slurm_token_position_correlation_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# NoPE Analysis: Token-Position Correlation in Natural Language
# Analyzes WikiText-103 to show how token distributions vary with position

cd /home/nlp/matan_avitan/git/nopos_locating_new

# Activate environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate nope_env

echo "Starting Token-Position Correlation Analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python analysis_scripts/token_position_correlation_natural_language.py

echo "Finished at: $(date)"
