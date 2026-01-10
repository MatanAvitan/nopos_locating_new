#!/bin/bash
#SBATCH --job-name=nope_higher_order
#SBATCH --output=logs/slurm_higher_order_stats_%j.out
#SBATCH --error=logs/slurm_higher_order_stats_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# NoPE Analysis: Higher-Order Statistics
# Investigates HOW position survives LayerNorm

cd /home/nlp/matan_avitan/git/nopos_locating_new

echo "Starting Higher-Order Statistics Analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python analysis_scripts/higher_order_statistics_analysis.py

echo "Finished at: $(date)"
