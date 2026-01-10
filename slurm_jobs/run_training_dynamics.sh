#!/bin/bash
#SBATCH --job-name=nope_dynamics
#SBATCH --output=logs/slurm_training_dynamics_%j.out
#SBATCH --error=logs/slurm_training_dynamics_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# NoPE Analysis: Training Dynamics Analysis
# Analyzes checkpoints from training to see how position encoding emerges

cd /home/nlp/matan_avitan/git/nopos_locating_new

echo "Starting Training Dynamics Analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python analysis_scripts/training_dynamics_analysis.py

echo "Finished at: $(date)"
