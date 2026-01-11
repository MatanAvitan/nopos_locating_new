#!/bin/bash
#SBATCH --job-name=norm_int
#SBATCH --output=logs/slurm_norm_intervention_%j.out
#SBATCH --error=logs/slurm_norm_intervention_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "Starting Norm Intervention Experiment"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /home/nlp/matan_avitan/git/nopos_locating_new

# Activate conda if needed
source ~/.bashrc

python analysis_scripts/norm_intervention_experiment.py

echo "Completed at: $(date)"
