#!/bin/bash
#SBATCH --job-name=ln_geom
#SBATCH --output=logs/slurm_ln_geometry_%j.out
#SBATCH --error=logs/slurm_ln_geometry_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "Starting LayerNorm Geometry Analysis"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /home/nlp/matan_avitan/git/nopos_locating_new

# Activate conda if needed
source ~/.bashrc

python analysis_scripts/layernorm_geometry_analysis.py

echo "Completed at: $(date)"
