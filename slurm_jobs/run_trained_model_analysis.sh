#!/bin/bash
#SBATCH --job-name=trained_analysis
#SBATCH --output=logs/slurm_trained_analysis_%j.out
#SBATCH --error=logs/slurm_trained_analysis_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "Starting Trained Model Direction vs Norm Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

cd /home/nlp/matan_avitan/git/nopos_locating_new

# Run the analysis
python analysis_scripts/trained_model_direction_norm.py

echo "Done!"
echo "End time: $(date)"
