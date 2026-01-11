#!/bin/bash
#SBATCH --job-name=dir_norm_indep
#SBATCH --output=logs/slurm_dir_norm_indep_%j.out
#SBATCH --error=logs/slurm_dir_norm_indep_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "Starting direction vs norm independence analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

cd /home/nlp/matan_avitan/git/nopos_locating_new

# Run the analysis
python analysis_scripts/direction_norm_independence.py

echo "Done!"
