#!/bin/bash
#SBATCH --job-name=long_ctx_analysis
#SBATCH --output=logs/slurm_long_ctx_%j.out
#SBATCH --error=logs/slurm_long_ctx_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "Starting Long Context Position Encoding Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

cd /home/nlp/matan_avitan/git/nopos_locating_new

# Run the analysis
python analysis_scripts/long_context_analysis.py

echo "Done!"
echo "End time: $(date)"
