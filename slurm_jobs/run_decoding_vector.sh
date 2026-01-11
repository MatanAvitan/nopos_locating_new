#!/bin/bash
#SBATCH --job-name=nope_decoding
#SBATCH --output=logs/slurm_decoding_vector_%j.out
#SBATCH --error=logs/slurm_decoding_vector_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# NoPE Analysis: Decoding Vector Experiments
# Validates decoding vector construction

cd /home/nlp/matan_avitan/git/nopos_locating_new

echo "Starting Decoding Vector Experiments..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python analysis_scripts/decoding_vector_experiments.py

echo "Finished at: $(date)"
