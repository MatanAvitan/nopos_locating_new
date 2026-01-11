#!/bin/bash
#SBATCH --job-name=single_sample
#SBATCH --output=logs/slurm_single_sample_%j.out
#SBATCH --error=logs/slurm_single_sample_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "Starting Single-Sample Position Analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/nlp/matan_avitan/git/nopos_locating_new

# Activate conda environment
source ~/.bashrc
conda activate py312

# Run the analysis
python analysis_scripts/single_sample_position_analysis.py \
    --n_samples 5000 \
    --seq_len 64 \
    --d_model 1024 \
    --batch_size 256

echo ""
echo "End time: $(date)"
echo "Job complete!"
