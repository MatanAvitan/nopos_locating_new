#!/bin/bash
#SBATCH --job-name=attn_pat
#SBATCH --output=logs/slurm_attention_pattern_%j.out
#SBATCH --error=logs/slurm_attention_pattern_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "Starting Attention Pattern Analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd /home/nlp/matan_avitan/git/nopos_locating_new
source ~/.bashrc
conda activate py312

python analysis_scripts/attention_pattern_analysis.py \
    --n_samples 5000 \
    --seq_len 64 \
    --d_model 1024

echo "End time: $(date)"
echo "Job complete!"
