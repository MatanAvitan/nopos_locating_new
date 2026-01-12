#!/bin/bash
#SBATCH --job-name=nope_probe
#SBATCH --output=logs/slurm_comprehensive_probe_a100_%j.out
#SBATCH --error=logs/slurm_comprehensive_probe_a100_%j.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# NoPE Analysis: Comprehensive Probe Analysis
# Trains linear and MLP probes at all activation points
# Running on A100-12h partition for longer runtime

cd /home/nlp/matan_avitan/git/nopos_locating_new

echo "Starting Comprehensive Probe Analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# Use larger sample size for thorough analysis
python analysis_scripts/comprehensive_probe_analysis.py --n_samples 30000 --seq_len 64

echo "Finished at: $(date)"
