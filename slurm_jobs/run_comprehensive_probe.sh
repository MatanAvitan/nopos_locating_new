#!/bin/bash
#SBATCH --job-name=nope_probe
#SBATCH --output=logs/slurm_comprehensive_probe_%j.out
#SBATCH --error=logs/slurm_comprehensive_probe_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00

# NoPE Analysis: Comprehensive Probe Analysis
# Trains linear and MLP probes at all activation points

cd /home/nlp/matan_avitan/git/nopos_locating_new

echo "Starting Comprehensive Probe Analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Use reduced sample size for faster execution
python analysis_scripts/comprehensive_probe_analysis.py --n_samples 10000 --seq_len 64

echo "Finished at: $(date)"
