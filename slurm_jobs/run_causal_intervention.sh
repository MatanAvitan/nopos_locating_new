#!/bin/bash
#SBATCH --job-name=nope_causal
#SBATCH --output=logs/slurm_causal_intervention_%j.out
#SBATCH --error=logs/slurm_causal_intervention_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# NoPE Analysis: Causal Intervention Experiments
# Uses mechanistic interpretability techniques

cd /home/nlp/matan_avitan/git/nopos_locating_new

echo "Starting Causal Intervention Experiments..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python analysis_scripts/causal_intervention_experiments.py

echo "Finished at: $(date)"
