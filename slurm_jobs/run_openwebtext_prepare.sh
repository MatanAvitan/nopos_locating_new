#!/bin/bash
#SBATCH --job-name=owt_prepare
#SBATCH --output=logs/slurm_owt_prepare_%j.out
#SBATCH --error=logs/slurm_owt_prepare_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting OpenWebText Dataset Preparation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext

# Run the preparation script
python prepare.py

echo "Done!"
echo "End time: $(date)"
