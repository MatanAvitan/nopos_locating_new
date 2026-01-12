#!/bin/bash
#SBATCH --job-name=prep_owt
#SBATCH --output=logs/slurm_prep_owt_%j.out
#SBATCH --error=logs/slurm_prep_owt_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00

# Prepare OpenWebText dataset for NoPE training
# This downloads ~54GB and processes it to ~17GB train.bin + ~8.5MB val.bin
# Should take several hours

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT

echo "Starting OpenWebText preparation at $(date)"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"

# Run the preparation script
python data/openwebtext/prepare.py

echo "Finished at $(date)"
echo "Output files:"
ls -lh data/openwebtext/*.bin 2>/dev/null || echo "No .bin files found yet"
