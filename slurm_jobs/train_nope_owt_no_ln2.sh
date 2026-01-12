#!/bin/bash
#SBATCH --job-name=train_no_ln2
#SBATCH --output=logs/slurm_train_no_ln2_%j.out
#SBATCH --error=logs/slurm_train_no_ln2_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Train NoPE transformer WITHOUT LN2 on OpenWebText
# Tests hypothesis: Does LN2 hurt positional encoding?

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT

echo "Starting NoPE training (NO LN2) on OpenWebText at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

# Check if data exists
if [ ! -f "data/openwebtext/train.bin" ]; then
    echo "ERROR: OpenWebText data not prepared! Run prepare_openwebtext.sh first."
    exit 1
fi

# Run training
python train_nope.py config/train_nope_owt_no_ln2.py

echo "Finished at $(date)"
echo "Checkpoint saved to: out-nope-owt-no-ln2/"
