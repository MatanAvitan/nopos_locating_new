#!/bin/bash
#SBATCH --job-name=train_pe
#SBATCH --output=logs/slurm_train_pe_%j.out
#SBATCH --error=logs/slurm_train_pe_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Train BASELINE transformer WITH positional embeddings on OpenWebText
# Standard transformer for comparison with NoPE variants

cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT

echo "Starting BASELINE training (with PE) on OpenWebText at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

# Check if data exists
if [ ! -f "data/openwebtext/train.bin" ]; then
    echo "ERROR: OpenWebText data not prepared! Run prepare_openwebtext.sh first."
    exit 1
fi

# Run training
python train_nope.py config/train_baseline_owt_pe.py

echo "Finished at $(date)"
echo "Checkpoint saved to: out-baseline-owt-pe/"
