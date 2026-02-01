# Experiments

This directory contains reproduction instructions for the key experiments in the paper.

## Models

| Name | Description | Heads | Layers | Special |
|------|-------------|-------|--------|---------|
| R0_FULL-12H | Full transformer, BOS-anchored position | 12 | 2 | Standard |
| R2_ATTN2-1H | Single-head, attention-only layer 2 | 1 | 2 | Mechanistic analysis |

## Prerequisites

1. Install dependencies:
   ```bash
   pip install torch numpy scipy matplotlib tiktoken wandb
   ```

2. Prepare OpenWebText data:
   ```bash
   cd nanoGPT/data/openwebtext
   python prepare.py
   ```

## Reproduction

See subdirectories for each experiment:
- `R0_FULL-12H/` - Full 12-head model (636 MB checkpoint)
- `R2_ATTN2-1H/` - Single-head attention-only model (222 MB checkpoint)

## Checkpoints

Pre-trained checkpoints are stored in `model_backups/` (not tracked in git due to size).
Original training locations:
- R0: `nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt`
- R2: `nanoGPT/out-2layer-mechanism-r2-1head-attnonly-fullblock-40k/R2/o4w7v8dv/best_ckpt.pt`
