# R0 Full 12-Head Model

## Overview

Full 2-layer transformer with 12 attention heads per layer, trained in regime R0
(position prediction at position 0 via BOS token).

## Architecture

| Parameter | Value |
|-----------|-------|
| `n_layer` | 2 |
| `n_head` | 12 |
| `n_embd` | 768 |
| `block_size` | 128 |
| Positional Embeddings | None (NoPE) |
| MLP | Full (in both layers) |

## Training

```bash
cd nanoGPT
python train_2layer_mechanism.py --regime R0 --wandb
```

### Key Training Parameters
- Dataset: OpenWebText
- Batch size: 32 × 4 gradient accumulation = 128 effective
- Learning rate: 6e-4 with cosine decay
- Warmup: 100 iterations
- Max iterations: 30,000

## Checkpoint

- Size: ~636 MB
- Location: `model_backups/R0_FULL-12H/best_ckpt.pt`

## Key Findings

1. **BOS-Anchored Attention**: Several heads develop near-uniform attention with BOS as anchor point
2. **Position Encoding**: Position information encoded via attention-weighted averaging
3. **Extrapolation**: Good generalization to sequences longer than training length
