# R2 Attention-Only 1-Head Model

## Overview

2-layer transformer with 1 attention head per layer, with layer 2 being attention-only
(MLP frozen/removed). Trained in regime R2 for mechanistic analysis of position encoding.

## Architecture

| Parameter | Value |
|-----------|-------|
| `n_layer` | 2 |
| `n_head` | 1 |
| `n_embd` | 768 |
| `block_size` | 128 |
| Positional Embeddings | None (NoPE) |
| Layer 1 MLP | Full |
| Layer 2 MLP | Frozen/Removed |

## Training

```bash
cd nanoGPT
python train_2layer_mechanism.py \
    --regime R2 \
    --n_head 1 \
    --max_iters 40000 \
    --r2_attn_head_only \
    --wandb
```

### Key Training Parameters
- Dataset: OpenWebText
- Batch size: 32 × 4 gradient accumulation = 128 effective
- Learning rate: 6e-4 with cosine decay
- Warmup: 100 iterations
- Max iterations: 40,000

## Checkpoint

- Size: ~222 MB
- Location: `model_backups/R2_ATTN2-1H/best_ckpt.pt`

## Key Findings

1. **Low-Rank Write Subspace**: Single attention head develops interpretable position encoding
2. **Geometric Clock**: Position encoded through rotation in low-dimensional subspace
3. **Write Bottleneck**: Limited expressivity reveals cleaner positional mechanism
