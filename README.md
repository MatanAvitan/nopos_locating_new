# How Causal Transformers Encode Position Without Positional Embeddings

**Authors**: Matan Avitan, Ido Nachum, Yoav Goldberg

This repository contains code for our paper on understanding how causal transformers encode absolute position without explicit positional embeddings (NoPE - No Positional Embeddings).

## Key Contribution

We discover a **complete 4-component circuit** explaining implicit position encoding:

1. **Token Embedding Diversity**: Sufficient vocabulary provides diverse representations
2. **Near-Uniform Causal Attention**: Proper initialization yields attention where variance decays as 1/(i+1)
3. **LayerNorm Paradox**: Position survives normalization through population statistics
4. **MLP Decoding**: Uses decoding vector exploiting orthogonality of random embeddings

## Repository Structure

```
├── analysis_scripts/          # Analysis and figure generation
│   ├── core/                  # Shared utilities
│   ├── decoding/              # Decoding vector analysis
│   ├── attention/             # Attention pattern analysis
│   ├── layernorm/             # LayerNorm mechanism
│   ├── probing/               # Linear probing experiments
│   ├── experiments/           # R0/R2 specific analyses
│   │   ├── r0_analysis/
│   │   ├── r2_analysis/
│   │   ├── extrapolation/
│   │   └── interventions/
│   ├── figures/               # Paper figure generation
│   └── archive/               # Old/experimental scripts
├── analysis_notebooks/        # Jupyter notebooks for exploration
├── experiments/               # Reproduction documentation
│   ├── R0_FULL-12H/          # Full 12-head model
│   └── R2_ATTN2-1H/          # Single-head attention-only
├── model_backups/            # Checkpoint backups (not in git)
├── nanoGPT/                  # Training code (modified nanoGPT)
├── overleaf/                 # Paper LaTeX source
└── results/                  # Generated figures and data
```

## Reproducing Experiments

### Prerequisites

```bash
pip install torch numpy scipy matplotlib tiktoken wandb scikit-learn
```

### Data Preparation

```bash
cd nanoGPT/data/openwebtext
python prepare.py
```

### Training

See `experiments/` directory for detailed instructions:

```bash
# R0: Full 12-head model
cd nanoGPT && python train_2layer_mechanism.py --regime R0 --wandb

# R2: Single-head attention-only model
cd nanoGPT && python train_2layer_mechanism.py --regime R2 --n_head 1 --max_iters 40000 --r2_attn_head_only --wandb
```

## Citation

```bibtex
@article{avitan2026nope,
  title={How Causal Transformers Encode Position Without Positional Embeddings},
  author={Avitan, Matan and Nachum, Ido and Goldberg, Yoav},
  year={2026}
}
```

## Related Work

This work builds on prior research showing that causal language models can learn position without explicit positional encodings:
- Haviv et al. (EMNLP 2022): First showed causal LMs without PE learn position
- Chi et al. (ACL 2023): Identified self-attention variance as carrier of positional info
- Zuo et al. (COLING 2025): Proposed adjacency patterns in embeddings
- Kazemnejad et al. (NeurIPS 2023): Length generalization properties of NoPE

Our contribution is a complete mechanistic explanation with surgical validation.
