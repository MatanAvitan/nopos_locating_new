# Project Context for Claude

## Paper Overview
**Title**: "How Causal Transformers Encode Position Without Positional Embeddings"
**Authors**: Matan Avitan, Ido Nachum, Yoav Goldberg
**Venue**: ACL paper (in preparation)

## Main Contribution
Discovery of a **complete 4-component circuit** explaining how causal transformers encode absolute position without explicit positional embeddings:

1. **Token Embedding Diversity**: Sufficient vocabulary provides diverse representations
2. **Near-Uniform Causal Attention**: Proper initialization (Xavier) yields attention where variance decays as 1/(i+1)
3. **LayerNorm Paradox**: Position survives normalization through population statistics (novel finding)
4. **MLP Decoding**: Uses decoding vector `w = W_V · Σ_j LN(E_j)` exploiting orthogonality of random embeddings

## Key Novel Findings
- **Information flow**: No signal → variance (post-attention) → mean (post-LN)
- **LayerNorm Paradox**: While LN normalizes each sample to zero mean/unit variance, population-level expectations differ by position due to position-correlated token distributions
- **Constructive decoding vector**: Explicit formula achieving Pearson r = 0.999
- **Vocabulary scaling**: min_samples ≈ 0.5 × vocab_size (linear relationship)

## Related Work (Important!)
Prior work this paper builds on and differentiates from:
- **Haviv et al. 2022 (EMNLP)**: First showed causal LMs without PE learn position (probing evidence)
- **Chi et al. 2023 (ACL)**: Identified self-attention variance as carrier of positional info
- **Zuo et al. 2025 (COLING)**: Proposed adjacency patterns (nearby embeddings more similar)
- **Kazemnejad et al. 2023 (NeurIPS)**: Length generalization properties of NoPE

**Key differentiation**: Prior work showed position *can* be encoded; this paper explains *how* with complete circuit + surgical validation.

## File Structure
- **Paper**: `/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/acl_latex.tex`
- **Bibliography**: `/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/custom.bib`
- **Plots directory**: `/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots/`
- **Analysis scripts**: `/home/nlp/matan_avitan/git/nopos_locating_new/analysis_scripts/`

## Analysis Scripts
- `attention_std_phases.py`: Visualizes 3 attention regimes (small/medium/large σ)
- `mlp_decoding_peaks.py`: Shows MLP position decoding with peaks
- `mlp_decoding_correct.py`: Correct formula for decoding vector
- `vocabulary_scaling_linear.py`: Linear relationship between vocab size and min samples

## Key Figures
- `attention_std_phases.png`: 3 attention patterns by initialization scale
- `variance_decay.png`: Post-attention variance decay
- `post_attn_ln_output_30K_samples_64_ctx.png`: LayerNorm paradox visualization
- `mlp_decoding_summary.png`: Decoding vector contributions
- `sample_convergence.png`: Sample size convergence study

## Writing Style Preferences (User: Matan)
- **DO**: Write fluent prose, elaborate on experiments with methodology and conclusions
- **DON'T**: Use excessive bullet points or `[noitemsep,topsep=0pt]`
- **Layer naming**: Use component names like "post-LN", "post-attention", "post-MLP" (NOT "layer 0")
- **Style reference**: Yoav Goldberg's concise but not terse style

## Citation Keys
- `haviv2022transformer` - Haviv et al. EMNLP 2022
- `chi-etal-2023-latent` - Chi et al. ACL 2023
- `zuo2025position` - Zuo et al. COLING 2025
- `kazemnejad2023lenghGeneralization` - Kazemnejad et al. NeurIPS 2023

## Recent Changes (Jan 2026)
1. Removed biology section (deemed irrelevant)
2. Added prior work citations and differentiation throughout paper
3. Expanded Related Work to 3 subsections (Implicit PE, Explicit PE, Mechanistic Interp.)
4. Updated abstract/intro/conclusion to acknowledge prior work while emphasizing novel contributions
5. Attention plots use normalized weights (after softmax), not unnormalized scores

---

## NoPE Training Experiment (Jan 2026)

### Purpose
Train a 1-layer wide transformer language model **without positional embeddings** to study whether the positional encoding mechanisms described in the paper emerge naturally during training on next-token prediction.

### Hypotheses to Test
1. **Uniform Attention Head**: Does at least one attention head maintain near-uniform attention (variance decays as 1/(i+1))?
2. **Decoding Vector**: Can we decode position using `w = W_V · Σ_j LN(E_j)` with high correlation?
3. **Population Statistics**: Does the network use population mean/std to infer position (LayerNorm paradox)?

### Architecture Choices
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_layer` | 1 | Single layer to study emergence clearly |
| `n_head` | 12 | Multiple heads to observe which become uniform |
| `n_embd` | 768 | Standard width for sufficient capacity |
| `block_size` | 256 | Moderate context for positional patterns |
| `vocab_size` | 50257 | GPT-2 BPE tokenizer |
| `dropout` | 0.0 | Clean analysis without stochasticity |
| `bias` | False | Modern practice |
| `use_positional_embedding` | False | **NoPE mode** |

### Two Experimental Variants
1. **LayerNorm variant** (`norm_type='layernorm'`): Standard centering + scaling
2. **RMSNorm variant** (`norm_type='rmsnorm'`): Only scaling, no centering

The RMSNorm comparison is critical because the LayerNorm paradox relies on population mean differences surviving despite per-sample zero-centering. RMSNorm doesn't center, so position information may flow differently.

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Dataset | Shakespeare BPE (~302K tokens) |
| Batch size | 32 × 4 = 128 effective |
| Learning rate | 1e-3 with cosine decay |
| Max iterations | 5000 |
| Warmup | 200 iterations |
| Checkpoint interval | 250 iterations |
| Seed | 42 (fixed for reproducibility) |

### Key Implementation Details
- **Xavier initialization** for Q, K, V matrices (critical for near-uniform attention at init)
- **Attention statistics logging** during training (entropy, uniformity per head)
- **Same seed** for both LN and RMSNorm experiments for fair comparison

### File Structure
```
nanoGPT/
├── model_nope.py                    # NoPE model with LN/RMSNorm option
├── train_nope.py                    # Training script with attention logging
├── config/
│   ├── train_nope_1layer_ln.py      # LayerNorm experiment config
│   └── train_nope_1layer_rms.py     # RMSNorm experiment config
├── out-nope-1layer-ln/              # LayerNorm checkpoints
└── out-nope-1layer-rms/             # RMSNorm checkpoints

analysis_scripts/
└── analyze_trained_nope.py          # Post-training hypothesis testing
```

### Running the Experiment
```bash
# Prepare data
cd nanoGPT/data/shakespeare && python prepare.py

# Train LayerNorm variant (GPU 0)
CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_1layer_ln.py

# Train RMSNorm variant (GPU 1)
CUDA_VISIBLE_DEVICES=1 python train_nope.py config/train_nope_1layer_rms.py

# Analyze results
python analysis_scripts/analyze_trained_nope.py --checkpoint nanoGPT/out-nope-1layer-ln/ckpt.pt
python analysis_scripts/analyze_trained_nope.py --checkpoint nanoGPT/out-nope-1layer-rms/ckpt.pt
```

### Expected Outcomes
| Hypothesis | LayerNorm Expected | RMSNorm Expected |
|------------|-------------------|------------------|
| H1: Uniform attention | 1-2 heads with r>0.9 | Similar or different pattern |
| H2: Decoding vector | r > 0.95 | May differ without centering |
| H3: Population stats | Strong position in mean | Different - no mean centering |
