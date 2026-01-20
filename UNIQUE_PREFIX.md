# Position Regression via Unique Prefix Diversity (NoPE Analysis)

This document details the experimental setup, findings, and hypotheses regarding how No-Positional Embedding (NoPE) causal transformers encode position through token diversity.

## 🧪 Experimental Setup

We analyze a specific training configuration designed to isolate the emergence of position information:
- **Model**: `nope-6layer-until-first-mlp`
- **Frozen Components**: Token embeddings (random init), Block 0 LayerNorm1, Attention, and LayerNorm2.
- **Trainable Components**: Block 0 MLP onwards.
- **Objective**: Regression task predicting the absolute position (0-127).
- **Core Question**: Can the MLP learn to "decode" position signal naturally arising from causal attention averaging of random embeddings?

## 📊 Analysis Methodology

We employ two main diagnostic scripts to track the emergence of position encoding across training:

### 1. Side-by-Side t-SNE Comparison (`tsne_unique_vs_random.py`)
Visualizes the geometry of activations at three key points: `post_ln2`, `mlp_hidden`, and `post_mlp`.
- **Unique Prefix Strategy**: Each sample $i$ has $i$ unique prefix tokens followed by repeated base tokens.
  - Sample 0: `[1000, 1000, 1000, ...]`
  - Sample $i$: `[1001, ..., 1000+i, 1000, 1000, ...]`
- **Random Sequence Strategy**: Completely random token sequences.
- **Dual-Encoding Visualization**:
  - **Color**: Position bucket (0-127 divided into 8 groups).
  - **Marker Shape**: Prefix diversity level (0-5, 6-11, 12-17, 18-23 unique tokens).
- **Optimization**: CPU-based t-SNE (`barnes_hut`, angle=0.2, 1000 iterations) for sharp, publication-quality clusters.

### 2. Mechanistic Metrics (`position_regression_metrics.py`)
Extracts three quantitative metrics across 20 checkpoints (every 1000 steps):
1. **Basis Validation**: Dot-product projection of activations onto the raw embeddings of unique prefix tokens.
2. **Pythagorean Numbers**: Tracking $L_2$ norm squared ($||v||^2$) before attention (value vectors) and after attention.
3. **Spectral Analysis**: PCA singular values and explained variance ratios at each layer to measure the dimensionality of the position signal.

## 💡 Findings & Hypotheses

### Finding 1: The "Unique Token Basis"
Causal attention creates a position signal by averaging varying numbers of unique embeddings. When the prefix is unique, it provides a stable "basis" in the activation space.
- **Hypothesis**: The MLP learns to project the high-dimensional residual signal onto these "basis directions" to recover the position index.

### Finding 2: LayerNorm's Linearization
While the position signal exists pre-LN (in the directional structure), it is highly non-linear.
- **Observation**: t-SNE clusters for position become significantly sharper and more linearly separable after `post_ln2`.
- **Hypothesis**: LayerNorm transforms the variance-based position signal into a mean-shifted signal that the MLP's first linear layer can easily process.

### Finding 3: The Pythagorean Transformation
- **Observation**: There is a consistent ratio change in $||v||^2$ before vs. after attention.
- **Hypothesis**: This "Pythagorean number" reflects the orthogonality of the random embeddings. As attention averages $i+1$ tokens, the norm scales according to the law of large numbers for random vectors, creating a monotonic "norm-clock" that the model utilizes.

## 🚀 Running the Analysis

```bash
# Run t-SNE visualization
python analysis_scripts/tsne_unique_vs_random.py --checkpoint-dir out-posreg-6layer-until-mlp --n-samples 24

# Run quantitative metrics
python analysis_scripts/position_regression_metrics.py --checkpoint-dir out-posreg-6layer-until-mlp
```

Results are logged to WandB projects:
- `nope-position-regression-tsne`
- `nope-position-regression-metrics`
