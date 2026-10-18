"""
MLP Position Decoding with LayerNorm Normalization

Mathematical Framework (Corrected):
====================================

Since the input to attention is normalized with LayerNorm, the decoding
vector must also account for this normalization.

1. Input normalization:
   x̂_j = (E_j - μ_{E_j}) / σ_{E_j}
   where E_j is the embedding for token j, normalized across dimensions.

2. Value vectors after normalization:
   v_j = W_V · x̂_j = W_V · (E_j - μ_{E_j}) / σ_{E_j}

3. Attention output at position i with uniform weights:
   z_i = (1/i) · Σ_{j=1}^{i} v_j

4. Decoding vector (sum of ALL normalized value vectors):
   w = Σ_{j=1}^{N} W_V · (E_j - μ_{E_j}) / σ_{E_j} = Σ_{j=1}^{N} v_j

5. Position decoding: Count positive dot products v_k · z_i
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
N_CTX = 64
D_MODEL = 1024
D_VOCAB = 5000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def xavier_init(shape, gain=1.0):
    """Xavier initialization."""
    fan_in, fan_out = shape[0], shape[1]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return torch.randn(*shape, device=DEVICE) * std


def apply_layernorm(x, eps=1e-5):
    """Apply LayerNorm normalization across the last dimension."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    return (x - mean) / std


def run_decoding_experiment():
    """
    Demonstrate position decoding with LayerNorm-normalized inputs.
    """
    print("=" * 70)
    print("MLP Position Decoding with LayerNorm Normalization")
    print("=" * 70)

    # Initialize
    torch.manual_seed(42)
    W_E = xavier_init((D_VOCAB, D_MODEL))  # Embedding matrix
    W_V = xavier_init((D_MODEL, D_MODEL))  # Value projection

    # Generate random sequence
    tokens = torch.randint(0, D_VOCAB, (N_CTX,), device=DEVICE)
    print(f"\nSequence length: {N_CTX}")
    print(f"Unique tokens: {len(tokens.unique())}")

    # Get embeddings
    E = W_E[tokens]  # [N_CTX, D_MODEL]

    # Apply LayerNorm to embeddings (as done before attention)
    E_normalized = apply_layernorm(E)  # [N_CTX, D_MODEL]

    # Compute value vectors: v_j = W_V @ LN(E_j)
    v = E_normalized @ W_V.T  # [N_CTX, D_MODEL]

    print(f"\nValue vector stats (after LN):")
    v_norms = v.norm(dim=1)
    print(f"  Mean ||v_j||: {v_norms.mean().item():.4f}")
    print(f"  Std ||v_j||: {v_norms.std().item():.4f}")

    # Compute attention outputs: z_i = (1/i) * Σ_{j=1}^{i} v_j
    cumsum_v = torch.cumsum(v, dim=0)  # [N_CTX, D_MODEL]
    positions = torch.arange(1, N_CTX + 1, device=DEVICE).float().unsqueeze(1)
    z = cumsum_v / positions  # [N_CTX, D_MODEL]

    # Decoding: For each position, compute dot products with each contributing token
    print("\n" + "=" * 70)
    print("Position Decoding Results")
    print("=" * 70)

    decoded_positions = []
    all_contributions = {}

    for pos_idx in range(N_CTX):
        z_i = z[pos_idx]  # [D_MODEL]

        # Compute v_j · z_i for each token j that contributed (j ≤ pos_idx)
        v_contributing = v[:pos_idx + 1]  # [pos_idx + 1, D_MODEL]
        dot_products = (v_contributing * z_i).sum(dim=1)  # [pos_idx + 1]

        # Count positive contributions
        positive_count = (dot_products > 0).sum().item()
        decoded_positions.append(positive_count)

        # Store for plotting
        if (pos_idx + 1) in [5, 10, 20, 32, 50, 64]:
            all_contributions[pos_idx + 1] = dot_products.cpu().numpy()

    # Compute correlation
    true_positions = np.arange(1, N_CTX + 1)
    from scipy.stats import pearsonr
    corr, pval = pearsonr(true_positions, decoded_positions)

    print(f"\nDecoding Results:")
    print(f"  Pearson correlation: {corr:.4f}")

    # Show sample positions
    for pos in [5, 10, 20, 32, 50, 64]:
        print(f"  Position {pos}: decoded = {decoded_positions[pos-1]}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    # Plot 1: Individual contributions showing peak at decoded token
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    positions_to_plot = [5, 10, 20, 32, 50, 64]

    for idx, pos in enumerate(positions_to_plot):
        ax = axes[idx]
        contributions = all_contributions[pos]

        # Bar plot with color based on sign
        x_vals = np.arange(len(contributions))
        colors = ['green' if c > 0 else 'red' for c in contributions]
        bars = ax.bar(x_vals, contributions, color=colors, alpha=0.7, width=1.0)

        # Highlight the peak
        peak_idx = np.argmax(contributions)
        ax.axhline(y=0, color='black', linewidth=0.5)

        positive_count = (contributions > 0).sum()
        ax.set_xlabel('Token Index j', fontsize=10)
        ax.set_ylabel('$v_j \\cdot z_i$', fontsize=10)
        ax.set_title(f'Position i={pos}\n(+count: {positive_count}, true: {pos})', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Dot Product Contributions $v_j \\cdot z_i$ with LayerNorm\n' +
                 '(Green = positive contribution from token at position j)', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_contributions.png")

    # Plot 2: Decoded vs true position
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(true_positions, decoded_positions, s=80, c='blue', alpha=0.8,
               label='Decoded (count of +ve)', zorder=3)
    ax.plot([0, N_CTX + 1], [0, N_CTX + 1], 'r--', linewidth=2.5,
            label='Perfect decoding (y = x)', zorder=2)

    ax.set_xlabel('True Position $i$', fontsize=14)
    ax.set_ylabel('Decoded Position', fontsize=14)
    ax.set_title('MLP Position Decoding with LayerNorm Normalization\n' +
                 r'$w = \sum_j W_V \cdot \frac{E_j - \mu_j}{\sigma_j}$, decode by counting $|\{j : v_j \cdot z_i > 0\}|$',
                 fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, N_CTX + 2)
    ax.set_ylim(0, N_CTX + 2)

    # Stats text box
    mae = np.abs(np.array(decoded_positions) - true_positions).mean()
    textstr = f'Pearson r = {corr:.4f}\nMAE = {mae:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_summary.png")

    # Plot 3: Detailed view of contributions for a single position
    fig, ax = plt.subplots(figsize=(12, 6))

    pos = 32
    contributions = all_contributions[pos]
    x_vals = np.arange(len(contributions))

    # Color by magnitude
    colors = plt.cm.RdYlGn((contributions - contributions.min()) /
                            (contributions.max() - contributions.min() + 1e-10))

    bars = ax.bar(x_vals, contributions, color=colors, alpha=0.8, width=1.0,
                  edgecolor='black', linewidth=0.3)
    ax.axhline(y=0, color='black', linewidth=1)

    # Mark threshold
    threshold = 0
    ax.axhline(y=threshold, color='blue', linestyle='--', linewidth=1.5,
               label='Threshold (0)')

    ax.set_xlabel('Token Index j', fontsize=12)
    ax.set_ylabel('Dot Product $v_j \\cdot z_{32}$', fontsize=12)
    ax.set_title(f'Detailed Contributions at Position 32\n' +
                 f'Each bar = contribution from token j to attention output at position 32',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_detailed.png")

    return {
        'decoded_positions': decoded_positions,
        'correlation': corr,
        'contributions': all_contributions
    }


def run_multiple_sequences():
    """Run on multiple sequences to verify robustness."""
    print("\n" + "=" * 70)
    print("Multi-Sequence Validation")
    print("=" * 70)

    n_sequences = 100
    all_correlations = []
    all_mae = []

    W_E = xavier_init((D_VOCAB, D_MODEL))
    W_V = xavier_init((D_MODEL, D_MODEL))

    for seq_idx in range(n_sequences):
        torch.manual_seed(seq_idx)
        tokens = torch.randint(0, D_VOCAB, (N_CTX,), device=DEVICE)

        E = W_E[tokens]
        E_normalized = apply_layernorm(E)
        v = E_normalized @ W_V.T

        cumsum_v = torch.cumsum(v, dim=0)
        positions = torch.arange(1, N_CTX + 1, device=DEVICE).float().unsqueeze(1)
        z = cumsum_v / positions

        decoded = []
        for pos_idx in range(N_CTX):
            z_i = z[pos_idx]
            v_contributing = v[:pos_idx + 1]
            dot_products = (v_contributing * z_i).sum(dim=1)
            positive_count = (dot_products > 0).sum().item()
            decoded.append(positive_count)

        true_positions = np.arange(1, N_CTX + 1)
        from scipy.stats import pearsonr
        corr, _ = pearsonr(true_positions, decoded)
        mae = np.abs(np.array(decoded) - true_positions).mean()

        all_correlations.append(corr)
        all_mae.append(mae)

    print(f"\nResults over {n_sequences} sequences:")
    print(f"  Mean correlation: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}")
    print(f"  Mean MAE: {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f}")

    return all_correlations, all_mae


if __name__ == "__main__":
    results = run_decoding_experiment()
    correlations, maes = run_multiple_sequences()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nSingle sequence correlation: {results['correlation']:.4f}")
    print(f"Multi-sequence mean correlation: {np.mean(correlations):.4f}")
    print(f"\nPlots saved to: {PLOTS_DIR}")
