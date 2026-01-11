"""
MLP Position Decoding - CORRECT Formula

Decoding Vector (User's Formula):
=================================
w = W_V * Σ_{i=1}^{N} (e_i - E(e_i)) / std(e_i)

where:
- e_i is the embedding vector for token i
- E(e_i) is the mean across the neuron dimension (per token)
- std(e_i) is the std across the neuron dimension (per token)

This is a SINGLE decoding vector, computed as:
1. Normalize each embedding: x̂_i = (e_i - mean(e_i)) / std(e_i)
2. Sum all normalized embeddings: s = Σ_i x̂_i
3. Apply W_V to get decoding vector: w = W_V @ s

Then compute w · z_i for each position i to decode.
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


def normalize_embedding(e, eps=1e-5):
    """
    Normalize embedding across neuron dimension.
    e: [D_MODEL] or [N, D_MODEL]
    Returns: (e - mean(e)) / std(e) computed per sample across neurons
    """
    if e.dim() == 1:
        mean = e.mean()
        std = e.std(unbiased=False) + eps
        return (e - mean) / std
    else:
        # e is [N, D_MODEL], normalize each row
        mean = e.mean(dim=-1, keepdim=True)
        std = e.std(dim=-1, keepdim=True, unbiased=False) + eps
        return (e - mean) / std


def run_decoding_experiment():
    """
    Demonstrate position decoding with the CORRECT decoding vector formula.

    Decoding vector: w = W_V @ Σ_i (e_i - mean(e_i)) / std(e_i)
    """
    print("=" * 70)
    print("MLP Position Decoding - CORRECT Formula")
    print("=" * 70)
    print("\nDecoding vector: w = W_V @ Σ_i [(e_i - E(e_i)) / std(e_i)]")

    # Initialize
    torch.manual_seed(42)
    W_E = xavier_init((D_VOCAB, D_MODEL))  # Embedding matrix
    W_V = xavier_init((D_MODEL, D_MODEL))  # Value projection

    # Generate random sequence
    tokens = torch.randint(0, D_VOCAB, (N_CTX,), device=DEVICE)
    print(f"\nSequence length: {N_CTX}")
    print(f"Unique tokens: {len(tokens.unique())}")

    # Get embeddings for each token
    e = W_E[tokens]  # [N_CTX, D_MODEL]

    # Step 1: Normalize each embedding across neuron dimension
    e_normalized = normalize_embedding(e)  # [N_CTX, D_MODEL]

    print(f"\nNormalized embeddings stats:")
    print(f"  Mean of means: {e_normalized.mean(dim=-1).mean().item():.6f}")
    print(f"  Mean of stds: {e_normalized.std(dim=-1).mean().item():.4f}")

    # Step 2: Sum all normalized embeddings
    sum_normalized = e_normalized.sum(dim=0)  # [D_MODEL]

    # Step 3: Apply W_V to get the SINGLE decoding vector
    w = sum_normalized @ W_V.T  # [D_MODEL] - THIS IS THE DECODING VECTOR

    print(f"\nDecoding vector:")
    print(f"  ||w|| = {w.norm().item():.4f}")

    # Compute value vectors for attention output
    # v_j = W_V @ (e_j - mean) / std
    v = e_normalized @ W_V.T  # [N_CTX, D_MODEL]

    # Compute attention outputs: z_i = (1/i) * Σ_{j=1}^{i} v_j
    cumsum_v = torch.cumsum(v, dim=0)  # [N_CTX, D_MODEL]
    positions = torch.arange(1, N_CTX + 1, device=DEVICE).float().unsqueeze(1)
    z = cumsum_v / positions  # [N_CTX, D_MODEL]

    # Decode: Compute w · z_i for each position
    print("\n" + "=" * 70)
    print("Position Decoding with Single Decoding Vector")
    print("=" * 70)

    # w · z_i for all positions
    dot_products = (z * w).sum(dim=1)  # [N_CTX]

    print(f"\nDot products w · z_i:")
    for pos in [1, 5, 10, 20, 32, 50, 64]:
        print(f"  Position {pos}: w · z_{pos} = {dot_products[pos-1].item():.4f}")

    # Check if dot product correlates with position
    true_positions = np.arange(1, N_CTX + 1)
    from scipy.stats import pearsonr, spearmanr

    dp_numpy = dot_products.cpu().numpy()
    pearson_r, p_val = pearsonr(true_positions, dp_numpy)
    spearman_r, _ = spearmanr(true_positions, dp_numpy)

    print(f"\nCorrelation of w · z_i with position:")
    print(f"  Pearson r: {pearson_r:.4f} (p = {p_val:.2e})")
    print(f"  Spearman r: {spearman_r:.4f}")

    # Also test individual v_j · z_i approach (for comparison)
    print("\n" + "=" * 70)
    print("Alternative: Individual Token Detection")
    print("=" * 70)

    decoded_by_counting = []
    contributions_dict = {}

    for pos_idx in range(N_CTX):
        z_i = z[pos_idx]  # [D_MODEL]

        # Compute v_j · z_i for each token j that contributed
        v_contributing = v[:pos_idx + 1]  # [pos_idx + 1, D_MODEL]
        individual_dots = (v_contributing * z_i).sum(dim=1)  # [pos_idx + 1]

        # Count positive contributions
        positive_count = (individual_dots > 0).sum().item()
        decoded_by_counting.append(positive_count)

        if (pos_idx + 1) in [5, 10, 20, 32, 50, 64]:
            contributions_dict[pos_idx + 1] = individual_dots.cpu().numpy()

    count_corr, _ = pearsonr(true_positions, decoded_by_counting)
    print(f"\nCounting positive v_j · z_i:")
    print(f"  Pearson r: {count_corr:.4f}")
    for pos in [5, 10, 20, 32, 50, 64]:
        print(f"  Position {pos}: decoded = {decoded_by_counting[pos-1]}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    # Plot 1: Single decoding vector dot products vs position
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(true_positions, dp_numpy, s=60, c='blue', alpha=0.8)
    ax1.set_xlabel('True Position $i$', fontsize=12)
    ax1.set_ylabel('$w \\cdot z_i$', fontsize=12)
    ax1.set_title(f'Single Decoding Vector: $w \\cdot z_i$ vs Position\n' +
                  f'(Pearson r = {pearson_r:.4f})', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Add trend line
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(true_positions, dp_numpy)
    ax1.plot(true_positions, slope * true_positions + intercept, 'r--',
             linewidth=2, label=f'Linear fit')
    ax1.legend()

    ax2 = axes[1]
    ax2.scatter(true_positions, decoded_by_counting, s=60, c='green', alpha=0.8)
    ax2.plot([0, N_CTX+1], [0, N_CTX+1], 'r--', linewidth=2, label='y = x')
    ax2.set_xlabel('True Position $i$', fontsize=12)
    ax2.set_ylabel('Decoded Position (count)', fontsize=12)
    ax2.set_title(f'Counting Method: $|\\{{j : v_j \\cdot z_i > 0\\}}|$\n' +
                  f'(Pearson r = {count_corr:.4f})', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, N_CTX+2)
    ax2.set_ylim(0, N_CTX+2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_comparison.png")

    # Plot 2: Individual contributions showing peaks
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    positions_to_plot = [5, 10, 20, 32, 50, 64]

    for idx, pos in enumerate(positions_to_plot):
        ax = axes[idx]
        contributions = contributions_dict[pos]

        x_vals = np.arange(len(contributions))
        colors = ['green' if c > 0 else 'red' for c in contributions]
        ax.bar(x_vals, contributions, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)

        positive_count = (contributions > 0).sum()
        ax.set_xlabel('Token Index j', fontsize=10)
        ax.set_ylabel('$v_j \\cdot z_i$', fontsize=10)
        ax.set_title(f'Position i={pos}\n(+count: {positive_count}, true: {pos})', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Individual Dot Product Contributions $v_j \\cdot z_i$', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_contributions.png")

    # Plot 3: Summary figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    ax.scatter(true_positions, decoded_by_counting, s=80, c='blue', alpha=0.8,
               label='Decoded position', zorder=3)
    ax.plot([0, N_CTX + 1], [0, N_CTX + 1], 'r--', linewidth=2.5,
            label='Perfect (y = x)', zorder=2)

    ax.set_xlabel('True Position $i$', fontsize=14)
    ax.set_ylabel('Decoded Position', fontsize=14)
    ax.set_title('MLP Position Decoding\n' +
                 r'$w = W_V \cdot \sum_j \frac{e_j - \mu_j}{\sigma_j}$, ' +
                 r'decode: $|\{j : v_j \cdot z_i > 0\}|$',
                 fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, N_CTX + 2)
    ax.set_ylim(0, N_CTX + 2)

    mae = np.abs(np.array(decoded_by_counting) - true_positions).mean()
    textstr = f'Pearson r = {count_corr:.4f}\nMAE = {mae:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_summary.png")

    return {
        'single_vector_correlation': pearson_r,
        'counting_correlation': count_corr,
        'dot_products': dp_numpy,
        'decoded_by_counting': decoded_by_counting
    }


if __name__ == "__main__":
    results = run_decoding_experiment()

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nSingle decoding vector (w · z_i) correlation: {results['single_vector_correlation']:.4f}")
    print(f"Counting method correlation: {results['counting_correlation']:.4f}")
    print(f"\nPlots saved to: {PLOTS_DIR}")
