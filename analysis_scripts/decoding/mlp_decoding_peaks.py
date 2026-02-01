"""
MLP Position Decoding - Correct Visualization with Peaks

The decoding vector w = W_V · Σ_j (e_j - μ_j) / σ_j

For each position i, we compute w · v_j for all j in the sequence.
This should show that each token j contributes positively, and the
cumulative sum up to position i gives approximately i.

The key visualization: Show w · v_j for all j, demonstrating that
at position 12, the sum of the first 12 contributions ≈ 12.
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
    """Normalize embedding across neuron dimension."""
    if e.dim() == 1:
        mean = e.mean()
        std = e.std(unbiased=False) + eps
        return (e - mean) / std
    else:
        mean = e.mean(dim=-1, keepdim=True)
        std = e.std(dim=-1, keepdim=True, unbiased=False) + eps
        return (e - mean) / std


def run_decoding_experiment():
    """
    Demonstrate position decoding showing peaks at each position.
    """
    print("=" * 70)
    print("MLP Position Decoding - Peak Visualization")
    print("=" * 70)

    # Initialize
    torch.manual_seed(42)
    W_E = xavier_init((D_VOCAB, D_MODEL))
    W_V = xavier_init((D_MODEL, D_MODEL))

    # Generate random sequence
    tokens = torch.randint(0, D_VOCAB, (N_CTX,), device=DEVICE)
    print(f"\nSequence length: {N_CTX}")

    # Get embeddings
    e = W_E[tokens]  # [N_CTX, D_MODEL]

    # Normalize each embedding
    e_normalized = normalize_embedding(e)  # [N_CTX, D_MODEL]

    # Compute value vectors: v_j = W_V @ LN(e_j)
    v = e_normalized @ W_V.T  # [N_CTX, D_MODEL]

    # Decoding vector: w = W_V @ Σ_j LN(e_j)
    sum_normalized = e_normalized.sum(dim=0)  # [D_MODEL]
    w = sum_normalized @ W_V.T  # [D_MODEL]

    print(f"\nDecoding vector ||w|| = {w.norm().item():.4f}")

    # Compute w · v_j for each position j
    # This shows the contribution of each token
    contributions = (w * v).sum(dim=1)  # [N_CTX]
    contributions_np = contributions.cpu().numpy()

    print(f"\nContributions w · v_j:")
    print(f"  Mean: {contributions_np.mean():.4f}")
    print(f"  Std: {contributions_np.std():.4f}")
    print(f"  Min: {contributions_np.min():.4f}")
    print(f"  Max: {contributions_np.max():.4f}")

    # Cumulative sum gives decoded position
    cumsum_contributions = np.cumsum(contributions_np)

    print(f"\nCumulative sum (decoded position):")
    for pos in [1, 5, 10, 12, 20, 32, 50, 64]:
        print(f"  Position {pos}: cumsum = {cumsum_contributions[pos-1]:.2f}")

    # Normalize to get position estimate
    # If each contribution is approximately constant c, then cumsum[i] ≈ i*c
    # So decoded_position = cumsum / c ≈ i
    avg_contribution = contributions_np.mean()
    decoded_positions = cumsum_contributions / avg_contribution

    print(f"\nNormalized decoded positions (cumsum / mean_contribution):")
    for pos in [1, 5, 10, 12, 20, 32, 50, 64]:
        print(f"  Position {pos}: decoded = {decoded_positions[pos-1]:.2f}")

    # Correlation
    from scipy.stats import pearsonr
    true_positions = np.arange(1, N_CTX + 1)
    corr, _ = pearsonr(true_positions, decoded_positions)
    print(f"\nPearson correlation: {corr:.4f}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    # Plot 1: Individual contributions w · v_j showing each token's contribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: All contributions as bar plot
    ax1 = axes[0, 0]
    colors = ['green' if c > 0 else 'red' for c in contributions_np]
    ax1.bar(range(N_CTX), contributions_np, color=colors, alpha=0.7, width=1.0)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=avg_contribution, color='blue', linestyle='--',
                label=f'Mean = {avg_contribution:.2f}')
    ax1.set_xlabel('Position j', fontsize=12)
    ax1.set_ylabel('$w \\cdot v_j$', fontsize=12)
    ax1.set_title('Individual Token Contributions', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Top right: Cumulative sum showing position decoding
    ax2 = axes[0, 1]
    ax2.plot(true_positions, cumsum_contributions, 'b-', linewidth=2, label='Cumulative sum')
    ax2.set_xlabel('Position i', fontsize=12)
    ax2.set_ylabel('$\\sum_{j=1}^{i} w \\cdot v_j$', fontsize=12)
    ax2.set_title('Cumulative Sum = Decoded Signal', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Bottom left: Normalized decoded position vs true position
    ax3 = axes[1, 0]
    ax3.scatter(true_positions, decoded_positions, s=50, c='blue', alpha=0.7)
    ax3.plot([0, N_CTX+1], [0, N_CTX+1], 'r--', linewidth=2, label='y = x')
    ax3.set_xlabel('True Position i', fontsize=12)
    ax3.set_ylabel('Decoded Position', fontsize=12)
    ax3.set_title(f'Decoded vs True Position (r = {corr:.4f})', fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, N_CTX+2)
    ax3.set_ylim(0, N_CTX+2)

    # Bottom right: Heatmap showing contribution at each position
    ax4 = axes[1, 1]
    # Create matrix: row i shows contributions 1..i, rest is 0
    contrib_matrix = np.zeros((N_CTX, N_CTX))
    for i in range(N_CTX):
        contrib_matrix[i, :i+1] = contributions_np[:i+1]

    im = ax4.imshow(contrib_matrix, aspect='auto', cmap='RdYlGn',
                     vmin=-abs(contributions_np).max(), vmax=abs(contributions_np).max())
    ax4.set_xlabel('Token j', fontsize=12)
    ax4.set_ylabel('Position i', fontsize=12)
    ax4.set_title('Contributions at Each Position\n(Row i shows tokens 1..i)', fontsize=13)
    plt.colorbar(im, ax=ax4, label='$w \\cdot v_j$')

    plt.suptitle('MLP Position Decoding: $w = W_V \\cdot \\sum_j \\frac{e_j - \\mu_j}{\\sigma_j}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_peaks.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_peaks.png")

    # Plot 2: Simple summary showing the key insight
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show contributions and cumulative sum on same plot
    ax.bar(range(N_CTX), contributions_np, color='lightblue', alpha=0.7,
           width=1.0, label='Individual: $w \\cdot v_j$')

    ax2 = ax.twinx()
    ax2.plot(range(N_CTX), cumsum_contributions, 'r-', linewidth=2.5,
             label='Cumulative: $\\sum_{j=1}^{i} w \\cdot v_j$')
    ax2.plot(range(N_CTX), true_positions * avg_contribution, 'g--', linewidth=2,
             alpha=0.7, label='Expected: $i \\times \\bar{c}$')

    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Individual Contribution $w \\cdot v_j$', fontsize=12, color='blue')
    ax2.set_ylabel('Cumulative Sum', fontsize=12, color='red')
    ax.set_title('Position Decoding: Cumulative Sum of Token Contributions', fontsize=14)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_summary.png")

    # Plot 3: Show specific positions with peaks
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    positions_to_show = [5, 12, 20, 32, 50, 64]

    for idx, pos in enumerate(positions_to_show):
        ax = axes[idx]

        # Show contributions up to position pos
        contrib_at_pos = np.zeros(N_CTX)
        contrib_at_pos[:pos] = contributions_np[:pos]

        colors = ['green' if j < pos else 'lightgray' for j in range(N_CTX)]
        ax.bar(range(N_CTX), contributions_np, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Mark the position
        ax.axvline(x=pos-0.5, color='red', linestyle='--', linewidth=2,
                   label=f'Position {pos}')

        cumsum_at_pos = cumsum_contributions[pos-1]
        decoded = cumsum_at_pos / avg_contribution

        ax.set_xlabel('Token j', fontsize=10)
        ax.set_ylabel('$w \\cdot v_j$', fontsize=10)
        ax.set_title(f'Position i={pos}\nSum={cumsum_at_pos:.1f}, Decoded={decoded:.1f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(-1, N_CTX)

    plt.suptitle('Token Contributions at Different Positions\n(Green = contributing tokens)', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: mlp_decoding_contributions.png")

    return {
        'contributions': contributions_np,
        'cumsum': cumsum_contributions,
        'decoded_positions': decoded_positions,
        'correlation': corr
    }


if __name__ == "__main__":
    results = run_decoding_experiment()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPearson correlation: {results['correlation']:.4f}")
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("  - mlp_decoding_peaks.png (4-panel analysis)")
    print("  - mlp_decoding_summary.png (contributions + cumsum)")
    print("  - mlp_decoding_contributions.png (position-specific views)")
