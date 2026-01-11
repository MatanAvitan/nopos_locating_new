"""
Attention Pattern Phases based on Embedding Initialization Scale

This script demonstrates how the attention pattern changes with the standard
deviation of embedding initialization:

1. Small σ (σ << 1/√d): Near-uniform attention weights
2. Medium σ (σ ≈ 1/√d, Xavier): Diagonal dominant + uniform lower triangle
3. Large σ (σ >> 1/√d): Random element dominates
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
N_CTX = 64
D_MODEL = 1024
N_HEADS = 1
D_HEAD = D_MODEL // N_HEADS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_attention_pattern(embeddings, W_Q, W_K, causal_mask=True):
    """Compute attention weights given embeddings and projection matrices."""
    # Project to queries and keys
    Q = embeddings @ W_Q  # [N_CTX, D_HEAD]
    K = embeddings @ W_K  # [N_CTX, D_HEAD]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(D_HEAD)  # [N_CTX, N_CTX]

    # Apply causal mask
    if causal_mask:
        mask = torch.triu(torch.ones(N_CTX, N_CTX, device=DEVICE), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    return attn_weights, scores


def run_attention_phase_experiment():
    """
    Generate attention patterns for three different initialization scales.
    """
    print("=" * 70)
    print("Attention Pattern Phases vs Initialization Scale")
    print("=" * 70)

    # Xavier scale: sqrt(2 / (fan_in + fan_out)) ≈ 1/sqrt(d) for square matrices
    xavier_scale = np.sqrt(2.0 / (D_MODEL + D_HEAD))

    # Three regimes
    scales = {
        'small': xavier_scale / 10,      # σ << 1/√d
        'medium': xavier_scale,           # σ ≈ 1/√d (Xavier)
        'large': xavier_scale * 10        # σ >> 1/√d
    }

    labels = {
        'small': f'Small σ = {scales["small"]:.4f}\n(Uniform attention)',
        'medium': f'Medium σ = {scales["medium"]:.4f}\n(Xavier, diagonal + uniform)',
        'large': f'Large σ = {scales["large"]:.4f}\n(Random dominant)'
    }

    # Store results
    attention_patterns = {}
    attention_scores = {}

    # Set seed for reproducibility
    torch.manual_seed(42)

    for scale_name, sigma in scales.items():
        print(f"\n{scale_name.upper()} scale: σ = {sigma:.6f}")

        # Initialize embeddings and projection matrices
        embeddings = torch.randn(N_CTX, D_MODEL, device=DEVICE) * sigma
        W_Q = torch.randn(D_MODEL, D_HEAD, device=DEVICE) * sigma
        W_K = torch.randn(D_MODEL, D_HEAD, device=DEVICE) * sigma

        # Compute attention
        attn_weights, scores = compute_attention_pattern(embeddings, W_Q, W_K)

        attention_patterns[scale_name] = attn_weights.cpu().numpy()
        attention_scores[scale_name] = scores.cpu().numpy()

        # Analyze pattern
        # Check uniformity: how close are off-diagonal elements?
        lower_tri_mask = np.tril(np.ones((N_CTX, N_CTX), dtype=bool), k=-1)
        off_diag = attn_weights.cpu().numpy()[lower_tri_mask]
        diag = np.diag(attn_weights.cpu().numpy())

        print(f"  Diagonal mean: {diag.mean():.4f}, std: {diag.std():.4f}")
        print(f"  Off-diagonal mean: {off_diag.mean():.4f}, std: {off_diag.std():.4f}")
        print(f"  Diagonal/Off-diag ratio: {diag.mean() / (off_diag.mean() + 1e-10):.2f}")

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Three attention patterns (normalized) side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (scale_name, label) in enumerate(zip(['small', 'medium', 'large'],
                                                   [labels['small'], labels['medium'], labels['large']])):
        ax = axes[idx]
        attn = attention_patterns[scale_name]

        im = ax.imshow(attn, cmap='Blues', aspect='auto', vmin=0, vmax=0.3)
        ax.set_xlabel('Key Position', fontsize=11)
        ax.set_ylabel('Query Position', fontsize=11)
        ax.set_title(label, fontsize=12)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Attention Patterns vs Initialization Scale σ', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'attention_std_phases.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: attention_std_phases.png")

    # Plot 2: Diagonal attention weights across positions for each regime
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(N_CTX)
    colors = {'small': 'blue', 'medium': 'green', 'large': 'red'}

    for scale_name in ['small', 'medium', 'large']:
        diag = np.diag(attention_patterns[scale_name])
        ax.plot(positions, diag, '-', linewidth=2, color=colors[scale_name],
                label=f'{scale_name.capitalize()} σ', alpha=0.8)

    # Theoretical uniform: 1/i for position i
    theoretical_uniform = 1.0 / (positions + 1)
    ax.plot(positions, theoretical_uniform, 'k--', linewidth=2,
            label='Theoretical 1/(i+1)', alpha=0.7)

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Diagonal Attention Weight $A_{ii}$', fontsize=12)
    ax.set_title('Diagonal Attention Weights vs Initialization Scale', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, N_CTX-1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'attention_diagonal_by_std.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: attention_diagonal_by_std.png")

    # Plot 3: Row-wise attention distribution for position 32 (middle)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    query_pos = 32

    for idx, scale_name in enumerate(['small', 'medium', 'large']):
        ax = axes[idx]
        row = attention_patterns[scale_name][query_pos, :query_pos+1]

        ax.bar(range(len(row)), row, color=colors[scale_name], alpha=0.7)
        ax.axhline(y=1.0/(query_pos+1), color='black', linestyle='--',
                   label=f'Uniform: 1/{query_pos+1}')
        ax.set_xlabel('Key Position', fontsize=11)
        ax.set_ylabel('Attention Weight', fontsize=11)
        ax.set_title(f'{scale_name.capitalize()} σ (Query pos={query_pos})', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.5, query_pos + 0.5)

    plt.suptitle(f'Attention Distribution at Query Position {query_pos}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'attention_row_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: attention_row_distribution.png")

    return attention_patterns, attention_scores


def run_multiple_samples():
    """Average attention patterns across multiple random initializations."""
    print("\n" + "=" * 70)
    print("Averaging over Multiple Samples")
    print("=" * 70)

    n_samples = 100
    xavier_scale = np.sqrt(2.0 / (D_MODEL + D_HEAD))

    scales = {
        'small': xavier_scale / 10,
        'medium': xavier_scale,
        'large': xavier_scale * 10
    }

    avg_patterns = {k: np.zeros((N_CTX, N_CTX)) for k in scales}

    for sample_idx in range(n_samples):
        torch.manual_seed(sample_idx)

        for scale_name, sigma in scales.items():
            embeddings = torch.randn(N_CTX, D_MODEL, device=DEVICE) * sigma
            W_Q = torch.randn(D_MODEL, D_HEAD, device=DEVICE) * sigma
            W_K = torch.randn(D_MODEL, D_HEAD, device=DEVICE) * sigma

            attn_weights, _ = compute_attention_pattern(embeddings, W_Q, W_K)
            avg_patterns[scale_name] += attn_weights.cpu().numpy() / n_samples

    # Plot averaged patterns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = ['Small σ (Uniform)', 'Medium σ (Xavier)', 'Large σ (Random)']

    for idx, (scale_name, label) in enumerate(zip(['small', 'medium', 'large'], labels)):
        ax = axes[idx]
        attn = avg_patterns[scale_name]

        im = ax.imshow(attn, cmap='Blues', aspect='auto', vmin=0, vmax=0.15)
        ax.set_xlabel('Key Position', fontsize=11)
        ax.set_ylabel('Query Position', fontsize=11)
        ax.set_title(f'{label}\n(Averaged over {n_samples} samples)', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Average Attention Patterns by Initialization Scale', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'attention_std_phases_averaged.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: attention_std_phases_averaged.png")

    return avg_patterns


if __name__ == "__main__":
    # Run single sample experiment
    patterns, scores = run_attention_phase_experiment()

    # Run averaged experiment
    avg_patterns = run_multiple_samples()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nGenerated plots in: {PLOTS_DIR}")
    print("  - attention_std_phases.png (3 regimes side by side)")
    print("  - attention_diagonal_by_std.png (diagonal weights comparison)")
    print("  - attention_row_distribution.png (attention at position 32)")
    print("  - attention_std_phases_averaged.png (averaged over 100 samples)")
