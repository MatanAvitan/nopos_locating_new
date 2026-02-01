"""
Evaluate position prediction with increasing prefix diversity.

Setup:
- All tokens the same (0 unique prefix tokens)
- First 1 token different, rest same
- First 2 tokens different, rest same
- ... and so on

This tests how prefix diversity affects position encoding.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import wandb
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_position_classifier import GPTPositionClassifier, GPTPositionClassifierConfig


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model args from checkpoint
    model_args = checkpoint['model_args']
    config = GPTPositionClassifierConfig(**model_args)

    model = GPTPositionClassifier(config)

    # Load state dict (handle _orig_mod prefix from torch.compile)
    state_dict = checkpoint['model']
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            unwrapped_state_dict[k[len('_orig_mod.'):]] = v
        else:
            unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model = model.to(device)
    model.eval()

    return model, config, checkpoint


def create_prefix_diversity_sequences(n_prefix_unique, block_size, base_token=1000, n_sequences=100):
    """
    Create sequences where the first n_prefix_unique tokens are unique,
    and the rest are all the same base_token.

    Args:
        n_prefix_unique: Number of unique tokens at the beginning (0 means all same)
        block_size: Total sequence length
        base_token: The token to fill the rest of the sequence
        n_sequences: Number of sequences to generate

    Returns:
        Tensor of shape (n_sequences, block_size)
    """
    sequences = torch.full((n_sequences, block_size), base_token, dtype=torch.long)

    if n_prefix_unique > 0:
        # Use different tokens for the prefix (avoiding base_token)
        # Use tokens starting from base_token + 1
        for i in range(min(n_prefix_unique, block_size)):
            sequences[:, i] = base_token + 1 + i

    return sequences


def evaluate_prefix_diversity(model, config, device='cuda', n_sequences=100):
    """
    Evaluate model on sequences with varying prefix diversity.

    Returns dict mapping n_prefix_unique -> results dict
    """
    block_size = config.block_size
    results = {}

    # Test prefix lengths: 0, 1, 2, 4, 8, 16, 32, 64, 128 (up to block_size)
    prefix_lengths = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    prefix_lengths = [p for p in prefix_lengths if p <= block_size]

    # Also add block_size if not already included
    if block_size not in prefix_lengths:
        prefix_lengths.append(block_size)

    print(f"\nEvaluating prefix diversity (block_size={block_size})...")
    print(f"Testing prefix lengths: {prefix_lengths}")

    with torch.no_grad():
        for n_prefix in prefix_lengths:
            # Create sequences
            sequences = create_prefix_diversity_sequences(
                n_prefix_unique=n_prefix,
                block_size=block_size,
                base_token=1000,
                n_sequences=n_sequences
            ).to(device)

            # Get predictions
            preds, _, _ = model(sequences)
            pred_positions = preds.squeeze(-1).cpu().numpy()  # (n_sequences, block_size)

            # True positions
            true_positions = np.arange(block_size)

            # Flatten for correlation
            pred_flat = pred_positions.flatten()
            true_flat = np.tile(true_positions, n_sequences)

            # Calculate metrics
            pearson_r, pearson_p = stats.pearsonr(pred_flat, true_flat)
            mse = np.mean((pred_flat - true_flat) ** 2)
            mae = np.mean(np.abs(pred_flat - true_flat))

            # Per-position mean prediction
            per_position_mean = pred_positions.mean(axis=0)
            per_position_std = pred_positions.std(axis=0)

            results[n_prefix] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'mse': mse,
                'mae': mae,
                'pred_positions': pred_positions,  # Keep all predictions for plotting
                'per_position_mean': per_position_mean,
                'per_position_std': per_position_std
            }

            print(f"  Prefix {n_prefix:3d} unique tokens: Pearson r = {pearson_r:.4f}, MSE = {mse:.2f}, MAE = {mae:.2f}")

    return results, prefix_lengths


def plot_regression_results(results, prefix_lengths, block_size):
    """Create regression scatter plots for each prefix length."""

    # Determine grid size
    n_plots = len(prefix_lengths)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    true_positions = np.arange(block_size)

    for idx, n_prefix in enumerate(prefix_lengths):
        ax = axes[idx]
        res = results[n_prefix]

        # Get mean and std per position
        mean_pred = res['per_position_mean']
        std_pred = res['per_position_std']

        # Plot mean prediction with error bars (std)
        ax.fill_between(true_positions, mean_pred - std_pred, mean_pred + std_pred,
                       alpha=0.3, color='steelblue', label='±1 std')
        ax.plot(true_positions, mean_pred, 'o-', color='steelblue', markersize=3,
               linewidth=1.5, label='Mean pred')

        # Plot ideal diagonal
        ax.plot([0, block_size-1], [0, block_size-1], 'r--', linewidth=2, alpha=0.7, label='Ideal')

        ax.set_xlabel('True Position', fontsize=10)
        ax.set_ylabel('Predicted Position', fontsize=10)
        ax.set_title(f'{n_prefix} unique prefix tokens\nr={res["pearson_r"]:.3f}, MAE={res["mae"]:.1f}', fontsize=11)
        ax.set_xlim(-2, block_size + 2)
        ax.set_ylim(-10, block_size + 10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        # Mark the prefix boundary
        if n_prefix > 0 and n_prefix < block_size:
            ax.axvline(x=n_prefix - 0.5, color='green', linestyle=':', linewidth=2, alpha=0.7)

    # Hide unused subplots
    for idx in range(len(prefix_lengths), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Position Regression by Prefix Diversity\n(6-layer NoPE, freeze-until-first-mlp)', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def plot_summary(results, prefix_lengths):
    """Create summary plot of metrics vs prefix diversity."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract metrics
    correlations = [results[p]['pearson_r'] for p in prefix_lengths]
    mses = [results[p]['mse'] for p in prefix_lengths]
    maes = [results[p]['mae'] for p in prefix_lengths]

    # Plot 1: Pearson correlation
    ax1 = axes[0]
    ax1.plot(prefix_lengths, correlations, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Number of Unique Prefix Tokens', fontsize=12)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Position Correlation vs Prefix Diversity', fontsize=12)
    ax1.set_ylim(-0.1, 1.05)
    ax1.grid(True, alpha=0.3)
    for x, y in zip(prefix_lengths, correlations):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    # Plot 2: MSE
    ax2 = axes[1]
    ax2.plot(prefix_lengths, mses, 'o-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Number of Unique Prefix Tokens', fontsize=12)
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_title('MSE vs Prefix Diversity', fontsize=12)
    ax2.grid(True, alpha=0.3)
    for x, y in zip(prefix_lengths, mses):
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    # Plot 3: MAE
    ax3 = axes[2]
    ax3.plot(prefix_lengths, maes, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Number of Unique Prefix Tokens', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('MAE vs Prefix Diversity', fontsize=12)
    ax3.grid(True, alpha=0.3)
    for x, y in zip(prefix_lengths, maes):
        ax3.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--n_sequences', type=int, default=100, help='Number of sequences per test')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='Wandb run ID to resume')
    parser.add_argument('--wandb_project', type=str, default='position-regression', help='Wandb project')
    args = parser.parse_args()

    # Load model
    model, config, checkpoint = load_checkpoint(args.checkpoint, args.device)
    print(f"Model config: {config.n_layer} layers, block_size={config.block_size}")
    print(f"Use regression: {config.use_regression}")

    # Initialize wandb
    if args.wandb_run_id:
        wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume='must')
    else:
        wandb.init(project=args.wandb_project, name='prefix-diversity-eval')

    # Run evaluation
    results, prefix_lengths = evaluate_prefix_diversity(
        model, config,
        device=args.device,
        n_sequences=args.n_sequences
    )

    # Create plots
    fig_regression = plot_regression_results(results, prefix_lengths, config.block_size)
    fig_summary = plot_summary(results, prefix_lengths)

    # Log to wandb
    wandb.log({
        "prefix_diversity/regression_by_prefix": wandb.Image(fig_regression),
        "prefix_diversity/metrics_summary": wandb.Image(fig_summary),
    })

    # Also log individual metrics
    for n_prefix in prefix_lengths:
        wandb.log({
            f"prefix_diversity/pearson_r_{n_prefix}_unique": results[n_prefix]['pearson_r'],
            f"prefix_diversity/mse_{n_prefix}_unique": results[n_prefix]['mse'],
            f"prefix_diversity/mae_{n_prefix}_unique": results[n_prefix]['mae'],
        })

    # Save plots locally
    out_dir = os.path.dirname(args.checkpoint)
    fig_regression.savefig(os.path.join(out_dir, 'prefix_diversity_regression.png'), dpi=150, bbox_inches='tight')
    fig_summary.savefig(os.path.join(out_dir, 'prefix_diversity_summary.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to {out_dir}")

    plt.close(fig_regression)
    plt.close(fig_summary)

    print("\nResults logged to wandb!")
    wandb.finish()


if __name__ == '__main__':
    main()
