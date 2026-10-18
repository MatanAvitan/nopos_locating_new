"""
Evaluate prefix diversity mechanism evolution during training.

Runs prefix diversity evaluation on all checkpoints to see how the
position encoding mechanism develops over training iterations.
"""

import os
import sys
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
import wandb
import argparse
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_position_classifier import GPTPositionClassifier, GPTPositionClassifierConfig


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
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

    iter_num = checkpoint.get('iter_num', 0)

    return model, config, iter_num


def create_prefix_diversity_sequences(n_prefix_unique, block_size, base_token=1000, n_sequences=100):
    """Create sequences with specified prefix diversity."""
    sequences = torch.full((n_sequences, block_size), base_token, dtype=torch.long)

    if n_prefix_unique > 0:
        for i in range(min(n_prefix_unique, block_size)):
            sequences[:, i] = base_token + 1 + i

    return sequences


def evaluate_checkpoint(model, config, device='cuda', n_sequences=100):
    """Evaluate a single checkpoint on prefix diversity."""
    block_size = config.block_size
    results = {}

    prefix_lengths = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    prefix_lengths = [p for p in prefix_lengths if p <= block_size]

    with torch.no_grad():
        for n_prefix in prefix_lengths:
            sequences = create_prefix_diversity_sequences(
                n_prefix_unique=n_prefix,
                block_size=block_size,
                base_token=1000,
                n_sequences=n_sequences
            ).to(device)

            preds, _, _ = model(sequences)
            pred_positions = preds.squeeze(-1).cpu().numpy()

            true_positions = np.arange(block_size)
            pred_flat = pred_positions.flatten()
            true_flat = np.tile(true_positions, n_sequences)

            pearson_r, _ = stats.pearsonr(pred_flat, true_flat)
            mse = np.mean((pred_flat - true_flat) ** 2)
            mae = np.mean(np.abs(pred_flat - true_flat))

            results[n_prefix] = {
                'pearson_r': pearson_r,
                'mse': mse,
                'mae': mae
            }

    return results, prefix_lengths


def get_checkpoint_files(checkpoint_dir):
    """Get all checkpoint files sorted by iteration number."""
    files = os.listdir(checkpoint_dir)
    checkpoints = []

    for f in files:
        if f.startswith('ckpt_') and f.endswith('.pt'):
            match = re.match(r'ckpt_(\d+)\.pt', f)
            if match:
                iter_num = int(match.group(1))
                checkpoints.append((iter_num, os.path.join(checkpoint_dir, f)))

    # Sort by iteration number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def create_random_model(config_from_checkpoint, device='cuda'):
    """Create a fresh randomly initialized model with the same config."""
    model = GPTPositionClassifier(config_from_checkpoint)
    model = model.to(device)
    model.eval()
    return model


def extract_post_ln_representations(model, sequences, device='cuda'):
    """Extract post-final-LayerNorm representations from the model."""
    model.eval()
    with torch.no_grad():
        # Token embeddings
        tok_emb = model.transformer.wte(sequences)

        # Add positional embeddings only if enabled
        if model.config.use_positional_embedding and hasattr(model.transformer, 'wpe'):
            t = sequences.size(1)
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = model.transformer.wpe(pos)
            x = model.transformer.drop(tok_emb + pos_emb)
        else:
            x = model.transformer.drop(tok_emb)

        # Pass through transformer blocks
        for block in model.transformer.h:
            x = block(x)

        # Final layer norm (post-LN representation)
        x = model.transformer.ln_f(x)

    return x


def evaluate_with_linear_probe(model, config, device='cuda', n_sequences=100, n_train_sequences=200):
    """
    Evaluate position prediction using a trained linear probe on post-LN representations.
    This is used for random initialization to show what position info is available.
    """
    block_size = config.block_size
    results = {}

    prefix_lengths = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    prefix_lengths = [p for p in prefix_lengths if p <= block_size]

    for n_prefix in prefix_lengths:
        # Generate training data for the probe
        train_sequences = create_prefix_diversity_sequences(
            n_prefix_unique=n_prefix,
            block_size=block_size,
            base_token=1000,
            n_sequences=n_train_sequences
        ).to(device)

        # Extract representations
        train_reps = extract_post_ln_representations(model, train_sequences, device)
        train_reps = train_reps.cpu().numpy()  # (n_train, block_size, n_embd)

        # Prepare training data: flatten sequences
        n_train, seq_len, n_embd = train_reps.shape
        X_train = train_reps.reshape(-1, n_embd)  # (n_train * seq_len, n_embd)
        y_train = np.tile(np.arange(seq_len), n_train)  # positions

        # Train linear probe (Ridge regression for stability)
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)

        # Generate test data
        test_sequences = create_prefix_diversity_sequences(
            n_prefix_unique=n_prefix,
            block_size=block_size,
            base_token=1000,
            n_sequences=n_sequences
        ).to(device)

        # Extract test representations
        test_reps = extract_post_ln_representations(model, test_sequences, device)
        test_reps = test_reps.cpu().numpy()

        # Predict with probe
        n_test = test_reps.shape[0]
        X_test = test_reps.reshape(-1, n_embd)
        pred_positions = probe.predict(X_test)

        # True positions
        true_positions = np.tile(np.arange(seq_len), n_test)

        # Calculate metrics
        pearson_r, _ = stats.pearsonr(pred_positions, true_positions)
        mse = np.mean((pred_positions - true_positions) ** 2)
        mae = np.mean(np.abs(pred_positions - true_positions))

        results[n_prefix] = {
            'pearson_r': pearson_r,
            'mse': mse,
            'mae': mae
        }

    return results, prefix_lengths


def plot_training_dynamics(all_results, prefix_lengths, use_probe_at_init=False):
    """Create plots showing how prefix diversity mechanism evolves during training."""

    iterations = sorted(all_results.keys())

    # Plot 1: Pearson correlation over training for each prefix length
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(prefix_lengths)))

    for idx, n_prefix in enumerate(prefix_lengths):
        correlations = [all_results[it][n_prefix]['pearson_r'] for it in iterations]
        ax1.plot(iterations, correlations, 'o-', color=colors[idx],
                linewidth=2, markersize=4, label=f'{n_prefix} unique')

    ax1.set_xlabel('Training Iteration', fontsize=12)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Position Correlation vs Training Progress\n(by Prefix Diversity)', fontsize=14)
    ax1.set_ylim(-0.1, 1.05)
    ax1.legend(title='Prefix tokens', loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot 2: Difference between 0-prefix and 1-prefix correlation
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 0-prefix vs 1-prefix correlation
    ax2a = axes[0]
    corr_0 = [all_results[it][0]['pearson_r'] for it in iterations]
    corr_1 = [all_results[it][1]['pearson_r'] for it in iterations]
    corr_diff = [c1 - c0 for c0, c1 in zip(corr_0, corr_1)]

    ax2a.plot(iterations, corr_0, 'o-', color='red', linewidth=2, markersize=4, label='0 unique (all same)')
    ax2a.plot(iterations, corr_1, 'o-', color='blue', linewidth=2, markersize=4, label='1 unique')
    ax2a.fill_between(iterations, corr_0, corr_1, alpha=0.2, color='green')
    ax2a.set_xlabel('Training Iteration', fontsize=12)
    ax2a.set_ylabel('Pearson Correlation', fontsize=12)
    ax2a.set_title('Emergence of Prefix-Dependent Position Encoding', fontsize=12)
    ax2a.legend(loc='lower right')
    ax2a.set_ylim(-0.1, 1.05)
    ax2a.grid(True, alpha=0.3)

    # Right: Correlation gap (benefit of having 1 unique token)
    ax2b = axes[1]
    ax2b.plot(iterations, corr_diff, 'o-', color='green', linewidth=2, markersize=5)
    ax2b.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2b.set_xlabel('Training Iteration', fontsize=12)
    ax2b.set_ylabel('Correlation Gap (1-unique minus 0-unique)', fontsize=12)
    ax2b.set_title('Benefit of Single Prefix Token', fontsize=12)
    ax2b.grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot 3: MAE over training
    fig3, ax3 = plt.subplots(figsize=(12, 7))

    for idx, n_prefix in enumerate(prefix_lengths):
        maes = [all_results[it][n_prefix]['mae'] for it in iterations]
        ax3.plot(iterations, maes, 'o-', color=colors[idx],
                linewidth=2, markersize=4, label=f'{n_prefix} unique')

    ax3.set_xlabel('Training Iteration', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('Position MAE vs Training Progress\n(by Prefix Diversity)', fontsize=14)
    ax3.legend(title='Prefix tokens', loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot 4: Heatmap of correlation over training x prefix diversity
    fig4, ax4 = plt.subplots(figsize=(14, 6))

    # Create matrix: rows = prefix lengths, cols = iterations
    corr_matrix = np.zeros((len(prefix_lengths), len(iterations)))
    for i, n_prefix in enumerate(prefix_lengths):
        for j, it in enumerate(iterations):
            corr_matrix[i, j] = all_results[it][n_prefix]['pearson_r']

    im = ax4.imshow(corr_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_xlabel('Training Iteration', fontsize=12)
    ax4.set_ylabel('Unique Prefix Tokens', fontsize=12)

    # Update title to indicate probe at init
    if use_probe_at_init:
        ax4.set_title('Position Correlation Heatmap\n(Training Progress x Prefix Diversity)\nIter 0 = Linear Probe on Random Init', fontsize=14)
    else:
        ax4.set_title('Position Correlation Heatmap\n(Training Progress x Prefix Diversity)', fontsize=14)

    # Set tick labels
    ax4.set_yticks(range(len(prefix_lengths)))
    ax4.set_yticklabels([str(p) for p in prefix_lengths])

    # Show every nth iteration on x-axis - mark iter 0 as "0 (probe)" if applicable
    n_ticks = min(10, len(iterations))
    tick_indices = np.linspace(0, len(iterations)-1, n_ticks, dtype=int)
    ax4.set_xticks(tick_indices)
    tick_labels = []
    for i in tick_indices:
        it = iterations[i]
        if it == 0 and use_probe_at_init:
            tick_labels.append('0\n(probe)')
        else:
            tick_labels.append(str(it))
    ax4.set_xticklabels(tick_labels)

    # Add vertical line after iteration 0 to separate probe from trained
    if use_probe_at_init and 0 in iterations:
        ax4.axvline(x=0.5, color='white', linestyle='--', linewidth=2)

    plt.colorbar(im, ax=ax4, label='Pearson r')
    plt.tight_layout()

    return fig1, fig2, fig3, fig4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory with checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--n_sequences', type=int, default=50, help='Number of sequences per test')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='Wandb run ID to resume')
    parser.add_argument('--wandb_project', type=str, default='position-regression', help='Wandb project')
    parser.add_argument('--sample_every', type=int, default=1, help='Evaluate every nth checkpoint')
    args = parser.parse_args()

    # Get all checkpoints
    checkpoints = get_checkpoint_files(args.checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    # Sample checkpoints if requested
    if args.sample_every > 1:
        checkpoints = checkpoints[::args.sample_every]
        print(f"Sampling every {args.sample_every}th checkpoint: {len(checkpoints)} remaining")

    # Initialize wandb
    if args.wandb_run_id:
        wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume='must')
    else:
        wandb.init(project=args.wandb_project, name='prefix-diversity-training-dynamics')

    # Evaluate all checkpoints
    all_results = {}
    prefix_lengths = None

    # First, evaluate random initialization (iteration 0)
    print("[0] Evaluating random initialization (iteration 0)...")
    # Load any checkpoint to get the config
    _, config, _ = load_checkpoint(checkpoints[0][1], args.device)

    # Create fresh random model
    random_model = create_random_model(config, args.device)
    results, prefix_lengths = evaluate_checkpoint(random_model, config, args.device, args.n_sequences)
    all_results[0] = results

    # Log random init metrics
    log_dict = {"training_dynamics/iter": 0}
    for n_prefix in prefix_lengths:
        log_dict[f"training_dynamics/pearson_r_{n_prefix}_prefix"] = results[n_prefix]['pearson_r']
        log_dict[f"training_dynamics/mae_{n_prefix}_prefix"] = results[n_prefix]['mae']
    wandb.log(log_dict)

    del random_model
    torch.cuda.empty_cache()
    print("  Random init: r_0={:.4f}, r_1={:.4f}, r_128={:.4f}".format(
        results[0]['pearson_r'], results[1]['pearson_r'], results[128]['pearson_r'] if 128 in results else results[max(results.keys())]['pearson_r']))

    # Now evaluate all training checkpoints
    for i, (iter_num, ckpt_path) in enumerate(checkpoints):
        print(f"[{i+1}/{len(checkpoints)}] Evaluating checkpoint at iteration {iter_num}...")

        model, config, _ = load_checkpoint(ckpt_path, args.device)
        results, prefix_lengths = evaluate_checkpoint(model, config, args.device, args.n_sequences)
        all_results[iter_num] = results

        # Log individual checkpoint metrics
        log_dict = {f"training_dynamics/iter": iter_num}
        for n_prefix in prefix_lengths:
            log_dict[f"training_dynamics/pearson_r_{n_prefix}_prefix"] = results[n_prefix]['pearson_r']
            log_dict[f"training_dynamics/mae_{n_prefix}_prefix"] = results[n_prefix]['mae']

        wandb.log(log_dict)

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Create summary plots
    print("\nCreating training dynamics plots...")
    fig1, fig2, fig3, fig4 = plot_training_dynamics(all_results, prefix_lengths, use_probe_at_init=False)

    # Log plots to wandb
    wandb.log({
        "training_dynamics/correlation_by_prefix": wandb.Image(fig1),
        "training_dynamics/prefix_emergence": wandb.Image(fig2),
        "training_dynamics/mae_by_prefix": wandb.Image(fig3),
        "training_dynamics/correlation_heatmap": wandb.Image(fig4),
    })

    # Save plots locally
    fig1.savefig(os.path.join(args.checkpoint_dir, 'training_dynamics_correlation.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(args.checkpoint_dir, 'training_dynamics_emergence.png'), dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(args.checkpoint_dir, 'training_dynamics_mae.png'), dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(args.checkpoint_dir, 'training_dynamics_heatmap.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to {args.checkpoint_dir}")

    plt.close('all')

    print("\nResults logged to wandb!")
    wandb.finish()


if __name__ == '__main__':
    main()
