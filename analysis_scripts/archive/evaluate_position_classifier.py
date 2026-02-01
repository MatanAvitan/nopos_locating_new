"""
Evaluation script for Position Classifier with Distinctive Token Analysis.

Evaluates the trained position classifier on sequences with varying numbers
of distinctive tokens:
- 1 token: All same token [A, A, A, ..., A]
- 2 tokens: Two alternating [A, B, A, B, ...]
- 4, 8, 16, 32, 64, 128 tokens: Increasing diversity

Generates wandb plots:
- Line plot: accuracy vs position for each distinctive count
- Heatmap: accuracy[position, n_distinctive]
- Summary table

Usage:
    python evaluate_position_classifier.py \
        --checkpoint out-position-classifier-full/ckpt.pt \
        --wandb_project position-classifier \
        --run_name eval-full
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_position_classifier import GPTPositionClassifier, GPTPositionClassifierConfig


def load_model(checkpoint_path, device="cuda"):
    """Load trained position classifier from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]

    config = GPTPositionClassifierConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 128),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        bias=model_args.get("bias", False),
        use_positional_embedding=model_args.get("use_positional_embedding", False),
        norm_type=model_args.get("norm_type", "layernorm"),
    )

    model = GPTPositionClassifier(config)

    # Handle compiled model prefix
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def generate_distinctive_sequences(n_distinctive, n_samples, seq_len, vocab_size):
    """
    Generate sequences with exactly n_distinctive unique tokens.

    Args:
        n_distinctive: Number of unique tokens in each sequence
        n_samples: Number of sequences to generate
        seq_len: Length of each sequence
        vocab_size: Total vocabulary size

    Returns:
        tokens: (n_samples, seq_len) tensor
    """
    if n_distinctive > vocab_size:
        n_distinctive = vocab_size

    # Sample n_distinctive unique tokens for each sequence
    tokens = torch.zeros(n_samples, seq_len, dtype=torch.long)

    for i in range(n_samples):
        # Pick n_distinctive random tokens from vocabulary
        unique_tokens = torch.randperm(vocab_size)[:n_distinctive]

        if n_distinctive == 1:
            # All same token
            tokens[i, :] = unique_tokens[0]
        else:
            # Fill sequence by cycling through unique tokens
            for j in range(seq_len):
                tokens[i, j] = unique_tokens[j % n_distinctive]

    return tokens


def evaluate_with_distinctive_tokens(model, n_distinctive, n_samples=1000, device="cuda"):
    """
    Evaluate model on sequences with n_distinctive unique tokens.

    Returns per-position accuracy array.
    """
    model.eval()
    seq_len = model.config.block_size
    vocab_size = model.config.vocab_size

    tokens = generate_distinctive_sequences(n_distinctive, n_samples, seq_len, vocab_size)
    tokens = tokens.to(device)

    # Position labels: [0, 1, 2, ..., seq_len-1]
    targets = torch.arange(seq_len).unsqueeze(0).expand(n_samples, -1).to(device)

    # Batch inference
    batch_size = 100
    per_position_correct = torch.zeros(seq_len)
    per_position_total = torch.zeros(seq_len)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_tokens = tokens[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            logits, _ = model(batch_tokens)
            preds = logits.argmax(dim=-1)

            correct = (preds == batch_targets).float().cpu()
            for pos in range(seq_len):
                per_position_correct[pos] += correct[:, pos].sum().item()
                per_position_total[pos] += batch_tokens.size(0)

    accuracy = per_position_correct / per_position_total
    return accuracy.numpy()


def create_wandb_plots(results, run_name):
    """Create and log wandb plots."""
    import wandb

    seq_len = len(list(results.values())[0])
    distinctive_counts = sorted(results.keys())

    # Create accuracy matrix for heatmap
    accuracy_matrix = np.zeros((len(distinctive_counts), seq_len))
    for i, n_dist in enumerate(distinctive_counts):
        accuracy_matrix[i, :] = results[n_dist]

    # 1. Line plot: accuracy vs position
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(distinctive_counts)))
    for i, n_dist in enumerate(distinctive_counts):
        ax1.plot(range(seq_len), results[n_dist],
                 label=f"{n_dist} tokens", color=colors[i], linewidth=1.5)
    ax1.set_xlabel("Position", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"Position Prediction Accuracy by Token Diversity - {run_name}", fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"accuracy_vs_position": wandb.Image(fig1)})
    plt.close(fig1)

    # 2. Heatmap: accuracy[n_distinctive, position]
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    sns.heatmap(accuracy_matrix, ax=ax2,
                xticklabels=10,  # Show every 10th position
                yticklabels=[str(n) for n in distinctive_counts],
                cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Accuracy"})
    ax2.set_xlabel("Position", fontsize=12)
    ax2.set_ylabel("Distinctive Tokens", fontsize=12)
    ax2.set_title(f"Position Accuracy Heatmap - {run_name}", fontsize=14)
    plt.tight_layout()
    wandb.log({"accuracy_heatmap": wandb.Image(fig2)})
    plt.close(fig2)

    # 3. Summary bar chart: mean accuracy per distinctive count
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    mean_accs = [results[n].mean() for n in distinctive_counts]
    bars = ax3.bar([str(n) for n in distinctive_counts], mean_accs, color=colors)
    ax3.set_xlabel("Number of Distinctive Tokens", fontsize=12)
    ax3.set_ylabel("Mean Accuracy", fontsize=12)
    ax3.set_title(f"Mean Position Accuracy by Token Diversity - {run_name}", fontsize=14)
    ax3.set_ylim(0, 1.05)
    for bar, acc in zip(bars, mean_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{acc:.3f}", ha="center", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    wandb.log({"mean_accuracy_by_diversity": wandb.Image(fig3)})
    plt.close(fig3)

    # 4. Log table with all results
    table = wandb.Table(columns=["distinctive_tokens", "position", "accuracy"])
    for n_dist in distinctive_counts:
        for pos, acc in enumerate(results[n_dist]):
            table.add_data(n_dist, pos, acc)
    wandb.log({"accuracy_table": table})

    # 5. Log summary metrics
    summary = {
        "summary/overall_mean_accuracy": np.mean([results[n].mean() for n in distinctive_counts]),
    }
    for n_dist in distinctive_counts:
        summary[f"summary/mean_acc_{n_dist}_tokens"] = results[n_dist].mean()
        summary[f"summary/min_acc_{n_dist}_tokens"] = results[n_dist].min()
        summary[f"summary/max_acc_{n_dist}_tokens"] = results[n_dist].max()
    wandb.log(summary)


def save_local_plots(results, output_dir, run_name):
    """Save plots locally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len = len(list(results.values())[0])
    distinctive_counts = sorted(results.keys())

    # Accuracy matrix
    accuracy_matrix = np.zeros((len(distinctive_counts), seq_len))
    for i, n_dist in enumerate(distinctive_counts):
        accuracy_matrix[i, :] = results[n_dist]

    # Line plot
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(distinctive_counts)))
    for i, n_dist in enumerate(distinctive_counts):
        ax1.plot(range(seq_len), results[n_dist],
                 label=f"{n_dist} tokens", color=colors[i], linewidth=1.5)
    ax1.set_xlabel("Position", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"Position Prediction Accuracy - {run_name}", fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_vs_position_{run_name}.png", dpi=150)
    plt.close(fig1)

    # Heatmap
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    sns.heatmap(accuracy_matrix, ax=ax2,
                xticklabels=10,
                yticklabels=[str(n) for n in distinctive_counts],
                cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Accuracy"})
    ax2.set_xlabel("Position", fontsize=12)
    ax2.set_ylabel("Distinctive Tokens", fontsize=12)
    ax2.set_title(f"Position Accuracy Heatmap - {run_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_heatmap_{run_name}.png", dpi=150)
    plt.close(fig2)

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Position Classifier")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--wandb_project", type=str, default="position-classifier",
                        help="Wandb project name")
    parser.add_argument("--run_name", type=str, default="eval",
                        help="Wandb run name")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples per distinctive token count")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="results/position_classifier",
                        help="Directory for local output")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    print("="*70)
    print("POSITION CLASSIFIER EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Samples per condition: {args.n_samples}")

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(args.checkpoint, args.device)
    seq_len = model.config.block_size

    print(f"Model loaded: {model.config.n_layer} layers, {model.config.n_head} heads")
    print(f"Sequence length: {seq_len}")
    if "val_accuracy" in checkpoint:
        print(f"Training val accuracy: {checkpoint['val_accuracy']:.4f}")

    # Distinctive token counts to evaluate
    distinctive_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    # Filter to valid values (can't have more distinctive tokens than seq_len)
    distinctive_counts = [n for n in distinctive_counts if n <= seq_len]

    # Run evaluation for each condition
    results = {}
    print("\nRunning evaluation...")
    for n_dist in distinctive_counts:
        print(f"  Evaluating with {n_dist} distinctive tokens...")
        accuracy = evaluate_with_distinctive_tokens(
            model, n_dist, n_samples=args.n_samples, device=args.device
        )
        results[n_dist] = accuracy
        print(f"    Mean accuracy: {accuracy.mean():.4f}, "
              f"Min: {accuracy.min():.4f}, Max: {accuracy.max():.4f}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Distinctive Tokens':<20} {'Mean Acc':>12} {'Min Acc':>12} {'Max Acc':>12}")
    print("-"*60)
    for n_dist in distinctive_counts:
        acc = results[n_dist]
        print(f"{n_dist:<20} {acc.mean():>12.4f} {acc.min():>12.4f} {acc.max():>12.4f}")

    # Save local plots
    save_local_plots(results, args.output_dir, args.run_name)

    # Wandb logging
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "checkpoint": args.checkpoint,
                "n_samples": args.n_samples,
                "seq_len": seq_len,
                "distinctive_counts": distinctive_counts,
            }
        )
        create_wandb_plots(results, args.run_name)
        wandb.finish()
        print(f"\nWandb logs uploaded to project: {args.wandb_project}")

    # Save results
    import json
    results_path = Path(args.output_dir) / f"results_{args.run_name}.json"
    with open(results_path, "w") as f:
        json.dump({str(k): v.tolist() for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
