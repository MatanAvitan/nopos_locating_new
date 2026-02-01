"""
Position Regression Extrapolation Evaluation

Evaluates position regression accuracy at various sequence lengths to test extrapolation.
Tests on context lengths up to 32x the training length.

Usage:
    python analysis_scripts/evaluate_posreg_extrapolation.py \
        --checkpoint nanoGPT/out-posreg-2layer-longctx-8192/ckpt.pt \
        --context_lengths 8192 16384 32768 65536 131072 262144 \
        --n_batches 50 \
        --wandb
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]

    from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

    config = TwoLayerMechanismConfig(**model_args)
    model = TwoLayerMechanismModel(config)

    # Handle state dict prefix from compile
    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    model.eval()

    train_iter = checkpoint.get("iter_num", 0)
    train_ctx = model_args.get("block_size", 128)

    return model, model_args, train_ctx, train_iter


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Get a random batch of data with position targets."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    # Position targets: 0, 1, 2, ..., block_size-1 for each sample
    positions = (
        torch.arange(block_size, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    return x.to(device), positions.to(device)


@torch.no_grad()
def evaluate_position_accuracy(
    model,
    data: np.ndarray,
    context_length: int,
    batch_size: int,
    n_batches: int,
    device: str,
    train_context: int,
) -> dict:
    """Evaluate position regression accuracy at a given context length."""
    model.eval()

    # Temporarily adjust model's block_size
    original_block_size = model.config.block_size
    model.config.block_size = context_length

    all_predictions = []
    all_targets = []

    for _ in range(n_batches):
        x, positions = get_batch(data, batch_size, context_length, device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Forward pass - assume model returns position predictions
            outputs = model(x)
            if isinstance(outputs, tuple):
                pos_preds = outputs[0]  # Position predictions
            else:
                pos_preds = outputs

        # Flatten predictions and targets
        all_predictions.append(pos_preds.float().cpu().numpy().flatten())
        all_targets.append(positions.cpu().numpy().flatten())

    # Restore original block_size
    model.config.block_size = original_block_size

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    # R² score
    ss_res = np.sum((predictions - targets) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Correlation
    r, p = stats.pearsonr(predictions, targets)

    # Per-position metrics (binned for long sequences)
    n_bins = min(100, context_length)
    bin_size = context_length // n_bins
    per_position_mae = []
    for i in range(n_bins):
        pos_start = i * bin_size
        pos_end = (i + 1) * bin_size
        mask = (targets >= pos_start) & (targets < pos_end)
        if mask.sum() > 0:
            per_position_mae.append(np.mean(np.abs(predictions[mask] - targets[mask])))
        else:
            per_position_mae.append(np.nan)

    return {
        "context_length": context_length,
        "extrapolation_ratio": context_length / train_context,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "correlation": r,
        "per_position_mae": per_position_mae,
        "n_bins": n_bins,
    }


def plot_extrapolation_results(results: list, train_ctx: int, save_path: str):
    """Create extrapolation visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    contexts = [r["context_length"] for r in results]
    extrapolation_ratios = [r["extrapolation_ratio"] for r in results]

    # Plot 1: MAE vs context length
    ax1 = axes[0]
    maes = [r["mae"] for r in results]
    ax1.plot(contexts, maes, "o-", linewidth=2, markersize=8)
    ax1.axvline(
        x=train_ctx,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Train ctx ({train_ctx})",
    )
    ax1.set_xlabel("Context Length", fontsize=12)
    ax1.set_ylabel("Mean Absolute Error", fontsize=12)
    ax1.set_title("Position Prediction MAE", fontsize=14)
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: R² vs context length
    ax2 = axes[1]
    r2s = [r["r2"] for r in results]
    ax2.plot(contexts, r2s, "o-", linewidth=2, markersize=8, color="green")
    ax2.axvline(
        x=train_ctx,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Train ctx ({train_ctx})",
    )
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Context Length", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=12)
    ax2.set_title("Position Prediction R²", fontsize=14)
    ax2.set_xscale("log", base=2)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Relative MAE (normalized by context length)
    ax3 = axes[2]
    relative_maes = [r["mae"] / r["context_length"] for r in results]
    ax3.plot(
        extrapolation_ratios,
        relative_maes,
        "o-",
        linewidth=2,
        markersize=8,
        color="purple",
    )
    ax3.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="In-distribution")
    ax3.set_xlabel("Extrapolation Ratio (ctx / train_ctx)", fontsize=12)
    ax3.set_ylabel("MAE / Context Length", fontsize=12)
    ax3.set_title("Relative Position Error", fontsize=14)
    ax3.set_xscale("log", base=2)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Extrapolation plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate position regression extrapolation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Context lengths to evaluate (include training length)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=50,
        help="Number of batches per context length",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/posreg_extrapolation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to W&B",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nope-position-regression",
        help="W&B project name",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_args, train_ctx, train_iter = load_model(args.checkpoint, device)
    print(f"Training context length: {train_ctx}")
    print(f"Training iterations: {train_iter}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Load validation data
    data_path = (
        Path(__file__).parent.parent / "nanoGPT" / "data" / args.dataset / "val.bin"
    )
    print(f"\nLoading validation data from: {data_path}")
    val_data = np.memmap(data_path, dtype=np.uint16, mode="r")
    print(f"Validation data size: {len(val_data):,} tokens")

    # Setup W&B
    if args.wandb:
        import wandb

        run_name = f"extrapolation-{Path(args.checkpoint).parent.name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "checkpoint": args.checkpoint,
                "train_context": train_ctx,
                "context_lengths": args.context_lengths,
                **model_args,
            },
        )

    # Evaluate at each context length
    results = []
    print("\nEvaluating position regression at different context lengths:")
    print("-" * 80)
    print(f"{'Context':<12} {'Extrap':<12} {'MAE':<12} {'R²':<12} {'Correlation':<12}")
    print("-" * 80)

    for ctx_len in sorted(args.context_lengths):
        if ctx_len > len(val_data) - 1:
            print(f"Skipping context {ctx_len}: exceeds data length")
            continue

        result = evaluate_position_accuracy(
            model=model,
            data=val_data,
            context_length=ctx_len,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            device=device,
            train_context=train_ctx,
        )
        results.append(result)

        extrap_str = (
            f"{result['extrapolation_ratio']:.1f}x"
            if result["extrapolation_ratio"] > 1
            else "in-dist"
        )
        print(
            f"{ctx_len:<12} {extrap_str:<12} {result['mae']:<12.2f} {result['r2']:<12.4f} {result['correlation']:<12.4f}"
        )

        if args.wandb:
            wandb.log(
                {
                    f"mae/ctx_{ctx_len}": result["mae"],
                    f"r2/ctx_{ctx_len}": result["r2"],
                    f"correlation/ctx_{ctx_len}": result["correlation"],
                    "context_length": ctx_len,
                }
            )

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRAPOLATION SUMMARY")
    print("=" * 80)

    in_dist = next((r for r in results if r["context_length"] == train_ctx), results[0])
    print(f"\nIn-distribution (ctx={in_dist['context_length']}):")
    print(f"  MAE: {in_dist['mae']:.2f}, R²: {in_dist['r2']:.4f}")

    extrap_results = [r for r in results if r["extrapolation_ratio"] > 1]
    if extrap_results:
        print(f"\nExtrapolation performance degradation:")
        for r in extrap_results:
            degradation = r["mae"] / in_dist["mae"]
            print(
                f"  {r['extrapolation_ratio']:.0f}x: MAE={r['mae']:.2f} ({degradation:.1f}x worse), R²={r['r2']:.4f}"
            )

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = Path(args.checkpoint).parent.name

    # Remove non-serializable per_position_mae from JSON
    json_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "per_position_mae"}
        json_results.append(r_copy)

    results_file = Path(args.save_dir) / f"extrapolation_{ckpt_name}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "train_context": train_ctx,
                "results": json_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_file}")

    # Create plot
    plot_path = Path(args.save_dir) / f"extrapolation_{ckpt_name}.png"
    plot_extrapolation_results(results, train_ctx, str(plot_path))

    if args.wandb:
        import wandb

        # Log summary table
        table = wandb.Table(
            columns=["Context", "Extrapolation", "MAE", "R²", "Correlation"],
            data=[
                [
                    r["context_length"],
                    f"{r['extrapolation_ratio']:.1f}x",
                    r["mae"],
                    r["r2"],
                    r["correlation"],
                ]
                for r in results
            ],
        )
        wandb.log({"extrapolation_table": table})
        wandb.log({"extrapolation_plot": wandb.Image(str(plot_path))})
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
