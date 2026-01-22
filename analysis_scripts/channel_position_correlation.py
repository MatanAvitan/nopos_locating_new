"""
Channel-wise Position Correlation Analysis

Analyzes which individual channels (dimensions) in the residual stream
have high correlation with position, before and after attention.

For R0 (full training) and R2 (attention-only training) with 12 heads.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    model_args = checkpoint.get("model_args", {})
    config = TwoLayerMechanismConfig(**model_args)

    # Create and load model
    model = TwoLayerMechanismModel(config)

    # Handle state dict with _orig_mod prefix (from torch.compile)
    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            unwrapped_state_dict[k[len("_orig_mod.") :]] = v
        else:
            unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    model.eval()

    return model, config


def load_owt_data(data_dir: str = "nanoGPT/data/openwebtext"):
    """Load OpenWebText data."""
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Get a batch of sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device)


def compute_channel_correlations(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    n_batches: int = 50,
    batch_size: int = 32,
    block_size: int = 128,
    device: str = "cuda",
):
    """
    Compute per-channel correlation with position for each activation tap point.

    Returns dict of layer_name -> (n_channels,) array of correlations
    """
    model.eval()

    # Collect activations across batches
    activations = {
        "block1_pre_attn": [],  # LN1 output (before attention)
        "block1_post_attn": [],  # After attention residual
        "block1_post_ln2": [],  # After LN2 (before MLP)
        "block1_out": [],  # After MLP residual
        "block2_pre_attn": [],  # LN1 output of block 2
        "block2_post_attn": [],  # After attention residual
        "block2_post_ln2": [],  # After LN2 (before MLP)
        "block2_out": [],  # Final output
    }
    positions_list = []

    with torch.no_grad():
        for _ in range(n_batches):
            tokens = get_batch(data, batch_size, block_size, device)
            B, T = tokens.shape

            # Forward pass with activation capture
            _ = model(tokens, capture_taps=True)

            # Get activations from tap points
            block1 = model.block1
            block2 = model.block2

            # Block 1 activations
            activations["block1_pre_attn"].append(block1.last_ln1_out.cpu())
            activations["block1_post_attn"].append(block1.last_post_attn.cpu())
            activations["block1_post_ln2"].append(block1.last_ln2_out.cpu())
            activations["block1_out"].append(block1.last_block_out.cpu())

            # Block 2 activations
            activations["block2_pre_attn"].append(block2.last_ln1_out.cpu())
            activations["block2_post_attn"].append(block2.last_post_attn.cpu())
            activations["block2_post_ln2"].append(block2.last_ln2_out.cpu())
            activations["block2_out"].append(block2.last_block_out.cpu())

            # Positions
            positions = torch.arange(T).unsqueeze(0).expand(B, -1)  # [B, T]
            positions_list.append(positions)

    # Concatenate all batches: [N, T, d_model]
    for key in activations:
        activations[key] = torch.cat(activations[key], dim=0)
    positions = torch.cat(positions_list, dim=0)  # [N, T]

    # Compute per-channel correlations
    # Flatten batch and position: [N*T, d_model]
    correlations = {}
    positions_flat = positions.reshape(-1).numpy()

    for layer_name, acts in activations.items():
        N, T, d_model = acts.shape
        acts_flat = acts.reshape(-1, d_model).numpy()  # [N*T, d_model]

        # Compute Pearson correlation for each channel
        channel_corrs = np.zeros(d_model)
        for c in range(d_model):
            r, _ = stats.pearsonr(positions_flat, acts_flat[:, c])
            channel_corrs[c] = r

        correlations[layer_name] = channel_corrs

    return correlations, activations, positions


def analyze_correlations(correlations: dict, model_name: str, save_dir: str):
    """Analyze and visualize channel correlations."""
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    for layer_name, corrs in correlations.items():
        abs_corrs = np.abs(corrs)

        # Statistics
        results[layer_name] = {
            "mean_abs_corr": float(np.mean(abs_corrs)),
            "max_abs_corr": float(np.max(abs_corrs)),
            "min_abs_corr": float(np.min(abs_corrs)),
            "std_abs_corr": float(np.std(abs_corrs)),
            "n_high_corr_0.3": int(np.sum(abs_corrs > 0.3)),
            "n_high_corr_0.5": int(np.sum(abs_corrs > 0.5)),
            "n_high_corr_0.7": int(np.sum(abs_corrs > 0.7)),
            "top_10_channels": np.argsort(abs_corrs)[-10:][::-1].tolist(),
            "top_10_corrs": corrs[np.argsort(abs_corrs)[-10:][::-1]].tolist(),
        }

        print(f"\n{layer_name}:")
        print(f"  Mean |r|: {results[layer_name]['mean_abs_corr']:.4f}")
        print(f"  Max |r|:  {results[layer_name]['max_abs_corr']:.4f}")
        print(f"  Channels with |r| > 0.3: {results[layer_name]['n_high_corr_0.3']}")
        print(f"  Channels with |r| > 0.5: {results[layer_name]['n_high_corr_0.5']}")
        print(f"  Channels with |r| > 0.7: {results[layer_name]['n_high_corr_0.7']}")
        print(f"  Top 5 channels: {results[layer_name]['top_10_channels'][:5]}")
        print(
            f"  Top 5 correlations: {[f'{c:.3f}' for c in results[layer_name]['top_10_corrs'][:5]]}"
        )

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    layer_order = [
        "block1_pre_attn",
        "block1_post_attn",
        "block1_post_ln2",
        "block1_out",
        "block2_pre_attn",
        "block2_post_attn",
        "block2_post_ln2",
        "block2_out",
    ]

    for idx, layer_name in enumerate(layer_order):
        ax = axes[idx]
        corrs = correlations[layer_name]

        # Histogram of correlations
        ax.hist(corrs, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
        ax.axvline(x=0.3, color="r", linestyle="--", alpha=0.5, label="|r|=0.3")
        ax.axvline(x=-0.3, color="r", linestyle="--", alpha=0.5)
        ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="|r|=0.5")
        ax.axvline(x=-0.5, color="orange", linestyle="--", alpha=0.5)

        ax.set_xlabel("Pearson r with position")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{layer_name}\nmax|r|={np.max(np.abs(corrs)):.3f}, n(|r|>0.3)={np.sum(np.abs(corrs) > 0.3)}"
        )
        ax.set_xlim(-1, 1)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(f"Channel-Position Correlations: {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"channel_corr_histogram_{model_name}.png"), dpi=150
    )
    plt.savefig(os.path.join(save_dir, f"channel_corr_histogram_{model_name}.pdf"))
    plt.close()

    # Create sorted correlation plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, layer_name in enumerate(layer_order):
        ax = axes[idx]
        corrs = correlations[layer_name]
        sorted_corrs = np.sort(corrs)[::-1]

        ax.plot(sorted_corrs, alpha=0.7)
        ax.axhline(y=0.3, color="r", linestyle="--", alpha=0.5, label="r=0.3")
        ax.axhline(y=-0.3, color="r", linestyle="--", alpha=0.5)
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        ax.set_xlabel("Channel (sorted by correlation)")
        ax.set_ylabel("Pearson r with position")
        ax.set_title(f"{layer_name}")
        ax.set_ylim(-1, 1)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(f"Sorted Channel-Position Correlations: {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"channel_corr_sorted_{model_name}.png"), dpi=150
    )
    plt.savefig(os.path.join(save_dir, f"channel_corr_sorted_{model_name}.pdf"))
    plt.close()

    # Save results
    with open(
        os.path.join(save_dir, f"channel_corr_stats_{model_name}.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)

    # Save raw correlations
    np.savez(
        os.path.join(save_dir, f"channel_correlations_{model_name}.npz"),
        **{k: v for k, v in correlations.items()},
    )

    return results


def compare_models(all_results: dict, save_dir: str):
    """Compare correlations across R0 and R2 models."""

    # Create comparison bar plot
    layer_order = [
        "block1_pre_attn",
        "block1_post_attn",
        "block1_post_ln2",
        "block1_out",
        "block2_pre_attn",
        "block2_post_attn",
        "block2_post_ln2",
        "block2_out",
    ]

    metrics = ["mean_abs_corr", "max_abs_corr", "n_high_corr_0.3", "n_high_corr_0.5"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(layer_order))
        width = 0.35

        r0_vals = [all_results["R0"][layer][metric] for layer in layer_order]
        r2_vals = [all_results["R2"][layer][metric] for layer in layer_order]

        ax.bar(x - width / 2, r0_vals, width, label="R0 (Full)", alpha=0.8)
        ax.bar(x + width / 2, r2_vals, width, label="R2 (Attn2-only)", alpha=0.8)

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(
            [l.replace("_", "\n") for l in layer_order],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.legend()

    plt.suptitle("R0 vs R2: Channel-Position Correlation Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "channel_corr_comparison.png"), dpi=150)
    plt.savefig(os.path.join(save_dir, "channel_corr_comparison.pdf"))
    plt.close()

    # Print summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: Channels with |r| > 0.3")
    print("=" * 80)
    print(f"{'Layer':<25} {'R0':<10} {'R2':<10} {'Diff':<10}")
    print("-" * 55)
    for layer in layer_order:
        r0 = all_results["R0"][layer]["n_high_corr_0.3"]
        r2 = all_results["R2"][layer]["n_high_corr_0.3"]
        print(f"{layer:<25} {r0:<10} {r2:<10} {r0 - r2:<10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r0_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
        help="Path to R0 checkpoint",
    )
    parser.add_argument(
        "--r2_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R2/best_ckpt.pt",
        help="Path to R2 checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="nanoGPT/data/openwebtext",
        help="Path to data directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/channel_position_correlation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--n_batches", type=int, default=50, help="Number of batches to process"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    _, val_data = load_owt_data(args.data_dir)

    all_results = {}

    # Analyze R0
    print("\n" + "=" * 60)
    print("Analyzing R0 (Full Training)")
    print("=" * 60)
    model_r0, config_r0 = load_model(args.r0_checkpoint, args.device)
    print(f"Model config: n_embd={config_r0.n_embd}, n_head={config_r0.n_head}")

    corrs_r0, _, _ = compute_channel_correlations(
        model_r0,
        val_data,
        args.n_batches,
        args.batch_size,
        config_r0.block_size,
        args.device,
    )
    all_results["R0"] = analyze_correlations(corrs_r0, "R0", args.save_dir)
    del model_r0
    torch.cuda.empty_cache()

    # Analyze R2
    print("\n" + "=" * 60)
    print("Analyzing R2 (Attn2-only Training)")
    print("=" * 60)
    model_r2, config_r2 = load_model(args.r2_checkpoint, args.device)

    corrs_r2, _, _ = compute_channel_correlations(
        model_r2,
        val_data,
        args.n_batches,
        args.batch_size,
        config_r2.block_size,
        args.device,
    )
    all_results["R2"] = analyze_correlations(corrs_r2, "R2", args.save_dir)
    del model_r2
    torch.cuda.empty_cache()

    # Compare models
    compare_models(all_results, args.save_dir)

    # Save combined results
    with open(os.path.join(args.save_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()
