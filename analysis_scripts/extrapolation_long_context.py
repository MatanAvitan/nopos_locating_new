"""
Long-Context Extrapolation Analysis

Tests position encoding extrapolation to very long sequences.
Uses adaptive batch sizes to handle memory constraints.

Memory constraints (attention is O(L²)):
- L=4096: ~800MB attention matrix (feasible with batch_size=16)
- L=8192: ~3.2GB attention matrix (feasible with batch_size=4)
- L=16384: ~12.8GB attention matrix (feasible with batch_size=1)
- L=32768: ~51GB attention matrix (might work on A100 80GB with batch_size=1)
- L=65536+: Requires different approach (not feasible with full attention)

We test up to the maximum feasible length and report the extrapolation trend.
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    config = TwoLayerMechanismConfig(**model_args)
    model = TwoLayerMechanismModel(config)

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
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Get a batch of sequences."""
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError(f"Data too short for block_size={block_size}")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device)


def get_adaptive_batch_params(context_length: int, base_batch_size: int = 32):
    """
    Get batch size and number of batches for a given context length.
    Keeps total samples roughly constant while avoiding OOM.
    """
    # Memory scales as L² for attention
    # Base: L=128 with batch_size=32
    base_memory = 128 * 128 * 32

    # Target similar total memory usage
    target_batch_size = max(1, int(base_memory / (context_length * context_length)))
    target_batch_size = min(target_batch_size, base_batch_size)

    # Adjust n_batches to maintain similar sample count
    base_samples = 32 * 50  # 1600 sequences
    target_n_batches = max(10, base_samples // target_batch_size)

    # Cap based on context length to avoid excessive runtime
    if context_length >= 16384:
        target_n_batches = min(target_n_batches, 20)
        target_batch_size = min(target_batch_size, 2)
    elif context_length >= 8192:
        target_n_batches = min(target_n_batches, 50)
        target_batch_size = min(target_batch_size, 4)
    elif context_length >= 4096:
        target_n_batches = min(target_n_batches, 100)
        target_batch_size = min(target_batch_size, 8)

    return target_batch_size, target_n_batches


def compute_extrapolation_metrics(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    context_length: int,
    device: str = "cuda",
):
    """
    Compute position encoding metrics at a given context length.
    Uses adaptive batch size based on memory constraints.
    """
    batch_size, n_batches = get_adaptive_batch_params(context_length)
    total_samples = batch_size * n_batches

    print(
        f"  L={context_length}: {n_batches} batches x {batch_size} = {total_samples} sequences"
    )

    model.eval()

    # Collect activations
    post_attn_acts = []
    positions_list = []

    try:
        with torch.no_grad():
            for batch_idx in range(n_batches):
                tokens = get_batch(data, batch_size, context_length, device)
                B_size, T = tokens.shape

                # Forward pass
                _ = model(tokens, capture_taps=True)

                block2 = model.block2
                post_attn_acts.append(block2.last_post_attn.cpu())

                positions = torch.arange(T).unsqueeze(0).expand(B_size, -1)
                positions_list.append(positions)

                # Clear cache periodically for long sequences
                if context_length >= 4096 and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        # Concatenate
        post_attn_acts = torch.cat(post_attn_acts, dim=0).numpy()
        positions = torch.cat(positions_list, dim=0).numpy()

        N, T, d_model = post_attn_acts.shape
        positions_flat = positions.reshape(-1)
        acts_flat = post_attn_acts.reshape(-1, d_model)

        # Linear probe R²
        X_train, X_test, y_train, y_test = train_test_split(
            acts_flat, positions_flat, test_size=0.2, random_state=42
        )
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        linear_probe_r2 = 1 - ss_res / ss_tot

        # Per-channel correlations
        neuron_corrs = np.array(
            [stats.pearsonr(positions_flat, acts_flat[:, c])[0] for c in range(d_model)]
        )

        # Threshold counts
        thresholds = [0.3, 0.5, 0.7, 0.9, 0.95]
        neuron_counts = {
            f"|r|>{t}": int(np.sum(np.abs(neuron_corrs) > t)) for t in thresholds
        }

        return {
            "context_length": context_length,
            "linear_probe_r2": float(linear_probe_r2),
            "neuron_counts": neuron_counts,
            "mean_abs_corr": float(np.mean(np.abs(neuron_corrs))),
            "max_abs_corr": float(np.max(np.abs(neuron_corrs))),
            "n_samples": total_samples,
            "success": True,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM at L={context_length}, skipping...")
            torch.cuda.empty_cache()
            return {
                "context_length": context_length,
                "linear_probe_r2": None,
                "success": False,
                "error": "OOM",
            }
        else:
            raise


def plot_long_extrapolation(results: dict, save_dir: str):
    """Create plot of extrapolation over log-scale context lengths."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): R² vs context length (log scale)
    ax = axes[0]

    for model_name, color, marker in [("R0", "#1f77b4", "o"), ("R2", "#ff7f0e", "s")]:
        lengths = []
        r2_values = []
        for L, data in sorted(results[model_name].items()):
            if data["success"]:
                lengths.append(int(L))
                r2_values.append(data["linear_probe_r2"])

        ax.plot(
            lengths,
            r2_values,
            f"{marker}-",
            color=color,
            label=model_name,
            linewidth=2,
            markersize=8,
        )

    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5, label="Training length")
    ax.set_xlabel("Context length", fontsize=11)
    ax.set_ylabel("Linear probe R²", fontsize=11)
    ax.set_title("Position Decoding vs Context Length", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (b): Mean |r| vs context length
    ax = axes[1]

    for model_name, color, marker in [("R0", "#1f77b4", "o"), ("R2", "#ff7f0e", "s")]:
        lengths = []
        mean_corr = []
        for L, data in sorted(results[model_name].items()):
            if data["success"]:
                lengths.append(int(L))
                mean_corr.append(data["mean_abs_corr"])

        ax.plot(
            lengths,
            mean_corr,
            f"{marker}-",
            color=color,
            label=model_name,
            linewidth=2,
            markersize=8,
        )

    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Context length", fontsize=11)
    ax.set_ylabel("Mean |r| with position", fontsize=11)
    ax.set_title("Average Channel Correlation vs Context Length", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "extrapolation_long_context.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(save_dir, "extrapolation_long_context.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print(f"Figure saved to {save_dir}/extrapolation_long_context.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r0_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument(
        "--r2_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R2/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/extrapolation_long_context"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Maximum context length to test (default: 16384)",
    )
    args = parser.parse_args()

    # Context lengths to test (powers of 2)
    context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    if args.max_length >= 16384:
        context_lengths.append(16384)
    if args.max_length >= 32768:
        context_lengths.append(32768)

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading data...")
    val_data = load_owt_data(args.data_dir)
    print(f"Validation data size: {len(val_data):,} tokens")

    all_results = {}

    for model_name, ckpt_path in [
        ("R0", args.r0_checkpoint),
        ("R2", args.r2_checkpoint),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Analyzing {model_name}")
        print(f"{'=' * 60}")

        model, config = load_model(ckpt_path, args.device)
        all_results[model_name] = {}

        for L in context_lengths:
            print(f"\nContext length: {L}")

            # Check if data is long enough
            if L > len(val_data) - 100:
                print(f"  Skipping L={L}: data too short")
                continue

            result = compute_extrapolation_metrics(model, val_data, L, args.device)
            all_results[model_name][L] = result

            if result["success"]:
                print(f"  R²: {result['linear_probe_r2']:.4f}")
                print(f"  Mean |r|: {result['mean_abs_corr']:.4f}")
                print(f"  |r|>0.9: {result['neuron_counts']['|r|>0.9']}")

            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(
        os.path.join(args.save_dir, "extrapolation_long_context_results.json"), "w"
    ) as f:
        json.dump(all_results, f, indent=2)

    # Plot
    plot_long_extrapolation(all_results, args.save_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXTRAPOLATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<8} {'L':<8} {'R²':<10} {'Mean |r|':<10}")
    print("-" * 40)

    for model_name in ["R0", "R2"]:
        for L in sorted(all_results[model_name].keys()):
            r = all_results[model_name][L]
            if r["success"]:
                print(
                    f"{model_name:<8} {L:<8} {r['linear_probe_r2']:<10.4f} {r['mean_abs_corr']:<10.4f}"
                )

    # Calculate degradation rate
    print(f"\n{'=' * 60}")
    print("DEGRADATION ANALYSIS")
    print(f"{'=' * 60}")

    for model_name in ["R0", "R2"]:
        results = all_results[model_name]
        if 128 in results and results[128]["success"]:
            base_r2 = results[128]["linear_probe_r2"]
            print(f"\n{model_name}:")
            print(f"  Base R² at L=128: {base_r2:.4f}")

            for L in sorted(results.keys()):
                if L > 128 and results[L]["success"]:
                    r2 = results[L]["linear_probe_r2"]
                    pct_change = (r2 - base_r2) / base_r2 * 100
                    print(f"  L={L}: R²={r2:.4f} ({pct_change:+.1f}%)")

    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()
