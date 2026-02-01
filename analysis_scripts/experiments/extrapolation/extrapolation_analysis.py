"""
Extrapolation Analysis: Testing Position Encoding at Longer Sequences

Tests whether the proposed positional encoding mechanisms (R0's BOS-anchoring,
R2's low-rank write subspace) extrapolate beyond the training context length (128 tokens).

Models were trained on block_size=128, we test on 128, 256, 512, 1024.
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

BOS_TOKEN_ID = 50256


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
    ix = torch.randint(len(data) - (block_size - 1), (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(
                np.concatenate(
                    [[BOS_TOKEN_ID], data[i : i + block_size - 1].astype(np.int64)]
                )
            )
            for i in ix
        ]
    )
    return x.to(device)


def get_block2_write_map(model: TwoLayerMechanismModel):
    """Extract B = W_O @ W_V from Block 2 attention."""
    attn = model.block2.attn
    c_attn_weight = attn.c_attn.weight
    d_model = c_attn_weight.shape[1]
    W_V = c_attn_weight[2 * d_model :, :]
    W_O = attn.c_proj.weight
    B = W_O @ W_V
    return B.detach().cpu().numpy()


def compute_extrapolation_metrics(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    B: np.ndarray,
    context_length: int,
    n_batches: int = 100,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Compute position encoding metrics at a given context length.

    Returns dict with:
    - linear_probe_r2: R² from linear regression on position
    - per_position_corr: Pearson r at each position
    - neuron_counts: dict of threshold -> count
    - svd_counts: dict of threshold -> count
    - mean_abs_corr_neuron: mean |r| in neuron basis
    - mean_abs_corr_svd: mean |r| in SVD basis
    """
    model.eval()

    # Compute SVD of B
    U, S, Vt = np.linalg.svd(B, full_matrices=True)

    # Collect activations
    post_attn_acts = []
    positions_list = []

    # Adjust batch size for longer sequences to avoid OOM
    effective_batch_size = batch_size
    if context_length > 512:
        effective_batch_size = max(8, batch_size // 4)
    elif context_length > 256:
        effective_batch_size = max(16, batch_size // 2)

    # Adjust n_batches to maintain similar total samples
    effective_n_batches = n_batches * (batch_size // effective_batch_size)

    print(
        f"  Context {context_length}: {effective_n_batches} batches x {effective_batch_size} = {effective_n_batches * effective_batch_size} sequences"
    )

    with torch.no_grad():
        for batch_idx in range(effective_n_batches):
            tokens = get_batch(data, effective_batch_size, context_length, device)
            B_size, T = tokens.shape

            # Forward pass with activation capture
            _ = model(tokens, capture_taps=True)

            block2 = model.block2
            post_attn_acts.append(block2.last_post_attn.cpu())

            # Positions
            positions = torch.arange(T).unsqueeze(0).expand(B_size, -1)
            positions_list.append(positions)

    # Concatenate
    post_attn_acts = torch.cat(post_attn_acts, dim=0).numpy()  # [N, T, d_model]
    positions = torch.cat(positions_list, dim=0).numpy()  # [N, T]

    N, T, d_model = post_attn_acts.shape

    # Flatten for correlation and probe
    positions_flat = positions.reshape(-1)  # [N*T]
    acts_flat = post_attn_acts.reshape(-1, d_model)  # [N*T, d_model]

    # 1. Linear probe R²
    X_train, X_test, y_train, y_test = train_test_split(
        acts_flat, positions_flat, test_size=0.2, random_state=42
    )
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    # Compute R²
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    linear_probe_r2 = 1 - ss_res / ss_tot

    # 2. Per-position correlation (average neuron correlation at each position)
    per_position_corr = np.zeros(T)
    for t in range(T):
        # Get activations at position t across all sequences
        acts_at_t = post_attn_acts[:, t, :]  # [N, d_model]
        # Compute mean absolute correlation across all neurons
        corrs = []
        for d in range(d_model):
            # All sequences have the same position t, so we correlate with sequence index
            # Actually, we want to see how well the activation predicts the position t
            # But t is constant, so we need a different approach
            pass
        # Instead, compute how "position-like" the activations are
        # by correlating each neuron with position across all (sequence, position) pairs
        pass

    # For per-position analysis, compute correlation of each neuron with position
    # and then look at how this varies
    neuron_corrs = np.zeros(d_model)
    for c in range(d_model):
        r, _ = stats.pearsonr(positions_flat, acts_flat[:, c])
        neuron_corrs[c] = r

    # SVD basis correlations
    acts_svd = acts_flat @ U
    svd_corrs = np.zeros(d_model)
    for c in range(d_model):
        r, _ = stats.pearsonr(positions_flat, acts_svd[:, c])
        svd_corrs[c] = r

    # 3. Count channels at thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9, 0.95]
    neuron_counts = {
        f"|r|>{t}": int(np.sum(np.abs(neuron_corrs) > t)) for t in thresholds
    }
    svd_counts = {f"|r|>{t}": int(np.sum(np.abs(svd_corrs) > t)) for t in thresholds}

    # 4. Per-position breakdown: use the global probe to predict positions in each bucket
    # This is faster than fitting separate probes
    n_buckets = min(8, T // 16)  # At most 8 buckets
    bucket_size = T // n_buckets
    per_bucket_r2 = []
    bucket_labels = []

    for i in range(n_buckets):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < n_buckets - 1 else T
        mask = (positions_flat >= start) & (positions_flat < end)

        if np.sum(mask) > 100:  # Need enough samples
            X_bucket = acts_flat[mask]
            y_bucket = positions_flat[mask]

            # Use the global probe to predict
            y_pred_bucket = probe.predict(X_bucket)

            ss_res = np.sum((y_bucket - y_pred_bucket) ** 2)
            ss_tot = np.sum((y_bucket - np.mean(y_bucket)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            per_bucket_r2.append(r2)
            bucket_labels.append(f"{start}-{end}")

    return {
        "context_length": context_length,
        "linear_probe_r2": float(linear_probe_r2),
        "neuron_counts": neuron_counts,
        "svd_counts": svd_counts,
        "mean_abs_corr_neuron": float(np.mean(np.abs(neuron_corrs))),
        "mean_abs_corr_svd": float(np.mean(np.abs(svd_corrs))),
        "max_abs_corr_neuron": float(np.max(np.abs(neuron_corrs))),
        "max_abs_corr_svd": float(np.max(np.abs(svd_corrs))),
        "per_bucket_r2": per_bucket_r2,
        "bucket_labels": bucket_labels,
    }


def plot_extrapolation_results(results: dict, save_dir: str):
    """Create plots comparing extrapolation performance."""
    os.makedirs(save_dir, exist_ok=True)

    context_lengths = [128, 256, 512, 1024]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel (a): Linear probe R² vs context length
    ax = axes[0, 0]
    for model_name, color in [("R0", "#1f77b4"), ("R2", "#ff7f0e")]:
        r2_values = [results[model_name][L]["linear_probe_r2"] for L in context_lengths]
        ax.plot(
            context_lengths,
            r2_values,
            "o-",
            color=color,
            label=model_name,
            linewidth=2,
            markersize=8,
        )

    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5, label="Training length")
    ax.set_xlabel("Context length", fontsize=10)
    ax.set_ylabel("Linear probe R²", fontsize=10)
    ax.set_title("(a) Position decoding accuracy", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(context_lengths)
    ax.set_xticklabels(context_lengths)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (b): Neuron count |r|>0.95 vs context length
    ax = axes[0, 1]
    for model_name, color in [("R0", "#1f77b4"), ("R2", "#ff7f0e")]:
        counts = [
            results[model_name][L]["neuron_counts"]["|r|>0.95"] for L in context_lengths
        ]
        ax.plot(
            context_lengths,
            counts,
            "o-",
            color=color,
            label=f"{model_name} neuron",
            linewidth=2,
            markersize=8,
        )

        counts_svd = [
            results[model_name][L]["svd_counts"]["|r|>0.95"] for L in context_lengths
        ]
        ax.plot(
            context_lengths,
            counts_svd,
            "s--",
            color=color,
            label=f"{model_name} SVD",
            linewidth=1.5,
            markersize=6,
            alpha=0.7,
        )

    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Context length", fontsize=10)
    ax.set_ylabel("Channels with |r|>0.95", fontsize=10)
    ax.set_title("(b) Strong position channels", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(context_lengths)
    ax.set_xticklabels(context_lengths)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (c): Mean |r| vs context length
    ax = axes[1, 0]
    for model_name, color in [("R0", "#1f77b4"), ("R2", "#ff7f0e")]:
        mean_corr = [
            results[model_name][L]["mean_abs_corr_neuron"] for L in context_lengths
        ]
        ax.plot(
            context_lengths,
            mean_corr,
            "o-",
            color=color,
            label=f"{model_name} neuron",
            linewidth=2,
            markersize=8,
        )

        mean_corr_svd = [
            results[model_name][L]["mean_abs_corr_svd"] for L in context_lengths
        ]
        ax.plot(
            context_lengths,
            mean_corr_svd,
            "s--",
            color=color,
            label=f"{model_name} SVD",
            linewidth=1.5,
            markersize=6,
            alpha=0.7,
        )

    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Context length", fontsize=10)
    ax.set_ylabel("Mean |r| with position", fontsize=10)
    ax.set_title("(c) Average correlation strength", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(context_lengths)
    ax.set_xticklabels(context_lengths)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (d): Per-bucket R² at L=1024
    ax = axes[1, 1]
    for model_name, color in [("R0", "#1f77b4"), ("R2", "#ff7f0e")]:
        bucket_r2 = results[model_name][1024]["per_bucket_r2"]
        bucket_labels = results[model_name][1024]["bucket_labels"]
        x = np.arange(len(bucket_r2))
        ax.bar(
            x + (0.2 if model_name == "R0" else -0.2),
            bucket_r2,
            0.35,
            label=model_name,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Position bucket", fontsize=10)
    ax.set_ylabel("R² in bucket", fontsize=10)
    ax.set_title("(d) Per-position R² at L=1024", fontsize=11)
    if bucket_labels:
        ax.set_xticks(np.arange(len(bucket_labels)))
        ax.set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "extrapolation_analysis.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(save_dir, "extrapolation_analysis.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print(f"Figure saved to {save_dir}/extrapolation_analysis.pdf")


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
    parser.add_argument(
        "--data_dir",
        type=str,
        default="nanoGPT/data/openwebtext",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/extrapolation_analysis",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--context_lengths",
        type=str,
        default="128,256,512,1024",
        help="Comma-separated list of context lengths to test",
    )
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    _, val_data = load_owt_data(args.data_dir)

    all_results = {}

    for model_name, ckpt_path in [
        ("R0", args.r0_checkpoint),
        ("R2", args.r2_checkpoint),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Analyzing {model_name}")
        print(f"{'=' * 60}")

        model, config = load_model(ckpt_path, args.device)
        B = get_block2_write_map(model)

        all_results[model_name] = {}

        for L in context_lengths:
            print(f"\nContext length: {L}")

            results = compute_extrapolation_metrics(
                model, val_data, B, L, args.n_batches, args.batch_size, args.device
            )

            all_results[model_name][L] = results

            print(f"  Linear probe R²: {results['linear_probe_r2']:.4f}")
            print(f"  Neuron |r|>0.95: {results['neuron_counts']['|r|>0.95']}")
            print(f"  SVD |r|>0.95: {results['svd_counts']['|r|>0.95']}")
            print(f"  Mean |r| neuron: {results['mean_abs_corr_neuron']:.4f}")

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(args.save_dir, "extrapolation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    plot_extrapolation_results(all_results, args.save_dir)

    # Print summary table
    print(f"\n{'=' * 60}")
    print("EXTRAPOLATION SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Model':<8} {'L':<6} {'R²':<8} {'|r|>0.95 (N)':<14} {'|r|>0.95 (SVD)':<14}"
    )
    print("-" * 50)
    for model_name in ["R0", "R2"]:
        for L in context_lengths:
            r = all_results[model_name][L]
            print(
                f"{model_name:<8} {L:<6} {r['linear_probe_r2']:<8.4f} {r['neuron_counts']['|r|>0.95']:<14} {r['svd_counts']['|r|>0.95']:<14}"
            )

    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()
