"""
SVD-Basis Position Correlation Analysis

Computes position correlations in the SVD basis of B = W_O @ W_V (the write map).
This distinguishes between:
- Intrinsic dimensionality (low-rank in rotated/SVD basis)
- Axis-alignment (high |r| in standard neuron basis)

For R0 (full training) and R2 (attention-only training) with 12 heads.
Focuses on post_attn as the enabling mechanism of positional encoding.
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


def get_block2_write_map(model: TwoLayerMechanismModel):
    """
    Extract B = W_O @ W_V from Block 2 attention.

    W_V: [d_model, d_model] (value projection)
    W_O: [d_model, d_model] (output projection)
    B = W_O @ W_V: [d_model, d_model]
    """
    attn = model.block2.attn

    # Get weight matrices
    # c_attn projects to [q, k, v] concatenated
    c_attn_weight = attn.c_attn.weight  # [3*d_model, d_model]
    d_model = c_attn_weight.shape[1]

    # Extract W_V (last third of c_attn)
    W_V = c_attn_weight[2 * d_model :, :]  # [d_model, d_model]

    # W_O is the output projection
    W_O = attn.c_proj.weight  # [d_model, d_model]

    # B = W_O @ W_V
    B = W_O @ W_V  # [d_model, d_model]

    return B.detach().cpu().numpy()


def compute_svd_basis_correlations(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    B: np.ndarray,
    n_batches: int = 50,
    batch_size: int = 32,
    block_size: int = 128,
    device: str = "cuda",
):
    """
    Compute per-component correlation with position in both:
    1. Standard neuron basis (raw activations)
    2. SVD basis of B = W_O @ W_V

    Focuses on post_attn (after attention residual) and attention output (before residual).

    Returns:
        neuron_corrs: dict of layer -> [d_model] correlations in neuron basis
        svd_corrs: dict of layer -> [d_model] correlations in SVD basis
        U: left singular vectors of B
        S: singular values of B
    """
    model.eval()

    # Compute SVD of B
    U, S, Vt = np.linalg.svd(B, full_matrices=True)

    # Collect activations
    attn_outputs = []  # o_i: attention output before residual
    post_attn_acts = []  # post_attn: after residual (x + o_i)
    positions_list = []

    with torch.no_grad():
        for _ in range(n_batches):
            tokens = get_batch(data, batch_size, block_size, device)
            B_size, T = tokens.shape

            # Forward pass with activation capture
            _ = model(tokens, capture_taps=True)

            block2 = model.block2

            # Get attention output (before residual) - need to compute from stored values
            # post_attn = x + attn_output, so attn_output = post_attn - x
            # x is the input to block2, which is block1.last_block_out after LN1

            # Actually, we need the raw attention output. Let me get it differently.
            # The model stores last_attn_out which is the attention output before projection
            # We need to use the stored values properly

            # From the model: post_attn = x + self.attn(self.ln1(x))
            # So attn_output = post_attn - x (where x is the block input)

            # Block 2 input is block1_out
            block2_input = model.block1.last_block_out  # [B, T, d_model]
            block2_post_attn = block2.last_post_attn  # [B, T, d_model]

            # Attention output (before adding to residual)
            attn_out = block2_post_attn - block2_input  # o_i = post_attn - x

            attn_outputs.append(attn_out.cpu())
            post_attn_acts.append(block2_post_attn.cpu())

            # Positions
            positions = torch.arange(T).unsqueeze(0).expand(B_size, -1)
            positions_list.append(positions)

    # Concatenate
    attn_outputs = torch.cat(attn_outputs, dim=0).numpy()  # [N, T, d_model]
    post_attn_acts = torch.cat(post_attn_acts, dim=0).numpy()
    positions = torch.cat(positions_list, dim=0).numpy()  # [N, T]

    # Flatten for correlation computation
    N, T, d_model = attn_outputs.shape
    positions_flat = positions.reshape(-1)  # [N*T]

    results = {}

    for name, acts in [("attn_output", attn_outputs), ("post_attn", post_attn_acts)]:
        acts_flat = acts.reshape(-1, d_model)  # [N*T, d_model]

        # 1. Neuron basis correlations
        neuron_corrs = np.zeros(d_model)
        for c in range(d_model):
            r, _ = stats.pearsonr(positions_flat, acts_flat[:, c])
            neuron_corrs[c] = r

        # 2. SVD basis correlations (project onto U)
        acts_svd = acts_flat @ U  # [N*T, d_model] - project onto left singular vectors
        svd_corrs = np.zeros(d_model)
        for c in range(d_model):
            r, _ = stats.pearsonr(positions_flat, acts_svd[:, c])
            svd_corrs[c] = r

        results[name] = {
            "neuron_corrs": neuron_corrs,
            "svd_corrs": svd_corrs,
        }

    return results, U, S


def count_high_corr(corrs: np.ndarray, thresholds: list = [0.3, 0.5, 0.7, 0.9, 0.95]):
    """Count channels exceeding various correlation thresholds."""
    abs_corrs = np.abs(corrs)
    return {f"|r|>{t}": int(np.sum(abs_corrs > t)) for t in thresholds}


def analyze_and_plot(
    results_r0: dict,
    results_r2: dict,
    S_r0: np.ndarray,
    S_r2: np.ndarray,
    save_dir: str,
):
    """Create analysis plots comparing neuron vs SVD basis correlations."""
    os.makedirs(save_dir, exist_ok=True)

    # Focus on post_attn as the key mechanism
    layer = "post_attn"

    # Collect statistics
    stats_table = []

    for model_name, results, S in [("R0", results_r0, S_r0), ("R2", results_r2, S_r2)]:
        neuron_corrs = results[layer]["neuron_corrs"]
        svd_corrs = results[layer]["svd_corrs"]

        neuron_counts = count_high_corr(neuron_corrs)
        svd_counts = count_high_corr(svd_corrs)

        stats_table.append(
            {
                "model": model_name,
                "layer": layer,
                "neuron": neuron_counts,
                "svd": svd_counts,
            }
        )

        print(f"\n{model_name} - {layer}:")
        print(f"  Neuron basis: {neuron_counts}")
        print(f"  SVD basis:    {svd_counts}")

    # Create ICML-compatible figure (single column width)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel (a): Sorted |r| curves for R0 and R2 in both bases
    ax = axes[0]

    for model_name, results, color, marker in [
        ("R0", results_r0, "#1f77b4", "o"),
        ("R2", results_r2, "#ff7f0e", "s"),
    ]:
        neuron_corrs = results[layer]["neuron_corrs"]
        svd_corrs = results[layer]["svd_corrs"]

        # Sort by absolute value (descending)
        neuron_sorted = np.sort(np.abs(neuron_corrs))[::-1]
        svd_sorted = np.sort(np.abs(svd_corrs))[::-1]

        # Plot first 100 components for clarity
        n_show = 100
        ax.plot(
            range(n_show),
            neuron_sorted[:n_show],
            color=color,
            linestyle="-",
            linewidth=1.5,
            label=f"{model_name} neuron",
        )
        ax.plot(
            range(n_show),
            svd_sorted[:n_show],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{model_name} SVD",
        )

    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.7, linewidth=1)
    ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.7, linewidth=1)
    ax.text(
        n_show - 2,
        0.96,
        r"$|r|=0.95$",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    ax.text(
        n_show - 2,
        0.31,
        r"$|r|=0.3$",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    ax.set_xlabel("Component rank", fontsize=9)
    ax.set_ylabel(r"$|r|$ with position", fontsize=9)
    ax.set_title("(a) Sorted correlations", fontsize=10)
    ax.set_xlim(0, n_show)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)

    # Panel (b): Bar chart comparing counts at thresholds
    ax = axes[1]

    thresholds = [0.3, 0.5, 0.7, 0.9, 0.95]
    x = np.arange(len(thresholds))
    width = 0.2

    for i, (model_name, results) in enumerate([("R0", results_r0), ("R2", results_r2)]):
        neuron_corrs = results[layer]["neuron_corrs"]
        svd_corrs = results[layer]["svd_corrs"]

        neuron_counts = [np.sum(np.abs(neuron_corrs) > t) for t in thresholds]
        svd_counts = [np.sum(np.abs(svd_corrs) > t) for t in thresholds]

        color = "#1f77b4" if model_name == "R0" else "#ff7f0e"
        offset = (i - 0.5) * 2 * width

        ax.bar(
            x + offset - width / 2,
            neuron_counts,
            width,
            label=f"{model_name} neuron",
            color=color,
            alpha=0.8,
        )
        ax.bar(
            x + offset + width / 2,
            svd_counts,
            width,
            label=f"{model_name} SVD",
            color=color,
            alpha=0.4,
            hatch="//",
        )

    ax.set_xlabel(r"Threshold $\tau$", fontsize=9)
    ax.set_ylabel(r"Channels with $|r|>\tau$", fontsize=9)
    ax.set_title("(b) Count by threshold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in thresholds], fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "svd_vs_neuron_correlation.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(save_dir, "svd_vs_neuron_correlation.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print(f"\nFigure saved to {save_dir}/svd_vs_neuron_correlation.pdf")

    # Save detailed statistics as JSON
    detailed_stats = {
        "R0": {
            "post_attn": {
                "neuron_counts": count_high_corr(
                    results_r0["post_attn"]["neuron_corrs"]
                ),
                "svd_counts": count_high_corr(results_r0["post_attn"]["svd_corrs"]),
                "neuron_max_abs": float(
                    np.max(np.abs(results_r0["post_attn"]["neuron_corrs"]))
                ),
                "svd_max_abs": float(
                    np.max(np.abs(results_r0["post_attn"]["svd_corrs"]))
                ),
            },
            "attn_output": {
                "neuron_counts": count_high_corr(
                    results_r0["attn_output"]["neuron_corrs"]
                ),
                "svd_counts": count_high_corr(results_r0["attn_output"]["svd_corrs"]),
                "neuron_max_abs": float(
                    np.max(np.abs(results_r0["attn_output"]["neuron_corrs"]))
                ),
                "svd_max_abs": float(
                    np.max(np.abs(results_r0["attn_output"]["svd_corrs"]))
                ),
            },
        },
        "R2": {
            "post_attn": {
                "neuron_counts": count_high_corr(
                    results_r2["post_attn"]["neuron_corrs"]
                ),
                "svd_counts": count_high_corr(results_r2["post_attn"]["svd_corrs"]),
                "neuron_max_abs": float(
                    np.max(np.abs(results_r2["post_attn"]["neuron_corrs"]))
                ),
                "svd_max_abs": float(
                    np.max(np.abs(results_r2["post_attn"]["svd_corrs"]))
                ),
            },
            "attn_output": {
                "neuron_counts": count_high_corr(
                    results_r2["attn_output"]["neuron_corrs"]
                ),
                "svd_counts": count_high_corr(results_r2["attn_output"]["svd_corrs"]),
                "neuron_max_abs": float(
                    np.max(np.abs(results_r2["attn_output"]["neuron_corrs"]))
                ),
                "svd_max_abs": float(
                    np.max(np.abs(results_r2["attn_output"]["svd_corrs"]))
                ),
            },
        },
    }

    with open(os.path.join(save_dir, "svd_basis_correlation_stats.json"), "w") as f:
        json.dump(detailed_stats, f, indent=2)

    # Save raw correlations
    np.savez(
        os.path.join(save_dir, "svd_basis_correlations.npz"),
        r0_post_attn_neuron=results_r0["post_attn"]["neuron_corrs"],
        r0_post_attn_svd=results_r0["post_attn"]["svd_corrs"],
        r0_attn_output_neuron=results_r0["attn_output"]["neuron_corrs"],
        r0_attn_output_svd=results_r0["attn_output"]["svd_corrs"],
        r2_post_attn_neuron=results_r2["post_attn"]["neuron_corrs"],
        r2_post_attn_svd=results_r2["post_attn"]["svd_corrs"],
        r2_attn_output_neuron=results_r2["attn_output"]["neuron_corrs"],
        r2_attn_output_svd=results_r2["attn_output"]["svd_corrs"],
        S_r0=S_r0,
        S_r2=S_r2,
    )

    return detailed_stats


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
        default="results/svd_basis_correlation",
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

    # Analyze R0
    print("\n" + "=" * 60)
    print("Analyzing R0 (Full Training)")
    print("=" * 60)
    model_r0, config_r0 = load_model(args.r0_checkpoint, args.device)
    print(f"Model config: n_embd={config_r0.n_embd}, n_head={config_r0.n_head}")

    B_r0 = get_block2_write_map(model_r0)
    print(f"B matrix shape: {B_r0.shape}")

    results_r0, U_r0, S_r0 = compute_svd_basis_correlations(
        model_r0,
        val_data,
        B_r0,
        args.n_batches,
        args.batch_size,
        config_r0.block_size,
        args.device,
    )
    del model_r0
    torch.cuda.empty_cache()

    # Analyze R2
    print("\n" + "=" * 60)
    print("Analyzing R2 (Attn2-only Training)")
    print("=" * 60)
    model_r2, config_r2 = load_model(args.r2_checkpoint, args.device)

    B_r2 = get_block2_write_map(model_r2)

    results_r2, U_r2, S_r2 = compute_svd_basis_correlations(
        model_r2,
        val_data,
        B_r2,
        args.n_batches,
        args.batch_size,
        config_r2.block_size,
        args.device,
    )
    del model_r2
    torch.cuda.empty_cache()

    # Analyze and plot
    print("\n" + "=" * 60)
    print("Comparing Results")
    print("=" * 60)

    detailed_stats = analyze_and_plot(results_r0, results_r2, S_r0, S_r2, args.save_dir)

    # Print summary table for paper
    print("\n" + "=" * 60)
    print("SUMMARY TABLE FOR PAPER (post_attn)")
    print("=" * 60)
    print(
        f"{'Model':<8} {'Basis':<10} {'|r|>0.3':<10} {'|r|>0.5':<10} {'|r|>0.95':<10}"
    )
    print("-" * 48)
    for model in ["R0", "R2"]:
        for basis in ["neuron", "svd"]:
            counts = detailed_stats[model]["post_attn"][f"{basis}_counts"]
            print(
                f"{model:<8} {basis:<10} {counts['|r|>0.3']:<10} {counts['|r|>0.5']:<10} {counts['|r|>0.95']:<10}"
            )

    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()
