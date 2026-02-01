"""
BOS Causal Intervention Plot for ICML 2026 Paper

Generates publication-quality figure showing the effect of masking BOS visibility
in Block 2 attention. The intervention sets attention weights to position 0 to zero
and renormalizes.

Usage:
    python analysis_scripts/generate_bos_intervention_plot.py
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
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colors (colorblind-friendly)
COLOR_BASELINE = "#0072B2"  # Blue
COLOR_MASKED = "#D55E00"  # Vermillion
COLOR_R0 = "#0072B2"
COLOR_R2 = "#D55E00"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOS_TOKEN_ID = 50256


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
    """Load OpenWebText validation data."""
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Get a batch of sequences with BOS token at position 0."""
    tokens_needed = block_size - 1
    ix = torch.randint(len(data) - tokens_needed, (batch_size,))
    sequences = []
    for i in ix:
        after_bos = data[i : i + tokens_needed].astype(np.int64)
        seq = np.concatenate([[BOS_TOKEN_ID], after_bos])
        sequences.append(torch.from_numpy(seq))
    x = torch.stack(sequences)
    return x.to(device)


def forward_with_bos_mask(
    model: TwoLayerMechanismModel,
    tokens: torch.Tensor,
    mask_bos: bool = False,
    return_attention: bool = False,
):
    """
    Forward pass with optional BOS masking in Block 2 attention.

    Args:
        model: The model
        tokens: Input tokens [B, T]
        mask_bos: If True, set attention to position 0 to zero and renormalize
        return_attention: If True, return Block 2 attention weights

    Returns:
        predictions: Position predictions [B, T]
        (optional) attn_weights: Block 2 attention weights [B, H, T, T]
    """
    B, T = tokens.shape
    d_model = model.config.n_embd
    n_head = model.config.n_head
    head_dim = d_model // n_head

    with torch.no_grad():
        # Embedding
        e = model.wte(tokens)

        # Block 1 forward (unchanged)
        ln1_out = model.block1.ln_1(e)
        attn1 = model.block1.attn

        qkv1 = attn1.c_attn(ln1_out)
        q1, k1, v1 = qkv1.split(d_model, dim=2)
        q1 = q1.view(B, T, n_head, head_dim).transpose(1, 2)
        k1 = k1.view(B, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B, T, n_head, head_dim).transpose(1, 2)

        att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        causal_mask = torch.triu(
            torch.ones(T, T, device=tokens.device), diagonal=1
        ).bool()
        att1 = att1.masked_fill(causal_mask, float("-inf"))
        att1 = F.softmax(att1, dim=-1)
        y1 = (att1 @ v1).transpose(1, 2).contiguous().view(B, T, d_model)
        attn_out1 = attn1.c_proj(y1)

        r1_attn = e + attn_out1
        ln2_out_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_out_b1)
        r1 = r1_attn + mlp_out1

        # Block 2 forward with BOS masking
        ln1_out_b2 = model.block2.ln_1(r1)
        attn2 = model.block2.attn

        qkv2 = attn2.c_attn(ln1_out_b2)
        q2, k2, v2 = qkv2.split(d_model, dim=2)
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        att2 = att2.masked_fill(causal_mask, float("-inf"))

        if mask_bos:
            # Mask attention to position 0 by setting it to -inf before softmax
            att2[:, :, :, 0] = float("-inf")
            # Handle position 0 query (can only attend to itself)
            # Set it to attend uniformly to itself (or keep it as is)
            att2[:, :, 0, 0] = 0.0  # Position 0 can only attend to itself

        att2 = F.softmax(att2, dim=-1)

        y2 = (att2 @ v2).transpose(1, 2).contiguous().view(B, T, d_model)
        attn_out2 = attn2.c_proj(y2)

        r2_attn = r1 + attn_out2
        ln2_out_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_out_b2)
        r2 = r2_attn + mlp_out2

        # Final layer norm and prediction head
        final = model.ln_f(r2)
        pred = model.pos_head(final).squeeze(-1)  # [B, T]

    if return_attention:
        return pred, att2
    return pred


def compute_r2(predictions, positions):
    """Compute R² between predictions and positions."""
    pred_flat = predictions.cpu().flatten().numpy()
    pos_flat = positions.cpu().flatten().numpy()
    r, _ = stats.pearsonr(pos_flat, pred_flat)
    return r**2


def run_bos_intervention_experiment(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    model_name: str,
    n_batches: int = 100,
    batch_size: int = 32,
):
    """Run BOS masking experiment."""
    print(f"\n{'=' * 60}")
    print(f"Running BOS intervention experiment for {model_name}")
    print(f"{'=' * 60}")

    block_size = model.config.block_size

    baseline_preds = []
    masked_preds = []
    all_positions = []
    sample_attn_baseline = None
    sample_attn_masked = None

    for i in tqdm(range(n_batches), desc=f"{model_name}"):
        tokens = get_batch(data, batch_size, block_size, DEVICE)
        positions = (
            torch.arange(block_size, device=DEVICE)
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Baseline (no intervention)
        pred_baseline, attn_baseline = forward_with_bos_mask(
            model, tokens, mask_bos=False, return_attention=True
        )

        # Masked BOS
        pred_masked, attn_masked = forward_with_bos_mask(
            model, tokens, mask_bos=True, return_attention=True
        )

        baseline_preds.append(pred_baseline.cpu())
        masked_preds.append(pred_masked.cpu())
        all_positions.append(positions.cpu())

        # Store sample attention for visualization (first batch only)
        if i == 0:
            sample_attn_baseline = attn_baseline[0].cpu()  # [H, T, T]
            sample_attn_masked = attn_masked[0].cpu()

    # Concatenate
    baseline_preds = torch.cat(baseline_preds, dim=0)
    masked_preds = torch.cat(masked_preds, dim=0)
    all_positions = torch.cat(all_positions, dim=0)

    # Compute R²
    baseline_r2 = compute_r2(baseline_preds, all_positions)
    masked_r2 = compute_r2(masked_preds, all_positions)

    print(f"Baseline R²: {baseline_r2:.4f}")
    print(f"Masked BOS R²: {masked_r2:.4f}")
    print(
        f"R² Drop: {baseline_r2 - masked_r2:.4f} ({100 * (baseline_r2 - masked_r2) / baseline_r2:.1f}%)"
    )

    # Compute per-position MAE
    baseline_mae = torch.abs(baseline_preds - all_positions).mean(dim=0).numpy()
    masked_mae = torch.abs(masked_preds - all_positions).mean(dim=0).numpy()

    return {
        "baseline_r2": float(baseline_r2),
        "masked_r2": float(masked_r2),
        "r2_drop": float(baseline_r2 - masked_r2),
        "r2_drop_pct": float(100 * (baseline_r2 - masked_r2) / baseline_r2),
        "baseline_mae_per_pos": baseline_mae.tolist(),
        "masked_mae_per_pos": masked_mae.tolist(),
        "sample_attn_baseline": sample_attn_baseline.numpy()
        if sample_attn_baseline is not None
        else None,
        "sample_attn_masked": sample_attn_masked.numpy()
        if sample_attn_masked is not None
        else None,
    }


def create_bos_intervention_plot(results: dict, save_path: str):
    """Create the main BOS intervention figure."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Panel (a): R² comparison bar chart
    ax = axes[0]
    x = [0, 1]
    values = [results["baseline_r2"], results["masked_r2"]]
    colors = [COLOR_BASELINE, COLOR_MASKED]
    labels = ["Baseline", "BOS Masked"]

    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Position $R^2$")
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add drop annotation
    drop_pct = results["r2_drop_pct"]
    ax.annotate(
        f"$\\Delta R^2$ = {results['r2_drop']:.3f}\n({drop_pct:.1f}% drop)",
        xy=(0.5, results["masked_r2"]),
        xytext=(1.3, 0.6),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        ha="center",
    )

    ax.set_title("(a) Position Decoding Accuracy")
    ax.grid(True, alpha=0.2, axis="y", linewidth=0.5)

    # Panel (b): Per-position MAE comparison
    ax = axes[1]
    positions = np.arange(len(results["baseline_mae_per_pos"]))

    ax.plot(
        positions,
        results["baseline_mae_per_pos"],
        "-",
        color=COLOR_BASELINE,
        label="Baseline",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.plot(
        positions,
        results["masked_mae_per_pos"],
        "-",
        color=COLOR_MASKED,
        label="BOS Masked",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("(b) Error by Position")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9)
    ax.set_xlim(0, len(positions))
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved figure to {save_path}")


def create_attention_comparison_plot(results: dict, save_path: str):
    """Create attention heatmap comparison (optional supplementary figure)."""
    if results["sample_attn_baseline"] is None:
        return

    attn_baseline = results["sample_attn_baseline"]  # [H, T, T]
    attn_masked = results["sample_attn_masked"]

    n_heads = attn_baseline.shape[0]

    # Find BOS heads (highest attention to position 0)
    bos_scores = attn_baseline[:, :, 0].mean(axis=1)  # Average over queries
    top_bos_heads = np.argsort(bos_scores)[-3:][::-1]  # Top 3 BOS heads

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4))

    for col, head_idx in enumerate(top_bos_heads):
        # Baseline
        ax = axes[0, col]
        im = ax.imshow(
            attn_baseline[head_idx], cmap="viridis", aspect="auto", vmin=0, vmax=1
        )
        ax.set_title(f"Head {head_idx} (BOS={bos_scores[head_idx]:.2f})", fontsize=9)
        if col == 0:
            ax.set_ylabel("Baseline\nQuery pos", fontsize=8)
        ax.set_xticks([])

        # Masked
        ax = axes[1, col]
        im = ax.imshow(
            attn_masked[head_idx], cmap="viridis", aspect="auto", vmin=0, vmax=1
        )
        if col == 0:
            ax.set_ylabel("BOS Masked\nQuery pos", fontsize=8)
        ax.set_xlabel("Key pos", fontsize=8)

    # Colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention Weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("BOS Intervention: Attention Pattern Changes", fontsize=10, y=1.02)

    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved attention comparison to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r0_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument("--save_dir", type=str, default="results/bos_intervention")
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    val_data = load_owt_data(args.data_dir)

    # Load R0 model (the one with BOS heads)
    print("\nLoading R0 model...")
    model_r0, config_r0 = load_model(args.r0_checkpoint, DEVICE)

    # Run experiment
    results = run_bos_intervention_experiment(
        model_r0, val_data, "R0", args.n_batches, args.batch_size
    )

    # Save results (without large arrays)
    results_to_save = {
        k: v for k, v in results.items() if not k.startswith("sample_attn")
    }
    with open(os.path.join(args.save_dir, "bos_intervention_results.json"), "w") as f:
        json.dump(results_to_save, f, indent=2)

    # Create main figure
    create_bos_intervention_plot(
        results, os.path.join(args.save_dir, "bos_intervention.pdf")
    )

    # Create attention comparison figure (supplementary)
    create_attention_comparison_plot(
        results, os.path.join(args.save_dir, "bos_attention_comparison.pdf")
    )

    # Copy to paper directory
    import shutil

    paper_dir = "overleaf/nopos_icml_2026/plots"
    shutil.copy(
        os.path.join(args.save_dir, "bos_intervention.pdf"),
        os.path.join(paper_dir, "bos_intervention.pdf"),
    )
    print(f"\nCopied figure to {paper_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"Baseline R²: {results['baseline_r2']:.4f}")
    print(f"BOS Masked R²: {results['masked_r2']:.4f}")
    print(f"R² Drop: {results['r2_drop']:.4f} ({results['r2_drop_pct']:.1f}%)")
    print(f"N samples: {args.n_batches * args.batch_size}")


if __name__ == "__main__":
    main()
