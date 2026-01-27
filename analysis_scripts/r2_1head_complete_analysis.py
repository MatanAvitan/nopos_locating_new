"""
Complete Analysis for R2-1-Head Geometric Clock Mechanism
=========================================================

This script generates all plots and data for the ICML 2026 paper:
1. Write bottleneck curves (retention/ablation interventions)
2. Attention maps showing the geometric clock mechanism
3. Directional rotation visualization
4. Extrapolation analysis to longer sequences

Usage:
    python analysis_scripts/r2_1head_complete_analysis.py
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
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))
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

# Colors
COLOR_PRIMARY = "#0072B2"  # Blue
COLOR_SECONDARY = "#D55E00"  # Vermillion
COLOR_POS0 = "#009E73"  # Green
COLOR_OTHERS = "#CC79A7"  # Pink

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_TOKEN_ID = 50256


def load_model(checkpoint_path: str, device: str = "cuda", post_attn: bool = True):
    """Load a trained R2-1head model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
    elif "model_args" in checkpoint:
        config_dict = checkpoint["model_args"]
    else:
        raise ValueError("Cannot find config in checkpoint")

    config = TwoLayerMechanismConfig(
        block_size=config_dict.get("block_size", 128),
        vocab_size=config_dict.get("vocab_size", 50304),
        n_embd=config_dict.get("n_embd", 768),
        n_head=config_dict.get("n_head", 1),
        dropout=0.0,
        norm_type=config_dict.get("norm_type", "layernorm"),
        bias=config_dict.get("bias", True),
        use_regression=True,
    )

    model = TwoLayerMechanismModel(config)
    if post_attn:
        model.set_post_attn_head(True)

    state_dict = checkpoint["model"]
    unwrapped = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped)
    model.to(device).eval()

    return model, config


def load_owt_data(data_dir: str = None):
    """Load OpenWebText validation data."""
    if data_dir is None:
        data_dir = ROOT_DIR / "nanoGPT/data/openwebtext"
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


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


def corrcoef(x, y):
    """Compute Pearson correlation coefficient."""
    x, y = x.flatten().float(), y.flatten().float()
    return ((x - x.mean()) @ (y - y.mean()) / (x.std() * y.std() * len(x))).item()


def r2_score(preds, targets):
    """Compute R^2 score."""
    preds, targets = preds.flatten().float(), targets.flatten().float()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


# =============================================================================
# Write Bottleneck Analysis
# =============================================================================


def get_block2_write_map_svd(model):
    """Compute SVD of B = W_O @ W_V from Block 2 attention."""
    attn = model.block2.attn
    c_attn_weight = attn.c_attn.weight
    d_model = model.config.n_embd

    W_V = c_attn_weight[2 * d_model :, :]  # [d_model, d_model]
    W_O = attn.c_proj.weight  # [d_model, d_model]
    B = W_O @ W_V

    U, S, Vt = torch.linalg.svd(B, full_matrices=True)
    return U, S, Vt, B


def forward_with_write_intervention(
    model, tokens, U, rank, intervention_type="retention"
):
    """Forward pass with intervention on Block 2 attention output."""
    B, T = tokens.shape
    d_model = model.config.n_embd
    n_head = model.config.n_head
    head_dim = d_model // n_head

    with torch.no_grad():
        e = model.wte(tokens)

        # Block 1 forward
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

        # Block 2 forward with intervention
        ln1_out_b2 = model.block2.ln_1(r1)
        attn2 = model.block2.attn
        qkv2 = attn2.c_attn(ln1_out_b2)
        q2, k2, v2 = qkv2.split(d_model, dim=2)
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        att2 = att2.masked_fill(causal_mask, float("-inf"))
        att2 = F.softmax(att2, dim=-1)
        y2 = (att2 @ v2).transpose(1, 2).contiguous().view(B, T, d_model)
        attn_out2 = attn2.c_proj(y2)

        # Apply write subspace intervention
        U_r = U[:, :rank]
        if intervention_type == "retention":
            attn_out2_int = attn_out2 @ U_r @ U_r.T
        elif intervention_type == "ablation":
            attn_out2_int = attn_out2 - attn_out2 @ U_r @ U_r.T
        else:
            attn_out2_int = attn_out2

        # For post-attn head, skip MLP2
        r2 = r1 + attn_out2_int
        final = model.ln_f(r2)
        pred = model.pos_head(final).squeeze(-1)

    return pred


def compute_r2_at_rank(
    model, data, U, rank, intervention_type, n_batches=50, batch_size=32
):
    """Compute R^2 for a given rank intervention."""
    block_size = model.config.block_size
    all_preds, all_positions = [], []

    for _ in range(n_batches):
        tokens = get_batch(data, batch_size, block_size, DEVICE)
        preds = forward_with_write_intervention(
            model, tokens, U, rank, intervention_type
        )
        positions = (
            torch.arange(block_size, device=DEVICE)
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        all_preds.append(preds.cpu())
        all_positions.append(positions.cpu())

    all_preds = torch.cat(all_preds, dim=0).flatten().numpy()
    all_positions = torch.cat(all_positions, dim=0).flatten().numpy()

    r, _ = stats.pearsonr(all_positions, all_preds)
    return r**2


def run_write_bottleneck(model, data, ranks, n_batches=50, batch_size=32):
    """Run full write bottleneck experiment."""
    print("\n" + "=" * 60)
    print("Running Write Bottleneck Experiment")
    print("=" * 60)

    U, S, Vt, B = get_block2_write_map_svd(model)

    # Baseline R^2
    baseline_r2 = compute_r2_at_rank(
        model, data, U, 768, "retention", n_batches, batch_size
    )
    print(f"Baseline R²: {baseline_r2:.4f}")

    retention_r2s = []
    ablation_r2s = []

    for rank in tqdm(ranks, desc="Computing R² at each rank"):
        ret_r2 = compute_r2_at_rank(
            model, data, U, rank, "retention", n_batches, batch_size
        )
        abl_r2 = compute_r2_at_rank(
            model, data, U, rank, "ablation", n_batches, batch_size
        )
        retention_r2s.append(ret_r2)
        ablation_r2s.append(abl_r2)

    # Find r_95
    threshold_95 = 0.95 * baseline_r2
    r_95 = None
    for i, r2 in enumerate(retention_r2s):
        if r2 >= threshold_95:
            r_95 = ranks[i]
            break

    print(f"r_95 (95% of baseline): {r_95}")

    return {
        "baseline_r2": float(baseline_r2),
        "ranks": ranks,
        "retention_r2s": [float(x) for x in retention_r2s],
        "ablation_r2s": [float(x) for x in ablation_r2s],
        "r_95": r_95,
        "singular_values": S.detach().cpu().numpy().tolist(),
    }


# =============================================================================
# Attention Map Analysis
# =============================================================================


def generate_attention_map(model, data, batch_size=64):
    """Generate attention maps for Block 1 and Block 2."""
    tokens = get_batch(data, batch_size, model.config.block_size, DEVICE)

    with torch.no_grad():
        model(tokens, capture_taps=True)
        attn1, attn2 = model.get_attention_weights()

    return attn1.mean(dim=0), attn2.mean(dim=0)


def plot_attention_maps(attn1, attn2, save_path):
    """Plot attention maps for the R2-1-head model."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    def strip_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Block 1 attention (frozen, random)
    im1 = axes[0].imshow(attn1[0].cpu().numpy(), cmap="cividis", aspect="auto")
    axes[0].set_title("Block 1 Attention (Frozen)")
    strip_axes(axes[0])
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Block 2 attention (trained)
    im2 = axes[1].imshow(attn2[0].cpu().numpy(), cmap="cividis", aspect="auto")
    axes[1].set_title("Block 2 Attention (Trained)")
    strip_axes(axes[1])
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved attention maps to {save_path}")


# =============================================================================
# Geometric Clock Visualization
# =============================================================================


def analyze_geometric_clock(model, data, batch_size=64):
    """Analyze the geometric clock mechanism."""
    tokens = get_batch(data, batch_size, model.config.block_size, DEVICE)
    D = model.config.n_embd
    T = model.config.block_size

    with torch.no_grad():
        model(tokens, capture_taps=True)
        taps = model.get_all_taps()

        # Get weights
        W_V = model.block2.attn.c_attn.weight[2 * D :, :].detach()
        W_O = model.block2.attn.c_proj.weight.detach()
        b_O = model.block2.attn.c_proj.bias.detach()
        w_head = model.pos_head.weight.detach().squeeze()

        # Block 2 attention input
        ln2_1 = taps["block2_ln1"]
        attn_out = taps["block2_attn"]

        # Compute projected values
        v2 = ln2_1 @ W_V.T
        Wo_v = v2 @ W_O.T

        # Position 0 and others directions
        pos0_Wo_v_mean = Wo_v[:, 0, :].mean(dim=0)
        others_Wo_v_mean = Wo_v[:, 1:, :].mean(dim=(0, 1))

        pos0_dir = F.normalize(pos0_Wo_v_mean.unsqueeze(0), dim=-1).squeeze()
        others_dir = F.normalize(others_Wo_v_mean.unsqueeze(0), dim=-1).squeeze()

        # Projections
        attn_out_no_bias = attn_out - b_O
        proj_pos0 = attn_out_no_bias @ pos0_dir
        proj_others = attn_out_no_bias @ others_dir

        positions = (
            torch.arange(T, device=DEVICE).float().unsqueeze(0).expand(batch_size, -1)
        )

        # Compute metrics
        cos_pos0_others = F.cosine_similarity(
            pos0_dir.unsqueeze(0), others_dir.unsqueeze(0)
        ).item()
        corr_pos0 = corrcoef(proj_pos0, positions)
        corr_others = corrcoef(proj_others, positions)
        cos_w_pos0 = F.cosine_similarity(
            w_head.unsqueeze(0), pos0_dir.unsqueeze(0)
        ).item()
        cos_w_others = F.cosine_similarity(
            w_head.unsqueeze(0), others_dir.unsqueeze(0)
        ).item()

    return {
        "proj_pos0": proj_pos0.mean(dim=0).cpu().numpy(),
        "proj_others": proj_others.mean(dim=0).cpu().numpy(),
        "cos_pos0_others": cos_pos0_others,
        "corr_pos0": corr_pos0,
        "corr_others": corr_others,
        "cos_w_pos0": cos_w_pos0,
        "cos_w_others": cos_w_others,
        "pos0_norm": pos0_Wo_v_mean.norm().item(),
        "others_norm": others_Wo_v_mean.norm().item(),
    }


def plot_geometric_clock(clock_data, save_path):
    """Plot the geometric clock mechanism."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    T = len(clock_data["proj_pos0"])
    positions = np.arange(T)

    # Projections
    axes[0].plot(
        positions,
        clock_data["proj_pos0"],
        label="Proj. pos-0 dir",
        color=COLOR_POS0,
        linewidth=2,
    )
    axes[0].plot(
        positions,
        clock_data["proj_others"],
        label="Proj. others dir",
        color=COLOR_OTHERS,
        linewidth=2,
    )
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Projection value")
    axes[0].set_title("Attention Output Projection")
    axes[0].legend(loc="center right")
    axes[0].grid(True, alpha=0.3)

    # Add correlation annotations
    axes[0].annotate(
        f"r={clock_data['corr_pos0']:.2f}",
        xy=(T * 0.1, clock_data["proj_pos0"][int(T * 0.1)]),
        fontsize=8,
        color=COLOR_POS0,
    )
    axes[0].annotate(
        f"r={clock_data['corr_others']:.2f}",
        xy=(T * 0.7, clock_data["proj_others"][int(T * 0.7)]),
        fontsize=8,
        color=COLOR_OTHERS,
    )

    # Directional rotation (2D view)
    theta = np.linspace(0, np.pi / 2, T)
    x = np.cos(theta)
    y = np.sin(theta)

    colors = plt.cm.viridis(positions / T)
    for i in range(T - 1):
        axes[1].plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=colors[i], linewidth=2)

    axes[1].scatter(
        [1], [0], s=100, c=COLOR_POS0, marker="o", label="Pos-0 dir", zorder=5
    )
    axes[1].scatter(
        [0], [1], s=100, c=COLOR_OTHERS, marker="s", label="Others dir", zorder=5
    )
    axes[1].set_xlabel("Pos-0 component")
    axes[1].set_ylabel("Others component")
    axes[1].set_title("Directional Rotation")
    axes[1].legend(loc="upper right")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, T - 1))
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Position")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved geometric clock to {save_path}")


# =============================================================================
# Extrapolation Analysis
# =============================================================================


def evaluate_extrapolation(model, data, context_lengths, n_batches=20, batch_size=32):
    """Evaluate position decoding at various context lengths."""
    results = {}
    original_block_size = model.config.block_size

    for L in context_lengths:
        print(f"\nEvaluating at L={L}...")
        all_preds, all_positions = [], []

        for _ in range(n_batches):
            # Get batch of length L
            ix = torch.randint(len(data) - L, (batch_size,))
            tokens = torch.stack(
                [torch.from_numpy(data[i : i + L].astype(np.int64)) for i in ix]
            ).to(DEVICE)

            with torch.no_grad():
                # Manual forward for variable length
                e = model.wte(tokens)
                x = model.block1(e)

                # Block 2 with post-attn head
                ln1 = model.block2.ln_1(x)
                attn_out = model.block2.attn(ln1)
                x = x + attn_out
                x = model.ln_f(x)
                preds = model.pos_head(x).squeeze(-1)

            positions = (
                torch.arange(L, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            all_preds.append(preds.cpu())
            all_positions.append(positions.cpu())

        all_preds = torch.cat(all_preds, dim=0).flatten().numpy()
        all_positions = torch.cat(all_positions, dim=0).flatten().numpy()

        r, _ = stats.pearsonr(all_positions, all_preds)
        results[L] = r**2
        print(f"  L={L}: R² = {results[L]:.4f}")

    return results


def plot_extrapolation(results, save_path):
    """Plot extrapolation results."""
    fig, ax = plt.subplots(figsize=(4, 3))

    lengths = sorted(results.keys())
    r2s = [results[L] for L in lengths]

    ax.plot(lengths, r2s, "o-", color=COLOR_PRIMARY, linewidth=2, markersize=6)
    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5, label="Training length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Position $R^2$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved extrapolation plot to {save_path}")


# =============================================================================
# Write Bottleneck Plot
# =============================================================================


def plot_write_bottleneck(results, save_path):
    """Create the main write bottleneck figure."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ranks = results["ranks"]

    ax.plot(
        ranks,
        results["retention_r2s"],
        "-",
        color=COLOR_PRIMARY,
        label="Retention",
        linewidth=2,
    )
    ax.plot(
        ranks,
        results["ablation_r2s"],
        "--",
        color=COLOR_SECONDARY,
        label="Ablation",
        linewidth=2,
        alpha=0.8,
    )

    # Mark r_95
    if results["r_95"]:
        idx = ranks.index(results["r_95"])
        ax.axvline(x=results["r_95"], color=COLOR_PRIMARY, linestyle=":", alpha=0.6)
        ax.annotate(
            f"$r_{{95}}$={results['r_95']}",
            xy=(results["r_95"], results["retention_r2s"][idx]),
            xytext=(results["r_95"] + 3, results["retention_r2s"][idx] - 0.1),
            fontsize=8,
            color=COLOR_PRIMARY,
        )

    # 95% baseline line
    ax.axhline(
        y=0.95 * results["baseline_r2"],
        color="gray",
        linestyle="--",
        alpha=0.4,
        linewidth=1,
        label="95% baseline",
    )

    ax.set_xlabel("Rank $r$")
    ax.set_ylabel("Position $R^2$")
    ax.set_xlim(0, max(ranks))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved write bottleneck to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism-r2-1head-postattn/R2/uv1hq205/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument("--save_dir", type=str, default="results/r2_1head_analysis")
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument(
        "--full_block",
        action="store_true",
        help="Use full Block 2 output for head (no post-attn head)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for saved filenames (e.g., 'fullblock')",
    )
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.paper_dir, exist_ok=True)

    tag_suffix = f"_{args.tag}" if args.tag else ""

    # Load model and data
    print("Loading model...")
    checkpoint_path = ROOT_DIR / args.checkpoint
    model, config = load_model(
        str(checkpoint_path), DEVICE, post_attn=not args.full_block
    )

    print("Loading data...")
    data = load_owt_data(ROOT_DIR / args.data_dir)

    all_results = {}

    # 1. Write Bottleneck Analysis
    print("\n" + "=" * 60)
    print("1. WRITE BOTTLENECK ANALYSIS")
    print("=" * 60)
    ranks = list(range(1, 21)) + list(range(25, 101, 5))
    wb_results = run_write_bottleneck(
        model, data, ranks, args.n_batches, args.batch_size
    )
    all_results["write_bottleneck"] = wb_results

    plot_write_bottleneck(
        wb_results,
        os.path.join(args.save_dir, f"write_bottleneck_r2_1head{tag_suffix}.pdf"),
    )

    # 2. Attention Maps
    print("\n" + "=" * 60)
    print("2. ATTENTION MAP ANALYSIS")
    print("=" * 60)
    attn1, attn2 = generate_attention_map(model, data)
    plot_attention_maps(
        attn1,
        attn2,
        os.path.join(args.save_dir, f"attention_maps_r2_1head{tag_suffix}.pdf"),
    )

    # 3. Geometric Clock Analysis
    print("\n" + "=" * 60)
    print("3. GEOMETRIC CLOCK ANALYSIS")
    print("=" * 60)
    clock_data = analyze_geometric_clock(model, data)
    all_results["geometric_clock"] = clock_data
    print(f"  cos(pos0_dir, others_dir): {clock_data['cos_pos0_others']:.3f}")
    print(f"  corr(proj_pos0, position): {clock_data['corr_pos0']:.3f}")
    print(f"  corr(proj_others, position): {clock_data['corr_others']:.3f}")
    print(f"  cos(w_head, pos0_dir): {clock_data['cos_w_pos0']:.3f}")
    print(f"  cos(w_head, others_dir): {clock_data['cos_w_others']:.3f}")

    plot_geometric_clock(
        clock_data, os.path.join(args.save_dir, f"geometric_clock{tag_suffix}.pdf")
    )

    # 4. Extrapolation Analysis
    print("\n" + "=" * 60)
    print("4. EXTRAPOLATION ANALYSIS")
    print("=" * 60)
    context_lengths = [128, 256, 512, 1024, 2048, 4096]
    extrap_results = evaluate_extrapolation(
        model, data, context_lengths, n_batches=20, batch_size=args.batch_size
    )
    all_results["extrapolation"] = extrap_results

    plot_extrapolation(
        extrap_results,
        os.path.join(args.save_dir, f"extrapolation_r2_1head{tag_suffix}.pdf"),
    )

    # Save all results
    with open(
        os.path.join(args.save_dir, f"r2_1head_analysis_results{tag_suffix}.json"), "w"
    ) as f:

        def to_serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.float32, np.float64)):
                return float(value)
            if isinstance(value, dict):
                return {k: to_serializable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_serializable(v) for v in value]
            return value

        json.dump(to_serializable(all_results), f, indent=2)

    # Copy to paper directory
    import shutil

    for fname in [
        f"write_bottleneck_r2_1head{tag_suffix}.pdf",
        f"attention_maps_r2_1head{tag_suffix}.pdf",
        f"geometric_clock{tag_suffix}.pdf",
        f"extrapolation_r2_1head{tag_suffix}.pdf",
    ]:
        src = os.path.join(args.save_dir, fname)
        dst = os.path.join(args.paper_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {fname} to {args.paper_dir}")

    # Print summary for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"\nWrite Bottleneck:")
    print(f"  Baseline R²: {wb_results['baseline_r2']:.4f}")
    print(f"  r_95: {wb_results['r_95']}")

    print(f"\nGeometric Clock:")
    print(f"  cos(pos0_dir, others_dir): {clock_data['cos_pos0_others']:.3f}")
    print(f"  ||pos0_Wo_v||: {clock_data['pos0_norm']:.1f}")
    print(f"  ||others_Wo_v||: {clock_data['others_norm']:.1f}")
    print(f"  cos(w_head, others_dir): {clock_data['cos_w_others']:.3f}")

    print(f"\nExtrapolation (R²):")
    for L, r2 in sorted(extrap_results.items()):
        ratio = L / 128
        print(f"  L={L:4d} ({ratio:2.0f}x): {r2:.4f}")


if __name__ == "__main__":
    main()
