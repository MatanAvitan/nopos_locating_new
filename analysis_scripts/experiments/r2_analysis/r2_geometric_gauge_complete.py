"""
Complete Analysis for Geometric Gauge Mechanism (12-head R2 model)
===================================================================

This script generates all plots and data for the ICML 2026 paper focusing on:
1. The complete geometric gauge mechanism
2. Write bottleneck curves
3. Attention maps
4. Extrapolation analysis
5. Comparison between R0 and R2

Usage:
    python analysis_scripts/r2_geometric_gauge_complete.py
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

ROOT_DIR = Path(__file__).parent.parent.parent.parent
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

COLOR_R0 = "#0072B2"
COLOR_R2 = "#D55E00"
COLOR_POS0 = "#009E73"
COLOR_OTHERS = "#CC79A7"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_TOKEN_ID = 50256


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", checkpoint.get("model_args", {}))

    config = TwoLayerMechanismConfig(
        block_size=cfg.get("block_size", 128),
        vocab_size=cfg.get("vocab_size", 50304),
        n_embd=cfg.get("n_embd", 768),
        n_head=cfg.get("n_head", 12),
        dropout=0.0,
        norm_type=cfg.get("norm_type", "layernorm"),
        bias=True,
        use_regression=True,
    )

    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model, config, checkpoint.get("best_metrics", {})


def load_owt_data(data_dir=None):
    if data_dir is None:
        data_dir = ROOT_DIR / "nanoGPT/data/openwebtext"
    return np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device):
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


def r2_score(preds, targets):
    preds, targets = preds.flatten().float(), targets.flatten().float()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


# =============================================================================
# Write Bottleneck Analysis
# =============================================================================


def get_block2_write_map_svd(model):
    """Compute SVD of B = W_O @ W_V from Block 2 attention."""
    d = model.config.n_embd
    W_V = model.block2.attn.c_attn.weight[2 * d :, :]
    W_O = model.block2.attn.c_proj.weight
    B = W_O @ W_V
    U, S, Vt = torch.linalg.svd(B, full_matrices=True)
    return U, S, Vt


def forward_with_write_intervention(
    model, tokens, U, rank, intervention_type="retention"
):
    """Forward pass with intervention on Block 2 attention output."""
    B, T = tokens.shape
    d = model.config.n_embd
    n_head = model.config.n_head
    head_dim = d // n_head

    with torch.no_grad():
        e = model.wte(tokens)

        # Block 1
        ln1 = model.block1.ln_1(e)
        attn1 = model.block1.attn
        qkv1 = attn1.c_attn(ln1)
        q1, k1, v1 = qkv1.split(d, dim=2)
        q1 = q1.view(B, T, n_head, head_dim).transpose(1, 2)
        k1 = k1.view(B, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B, T, n_head, head_dim).transpose(1, 2)

        att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        att1 = att1.masked_fill(mask, float("-inf"))
        att1 = F.softmax(att1, dim=-1)
        y1 = (att1 @ v1).transpose(1, 2).contiguous().view(B, T, d)
        attn_out1 = attn1.c_proj(y1)

        r1_attn = e + attn_out1
        ln2_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_b1)
        r1 = r1_attn + mlp_out1

        # Block 2
        ln1_b2 = model.block2.ln_1(r1)
        attn2 = model.block2.attn
        qkv2 = attn2.c_attn(ln1_b2)
        q2, k2, v2 = qkv2.split(d, dim=2)
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        att2 = att2.masked_fill(mask, float("-inf"))
        att2 = F.softmax(att2, dim=-1)
        y2 = (att2 @ v2).transpose(1, 2).contiguous().view(B, T, d)
        attn_out2 = attn2.c_proj(y2)

        # Intervention
        U_r = U[:, :rank]
        if intervention_type == "retention":
            attn_out2 = attn_out2 @ U_r @ U_r.T
        elif intervention_type == "ablation":
            attn_out2 = attn_out2 - attn_out2 @ U_r @ U_r.T

        r2_attn = r1 + attn_out2
        ln2_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_b2)
        r2 = r2_attn + mlp_out2

        final = model.ln_f(r2)
        pred = model.pos_head(final).squeeze(-1)

    return pred


def run_write_bottleneck(
    model, data, ranks, n_batches=50, batch_size=32, model_name=""
):
    """Run full write bottleneck experiment."""
    print(f"\n{'=' * 60}")
    print(f"Write Bottleneck: {model_name}")
    print(f"{'=' * 60}")

    U, S, Vt = get_block2_write_map_svd(model)
    block_size = model.config.block_size

    # Baseline
    all_preds, all_pos = [], []
    for _ in range(n_batches):
        tokens = get_batch(data, batch_size, block_size, DEVICE)
        with torch.no_grad():
            output, _ = model(tokens)
            preds = output.squeeze(-1)
        positions = (
            torch.arange(block_size, device=DEVICE)
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        all_preds.append(preds.cpu())
        all_pos.append(positions.cpu())

    baseline_r2 = r2_score(torch.cat(all_preds), torch.cat(all_pos))
    print(f"Baseline R²: {baseline_r2:.4f}")

    retention_r2s, ablation_r2s = [], []

    for rank in tqdm(ranks, desc=f"{model_name} ranks"):
        # Retention
        all_preds, all_pos = [], []
        for _ in range(n_batches):
            tokens = get_batch(data, batch_size, block_size, DEVICE)
            preds = forward_with_write_intervention(model, tokens, U, rank, "retention")
            positions = (
                torch.arange(block_size, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            all_preds.append(preds.cpu())
            all_pos.append(positions.cpu())
        retention_r2s.append(r2_score(torch.cat(all_preds), torch.cat(all_pos)))

        # Ablation
        all_preds, all_pos = [], []
        for _ in range(n_batches):
            tokens = get_batch(data, batch_size, block_size, DEVICE)
            preds = forward_with_write_intervention(model, tokens, U, rank, "ablation")
            positions = (
                torch.arange(block_size, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            all_preds.append(preds.cpu())
            all_pos.append(positions.cpu())
        ablation_r2s.append(r2_score(torch.cat(all_preds), torch.cat(all_pos)))

    # Find r_95
    r_95 = None
    threshold = 0.95 * baseline_r2
    for i, r2 in enumerate(retention_r2s):
        if r2 >= threshold:
            r_95 = ranks[i]
            break

    print(f"r_95: {r_95}")

    return {
        "baseline_r2": float(baseline_r2),
        "ranks": ranks,
        "retention_r2s": [float(x) for x in retention_r2s],
        "ablation_r2s": [float(x) for x in ablation_r2s],
        "r_95": r_95,
        "singular_values": S.detach().cpu().numpy().tolist(),
    }


# =============================================================================
# Attention Maps
# =============================================================================


def generate_attention_maps(model, data, batch_size=64):
    """Generate attention maps."""
    tokens = get_batch(data, batch_size, model.config.block_size, DEVICE)
    with torch.no_grad():
        model(tokens, capture_taps=True)
        attn1, attn2 = model.get_attention_weights()
    return attn1.mean(dim=0), attn2.mean(dim=0)


def plot_attention_maps(attn1, attn2, model_name, save_path, n_heads_to_show=4):
    """Plot attention maps."""
    n_heads = attn2.shape[0]

    fig, axes = plt.subplots(2, n_heads_to_show, figsize=(n_heads_to_show * 2, 4))

    def strip_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for h in range(n_heads_to_show):
        # Block 1
        im1 = axes[0, h].imshow(
            attn1[h].cpu().numpy(), cmap="cividis", aspect="auto", vmin=0, vmax=0.1
        )
        axes[0, h].set_title(f"B1 H{h}")
        strip_axes(axes[0, h])

        # Block 2
        im2 = axes[1, h].imshow(
            attn2[h].cpu().numpy(), cmap="cividis", aspect="auto", vmin=0, vmax=0.5
        )
        axes[1, h].set_title(f"B2 H{h}")
        strip_axes(axes[1, h])

    plt.suptitle(f"{model_name} Attention Maps", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved attention maps to {save_path}")


# =============================================================================
# Geometric Gauge Analysis
# =============================================================================


def analyze_geometric_gauge(model, data, batch_size=64, n_batches=10):
    """Analyze geometric gauge mechanism for R2."""
    D = model.config.n_embd
    T = model.config.block_size
    n_head = model.config.n_head
    head_dim = D // n_head

    all_proj_pos0, all_proj_others = [], []
    all_attn_to_pos0 = []

    for _ in range(n_batches):
        tokens = get_batch(data, batch_size, T, DEVICE)

        with torch.no_grad():
            model(tokens, capture_taps=True)
            taps = model.get_all_taps()
            attn1, attn2 = model.get_attention_weights()

            # Block 2 value computation
            W_V = model.block2.attn.c_attn.weight[2 * D :, :]
            W_O = model.block2.attn.c_proj.weight
            b_O = model.block2.attn.c_proj.bias

            ln2_1 = taps["block2_ln1"]
            attn_out = taps["block2_attn"]

            # Compute projected values per head
            v2 = ln2_1 @ W_V.T
            Wo_v = v2 @ W_O.T

            # Mean directions
            pos0_mean = Wo_v[:, 0, :].mean(dim=0)
            others_mean = Wo_v[:, 1:, :].mean(dim=(0, 1))

            pos0_dir = F.normalize(pos0_mean.unsqueeze(0), dim=-1).squeeze()
            others_dir = F.normalize(others_mean.unsqueeze(0), dim=-1).squeeze()

            # Project attention output
            attn_out_no_bias = attn_out - b_O
            proj_pos0 = (attn_out_no_bias @ pos0_dir).mean(dim=0)
            proj_others = (attn_out_no_bias @ others_dir).mean(dim=0)

            all_proj_pos0.append(proj_pos0.cpu())
            all_proj_others.append(proj_others.cpu())

            # Attention to position 0 (averaged across heads)
            attn_to_pos0 = attn2[:, :, :, 0].mean(dim=(0, 1))
            all_attn_to_pos0.append(attn_to_pos0.cpu())

    # Average across batches
    proj_pos0 = torch.stack(all_proj_pos0).mean(dim=0).numpy()
    proj_others = torch.stack(all_proj_others).mean(dim=0).numpy()
    attn_to_pos0 = torch.stack(all_attn_to_pos0).mean(dim=0).numpy()

    # Correlations
    positions = np.arange(T)
    corr_pos0 = np.corrcoef(proj_pos0, positions)[0, 1]
    corr_others = np.corrcoef(proj_others, positions)[0, 1]

    # Head alignment
    w_head = model.pos_head.weight.detach().squeeze().cpu()
    cos_w_pos0 = F.cosine_similarity(
        w_head.unsqueeze(0), pos0_dir.cpu().unsqueeze(0)
    ).item()
    cos_w_others = F.cosine_similarity(
        w_head.unsqueeze(0), others_dir.cpu().unsqueeze(0)
    ).item()
    cos_pos0_others = F.cosine_similarity(
        pos0_dir.cpu().unsqueeze(0), others_dir.cpu().unsqueeze(0)
    ).item()

    return {
        "proj_pos0": proj_pos0,
        "proj_others": proj_others,
        "attn_to_pos0": attn_to_pos0,
        "corr_pos0": corr_pos0,
        "corr_others": corr_others,
        "cos_w_pos0": cos_w_pos0,
        "cos_w_others": cos_w_others,
        "cos_pos0_others": cos_pos0_others,
        "pos0_norm": pos0_mean.norm().item(),
        "others_norm": others_mean.norm().item(),
    }


def plot_geometric_gauge(gauge_data, save_path):
    """Plot geometric gauge mechanism."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))

    T = len(gauge_data["proj_pos0"])
    positions = np.arange(T)

    # Panel A: Projections
    axes[0].plot(
        positions,
        gauge_data["proj_pos0"],
        label=f"Pos-0 dir (r={gauge_data['corr_pos0']:.2f})",
        color=COLOR_POS0,
        linewidth=1.5,
    )
    axes[0].plot(
        positions,
        gauge_data["proj_others"],
        label=f"Others dir (r={gauge_data['corr_others']:.2f})",
        color=COLOR_OTHERS,
        linewidth=1.5,
    )
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Projection")
    axes[0].set_title("(a) Attention Output Projection")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Panel B: Attention to position 0
    axes[1].plot(positions, gauge_data["attn_to_pos0"], color=COLOR_R2, linewidth=1.5)
    axes[1].plot(
        positions, 1.0 / (positions + 1), "--", color="gray", alpha=0.7, label="Uniform"
    )
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Attention to pos 0")
    axes[1].set_title("(b) Attention to Position 0")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel C: Directional rotation schematic
    theta = np.linspace(0, np.pi / 2, T)
    x = np.cos(theta)
    y = np.sin(theta)
    colors = plt.cm.viridis(positions / T)

    for i in range(T - 1):
        axes[2].plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=colors[i], linewidth=2)

    axes[2].scatter(
        [1], [0], s=80, c=COLOR_POS0, marker="o", label="Pos-0 dir", zorder=5
    )
    axes[2].scatter(
        [0], [1], s=80, c=COLOR_OTHERS, marker="s", label="Others dir", zorder=5
    )
    axes[2].set_xlabel("Pos-0 component")
    axes[2].set_ylabel("Others component")
    axes[2].set_title("(c) Directional Rotation")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, T - 1))
    cbar = plt.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Position", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved geometric gauge to {save_path}")


# =============================================================================
# Extrapolation Analysis
# =============================================================================


def evaluate_extrapolation(model, data, context_lengths, n_batches=30, batch_size=32):
    """Evaluate position decoding at various context lengths."""
    results = {}

    for L in context_lengths:
        if L >= 4096:
            effective_batch_size = max(1, batch_size // 16)
            effective_batches = max(1, n_batches // 8)
        elif L >= 2048:
            effective_batch_size = max(1, batch_size // 8)
            effective_batches = max(1, n_batches // 4)
        elif L >= 1024:
            effective_batch_size = max(1, batch_size // 4)
            effective_batches = max(1, n_batches // 2)
        elif L >= 512:
            effective_batch_size = max(1, batch_size // 2)
            effective_batches = n_batches
        else:
            effective_batch_size = batch_size
            effective_batches = n_batches

        print(f"  Evaluating L={L}...", end=" ")
        all_preds, all_pos = [], []

        for _ in range(effective_batches):
            ix = torch.randint(len(data) - (L - 1), (effective_batch_size,))
            tokens = torch.stack(
                [
                    torch.from_numpy(
                        np.concatenate(
                            [[BOS_TOKEN_ID], data[i : i + L - 1].astype(np.int64)]
                        )
                    )
                    for i in ix
                ]
            ).to(DEVICE)

            with torch.no_grad():
                e = model.wte(tokens)
                x = model.block1(e)
                x = model.block2(x)
                x = model.ln_f(x)
                preds = model.pos_head(x).squeeze(-1)

            positions = (
                torch.arange(L, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(effective_batch_size, -1)
            )
            all_preds.append(preds.cpu())
            all_pos.append(positions.cpu())

        torch.cuda.empty_cache()

        r2 = r2_score(torch.cat(all_preds), torch.cat(all_pos))
        results[L] = r2
        print(f"R²={r2:.4f}")

    return results


def plot_extrapolation(results_r0, results_r2, save_path):
    """Plot extrapolation comparison."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    lengths = sorted(results_r0.keys())
    r2s_r0 = [results_r0[L] for L in lengths]
    r2s_r2 = [results_r2[L] for L in lengths]

    ax.plot(
        lengths,
        r2s_r0,
        "o-",
        color=COLOR_R0,
        linewidth=1.5,
        markersize=5,
        label="R0 (Full)",
    )
    ax.plot(
        lengths,
        r2s_r2,
        "s-",
        color=COLOR_R2,
        linewidth=1.5,
        markersize=5,
        label="R2 (Attn2-only)",
    )
    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5, label="Train length")

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Position $R^2$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths], rotation=45)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved extrapolation to {save_path}")


# =============================================================================
# Write Bottleneck Plot
# =============================================================================


def plot_write_bottleneck_comparison(results_r0, results_r2, save_path):
    """Plot write bottleneck comparison."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ranks = results_r0["ranks"]

    ax.plot(
        ranks,
        results_r0["retention_r2s"],
        "-",
        color=COLOR_R0,
        label="R0 retention",
        linewidth=1.5,
    )
    ax.plot(
        ranks,
        results_r0["ablation_r2s"],
        "--",
        color=COLOR_R0,
        label="R0 ablation",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        ranks,
        results_r2["retention_r2s"],
        "-",
        color=COLOR_R2,
        label="R2 retention",
        linewidth=1.5,
    )
    ax.plot(
        ranks,
        results_r2["ablation_r2s"],
        "--",
        color=COLOR_R2,
        label="R2 ablation",
        linewidth=1.5,
        alpha=0.7,
    )

    # Mark r_95
    if results_r0["r_95"]:
        ax.axvline(x=results_r0["r_95"], color=COLOR_R0, linestyle=":", alpha=0.5)
        ax.text(
            results_r0["r_95"] + 1,
            0.15,
            f"$r_{{95}}$={results_r0['r_95']}",
            fontsize=7,
            color=COLOR_R0,
        )

    if results_r2["r_95"]:
        ax.axvline(x=results_r2["r_95"], color=COLOR_R2, linestyle=":", alpha=0.5)
        ax.text(
            results_r2["r_95"] + 1,
            0.25,
            f"$r_{{95}}$={results_r2['r_95']}",
            fontsize=7,
            color=COLOR_R2,
        )

    ax.axhline(
        y=0.95 * results_r0["baseline_r2"],
        color="gray",
        linestyle="--",
        alpha=0.3,
        linewidth=0.8,
    )

    ax.set_xlabel("Rank $r$")
    ax.set_ylabel("Position $R^2$")
    ax.set_xlim(0, max(ranks))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=7)
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
        "--save_dir", type=str, default="results/geometric_gauge_analysis"
    )
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.paper_dir, exist_ok=True)

    # Load models
    print("Loading R0 model...")
    model_r0, config_r0, metrics_r0 = load_model(
        str(ROOT_DIR / args.r0_checkpoint), DEVICE
    )
    print(f"  R0 Val R²: {metrics_r0.get('val_r2', 'N/A')}")

    print("Loading R2 model...")
    model_r2, config_r2, metrics_r2 = load_model(
        str(ROOT_DIR / args.r2_checkpoint), DEVICE
    )
    print(f"  R2 Val R²: {metrics_r2.get('val_r2', 'N/A')}")

    print("Loading data...")
    data = load_owt_data()

    all_results = {
        "r0_metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in metrics_r0.items()
            if not isinstance(v, list)
        },
        "r2_metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in metrics_r2.items()
            if not isinstance(v, list)
        },
    }

    # 1. Write Bottleneck
    print("\n" + "=" * 60)
    print("1. WRITE BOTTLENECK ANALYSIS")
    print("=" * 60)
    ranks = list(range(1, 21)) + list(range(25, 101, 5))

    wb_r0 = run_write_bottleneck(
        model_r0, data, ranks, args.n_batches, args.batch_size, "R0"
    )
    wb_r2 = run_write_bottleneck(
        model_r2, data, ranks, args.n_batches, args.batch_size, "R2"
    )

    all_results["write_bottleneck_r0"] = wb_r0
    all_results["write_bottleneck_r2"] = wb_r2

    plot_write_bottleneck_comparison(
        wb_r0, wb_r2, os.path.join(args.save_dir, "write_bottleneck_curves.pdf")
    )

    # 2. Attention Maps
    print("\n" + "=" * 60)
    print("2. ATTENTION MAPS")
    print("=" * 60)
    attn1_r0, attn2_r0 = generate_attention_maps(model_r0, data)
    plot_attention_maps(
        attn1_r0, attn2_r0, "R0", os.path.join(args.save_dir, "attention_maps_r0.pdf")
    )

    attn1_r2, attn2_r2 = generate_attention_maps(model_r2, data)
    plot_attention_maps(
        attn1_r2, attn2_r2, "R2", os.path.join(args.save_dir, "attention_maps_r2.pdf")
    )

    # 3. Geometric Gauge (R2)
    print("\n" + "=" * 60)
    print("3. GEOMETRIC GAUGE ANALYSIS (R2)")
    print("=" * 60)
    gauge_data = analyze_geometric_gauge(model_r2, data)

    print(f"  cos(pos0_dir, others_dir): {gauge_data['cos_pos0_others']:.3f}")
    print(f"  corr(proj_pos0, position): {gauge_data['corr_pos0']:.3f}")
    print(f"  corr(proj_others, position): {gauge_data['corr_others']:.3f}")
    print(f"  cos(w_head, others_dir): {gauge_data['cos_w_others']:.3f}")

    all_results["geometric_gauge"] = {
        k: float(v)
        if isinstance(v, (int, float, np.floating))
        else v.tolist()
        if isinstance(v, np.ndarray)
        else v
        for k, v in gauge_data.items()
    }

    plot_geometric_gauge(gauge_data, os.path.join(args.save_dir, "geometric_gauge.pdf"))

    # 4. Extrapolation
    print("\n" + "=" * 60)
    print("4. EXTRAPOLATION ANALYSIS")
    print("=" * 60)
    context_lengths = [128, 256, 512, 1024, 2048, 4096]

    print("R0:")
    extrap_r0 = evaluate_extrapolation(
        model_r0, data, context_lengths, n_batches=30, batch_size=args.batch_size
    )

    print("R2:")
    extrap_r2 = evaluate_extrapolation(
        model_r2, data, context_lengths, n_batches=30, batch_size=args.batch_size
    )

    all_results["extrapolation_r0"] = {str(k): float(v) for k, v in extrap_r0.items()}
    all_results["extrapolation_r2"] = {str(k): float(v) for k, v in extrap_r2.items()}

    plot_extrapolation(
        extrap_r0, extrap_r2, os.path.join(args.save_dir, "extrapolation.pdf")
    )

    # Save results
    with open(os.path.join(args.save_dir, "analysis_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Copy to paper directory
    import shutil

    for fname in [
        "write_bottleneck_curves.pdf",
        "attention_maps_r0.pdf",
        "attention_maps_r2.pdf",
        "geometric_gauge.pdf",
        "extrapolation.pdf",
    ]:
        src = os.path.join(args.save_dir, fname)
        dst = os.path.join(args.paper_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
    print(f"\nCopied plots to {args.paper_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"\nWrite Bottleneck:")
    print(f"  R0: baseline R²={wb_r0['baseline_r2']:.4f}, r_95={wb_r0['r_95']}")
    print(f"  R2: baseline R²={wb_r2['baseline_r2']:.4f}, r_95={wb_r2['r_95']}")

    print(f"\nGeometric Gauge (R2):")
    print(f"  cos(pos0_dir, others_dir): {gauge_data['cos_pos0_others']:.3f}")
    print(f"  corr(proj_others, position): {gauge_data['corr_others']:.3f}")

    print(f"\nExtrapolation (R²):")
    print(f"  {'Length':<8} {'R0':<10} {'R2':<10}")
    for L in context_lengths:
        ratio = L // 128
        print(f"  {L:<8} {extrap_r0[L]:.4f}     {extrap_r2[L]:.4f}   ({ratio}x)")


if __name__ == "__main__":
    main()
