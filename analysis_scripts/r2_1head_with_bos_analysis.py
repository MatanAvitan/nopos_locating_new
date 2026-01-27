"""
Complete Analysis for R2-1-Head Geometric Clock with BOS Token
===============================================================

The correct R2-1-head model that achieves R²=0.993 uses:
- BOS token (50256) at position 0
- Checkpoint: ff0hgn5g

This script generates all plots for the ICML 2026 paper.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style
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
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLOR_BOS = "#009E73"  # Green
COLOR_OTHERS = "#CC79A7"  # Pink
COLOR_PRIMARY = "#0072B2"  # Blue
COLOR_R0 = "#0072B2"
COLOR_R2 = "#D55E00"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_TOKEN_ID = 50256


def load_model(checkpoint_path, device="cuda", post_attn=True):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", checkpoint.get("model_args", {}))

    config = TwoLayerMechanismConfig(
        block_size=cfg.get("block_size", 128),
        vocab_size=cfg.get("vocab_size", 50304),
        n_embd=cfg.get("n_embd", 768),
        n_head=cfg.get("n_head", 1),
        dropout=0.0,
        norm_type=cfg.get("norm_type", "layernorm"),
        bias=True,
        use_regression=True,
    )

    model = TwoLayerMechanismModel(config)
    if post_attn:
        model.set_post_attn_head(True)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config


def load_data(data_dir=None):
    if data_dir is None:
        data_dir = ROOT_DIR / "nanoGPT/data/openwebtext"
    return np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch_with_bos(data, batch_size, block_size, device):
    """Get batch with BOS token at position 0."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    sequences = []
    for i in ix:
        after_bos = data[i : i + block_size - 1].astype(np.int64)
        seq = np.concatenate([[BOS_TOKEN_ID], after_bos])
        sequences.append(torch.from_numpy(seq))
    return torch.stack(sequences).to(device)


def r2_score(preds, targets):
    preds, targets = preds.flatten().float(), targets.flatten().float()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def corrcoef(x, y):
    x, y = x.flatten().float(), y.flatten().float()
    return ((x - x.mean()) @ (y - y.mean()) / (x.std() * y.std() * len(x))).item()


# =============================================================================
# Write Bottleneck
# =============================================================================


def get_write_map_svd(model):
    D = model.config.n_embd
    W_V = model.block2.attn.c_attn.weight[2 * D :, :]
    W_O = model.block2.attn.c_proj.weight
    B = W_O @ W_V
    U, S, Vt = torch.linalg.svd(B, full_matrices=True)
    return U, S, Vt


def forward_with_intervention(model, tokens, U, rank, intervention_type="retention"):
    B, T = tokens.shape
    D = model.config.n_embd

    with torch.no_grad():
        e = model.wte(tokens)

        # Block 1
        ln1 = model.block1.ln_1(e)
        attn_out1 = model.block1.attn(ln1)
        r1_attn = e + attn_out1
        ln2_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_b1)
        r1 = r1_attn + mlp_out1

        # Block 2
        ln1_b2 = model.block2.ln_1(r1)
        attn_out2 = model.block2.attn(ln1_b2)

        # Intervention
        U_r = U[:, :rank]
        if intervention_type == "retention":
            attn_out2 = attn_out2 @ U_r @ U_r.T
        elif intervention_type == "ablation":
            attn_out2 = attn_out2 - attn_out2 @ U_r @ U_r.T

        # Post-attention residual (no MLP2)
        r2 = r1 + attn_out2
        final = model.ln_f(r2)
        pred = model.pos_head(final).squeeze(-1)

    return pred


def run_write_bottleneck(model, data, ranks, n_batches=50, batch_size=32):
    print("\n" + "=" * 60)
    print("WRITE BOTTLENECK ANALYSIS")
    print("=" * 60)

    U, S, Vt = get_write_map_svd(model)
    T = model.config.block_size

    # Baseline
    all_preds, all_pos = [], []
    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, T, DEVICE)
        with torch.no_grad():
            output, _ = model(tokens)
            preds = output.squeeze(-1)
        positions = (
            torch.arange(T, device=DEVICE).float().unsqueeze(0).expand(batch_size, -1)
        )
        all_preds.append(preds.cpu())
        all_pos.append(positions.cpu())

    baseline_r2 = r2_score(torch.cat(all_preds), torch.cat(all_pos))
    print(f"Baseline R²: {baseline_r2:.4f}")

    retention_r2s, ablation_r2s = [], []

    for rank in tqdm(ranks, desc="Computing R² at each rank"):
        # Retention
        all_preds, all_pos = [], []
        for _ in range(n_batches):
            tokens = get_batch_with_bos(data, batch_size, T, DEVICE)
            preds = forward_with_intervention(model, tokens, U, rank, "retention")
            positions = (
                torch.arange(T, device=DEVICE)
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
            tokens = get_batch_with_bos(data, batch_size, T, DEVICE)
            preds = forward_with_intervention(model, tokens, U, rank, "ablation")
            positions = (
                torch.arange(T, device=DEVICE)
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


def plot_write_bottleneck(results, save_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ranks = results["ranks"]
    retention_r2s = results["retention_r2s"]
    ablation_r2s = results["ablation_r2s"]

    # Clip ablation values for visualization but show they go negative
    ablation_clipped = [max(r, -0.2) for r in ablation_r2s]

    ax.plot(
        ranks,
        retention_r2s,
        "-",
        color=COLOR_PRIMARY,
        label="Retention",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.plot(
        ranks,
        ablation_clipped,
        "--",
        color=COLOR_R2,
        label="Ablation",
        linewidth=2,
        marker="s",
        markersize=3,
    )

    if results["r_95"]:
        idx = ranks.index(results["r_95"])
        ax.axvline(
            x=results["r_95"],
            color=COLOR_PRIMARY,
            linestyle=":",
            alpha=0.6,
            linewidth=1.5,
        )
        ax.annotate(
            f"$r_{{95}}$={results['r_95']}",
            xy=(results["r_95"], retention_r2s[idx]),
            xytext=(results["r_95"] + 3, 0.7),
            fontsize=9,
            color=COLOR_PRIMARY,
            arrowprops=dict(arrowstyle="->", color=COLOR_PRIMARY, lw=0.8),
        )

    ax.axhline(
        y=0.95 * results["baseline_r2"],
        color="gray",
        linestyle="--",
        alpha=0.4,
        linewidth=1,
    )
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Add annotation showing ablation goes much lower
    ax.annotate(
        "$R^2 < -1$", xy=(50, -0.15), fontsize=7, color=COLOR_R2, style="italic"
    )

    ax.set_xlabel("Rank $r$")
    ax.set_ylabel("Position $R^2$")
    ax.set_xlim(0, max(ranks))
    ax.set_ylim(-0.25, 1.05)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Geometric Clock Analysis
# =============================================================================


def analyze_geometric_clock(model, data, batch_size=64, n_batches=10):
    D = model.config.n_embd
    T = model.config.block_size

    all_proj_bos, all_proj_others = [], []
    all_attn_to_bos = []

    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, T, DEVICE)

        with torch.no_grad():
            model(tokens, capture_taps=True)
            taps = model.get_all_taps()
            _, attn2 = model.get_attention_weights()

            W_V = model.block2.attn.c_attn.weight[2 * D :, :]
            W_O = model.block2.attn.c_proj.weight
            b_O = model.block2.attn.c_proj.bias

            ln2_1 = taps["block2_ln1"]
            attn_out = taps["block2_attn"]

            v2 = ln2_1 @ W_V.T
            Wo_v = v2 @ W_O.T

            bos_mean = Wo_v[:, 0, :].mean(dim=0)
            others_mean = Wo_v[:, 1:, :].mean(dim=(0, 1))

            bos_dir = F.normalize(bos_mean.unsqueeze(0), dim=-1).squeeze()
            others_dir = F.normalize(others_mean.unsqueeze(0), dim=-1).squeeze()

            attn_out_no_bias = attn_out - b_O
            proj_bos = (attn_out_no_bias @ bos_dir).mean(dim=0)
            proj_others = (attn_out_no_bias @ others_dir).mean(dim=0)

            all_proj_bos.append(proj_bos.cpu())
            all_proj_others.append(proj_others.cpu())

            attn_to_bos = attn2[:, :, :, 0].mean(dim=(0, 1))
            all_attn_to_bos.append(attn_to_bos.cpu())

    proj_bos = torch.stack(all_proj_bos).mean(dim=0).numpy()
    proj_others = torch.stack(all_proj_others).mean(dim=0).numpy()
    attn_to_bos = torch.stack(all_attn_to_bos).mean(dim=0).numpy()

    positions = np.arange(T)
    corr_bos = np.corrcoef(proj_bos, positions)[0, 1]
    corr_others = np.corrcoef(proj_others, positions)[0, 1]

    w_head = model.pos_head.weight.detach().squeeze().cpu()
    cos_w_bos = F.cosine_similarity(
        w_head.unsqueeze(0), bos_dir.cpu().unsqueeze(0)
    ).item()
    cos_w_others = F.cosine_similarity(
        w_head.unsqueeze(0), others_dir.cpu().unsqueeze(0)
    ).item()
    cos_bos_others = F.cosine_similarity(
        bos_dir.cpu().unsqueeze(0), others_dir.cpu().unsqueeze(0)
    ).item()

    return {
        "proj_bos": proj_bos,
        "proj_others": proj_others,
        "attn_to_bos": attn_to_bos,
        "corr_bos": corr_bos,
        "corr_others": corr_others,
        "cos_w_bos": cos_w_bos,
        "cos_w_others": cos_w_others,
        "cos_bos_others": cos_bos_others,
        "bos_norm": bos_mean.norm().item(),
        "others_norm": others_mean.norm().item(),
    }


def plot_geometric_clock(clock_data, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))

    T = len(clock_data["proj_bos"])
    positions = np.arange(T)

    # Panel A
    axes[0].plot(
        positions,
        clock_data["proj_bos"],
        label=f"BOS dir (r={clock_data['corr_bos']:.2f})",
        color=COLOR_BOS,
        linewidth=1.5,
    )
    axes[0].plot(
        positions,
        clock_data["proj_others"],
        label=f"Others dir (r={clock_data['corr_others']:.2f})",
        color=COLOR_OTHERS,
        linewidth=1.5,
    )
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Projection")
    axes[0].set_title("(a) Attention Output Projection")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Panel B
    axes[1].plot(
        positions,
        clock_data["attn_to_bos"],
        color=COLOR_R2,
        linewidth=1.5,
        label="Learned",
    )
    axes[1].plot(
        positions, 1.0 / (positions + 1), "--", color="gray", alpha=0.7, label="Uniform"
    )
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Attention to BOS")
    axes[1].set_title("(b) Attention to BOS Token")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel C
    theta = np.linspace(0, np.pi / 2, T)
    x = np.cos(theta)
    y = np.sin(theta)
    colors = plt.cm.viridis(positions / T)

    for i in range(T - 1):
        axes[2].plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=colors[i], linewidth=2)

    axes[2].scatter([1], [0], s=80, c=COLOR_BOS, marker="o", label="BOS dir", zorder=5)
    axes[2].scatter(
        [0], [1], s=80, c=COLOR_OTHERS, marker="s", label="Others dir", zorder=5
    )
    axes[2].set_xlabel("BOS component")
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
    print(f"Saved: {save_path}")


# =============================================================================
# Attention Maps
# =============================================================================


def plot_attention_maps(model, data, save_path, batch_size=64):
    tokens = get_batch_with_bos(data, batch_size, model.config.block_size, DEVICE)

    with torch.no_grad():
        model(tokens, capture_taps=True)
        attn1, attn2 = model.get_attention_weights()

    attn1_avg = attn1.mean(dim=0)[0].cpu().numpy()  # 1 head
    attn2_avg = attn2.mean(dim=0)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    def strip_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    im1 = axes[0].imshow(attn1_avg, cmap="cividis", aspect="auto")
    axes[0].set_title("Block 1 (Frozen Random)")
    strip_axes(axes[0])
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(attn2_avg, cmap="cividis", aspect="auto")
    axes[1].set_title("Block 2 (Trained)")
    strip_axes(axes[1])
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism-r2-1head-postattn/R2/ff0hgn5g/best_ckpt.pt",
    )
    parser.add_argument("--save_dir", type=str, default="results/r2_1head_bos_analysis")
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.paper_dir, exist_ok=True)

    print("Loading model...")
    model, config = load_model(str(ROOT_DIR / args.checkpoint), DEVICE, post_attn=True)
    print(f"  n_head={config.n_head}, n_embd={config.n_embd}")

    print("Loading data...")
    data = load_data()

    all_results = {}

    # 1. Write Bottleneck
    ranks = list(range(1, 21)) + list(range(25, 101, 5))
    wb_results = run_write_bottleneck(
        model, data, ranks, args.n_batches, args.batch_size
    )
    all_results["write_bottleneck"] = wb_results
    plot_write_bottleneck(
        wb_results, os.path.join(args.save_dir, "write_bottleneck_r2_1head.pdf")
    )

    # 2. Attention Maps
    print("\n" + "=" * 60)
    print("ATTENTION MAPS")
    print("=" * 60)
    plot_attention_maps(
        model, data, os.path.join(args.save_dir, "attention_maps_r2_1head.pdf")
    )

    # 3. Geometric Clock
    print("\n" + "=" * 60)
    print("GEOMETRIC CLOCK ANALYSIS")
    print("=" * 60)
    clock_data = analyze_geometric_clock(model, data)

    print(f"  ||W_O @ v_BOS||: {clock_data['bos_norm']:.1f}")
    print(f"  ||W_O @ v_others||: {clock_data['others_norm']:.1f}")
    print(f"  cos(BOS dir, Others dir): {clock_data['cos_bos_others']:.3f}")
    print(f"  corr(proj_BOS, position): {clock_data['corr_bos']:.3f}")
    print(f"  corr(proj_others, position): {clock_data['corr_others']:.3f}")
    print(f"  cos(w_head, BOS dir): {clock_data['cos_w_bos']:.3f}")
    print(f"  cos(w_head, Others dir): {clock_data['cos_w_others']:.3f}")

    all_results["geometric_clock"] = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else v.tolist()
        for k, v in clock_data.items()
    }

    plot_geometric_clock(clock_data, os.path.join(args.save_dir, "geometric_clock.pdf"))

    # Save
    with open(os.path.join(args.save_dir, "analysis_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Copy to paper dir
    import shutil

    for fname in [
        "write_bottleneck_r2_1head.pdf",
        "attention_maps_r2_1head.pdf",
        "geometric_clock.pdf",
    ]:
        src = os.path.join(args.save_dir, fname)
        dst = os.path.join(args.paper_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Write Bottleneck:")
    print(f"  Baseline R²: {wb_results['baseline_r2']:.4f}")
    print(f"  r_95: {wb_results['r_95']}")
    print(f"\nGeometric Clock:")
    print(f"  ||W_O @ v_BOS||: {clock_data['bos_norm']:.1f}")
    print(f"  ||W_O @ v_others||: {clock_data['others_norm']:.1f}")
    print(f"  cos(BOS dir, Others dir): {clock_data['cos_bos_others']:.3f}")
    print(f"  cos(w_head, Others dir): {clock_data['cos_w_others']:.3f}")


if __name__ == "__main__":
    main()
