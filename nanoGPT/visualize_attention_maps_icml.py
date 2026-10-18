"""
ICML-Quality Attention Map Visualizations

Publication-ready figures with:
- Clear colormaps (sequential for attention, diverging where appropriate)
- Proper font sizes and family (serif for ICML)
- Clean axis labels and titles
- Legends outside plot area to avoid overlap
- High DPI for print quality
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import wandb
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "text.usetex": False,  # Set True if LaTeX is available
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Custom colormaps
ATTENTION_CMAP = "cividis"  # Sequential, higher-contrast attention weights [0, 1]
DIVERGING_CMAP = "RdBu_r"  # Diverging for correlations [-1, 1]
VIRIDIS_CMAP = "viridis"  # For general heatmaps
BOS_TOKEN_ID = 50256


def load_model(regime, device="cuda"):
    """Load a trained model for a specific regime."""
    checkpoint_path = f"out-2layer-mechanism/{regime}/best_ckpt.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]

    model_config_fields = [
        "block_size",
        "vocab_size",
        "n_embd",
        "n_head",
        "dropout",
        "norm_type",
    ]
    filtered_config = {
        k: config_dict[k] for k in model_config_fields if k in config_dict
    }
    filtered_config["bias"] = True
    filtered_config["use_regression"] = True

    config = TwoLayerMechanismConfig(**filtered_config)
    model = TwoLayerMechanismModel(config)

    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, config


def get_attention_weights(model, tokens, device="cuda"):
    """Extract attention weights from both blocks."""
    config = model.config
    B, T = tokens.shape
    D = config.n_embd
    n_head = config.n_head
    head_dim = D // n_head

    with torch.no_grad():
        e = model.wte(tokens)

        # Block 1
        ln1_out = model.block1.ln_1(e)
        attn1 = model.block1.attn

        qkv1 = attn1.c_attn(ln1_out)
        q1, k1, v1 = qkv1.split(D, dim=2)

        q1 = q1.view(B, T, n_head, head_dim).transpose(1, 2)
        k1 = k1.view(B, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B, T, n_head, head_dim).transpose(1, 2)

        scores1 = (q1 @ k1.transpose(-2, -1)) / np.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores1 = scores1.masked_fill(causal_mask, float("-inf"))
        attn_weights1 = F.softmax(scores1, dim=-1)

        # Block 1 output
        attn_out1 = (attn_weights1 @ v1).transpose(1, 2).reshape(B, T, D)
        attn_out1 = attn1.c_proj(attn_out1)
        r1_attn = e + attn_out1
        ln2_out_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_out_b1)
        r1 = r1_attn + mlp_out1

        # Block 2
        ln1_out_b2 = model.block2.ln_1(r1)
        attn2 = model.block2.attn

        qkv2 = attn2.c_attn(ln1_out_b2)
        q2, k2, v2 = qkv2.split(D, dim=2)

        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        scores2 = (q2 @ k2.transpose(-2, -1)) / np.sqrt(head_dim)
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))
        attn_weights2 = F.softmax(scores2, dim=-1)

    return attn_weights1, attn_weights2


def strip_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_single_attention_head(
    ax,
    weights,
    title,
    show_xlabel=True,
    show_ylabel=True,
    show_colorbar=False,
    vmin=0,
    vmax=None,
):
    """Plot a single attention head with ICML styling."""
    if vmax is None:
        vmax = weights.max()

    im = ax.imshow(
        weights,
        cmap=ATTENTION_CMAP,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=10, pad=3)

    if show_xlabel:
        ax.set_xlabel("Key Position", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Query Position", fontsize=9)

    strip_axes(ax)

    return im


def plot_attention_maps_icml(attn_weights, block_name, regime, sample_idx=0):
    """Plot attention maps for all heads - ICML quality."""
    n_head = attn_weights.shape[1]
    T = attn_weights.shape[2]

    n_cols = 4
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 5.5))

    weights = attn_weights[sample_idx].cpu().numpy()

    # Find global max for consistent colorbar
    vmax = weights.max()

    for h in range(n_head):
        row, col = h // n_cols, h % n_cols
        ax = axes[row, col]

        show_xlabel = row == n_rows - 1
        show_ylabel = col == 0

        im = plot_single_attention_head(
            ax,
            weights[h],
            f"Head {h}",
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            vmin=0,
            vmax=vmax,
        )

    # Add single colorbar on the right
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention Weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"{regime}: {block_name} Attention Patterns", fontsize=12, y=0.98)

    return fig


def plot_attention_summary_icml(attn_weights1, attn_weights2, regime, sample_idx=0):
    """Create a summary visualization - ICML quality."""
    n_head = attn_weights1.shape[1]
    T = attn_weights1.shape[2]

    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.35,
        wspace=0.35,
        left=0.08,
        right=0.92,
        top=0.88,
        bottom=0.1,
    )

    weights1 = attn_weights1[sample_idx].cpu().numpy()
    weights2 = attn_weights2[sample_idx].cpu().numpy()

    # Row 1: Average attention patterns
    ax1 = fig.add_subplot(gs[0, 0])
    avg1 = weights1.mean(axis=0)
    im1 = ax1.imshow(avg1, cmap=ATTENTION_CMAP, aspect="auto", vmin=0)
    ax1.set_title("Block 1 (Avg)", fontsize=10)
    ax1.set_xlabel("Key Position", fontsize=9)
    ax1.set_ylabel("Query Position", fontsize=9)
    strip_axes(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    avg2 = weights2.mean(axis=0)
    im2 = ax2.imshow(avg2, cmap=ATTENTION_CMAP, aspect="auto", vmin=0)
    ax2.set_title("Block 2 (Avg)", fontsize=10)
    ax2.set_xlabel("Key Position", fontsize=9)
    strip_axes(ax2)

    # Add colorbar for heatmaps
    cbar_ax1 = fig.add_axes([0.64, 0.58, 0.01, 0.25])
    cbar1 = fig.colorbar(im2, cax=cbar_ax1)
    cbar1.ax.tick_params(labelsize=7)

    # Row 1, Col 3: BOS attention comparison
    ax3 = fig.add_subplot(gs[0, 2])
    bos_attn1 = weights1[:, :, 0].mean(axis=1)
    bos_attn2 = weights2[:, :, 0].mean(axis=1)

    x = np.arange(n_head)
    width = 0.35
    bars1 = ax3.bar(
        x - width / 2, bos_attn1, width, label="Block 1", color="#1f77b4", alpha=0.8
    )
    bars2 = ax3.bar(
        x + width / 2, bos_attn2, width, label="Block 2", color="#ff7f0e", alpha=0.8
    )
    ax3.set_xlabel("Head", fontsize=9)
    ax3.set_ylabel("Attention to Pos 0", fontsize=9)
    ax3.set_title("BOS Attention", fontsize=10)
    ax3.legend(
        loc="upper left", fontsize=7, frameon=True, fancybox=False, edgecolor="#cccccc"
    )
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels(x[::2])
    ax3.set_ylim(0, 1.05)

    # Row 2: Attention from specific query positions
    query_positions = [32, 64, 96]
    colors = ["#2ca02c", "#d62728", "#9467bd"]

    ax4 = fig.add_subplot(gs[1, 0])
    for i, qpos in enumerate(query_positions):
        attn_at_qpos = weights1[:, qpos, : qpos + 1].mean(axis=0)
        ax4.plot(
            range(qpos + 1),
            attn_at_qpos,
            color=colors[i],
            label=f"q={qpos}",
            alpha=0.8,
            linewidth=1.2,
        )
    ax4.set_xlabel("Key Position", fontsize=9)
    ax4.set_ylabel("Attention Weight", fontsize=9)
    ax4.set_title("Block 1 Attn Pattern", fontsize=10)
    ax4.legend(
        loc="upper left",
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )
    ax4.set_xlim(0, 100)

    ax5 = fig.add_subplot(gs[1, 1])
    for i, qpos in enumerate(query_positions):
        attn_at_qpos = weights2[:, qpos, : qpos + 1].mean(axis=0)
        ax5.plot(
            range(qpos + 1),
            attn_at_qpos,
            color=colors[i],
            label=f"q={qpos}",
            alpha=0.8,
            linewidth=1.2,
        )
    ax5.set_xlabel("Key Position", fontsize=9)
    ax5.set_ylabel("Attention Weight", fontsize=9)
    ax5.set_title("Block 2 Attn Pattern", fontsize=10)
    ax5.legend(
        loc="upper left",
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )
    ax5.set_xlim(0, 100)

    # Row 2, Col 3: Self-attention comparison
    ax6 = fig.add_subplot(gs[1, 2])
    self_attn1 = np.array([np.diag(weights1[h]).mean() for h in range(n_head)])
    self_attn2 = np.array([np.diag(weights2[h]).mean() for h in range(n_head)])

    bars1 = ax6.bar(
        x - width / 2, self_attn1, width, label="Block 1", color="#1f77b4", alpha=0.8
    )
    bars2 = ax6.bar(
        x + width / 2, self_attn2, width, label="Block 2", color="#ff7f0e", alpha=0.8
    )
    ax6.set_xlabel("Head", fontsize=9)
    ax6.set_ylabel("Self-Attention", fontsize=9)
    ax6.set_title("Diagonal Attention", fontsize=10)
    ax6.legend(
        loc="upper left",
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )
    ax6.set_xticks(x[::2])
    ax6.set_xticklabels(x[::2])

    fig.suptitle(f"{regime}: Attention Pattern Summary", fontsize=12, y=0.98)

    return fig


def plot_r2_mechanism_icml(model, tokens, device="cuda"):
    """ICML-quality R2 mechanism visualization."""
    config = model.config
    B, T = tokens.shape
    D = config.n_embd
    n_head = config.n_head
    head_dim = D // n_head

    positions = torch.arange(T, device=device).float()

    with torch.no_grad():
        # Full forward pass
        e = model.wte(tokens)

        # Block 1
        ln1_out = model.block1.ln_1(e)
        attn1 = model.block1.attn
        qkv1 = attn1.c_attn(ln1_out)
        q1, k1, v1 = qkv1.split(D, dim=2)
        q1 = q1.view(B, T, n_head, head_dim).transpose(1, 2)
        k1 = k1.view(B, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B, T, n_head, head_dim).transpose(1, 2)
        scores1 = (q1 @ k1.transpose(-2, -1)) / np.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores1 = scores1.masked_fill(causal_mask, float("-inf"))
        attn_weights1 = F.softmax(scores1, dim=-1)
        attn_out1 = (attn_weights1 @ v1).transpose(1, 2).reshape(B, T, D)
        attn_out1 = attn1.c_proj(attn_out1)
        r1_attn = e + attn_out1
        ln2_out_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_out_b1)
        r1 = r1_attn + mlp_out1

        # Block 2
        ln1_out_b2 = model.block2.ln_1(r1)
        attn2 = model.block2.attn
        qkv2 = attn2.c_attn(ln1_out_b2)
        q2, k2, v2 = qkv2.split(D, dim=2)
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)
        scores2 = (q2 @ k2.transpose(-2, -1)) / np.sqrt(head_dim)
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))
        attn_weights2 = F.softmax(scores2, dim=-1)
        attn_out2 = (attn_weights2 @ v2).transpose(1, 2).reshape(B, T, D)
        attn_out2_proj = attn2.c_proj(attn_out2)
        r2_attn = r1 + attn_out2_proj
        ln2_out_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_out_b2)
        r2 = r2_attn + mlp_out2
        final = model.ln_f(r2)

        pred = model.pos_head(final).squeeze(-1)

    fig = plt.figure(figsize=(7.5, 8))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.4,
        wspace=0.35,
        left=0.1,
        right=0.95,
        top=0.92,
        bottom=0.08,
    )

    sample_idx = 0
    pos_np = positions.cpu().numpy()

    # Colors for consistent styling
    colors = {
        "embed": "#1f77b4",
        "block1": "#ff7f0e",
        "block2_ln": "#2ca02c",
        "final": "#d62728",
        "pred": "#9467bd",
    }

    # Panel A: Activation norms through layers
    ax1 = fig.add_subplot(gs[0, 0])

    e_norms = e[sample_idx].norm(dim=-1).cpu().numpy()
    r1_norms = r1[sample_idx].norm(dim=-1).cpu().numpy()
    ln1_b2_norms = ln1_out_b2[sample_idx].norm(dim=-1).cpu().numpy()
    final_norms = final[sample_idx].norm(dim=-1).cpu().numpy()

    r_e = np.corrcoef(pos_np, e_norms)[0, 1]
    r_r1 = np.corrcoef(pos_np, r1_norms)[0, 1]
    r_ln = np.corrcoef(pos_np, ln1_b2_norms)[0, 1]
    r_f = np.corrcoef(pos_np, final_norms)[0, 1]

    ax1.plot(
        pos_np,
        e_norms,
        color=colors["embed"],
        label=f"Embed (r={r_e:.2f})",
        linewidth=1.5,
        alpha=0.8,
    )
    ax1.plot(
        pos_np,
        r1_norms,
        color=colors["block1"],
        label=f"Block1 (r={r_r1:.2f})",
        linewidth=1.5,
        alpha=0.8,
    )
    ax1.plot(
        pos_np,
        ln1_b2_norms,
        color=colors["block2_ln"],
        label=f"Block2 LN (r={r_ln:.2f})",
        linewidth=1.5,
        alpha=0.8,
    )
    ax1.plot(
        pos_np,
        final_norms,
        color=colors["final"],
        label=f"Final (r={r_f:.2f})",
        linewidth=1.5,
        alpha=0.8,
    )

    ax1.set_xlabel("Position", fontsize=10)
    ax1.set_ylabel("Activation Norm", fontsize=10)
    ax1.set_title("(A) Norm vs Position", fontsize=11)
    ax1.legend(
        loc="upper right",
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        ncol=2,
    )
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # Panel B: Predictions vs actual
    ax2 = fig.add_subplot(gs[0, 1])
    pred_np = pred[sample_idx].cpu().numpy()
    r2_score = np.corrcoef(pos_np, pred_np)[0, 1] ** 2

    ax2.scatter(
        pos_np, pred_np, alpha=0.5, s=15, color=colors["pred"], edgecolors="none"
    )
    ax2.plot([0, T], [0, T], "k--", linewidth=1, alpha=0.7, label="Perfect")
    ax2.set_xlabel("Actual Position", fontsize=10)
    ax2.set_ylabel("Predicted Position", fontsize=10)
    ax2.set_title(f"(B) Prediction Quality (R²={r2_score:.3f})", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.set_xlim(-5, T + 5)
    ax2.set_ylim(-5, T + 5)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    # Panel C: Block 2 attention from query position 64
    ax3 = fig.add_subplot(gs[1, 0])
    query_pos = 64

    # Use a colormap for heads - only plot a few representative heads
    representative_heads = [0, 3, 6, 9, 11]
    head_colors = plt.cm.tab10(np.linspace(0, 1, len(representative_heads)))

    for i, h in enumerate(representative_heads):
        weights_at_pos = (
            attn_weights2[sample_idx, h, query_pos, : query_pos + 1].cpu().numpy()
        )
        ax3.plot(
            range(query_pos + 1),
            weights_at_pos,
            color=head_colors[i],
            alpha=0.8,
            linewidth=1.5,
            label=f"H{h}",
        )

    ax3.set_xlabel("Key Position", fontsize=10)
    ax3.set_ylabel("Attention Weight", fontsize=10)
    ax3.set_title(f"(C) Block 2 Attn (query={query_pos})", fontsize=11)
    ax3.legend(
        loc="upper left",
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        ncol=1,
    )
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    ax3.set_xlim(0, query_pos)

    # Panel D: Per-dimension correlation histogram
    ax4 = fig.add_subplot(gs[1, 1])

    attn_out2_cpu = attn_out2_proj[sample_idx].cpu()
    r1_cpu = r1[sample_idx].cpu()

    r1_corrs = []
    attn2_corrs = []
    for d in range(D):
        r1_corr = torch.corrcoef(torch.stack([r1_cpu[:, d], positions.cpu()]))[
            0, 1
        ].item()
        attn2_corr = torch.corrcoef(
            torch.stack([attn_out2_cpu[:, d], positions.cpu()])
        )[0, 1].item()
        if not np.isnan(r1_corr):
            r1_corrs.append(r1_corr)
        if not np.isnan(attn2_corr):
            attn2_corrs.append(attn2_corr)

    ax4.hist(
        r1_corrs,
        bins=40,
        alpha=0.6,
        color=colors["block1"],
        label=f"Block1 (|r|={max(np.abs(r1_corrs)):.2f})",
        density=True,
    )
    ax4.hist(
        attn2_corrs,
        bins=40,
        alpha=0.6,
        color=colors["block2_ln"],
        label=f"Attn2 (|r|={max(np.abs(attn2_corrs)):.2f})",
        density=True,
    )
    ax4.axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
    ax4.set_xlabel("Correlation with Position", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.set_title("(D) Per-Dimension Correlations", fontsize=11)
    ax4.legend(
        loc="upper left", fontsize=7, frameon=True, fancybox=False, edgecolor="#cccccc"
    )

    # Panel E: Top correlated dimensions
    ax5 = fig.add_subplot(gs[2, 0])

    attn2_dim_corrs = [
        (
            d,
            torch.corrcoef(torch.stack([attn_out2_cpu[:, d], positions.cpu()]))[
                0, 1
            ].item(),
        )
        for d in range(D)
    ]
    attn2_dim_corrs = [(d, c) for d, c in attn2_dim_corrs if not np.isnan(c)]
    attn2_dim_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    top_colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
    for i, (d, corr) in enumerate(attn2_dim_corrs[:5]):
        vals = attn_out2_cpu[:, d].numpy()
        vals_norm = (vals - vals.mean()) / vals.std()
        ax5.plot(
            pos_np,
            vals_norm,
            color=top_colors[i],
            label=f"d={d} (r={corr:.3f})",
            linewidth=1.2,
            alpha=0.8,
        )

    ax5.set_xlabel("Position", fontsize=10)
    ax5.set_ylabel("Normalized Activation", fontsize=10)
    ax5.set_title("(E) Top Position-Correlated Dims", fontsize=11)
    ax5.legend(
        loc="lower left", fontsize=7, frameon=True, fancybox=False, edgecolor="#cccccc"
    )
    ax5.grid(True, alpha=0.3, linewidth=0.5)

    # Panel F: Mechanism summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    mechanism_text = """
    R2 Mechanism Summary
    ====================
    
    1. Block 1 (frozen, random):
       Causal averaging creates
       variance decay: var ~ 1/(i+1)
    
    2. LayerNorm amplifies signal:
       Norm-position corr: r = -0.87
    
    3. Block 2 Attn (trained):
       Projects signal to dims with
       |r| > 0.99 position correlation
    
    4. Linear head extracts position:
       Final R^2 = 0.95
    """

    ax6.text(
        0.05,
        0.5,
        mechanism_text,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="center",
        transform=ax6.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f0f0f0",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    fig.suptitle(
        "R2 Mechanism: Position Encoding via Variance Decay", fontsize=12, y=0.98
    )

    return fig


def plot_r0_vs_r2_comparison_icml(model_r0, model_r2, tokens, device="cuda"):
    """ICML-quality comparison of R0 (BOS) vs R2 (Variance) mechanisms."""

    _, attn_r0_b2 = get_attention_weights(model_r0, tokens, device)
    _, attn_r2_b2 = get_attention_weights(model_r2, tokens, device)

    n_head = 12

    fig = plt.figure(figsize=(7.5, 6))
    gs = GridSpec(
        4,
        6,
        figure=fig,
        hspace=0.25,
        wspace=0.15,
        left=0.06,
        right=0.88,
        top=0.90,
        bottom=0.08,
    )

    weights_r0 = attn_r0_b2[0].cpu().numpy()
    weights_r2 = attn_r2_b2[0].cpu().numpy()

    vmax = max(weights_r0.max(), weights_r2.max())

    # R0 Block 2 heads (rows 0-1)
    for h in range(12):
        row = h // 6
        col = h % 6
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            weights_r0[h], cmap=ATTENTION_CMAP, aspect="auto", vmin=0, vmax=vmax
        )
        ax.set_title(f"H{h}", fontsize=8, pad=2)

        if col == 0 and row == 0:
            ax.set_ylabel("R0 (BOS)", fontsize=9)
        elif col == 0 and row == 1:
            ax.set_ylabel("", fontsize=9)

        strip_axes(ax)

    # R2 Block 2 heads (rows 2-3)
    for h in range(12):
        row = 2 + h // 6
        col = h % 6
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            weights_r2[h], cmap=ATTENTION_CMAP, aspect="auto", vmin=0, vmax=vmax
        )
        ax.set_title(f"H{h}", fontsize=8, pad=2)

        if col == 0 and row == 2:
            ax.set_ylabel("R2 (Var)", fontsize=9)
        elif col == 0 and row == 3:
            ax.set_ylabel("", fontsize=9)

        strip_axes(ax)

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Add row labels
    fig.text(
        0.02,
        0.78,
        "R0\n(BOS)",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )
    fig.text(
        0.02,
        0.35,
        "R2\n(Var)",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )

    fig.suptitle(
        "Block 2 Attention: R0 (BOS Reference) vs R2 (Variance Reading)",
        fontsize=11,
        y=0.96,
    )

    return fig


def plot_bos_heads_highlight_icml(model, tokens, regime, device="cuda"):
    """Highlight BOS heads in R0 model - ICML quality."""

    _, attn_weights2 = get_attention_weights(model, tokens, device)
    weights = attn_weights2[0].cpu().numpy()
    n_head = weights.shape[0]

    # Identify BOS heads (>50% attention to position 0 on average)
    bos_scores = weights[:, :, 0].mean(axis=1)  # Average over queries
    bos_heads = np.where(bos_scores > 0.5)[0]

    fig = plt.figure(figsize=(7, 3.5))

    if len(bos_heads) >= 2:
        gs = GridSpec(
            1, 3, figure=fig, wspace=0.3, left=0.08, right=0.92, top=0.85, bottom=0.15
        )

        # Plot top 2 BOS heads
        for i, h in enumerate(bos_heads[:2]):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(
                weights[h], cmap=ATTENTION_CMAP, aspect="auto", vmin=0, vmax=1
            )
            ax.plot(
                0,
                -3,
                marker="v",
                color="#C44E52",
                markersize=5,
                clip_on=False,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )
            ax.set_title(f"Head {h} (BOS score: {bos_scores[h]:.2f})", fontsize=10)
            strip_axes(ax)

        # Bar chart of BOS scores
        ax3 = fig.add_subplot(gs[0, 2])
        colors = ["#d62728" if h in bos_heads else "#1f77b4" for h in range(n_head)]
        ax3.bar(range(n_head), bos_scores, color=colors, alpha=0.8)
        ax3.set_xlabel("Head", fontsize=9)
        ax3.set_ylabel("BOS Score", fontsize=9)
        ax3.set_title("BOS Attention by Head", fontsize=10)
        ax3.set_xticks(range(0, n_head, 2))

    else:
        # No clear BOS heads - show top 2 by BOS score anyway
        gs = GridSpec(
            1, 3, figure=fig, wspace=0.3, left=0.08, right=0.92, top=0.85, bottom=0.15
        )

        top2 = np.argsort(bos_scores)[-2:][::-1]
        for i, h in enumerate(top2):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(
                weights[h], cmap=ATTENTION_CMAP, aspect="auto", vmin=0, vmax=1
            )
            ax.plot(
                0,
                -3,
                marker="v",
                color="#C44E52",
                markersize=5,
                clip_on=False,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )
            ax.set_title(f"Head {h} (BOS: {bos_scores[h]:.2f})", fontsize=10)
            strip_axes(ax)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(range(n_head), bos_scores, color="#1f77b4", alpha=0.8)
        ax3.set_xlabel("Head", fontsize=9)
        ax3.set_ylabel("BOS Score", fontsize=9)
        ax3.set_title("BOS Attention by Head", fontsize=10)

    fig.suptitle(f"{regime}: BOS Head Analysis", fontsize=11, y=0.98)

    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    data_path = "data/openwebtext/train.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    batch_size = 8
    block_size = 128

    torch.manual_seed(42)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    tokens = torch.stack(
        [
            torch.from_numpy(
                np.concatenate(
                    [[BOS_TOKEN_ID], data[i : i + block_size - 1].astype(np.int64)]
                )
            )
            for i in ix
        ]
    ).to(device)

    regimes = ["R0", "R1", "R2", "R3", "R4_linear", "R4_mlp"]

    output_dir = "out-2layer-mechanism/attention_maps_icml"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B
    wandb.init(
        project="nope-2layer-mechanism",
        name="attention-maps-icml-quality",
        config={
            "batch_size": batch_size,
            "block_size": block_size,
            "regimes": regimes,
            "style": "ICML",
        },
    )

    for regime in regimes:
        print(f"\n{'=' * 60}")
        print(f"Processing {regime}")
        print(f"{'=' * 60}")

        model, config = load_model(regime, device)
        if model is None:
            print(f"Skipping {regime} - model not found")
            continue

        attn_weights1, attn_weights2 = get_attention_weights(model, tokens, device)

        # Block 1 attention maps
        fig1 = plot_attention_maps_icml(attn_weights1, "Block 1", regime)
        fig1_path = f"{output_dir}/{regime}_block1_attention.png"
        fig1.savefig(fig1_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig1.savefig(
            fig1_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white"
        )
        wandb.log({f"{regime}/block1_attention": wandb.Image(fig1)})
        plt.close(fig1)
        print(f"  Saved Block 1 attention maps")

        # Block 2 attention maps
        fig2 = plot_attention_maps_icml(attn_weights2, "Block 2", regime)
        fig2_path = f"{output_dir}/{regime}_block2_attention.png"
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig2.savefig(
            fig2_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white"
        )
        wandb.log({f"{regime}/block2_attention": wandb.Image(fig2)})
        plt.close(fig2)
        print(f"  Saved Block 2 attention maps")

        # Attention summary
        fig3 = plot_attention_summary_icml(attn_weights1, attn_weights2, regime)
        fig3_path = f"{output_dir}/{regime}_attention_summary.png"
        fig3.savefig(fig3_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig3.savefig(
            fig3_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white"
        )
        wandb.log({f"{regime}/attention_summary": wandb.Image(fig3)})
        plt.close(fig3)
        print(f"  Saved attention summary")

        # BOS head analysis for R0
        if regime == "R0":
            fig_bos = plot_bos_heads_highlight_icml(model, tokens, regime, device)
            fig_bos_path = f"{output_dir}/{regime}_bos_heads.png"
            fig_bos.savefig(
                fig_bos_path, dpi=300, bbox_inches="tight", facecolor="white"
            )
            fig_bos.savefig(
                fig_bos_path.replace(".png", ".pdf"),
                bbox_inches="tight",
                facecolor="white",
            )
            wandb.log({f"{regime}/bos_heads": wandb.Image(fig_bos)})
            plt.close(fig_bos)
            print(f"  Saved BOS head analysis")

        # R2 mechanism visualization
        if regime == "R2":
            print(f"\n  Creating R2 mechanism visualization...")
            fig4 = plot_r2_mechanism_icml(model, tokens, device)
            fig4_path = f"{output_dir}/R2_mechanism.png"
            fig4.savefig(fig4_path, dpi=300, bbox_inches="tight", facecolor="white")
            fig4.savefig(
                fig4_path.replace(".png", ".pdf"),
                bbox_inches="tight",
                facecolor="white",
            )
            wandb.log({"R2/mechanism": wandb.Image(fig4)})
            plt.close(fig4)
            print(f"  Saved R2 mechanism visualization")

    # R0 vs R2 comparison
    print(f"\n{'=' * 60}")
    print("Creating R0 vs R2 comparison")
    print(f"{'=' * 60}")

    model_r0, _ = load_model("R0", device)
    model_r2, _ = load_model("R2", device)

    if model_r0 is not None and model_r2 is not None:
        fig5 = plot_r0_vs_r2_comparison_icml(model_r0, model_r2, tokens, device)
        fig5_path = f"{output_dir}/R0_vs_R2_comparison.png"
        fig5.savefig(fig5_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig5.savefig(
            fig5_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white"
        )
        wandb.log({"comparison/R0_vs_R2": wandb.Image(fig5)})
        plt.close(fig5)
        print(f"  Saved R0 vs R2 comparison")

        # Also create R0 mechanism visualization
        fig6 = plot_r2_mechanism_icml(model_r0, tokens, device)
        fig6_path = f"{output_dir}/R0_mechanism.png"
        fig6.savefig(fig6_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig6.savefig(
            fig6_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white"
        )
        wandb.log({"R0/mechanism": wandb.Image(fig6)})
        plt.close(fig6)
        print(f"  Saved R0 mechanism visualization")

    wandb.finish()
    print(f"\n{'=' * 60}")
    print("All ICML-quality visualizations complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
