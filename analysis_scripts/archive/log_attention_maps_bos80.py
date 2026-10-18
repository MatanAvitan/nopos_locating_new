"""
Log Attention Maps to WandB for BOS 80 Experiment

Loads the trained BOS 80 model and logs attention maps to the WandB run.

Usage:
    python analysis_scripts/log_attention_maps_bos80.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

# Add nanoGPT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)

BOS_TOKEN_ID = 50256
BOS_POSITION = 80


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = TwoLayerMechanismConfig(
        block_size=128,
        vocab_size=50304,
        n_embd=768,
        n_head=12,
        dropout=0.0,
        norm_type="layernorm",
        use_regression=True,
    )

    model = TwoLayerMechanismModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, checkpoint.get("config", {})


def load_data(data_path: str):
    """Load memory-mapped data."""
    return np.memmap(data_path, dtype=np.uint16, mode="r")


def get_batch_with_bos(
    data, batch_size: int, block_size: int, bos_position: int, device: str
):
    """Get a batch with BOS token at specified position."""
    tokens_needed = block_size - 1
    ix = np.random.randint(0, len(data) - tokens_needed, size=batch_size)

    sequences = []
    for i in ix:
        before_bos = data[i : i + bos_position].astype(np.int64)
        after_bos = data[i + bos_position : i + tokens_needed].astype(np.int64)
        seq = np.concatenate([before_bos, [BOS_TOKEN_ID], after_bos])
        sequences.append(torch.from_numpy(seq))

    return torch.stack(sequences).to(device)


def get_attention_weights(model, x, device):
    """Forward pass and capture attention weights from both blocks."""
    with torch.no_grad():
        tok_emb = model.wte(x)
        h = model.drop(tok_emb)

        # Block 1
        block1_out = model.block1(h, capture_taps=True)
        attn1_weights = model.block1.attn.last_attention_weights  # [B, n_head, T, T]

        # Block 2
        block2_out = model.block2(block1_out, capture_taps=True)
        attn2_weights = model.block2.attn.last_attention_weights  # [B, n_head, T, T]

    return attn1_weights, attn2_weights


def create_attention_heatmap(
    attn_weights,
    block_name: str,
    head_idx: int,
    bos_position: int = 80,
    block_size: int = 128,
):
    """Create attention heatmap for a single head."""
    # Average over batch
    attn = attn_weights[:, head_idx].mean(dim=0).cpu().numpy()  # [T, T]

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0)

    # Add BOS position markers on the margins (triangles pointing inward)
    # Top margin - mark key position
    ax.annotate(
        "",
        xy=(bos_position, -1),
        xytext=(bos_position, -6),
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2),
        annotation_clip=False,
    )
    ax.text(
        bos_position,
        -8,
        "BOS",
        ha="center",
        va="top",
        fontsize=8,
        color="#C44E52",
        fontweight="bold",
        clip_on=False,
    )

    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"{block_name} Head {head_idx}", fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight")

    # Expand axis limits to show annotations
    ax.set_xlim(-0.5, block_size - 0.5)
    ax.set_ylim(block_size - 0.5, -0.5)

    plt.tight_layout()
    return fig


def create_bos_attention_summary(
    attn_weights, block_name: str, bos_position: int = 80, n_heads: int = 12
):
    """Create summary of how much each head attends to BOS position."""
    # Average over batch: [n_head, T, T]
    attn = attn_weights.mean(dim=0).cpu().numpy()

    # For each head, compute mean attention to BOS position (column bos_position)
    # Only for queries after BOS position (they can see it)
    attn_to_bos = []
    for h in range(n_heads):
        # Queries from positions > bos_position attending to key at bos_position
        attn_to_bos.append(attn[h, bos_position + 1 :, bos_position].mean())

    fig, ax = plt.subplots(figsize=(5, 3))

    bars = ax.bar(
        range(n_heads), attn_to_bos, color="#4C72B0", edgecolor="black", linewidth=0.5
    )

    # Highlight heads with high BOS attention
    for i, val in enumerate(attn_to_bos):
        if val > 0.1:  # Threshold for "BOS head"
            bars[i].set_color("#C44E52")

    ax.set_xlabel("Head Index")
    ax.set_ylabel("Mean Attention to BOS")
    ax.set_title(
        f"{block_name}: Attention to BOS Position ({bos_position})", fontweight="bold"
    )
    ax.set_xticks(range(n_heads))

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, attn_to_bos


def create_full_attention_grid(
    attn_weights,
    block_name: str,
    bos_position: int = 80,
    pos0_position: int = 0,
    pos0_threshold: float = 0.1,
    bos_threshold: float = 0.1,
):
    """Create a grid of all attention heads with head-specific anchor markers."""
    n_heads = attn_weights.shape[1]
    block_size = attn_weights.shape[2]
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    axes = axes.flatten()

    attn = attn_weights.mean(dim=0).cpu().numpy()  # [n_head, T, T]

    attn_to_pos0 = attn[:, :, pos0_position].mean(axis=1)
    if bos_position + 1 < block_size:
        attn_to_bos = attn[:, bos_position + 1 :, bos_position].mean(axis=1)
    else:
        attn_to_bos = np.zeros(n_heads)

    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap="Blues", aspect="auto", vmin=0)

        marker_pos = None
        if attn_to_pos0[h] >= pos0_threshold and attn_to_pos0[h] >= attn_to_bos[h]:
            marker_pos = pos0_position
        elif attn_to_bos[h] >= bos_threshold:
            marker_pos = bos_position

        if marker_pos is not None:
            ax.plot(
                marker_pos,
                -3,
                marker="v",
                color="#C44E52",
                markersize=5,
                clip_on=False,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )
            ax.plot(
                -3,
                marker_pos,
                marker=">",
                color="#C44E52",
                markersize=5,
                clip_on=False,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )

        ax.set_title(f"Head {h}", fontsize=9)
        ax.set_xticks([0, bos_position, block_size - 1])
        ax.set_yticks([0, bos_position, block_size - 1])
        ax.tick_params(labelsize=7)

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(
        f"{block_name} Attention Maps (BOS @ pos {bos_position})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def main():
    # Paths
    checkpoint_path = "/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism-bos80/R0/best_ckpt.pt"
    data_path = (
        "/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext/val.bin"
    )
    output_dir = (
        "/home/nlp/matan_avitan/git/nopos_locating_new/results/attention_maps_bos80"
    )
    overleaf_plot_dir = (
        "/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos_icml_2026/plots"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize WandB (resume the run or create new)
    wandb.init(
        project="nope-2layer-mechanism-bos80",
        name="R0_bos80_attention_maps",
        job_type="analysis",
        tags=["attention_maps", "bos80", "analysis"],
    )

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)

    # Load data
    data = load_data(data_path)
    print(f"Loaded {len(data):,} tokens")

    # Get batch with BOS at position 80
    np.random.seed(42)
    x = get_batch_with_bos(
        data, batch_size=64, block_size=128, bos_position=BOS_POSITION, device=device
    )

    # Get attention weights
    attn1, attn2 = get_attention_weights(model, x, device)

    print(f"Block 1 attention shape: {attn1.shape}")
    print(f"Block 2 attention shape: {attn2.shape}")

    # Create and log attention maps
    print("\nLogging attention maps to WandB...")

    # Block 1 grid
    fig1_grid = create_full_attention_grid(attn1, "Block 1", BOS_POSITION)
    wandb.log({"attention/block1_all_heads": wandb.Image(fig1_grid)})
    fig1_grid.savefig(
        os.path.join(output_dir, "block1_attention_grid.png"),
        dpi=150,
        bbox_inches="tight",
    )
    fig1_grid.savefig(
        os.path.join(output_dir, "block1_attention_grid.pdf"), bbox_inches="tight"
    )
    plt.close(fig1_grid)

    # Block 2 grid
    fig2_grid = create_full_attention_grid(attn2, "Block 2", BOS_POSITION)
    wandb.log({"attention/block2_all_heads": wandb.Image(fig2_grid)})
    fig2_grid.savefig(
        os.path.join(output_dir, "block2_attention_grid.png"),
        dpi=150,
        bbox_inches="tight",
    )
    fig2_grid.savefig(
        os.path.join(output_dir, "block2_attention_grid.pdf"), bbox_inches="tight"
    )
    fig2_grid.savefig(
        os.path.join(overleaf_plot_dir, "bos80_attention_grid.png"),
        dpi=150,
        bbox_inches="tight",
    )
    fig2_grid.savefig(
        os.path.join(overleaf_plot_dir, "bos80_attention_grid.pdf"), bbox_inches="tight"
    )
    plt.close(fig2_grid)

    # BOS attention summary
    fig1_bos, attn1_to_bos = create_bos_attention_summary(
        attn1, "Block 1", BOS_POSITION
    )
    wandb.log({"attention/block1_bos_attention": wandb.Image(fig1_bos)})
    fig1_bos.savefig(
        os.path.join(output_dir, "block1_bos_attention.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig1_bos)

    fig2_bos, attn2_to_bos = create_bos_attention_summary(
        attn2, "Block 2", BOS_POSITION
    )
    wandb.log({"attention/block2_bos_attention": wandb.Image(fig2_bos)})
    fig2_bos.savefig(
        os.path.join(output_dir, "block2_bos_attention.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig2_bos)

    # Log individual heads with high BOS attention
    for h, val in enumerate(attn2_to_bos):
        if val > 0.05:
            fig_head = create_attention_heatmap(attn2, "Block 2", h, BOS_POSITION)
            wandb.log({f"attention/block2_head{h}_detail": wandb.Image(fig_head)})
            fig_head.savefig(
                os.path.join(output_dir, f"block2_head{h}_detail.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig_head)

    # Log summary statistics
    wandb.run.summary["block1_max_bos_attention"] = max(attn1_to_bos)
    wandb.run.summary["block2_max_bos_attention"] = max(attn2_to_bos)
    wandb.run.summary["block1_bos_heads"] = [
        i for i, v in enumerate(attn1_to_bos) if v > 0.1
    ]
    wandb.run.summary["block2_bos_heads"] = [
        i for i, v in enumerate(attn2_to_bos) if v > 0.1
    ]

    print("\n" + "=" * 50)
    print("BOS Attention Analysis (BOS @ position 80)")
    print("=" * 50)
    print(f"Block 1 - Max BOS attention: {max(attn1_to_bos):.4f}")
    print(
        f"Block 1 - BOS heads (>0.1): {[i for i, v in enumerate(attn1_to_bos) if v > 0.1]}"
    )
    print(f"Block 2 - Max BOS attention: {max(attn2_to_bos):.4f}")
    print(
        f"Block 2 - BOS heads (>0.1): {[i for i, v in enumerate(attn2_to_bos) if v > 0.1]}"
    )
    print("=" * 50)

    wandb.finish()
    print(f"\nAttention maps saved to: {output_dir}")
    print("WandB run complete!")


if __name__ == "__main__":
    main()
