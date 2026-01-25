"""
Improved Attention Map Visualizations for ICML 2026 Paper

Generates:
1. Improved BOS heads figure with viridis colormap
2. Appendix attention maps (one page per regime)

Uses viridis for better contrast and colorblind-friendliness.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Colors (colorblind-friendly)
COLOR_BOS = '#D55E00'     # Vermillion for BOS heads
COLOR_NON_BOS = '#0072B2'  # Blue for non-BOS heads

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    config = TwoLayerMechanismConfig(**model_args)
    model = TwoLayerMechanismModel(config)

    state_dict = checkpoint["model"]
    unwrapped = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped)
    model.to(device)
    model.eval()

    return model, config


def load_owt_data(data_dir: str = "nanoGPT/data/openwebtext"):
    """Load OpenWebText validation data."""
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str, seed: int = 42):
    """Get a batch of sequences with fixed seed for reproducibility."""
    np.random.seed(seed)
    ix = np.random.randint(0, len(data) - block_size, size=batch_size)
    x = torch.stack([
        torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix
    ])
    return x.to(device)


def get_attention_weights(model: TwoLayerMechanismModel, tokens: torch.Tensor, device: str = "cuda"):
    """Extract attention weights from both blocks."""
    B, T = tokens.shape
    D = model.config.n_embd
    n_head = model.config.n_head
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
        scores1 = scores1.masked_fill(causal_mask, float('-inf'))
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
        scores2 = scores2.masked_fill(causal_mask, float('-inf'))
        attn_weights2 = F.softmax(scores2, dim=-1)

    return attn_weights1, attn_weights2


def plot_improved_bos_heads(model: TwoLayerMechanismModel, tokens: torch.Tensor,
                            regime: str, save_path: str, device: str = "cuda"):
    """
    Create improved BOS heads figure with viridis colormap.

    Shows:
    - Two attention heatmaps for top BOS heads (using viridis)
    - Bar chart showing BOS attention score per head
    """
    _, attn_weights2 = get_attention_weights(model, tokens, device)

    # Average over batch for smoother patterns
    weights = attn_weights2.mean(dim=0).cpu().numpy()  # [H, T, T]
    n_head = weights.shape[0]
    T = weights.shape[1]

    # Compute BOS scores (average attention to position 0)
    bos_scores = weights[:, :, 0].mean(axis=1)

    # Identify BOS heads (>50% attention to position 0)
    bos_heads = np.where(bos_scores > 0.5)[0]
    top_bos = np.argsort(bos_scores)[-2:][::-1]  # Top 2 by BOS score

    # Create figure
    fig = plt.figure(figsize=(6.5, 2.8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35, left=0.08, right=0.95, top=0.85, bottom=0.15)

    # Plot top 2 BOS heads with viridis
    for i, h in enumerate(top_bos):
        ax = fig.add_subplot(gs[0, i])

        # Use viridis colormap for better contrast
        im = ax.imshow(weights[h], cmap='viridis', aspect='auto', vmin=0, vmax=1)

        # Add title with BOS score
        is_bos = h in bos_heads
        title_color = COLOR_BOS if is_bos else 'black'
        ax.set_title(f'Head {h} (BOS: {bos_scores[h]:.2f})', fontsize=10, color=title_color)

        ax.set_xlabel('Key Position', fontsize=9)
        if i == 0:
            ax.set_ylabel('Query Position', fontsize=9)

        # Clean tick formatting
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        # Add colorbar for first subplot only
        if i == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention', fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    # Bar chart of BOS scores
    ax3 = fig.add_subplot(gs[0, 2])

    # Color bars based on BOS status
    colors = [COLOR_BOS if h in bos_heads else COLOR_NON_BOS for h in range(n_head)]
    bars = ax3.bar(range(n_head), bos_scores, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add threshold line
    ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.text(n_head - 0.5, 0.52, 'Threshold', fontsize=7, ha='right', va='bottom', color='gray')

    ax3.set_xlabel('Head', fontsize=9)
    ax3.set_ylabel('BOS Score', fontsize=9)
    ax3.set_title('BOS Attention by Head', fontsize=10)
    ax3.set_xticks(range(0, n_head, 2))
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_BOS, edgecolor='black', label='BOS head (>0.5)'),
        Patch(facecolor=COLOR_NON_BOS, edgecolor='black', label='Non-BOS head')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=7, frameon=True, framealpha=0.9)

    fig.suptitle(f'{regime}: BOS Head Analysis', fontsize=11, y=0.98)

    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved improved BOS heads figure to {save_path}")

    return {
        "bos_heads": bos_heads.tolist(),
        "bos_scores": bos_scores.tolist(),
        "n_bos_heads": len(bos_heads),
    }


def plot_appendix_attention_maps(model: TwoLayerMechanismModel, tokens: torch.Tensor,
                                  regime: str, save_path: str, device: str = "cuda"):
    """
    Create full-page attention map figure for appendix.
    Shows all 12 heads for both Block 1 and Block 2.
    """
    attn_weights1, attn_weights2 = get_attention_weights(model, tokens, device)

    # Average over batch
    weights1 = attn_weights1.mean(dim=0).cpu().numpy()  # [H, T, T]
    weights2 = attn_weights2.mean(dim=0).cpu().numpy()
    n_head = weights1.shape[0]

    # Compute BOS scores for annotations
    bos_scores = weights2[:, :, 0].mean(axis=1)

    # Create figure: 4 rows x 6 columns (2 rows per block)
    fig, axes = plt.subplots(4, 6, figsize=(10, 8))

    # Block 1 attention (rows 0-1)
    for h in range(12):
        row = h // 6
        col = h % 6
        ax = axes[row, col]

        im = ax.imshow(weights1[h], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'H{h}', fontsize=9, pad=2)

        if col == 0 and row == 0:
            ax.set_ylabel('Block 1\nQuery', fontsize=9)
        elif col == 0 and row == 1:
            ax.set_ylabel('Query', fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

    # Block 2 attention (rows 2-3)
    for h in range(12):
        row = 2 + h // 6
        col = h % 6
        ax = axes[row, col]

        im = ax.imshow(weights2[h], cmap='viridis', aspect='auto', vmin=0, vmax=1)

        # Highlight BOS heads with colored title
        is_bos = bos_scores[h] > 0.5
        title_color = COLOR_BOS if is_bos else 'black'
        bos_marker = '*' if is_bos else ''
        ax.set_title(f'H{h}{bos_marker} ({bos_scores[h]:.2f})', fontsize=8, pad=2, color=title_color)

        if col == 0 and row == 2:
            ax.set_ylabel('Block 2\nQuery', fontsize=9)
        elif col == 0 and row == 3:
            ax.set_ylabel('Query', fontsize=9)

        if row == 3:
            ax.set_xlabel('Key', fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    fig.subplots_adjust(right=0.92, hspace=0.25, wspace=0.1)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f'{regime}: Full Attention Maps (Block 1 top, Block 2 bottom)\n'
                 f'* indicates BOS head (>50% attention to position 0)', fontsize=11, y=0.98)

    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved appendix attention maps for {regime} to {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument("--checkpoint_dir", type=str, default="nanoGPT/out-2layer-mechanism")
    parser.add_argument("--save_dir", type=str, default="overleaf/nopos_icml_2026/plots")
    parser.add_argument("--batch_size", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    val_data = load_owt_data(args.data_dir)

    # Get batch (fixed seed for reproducibility)
    regimes = ["R0", "R1", "R2", "R3"]

    for regime in regimes:
        checkpoint_path = f"{args.checkpoint_dir}/{regime}/best_ckpt.pt"

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for {regime}, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {regime}")
        print(f"{'='*60}")

        model, config = load_model(checkpoint_path, DEVICE)
        block_size = config.block_size

        tokens = get_batch(val_data, args.batch_size, block_size, DEVICE, seed=42)

        # Generate improved BOS heads figure (only for R0)
        if regime == "R0":
            bos_info = plot_improved_bos_heads(
                model, tokens, regime,
                os.path.join(args.save_dir, "R0_bos_heads.pdf"),
                DEVICE
            )
            print(f"  BOS heads: {bos_info['bos_heads']}")
            print(f"  BOS scores: {[f'{s:.2f}' for s in bos_info['bos_scores']]}")

        # Generate appendix attention maps
        plot_appendix_attention_maps(
            model, tokens, regime,
            os.path.join(args.save_dir, f"appendix_attention_{regime}.pdf"),
            DEVICE
        )

    print(f"\n{'='*60}")
    print("All visualizations complete!")
    print(f"Output directory: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
