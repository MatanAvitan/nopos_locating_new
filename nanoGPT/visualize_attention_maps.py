"""
Visualize Attention Maps for All Regimes and R2 Mechanism

Generates:
1. Attention maps for all heads in Block 1 and Block 2 for each regime
2. R2 mechanism visualization showing how position is extracted

Uploads all visualizations to W&B.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wandb
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig


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
        # Embedding
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


def plot_attention_maps(attn_weights, block_name, regime, sample_idx=0):
    """Plot attention maps for all heads in a block."""
    # attn_weights: [B, n_head, T, T]
    n_head = attn_weights.shape[1]
    T = attn_weights.shape[2]

    # Create figure with subplots for all heads
    n_cols = 4
    n_rows = (n_head + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    weights = attn_weights[sample_idx].cpu().numpy()  # [n_head, T, T]

    for h in range(n_head):
        ax = axes[h]
        im = ax.imshow(weights[h], cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Head {h}", fontsize=12)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for h in range(n_head, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"{regime} - {block_name} Attention Maps", fontsize=16, y=1.02)
    plt.tight_layout()

    return fig


def plot_attention_summary(attn_weights1, attn_weights2, regime, sample_idx=0):
    """Create a summary visualization showing key attention patterns."""
    n_head = attn_weights1.shape[1]
    T = attn_weights1.shape[2]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    weights1 = attn_weights1[sample_idx].cpu().numpy()
    weights2 = attn_weights2[sample_idx].cpu().numpy()

    # Row 1: Block 1 - sample heads and average
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(weights1[0], cmap="viridis", aspect="auto")
    ax1.set_title("Block 1, Head 0")
    ax1.set_xlabel("Key")
    ax1.set_ylabel("Query")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(weights1[5], cmap="viridis", aspect="auto")
    ax2.set_title("Block 1, Head 5")
    ax2.set_xlabel("Key")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(weights1[11], cmap="viridis", aspect="auto")
    ax3.set_title("Block 1, Head 11")
    ax3.set_xlabel("Key")

    ax4 = fig.add_subplot(gs[0, 3])
    avg1 = weights1.mean(axis=0)
    ax4.imshow(avg1, cmap="viridis", aspect="auto")
    ax4.set_title("Block 1, Average")
    ax4.set_xlabel("Key")

    # Row 2: Block 2 - sample heads and average
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(weights2[0], cmap="viridis", aspect="auto")
    ax5.set_title("Block 2, Head 0")
    ax5.set_xlabel("Key")
    ax5.set_ylabel("Query")

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(weights2[5], cmap="viridis", aspect="auto")
    ax6.set_title("Block 2, Head 5")
    ax6.set_xlabel("Key")

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(weights2[11], cmap="viridis", aspect="auto")
    ax7.set_title("Block 2, Head 11")
    ax7.set_xlabel("Key")

    ax8 = fig.add_subplot(gs[1, 3])
    avg2 = weights2.mean(axis=0)
    ax8.imshow(avg2, cmap="viridis", aspect="auto")
    ax8.set_title("Block 2, Average")
    ax8.set_xlabel("Key")

    # Row 3: Attention statistics
    # BOS attention (attention to position 0)
    ax9 = fig.add_subplot(gs[2, 0:2])
    bos_attn1 = weights1[:, :, 0].mean(axis=1)  # [n_head] avg over queries
    bos_attn2 = weights2[:, :, 0].mean(axis=1)

    x = np.arange(n_head)
    width = 0.35
    ax9.bar(x - width / 2, bos_attn1, width, label="Block 1", alpha=0.8)
    ax9.bar(x + width / 2, bos_attn2, width, label="Block 2", alpha=0.8)
    ax9.set_xlabel("Head")
    ax9.set_ylabel("Avg Attention to Pos 0")
    ax9.set_title("BOS Attention by Head")
    ax9.legend()
    ax9.set_xticks(x)

    # Self attention (attention to current position)
    ax10 = fig.add_subplot(gs[2, 2:4])
    # Diagonal attention (self-attention) - average over positions
    self_attn1 = np.array([np.diag(weights1[h]).mean() for h in range(n_head)])
    self_attn2 = np.array([np.diag(weights2[h]).mean() for h in range(n_head)])

    ax10.bar(x - width / 2, self_attn1, width, label="Block 1", alpha=0.8)
    ax10.bar(x + width / 2, self_attn2, width, label="Block 2", alpha=0.8)
    ax10.set_xlabel("Head")
    ax10.set_ylabel("Avg Self-Attention")
    ax10.set_title("Self-Attention by Head")
    ax10.legend()
    ax10.set_xticks(x)

    fig.suptitle(f"{regime} - Attention Pattern Summary", fontsize=16, y=1.02)

    return fig


def plot_r2_mechanism(model, tokens, device="cuda"):
    """Create comprehensive visualization of the R2 mechanism."""
    config = model.config
    B, T = tokens.shape
    D = config.n_embd
    n_head = config.n_head
    head_dim = D // n_head

    positions = torch.arange(T, device=device).float()

    with torch.no_grad():
        # Forward pass collecting all intermediate states
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

        # Predictions
        pred = model.pos_head(final).squeeze(-1)

    # Create the visualization
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

    sample_idx = 0

    # Row 1: Position signal through layers (norms)
    ax1 = fig.add_subplot(gs[0, :2])

    e_norms = e[sample_idx].norm(dim=-1).cpu().numpy()
    r1_norms = r1[sample_idx].norm(dim=-1).cpu().numpy()
    ln1_b2_norms = ln1_out_b2[sample_idx].norm(dim=-1).cpu().numpy()
    attn_out2_norms = attn_out2_proj[sample_idx].norm(dim=-1).cpu().numpy()
    final_norms = final[sample_idx].norm(dim=-1).cpu().numpy()

    pos_np = positions.cpu().numpy()
    ax1.plot(
        pos_np,
        e_norms,
        label=f"Embedding (r={np.corrcoef(pos_np, e_norms)[0, 1]:.3f})",
        alpha=0.7,
    )
    ax1.plot(
        pos_np,
        r1_norms,
        label=f"Block1 out (r={np.corrcoef(pos_np, r1_norms)[0, 1]:.3f})",
        alpha=0.7,
    )
    ax1.plot(
        pos_np,
        ln1_b2_norms,
        label=f"Block2 LN in (r={np.corrcoef(pos_np, ln1_b2_norms)[0, 1]:.3f})",
        alpha=0.7,
    )
    ax1.plot(
        pos_np,
        final_norms,
        label=f"Final (r={np.corrcoef(pos_np, final_norms)[0, 1]:.3f})",
        alpha=0.7,
    )
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Norm")
    ax1.set_title("Activation Norms Through Layers")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Row 1: Predictions vs actual
    ax2 = fig.add_subplot(gs[0, 2:])
    pred_np = pred[sample_idx].cpu().numpy()
    ax2.scatter(pos_np, pred_np, alpha=0.5, s=20)
    ax2.plot([0, T], [0, T], "r--", label="Perfect prediction")
    ax2.set_xlabel("Actual Position")
    ax2.set_ylabel("Predicted Position")
    ax2.set_title(
        f"Position Prediction (R²={np.corrcoef(pos_np, pred_np)[0, 1] ** 2:.4f})"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 2: Block 1 attention (averaged over heads)
    ax3 = fig.add_subplot(gs[1, 0])
    avg_attn1 = attn_weights1[sample_idx].mean(dim=0).cpu().numpy()
    im3 = ax3.imshow(avg_attn1, cmap="viridis", aspect="auto")
    ax3.set_title("Block 1 Avg Attention")
    ax3.set_xlabel("Key Position")
    ax3.set_ylabel("Query Position")
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: Block 2 attention (averaged over heads)
    ax4 = fig.add_subplot(gs[1, 1])
    avg_attn2 = attn_weights2[sample_idx].mean(dim=0).cpu().numpy()
    im4 = ax4.imshow(avg_attn2, cmap="viridis", aspect="auto")
    ax4.set_title("Block 2 Avg Attention")
    ax4.set_xlabel("Key Position")
    ax4.set_ylabel("Query Position")
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Row 2: Block 2 head patterns - where they attend at position 64
    ax5 = fig.add_subplot(gs[1, 2:])
    query_pos = 64
    for h in range(n_head):
        weights_at_pos = (
            attn_weights2[sample_idx, h, query_pos, : query_pos + 1].cpu().numpy()
        )
        ax5.plot(range(query_pos + 1), weights_at_pos, label=f"H{h}", alpha=0.7)
    ax5.set_xlabel("Key Position")
    ax5.set_ylabel("Attention Weight")
    ax5.set_title(f"Block 2 Attention from Query Position {query_pos}")
    ax5.legend(loc="upper left", ncol=4, fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Row 3: Dimension-position correlations
    ax6 = fig.add_subplot(gs[2, :2])

    # Compute correlations for each dimension at different layers
    r1_cpu = r1[sample_idx].cpu()
    ln1_b2_cpu = ln1_out_b2[sample_idx].cpu()
    attn_out2_cpu = attn_out2_proj[sample_idx].cpu()

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

    ax6.hist(
        r1_corrs,
        bins=50,
        alpha=0.5,
        label=f"Block1 out (max |r|={max(abs(min(r1_corrs)), abs(max(r1_corrs))):.3f})",
    )
    ax6.hist(
        attn2_corrs,
        bins=50,
        alpha=0.5,
        label=f"Attn2 out (max |r|={max(abs(min(attn2_corrs)), abs(max(attn2_corrs))):.3f})",
    )
    ax6.set_xlabel("Correlation with Position")
    ax6.set_ylabel("Count")
    ax6.set_title("Per-Dimension Position Correlations")
    ax6.legend()
    ax6.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    # Row 3: Top correlated dimensions in Attn2 output
    ax7 = fig.add_subplot(gs[2, 2:])

    # Get top 5 correlated dimensions
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

    for i, (d, corr) in enumerate(attn2_dim_corrs[:5]):
        vals = attn_out2_cpu[:, d].numpy()
        ax7.plot(
            pos_np,
            (vals - vals.mean()) / vals.std(),
            label=f"Dim {d} (r={corr:.3f})",
            alpha=0.7,
        )
    ax7.set_xlabel("Position")
    ax7.set_ylabel("Normalized Value")
    ax7.set_title("Top Position-Correlated Dimensions in Attn2 Output")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Row 4: Linear head weights and mechanism summary
    ax8 = fig.add_subplot(gs[3, :2])

    linear_weight = model.pos_head.weight.squeeze(0).detach().cpu().numpy()

    # Sort dimensions by |weight|
    sorted_dims = np.argsort(np.abs(linear_weight))[::-1]
    top_dims = sorted_dims[:30]

    ax8.bar(range(30), linear_weight[top_dims])
    ax8.set_xlabel("Rank (by |weight|)")
    ax8.set_ylabel("Weight Value")
    ax8.set_title("Top 30 Linear Head Weights")
    ax8.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Row 4: Mechanism flow diagram (text)
    ax9 = fig.add_subplot(gs[3, 2:])
    ax9.axis("off")

    mechanism_text = """
    R2 MECHANISM FLOW
    ═════════════════
    
    1. EMBEDDING: e_i (no position info)
       ↓
    2. BLOCK 1 (frozen, random):
       • Causal attention averages over 0..i
       • Creates variance ~ 1/(i+1)
       • Position correlation: r ≈ -0.42
       ↓
    3. LAYERNORM: Amplifies signal
       • Position correlation: r ≈ -0.87
       ↓
    4. BLOCK 2 ATTENTION (trained):
       • Attends to specific earlier positions
       • Creates near-perfect correlations (|r| > 0.99)
       ↓
    5. LINEAR HEAD: Extracts position
       • Distributed weighting
       • Final R² ≈ 0.95
    """
    ax9.text(
        0.1,
        0.5,
        mechanism_text,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        transform=ax9.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle("R2 Mechanism: How Attn2-Only Extracts Position", fontsize=16, y=1.01)

    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    data_path = "data/openwebtext/train.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Get a sample batch
    batch_size = 8
    block_size = 128

    torch.manual_seed(42)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    tokens = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    ).to(device)

    # Regimes to analyze
    regimes = ["R0", "R1", "R2", "R3", "R4_linear", "R4_mlp"]

    # Create output directory
    output_dir = "out-2layer-mechanism/attention_maps"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B
    wandb.init(
        project="nope-2layer-mechanism",
        name="attention-maps-visualization",
        config={
            "batch_size": batch_size,
            "block_size": block_size,
            "regimes": regimes,
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

        # Get attention weights
        attn_weights1, attn_weights2 = get_attention_weights(model, tokens, device)

        # Plot Block 1 attention maps
        fig1 = plot_attention_maps(attn_weights1, "Block 1", regime)
        fig1_path = f"{output_dir}/{regime}_block1_attention.png"
        fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
        wandb.log({f"{regime}/block1_attention_maps": wandb.Image(fig1)})
        plt.close(fig1)
        print(f"  Saved Block 1 attention maps")

        # Plot Block 2 attention maps
        fig2 = plot_attention_maps(attn_weights2, "Block 2", regime)
        fig2_path = f"{output_dir}/{regime}_block2_attention.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        wandb.log({f"{regime}/block2_attention_maps": wandb.Image(fig2)})
        plt.close(fig2)
        print(f"  Saved Block 2 attention maps")

        # Plot attention summary
        fig3 = plot_attention_summary(attn_weights1, attn_weights2, regime)
        fig3_path = f"{output_dir}/{regime}_attention_summary.png"
        fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
        wandb.log({f"{regime}/attention_summary": wandb.Image(fig3)})
        plt.close(fig3)
        print(f"  Saved attention summary")

        # For R2, also create the mechanism visualization
        if regime == "R2":
            print(f"\n  Creating R2 mechanism visualization...")
            fig4 = plot_r2_mechanism(model, tokens, device)
            fig4_path = f"{output_dir}/R2_mechanism_visualization.png"
            fig4.savefig(fig4_path, dpi=150, bbox_inches="tight")
            wandb.log({"R2/mechanism_visualization": wandb.Image(fig4)})
            plt.close(fig4)
            print(f"  Saved R2 mechanism visualization")

    # Also create R0 mechanism visualization for comparison
    print(f"\n{'=' * 60}")
    print("Creating R0 mechanism visualization for comparison")
    print(f"{'=' * 60}")

    model_r0, _ = load_model("R0", device)
    if model_r0 is not None:
        fig5 = plot_r2_mechanism(
            model_r0, tokens, device
        )  # Same function, different model
        fig5_path = f"{output_dir}/R0_mechanism_visualization.png"
        fig5.savefig(fig5_path, dpi=150, bbox_inches="tight")
        wandb.log({"R0/mechanism_visualization": wandb.Image(fig5)})
        plt.close(fig5)
        print(f"  Saved R0 mechanism visualization")

    # Create a comparison figure: R0 vs R2 Block 2 attention
    print(f"\n{'=' * 60}")
    print("Creating R0 vs R2 comparison")
    print(f"{'=' * 60}")

    model_r0, _ = load_model("R0", device)
    model_r2, _ = load_model("R2", device)

    if model_r0 is not None and model_r2 is not None:
        _, attn_r0_b2 = get_attention_weights(model_r0, tokens, device)
        _, attn_r2_b2 = get_attention_weights(model_r2, tokens, device)

        fig6 = plt.figure(figsize=(20, 10))

        # R0 Block 2 heads
        for h in range(12):
            ax = fig6.add_subplot(4, 6, h + 1)
            ax.imshow(attn_r0_b2[0, h].cpu().numpy(), cmap="viridis", aspect="auto")
            ax.set_title(f"R0 H{h}", fontsize=10)
            if h % 6 == 0:
                ax.set_ylabel("Query")
            if h >= 6:
                ax.set_xlabel("Key")

        # R2 Block 2 heads
        for h in range(12):
            ax = fig6.add_subplot(4, 6, h + 13)
            ax.imshow(attn_r2_b2[0, h].cpu().numpy(), cmap="viridis", aspect="auto")
            ax.set_title(f"R2 H{h}", fontsize=10)
            if h % 6 == 0:
                ax.set_ylabel("Query")
            ax.set_xlabel("Key")

        fig6.suptitle(
            "Block 2 Attention: R0 (BOS Reference) vs R2 (Variance Reading)",
            fontsize=14,
        )
        plt.tight_layout()

        fig6_path = f"{output_dir}/R0_vs_R2_block2_comparison.png"
        fig6.savefig(fig6_path, dpi=150, bbox_inches="tight")
        wandb.log({"comparison/R0_vs_R2_block2": wandb.Image(fig6)})
        plt.close(fig6)
        print(f"  Saved R0 vs R2 comparison")

    wandb.finish()
    print(f"\n{'=' * 60}")
    print("All visualizations complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
