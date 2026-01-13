"""
Scatter plot of post-LN2 norm vs position for random init model.

This visualizes the -0.97 correlation between norm and position that
appears at random initialization in NoPE transformers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOTS_DIR = (
    Path(__file__).parent.parent / "overleaf" / "nopos---claude-version" / "plots"
)


def get_activations(model, tokens):
    """Get post-LN2 activations."""
    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)
        block = model.transformer.h[0]

        x = block.ln_1(tok_emb)
        attn_out = block.attn(x)
        x = tok_emb + attn_out  # post_attn_residual
        x_ln2 = block.ln_2(x)  # post_ln2

    return {
        "post_attn": attn_out.detach(),
        "post_attn_residual": x.detach()
        - attn_out.detach()
        + attn_out.detach(),  # = tok_emb + attn_out
        "post_ln2": x_ln2.detach(),
    }


def main():
    print("=" * 70)
    print("SCATTER PLOT: POST-LN2 NORM VS POSITION")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Create random model (same config as in trained_model_direction_norm.py)
    config = GPTConfig(
        n_layer=1,
        n_head=12,
        n_embd=768,
        block_size=256,
        vocab_size=50257,
        dropout=0.0,
        use_positional_embedding=False,
        norm_type="layernorm",
    )

    model = GPT(config)
    model.eval()
    model.to(DEVICE)

    n_ctx = config.block_size
    vocab_size = config.vocab_size
    n_samples = 500

    print(f"\nConfig: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
    print(f"Context length: {n_ctx}")
    print(f"Number of samples: {n_samples}")

    # Collect data
    all_post_ln2_norms = []
    all_post_attn_norms = []
    all_positions = []

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Processing sample {i + 1}/{n_samples}...")

        tokens = torch.randint(0, vocab_size, (1, n_ctx), device=DEVICE)
        acts = get_activations(model, tokens)

        # Shape: (1, seq_len, d_model) -> (seq_len, d_model)
        post_ln2 = acts["post_ln2"][0].cpu().numpy()
        post_attn = acts["post_attn"][0].cpu().numpy()

        # Compute norms: (seq_len,)
        post_ln2_norms = np.linalg.norm(post_ln2, axis=1)
        post_attn_norms = np.linalg.norm(post_attn, axis=1)

        all_post_ln2_norms.append(post_ln2_norms)
        all_post_attn_norms.append(post_attn_norms)
        all_positions.append(np.arange(n_ctx))

    # Stack all data
    post_ln2_norms = np.concatenate(all_post_ln2_norms)
    post_attn_norms = np.concatenate(all_post_attn_norms)
    positions = np.concatenate(all_positions)

    print(f"\nTotal data points: {len(positions)}")

    # Compute correlations
    corr_ln2 = np.corrcoef(post_ln2_norms, positions)[0, 1]
    corr_attn = np.corrcoef(post_attn_norms, positions)[0, 1]

    print(f"\nCorrelations:")
    print(f"  Post-LN2 norm vs position: r = {corr_ln2:.4f}")
    print(f"  Post-attn norm vs position: r = {corr_attn:.4f}")

    # Compute mean norm by position for cleaner visualization
    mean_ln2_by_pos = np.array(
        [post_ln2_norms[positions == i].mean() for i in range(n_ctx)]
    )
    std_ln2_by_pos = np.array(
        [post_ln2_norms[positions == i].std() for i in range(n_ctx)]
    )

    mean_attn_by_pos = np.array(
        [post_attn_norms[positions == i].mean() for i in range(n_ctx)]
    )
    std_attn_by_pos = np.array(
        [post_attn_norms[positions == i].std() for i in range(n_ctx)]
    )

    # ====== CREATE FIGURES ======

    # Figure 1: Scatter plot with all points (subsample for visibility)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subsample for scatter plot (too many points otherwise)
    subsample_idx = np.random.choice(
        len(positions), size=min(10000, len(positions)), replace=False
    )

    # Post-LN2 scatter
    ax = axes[0]
    ax.scatter(
        positions[subsample_idx],
        post_ln2_norms[subsample_idx],
        alpha=0.1,
        s=1,
        c="blue",
        label="Individual samples",
    )
    ax.plot(np.arange(n_ctx), mean_ln2_by_pos, "r-", linewidth=2, label="Mean")
    ax.fill_between(
        np.arange(n_ctx),
        mean_ln2_by_pos - std_ln2_by_pos,
        mean_ln2_by_pos + std_ln2_by_pos,
        alpha=0.3,
        color="red",
        label="±1 std",
    )
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Post-LN2 Norm", fontsize=12)
    ax.set_title(
        f"Post-LN2 Norm vs Position\n(r = {corr_ln2:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Post-attention scatter
    ax = axes[1]
    ax.scatter(
        positions[subsample_idx],
        post_attn_norms[subsample_idx],
        alpha=0.1,
        s=1,
        c="green",
        label="Individual samples",
    )
    ax.plot(np.arange(n_ctx), mean_attn_by_pos, "r-", linewidth=2, label="Mean")
    ax.fill_between(
        np.arange(n_ctx),
        mean_attn_by_pos - std_attn_by_pos,
        mean_attn_by_pos + std_attn_by_pos,
        alpha=0.3,
        color="red",
        label="±1 std",
    )
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Post-Attention Norm", fontsize=12)
    ax.set_title(
        f"Post-Attention Norm vs Position\n(r = {corr_attn:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Random Initialization: Norm-Position Correlation",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig.savefig(PLOTS_DIR / "norm_position_scatter.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "norm_position_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {PLOTS_DIR / 'norm_position_scatter.pdf'}")

    # Figure 2: Mean norm with theoretical curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pos_array = np.arange(n_ctx)

    # Post-LN2
    ax = axes[0]
    ax.plot(pos_array, mean_ln2_by_pos, "b-", linewidth=2, label="Empirical mean")
    ax.fill_between(
        pos_array,
        mean_ln2_by_pos - std_ln2_by_pos,
        mean_ln2_by_pos + std_ln2_by_pos,
        alpha=0.3,
        color="blue",
    )

    # Theoretical: After LN, norm should be approximately sqrt(d_model) but with
    # small variations based on the input variance pattern
    # The correlation comes from: variance decreases with position -> LN gain increases
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Mean Post-LN2 Norm", fontsize=12)
    ax.set_title(
        f"Post-LN2: Mean Norm by Position\n(r = {corr_ln2:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add text annotation explaining the correlation
    ax.text(
        0.05,
        0.95,
        f"Correlation with position: r = {corr_ln2:.4f}\n"
        f"Norm range: [{mean_ln2_by_pos.min():.2f}, {mean_ln2_by_pos.max():.2f}]",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Post-attention with theoretical 1/sqrt(i+1) curve
    ax = axes[1]
    ax.plot(pos_array, mean_attn_by_pos, "g-", linewidth=2, label="Empirical mean")
    ax.fill_between(
        pos_array,
        mean_attn_by_pos - std_attn_by_pos,
        mean_attn_by_pos + std_attn_by_pos,
        alpha=0.3,
        color="green",
    )

    # Theoretical: 1/sqrt(i+1) scaled
    theoretical = 1 / np.sqrt(pos_array + 1)
    theoretical_scaled = theoretical * (mean_attn_by_pos[0] / theoretical[0])
    ax.plot(
        pos_array,
        theoretical_scaled,
        "r--",
        linewidth=2,
        label=r"Theoretical $\propto 1/\sqrt{i+1}$",
    )

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Mean Post-Attention Norm", fontsize=12)
    ax.set_title(
        f"Post-Attention: Mean Norm by Position\n(r = {corr_attn:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Compute correlation with theoretical
    corr_with_theory = np.corrcoef(mean_attn_by_pos, theoretical_scaled)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation with position: r = {corr_attn:.4f}\n"
        f"Correlation with theory: r = {corr_with_theory:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        "Random Initialization: Mean Norm vs Position",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig.savefig(PLOTS_DIR / "norm_position_mean.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "norm_position_mean.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {PLOTS_DIR / 'norm_position_mean.pdf'}")

    # Figure 3: Distribution of norms at specific positions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    positions_to_show = [0, n_ctx // 2, n_ctx - 1]
    colors = ["blue", "green", "red"]

    for ax, pos, color in zip(axes, positions_to_show, colors):
        norms_at_pos = post_ln2_norms[positions == pos]
        ax.hist(norms_at_pos, bins=50, alpha=0.7, color=color, edgecolor="black")
        ax.axvline(
            norms_at_pos.mean(),
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {norms_at_pos.mean():.2f}",
        )
        ax.set_xlabel("Post-LN2 Norm", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Position {pos}", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Distribution of Post-LN2 Norms at Different Positions",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig.savefig(
        PLOTS_DIR / "norm_distribution_by_position.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        PLOTS_DIR / "norm_distribution_by_position.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved: {PLOTS_DIR / 'norm_distribution_by_position.pdf'}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nPost-LN2 Norm:")
    print(
        f"  Position 0: mean = {mean_ln2_by_pos[0]:.4f}, std = {std_ln2_by_pos[0]:.4f}"
    )
    print(
        f"  Position {n_ctx // 2}: mean = {mean_ln2_by_pos[n_ctx // 2]:.4f}, std = {std_ln2_by_pos[n_ctx // 2]:.4f}"
    )
    print(
        f"  Position {n_ctx - 1}: mean = {mean_ln2_by_pos[-1]:.4f}, std = {std_ln2_by_pos[-1]:.4f}"
    )
    print(f"  Total variation: {mean_ln2_by_pos.max() - mean_ln2_by_pos.min():.4f}")
    print(
        f"  Percent variation: {(mean_ln2_by_pos.max() - mean_ln2_by_pos.min()) / mean_ln2_by_pos.mean() * 100:.2f}%"
    )

    print(f"\nPost-Attention Norm:")
    print(
        f"  Position 0: mean = {mean_attn_by_pos[0]:.4f}, std = {std_attn_by_pos[0]:.4f}"
    )
    print(
        f"  Position {n_ctx // 2}: mean = {mean_attn_by_pos[n_ctx // 2]:.4f}, std = {std_attn_by_pos[n_ctx // 2]:.4f}"
    )
    print(
        f"  Position {n_ctx - 1}: mean = {mean_attn_by_pos[-1]:.4f}, std = {std_attn_by_pos[-1]:.4f}"
    )
    print(f"  Total variation: {mean_attn_by_pos.max() - mean_attn_by_pos.min():.4f}")

    print("\n" + "=" * 70)
    print("HOW THE -0.97 CORRELATION IS CALCULATED")
    print("=" * 70)
    print("""
1. Generate n_samples=500 random sequences of length n_ctx=256
2. For each sequence, compute activations at post-LN2 layer
   Shape: (seq_len, d_model) = (256, 768)
3. For each position in each sequence, compute the L2 norm:
   norm[i] = sqrt(sum(activation[i, :]^2))
4. Stack all norms and positions:
   - post_ln2_norms: shape (500 * 256,) = (128000,)
   - positions: shape (128000,) with values [0,1,2,...,255, 0,1,2,...,255, ...]
5. Compute Pearson correlation:
   r = np.corrcoef(post_ln2_norms, positions)[0, 1]
   
The negative correlation (-0.97) means:
- Higher positions → Lower norms
- This happens because causal attention averages more tokens at later positions,
  reducing variance, and LayerNorm normalizes by variance, amplifying small
  variations into the norm.
""")

    return {
        "post_ln2_norm_position_corr": corr_ln2,
        "post_attn_norm_position_corr": corr_attn,
        "mean_ln2_by_pos": mean_ln2_by_pos.tolist(),
        "mean_attn_by_pos": mean_attn_by_pos.tolist(),
    }


if __name__ == "__main__":
    results = main()
