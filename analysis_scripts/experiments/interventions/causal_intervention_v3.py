"""
Causal Intervention V3: Value Vector Replacement in Layer 0 Attention

Experiment:
1. Generate 24 sequences with unique prefix pattern (sample K has K unique prefix tokens)
2. Intervention: Replace the VALUE vector at position K in layer 0 attention with
   the value vector of a NEW unique token (W_V @ embedding[new_token])
3. This propagates through W_O automatically in the attention output
4. Due to causal attention, intervention at K only affects positions >= K
5. Measure whether intervention improves or worsens position predictions for pos >= K

The intervention is:
    V[:, K, :] = W_V @ embedding[intervention_token]

Where W_V is part of c_attn projection and W_O (c_proj) is applied to attention output.

Usage:
    CUDA_VISIBLE_DEVICES=3 python analysis_scripts/causal_intervention_v3.py \
        --checkpoint-dir out-posreg-6layer-until-mlp \
        --experiment-name causal-intervention-v3
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import warnings
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "nanoGPT"))
from model_position_classifier import (
    GPTPositionClassifier as GPT,
    GPTPositionClassifierConfig as GPTConfig,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# ICML Publication Style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 0.8,
    }
)

# ICML column width ~3.25in, full width ~6.75in
ICML_COLUMN_WIDTH = 3.25
ICML_FULL_WIDTH = 6.75


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[GPT, dict]:
    """Load a model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    gptconf = GPTConfig(
        n_layer=model_args.get("n_layer", 6),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 128),
        bias=model_args.get("bias", False),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=model_args.get("use_positional_embedding", False),
        norm_type=model_args.get("norm_type", "layernorm"),
        use_regression=model_args.get("use_regression", True),
        compute_lm_loss=model_args.get("compute_lm_loss", False),
        use_ln2=model_args.get("use_ln2", True),
    )

    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, {"step": checkpoint.get("iter_num", 0)}


def generate_unique_prefix_sequences(
    n_samples: int,
    seq_len: int,
    base_token: int = 1000,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate sequences where sample i has i unique prefix tokens.

    - Sample 0: [1000, 1000, 1000, ...] (all base_token)
    - Sample 1: [1001, 1000, 1000, ...] (1 unique, rest base_token)
    - Sample i: [1001, ..., 1000+i, 1000, ...] (i unique, rest base_token)

    Tokens used: base_token (1000) and 1001 to 1000+n_samples-1
    """
    sequences = torch.full(
        (n_samples, seq_len), base_token, dtype=torch.long, device=device
    )

    for sample_idx in range(n_samples):
        n_unique = min(sample_idx, seq_len)
        for pos in range(n_unique):
            sequences[sample_idx, pos] = base_token + 1 + pos

    return sequences


def attention_forward_with_value_intervention(
    attn_module,
    x: torch.Tensor,
    intervention_positions: torch.Tensor,
    v_new: torch.Tensor,
) -> torch.Tensor:
    """
    Custom attention forward that replaces the value vector at specified positions.

    Args:
        attn_module: The CausalSelfAttention module
        x: Input tensor (B, T, C)
        intervention_positions: (B,) position to intervene for each batch element
        v_new: (C,) the new value vector to insert (already computed as W_V @ emb[new_token])

    Returns:
        Attention output (B, T, C)
    """
    B, T, C = x.size()
    n_head = attn_module.n_head
    head_dim = C // n_head

    # Compute Q, K, V
    qkv = attn_module.c_attn(x)
    q, k, v = qkv.split(attn_module.n_embd, dim=2)

    # Replace V at intervention positions
    # v has shape (B, T, C)
    for b in range(B):
        pos = intervention_positions[b].item()
        if 0 <= pos < T:
            v[b, pos, :] = v_new

    # Reshape for multi-head attention
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

    # Compute attention (manual implementation to avoid flash attention)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
    )
    att = att.masked_fill(causal_mask, float("-inf"))
    att = F.softmax(att, dim=-1)

    # Apply attention to values
    y = att @ v  # (B, n_head, T, head_dim)

    # Reshape and project
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = attn_module.c_proj(y)  # This is W_O

    return y


def get_position_predictions(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """
    Get position predictions using the full transformer model.
    Applies sigmoid + scaling as done in training.

    Returns:
        pos_out: (batch, seq_len) in [0, block_size-1]
    """
    block_size = model.config.block_size

    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)
        x = model.transformer.drop(tok_emb)

        for block in model.transformer.h:
            x = block(x)

        x = model.transformer.ln_f(x)
        raw_out = model.pos_head(x).squeeze(-1)
        pos_out = torch.sigmoid(raw_out) * (block_size - 1)

    return pos_out


def get_position_predictions_with_value_intervention(
    model: GPT,
    tokens: torch.Tensor,
    intervention_positions: torch.Tensor,
    intervention_token: int,
) -> torch.Tensor:
    """
    Get predictions with VALUE vector intervention in layer 0 attention.

    The intervention replaces V[:, K, :] with W_V @ embedding[intervention_token]
    for each sample, where K is the intervention position for that sample.

    Args:
        model: The GPT model
        tokens: (batch, seq_len) input tokens
        intervention_positions: (batch,) position to intervene for each sample
        intervention_token: Token ID whose embedding to use for the new value vector

    Returns:
        pos_out: (batch, seq_len) in [0, block_size-1]
    """
    block_size = model.config.block_size
    n_embd = model.config.n_embd

    with torch.no_grad():
        # Get the intervention token's embedding
        intervention_emb = model.transformer.wte.weight[intervention_token]  # (n_embd,)

        # Compute the new VALUE vector: this is W_V @ emb
        # c_attn projects to [Q, K, V], so V part is the last n_embd
        # c_attn.weight has shape (3*n_embd, n_embd)
        # For a single embedding: qkv = c_attn(emb.unsqueeze(0).unsqueeze(0))
        layer0_attn = model.transformer.h[0].attn

        # Get V projection for the intervention embedding
        qkv_new = layer0_attn.c_attn(intervention_emb.unsqueeze(0))  # (1, 3*n_embd)
        v_new = qkv_new[0, 2 * n_embd :]  # (n_embd,) - the V part

        # Forward pass with intervention
        tok_emb = model.transformer.wte(tokens)
        x = model.transformer.drop(tok_emb)

        # Layer 0: Apply LN1, then attention with intervention
        block0 = model.transformer.h[0]
        x_ln1 = block0.ln_1(x)
        attn_out = attention_forward_with_value_intervention(
            block0.attn, x_ln1, intervention_positions, v_new
        )
        x = x + attn_out

        # Layer 0: MLP
        if block0.use_ln2:
            x = x + block0.mlp(block0.ln_2(x))
        else:
            x = x + block0.mlp(x)

        # Remaining layers (no intervention)
        for block in model.transformer.h[1:]:
            x = block(x)

        x = model.transformer.ln_f(x)
        raw_out = model.pos_head(x).squeeze(-1)
        pos_out = torch.sigmoid(raw_out) * (block_size - 1)

    return pos_out


def compute_causal_metrics(
    pred_orig: torch.Tensor,
    pred_interv: torch.Tensor,
    true_positions: torch.Tensor,
    k: int,
) -> Dict:
    """
    Compute metrics for positions >= K only (due to causal masking).

    Returns:
        Dict with:
        - mae_orig: MAE of original predictions for pos >= K
        - mae_interv: MAE of intervened predictions for pos >= K
        - improvement: mae_orig - mae_interv (positive = intervention helped)
        - pct_improved: % of positions where intervention moved prediction closer to truth
    """
    seq_len = pred_orig.shape[0]

    if k >= seq_len:
        return {"mae_orig": 0, "mae_interv": 0, "improvement": 0, "pct_improved": 0}

    # Only consider positions >= K
    pred_orig_k = pred_orig[k:]
    pred_interv_k = pred_interv[k:]
    true_k = true_positions[k:]

    # Absolute errors
    error_orig = torch.abs(pred_orig_k - true_k)
    error_interv = torch.abs(pred_interv_k - true_k)

    mae_orig = error_orig.mean().item()
    mae_interv = error_interv.mean().item()
    improvement = mae_orig - mae_interv  # Positive = intervention helped

    # Percentage of positions where intervention improved prediction
    improved_mask = error_interv < error_orig
    pct_improved = improved_mask.float().mean().item() * 100

    return {
        "mae_orig": mae_orig,
        "mae_interv": mae_interv,
        "improvement": improvement,
        "pct_improved": pct_improved,
    }


def create_per_k_grid_plot(
    predictions_original: torch.Tensor,
    predictions_intervened: torch.Tensor,
    seq_len: int,
    step: int,
    n_samples: int = 24,
) -> Figure:
    """
    Create ICML-ready grid of subplots, one per K value.
    Each subplot shows original vs intervened predictions.
    Color indicates whether intervention improved or worsened predictions.
    """
    n_cols = 6
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(ICML_FULL_WIDTH, 4.0))
    axes = axes.flatten()

    true_positions = torch.arange(seq_len, device=predictions_original.device).float()

    for k in range(min(n_samples, n_rows * n_cols)):
        ax = axes[k]

        pred_orig = predictions_original[k]
        pred_interv = predictions_intervened[k]

        # Compute metrics for pos >= K
        metrics = compute_causal_metrics(pred_orig, pred_interv, true_positions, k)

        # Plot predictions
        positions_np = np.arange(seq_len)
        ax.plot(
            positions_np,
            pred_orig.cpu().numpy(),
            "b-",
            linewidth=0.6,
            alpha=0.8,
            label="Original",
        )
        ax.plot(
            positions_np,
            pred_interv.cpu().numpy(),
            "r-",
            linewidth=0.6,
            alpha=0.8,
            label="Intervened",
        )

        # Perfect prediction line
        ax.plot(positions_np, positions_np, "k--", linewidth=0.4, alpha=0.4)

        # Mark intervention position with vertical line
        if k < seq_len:
            ax.axvline(x=k, color="green", linestyle=":", linewidth=0.5, alpha=0.7)

        # Shade the causal region (positions >= K)
        if k < seq_len:
            ax.axvspan(k, seq_len, alpha=0.05, color="gray")

        # Title with improvement indicator
        if metrics["improvement"] > 0.1:
            title_color = "darkgreen"
            arrow = r"$\uparrow$"
        elif metrics["improvement"] < -0.1:
            title_color = "darkred"
            arrow = r"$\downarrow$"
        else:
            title_color = "black"
            arrow = r"$\approx$"

        ax.set_title(f"$K={k}$ {arrow}", fontsize=8, color=title_color, pad=2)

        # Minimal axis labels
        if k >= (n_rows - 1) * n_cols:  # Bottom row
            ax.set_xlabel("Pos", fontsize=7)
        if k % n_cols == 0:  # Left column
            ax.set_ylabel("Pred", fontsize=7)

        # Set axis limits
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, seq_len)

        # Sparse ticks
        ax.set_xticks([0, seq_len // 2, seq_len])
        ax.set_yticks([0, seq_len // 2, seq_len])

        ax.grid(True, alpha=0.1, linewidth=0.3)

    # Add legend to first subplot
    axes[0].legend(loc="lower right", fontsize=5, framealpha=0.9)

    fig.suptitle(
        f"Value Vector Intervention in Layer 0 Attention (Step {step})\n"
        r"Green $\uparrow$ = improved, Red $\downarrow$ = worsened (pos $\geq$ K)",
        fontsize=9,
        y=1.02,
    )

    plt.tight_layout()
    return fig


def create_improvement_summary_plot(
    predictions_original: torch.Tensor,
    predictions_intervened: torch.Tensor,
    seq_len: int,
    step: int,
    n_samples: int = 24,
) -> Figure:
    """
    Create ICML-ready summary plot showing intervention effect per K.
    Only measures effect for positions >= K (due to causal masking).
    """
    fig, axes = plt.subplots(1, 3, figsize=(ICML_FULL_WIDTH, 2.0))

    true_positions = torch.arange(seq_len, device=predictions_original.device).float()

    improvements = []
    pct_improved_list = []
    mae_orig_list = []
    mae_interv_list = []

    for k in range(n_samples):
        pred_orig = predictions_original[k]
        pred_interv = predictions_intervened[k]
        metrics = compute_causal_metrics(pred_orig, pred_interv, true_positions, k)

        improvements.append(metrics["improvement"])
        pct_improved_list.append(metrics["pct_improved"])
        mae_orig_list.append(metrics["mae_orig"])
        mae_interv_list.append(metrics["mae_interv"])

    k_values = np.arange(n_samples)

    # Plot 1: Improvement (MAE reduction)
    ax = axes[0]
    colors = [
        "darkgreen" if x > 0.1 else ("darkred" if x < -0.1 else "gray")
        for x in improvements
    ]
    ax.bar(
        k_values,
        improvements,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$K$ (unique prefix)", fontsize=9)
    ax.set_ylabel("MAE Reduction", fontsize=9)
    ax.set_title("Intervention Effect\n(+) = closer to truth", fontsize=9)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.grid(True, alpha=0.15, axis="y", linewidth=0.3)

    # Plot 2: Percentage of positions improved
    ax = axes[1]
    colors = ["darkgreen" if x > 50 else "darkred" for x in pct_improved_list]
    ax.bar(
        k_values,
        pct_improved_list,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axhline(y=50, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$K$ (unique prefix)", fontsize=9)
    ax.set_ylabel("% Positions Improved", fontsize=9)
    ax.set_title("Positions Where\nIntervention Helped", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.grid(True, alpha=0.15, axis="y", linewidth=0.3)

    # Plot 3: MAE comparison
    ax = axes[2]
    width = 0.35
    ax.bar(
        k_values - width / 2,
        mae_orig_list,
        width,
        label="Original",
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.bar(
        k_values + width / 2,
        mae_interv_list,
        width,
        label="Intervened",
        color="indianred",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel(r"$K$ (unique prefix)", fontsize=9)
    ax.set_ylabel(r"MAE (pos $\geq$ K)", fontsize=9)
    ax.set_title("MAE Comparison\n(lower = better)", fontsize=9)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.15, axis="y", linewidth=0.3)

    fig.suptitle(
        f"Value Vector Intervention Analysis (Step {step})", fontsize=9, y=1.08
    )

    plt.tight_layout()
    return fig


def fig_to_image(fig: Figure):
    """Convert matplotlib figure to PIL Image for wandb."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def main():
    parser = argparse.ArgumentParser(
        description="Causal Intervention V3 - Value Vector Replacement in Layer 0 Attention"
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default="causal-intervention-v3")
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument(
        "--intervention-token",
        type=int,
        default=2000,
        help="Token ID for intervention (should not appear in sequences)",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Specific steps to analyze (default: all available)",
    )
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "nanoGPT" / args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist!")
        return

    # Find available checkpoints
    checkpoint_steps = [0] + list(range(500, 21000, 500))
    available_steps = [
        s for s in checkpoint_steps if (checkpoint_dir / f"ckpt_{s:05d}.pt").exists()
    ]

    if args.steps:
        available_steps = [s for s in args.steps if s in available_steps]

    if not available_steps:
        print("Error: No checkpoints found!")
        return

    print(f"\n{'=' * 70}")
    print(f"Causal Intervention V3 - Value Vector Replacement")
    print(f"{'=' * 70}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Analyzing {len(available_steps)} checkpoints")
    print(f"N samples: {args.n_samples} (K=0 to K={args.n_samples - 1})")
    print(
        f"Intervention: Replace V[K] = W_V @ emb[{args.intervention_token}] in layer 0"
    )
    print(f"Note: Effect measured only for positions >= K (causal masking)")
    print(f"Device: {DEVICE}")

    # Create save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = PROJECT_ROOT / "results" / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="nope-position-regression-metrics",
            name=args.experiment_name,
            config={
                "n_samples": args.n_samples,
                "intervention_token": args.intervention_token,
                "intervention_type": "value_vector_replacement_layer0",
                "checkpoint_steps": available_steps,
            },
        )
        print(f"WandB: https://wandb.ai/matan_avitan/nope-position-regression-metrics")

    for step in tqdm(available_steps, desc="Analyzing checkpoints"):
        ckpt_path = checkpoint_dir / f"ckpt_{step:05d}.pt"

        try:
            model, meta = load_checkpoint(str(ckpt_path), DEVICE)
            seq_len = model.config.block_size

            # Generate sequences
            tokens = generate_unique_prefix_sequences(
                args.n_samples, seq_len, base_token=1000, device=DEVICE
            )

            # Intervention positions: for sample K, intervene at position K
            intervention_positions = torch.arange(args.n_samples, device=DEVICE)

            # Get predictions
            predictions_original = get_position_predictions(model, tokens)
            predictions_intervened = get_position_predictions_with_value_intervention(
                model, tokens, intervention_positions, args.intervention_token
            )

            # Create plots
            fig_grid = create_per_k_grid_plot(
                predictions_original,
                predictions_intervened,
                seq_len,
                step,
                args.n_samples,
            )
            fig_summary = create_improvement_summary_plot(
                predictions_original,
                predictions_intervened,
                seq_len,
                step,
                args.n_samples,
            )

            # Save in both PNG and PDF (separate files)
            fig_grid.savefig(
                save_dir / f"grid_step{step:05d}.png", dpi=300, bbox_inches="tight"
            )
            fig_grid.savefig(save_dir / f"grid_step{step:05d}.pdf", bbox_inches="tight")
            fig_summary.savefig(
                save_dir / f"summary_step{step:05d}.png", dpi=300, bbox_inches="tight"
            )
            fig_summary.savefig(
                save_dir / f"summary_step{step:05d}.pdf", bbox_inches="tight"
            )

            # Log to wandb
            if use_wandb:
                wandb.log(
                    {
                        "checkpoint/step": step,
                        "plots/grid": wandb.Image(fig_to_image(fig_grid)),
                        "plots/summary": wandb.Image(fig_to_image(fig_summary)),
                    },
                    commit=True,
                )
            else:
                plt.close(fig_grid)
                plt.close(fig_summary)

            print(f"  Step {step}: Done")

        except Exception as e:
            print(f"  Step {step}: Error - {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Plots saved to: {save_dir}")
    print(f"  - grid_stepXXXXX.png/pdf: 24 subplots, one per K")
    print(f"  - summary_stepXXXXX.png/pdf: Improvement metrics per K")

    if use_wandb:
        wandb.finish()
        print("WandB run finished.")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
