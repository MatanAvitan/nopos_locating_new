"""
Causal Intervention Experiment: Effect of K (unique prefix tokens) on Position Prediction

Tests the hypothesis: Changing K (number of unique prefix tokens) causally affects
the model's position predictions. This effect should be stronger at early checkpoints.

Experiment Design:
1. For each checkpoint, generate sequences with varying K (0 to 23)
2. Get position predictions from the model
3. Measure how prediction accuracy changes with K
4. Compare across checkpoints to see if effect diminishes with training

Usage:
    CUDA_VISIBLE_DEVICES=2 python analysis_scripts/causal_intervention_k.py \
        --checkpoint-dir out-posreg-6layer-until-mlp \
        --experiment-name causal-intervention-k
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

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
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, {"step": checkpoint.get("iter_num", 0)}


def generate_sequences_with_k_unique(
    k: int,
    n_copies: int,
    seq_len: int,
    base_token: int = 1000,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate n_copies sequences, each with exactly K unique prefix tokens.

    Sequence structure:
    - Positions 0..K-1: unique tokens [1001, 1002, ..., 1000+K]
    - Positions K..seq_len-1: base_token (1000)
    """
    sequences = torch.full(
        (n_copies, seq_len), base_token, dtype=torch.long, device=device
    )
    for pos in range(min(k, seq_len)):
        sequences[:, pos] = base_token + 1 + pos
    return sequences


def get_position_predictions(
    model: GPT, tokens: torch.Tensor
) -> torch.Tensor:
    """
    Get position predictions from the model (regression output).

    Returns:
        pos_out: (batch, seq_len) - continuous position predictions
    """
    with torch.no_grad():
        # Forward through transformer
        tok_emb = model.transformer.wte(tokens)
        x = model.transformer.drop(tok_emb)

        for block in model.transformer.h:
            x = block(x)

        x = model.transformer.ln_f(x)

        # Position regressor head outputs (batch, seq_len, 1)
        pos_out = model.pos_head(x).squeeze(-1)  # (batch, seq_len)

    return pos_out


def compute_regression_metrics(
    pos_out: torch.Tensor, seq_len: int
) -> Dict:
    """
    Compute regression metrics for position prediction.

    Args:
        pos_out: (batch, seq_len) - continuous position predictions
        seq_len: sequence length

    Returns:
        Dict with MAE, RMSE, Pearson correlation, etc.
    """
    true_positions = torch.arange(seq_len, device=pos_out.device).float()
    true_positions = true_positions.unsqueeze(0).expand_as(pos_out)

    errors = pos_out - true_positions

    mae = torch.abs(errors).mean().item()
    rmse = torch.sqrt((errors ** 2).mean()).item()
    mae_per_pos = torch.abs(errors).mean(dim=0).cpu().numpy()

    # Pearson correlation per sample, then average
    mean_pred = pos_out.mean(dim=1, keepdim=True)
    mean_true = true_positions.mean(dim=1, keepdim=True)

    cov = ((pos_out - mean_pred) * (true_positions - mean_true)).sum(dim=1)
    std_pred = torch.sqrt(((pos_out - mean_pred) ** 2).sum(dim=1))
    std_true = torch.sqrt(((true_positions - mean_true) ** 2).sum(dim=1))

    pearson = (cov / (std_pred * std_true + 1e-8)).mean().item()

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson,
        "mean_error": errors.mean().item(),
        "mae_per_pos": mae_per_pos,
    }


def run_intervention_experiment(
    model: GPT,
    max_k: int = 24,
    n_copies: int = 10,
    seq_len: int = 128,
    device: str = "cuda",
) -> Dict:
    """
    Run the causal intervention experiment for all K values.

    Returns:
        Dict with regression metrics for each K
    """
    results = {
        "mae_by_k": [],           # (max_k,) mean absolute error
        "rmse_by_k": [],          # (max_k,) root mean squared error
        "pearson_by_k": [],       # (max_k,) Pearson correlation
        "mae_by_k_pos": [],       # (max_k, seq_len) MAE per position
        "mae_prefix_by_k": [],    # MAE on prefix positions only
        "mae_suffix_by_k": [],    # MAE on suffix positions only
    }

    for k in range(max_k):
        tokens = generate_sequences_with_k_unique(k, n_copies, seq_len, device=device)
        pos_out = get_position_predictions(model, tokens)

        # Compute regression metrics
        metrics = compute_regression_metrics(pos_out, seq_len)

        results["mae_by_k"].append(metrics["mae"])
        results["rmse_by_k"].append(metrics["rmse"])
        results["pearson_by_k"].append(metrics["pearson_r"])
        results["mae_by_k_pos"].append(metrics["mae_per_pos"])

        # MAE on prefix (positions 0..K-1) vs suffix (positions K..end)
        if k > 0:
            results["mae_prefix_by_k"].append(float(np.mean(metrics["mae_per_pos"][:k])))
        else:
            results["mae_prefix_by_k"].append(0.0)

        if k < seq_len:
            results["mae_suffix_by_k"].append(float(np.mean(metrics["mae_per_pos"][k:])))
        else:
            results["mae_suffix_by_k"].append(0.0)

    return results


def create_intervention_summary_plot(
    all_results: List[Dict],
    steps: List[int],
) -> Figure:
    """
    Create summary plot showing causal effect of K across checkpoints.
    Uses regression metrics: MAE, Pearson R. Paper-ready styling.
    """
    import matplotlib.gridspec as gridspec

    n_checkpoints = len(all_results)
    max_k = len(all_results[0]["mae_by_k"])
    cmap = plt.cm.plasma

    # Create figure with colorbar space on the right
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.35, hspace=0.35)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cbar_ax = fig.add_subplot(gs[:, 2])

    # Plot 1: MAE vs K
    ax = axes[0]
    for i, (result, step) in enumerate(zip(all_results, steps)):
        color = cmap(i / max(1, n_checkpoints - 1))
        ax.plot(range(max_k), result["mae_by_k"], "-",
                color=color, alpha=0.8, linewidth=1.2)
    ax.set_xlabel(r"$K$ (unique prefix tokens)", fontsize=10)
    ax.set_ylabel("Mean Absolute Error", fontsize=10)
    ax.set_title("MAE vs $K$", fontsize=11, fontweight="medium")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 2: Pearson R vs K
    ax = axes[1]
    for i, (result, step) in enumerate(zip(all_results, steps)):
        color = cmap(i / max(1, n_checkpoints - 1))
        ax.plot(range(max_k), result["pearson_by_k"], "-",
                color=color, alpha=0.8, linewidth=1.2)
    ax.set_xlabel(r"$K$ (unique prefix tokens)", fontsize=10)
    ax.set_ylabel("Pearson $r$", fontsize=10)
    ax.set_title("Pearson $r$ vs $K$", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 3: Prefix vs Suffix MAE
    ax = axes[2]
    for i, (result, step) in enumerate(zip(all_results, steps)):
        color = cmap(i / max(1, n_checkpoints - 1))
        ax.plot(range(max_k), result["mae_prefix_by_k"], "-",
                color=color, alpha=0.8, linewidth=1.2)
        ax.plot(range(max_k), result["mae_suffix_by_k"], "--",
                color=color, alpha=0.5, linewidth=1.0)
    ax.set_xlabel(r"$K$ (unique prefix tokens)", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("Prefix (solid) vs Suffix (dashed)", fontsize=11, fontweight="medium")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 4: MAE heatmap
    ax = axes[3]
    latest_result = all_results[-1]
    mae_matrix = np.array(latest_result["mae_by_k_pos"])
    pos_stride = max(1, mae_matrix.shape[1] // 32)
    mae_subsampled = mae_matrix[:, ::pos_stride]
    im = ax.imshow(mae_subsampled, aspect="auto", cmap="plasma_r", origin="lower")
    ax.set_xlabel("Position (subsampled)", fontsize=10)
    ax.set_ylabel(r"$K$", fontsize=10)
    ax.set_title(f"MAE Heatmap (Final)", fontsize=11, fontweight="medium")
    ax.tick_params(axis="both", labelsize=9)
    cbar_hm = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar_hm.ax.tick_params(labelsize=8)

    # Shared colorbar for training step
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(steps), vmax=max(steps)))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Training Step", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    return fig


def create_causal_effect_plot(
    all_results: List[Dict],
    steps: List[int],
) -> Figure:
    """
    Plot the 'causal effect' of K on position prediction.
    Shows regression slope (dMAE/dK) and R² for MAE vs K relationship.
    Paper-ready styling.
    """
    from scipy import stats

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    k_values = np.arange(len(all_results[0]["mae_by_k"]))

    # Compute regression stats for each checkpoint
    mae_slopes = []
    mae_r2s = []
    pearson_slopes = []
    pearson_r2s = []

    for result in all_results:
        # MAE vs K regression
        slope, intercept, r_val, p_val, std_err = stats.linregress(k_values, result["mae_by_k"])
        mae_slopes.append(slope)
        mae_r2s.append(r_val ** 2)

        # Pearson vs K regression
        slope, intercept, r_val, p_val, std_err = stats.linregress(k_values, result["pearson_by_k"])
        pearson_slopes.append(slope)
        pearson_r2s.append(r_val ** 2)

    # Plot 1: MAE slope (dMAE/dK) - negative means K helps
    ax = axes[0]
    ax.plot(steps, mae_slopes, "o-", color="#2E86AB", linewidth=1.5, markersize=5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel(r"$\frac{\partial \mathrm{MAE}}{\partial K}$ (slope)", fontsize=10)
    ax.set_title("MAE Sensitivity to $K$", fontsize=11, fontweight="medium")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 2: Pearson slope (dR/dK) - positive means K helps
    ax = axes[1]
    ax.plot(steps, pearson_slopes, "o-", color="#A23B72", linewidth=1.5, markersize=5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel(r"$\frac{\partial r}{\partial K}$ (slope)", fontsize=10)
    ax.set_title("Pearson $r$ Sensitivity to $K$", fontsize=11, fontweight="medium")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 3: R² showing how well K explains variance
    ax = axes[2]
    ax.plot(steps, mae_r2s, "o-", color="#2E86AB", linewidth=1.5, markersize=5, label="MAE")
    ax.plot(steps, pearson_r2s, "s-", color="#A23B72", linewidth=1.5, markersize=5, label="Pearson $r$")
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel("$R^2$ (K explains variance)", fontsize=10)
    ax.set_title("Linearity of $K$ Effect", fontsize=11, fontweight="medium")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9)

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
        description="Causal Intervention Experiment - Effect of K on Position Prediction"
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default="causal-intervention-k")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--n-copies", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "nanoGPT" / args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist!")
        return

    # Find available checkpoints
    checkpoint_steps = [0] + list(range(1000, 21000, 1000))
    available_steps = [
        s for s in checkpoint_steps if (checkpoint_dir / f"ckpt_{s:05d}.pt").exists()
    ]

    if not available_steps:
        print("Error: No checkpoints found!")
        return

    print(f"\n{'=' * 70}")
    print(f"Causal Intervention Experiment - {args.experiment_name}")
    print(f"{'=' * 70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Found {len(available_steps)} checkpoints: {available_steps}")

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="nope-position-regression-metrics",
            name=f"causal_{args.experiment_name}",
            config={
                "max_k": args.max_k,
                "n_copies": args.n_copies,
                "checkpoint_steps": available_steps,
            },
        )
        print(f"WandB: https://wandb.ai/matan_avitan/nope-position-regression-metrics")

    all_results = []
    all_steps = []

    for step in tqdm(available_steps, desc="Analyzing"):
        ckpt_path = checkpoint_dir / f"ckpt_{step:05d}.pt"

        try:
            model, meta = load_checkpoint(str(ckpt_path), DEVICE)
            seq_len = model.config.block_size

            # Run intervention experiment
            result = run_intervention_experiment(
                model, args.max_k, args.n_copies, seq_len, DEVICE
            )
            result["step"] = step

            all_results.append(result)
            all_steps.append(step)

            # Log to wandb
            if use_wandb:
                metrics = {"checkpoint/step": step}

                # Summary metrics only (no individual K scalars - use aggregative plots)
                metrics["summary/mean_mae"] = np.mean(result["mae_by_k"])
                metrics["summary/mean_pearson"] = np.mean(result["pearson_by_k"])
                metrics["summary/mae_k0"] = result["mae_by_k"][0]
                metrics["summary/mae_k23"] = result["mae_by_k"][-1]
                metrics["summary/pearson_k0"] = result["pearson_by_k"][0]
                metrics["summary/pearson_k23"] = result["pearson_by_k"][-1]
                metrics["summary/mae_effect"] = result["mae_by_k"][0] - result["mae_by_k"][-1]
                metrics["summary/pearson_effect"] = result["pearson_by_k"][-1] - result["pearson_by_k"][0]

                # Aggregative plots (only after we have 2+ checkpoints)
                if len(all_results) > 1:
                    fig_summary = create_intervention_summary_plot(all_results, all_steps)
                    metrics["plots/intervention_summary"] = wandb.Image(fig_to_image(fig_summary))

                    fig_effect = create_causal_effect_plot(all_results, all_steps)
                    metrics["plots/causal_effect"] = wandb.Image(fig_to_image(fig_effect))

                wandb.log(metrics, commit=True)

            print(f"  Step {step}: MAE[K=0]={result['mae_by_k'][0]:.2f}, "
                  f"MAE[K=23]={result['mae_by_k'][-1]:.2f}, "
                  f"R[K=0]={result['pearson_by_k'][0]:.3f}, "
                  f"R[K=23]={result['pearson_by_k'][-1]:.3f}")

        except Exception as e:
            print(f"  Step {step}: Error - {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"Analysis complete! {len(all_results)} checkpoints analyzed.")

    if use_wandb:
        wandb.finish()
        print("WandB run finished.")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
