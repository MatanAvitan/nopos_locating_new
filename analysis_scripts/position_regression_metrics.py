"""
Position Regression Metrics Analysis - Per-K Basis Hypothesis Testing

Tests the hypothesis: K unique tokens form a basis that the MLP
uses to decode position.

Key metrics computed per-K (0-23 unique tokens):
1. Basis contribution: |activation · E_basis| (Metric 1)
2. Pythagorean numbers: ||v||² before/after attention (Metric 2)
3. PCA/Singular values at each layer (Metric 3)

Usage:
    CUDA_VISIBLE_DEVICES=4 python analysis_scripts/position_regression_metrics.py \
        --checkpoint-dir out-posreg-6layer-until-mlp \
        --experiment-name nope-6layer-until-first-mlp \
        --n-samples 24 \
        --mode unique
"""

import io
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

# Use non-interactive backend BEFORE importing pyplot
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_position_classifier import (
    GPTPositionClassifier as GPT,
    GPTPositionClassifierConfig as GPTConfig,
)

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Results will only be saved locally.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[GPT, dict]:
    """Load a model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    gptconf = GPTConfig(
        n_layer=model_args.get("n_layer", 6),
        n_head=model_args.get("n_head", 1),
        n_embd=model_args.get("n_embd", 7),
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

    # Handle state dict with _orig_mod prefix (from torch.compile)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {
        "step": checkpoint.get("iter_num", 0),
        "train_loss": checkpoint.get("best_val_loss", None),
        "config": model_args,
    }

    return model, meta


def generate_unique_prefix_sequences(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
    base_token: int = 1000,
) -> torch.Tensor:
    """
    Generate sequences where sample i (0-indexed) contains exactly K=i+1
    unique tokens counting the shared base_token.

    Examples:
        Sample 0 (K=1): [1000, 1000, 1000, ...]
        Sample 1 (K=2): [1001, 1000, 1000, ...]
        Sample 2 (K=3): [1001, 1002, 1000, ...]
    """
    sequences = torch.full(
        (n_samples, seq_len), base_token, dtype=torch.long, device=device
    )
    for sample_idx in range(n_samples):
        target_k = min(sample_idx + 1, seq_len)
        n_new_tokens = max(target_k - 1, 0)
        for pos in range(n_new_tokens):
            sequences[sample_idx, pos] = base_token + 1 + pos
    return sequences


def generate_random_sequences(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate completely random sequences."""
    return torch.randint(0, vocab_size, (n_samples, seq_len), device=device)


def extract_basis_embeddings(
    model: GPT, base_token: int = 1000, n_basis: int = 24
) -> torch.Tensor:
    """
    Extract embeddings for unique tokens including base_token.

    Returns embeddings for [base_token, base_token+1, ..., base_token+n_basis-1].
    """
    unique_tokens = torch.arange(
        base_token, base_token + n_basis, device=next(model.parameters()).device
    )
    basis = model.transformer.wte(unique_tokens)
    return basis


def extract_weight_matrices(
    model: GPT, layer_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract W_v and W_o from attention layer."""
    block = model.transformer.h[layer_idx]
    c_attn_weight = block.attn.c_attn.weight.data
    n_embd = model.config.n_embd
    W_v = c_attn_weight[2 * n_embd :, :].T
    W_o = block.attn.c_proj.weight.data.T
    return W_v, W_o


def get_detailed_activations(
    model: GPT, tokens: torch.Tensor, layer_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """Extract activations at key points including mlp_hidden."""
    activations = {}

    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)
        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[layer_idx]

        # Post-LN1 and Attention
        x_ln1 = block.ln_1(x)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # After first residual
        x = x + attn_out

        # Post-LN2
        if block.use_ln2:
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.clone()
            mlp_input = x

        # MLP hidden activations (after c_fc and GELU)
        mlp_hidden = block.mlp.c_fc(mlp_input)
        mlp_hidden = block.mlp.gelu(mlp_hidden)
        activations["mlp_hidden"] = mlp_hidden.clone()

        # Post-MLP
        mlp_out = block.mlp.c_proj(mlp_hidden)
        if hasattr(block.mlp, "dropout"):
            mlp_out = block.mlp.dropout(mlp_out)
        activations["post_mlp"] = mlp_out.clone()

    return activations


def compute_pythagorean_norms(
    model: GPT, tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Metric 2: Compute ||v||² before and after attention."""
    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)
        block = model.transformer.h[0]
        ln1_out = block.ln_1(tok_emb)
        W_v, _ = extract_weight_matrices(model)
        value_vectors = ln1_out @ W_v.T
        norms_before = (value_vectors**2).sum(dim=-1)
        attn_out = block.attn(ln1_out)
        norms_after = (attn_out**2).sum(dim=-1)
    return norms_before, norms_after


def compute_per_k_metrics(
    model: GPT,
    tokens: torch.Tensor,
    basis: torch.Tensor,
    n_samples: int,
    seq_len: int,
) -> Dict:
    """Compute metrics grouped by K (number of unique tokens)."""
    layers = ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]
    results = {
        "basis_contrib_by_k": {},
        "projections_heatmap": {},
        "pca_cumvar_curves": {},
        "activations": {},
    }

    all_acts = {layer: [] for layer in layers}
    for k in range(n_samples):
        acts = get_detailed_activations(model, tokens[k : k + 1])
        for layer in layers:
            all_acts[layer].append(acts[layer])

    for layer in layers:
        results["activations"][layer] = torch.cat(all_acts[layer], dim=0)

    for layer in layers:
        acts = results["activations"][layer]
        D = acts.shape[-1]

        # Metric 1: Projections onto basis vectors
        if D == basis.shape[-1]:
            basis_norms = torch.norm(basis, dim=1, keepdim=True)
            basis_normalized = basis / (basis_norms + 1e-8)
            projections_full = torch.zeros(
                (n_samples, seq_len, n_samples), device=acts.device, dtype=acts.dtype
            )
            contrib_by_k = []
            for k in range(n_samples):
                basis_k = basis_normalized[: k + 1]
                proj = torch.einsum("sd,bd->sb", acts[k], basis_k)
                projections_full[k, :, : k + 1] = proj
                contrib_by_k.append(torch.abs(proj).mean().item())
            results["basis_contrib_by_k"][layer] = contrib_by_k
            heatmap = torch.abs(projections_full).mean(dim=0)
            results["projections_heatmap"][layer] = heatmap.T.detach().cpu().numpy()
        else:
            # If dimensions don't match (mlp_hidden), skip projection
            results["basis_contrib_by_k"][layer] = [0.0] * n_samples
            results["projections_heatmap"][layer] = np.zeros((basis.shape[0], seq_len))

        # Metric 3: Per-K PCA (Cumulative variance curves)
        pca_cumvar_curves = []
        max_components = 100

        for k in range(n_samples):
            sample_acts = acts[k].detach().cpu().numpy().astype(np.float64)
            sample_centered = sample_acts - sample_acts.mean(axis=0, keepdims=True)

            try:
                from sklearn.decomposition import TruncatedSVD

                n_comp = min(max_components, sample_acts.shape[0] - 1)
                svd = TruncatedSVD(
                    n_components=n_comp,
                    algorithm="randomized",
                    n_iter=5,
                    random_state=42,
                )
                svd.fit(sample_centered)
                cumvar = np.cumsum(svd.explained_variance_ratio_).tolist()
                pca_cumvar_curves.append(cumvar)
            except Exception:
                pca_cumvar_curves.append([0.0])

        results["pca_cumvar_curves"][layer] = pca_cumvar_curves

    return results


def compute_pca_analysis(activations: torch.Tensor, max_components: int = 100) -> Dict:
    """Metric 3: Global PCA analysis."""
    from sklearn.decomposition import TruncatedSVD

    n_samples, seq_len, d_model = activations.shape
    acts_flat = (
        activations.reshape(-1, d_model).detach().cpu().numpy().astype(np.float64)
    )
    acts_centered = acts_flat - acts_flat.mean(axis=0, keepdims=True)
    k = min(max_components, min(acts_flat.shape) - 1)
    svd = TruncatedSVD(
        n_components=k, algorithm="randomized", n_iter=5, random_state=42
    )
    svd.fit(acts_centered)
    return {
        "singular_values": svd.singular_values_.tolist(),
        "explained_variance": svd.explained_variance_ratio_.tolist(),
    }


def create_pca_cumvar_curves_plot(result: Dict, layer: str = "post_mlp") -> Figure:
    """Plot cumulative variance curves with K markers."""
    fig, ax = plt.subplots(figsize=(12, 7))
    curves = result["per_k_metrics"]["pca_cumvar_curves"][layer]
    n_samples = len(curves)

    for k in range(n_samples):
        cumvar = curves[k]
        if len(cumvar) > 0:
            x_vals = list(range(1, len(cumvar) + 1))
            color = plt.cm.viridis((k + 1) / n_samples)
            ax.plot(x_vals, cumvar, "-", color=color, alpha=0.4, linewidth=1)
            marker_idx = k
            if marker_idx < len(cumvar):
                ax.scatter(
                    [k + 1],
                    [cumvar[marker_idx]],
                    color=color,
                    s=40,
                    marker="o",
                    zorder=5,
                )

    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90%")
    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=12)
    ax.set_title(
        f"PCA Cumulative Variance Curves by K ({layer}, step {result['step']})\nDots mark the K-th component",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_samples)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("K (unique tokens)", fontsize=11)
    plt.tight_layout()
    return fig


def create_pythagorean_norms_plot(all_results: List[Dict]) -> Figure:
    """Metric 2: Plot Pythagorean norms evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = [r["step"] for r in all_results]

    ax = axes[0]
    mean_before = [np.mean(r["norms_before_attn"]) for r in all_results]
    mean_after = [np.mean(r["norms_after_attn"]) for r in all_results]
    ax.plot(steps, mean_before, "o-", label="Before Attention")
    ax.plot(steps, mean_after, "s-", label="After Attention")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean ||v||²")
    ax.set_title("Pythagorean Norms Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    mean_ratio = [np.mean(r["norm_ratios"]) for r in all_results]
    ax.plot(steps, mean_ratio, "o-", color="purple")
    ax.axhline(y=1.0, color="red", linestyle="--")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Norm Ratio (After/Before)")
    ax.set_title("Attention Norm Transformation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_k_scaling_plot(all_results: List[Dict], layer: str) -> Figure:
    """
    Aggregative plot: K on x-axis, basis contribution on y-axis.
    One line per checkpoint, color-coded by training step.
    Paper-ready styling.
    """
    import matplotlib.gridspec as gridspec

    layer_names = {
        "post_attn": "Post-Attention",
        "post_ln2": "Post-LN2",
        "mlp_hidden": "MLP Hidden",
        "post_mlp": "Post-MLP",
    }

    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    steps = [r["step"] for r in all_results]
    n_ckpts = len(steps)
    cmap = plt.cm.plasma

    for i, result in enumerate(all_results):
        color = cmap(i / max(1, n_ckpts - 1))
        contrib_by_k = result["per_k_metrics"]["basis_contrib_by_k"][layer]
        ax.plot(
            range(1, len(contrib_by_k) + 1),
            contrib_by_k,
            "-",
            color=color,
            alpha=0.8,
            linewidth=1.2,
        )

    ax.set_xlabel(r"$K$ (unique tokens)", fontsize=11)
    ax.set_ylabel("Mean Basis Contribution", fontsize=11)
    ax.set_title(layer_names.get(layer, layer), fontsize=12, fontweight="medium")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min(steps), vmax=max(steps))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Training Step", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    return fig


def create_pca_var_at_k_plot(all_results: List[Dict], layer: str) -> Figure:
    """
    Shows PCA cumulative variance AT the K-th component for each K.
    X-axis: K, Y-axis: cumvar[K], lines by checkpoint.
    Paper-ready styling.
    """
    import matplotlib.gridspec as gridspec

    layer_names = {
        "post_attn": "Post-Attention",
        "post_ln2": "Post-LN2",
        "mlp_hidden": "MLP Hidden",
        "post_mlp": "Post-MLP",
    }

    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    steps = [r["step"] for r in all_results]
    n_ckpts = len(steps)
    cmap = plt.cm.plasma

    for i, result in enumerate(all_results):
        color = cmap(i / max(1, n_ckpts - 1))
        curves = result["per_k_metrics"]["pca_cumvar_curves"][layer]

        # Extract cumvar[K] for each K
        var_at_k = []
        for k in range(len(curves)):
            if len(curves[k]) > 0:
                if k == 0:
                    var_at_k.append(curves[k][0])
                elif len(curves[k]) >= k:
                    var_at_k.append(curves[k][k - 1])  # cumvar at K-th component
                else:
                    var_at_k.append(0.0)
            else:
                var_at_k.append(0.0)

        ax.plot(
            range(1, len(var_at_k) + 1),
            var_at_k,
            "-",
            color=color,
            alpha=0.8,
            linewidth=1.2,
        )

    ax.set_xlabel(r"$K$ (unique tokens)", fontsize=11)
    ax.set_ylabel("Cumulative Variance at $K$-th Component", fontsize=11)
    ax.set_title(layer_names.get(layer, layer), fontsize=12, fontweight="medium")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min(steps), vmax=max(steps))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Training Step", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    return fig


def create_summary_grid_plot(all_results: List[Dict]) -> Figure:
    """
    2x2 grid comparing K-scaling across all 4 layers.
    Paper-ready styling for ICML/NeurIPS.
    """
    import matplotlib.gridspec as gridspec

    # Paper-friendly layer names
    layer_names = {
        "post_attn": "Post-Attention",
        "post_ln2": "Post-LN2",
        "mlp_hidden": "MLP Hidden",
        "post_mlp": "Post-MLP",
    }
    layers = ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]

    # Set up figure with space for colorbar on the right
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.35, hspace=0.3)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cbar_ax = fig.add_subplot(gs[:, 2])

    steps = [r["step"] for r in all_results]
    n_ckpts = len(steps)

    # Use a perceptually uniform colormap
    cmap = plt.cm.plasma

    for ax, layer in zip(axes, layers):
        for i, result in enumerate(all_results):
            color = cmap(i / max(1, n_ckpts - 1))
            contrib_by_k = result["per_k_metrics"]["basis_contrib_by_k"][layer]
            ax.plot(
                range(1, len(contrib_by_k) + 1),
                contrib_by_k,
                "-",
                color=color,
                alpha=0.8,
                linewidth=1.2,
            )

        # Professional styling
        ax.set_xlabel(r"$K$ (unique tokens)", fontsize=10)
        ax.set_ylabel("Mean Basis Contribution", fontsize=10)
        ax.set_title(layer_names[layer], fontsize=11, fontweight="medium")
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Colorbar in dedicated axis (doesn't overlap plots)
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min(steps), vmax=max(steps))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Training Step", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    return fig


def fig_to_image(fig: Figure):
    """Convert matplotlib figure to PIL Image for wandb."""
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def analyze_checkpoint(
    ckpt_path: Path, n_samples: int = 24, device: str = "cuda", mode: str = "unique"
) -> Dict:
    """Analyze single checkpoint with all metrics."""
    model, meta = load_checkpoint(str(ckpt_path), device)
    seq_len = model.config.block_size
    vocab_size = model.config.vocab_size
    n_embd = model.config.n_embd

    if mode == "unique":
        tokens = generate_unique_prefix_sequences(
            n_samples, seq_len, vocab_size, device
        )
    else:
        tokens = generate_random_sequences(n_samples, seq_len, vocab_size, device)

    basis = extract_basis_embeddings(model, n_basis=n_samples)

    # Metric 2
    norms_before, norms_after = compute_pythagorean_norms(model, tokens)

    # Per-K metrics
    per_k = compute_per_k_metrics(model, tokens, basis, n_samples, seq_len)

    pca_results = {}
    for layer in ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]:
        pca_results[f"pca_{layer}"] = compute_pca_analysis(per_k["activations"][layer])

    return {
        "step": meta["step"],
        "n_embd": n_embd,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "norms_before_attn": norms_before.detach().cpu().numpy(),
        "norms_after_attn": norms_after.detach().cpu().numpy(),
        "norm_ratios": (norms_after / (norms_before + 1e-8)).detach().cpu().numpy(),
        "per_k_metrics": {
            "basis_contrib_by_k": per_k["basis_contrib_by_k"],
            "projections_heatmap": per_k["projections_heatmap"],
            "pca_cumvar_curves": per_k["pca_cumvar_curves"],
        },
        **pca_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Position Regression Metrics - Per-K Analysis"
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--mode", type=str, choices=["unique", "random"], default="unique"
    )
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "nanoGPT" / args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist!")
        return

    experiment_name = args.experiment_name or f"{args.checkpoint_dir}_{args.mode}"
    results_dir = (
        PROJECT_ROOT / "results" / f"position_regression_metrics_{experiment_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_steps = [0] + list(range(1000, 21000, 1000))
    available_steps = [
        s for s in checkpoint_steps if (checkpoint_dir / f"ckpt_{s:05d}.pt").exists()
    ]

    if not available_steps:
        print("Error: No checkpoints found!")
        return

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="nope-position-regression-metrics",
            name=f"per_k_{experiment_name}",
            config={
                "n_samples": args.n_samples,
                "checkpoint_steps": available_steps,
                "experiment": experiment_name,
                "mode": args.mode,
            },
        )

    all_results = []
    for step in tqdm(available_steps, desc="Analyzing"):
        ckpt_path = checkpoint_dir / f"ckpt_{step:05d}.pt"
        try:
            result = analyze_checkpoint(
                ckpt_path, args.n_samples, DEVICE, mode=args.mode
            )
            all_results.append(result)

            if use_wandb:
                layers = ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]
                metrics = {
                    "checkpoint/step": step,
                    "summary/mean_norm_ratio": float(np.mean(result["norm_ratios"])),
                }

                for layer in layers:
                    pca = result[f"pca_{layer}"]
                    metrics[f"pca/{layer}_sv1"] = pca["singular_values"][0]
                    metrics[f"pca/{layer}_top10_var"] = sum(
                        pca["explained_variance"][:10]
                    )

                    # Summary scalar only (removed individual K scalars)
                    contrib_by_k = result["per_k_metrics"]["basis_contrib_by_k"][layer]
                    metrics[f"summary/mean_basis_contrib_{layer}"] = float(
                        np.mean(contrib_by_k)
                    )

                # Aggregative plots (only after we have 2+ checkpoints)
                if len(all_results) > 1:
                    # K-Scaling plots per layer
                    for layer in layers:
                        fig_kscaling = create_k_scaling_plot(all_results, layer)
                        metrics[f"plots/k_scaling_{layer}"] = wandb.Image(
                            fig_to_image(fig_kscaling)
                        )

                        fig_pca_at_k = create_pca_var_at_k_plot(all_results, layer)
                        metrics[f"plots/pca_var_at_k_{layer}"] = wandb.Image(
                            fig_to_image(fig_pca_at_k)
                        )

                    # Summary grid comparing all layers
                    fig_grid = create_summary_grid_plot(all_results)
                    metrics["plots/k_scaling_all_layers"] = wandb.Image(
                        fig_to_image(fig_grid)
                    )

                    # Pythagorean norms
                    fig_norms = create_pythagorean_norms_plot(all_results)
                    metrics["plots/pythagorean_norms"] = wandb.Image(
                        fig_to_image(fig_norms)
                    )

                wandb.log(metrics, commit=True)
        except Exception as e:
            print(f"Step {step}: Error - {e}")
            import traceback

            traceback.print_exc()

    if all_results:
        layers = ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]
        final_result = all_results[-1]
        for layer in layers:
            fig_kscaling = create_k_scaling_plot(all_results, layer)
            fig_kscaling.savefig(
                results_dir / f"k_scaling_{layer}.png", dpi=300, bbox_inches="tight"
            )
            fig_kscaling.savefig(
                results_dir / f"k_scaling_{layer}.pdf", bbox_inches="tight"
            )
            plt.close(fig_kscaling)

            fig_pca_at_k = create_pca_var_at_k_plot(all_results, layer)
            fig_pca_at_k.savefig(
                results_dir / f"pca_var_at_k_{layer}.png", dpi=300, bbox_inches="tight"
            )
            fig_pca_at_k.savefig(
                results_dir / f"pca_var_at_k_{layer}.pdf", bbox_inches="tight"
            )
            plt.close(fig_pca_at_k)

            fig_pca_curves = create_pca_cumvar_curves_plot(final_result, layer)
            fig_pca_curves.savefig(
                results_dir
                / f"pca_cumvar_curves_{layer}_step{final_result['step']}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_pca_curves.savefig(
                results_dir
                / f"pca_cumvar_curves_{layer}_step{final_result['step']}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig_pca_curves)

        fig_grid = create_summary_grid_plot(all_results)
        fig_grid.savefig(
            results_dir / "k_scaling_all_layers.png", dpi=300, bbox_inches="tight"
        )
        fig_grid.savefig(results_dir / "k_scaling_all_layers.pdf", bbox_inches="tight")
        plt.close(fig_grid)

        fig_norms = create_pythagorean_norms_plot(all_results)
        fig_norms.savefig(
            results_dir / "pythagorean_norms.png", dpi=300, bbox_inches="tight"
        )
        fig_norms.savefig(results_dir / "pythagorean_norms.pdf", bbox_inches="tight")
        plt.close(fig_norms)

    if use_wandb:
        wandb.finish()
    print(f"\nAnalysis complete! Results in {results_dir}")


if __name__ == "__main__":
    main()
