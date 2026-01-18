"""
Position Regression Metrics Analysis

Extracts 3 key metrics across training checkpoints for nope-6layer-until-first-mlp model:
1. Basis validation via decoding vector theory
2. Pythagorean numbers (L2 norm squared) before/after attention
3. PCA/Singular values at each layer

Usage:
    CUDA_VISIBLE_DEVICES=4 python analysis_scripts/position_regression_metrics.py \
        --checkpoint-dir out-posreg-6layer-until-mlp \
        --experiment-name nope-6layer-until-first-mlp \
        --n-samples 24
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

    # Get model config from checkpoint
    model_args = checkpoint.get("model_args", {})

    # Create config
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

    # Create and load model
    model = GPT(gptconf)

    # Handle state dict with _orig_mod prefix (from torch.compile)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Extract training metadata
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
    Generate sequences where EACH sample i has i unique prefix tokens.

    Returns:
        tokens: (n_samples, seq_len) tensor - each row has different prefix length
    """
    sequences = torch.full(
        (n_samples, seq_len), base_token, dtype=torch.long, device=device
    )

    # For each sample i, fill first i positions with unique tokens
    for sample_idx in range(n_samples):
        n_unique = min(sample_idx, seq_len)
        for pos in range(n_unique):
            sequences[sample_idx, pos] = base_token + 1 + pos

    return sequences


def extract_basis_embeddings(
    model: GPT, base_token: int = 1000, n_basis: int = 24
) -> torch.Tensor:
    """
    Extract embeddings for unique prefix tokens: [1001, 1002, ..., 1024]

    Returns:
        basis: (n_basis, n_embd) - Raw embeddings of unique prefix tokens
    """
    unique_tokens = torch.arange(
        base_token + 1, base_token + n_basis + 1, device=next(model.parameters()).device
    )
    basis = model.transformer.wte(unique_tokens)
    return basis


def extract_weight_matrices(
    model: GPT, layer_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract W_v and W_o from attention layer.

    Returns:
        W_v: (n_embd, n_embd) - Value projection
        W_o: (n_embd, n_embd) - Output projection
    """
    block = model.transformer.h[layer_idx]
    c_attn_weight = block.attn.c_attn.weight.data  # (3*n_embd, n_embd)
    n_embd = model.config.n_embd

    # Extract W_v (last third of c_attn)
    W_v = c_attn_weight[2 * n_embd :, :].T  # (n_embd, n_embd)

    # Extract W_o
    W_o = block.attn.c_proj.weight.data.T  # (n_embd, n_embd)

    return W_v, W_o


def get_detailed_activations(
    model: GPT,
    tokens: torch.Tensor,
    layer_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Extract activations at all key points.

    Returns dict with:
    - post_attn: After attention (before residual)
    - post_ln2: After second LayerNorm
    - mlp_hidden: After GELU activation inside MLP
    - post_mlp: After whole MLP (before residual)
    """
    activations = {}

    with torch.no_grad():
        # Embedding
        tok_emb = model.transformer.wte(tokens)
        x = model.transformer.drop(tok_emb)

        # Get the specified block
        block = model.transformer.h[layer_idx]

        # Post-LN1 and Attention
        x_ln1 = block.ln_1(x)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # After first residual connection
        x = x + attn_out

        # Post-LN2 (BEFORE MLP)
        if block.use_ln2:
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.clone()
            mlp_input = x

        # Inside MLP - extract hidden activation after GELU
        mlp = block.mlp
        mlp_fc = mlp.c_fc(mlp_input)
        mlp_after_gelu = mlp.gelu(mlp_fc)
        activations["mlp_hidden"] = mlp_after_gelu.clone()

        # Complete MLP forward
        mlp_proj = mlp.c_proj(mlp_after_gelu)
        mlp_out = mlp.dropout(mlp_proj)
        activations["post_mlp"] = mlp_out.clone()

    return activations


def compute_basis_projections(
    activations: torch.Tensor, basis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dot product projection of activations onto basis vectors.

    Args:
        activations: (n_samples, seq_len, n_embd)
        basis: (n_basis, n_embd)

    Returns:
        projections: (n_samples, seq_len, n_basis) - Dot product with each basis vector
        total_contribution: (n_samples, seq_len) - Sum of absolute projections
    """
    # projections[i, j, k] = dot(activations[i, j], basis[k])
    projections = torch.einsum("ijk,mk->ijm", activations, basis)
    total_contribution = torch.abs(projections).sum(dim=-1)
    return projections, total_contribution


def compute_pythagorean_norms(
    model: GPT, tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ||v||² before and after attention.

    Returns:
        norms_before: (n_samples, seq_len) - ||W_v @ LN1(e)||²
        norms_after: (n_samples, seq_len) - ||attn_out||²
    """
    with torch.no_grad():
        # Get embeddings
        tok_emb = model.transformer.wte(tokens)

        # First block
        block = model.transformer.h[0]

        # Apply LN1
        ln1_out = block.ln_1(tok_emb)

        # Extract W_v
        W_v, _ = extract_weight_matrices(model)

        # Compute value vectors: v = ln1_out @ W_v.T
        value_vectors = ln1_out @ W_v.T

        # Before attention: ||v||²
        norms_before = (value_vectors**2).sum(dim=-1)

        # After attention
        attn_out = block.attn(ln1_out)
        norms_after = (attn_out**2).sum(dim=-1)

    return norms_before, norms_after


def compute_pca_analysis(
    activations_dict: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """
    Compute singular value decomposition at each layer.

    Args:
        activations_dict: {
            'post_attn': (n_samples, seq_len, n_embd),
            'post_ln2': (n_samples, seq_len, n_embd),
            'mlp_hidden': (n_samples, seq_len, 4*n_embd),
            'post_mlp': (n_samples, seq_len, n_embd),
        }

    Returns:
        results: Dict with singular values and explained variance for each layer
    """
    results = {}

    for layer_name, acts in activations_dict.items():
        # Reshape to (n_samples * seq_len, d_model)
        n_samples, seq_len, d_model = acts.shape
        acts_flat = acts.reshape(-1, d_model).cpu().numpy()

        # Center the data
        acts_centered = acts_flat - acts_flat.mean(axis=0, keepdims=True)

        # SVD
        U, S, Vh = np.linalg.svd(acts_centered, full_matrices=False)

        # Explained variance
        explained_var = (S**2) / (S**2).sum()

        results[f"singular_values_{layer_name}"] = S.tolist()
        results[f"explained_variance_{layer_name}"] = explained_var.tolist()

    return results


def analyze_checkpoint(
    ckpt_path: Path, n_samples: int = 24, device: str = "cuda"
) -> Dict:
    """Analyze single checkpoint - extract all 3 metrics."""

    # Load model
    model, meta = load_checkpoint(str(ckpt_path), device)
    vocab_size = model.config.vocab_size
    seq_len = model.config.block_size
    n_embd = model.config.n_embd

    # Generate sequences (same as t-SNE)
    tokens = generate_unique_prefix_sequences(n_samples, seq_len, vocab_size, device)

    # Extract basis
    basis = extract_basis_embeddings(model, n_basis=n_samples)

    # Get activations at all layers
    all_activations = []
    for i in range(n_samples):
        acts = get_detailed_activations(model, tokens[i : i + 1])
        all_activations.append(acts)

    # Stack activations: {layer: (n_samples, seq_len, d_model)}
    stacked_acts = {}
    for layer in ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]:
        stacked_acts[layer] = torch.cat([a[layer] for a in all_activations], dim=0)

    # Metric 1: Basis projections
    proj_attn, contrib_attn = compute_basis_projections(
        stacked_acts["post_attn"], basis
    )
    proj_ln2, contrib_ln2 = compute_basis_projections(stacked_acts["post_ln2"], basis)

    # Metric 2: Pythagorean norms
    norms_before, norms_after = compute_pythagorean_norms(model, tokens)

    # Metric 3: PCA
    pca_results = compute_pca_analysis(stacked_acts)

    return {
        "step": meta["step"],
        "n_embd": n_embd,
        "n_samples": n_samples,
        "seq_len": seq_len,
        # Metric 1
        "basis_projections_post_attn": proj_attn.detach().cpu().numpy(),
        "basis_contributions_post_attn": contrib_attn.detach().cpu().numpy(),
        "basis_projections_post_ln2": proj_ln2.detach().cpu().numpy(),
        "basis_contributions_post_ln2": contrib_ln2.detach().cpu().numpy(),
        # Metric 2
        "norms_before_attn": norms_before.detach().cpu().numpy(),
        "norms_after_attn": norms_after.detach().cpu().numpy(),
        "norm_ratios": (norms_after / (norms_before + 1e-8)).detach().cpu().numpy(),
        # Metric 3
        **pca_results,
    }


def create_visualizations(all_results, save_dir: Path):
    """Create all visualization plots."""
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in all_results]

    # 1. Basis Contributions Evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (layer, ax) in enumerate(zip(["post_attn", "post_ln2"], axes)):
        key = f"basis_contributions_{layer}"
        # Average over samples and positions
        mean_contrib = [r[key].mean() for r in all_results]
        std_contrib = [r[key].std() for r in all_results]

        ax.plot(steps, mean_contrib, "o-", linewidth=2, markersize=6, label="Mean")
        ax.fill_between(
            steps,
            np.array(mean_contrib) - np.array(std_contrib),
            np.array(mean_contrib) + np.array(std_contrib),
            alpha=0.3,
        )
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Total Basis Contribution", fontsize=11)
        ax.set_title(
            f"{layer.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        plots_dir / "basis_contributions_evolution.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # 2. Pythagorean Norms Evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before/After norms
    ax = axes[0]
    mean_before = [r["norms_before_attn"].mean() for r in all_results]
    mean_after = [r["norms_after_attn"].mean() for r in all_results]
    ax.plot(
        steps, mean_before, "o-", linewidth=2, markersize=6, label="Before Attention"
    )
    ax.plot(steps, mean_after, "s-", linewidth=2, markersize=6, label="After Attention")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Mean ||v||²", fontsize=11)
    ax.set_title("L2 Norm Squared Evolution", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Norm ratio
    ax = axes[1]
    mean_ratio = [r["norm_ratios"].mean() for r in all_results]
    ax.plot(steps, mean_ratio, "o-", linewidth=2, markersize=6, color="purple")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Ratio=1")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Norm Ratio (After/Before)", fontsize=11)
    ax.set_title("Attention Norm Transformation", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        plots_dir / "pythagorean_norms_evolution.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # 3. Singular Values Spectrum (4 layers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    layers = ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]

    for layer, ax in zip(layers, axes):
        key = f"singular_values_{layer}"

        # Plot top 50 singular values for each checkpoint
        for i, result in enumerate(all_results):
            s_vals = np.array(result[key])[:50]  # Top 50
            color = plt.cm.viridis(i / len(all_results))
            ax.plot(s_vals, alpha=0.7, color=color, linewidth=1.5)

        ax.set_xlabel("Component Index", fontsize=10)
        ax.set_ylabel("Singular Value", fontsize=10)
        ax.set_title(
            f"{layer.replace('_', ' ').title()}", fontsize=11, fontweight="bold"
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Add colorbar for steps
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(steps), vmax=max(steps))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Training Step", fontsize=11)

    plt.tight_layout()
    fig.savefig(
        plots_dir / "singular_values_spectrum.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # 4. Cumulative Explained Variance
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for layer, ax in zip(layers, axes):
        key = f"explained_variance_{layer}"

        for i, result in enumerate(all_results):
            exp_var = np.array(result[key])
            cumsum = np.cumsum(exp_var)[:100]  # Top 100 components
            color = plt.cm.viridis(i / len(all_results))
            ax.plot(cumsum, alpha=0.7, color=color, linewidth=1.5)

        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90%")
        ax.axhline(y=0.99, color="orange", linestyle="--", alpha=0.5, label="99%")
        ax.set_xlabel("Number of Components", fontsize=10)
        ax.set_ylabel("Cumulative Explained Variance", fontsize=10)
        ax.set_title(
            f"{layer.replace('_', ' ').title()}", fontsize=11, fontweight="bold"
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(steps), vmax=max(steps))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Training Step", fontsize=11)

    plt.tight_layout()
    fig.savefig(
        plots_dir / "explained_variance_cumulative.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"\n✓ All visualizations saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Position Regression Metrics Analysis")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory (e.g., out-posreg-6layer-until-mlp)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for wandb and results",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=24,
        help="Number of samples (each with varying prefix diversity)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    args = parser.parse_args()

    # Set up directories
    checkpoint_dir = PROJECT_ROOT / "nanoGPT" / args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist!")
        return

    experiment_name = args.experiment_name or args.checkpoint_dir
    results_dir = (
        PROJECT_ROOT / "results" / f"position_regression_metrics_{experiment_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Verify checkpoints exist
    checkpoint_steps = list(range(1000, 21000, 1000))  # Every 1000 steps
    print(f"\n{'=' * 70}")
    print(f"Position Regression Metrics - {experiment_name}")
    print(f"{'=' * 70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Verifying checkpoints...")

    available_steps = []
    for step in checkpoint_steps:
        ckpt_path = checkpoint_dir / f"ckpt_{step:05d}.pt"
        if ckpt_path.exists():
            available_steps.append(step)
        else:
            print(f"  Warning: Checkpoint {step} not found")

    if not available_steps:
        print("Error: No checkpoints found!")
        return

    print(f"Found {len(available_steps)} checkpoints: {available_steps}")

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="nope-position-regression-metrics",
            name=f"metrics_{experiment_name}",
            config={
                "n_samples": args.n_samples,
                "checkpoint_steps": available_steps,
                "experiment": experiment_name,
            },
        )
        print(
            f"\nWandB initialized: nope-position-regression-metrics/metrics_{experiment_name}"
        )

    # Analyze all checkpoints
    all_results = []

    for step in tqdm(available_steps, desc="Analyzing checkpoints"):
        ckpt_path = checkpoint_dir / f"ckpt_{step:05d}.pt"

        print(f"\n--- Checkpoint {step} ---")
        try:
            result = analyze_checkpoint(ckpt_path, args.n_samples, DEVICE)
            all_results.append(result)

            # Log to wandb
            if use_wandb:
                # Log scalar metrics
                wandb.log(
                    {
                        "step": step,
                        "basis_contrib_post_attn_mean": result[
                            "basis_contributions_post_attn"
                        ].mean(),
                        "basis_contrib_post_ln2_mean": result[
                            "basis_contributions_post_ln2"
                        ].mean(),
                        "norm_before_mean": result["norms_before_attn"].mean(),
                        "norm_after_mean": result["norms_after_attn"].mean(),
                        "norm_ratio_mean": result["norm_ratios"].mean(),
                        "singular_value_1_post_ln2": result["singular_values_post_ln2"][
                            0
                        ],
                        "explained_var_top10_post_ln2": sum(
                            result["explained_variance_post_ln2"][:10]
                        ),
                    }
                )

            print(f"  ✓ Analysis complete")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("Creating visualizations...")
    create_visualizations(all_results, results_dir)

    # Save summary
    import json

    summary_path = results_dir / "summary.json"
    summary = {
        "experiment": experiment_name,
        "n_samples": args.n_samples,
        "checkpoints_analyzed": [r["step"] for r in all_results],
        "n_embd": all_results[0]["n_embd"] if all_results else None,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_path}")

    if use_wandb:
        # Log final visualizations to wandb
        plots_dir = results_dir / "plots"
        for plot_file in plots_dir.glob("*.png"):
            wandb.log({f"final_plots/{plot_file.stem}": wandb.Image(str(plot_file))})

        wandb.finish()
        print("\nWandB run finished.")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Results saved to: {results_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
