"""
OWT Sample-Level Position Probing Analysis

Trains linear probes on each component (layer) of all 4 OWT architectures to understand
how position is encoded at the sample level throughout the network.

For each architecture (NoPE+LN, NoPE+BN2, NoPE+NoLN2, Baseline+PE):
  - Extract activations at each layer for individual samples
  - Train Ridge regression probes to predict position
  - Report Full R², Direction R², Norm R² at each layer

This provides a layer-by-layer view of positional information flow.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import Ridge
from scipy import stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "owt_sample_level_probing"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
PLOTS_DIR = PROJECT_ROOT / "overleaf" / "nopos---claude-version" / "plots"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    short_name: str  # For plot labels
    checkpoint_path: str
    use_positional_embedding: bool
    use_batchnorm_ln2: bool
    skip_ln2: bool


EXPERIMENTS = [
    ExperimentConfig(
        name="NoPE + LayerNorm",
        short_name="NoPE+LN",
        checkpoint_path="out-nope-owt-ln/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=False,
        skip_ln2=False,
    ),
    ExperimentConfig(
        name="NoPE + BatchNorm2",
        short_name="NoPE+BN2",
        checkpoint_path="out-nope-owt-bn2/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=True,
        skip_ln2=False,
    ),
    ExperimentConfig(
        name="NoPE + No LN2",
        short_name="NoPE+NoLN2",
        checkpoint_path="out-nope-owt-no-ln2/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=False,
        skip_ln2=True,
    ),
    ExperimentConfig(
        name="Baseline + PE",
        short_name="Baseline+PE",
        checkpoint_path="out-baseline-owt-pe/ckpt.pt",
        use_positional_embedding=True,
        use_batchnorm_ln2=False,
        skip_ln2=False,
    ),
]


# Layer display names for publication
LAYER_NAMES = {
    "embed": "Embedding",
    "post_ln1": "Post-LN1",
    "post_attn": "Post-Attn",
    "post_attn_residual": "Post-Attn+Res",
    "post_ln2": "Post-LN2",
    "post_mlp_residual": "Post-MLP+Res",
}

LAYERS = [
    "embed",
    "post_ln1",
    "post_attn",
    "post_attn_residual",
    "post_ln2",
    "post_mlp_residual",
]


def load_trained_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Load a trained model from checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / exp.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 512),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=exp.use_positional_embedding,
        norm_type=model_args.get("norm_type", "layernorm"),
        bias=model_args.get("bias", False),
        skip_ln2=exp.skip_ln2,
        use_batchnorm_ln2=exp.use_batchnorm_ln2,
    )

    model = GPT(config)

    # Unwrap torch.compile prefix if present
    state_dict = checkpoint["model"]
    unwrapped = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[10:]
        unwrapped[k] = v

    model.load_state_dict(unwrapped)
    model.eval()
    model.to(DEVICE)

    return model, config


def get_activations(
    model: GPT, tokens: torch.Tensor, skip_ln2: bool = False
) -> Dict[str, torch.Tensor]:
    """Get activations at key layers."""
    activations = {}

    with torch.no_grad():
        # Token embeddings
        tok_emb = model.transformer.wte(tokens)

        # Add positional embeddings if available
        if hasattr(model.transformer, "wpe") and model.config.use_positional_embedding:
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        activations["embed"] = x.detach()

        block = model.transformer.h[0]

        # Post LN1
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.detach()

        # Post attention (before residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.detach()

        # Post attention residual
        x = x + attn_out
        activations["post_attn_residual"] = x.detach()

        # Post LN2 (if not skipped)
        if not skip_ln2 and hasattr(block, "ln_2"):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.detach()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.detach()  # Same as post_attn_residual
            mlp_input = x

        # Post MLP residual
        mlp_out = block.mlp(mlp_input)
        x = x + mlp_out
        activations["post_mlp_residual"] = x.detach()

    return activations


def compute_probing_metrics(
    activations: np.ndarray,
    positions: np.ndarray,
    train_ratio: float = 0.8,
) -> Dict[str, float]:
    """Compute probing metrics for a set of activations."""
    n_samples = len(positions)
    n_train = int(train_ratio * n_samples)

    # Random split
    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Compute norms and directions
    norms = np.linalg.norm(activations, axis=1)
    directions = activations / (norms[:, np.newaxis] + 1e-8)

    # Train probes
    def fit_ridge(X_train, y_train, X_test, y_test, alpha=1.0):
        probe = Ridge(alpha=alpha)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return max(0, r2)

    # Full probe
    full_r2 = fit_ridge(
        activations[train_idx],
        positions[train_idx],
        activations[test_idx],
        positions[test_idx],
    )

    # Direction probe
    dir_r2 = fit_ridge(
        directions[train_idx],
        positions[train_idx],
        directions[test_idx],
        positions[test_idx],
    )

    # Norm probe
    norm_r2 = fit_ridge(
        norms[train_idx].reshape(-1, 1),
        positions[train_idx],
        norms[test_idx].reshape(-1, 1),
        positions[test_idx],
    )

    return {
        "full_r2": float(full_r2),
        "direction_r2": float(dir_r2),
        "norm_r2": float(norm_r2),
    }


def analyze_model(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    skip_ln2: bool = False,
    n_samples: int = 500,
    context_length: Optional[int] = None,
) -> Dict:
    """Run full analysis on a model."""
    ctx = context_length or config.block_size
    vocab_size = config.vocab_size

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(f"Config: {config.n_layer}L, {config.n_head}H, {config.n_embd}D, ctx={ctx}")
    print(f"{'=' * 60}")

    # Collect activations
    all_activations = {layer: [] for layer in LAYERS}
    all_positions = []

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Processing sample {i + 1}/{n_samples}...")

        tokens = torch.randint(0, vocab_size, (1, ctx), device=DEVICE)
        acts = get_activations(model, tokens, skip_ln2=skip_ln2)

        for layer in LAYERS:
            if layer in acts:
                all_activations[layer].append(acts[layer][0].cpu().numpy())

        all_positions.append(np.arange(ctx))

    # Stack activations
    for layer in LAYERS:
        all_activations[layer] = np.vstack(all_activations[layer])

    positions = np.concatenate(all_positions)

    # Compute metrics for each layer
    results = {"model_name": model_name, "n_samples": n_samples, "context": ctx}

    print(f"\n{'Layer':<20} {'Full R²':>10} {'Dir R²':>10} {'Norm R²':>10}")
    print("-" * 54)

    for layer in LAYERS:
        metrics = compute_probing_metrics(all_activations[layer], positions)
        results[layer] = metrics

        print(
            f"{layer:<20} {metrics['full_r2']:>10.4f} {metrics['direction_r2']:>10.4f} "
            f"{metrics['norm_r2']:>10.4f}"
        )

    return results


def create_layerwise_figure(all_results: Dict, output_path: Path):
    """Create publication-quality figure showing layer-wise R² for all architectures."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors for different architectures
    colors = {
        "NoPE + LayerNorm": "#1f77b4",
        "NoPE + BatchNorm2": "#ff7f0e",
        "NoPE + No LN2": "#2ca02c",
        "Baseline + PE": "#d62728",
    }

    markers = {
        "NoPE + LayerNorm": "o",
        "NoPE + BatchNorm2": "s",
        "NoPE + No LN2": "^",
        "Baseline + PE": "D",
    }

    x_positions = np.arange(len(LAYERS))
    x_labels = [LAYER_NAMES[l] for l in LAYERS]

    metrics = ["full_r2", "direction_r2", "norm_r2"]
    titles = ["Full Activation R²", "Direction-only R²", "Norm-only R²"]

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]

        for exp_name, data in all_results.items():
            if "trained" not in data:
                continue

            trained = data["trained"]
            y_values = [trained[layer][metric] for layer in LAYERS]

            ax.plot(
                x_positions,
                y_values,
                marker=markers[exp_name],
                color=colors[exp_name],
                linewidth=2,
                markersize=8,
                label=exp_name,
            )

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        if ax_idx == 2:  # Add legend to rightmost plot
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path.with_suffix('.pdf')}")


def create_comparison_heatmap(all_results: Dict, output_path: Path):
    """Create heatmap comparing Full R² across layers and architectures."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ["full_r2", "direction_r2", "norm_r2"]
    titles = ["Full R²", "Direction R²", "Norm R²"]

    exp_names = [exp.name for exp in EXPERIMENTS if exp.name in all_results]

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        # Build matrix
        matrix = np.zeros((len(exp_names), len(LAYERS)))

        for i, exp_name in enumerate(exp_names):
            if "trained" in all_results[exp_name]:
                trained = all_results[exp_name]["trained"]
                for j, layer in enumerate(LAYERS):
                    matrix[i, j] = trained[layer][metric]

        im = axes[ax_idx].imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)

        axes[ax_idx].set_xticks(np.arange(len(LAYERS)))
        axes[ax_idx].set_xticklabels(
            [LAYER_NAMES[l] for l in LAYERS], rotation=45, ha="right", fontsize=10
        )
        axes[ax_idx].set_yticks(np.arange(len(exp_names)))
        axes[ax_idx].set_yticklabels(exp_names, fontsize=10)
        axes[ax_idx].set_title(title, fontsize=14, fontweight="bold")

        # Add text annotations
        for i in range(len(exp_names)):
            for j in range(len(LAYERS)):
                text = axes[ax_idx].text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 0.5 else "black",
                    fontsize=8,
                )

        plt.colorbar(im, ax=axes[ax_idx], shrink=0.8)

    plt.tight_layout()

    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap: {output_path.with_suffix('.pdf')}")


def main():
    print("=" * 70)
    print("OWT SAMPLE-LEVEL POSITION PROBING ANALYSIS")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'#' * 70}")
        print(f"# Experiment: {exp.name}")
        print(f"{'#' * 70}")

        # Load trained model
        try:
            trained_model, config = load_trained_model(exp)
            print(f"\nLoaded checkpoint: {exp.checkpoint_path}")
        except FileNotFoundError as e:
            print(f"\nSkipping {exp.name}: {e}")
            continue

        # Analyze trained model
        trained_results = analyze_model(
            trained_model,
            config,
            f"{exp.name} (trained)",
            skip_ln2=exp.skip_ln2,
            n_samples=500,
            context_length=512,
        )

        # Store results
        all_results[exp.name] = {
            "trained": trained_results,
            "config": {
                "use_positional_embedding": exp.use_positional_embedding,
                "use_batchnorm_ln2": exp.use_batchnorm_ln2,
                "skip_ln2": exp.skip_ln2,
            },
        }

        # Clean up
        del trained_model
        torch.cuda.empty_cache()

    # Save results
    output_path = RESULTS_DIR / "owt_sample_level_probing_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 70}")

    # Create figures
    print("\nGenerating figures...")

    # Line plot showing layer-wise progression
    create_layerwise_figure(all_results, PLOTS_DIR / "owt_sample_level_layerwise")

    # Heatmap comparison
    create_comparison_heatmap(all_results, PLOTS_DIR / "owt_sample_level_heatmap")

    # Print final summary table
    print("\n" + "=" * 90)
    print("FINAL SUMMARY: Sample-Level Position Probing R² by Layer")
    print("=" * 90)

    # Header
    header = f"{'Architecture':<20}"
    for layer in LAYERS:
        header += f" {LAYER_NAMES[layer]:>12}"
    print(header)
    print("-" * (20 + 13 * len(LAYERS)))

    # Full R² for each architecture
    print("\nFull R²:")
    for name, data in all_results.items():
        if "trained" in data:
            row = f"{name:<20}"
            for layer in LAYERS:
                row += f" {data['trained'][layer]['full_r2']:>12.3f}"
            print(row)

    # Direction R² for each architecture
    print("\nDirection R²:")
    for name, data in all_results.items():
        if "trained" in data:
            row = f"{name:<20}"
            for layer in LAYERS:
                row += f" {data['trained'][layer]['direction_r2']:>12.3f}"
            print(row)

    # Norm R² for each architecture
    print("\nNorm R²:")
    for name, data in all_results.items():
        if "trained" in data:
            row = f"{name:<20}"
            for layer in LAYERS:
                row += f" {data['trained'][layer]['norm_r2']:>12.3f}"
            print(row)

    return all_results


if __name__ == "__main__":
    results = main()
