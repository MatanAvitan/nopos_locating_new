"""
OWT Sample-Level Position Probing Analysis - ACCURACY METRIC

Trains linear classifiers on each component (layer) of all 4 OWT architectures
to understand how well position can be decoded using CLASSIFICATION ACCURACY.

This version treats position prediction as a 512-class classification problem
rather than regression, providing a more interpretable metric.

For each architecture (NoPE+LN, NoPE+BN2, NoPE+NoLN2, Baseline+PE):
  - Extract activations at each layer for individual samples
  - Train linear classifiers to predict discrete position
  - Report classification accuracy at each layer
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

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
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


def create_random_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Create a randomly initialized model with the same config as trained."""
    # Use same config as trained models
    config = GPTConfig(
        n_layer=1,
        n_head=12,
        n_embd=768,
        block_size=512,
        vocab_size=50304,
        dropout=0.0,
        use_positional_embedding=exp.use_positional_embedding,
        norm_type="layernorm",
        bias=False,
        skip_ln2=exp.skip_ln2,
        use_batchnorm_ln2=exp.use_batchnorm_ln2,
    )

    model = GPT(config)
    model.eval()
    model.to(DEVICE)

    return model, config


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
    context_length: int = 512,
    n_bins: int = 32,  # Bin positions into 32 groups for faster classification
) -> Dict[str, float]:
    """Compute probing metrics using classification accuracy.

    We bin positions into n_bins groups for computational efficiency.
    E.g., with context_length=512 and n_bins=32, each bin covers 16 positions.
    """
    n_samples = len(positions)
    n_train = int(train_ratio * n_samples)

    # Bin positions into n_bins classes
    bin_size = context_length / n_bins
    binned_positions = (positions / bin_size).astype(int)
    binned_positions = np.clip(binned_positions, 0, n_bins - 1)  # Safety clamp

    # Random split
    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Compute norms and directions
    norms = np.linalg.norm(activations, axis=1)
    directions = activations / (norms[:, np.newaxis] + 1e-8)

    # Scale features for better convergence
    scaler_full = StandardScaler()
    scaler_dir = StandardScaler()
    scaler_norm = StandardScaler()

    # Prepare data
    X_train_full = scaler_full.fit_transform(activations[train_idx])
    X_test_full = scaler_full.transform(activations[test_idx])

    X_train_dir = scaler_dir.fit_transform(directions[train_idx])
    X_test_dir = scaler_dir.transform(directions[test_idx])

    X_train_norm = scaler_norm.fit_transform(norms[train_idx].reshape(-1, 1))
    X_test_norm = scaler_norm.transform(norms[test_idx].reshape(-1, 1))

    y_train = binned_positions[train_idx]
    y_test = binned_positions[test_idx]

    # Check if there's any variance in activations
    def has_variance(X):
        return np.std(X) > 1e-6

    results = {}

    # Full probe (classification)
    if has_variance(X_train_full):
        try:
            clf_full = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
            clf_full.fit(X_train_full, y_train)
            full_acc = clf_full.score(X_test_full, y_test)
        except Exception as e:
            print(f"    Full probe failed: {e}")
            full_acc = 1 / n_bins  # Random chance
    else:
        full_acc = 1 / n_bins

    # Direction probe (classification)
    if has_variance(X_train_dir):
        try:
            clf_dir = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
            clf_dir.fit(X_train_dir, y_train)
            dir_acc = clf_dir.score(X_test_dir, y_test)
        except Exception as e:
            print(f"    Direction probe failed: {e}")
            dir_acc = 1 / n_bins
    else:
        dir_acc = 1 / n_bins

    # Norm probe (classification)
    if has_variance(X_train_norm):
        try:
            clf_norm = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
            clf_norm.fit(X_train_norm, y_train)
            norm_acc = clf_norm.score(X_test_norm, y_test)
        except Exception as e:
            print(f"    Norm probe failed: {e}")
            norm_acc = 1 / n_bins
    else:
        norm_acc = 1 / n_bins

    # Random baseline accuracy
    random_acc = 1 / n_bins

    return {
        "full_accuracy": float(full_acc),
        "direction_accuracy": float(dir_acc),
        "norm_accuracy": float(norm_acc),
        "random_baseline": float(random_acc),
        "n_bins": n_bins,
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
    results = {
        "model_name": model_name,
        "n_samples": n_samples,
        "context": ctx,
        "n_classes": ctx,
    }

    print(
        f"\n{'Layer':<20} {'Full Acc':>10} {'Dir Acc':>10} {'Norm Acc':>10} {'Random':>10}"
    )
    print("-" * 64)

    for layer in LAYERS:
        print(f"  Training probes for {layer}...")
        metrics = compute_probing_metrics(
            all_activations[layer], positions, context_length=ctx
        )
        results[layer] = metrics

        print(
            f"{layer:<20} {metrics['full_accuracy']:>10.4f} {metrics['direction_accuracy']:>10.4f} "
            f"{metrics['norm_accuracy']:>10.4f} {metrics['random_baseline']:>10.4f}"
        )

    return results


def create_layerwise_figure(
    all_results: Dict, output_path: Path, init_type: str = "random"
):
    """Create publication-quality figure showing layer-wise accuracy for all architectures."""

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

    metrics = ["full_accuracy", "direction_accuracy", "norm_accuracy"]
    titles = ["Full Activation Acc", "Direction-only Acc", "Norm-only Acc"]

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]

        for exp_name, data in all_results.items():
            if init_type not in data:
                continue

            init_data = data[init_type]
            y_values = [init_data[layer][metric] for layer in LAYERS]

            ax.plot(
                x_positions,
                y_values,
                marker=markers[exp_name],
                color=colors[exp_name],
                linewidth=2,
                markersize=8,
                label=exp_name,
            )

        # Add random baseline (32 bins)
        random_baseline = 1 / 32
        ax.axhline(
            y=random_baseline,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Random (3.1%)",
        )

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(-0.01, 1.05)
        ax.grid(True, alpha=0.3)

        if ax_idx == 2:  # Add legend to rightmost plot
            ax.legend(loc="upper right", fontsize=9)

    title_suffix = (
        "Random Initialization" if init_type == "random" else "Trained Models"
    )
    plt.suptitle(
        f"Position Classification Accuracy ({title_suffix})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path.with_suffix('.pdf')}")


def create_comparison_heatmap(
    all_results: Dict, output_path: Path, init_type: str = "random"
):
    """Create heatmap comparing accuracy across layers and architectures."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ["full_accuracy", "direction_accuracy", "norm_accuracy"]
    titles = ["Full Acc", "Direction Acc", "Norm Acc"]

    exp_names = [exp.name for exp in EXPERIMENTS if exp.name in all_results]

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        # Build matrix
        matrix = np.zeros((len(exp_names), len(LAYERS)))

        for i, exp_name in enumerate(exp_names):
            if init_type in all_results[exp_name]:
                init_data = all_results[exp_name][init_type]
                for j, layer in enumerate(LAYERS):
                    matrix[i, j] = init_data[layer][metric]

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
                val = matrix[i, j]
                text = axes[ax_idx].text(
                    j,
                    i,
                    f"{val:.2f}" if val >= 0.01 else "<.01",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=8,
                )

        plt.colorbar(im, ax=axes[ax_idx], shrink=0.8)

    title_suffix = (
        "Random Initialization" if init_type == "random" else "Trained Models"
    )
    plt.suptitle(
        f"Position Classification ({title_suffix})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap: {output_path.with_suffix('.pdf')}")


def main(analyze_trained: bool = False, analyze_random: bool = True):
    print("=" * 70)
    print("OWT SAMPLE-LEVEL POSITION PROBING - ACCURACY METRIC")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Analyzing random init: {analyze_random}")
    print(f"Analyzing trained: {analyze_trained}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'#' * 70}")
        print(f"# Experiment: {exp.name}")
        print(f"{'#' * 70}")

        all_results[exp.name] = {
            "config": {
                "use_positional_embedding": exp.use_positional_embedding,
                "use_batchnorm_ln2": exp.use_batchnorm_ln2,
                "skip_ln2": exp.skip_ln2,
            },
        }

        # Analyze random initialization
        if analyze_random:
            print(f"\n--- Random Initialization ---")
            random_model, config = create_random_model(exp)
            random_results = analyze_model(
                random_model,
                config,
                f"{exp.name} (random)",
                skip_ln2=exp.skip_ln2,
                n_samples=200,  # Fewer samples for faster classification
                context_length=512,
            )
            all_results[exp.name]["random"] = random_results
            del random_model
            torch.cuda.empty_cache()

        # Analyze trained model
        if analyze_trained:
            print(f"\n--- Trained Model ---")
            try:
                trained_model, config = load_trained_model(exp)
                trained_results = analyze_model(
                    trained_model,
                    config,
                    f"{exp.name} (trained)",
                    skip_ln2=exp.skip_ln2,
                    n_samples=200,
                    context_length=512,
                )
                all_results[exp.name]["trained"] = trained_results
                del trained_model
                torch.cuda.empty_cache()
            except FileNotFoundError as e:
                print(f"  Skipping trained: {e}")

    # Save results
    output_path = RESULTS_DIR / "owt_sample_level_probing_accuracy_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 70}")

    # Create figures
    print("\nGenerating figures...")

    if analyze_random:
        create_layerwise_figure(
            all_results,
            PLOTS_DIR / "owt_sample_level_accuracy_random",
            init_type="random",
        )
        create_comparison_heatmap(
            all_results,
            PLOTS_DIR / "owt_sample_level_accuracy_heatmap_random",
            init_type="random",
        )

    if analyze_trained:
        create_layerwise_figure(
            all_results,
            PLOTS_DIR / "owt_sample_level_accuracy_trained",
            init_type="trained",
        )
        create_comparison_heatmap(
            all_results,
            PLOTS_DIR / "owt_sample_level_accuracy_heatmap_trained",
            init_type="trained",
        )

    # Print final summary table
    n_bins = 32  # Must match compute_probing_metrics default
    print("\n" + "=" * 90)
    print("FINAL SUMMARY: Position Classification Accuracy by Layer")
    print(f"(Positions binned into {n_bins} groups)")
    print("=" * 90)
    print(f"Random baseline (chance): {1 / n_bins:.4f} = {100 / n_bins:.2f}%")

    for init_type in ["random", "trained"]:
        has_data = any(init_type in data for data in all_results.values())
        if not has_data:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {init_type.upper()} INITIALIZATION")
        print(f"{'=' * 60}")

        # Header
        header = f"{'Architecture':<20}"
        for layer in LAYERS:
            header += f" {LAYER_NAMES[layer]:>12}"
        print(header)
        print("-" * (20 + 13 * len(LAYERS)))

        # Full accuracy for each architecture
        print("\nFull Accuracy:")
        for name, data in all_results.items():
            if init_type in data:
                row = f"{name:<20}"
                for layer in LAYERS:
                    row += f" {data[init_type][layer]['full_accuracy']:>12.3f}"
                print(row)

        # Direction accuracy for each architecture
        print("\nDirection Accuracy:")
        for name, data in all_results.items():
            if init_type in data:
                row = f"{name:<20}"
                for layer in LAYERS:
                    row += f" {data[init_type][layer]['direction_accuracy']:>12.3f}"
                print(row)

        # Norm accuracy for each architecture
        print("\nNorm Accuracy:")
        for name, data in all_results.items():
            if init_type in data:
                row = f"{name:<20}"
                for layer in LAYERS:
                    row += f" {data[init_type][layer]['norm_accuracy']:>12.3f}"
                print(row)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", action="store_true", help="Analyze trained models")
    parser.add_argument(
        "--random", action="store_true", help="Analyze random init (default)"
    )
    parser.add_argument("--both", action="store_true", help="Analyze both")
    args = parser.parse_args()

    if args.both:
        results = main(analyze_trained=True, analyze_random=True)
    elif args.trained:
        results = main(analyze_trained=True, analyze_random=False)
    else:
        # Default: analyze random init
        results = main(analyze_trained=False, analyze_random=True)
