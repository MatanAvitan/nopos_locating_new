"""
Fast OWT Trained Model Probing - Uses fewer samples and simpler classifier.

Compares random vs trained models on position classification accuracy.
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "owt_sample_level_probing"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
PLOTS_DIR = PROJECT_ROOT / "overleaf" / "nopos---claude-version" / "plots"


@dataclass
class ExperimentConfig:
    name: str
    short_name: str
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

LAYERS = [
    "embed",
    "post_ln1",
    "post_attn",
    "post_attn_residual",
    "post_ln2",
    "post_mlp_residual",
]
LAYER_NAMES = {
    "embed": "Embed",
    "post_ln1": "Post-LN1",
    "post_attn": "Post-Attn",
    "post_attn_residual": "Attn+Res",
    "post_ln2": "Post-LN2",
    "post_mlp_residual": "MLP+Res",
}


def create_random_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Create a randomly initialized model."""
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
    state_dict = checkpoint["model"]
    unwrapped = {
        (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()
    }
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
        tok_emb = model.transformer.wte(tokens)
        if hasattr(model.transformer, "wpe") and model.config.use_positional_embedding:
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        activations["embed"] = x.detach()
        block = model.transformer.h[0]

        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.detach()

        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.detach()

        x = x + attn_out
        activations["post_attn_residual"] = x.detach()

        if not skip_ln2 and hasattr(block, "ln_2"):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.detach()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.detach()
            mlp_input = x

        mlp_out = block.mlp(mlp_input)
        x = x + mlp_out
        activations["post_mlp_residual"] = x.detach()

    return activations


def compute_probing_metrics(
    activations: np.ndarray, positions: np.ndarray, n_bins: int = 32
) -> Dict[str, float]:
    """Compute probing metrics using fast RidgeClassifier."""
    n_samples = len(positions)
    n_train = int(0.8 * n_samples)

    # Bin positions
    bin_size = 512 / n_bins
    binned_positions = np.clip((positions / bin_size).astype(int), 0, n_bins - 1)

    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    norms = np.linalg.norm(activations, axis=1)
    directions = activations / (norms[:, np.newaxis] + 1e-8)

    # Prepare data with standard scaling
    scaler_full = StandardScaler()
    scaler_dir = StandardScaler()
    scaler_norm = StandardScaler()

    X_train_full = scaler_full.fit_transform(activations[train_idx])
    X_test_full = scaler_full.transform(activations[test_idx])
    X_train_dir = scaler_dir.fit_transform(directions[train_idx])
    X_test_dir = scaler_dir.transform(directions[test_idx])
    X_train_norm = scaler_norm.fit_transform(norms[train_idx].reshape(-1, 1))
    X_test_norm = scaler_norm.transform(norms[test_idx].reshape(-1, 1))

    y_train = binned_positions[train_idx]
    y_test = binned_positions[test_idx]

    # Use RidgeClassifier - much faster than LogisticRegression
    clf_full = RidgeClassifier(alpha=1.0)
    clf_full.fit(X_train_full, y_train)
    full_acc = clf_full.score(X_test_full, y_test)

    clf_dir = RidgeClassifier(alpha=1.0)
    clf_dir.fit(X_train_dir, y_train)
    dir_acc = clf_dir.score(X_test_dir, y_test)

    clf_norm = RidgeClassifier(alpha=1.0)
    clf_norm.fit(X_train_norm, y_train)
    norm_acc = clf_norm.score(X_test_norm, y_test)

    return {
        "full_accuracy": float(full_acc),
        "direction_accuracy": float(dir_acc),
        "norm_accuracy": float(norm_acc),
        "random_baseline": 1.0 / n_bins,
    }


def analyze_model(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    skip_ln2: bool = False,
    n_samples: int = 50,
) -> Dict:
    """Run analysis on a model."""
    ctx = config.block_size
    vocab_size = config.vocab_size

    print(f"\n  Analyzing: {model_name} ({n_samples} samples)")

    all_activations = {layer: [] for layer in LAYERS}
    all_positions = []

    for i in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, ctx), device=DEVICE)
        acts = get_activations(model, tokens, skip_ln2=skip_ln2)
        for layer in LAYERS:
            all_activations[layer].append(acts[layer][0].cpu().numpy())
        all_positions.append(np.arange(ctx))

    for layer in LAYERS:
        all_activations[layer] = np.vstack(all_activations[layer])
    positions = np.concatenate(all_positions)

    results = {"model_name": model_name}

    for layer in LAYERS:
        metrics = compute_probing_metrics(all_activations[layer], positions)
        results[layer] = metrics

    return results


def main():
    print("=" * 70)
    print("FAST OWT PROBING: RANDOM vs TRAINED COMPARISON")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'#' * 60}")
        print(f"# {exp.name}")
        print(f"{'#' * 60}")

        all_results[exp.name] = {}

        # Random init
        print("\n  [Random Init]")
        random_model, config = create_random_model(exp)
        all_results[exp.name]["random"] = analyze_model(
            random_model,
            config,
            f"{exp.name} (random)",
            skip_ln2=exp.skip_ln2,
            n_samples=50,
        )
        del random_model
        torch.cuda.empty_cache()

        # Trained
        print("\n  [Trained]")
        trained_model, config = load_trained_model(exp)
        all_results[exp.name]["trained"] = analyze_model(
            trained_model,
            config,
            f"{exp.name} (trained)",
            skip_ln2=exp.skip_ln2,
            n_samples=50,
        )
        del trained_model
        torch.cuda.empty_cache()

    # Save results
    output_path = RESULTS_DIR / "owt_random_vs_trained_accuracy.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY: Position Classification Accuracy (32 bins, random baseline = 3.1%)")
    print("=" * 90)

    key_layers = ["post_attn", "post_ln2", "post_mlp_residual"]

    for init_type in ["random", "trained"]:
        print(f"\n{'─' * 90}")
        print(f"  {init_type.upper()}")
        print(f"{'─' * 90}")

        # Header
        print(f"{'Model':<20}", end="")
        for layer in key_layers:
            print(f" {LAYER_NAMES[layer]:>20}", end="")
        print()

        print(f"{'':<20}", end="")
        for _ in key_layers:
            print(f" {'Full/Dir/Norm':>20}", end="")
        print()
        print("-" * 90)

        for name in [e.name for e in EXPERIMENTS]:
            data = all_results[name][init_type]
            print(f"{name:<20}", end="")
            for layer in key_layers:
                m = data[layer]
                cell = f"{m['full_accuracy']:.2f}/{m['direction_accuracy']:.2f}/{m['norm_accuracy']:.2f}"
                print(f" {cell:>20}", end="")
            print()

    # Create comparison figure
    create_comparison_figure(all_results)

    return all_results


def create_comparison_figure(all_results: Dict):
    """Create figure comparing random vs trained."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

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
    titles = ["Full Activation", "Direction Only", "Norm Only"]

    for row, init_type in enumerate(["random", "trained"]):
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row, col]

            for exp_name, data in all_results.items():
                y_values = [data[init_type][layer][metric] for layer in LAYERS]
                ax.plot(
                    x_positions,
                    y_values,
                    marker=markers[exp_name],
                    color=colors[exp_name],
                    linewidth=2,
                    markersize=8,
                    label=exp_name,
                )

            ax.axhline(
                y=1 / 32, color="gray", linestyle="--", alpha=0.7, label="Random (3.1%)"
            )
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_title(
                f"{title} ({init_type.capitalize()})", fontsize=12, fontweight="bold"
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(-0.01, 1.05)
            ax.grid(True, alpha=0.3)

            if row == 0 and col == 2:
                ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(
        "Position Classification: Random Init vs Trained",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    fig.savefig(
        PLOTS_DIR / "owt_random_vs_trained_comparison.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        PLOTS_DIR / "owt_random_vs_trained_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"\nSaved figure: {PLOTS_DIR / 'owt_random_vs_trained_comparison.pdf'}")


if __name__ == "__main__":
    main()
