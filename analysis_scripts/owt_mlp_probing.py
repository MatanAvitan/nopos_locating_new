"""
OWT MLP Probing - Uses MLP probe for position classification.

MLP probes have better expressivity than linear probes but cannot encode
position by themselves, making them reliable probes for detecting position
information in activations.

Compares random vs trained models on position classification accuracy.
"""

import torch
import torch.nn as nn
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
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "owt_mlp_probing"
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


class MLPProbe(nn.Module):
    """2-layer MLP probe for position classification.

    MLP probes have more expressivity than linear probes but cannot encode
    position information on their own (no positional inputs), making them
    reliable for detecting position information in activations.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_classes: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


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


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    n_classes: int = 32,
    hidden_dim: int = 256,
    n_epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> Tuple[float, MLPProbe]:
    """Train MLP probe and return test accuracy."""

    probe = MLPProbe(input_dim, hidden_dim, n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    n_samples = len(X_train_t)
    best_acc = 0.0

    for epoch in range(n_epochs):
        probe.train()

        # Shuffle
        perm = torch.randperm(n_samples)
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            batch_X = X_train_t[i : i + batch_size]
            batch_y = y_train_t[i : i + batch_size]

            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            logits = probe(X_test_t)
            preds = logits.argmax(dim=1)
            acc = (preds == y_test_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc

    return best_acc, probe


def compute_mlp_probing_metrics(
    activations: np.ndarray, positions: np.ndarray, n_bins: int = 32
) -> Dict[str, float]:
    """Compute MLP probing metrics."""
    n_samples = len(positions)
    n_train = int(0.8 * n_samples)
    input_dim = activations.shape[1]

    # Bin positions
    bin_size = 512 / n_bins
    binned_positions = np.clip((positions / bin_size).astype(int), 0, n_bins - 1)

    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Compute norm and direction
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

    # Train MLP probes
    full_acc, _ = train_mlp_probe(
        X_train_full,
        y_train,
        X_test_full,
        y_test,
        input_dim=input_dim,
        n_classes=n_bins,
    )

    dir_acc, _ = train_mlp_probe(
        X_train_dir, y_train, X_test_dir, y_test, input_dim=input_dim, n_classes=n_bins
    )

    # For norm, use smaller hidden dim since input is 1D
    norm_acc, _ = train_mlp_probe(
        X_train_norm,
        y_train,
        X_test_norm,
        y_test,
        input_dim=1,
        n_classes=n_bins,
        hidden_dim=64,
    )

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
    n_samples: int = 100,
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
        print(f"    Training MLP probe for {layer}...")
        metrics = compute_mlp_probing_metrics(all_activations[layer], positions)
        results[layer] = metrics
        print(
            f"      Full: {metrics['full_accuracy']:.3f}, Dir: {metrics['direction_accuracy']:.3f}, Norm: {metrics['norm_accuracy']:.3f}"
        )

    return results


def create_comparison_figure(all_results: Dict):
    """Create figure comparing random vs trained with MLP probes."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    colors = {
        "full_accuracy": "#1f77b4",
        "direction_accuracy": "#ff7f0e",
        "norm_accuracy": "#2ca02c",
    }

    x_positions = np.arange(len(LAYERS))
    x_labels = [LAYER_NAMES[l] for l in LAYERS]

    exp_names = [e.name for e in EXPERIMENTS]

    for row, init_type in enumerate(["random", "trained"]):
        for col, exp_name in enumerate(exp_names):
            ax = axes[row, col]
            data = all_results[exp_name][init_type]

            for metric, color in colors.items():
                y_values = [data[layer][metric] for layer in LAYERS]
                label = metric.replace("_accuracy", "").capitalize()
                ax.plot(
                    x_positions,
                    y_values,
                    marker="o",
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=label,
                )

            ax.axhline(
                y=1 / 32, color="gray", linestyle="--", alpha=0.7, label="Random (3.1%)"
            )
            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)

            short_name = next(e.short_name for e in EXPERIMENTS if e.name == exp_name)
            ax.set_title(
                f"{short_name} ({init_type.capitalize()})",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)

            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7)

    plt.suptitle(
        "MLP Probe Position Classification: Random Init vs Trained",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    fig.savefig(
        PLOTS_DIR / "owt_mlp_probe_comparison.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        PLOTS_DIR / "owt_mlp_probe_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"\nSaved figure: {PLOTS_DIR / 'owt_mlp_probe_comparison.pdf'}")


def create_separate_figures(all_results: Dict):
    """Create separate figures for random and trained (for paper)."""

    colors = {
        "full_accuracy": "#1f77b4",
        "direction_accuracy": "#ff7f0e",
        "norm_accuracy": "#2ca02c",
    }

    x_positions = np.arange(len(LAYERS))
    x_labels = [LAYER_NAMES[l] for l in LAYERS]
    exp_names = [e.name for e in EXPERIMENTS]

    for init_type in ["random", "trained"]:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

        for col, exp_name in enumerate(exp_names):
            ax = axes[col]
            data = all_results[exp_name][init_type]

            for metric, color in colors.items():
                y_values = [data[layer][metric] for layer in LAYERS]
                label = metric.replace("_accuracy", "").capitalize()
                ax.plot(
                    x_positions,
                    y_values,
                    marker="o",
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=label,
                )

            ax.axhline(y=1 / 32, color="gray", linestyle="--", alpha=0.7)
            ax.set_xlabel("Layer", fontsize=10)
            if col == 0:
                ax.set_ylabel("Accuracy", fontsize=10)

            short_name = next(e.short_name for e in EXPERIMENTS if e.name == exp_name)
            ax.set_title(short_name, fontsize=11, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)

            if col == 3:
                ax.legend(loc="upper right", fontsize=7)

        title = "Random Initialization" if init_type == "random" else "Trained Models"
        plt.suptitle(
            f"MLP Probe Position Classification ({title})",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        fig.savefig(
            PLOTS_DIR / f"owt_mlp_probe_{init_type}.pdf", dpi=300, bbox_inches="tight"
        )
        fig.savefig(
            PLOTS_DIR / f"owt_mlp_probe_{init_type}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved figure: {PLOTS_DIR / f'owt_mlp_probe_{init_type}.pdf'}")


def main():
    print("=" * 70)
    print("MLP PROBING: RANDOM vs TRAINED COMPARISON")
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
            n_samples=100,
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
            n_samples=100,
        )
        del trained_model
        torch.cuda.empty_cache()

    # Save results
    output_path = RESULTS_DIR / "owt_mlp_probe_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 90)
    print(
        "SUMMARY: MLP Probe Position Classification Accuracy (32 bins, random baseline = 3.1%)"
    )
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

    # Create figures
    create_comparison_figure(all_results)
    create_separate_figures(all_results)

    return all_results


if __name__ == "__main__":
    main()
