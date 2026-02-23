"""
Linear Probing on Post-LN Activations (Before Residual) with t-SNE Visualization

This script:
1. Trains linear probes on post-LN activations (before adding residual) for:
   - LayerNorm models (LN1 and LN2)
   - RMSNorm models
2. Tests on both randomly initialized and trained models
3. Logs results and plots to wandb
4. Creates t-SNE visualizations with position bucket coloring

Key insight: We extract activations AFTER the LayerNorm/RMSNorm but BEFORE adding
the residual connection. This isolates the effect of normalization on position encoding.
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
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import argparse

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig, LayerNorm, RMSNorm

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Plots will only be saved locally.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "linear_probe_post_ln"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
PLOTS_DIR = PROJECT_ROOT / "results" / "linear_probe_post_ln" / "plots"


@dataclass
class ModelConfig:
    """Configuration for a model experiment."""

    name: str
    short_name: str
    checkpoint_path: Optional[str]  # None for random init only
    norm_type: str  # "layernorm" or "rmsnorm"
    use_positional_embedding: bool = False
    skip_ln2: bool = False
    use_batchnorm_ln2: bool = False


# Experiments to run
EXPERIMENTS = [
    ModelConfig(
        name="NoPE + LayerNorm",
        short_name="NoPE_LN",
        checkpoint_path="out-nope-owt-ln/ckpt.pt",
        norm_type="layernorm",
    ),
    ModelConfig(
        name="NoPE + RMSNorm",
        short_name="NoPE_RMS",
        checkpoint_path="out-nope-1layer-rms/ckpt.pt",  # Correct path for RMSNorm
        norm_type="rmsnorm",
    ),
]

# Layers to extract - focus on post-LN before residual
EXTRACTION_POINTS = [
    "embed",
    "post_ln1",  # After LN1, before attention
    "post_attn",  # After attention, before residual add
    "pre_ln2",  # Before LN2 (after first residual)
    "post_ln2",  # After LN2, before MLP
    "post_mlp",  # After MLP, before residual add
]


def create_random_model(config: ModelConfig, n_layer: int = 1) -> Tuple[GPT, GPTConfig]:
    """Create a randomly initialized model."""
    gpt_config = GPTConfig(
        n_layer=n_layer,
        n_head=12,
        n_embd=768,
        block_size=256,
        vocab_size=50304,
        dropout=0.0,
        use_positional_embedding=config.use_positional_embedding,
        norm_type=config.norm_type,
        bias=False,
        skip_ln2=config.skip_ln2,
        use_batchnorm_ln2=config.use_batchnorm_ln2,
    )
    model = GPT(gpt_config)
    model.eval()
    model.to(DEVICE)
    return model, gpt_config


def load_trained_model(config: ModelConfig) -> Tuple[GPT, GPTConfig]:
    """Load a trained model from checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / config.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    gpt_config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 256),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=config.use_positional_embedding,
        norm_type=config.norm_type,
        bias=model_args.get("bias", False),
        skip_ln2=config.skip_ln2,
        use_batchnorm_ln2=config.use_batchnorm_ln2,
    )

    model = GPT(gpt_config)
    state_dict = checkpoint["model"]
    # Remove _orig_mod. prefix if present (from torch.compile)
    unwrapped = {
        (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(unwrapped)
    model.eval()
    model.to(DEVICE)
    return model, gpt_config


def get_post_ln_activations(
    model: GPT,
    tokens: torch.Tensor,
    layer_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Extract activations at key points, focusing on POST-LN BEFORE RESIDUAL.

    This is critical: we want to see how LN transforms the representation
    BEFORE it gets mixed with the residual stream.

    Architecture flow:
        x -> LN1 -> Attention -> (+x) -> LN2 -> MLP -> (+x)
             ^                          ^
             post_ln1                   post_ln2
             (before residual)          (before residual)
    """
    activations = {}

    with torch.no_grad():
        # Embedding
        tok_emb = model.transformer.wte(tokens)
        if hasattr(model.transformer, "wpe") and model.config.use_positional_embedding:
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        activations["embed"] = x.clone()

        # Get the specified block
        block = model.transformer.h[layer_idx]

        # Post-LN1 (BEFORE attention, BEFORE residual)
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.clone()

        # Post-Attention output (BEFORE adding residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # After first residual connection (input to LN2)
        x = x + attn_out
        activations["pre_ln2"] = x.clone()

        # Post-LN2 (BEFORE MLP, BEFORE second residual)
        if hasattr(block, "ln_2") and not block.skip_ln2:
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.clone()
            mlp_input = x

        # Post-MLP output (BEFORE adding residual)
        mlp_out = block.mlp(mlp_input)
        activations["post_mlp"] = mlp_out.clone()

    return activations


def collect_activations(
    model: GPT,
    config: GPTConfig,
    n_samples: int = 50,
    seq_len: int = 256,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Collect activations from multiple random sequences."""
    vocab_size = config.vocab_size

    all_activations = {layer: [] for layer in EXTRACTION_POINTS}
    all_positions = []

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)
        acts = get_post_ln_activations(model, tokens)

        for layer in EXTRACTION_POINTS:
            if layer in acts:
                all_activations[layer].append(acts[layer][0].cpu().numpy())

        all_positions.append(np.arange(seq_len))

    # Stack all samples
    for layer in EXTRACTION_POINTS:
        if all_activations[layer]:
            all_activations[layer] = np.vstack(all_activations[layer])

    positions = np.concatenate(all_positions)

    return all_activations, positions


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = 32,
) -> Tuple[float, LogisticRegression]:
    """Train a linear probe (logistic regression) for position classification."""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    accuracy = clf.score(X_test_scaled, y_test)

    return accuracy, clf


def compute_probing_metrics(
    activations: np.ndarray,
    positions: np.ndarray,
    n_bins: int = 32,
) -> Dict[str, float]:
    """Compute linear probing metrics for position classification."""
    n_samples = len(positions)
    n_train = int(0.8 * n_samples)

    # Bin positions
    max_pos = positions.max() + 1
    bin_size = max_pos / n_bins
    binned_positions = np.clip((positions / bin_size).astype(int), 0, n_bins - 1)

    # Shuffle and split
    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Compute norm and direction
    norms = np.linalg.norm(activations, axis=1)
    directions = activations / (norms[:, np.newaxis] + 1e-8)

    # Prepare data
    y_train = binned_positions[train_idx]
    y_test = binned_positions[test_idx]

    # Train probes on full activation, direction only, norm only
    full_acc, _ = train_linear_probe(
        activations[train_idx], y_train, activations[test_idx], y_test, n_classes=n_bins
    )

    dir_acc, _ = train_linear_probe(
        directions[train_idx], y_train, directions[test_idx], y_test, n_classes=n_bins
    )

    norm_acc, _ = train_linear_probe(
        norms[train_idx].reshape(-1, 1),
        y_train,
        norms[test_idx].reshape(-1, 1),
        y_test,
        n_classes=n_bins,
    )

    return {
        "full_accuracy": float(full_acc),
        "direction_accuracy": float(dir_acc),
        "norm_accuracy": float(norm_acc),
        "random_baseline": 1.0 / n_bins,
    }


def create_tsne_visualization(
    activations: np.ndarray,
    positions: np.ndarray,
    title: str,
    n_buckets: int = 8,
    n_samples_tsne: int = 1000,  # Reduced for speed
    perplexity: int = 30,
) -> plt.Figure:
    """
    Create t-SNE visualization with position buckets colored.

    Args:
        activations: (N, D) activation vectors
        positions: (N,) position indices
        title: Plot title
        n_buckets: Number of position buckets for coloring
        n_samples_tsne: Max samples for t-SNE (for speed)
        perplexity: t-SNE perplexity parameter
    """
    # Subsample if needed
    if len(positions) > n_samples_tsne:
        idx = np.random.choice(len(positions), n_samples_tsne, replace=False)
        activations = activations[idx]
        positions = positions[idx]

    # Bin positions for coloring
    max_pos = positions.max() + 1
    bucket_size = max_pos / n_buckets
    position_buckets = np.clip((positions / bucket_size).astype(int), 0, n_buckets - 1)

    # Run t-SNE
    print(f"    Running t-SNE ({len(positions)} samples)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    embeddings = tsne.fit_transform(activations)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map
    cmap = plt.cm.get_cmap("viridis", n_buckets)

    # Plot each bucket with its color
    for bucket in range(n_buckets):
        mask = position_buckets == bucket
        if mask.sum() > 0:
            start_pos = int(bucket * bucket_size)
            end_pos = int((bucket + 1) * bucket_size) - 1
            label = f"pos {start_pos}-{end_pos}"
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[cmap(bucket)],
                label=label,
                alpha=0.6,
                s=10,
            )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_comparison_tsne_figure(
    random_activations: Dict[str, np.ndarray],
    trained_activations: Dict[str, np.ndarray],
    positions: np.ndarray,
    model_name: str,
    layers_to_plot: List[str] = ["post_ln1", "post_ln2"],
    n_buckets: int = 8,
) -> plt.Figure:
    """
    Create a comparison figure showing t-SNE for random vs trained model
    for specified layers (before and after LN).
    """
    n_layers = len(layers_to_plot)
    fig, axes = plt.subplots(2, n_layers, figsize=(6 * n_layers, 10))

    if n_layers == 1:
        axes = axes.reshape(2, 1)

    cmap = plt.cm.get_cmap("viridis", n_buckets)

    for col, layer in enumerate(layers_to_plot):
        for row, (init_type, acts) in enumerate(
            [("Random", random_activations), ("Trained", trained_activations)]
        ):
            ax = axes[row, col]

            if layer not in acts or acts[layer] is None or len(acts[layer]) == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_title(f"{init_type} - {layer}")
                continue

            activation_data = acts[layer]
            pos_data = positions[: len(activation_data)]

            # Subsample
            n_samples = min(1000, len(pos_data))  # Reduced for speed
            idx = np.random.choice(len(pos_data), n_samples, replace=False)
            activation_subset = activation_data[idx]
            pos_subset = pos_data[idx]

            # Bin positions
            max_pos = pos_subset.max() + 1
            bucket_size = max_pos / n_buckets
            position_buckets = np.clip(
                (pos_subset / bucket_size).astype(int), 0, n_buckets - 1
            )

            # t-SNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
            embeddings = tsne.fit_transform(activation_subset)

            # Plot
            for bucket in range(n_buckets):
                mask = position_buckets == bucket
                if mask.sum() > 0:
                    ax.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        c=[cmap(bucket)],
                        alpha=0.6,
                        s=10,
                    )

            layer_display = layer.replace("_", " ").title()
            ax.set_title(
                f"{init_type} - {layer_display}", fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("t-SNE 1", fontsize=9)
            ax.set_ylabel("t-SNE 2", fontsize=9)
            ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_buckets))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Position Bucket", fontsize=10)

    fig.suptitle(
        f"t-SNE Visualization: {model_name}\nPost-LN Activations (Before Residual)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    return fig


def analyze_model_pair(
    config: ModelConfig,
    n_samples: int = 50,
    use_wandb: bool = True,
    skip_tsne: bool = False,
) -> Dict:
    """Run full analysis on random and trained model pair."""
    results = {
        "config": config.name,
        "norm_type": config.norm_type,
        "random": {},
        "trained": {},
    }

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {config.name}")
    print(f"{'=' * 60}")

    # === Random Model ===
    print("\n[Random Initialization]")
    random_model, gpt_config = create_random_model(config)
    random_activations, positions = collect_activations(
        random_model, gpt_config, n_samples=n_samples
    )

    print("  Computing linear probe metrics...")
    for layer in EXTRACTION_POINTS:
        if layer in random_activations and len(random_activations[layer]) > 0:
            metrics = compute_probing_metrics(random_activations[layer], positions)
            results["random"][layer] = metrics
            print(
                f"    {layer}: Full={metrics['full_accuracy']:.3f}, Dir={metrics['direction_accuracy']:.3f}, Norm={metrics['norm_accuracy']:.3f}"
            )

    del random_model
    torch.cuda.empty_cache()

    # === Trained Model ===
    trained_activations = None
    if config.checkpoint_path:
        try:
            print("\n[Trained Model]")
            trained_model, gpt_config = load_trained_model(config)
            trained_activations, positions = collect_activations(
                trained_model, gpt_config, n_samples=n_samples
            )

            print("  Computing linear probe metrics...")
            for layer in EXTRACTION_POINTS:
                if layer in trained_activations and len(trained_activations[layer]) > 0:
                    metrics = compute_probing_metrics(
                        trained_activations[layer], positions
                    )
                    results["trained"][layer] = metrics
                    print(
                        f"    {layer}: Full={metrics['full_accuracy']:.3f}, Dir={metrics['direction_accuracy']:.3f}, Norm={metrics['norm_accuracy']:.3f}"
                    )

            del trained_model
            torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Skipping trained model analysis.")

    # === Create t-SNE Visualizations ===
    if not skip_tsne:
        print("\n[Creating t-SNE visualizations...]")

        # Layers to visualize: before and after each LN (showing the effect)
        tsne_layers = ["embed", "post_ln1", "pre_ln2", "post_ln2"]

        # Create comparison figure
        if trained_activations is not None:
            # Random init
            random_activations_for_tsne, positions = collect_activations(
                create_random_model(config)[0], gpt_config, n_samples=30
            )

            fig = create_comparison_tsne_figure(
                random_activations_for_tsne,
                trained_activations,
                positions,
                config.name,
                layers_to_plot=["post_ln1", "post_ln2"],
            )

            # Save locally
            save_path = PLOTS_DIR / f"tsne_{config.short_name}_comparison.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({f"tsne/{config.short_name}_comparison": wandb.Image(fig)})

            plt.close(fig)

        # Individual t-SNE plots for each layer
        for layer in tsne_layers:
            if layer in random_activations and len(random_activations[layer]) > 0:
                # Random model t-SNE
                fig = create_tsne_visualization(
                    random_activations[layer],
                    positions,
                    f"{config.name} (Random) - {layer.replace('_', ' ').title()}",
                )
                save_path = PLOTS_DIR / f"tsne_{config.short_name}_random_{layer}.png"
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

                if use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {f"tsne/{config.short_name}_random_{layer}": wandb.Image(fig)}
                    )
                plt.close(fig)

            if trained_activations is not None and layer in trained_activations:
                if len(trained_activations[layer]) > 0:
                    # Trained model t-SNE
                    fig = create_tsne_visualization(
                        trained_activations[layer],
                        positions,
                        f"{config.name} (Trained) - {layer.replace('_', ' ').title()}",
                    )
                    save_path = (
                        PLOTS_DIR / f"tsne_{config.short_name}_trained_{layer}.png"
                    )
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")

                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log(
                            {
                                f"tsne/{config.short_name}_trained_{layer}": wandb.Image(
                                    fig
                                )
                            }
                        )
                    plt.close(fig)
    else:
        print("\n[Skipping t-SNE visualizations (--no-tsne)]")

    return results


def create_summary_bar_plot(all_results: Dict, use_wandb: bool = True) -> plt.Figure:
    """Create summary bar plot comparing all models and layers."""
    n_models = len(all_results)
    n_init_types = 2  # random, trained

    layers_to_plot = ["post_ln1", "post_ln2", "post_attn", "post_mlp"]
    layer_labels = ["Post-LN1", "Post-LN2", "Post-Attn", "Post-MLP"]

    fig, axes = plt.subplots(
        1, len(layers_to_plot), figsize=(4 * len(layers_to_plot), 5)
    )

    bar_width = 0.35
    colors = {"random": "#3498db", "trained": "#e74c3c"}

    model_names = list(all_results.keys())
    x = np.arange(len(model_names))

    for idx, (layer, layer_label) in enumerate(zip(layers_to_plot, layer_labels)):
        ax = axes[idx]

        random_accs = []
        trained_accs = []

        for model_name in model_names:
            result = all_results[model_name]
            random_acc = result.get("random", {}).get(layer, {}).get("full_accuracy", 0)
            trained_acc = (
                result.get("trained", {}).get(layer, {}).get("full_accuracy", 0)
            )
            random_accs.append(random_acc)
            trained_accs.append(trained_acc)

        ax.bar(
            x - bar_width / 2,
            random_accs,
            bar_width,
            label="Random",
            color=colors["random"],
            alpha=0.8,
        )
        ax.bar(
            x + bar_width / 2,
            trained_accs,
            bar_width,
            label="Trained",
            color=colors["trained"],
            alpha=0.8,
        )

        ax.axhline(
            y=1 / 32, color="gray", linestyle="--", alpha=0.7, label="Chance (3.1%)"
        )
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(layer_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.split()[0] for m in model_names], rotation=45, ha="right", fontsize=9
        )
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")

        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Linear Probe Accuracy on Post-LN Activations (Before Residual)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save
    save_path = PLOTS_DIR / "summary_linear_probe_accuracy.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary plot: {save_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary/linear_probe_accuracy": wandb.Image(fig)})

    return fig


def create_probing_line_plot(all_results: Dict, use_wandb: bool = True) -> plt.Figure:
    """Create line plot showing accuracy across layers."""
    layers_order = ["embed", "post_ln1", "post_attn", "pre_ln2", "post_ln2", "post_mlp"]
    layer_labels = ["Embed", "Post-LN1", "Post-Attn", "Pre-LN2", "Post-LN2", "Post-MLP"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "NoPE + LayerNorm": ("#1f77b4", "o"),
        "NoPE + RMSNorm": ("#ff7f0e", "s"),
    }

    for ax_idx, init_type in enumerate(["random", "trained"]):
        ax = axes[ax_idx]

        for model_name in all_results.keys():
            result = all_results[model_name].get(init_type, {})
            if not result:
                continue

            color, marker = colors.get(model_name, ("#2ca02c", "d"))

            full_accs = []
            dir_accs = []
            for layer in layers_order:
                layer_data = result.get(layer, {})
                full_accs.append(layer_data.get("full_accuracy", 0))
                dir_accs.append(layer_data.get("direction_accuracy", 0))

            ax.plot(
                range(len(layers_order)),
                full_accs,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=8,
                label=f"{model_name} (Full)",
            )
            ax.plot(
                range(len(layers_order)),
                dir_accs,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=8,
                linestyle="--",
                alpha=0.6,
                label=f"{model_name} (Dir)",
            )

        ax.axhline(y=1 / 32, color="gray", linestyle=":", alpha=0.7, label="Chance")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(
            f"{init_type.capitalize()} Initialization", fontsize=13, fontweight="bold"
        )
        ax.set_xticks(range(len(layers_order)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Linear Probe Position Classification: LayerNorm vs RMSNorm",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = PLOTS_DIR / "probing_line_plot.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved line plot: {save_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary/probing_line_plot": wandb.Image(fig)})

    return fig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Linear probing on post-LN activations with t-SNE"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50, help="Number of samples for probing"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--no-tsne", action="store_true", help="Skip t-SNE visualizations (faster)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer samples, no t-SNE"
    )
    args = parser.parse_args()

    if args.quick:
        args.n_samples = 20
        args.no_tsne = True

    print("=" * 70)
    print("LINEAR PROBING ON POST-LN ACTIVATIONS (BEFORE RESIDUAL)")
    print("With t-SNE Visualization and WandB Logging")
    print("=" * 70)
    print(
        f"  Samples: {args.n_samples}, WandB: {not args.no_wandb}, t-SNE: {not args.no_tsne}"
    )

    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="nope-linear-probing",
            name="post_ln_probing_with_tsne",
            config={
                "experiments": [e.name for e in EXPERIMENTS],
                "n_samples": args.n_samples,
                "n_bins": 32,
            },
        )
        print("\nWandB initialized. Logging to project: nope-linear-probing")

    # Run experiments
    all_results = {}

    for exp in EXPERIMENTS:
        results = analyze_model_pair(
            exp,
            n_samples=args.n_samples,
            use_wandb=use_wandb,
            skip_tsne=args.no_tsne,
        )
        all_results[exp.name] = results

        # Log metrics to wandb
        if use_wandb:
            for init_type in ["random", "trained"]:
                for layer, metrics in results.get(init_type, {}).items():
                    wandb.log(
                        {
                            f"metrics/{exp.short_name}/{init_type}/{layer}/full_accuracy": metrics.get(
                                "full_accuracy", 0
                            ),
                            f"metrics/{exp.short_name}/{init_type}/{layer}/direction_accuracy": metrics.get(
                                "direction_accuracy", 0
                            ),
                            f"metrics/{exp.short_name}/{init_type}/{layer}/norm_accuracy": metrics.get(
                                "norm_accuracy", 0
                            ),
                        }
                    )

    # Save results
    results_path = RESULTS_DIR / "linear_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Create summary plots
    print("\n[Creating summary plots...]")
    create_summary_bar_plot(all_results, use_wandb)
    create_probing_line_plot(all_results, use_wandb)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY: Linear Probe Position Classification (32 bins, chance = 3.1%)")
    print("=" * 90)

    key_layers = ["post_ln1", "post_ln2", "post_mlp"]

    for init_type in ["random", "trained"]:
        print(f"\n--- {init_type.upper()} ---")
        print(f"{'Model':<25} {'Post-LN1':>20} {'Post-LN2':>20} {'Post-MLP':>20}")
        print(
            f"{'':>25} {'Full/Dir/Norm':>20} {'Full/Dir/Norm':>20} {'Full/Dir/Norm':>20}"
        )
        print("-" * 90)

        for name, results in all_results.items():
            data = results.get(init_type, {})
            if not data:
                continue

            print(f"{name:<25}", end="")
            for layer in key_layers:
                m = data.get(layer, {})
                cell = f"{m.get('full_accuracy', 0):.2f}/{m.get('direction_accuracy', 0):.2f}/{m.get('norm_accuracy', 0):.2f}"
                print(f" {cell:>20}", end="")
            print()

    # Finish wandb
    if use_wandb:
        wandb.finish()
        print("\nWandB run finished. Check dashboard for plots.")

    print(f"\nAll plots saved to: {PLOTS_DIR}")

    return all_results


if __name__ == "__main__":
    main()
