"""
Comprehensive Probe Analysis Across All Activation Points

This script trains linear and MLP probes at every hidden representation of the model
to trace where positional information emerges, persists, and potentially gets destroyed.

Activation Points Probed:
- raw_embed: After token embedding lookup
- post_ln1: After first LayerNorm (pre-attention)
- post_attn: After attention (pre-residual)
- post_attn_residual: After attention + residual
- post_ln2: After second LayerNorm (pre-MLP)
- post_mlp: After MLP (pre-residual)
- post_mlp_residual: After MLP + residual
- post_final_ln: After final LayerNorm

Probe Types:
- Linear (Ridge): Tests linear decodability
- MLP (2-layer): Tests nonlinear decodability
- Mean-Subtracted Linear: Tests if info survives without population mean

Usage:
    python comprehensive_probe_analysis.py --n_samples 30000 --seq_len 64
"""

import os

# Only set GPU if not already specified by environment
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/comprehensive_probe_analysis")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ProbeConfig:
    """Configuration for probing experiments."""

    n_samples: int = 30000
    seq_len: int = 64
    d_model: int = 1024
    n_heads: int = 1
    d_head: int = 1024
    d_mlp: int = 4096
    vocab_size: int = 50257
    batch_size: int = 512
    train_ratio: float = 0.8
    seed: int = 42


# Activation points to probe
ACTIVATION_POINTS = [
    "raw_embed",  # hook_embed
    "post_ln1",  # blocks.0.ln1.hook_normalized
    "post_attn",  # blocks.0.attn.hook_result
    "post_attn_residual",  # blocks.0.hook_resid_mid
    "post_ln2",  # blocks.0.ln2.hook_normalized
    "post_mlp",  # blocks.0.mlp.hook_post
    "post_mlp_residual",  # blocks.0.hook_resid_post
    "post_final_ln",  # ln_final.hook_normalized
]

# Mapping to transformer_lens hook names
# Note: blocks.0.attn.hook_result is not cached by default, use blocks.0.hook_attn_out instead
HOOK_NAMES = {
    "raw_embed": "hook_embed",
    "post_ln1": "blocks.0.ln1.hook_normalized",
    "post_attn": "blocks.0.hook_attn_out",  # Changed from blocks.0.attn.hook_result
    "post_attn_residual": "blocks.0.hook_resid_mid",
    "post_ln2": "blocks.0.ln2.hook_normalized",
    "post_mlp": "blocks.0.mlp.hook_post",
    "post_mlp_residual": "blocks.0.hook_resid_post",
    "post_final_ln": "ln_final.hook_normalized",
}


def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(cfg: ProbeConfig, norm_type: str = "LN"):
    """Create a HookedTransformer model without positional embeddings."""
    model_cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        d_vocab=cfg.vocab_size,
        n_ctx=cfg.seq_len,
        act_fn="gelu",
        normalization_type=norm_type,
        device=device,
    )
    model = HookedTransformer(model_cfg)

    # Zero out positional embeddings for NoPE
    model.pos_embed.W_pos.data.zero_()
    model.pos_embed.W_pos.requires_grad = False

    return model


def generate_random_tokens(cfg: ProbeConfig):
    """Generate random token sequences for analysis."""
    torch.manual_seed(cfg.seed)
    return torch.randint(0, cfg.vocab_size, (cfg.n_samples, cfg.seq_len), device=device)


def extract_all_activations(model, tokens: torch.Tensor, batch_size: int = 256):
    """
    Extract activations at all hook points.

    Returns:
        activations: dict mapping activation_point_name -> [n_samples, seq_len, d_model]
    """
    model.eval()
    n_samples = tokens.shape[0]

    # Initialize storage
    activations = {name: [] for name in ACTIVATION_POINTS}

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting activations"):
            batch_tokens = tokens[i : i + batch_size]

            # Get all hook names we need
            hook_names = list(HOOK_NAMES.values())

            # Run with cache
            _, cache = model.run_with_cache(batch_tokens, names_filter=hook_names)

            # Extract and store activations
            for point_name, hook_name in HOOK_NAMES.items():
                act = cache[hook_name].detach().cpu()  # [batch, seq_len, d_model]
                activations[point_name].append(act)

            # Clear cache to free memory
            del cache
            torch.cuda.empty_cache()

    # Concatenate batches
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0).numpy()

    return activations


def compute_population_means(activations: dict):
    """
    Compute population mean at each position for each activation point.

    Returns:
        pop_means: dict mapping name -> [seq_len, d_model]
    """
    pop_means = {}
    for name, acts in activations.items():
        # acts: [n_samples, seq_len, d_model]
        pop_means[name] = acts.mean(axis=0)  # [seq_len, d_model]
    return pop_means


def prepare_probe_data(
    activations: np.ndarray,
    positions: np.ndarray,
    train_ratio: float = 0.8,
    mean_subtract: bool = False,
    pop_mean: np.ndarray = None,
):
    """
    Prepare data for probe training.

    Args:
        activations: [n_samples, seq_len, d_model]
        positions: [seq_len] position indices
        train_ratio: fraction for training
        mean_subtract: whether to subtract population mean
        pop_mean: [seq_len, d_model] population mean per position

    Returns:
        X_train, X_test, y_train, y_test
    """
    n_samples, seq_len, d_model = activations.shape

    # Flatten to [n_samples * seq_len, d_model]
    X = activations.reshape(-1, d_model)

    # Create position labels
    y = np.tile(positions, n_samples)

    if mean_subtract and pop_mean is not None:
        # Subtract position-specific mean
        pop_mean_expanded = np.tile(
            pop_mean, (n_samples, 1, 1)
        )  # [n_samples, seq_len, d_model]
        X_centered = activations - pop_mean_expanded
        X = X_centered.reshape(-1, d_model)

    # Split by samples (not by flattened indices) to avoid leakage
    n_train_samples = int(n_samples * train_ratio)

    X_train = activations[:n_train_samples].reshape(-1, d_model)
    X_test = activations[n_train_samples:].reshape(-1, d_model)
    y_train = np.tile(positions, n_train_samples)
    y_test = np.tile(positions, n_samples - n_train_samples)

    if mean_subtract and pop_mean is not None:
        # Compute mean on training data only (to avoid leakage)
        train_acts = activations[:n_train_samples]
        train_pop_mean = train_acts.mean(axis=0)

        X_train = (train_acts - train_pop_mean).reshape(-1, d_model)
        X_test = (activations[n_train_samples:] - train_pop_mean).reshape(-1, d_model)

    return X_train, X_test, y_train, y_test


def train_linear_probe(X_train, y_train, alpha: float = 1.0):
    """Train a Ridge regression probe."""
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    return probe


def train_mlp_probe(X_train, y_train, hidden_sizes=(256, 128), max_iter=500):
    """Train an MLP probe."""
    probe = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    probe.fit(X_train, y_train)
    return probe


def evaluate_probe(probe, X_test, y_test):
    """
    Evaluate probe performance.

    Returns:
        dict with r2, mae, pearson_r, accuracy (within 1 position)
    """
    y_pred = probe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    pearson_r, _ = pearsonr(y_test, y_pred)

    # Accuracy: within 1 position
    accuracy = np.mean(np.abs(y_pred - y_test) < 1)

    # Exact accuracy (rounded prediction)
    exact_accuracy = np.mean(np.round(y_pred) == y_test)

    return {
        "r2": float(r2),
        "mae": float(mae),
        "pearson_r": float(pearson_r),
        "accuracy_within_1": float(accuracy),
        "exact_accuracy": float(exact_accuracy),
    }


def run_probe_experiment(activations: dict, cfg: ProbeConfig, norm_type: str):
    """
    Run full probe experiment at all activation points.

    Returns:
        results: dict mapping (activation_point, probe_type) -> metrics
    """
    positions = np.arange(cfg.seq_len)
    pop_means = compute_population_means(activations)

    results = {}

    probe_types = ["linear", "mlp", "linear_mean_subtracted"]

    for act_name in ACTIVATION_POINTS:
        print(f"\n{'=' * 40}")
        print(f"Probing: {act_name}")
        print(f"{'=' * 40}")

        acts = activations[act_name]
        pop_mean = pop_means[act_name]

        for probe_type in probe_types:
            print(f"\n  Training {probe_type} probe...")

            mean_subtract = probe_type == "linear_mean_subtracted"

            # Prepare data
            X_train, X_test, y_train, y_test = prepare_probe_data(
                acts,
                positions,
                cfg.train_ratio,
                mean_subtract=mean_subtract,
                pop_mean=pop_mean,
            )

            # Train probe
            if "linear" in probe_type:
                probe = train_linear_probe(X_train, y_train)
            else:
                probe = train_mlp_probe(X_train, y_train)

            # Evaluate
            metrics = evaluate_probe(probe, X_test, y_test)

            key = f"{act_name}_{probe_type}"
            results[key] = {
                "activation_point": act_name,
                "probe_type": probe_type,
                "norm_type": norm_type,
                **metrics,
            }

            print(f"    R²: {metrics['r2']:.4f}")
            print(f"    Pearson r: {metrics['pearson_r']:.4f}")
            print(f"    MAE: {metrics['mae']:.2f}")
            print(f"    Exact accuracy: {metrics['exact_accuracy']:.2%}")

    return results


def plot_probe_results_heatmap(results: dict, save_path: str, metric: str = "r2"):
    """Plot heatmap of probe results (activation point × probe type)."""
    # Organize data
    probe_types = ["linear", "mlp", "linear_mean_subtracted"]

    matrix = np.zeros((len(ACTIVATION_POINTS), len(probe_types)))

    for i, act_name in enumerate(ACTIVATION_POINTS):
        for j, probe_type in enumerate(probe_types):
            key = f"{act_name}_{probe_type}"
            if key in results:
                matrix[i, j] = results[key][metric]

    # Create heatmap
    fig = px.imshow(
        matrix,
        x=["Linear", "MLP", "Mean-Subtracted"],
        y=ACTIVATION_POINTS,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        labels=dict(x="Probe Type", y="Activation Point", color=metric.upper()),
        text_auto=".3f",
        aspect="auto",
    )

    fig.update_layout(
        title=dict(
            text=f"Position Information by Layer ({metric.upper()})",
            font=dict(size=20, family="Serif"),
        ),
        width=700,
        height=600,
        template="plotly_white",
        margin=dict(l=150, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=700, height=600, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_probe_trajectory(results: dict, save_path: str):
    """Plot probe accuracy trajectory through the network."""
    fig = go.Figure()

    colors = {"linear": "blue", "mlp": "green", "linear_mean_subtracted": "red"}
    names = {
        "linear": "Linear Probe",
        "mlp": "MLP Probe",
        "linear_mean_subtracted": "Mean-Subtracted Linear",
    }

    for probe_type in colors:
        r2_values = []
        for act_name in ACTIVATION_POINTS:
            key = f"{act_name}_{probe_type}"
            if key in results:
                r2_values.append(results[key]["r2"])
            else:
                r2_values.append(0)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(ACTIVATION_POINTS))),
                y=r2_values,
                mode="lines+markers",
                name=names[probe_type],
                line=dict(width=3, color=colors[probe_type]),
                marker=dict(size=10),
            )
        )

    fig.update_layout(
        title=dict(
            text="Position Information Flow Through Network",
            font=dict(size=20, family="Serif"),
        ),
        xaxis=dict(
            title="Activation Point",
            tickmode="array",
            tickvals=list(range(len(ACTIVATION_POINTS))),
            ticktext=[name.replace("_", "\n") for name in ACTIVATION_POINTS],
            tickangle=45,
            title_font=dict(size=16),
            tickfont=dict(size=10),
        ),
        yaxis=dict(title="R² Score", title_font=dict(size=16), tickfont=dict(size=14)),
        width=1000,
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=120),
    )

    fig.write_image(f"{save_path}.png", width=1000, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_ln_vs_rmsln_comparison(results_ln: dict, results_rms: dict, save_path: str):
    """Compare LayerNorm vs RMSNorm probe results."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("LayerNorm", "RMSNorm"), horizontal_spacing=0.1
    )

    probe_types = ["linear", "mlp", "linear_mean_subtracted"]
    colors = {"linear": "blue", "mlp": "green", "linear_mean_subtracted": "red"}

    for col, (results, title) in enumerate(
        [(results_ln, "LN"), (results_rms, "RMS")], 1
    ):
        for probe_type in probe_types:
            r2_values = []
            for act_name in ACTIVATION_POINTS:
                key = f"{act_name}_{probe_type}"
                r2_values.append(results.get(key, {}).get("r2", 0))

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(ACTIVATION_POINTS))),
                    y=r2_values,
                    mode="lines+markers",
                    name=probe_type if col == 1 else None,
                    showlegend=(col == 1),
                    line=dict(width=2, color=colors[probe_type]),
                    marker=dict(size=8),
                ),
                row=1,
                col=col,
            )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(ACTIVATION_POINTS))),
        ticktext=[name[:10] for name in ACTIVATION_POINTS],
        tickangle=45,
    )
    fig.update_yaxes(title_text="R² Score", row=1, col=1)

    fig.update_layout(
        title=dict(
            text="LayerNorm vs RMSNorm: Position Information Flow",
            font=dict(size=20, family="Serif"),
        ),
        width=1200,
        height=500,
        template="plotly_white",
        legend=dict(x=0.4, y=0.98),
        margin=dict(l=60, r=50, t=80, b=120),
    )

    fig.write_image(f"{save_path}.png", width=1200, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def save_results(results: dict, save_path: str):
    """Save results to JSON."""
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {save_path}")


def print_summary_table(results: dict):
    """Print a summary table of results."""
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(
        f"{'Activation Point':<20} {'Probe Type':<25} {'R²':>8} {'Pearson r':>10} {'MAE':>8}"
    )
    print(f"{'-' * 80}")

    for act_name in ACTIVATION_POINTS:
        for probe_type in ["linear", "mlp", "linear_mean_subtracted"]:
            key = f"{act_name}_{probe_type}"
            if key in results:
                r = results[key]
                print(
                    f"{act_name:<20} {probe_type:<25} {r['r2']:>8.4f} {r['pearson_r']:>10.4f} {r['mae']:>8.2f}"
                )
        print()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Probe Analysis")
    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    setup_dirs()

    cfg = ProbeConfig(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        d_model=args.d_model,
        batch_size=args.batch_size,
    )

    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE PROBE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Samples: {cfg.n_samples}")
    print(f"Sequence length: {cfg.seq_len}")
    print(f"Model dimension: {cfg.d_model}")
    print(f"{'=' * 60}\n")

    # Generate random tokens
    tokens = generate_random_tokens(cfg)

    all_results = {}

    # ─── LayerNorm Model ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT: LayerNorm Model")
    print("=" * 60)

    model_ln = create_model(cfg, norm_type="LN")
    print("Extracting activations (LayerNorm)...")
    activations_ln = extract_all_activations(model_ln, tokens, cfg.batch_size)

    results_ln = run_probe_experiment(activations_ln, cfg, "LayerNorm")
    all_results["LayerNorm"] = results_ln

    del model_ln, activations_ln
    torch.cuda.empty_cache()

    # ─── RMSNorm Model ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT: RMSNorm Model")
    print("=" * 60)

    model_rms = create_model(cfg, norm_type="RMS")
    print("Extracting activations (RMSNorm)...")
    activations_rms = extract_all_activations(model_rms, tokens, cfg.batch_size)

    results_rms = run_probe_experiment(activations_rms, cfg, "RMSNorm")
    all_results["RMSNorm"] = results_rms

    del model_rms, activations_rms
    torch.cuda.empty_cache()

    # ─── Print Summaries ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYERNORM RESULTS")
    print("=" * 60)
    print_summary_table(results_ln)

    print("\n" + "=" * 60)
    print("RMSNORM RESULTS")
    print("=" * 60)
    print_summary_table(results_rms)

    # ─── Generate Plots ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Heatmaps
    plot_probe_results_heatmap(results_ln, str(RESULTS_DIR / "probe_heatmap_ln"))
    plot_probe_results_heatmap(results_rms, str(RESULTS_DIR / "probe_heatmap_rms"))

    # Trajectory plots
    plot_probe_trajectory(results_ln, str(RESULTS_DIR / "probe_trajectory_ln"))
    plot_probe_trajectory(results_rms, str(RESULTS_DIR / "probe_trajectory_rms"))

    # Comparison plot for paper
    plot_ln_vs_rmsln_comparison(
        results_ln, results_rms, str(PLOTS_DIR / "probe_ln_vs_rms_comparison")
    )

    # Also save trajectory to paper plots
    plot_probe_trajectory(results_ln, str(PLOTS_DIR / "position_info_trajectory"))

    # ─── Save Results ───────────────────────────────────────────────────────────
    save_results(all_results, str(RESULTS_DIR / "comprehensive_probe_results.json"))

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Paper figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
