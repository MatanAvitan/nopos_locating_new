"""
Hyperparameter Sweep: Effect on Position Encoding Ability

This script systematically varies model hyperparameters to study their effect
on the position encoding mechanism in NoPE transformers.

Hyperparameters Studied:
- d_model: Model dimension (256, 512, 768, 1024, 2048)
- n_heads: Number of attention heads (1, 2, 4, 8, 12)
- seq_len: Context length (32, 64, 128, 256, 512)
- vocab_size: Vocabulary size (1024, 4096, 16384, 50257)

Key Metrics:
- R² at post_attn (pre-LN position signal)
- R² at post_ln2 (post-LN position signal)
- R² from norm only vs direction only
- Correlation between theoretical variance and empirical

Usage:
    # Full sweep (takes ~2-4 hours on A100)
    CUDA_VISIBLE_DEVICES=0 python hyperparameter_sweep_probing.py --full

    # Quick test
    CUDA_VISIBLE_DEVICES=0 python hyperparameter_sweep_probing.py --quick

    # Single HP sweep
    CUDA_VISIBLE_DEVICES=0 python hyperparameter_sweep_probing.py --sweep d_model
"""

import os

# Only set GPU if not already specified by environment
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import itertools
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/hyperparameter_sweep")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Default HP values
DEFAULT_D_MODEL = 768
DEFAULT_N_HEADS = 1
DEFAULT_SEQ_LEN = 64
DEFAULT_VOCAB_SIZE = 50257
DEFAULT_N_SAMPLES = 10000
DEFAULT_BATCH_SIZE = 256

# Sweep ranges
D_MODEL_VALUES = [256, 512, 768, 1024, 2048]
N_HEADS_VALUES = [1, 2, 4, 8, 12]
SEQ_LEN_VALUES = [32, 64, 128, 256, 512]
VOCAB_SIZE_VALUES = [1024, 4096, 16384, 50257]

# Quick test values
QUICK_D_MODEL = [256, 768]
QUICK_N_HEADS = [1, 4]
QUICK_SEQ_LEN = [32, 64]
QUICK_VOCAB_SIZE = [1024, 50257]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    d_model: int = DEFAULT_D_MODEL
    n_heads: int = DEFAULT_N_HEADS
    seq_len: int = DEFAULT_SEQ_LEN
    vocab_size: int = DEFAULT_VOCAB_SIZE
    n_samples: int = DEFAULT_N_SAMPLES
    batch_size: int = DEFAULT_BATCH_SIZE
    norm_type: str = "LN"
    seed: int = 42

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_mlp(self) -> int:
        return self.d_model * 4


def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(cfg: ExperimentConfig):
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
        normalization_type=cfg.norm_type,
        device=device,
    )
    model = HookedTransformer(model_cfg)

    # Zero out positional embeddings for NoPE
    model.pos_embed.W_pos.data.zero_()
    model.pos_embed.W_pos.requires_grad = False

    return model


def generate_random_tokens(cfg: ExperimentConfig):
    """Generate random token sequences."""
    torch.manual_seed(cfg.seed)
    return torch.randint(0, cfg.vocab_size, (cfg.n_samples, cfg.seq_len), device=device)


def extract_activations(model, tokens: torch.Tensor, batch_size: int = 256):
    """Extract key activations for probing."""
    model.eval()
    n_samples = tokens.shape[0]

    # Storage
    post_attn_acts = []
    post_ln2_acts = []

    hook_names = [
        "blocks.0.hook_attn_out",  # post_attn
        "blocks.0.ln2.hook_normalized",  # post_ln2
    ]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_tokens = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(batch_tokens, names_filter=hook_names)

            post_attn_acts.append(cache["blocks.0.hook_attn_out"].detach().cpu())
            post_ln2_acts.append(cache["blocks.0.ln2.hook_normalized"].detach().cpu())

            del cache
            torch.cuda.empty_cache()

    return {
        "post_attn": torch.cat(post_attn_acts, dim=0).numpy(),
        "post_ln2": torch.cat(post_ln2_acts, dim=0).numpy(),
    }


def compute_theoretical_variance(seq_len: int, d_model: int):
    """
    Compute theoretical variance at each position.

    Theory: At position i, the attention output averages i+1 embeddings.
    Variance of mean of i+1 iid vectors with variance 1/d_model is 1/(d_model * (i+1))
    """
    positions = np.arange(seq_len)
    # Variance of output at position i ~ 1/(i+1) (normalized)
    theoretical_var = 1.0 / (positions + 1)
    return theoretical_var


def train_and_evaluate_probe(
    activations: np.ndarray, seq_len: int, train_ratio: float = 0.8
):
    """
    Train probe and return metrics.

    Returns dict with:
        - full_r2: R² using full activation
        - norm_r2: R² using only norm
        - direction_r2: R² using unit vector
        - theory_corr: Correlation with theoretical variance
    """
    n_samples, _, d_model = activations.shape
    positions = np.arange(seq_len)

    # Split data
    n_train = int(n_samples * train_ratio)

    train_acts = activations[:n_train]
    test_acts = activations[n_train:]

    # Flatten
    X_train = train_acts.reshape(-1, d_model)
    X_test = test_acts.reshape(-1, d_model)
    y_train = np.tile(positions, n_train)
    y_test = np.tile(positions, n_samples - n_train)

    # Full probe
    probe_full = Ridge(alpha=1.0)
    probe_full.fit(X_train, y_train)
    full_r2 = r2_score(y_test, probe_full.predict(X_test))

    # Norm only probe
    train_norms = np.linalg.norm(train_acts, axis=2).reshape(-1, 1)
    test_norms = np.linalg.norm(test_acts, axis=2).reshape(-1, 1)
    probe_norm = Ridge(alpha=1.0)
    probe_norm.fit(train_norms, y_train)
    norm_r2 = r2_score(y_test, probe_norm.predict(test_norms))

    # Direction only probe (unit vectors)
    train_dirs = train_acts / (np.linalg.norm(train_acts, axis=2, keepdims=True) + 1e-8)
    test_dirs = test_acts / (np.linalg.norm(test_acts, axis=2, keepdims=True) + 1e-8)
    X_train_dir = train_dirs.reshape(-1, d_model)
    X_test_dir = test_dirs.reshape(-1, d_model)
    probe_dir = Ridge(alpha=1.0)
    probe_dir.fit(X_train_dir, y_train)
    direction_r2 = r2_score(y_test, probe_dir.predict(X_test_dir))

    # Correlation with theoretical variance
    mean_norms_by_pos = np.linalg.norm(activations, axis=2).mean(axis=0)  # [seq_len]
    theoretical_var = compute_theoretical_variance(seq_len, d_model)
    # Norm should correlate negatively with position (higher pos -> lower norm)
    theory_corr, _ = pearsonr(mean_norms_by_pos, theoretical_var)

    # Also compute norm-position correlation
    all_norms = np.linalg.norm(activations, axis=2).flatten()
    all_positions = np.tile(positions, n_samples)
    norm_pos_corr, _ = pearsonr(all_norms, all_positions)

    return {
        "full_r2": float(full_r2),
        "norm_r2": float(norm_r2),
        "direction_r2": float(direction_r2),
        "theory_corr": float(theory_corr),
        "norm_position_corr": float(norm_pos_corr),
    }


def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run a single experiment with given config."""
    # Create model and generate tokens
    model = create_model(cfg)
    tokens = generate_random_tokens(cfg)

    # Extract activations
    activations = extract_activations(model, tokens, cfg.batch_size)

    # Evaluate probes at each layer
    post_attn_metrics = train_and_evaluate_probe(activations["post_attn"], cfg.seq_len)
    post_ln2_metrics = train_and_evaluate_probe(activations["post_ln2"], cfg.seq_len)

    # Cleanup
    del model, activations
    torch.cuda.empty_cache()

    return {
        "config": {
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_head": cfg.d_head,
            "seq_len": cfg.seq_len,
            "vocab_size": cfg.vocab_size,
            "n_samples": cfg.n_samples,
            "norm_type": cfg.norm_type,
        },
        "post_attn": post_attn_metrics,
        "post_ln2": post_ln2_metrics,
    }


def run_sweep(
    sweep_param: str,
    values: List,
    n_samples: int = DEFAULT_N_SAMPLES,
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """Run sweep over a single hyperparameter."""
    results = []

    for val in tqdm(values, desc=f"Sweeping {sweep_param}"):
        cfg = ExperimentConfig(n_samples=n_samples)
        setattr(cfg, sweep_param, val)

        # Ensure d_head is valid
        if sweep_param in ["d_model", "n_heads"]:
            if cfg.d_model % cfg.n_heads != 0:
                print(
                    f"  Skipping d_model={cfg.d_model}, n_heads={cfg.n_heads} (not divisible)"
                )
                continue

        print(f"\n  Running: {sweep_param}={val}")
        result = run_single_experiment(cfg)
        results.append(result)

    return results


def run_full_grid_search(
    quick: bool = False, n_samples: int = DEFAULT_N_SAMPLES
) -> Dict[str, List]:
    """Run full grid search over all hyperparameter combinations."""

    d_models = QUICK_D_MODEL if quick else D_MODEL_VALUES
    n_heads_list = QUICK_N_HEADS if quick else N_HEADS_VALUES
    seq_lens = QUICK_SEQ_LEN if quick else SEQ_LEN_VALUES
    vocab_sizes = QUICK_VOCAB_SIZE if quick else VOCAB_SIZE_VALUES

    all_results = {
        "d_model_sweep": [],
        "n_heads_sweep": [],
        "seq_len_sweep": [],
        "vocab_size_sweep": [],
        "grid_search": [],
    }

    # Individual sweeps (varying one HP at a time)
    print("\n" + "=" * 60)
    print("SWEEP: d_model")
    print("=" * 60)
    all_results["d_model_sweep"] = run_sweep("d_model", d_models, n_samples)

    print("\n" + "=" * 60)
    print("SWEEP: n_heads")
    print("=" * 60)
    all_results["n_heads_sweep"] = run_sweep("n_heads", n_heads_list, n_samples)

    print("\n" + "=" * 60)
    print("SWEEP: seq_len")
    print("=" * 60)
    all_results["seq_len_sweep"] = run_sweep("seq_len", seq_lens, n_samples)

    print("\n" + "=" * 60)
    print("SWEEP: vocab_size")
    print("=" * 60)
    all_results["vocab_size_sweep"] = run_sweep("vocab_size", vocab_sizes, n_samples)

    # Grid search (for d_model x seq_len interaction - most important)
    print("\n" + "=" * 60)
    print("GRID SEARCH: d_model × seq_len")
    print("=" * 60)

    for d_model, seq_len in tqdm(
        list(itertools.product(d_models, seq_lens)), desc="Grid search"
    ):
        cfg = ExperimentConfig(
            d_model=d_model,
            seq_len=seq_len,
            n_samples=n_samples,
        )
        print(f"\n  Running: d_model={d_model}, seq_len={seq_len}")
        result = run_single_experiment(cfg)
        all_results["grid_search"].append(result)

    return all_results


def plot_single_sweep(
    results: List[Dict], sweep_param: str, save_path: str, metric: str = "full_r2"
):
    """Plot results of a single parameter sweep."""
    x_vals = [r["config"][sweep_param] for r in results]

    post_attn_vals = [r["post_attn"][metric] for r in results]
    post_ln2_vals = [r["post_ln2"][metric] for r in results]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=post_attn_vals,
            mode="lines+markers",
            name="post_attn",
            line=dict(width=3, color="blue"),
            marker=dict(size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=post_ln2_vals,
            mode="lines+markers",
            name="post_ln2",
            line=dict(width=3, color="red"),
            marker=dict(size=10),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Position Encoding R² vs {sweep_param}",
            font=dict(size=20, family="Serif"),
        ),
        xaxis=dict(
            title=sweep_param,
            title_font=dict(size=16),
            tickfont=dict(size=14),
            type="log" if sweep_param == "vocab_size" else "linear",
        ),
        yaxis=dict(
            title="R² Score",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[0, 1],
        ),
        width=800,
        height=500,
        template="plotly_white",
        legend=dict(x=0.7, y=0.98),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_grid_search_heatmap(
    results: List[Dict],
    save_path: str,
    layer: str = "post_ln2",
    metric: str = "full_r2",
):
    """Plot heatmap of grid search results."""
    # Extract unique values
    d_models = sorted(set(r["config"]["d_model"] for r in results))
    seq_lens = sorted(set(r["config"]["seq_len"] for r in results))

    # Build matrix
    matrix = np.zeros((len(seq_lens), len(d_models)))
    for r in results:
        i = seq_lens.index(r["config"]["seq_len"])
        j = d_models.index(r["config"]["d_model"])
        matrix[i, j] = r[layer][metric]

    fig = px.imshow(
        matrix,
        x=[str(d) for d in d_models],
        y=[str(s) for s in seq_lens],
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        labels=dict(x="d_model", y="seq_len", color="R²"),
        text_auto=".2f",
        aspect="auto",
    )

    fig.update_layout(
        title=dict(
            text=f"Position Encoding R² ({layer}): d_model × seq_len",
            font=dict(size=20, family="Serif"),
        ),
        xaxis=dict(title="d_model", title_font=dict(size=16)),
        yaxis=dict(title="seq_len", title_font=dict(size=16)),
        width=700,
        height=500,
        template="plotly_white",
    )

    fig.write_image(f"{save_path}.png", width=700, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_all_sweeps_combined(all_results: Dict, save_path: str):
    """Plot all sweeps in a 2x2 grid."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("d_model", "n_heads", "seq_len", "vocab_size"),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    sweep_configs = [
        ("d_model_sweep", "d_model", 1, 1, False),
        ("n_heads_sweep", "n_heads", 1, 2, False),
        ("seq_len_sweep", "seq_len", 2, 1, False),
        ("vocab_size_sweep", "vocab_size", 2, 2, True),
    ]

    for sweep_name, param, row, col, log_x in sweep_configs:
        results = all_results[sweep_name]
        if not results:
            continue

        x_vals = [r["config"][param] for r in results]
        post_attn_vals = [r["post_attn"]["full_r2"] for r in results]
        post_ln2_vals = [r["post_ln2"]["full_r2"] for r in results]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=post_attn_vals,
                mode="lines+markers",
                name="post_attn" if (row == 1 and col == 1) else None,
                showlegend=(row == 1 and col == 1),
                line=dict(width=2, color="blue"),
                marker=dict(size=8),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=post_ln2_vals,
                mode="lines+markers",
                name="post_ln2" if (row == 1 and col == 1) else None,
                showlegend=(row == 1 and col == 1),
                line=dict(width=2, color="red"),
                marker=dict(size=8),
            ),
            row=row,
            col=col,
        )

        if log_x:
            fig.update_xaxes(type="log", row=row, col=col)
        fig.update_yaxes(range=[0, 1], row=row, col=col)

    fig.update_layout(
        title=dict(
            text="Position Encoding R² vs Hyperparameters",
            font=dict(size=22, family="Serif"),
        ),
        width=1000,
        height=800,
        template="plotly_white",
        legend=dict(x=0.85, y=0.98),
    )

    fig.write_image(f"{save_path}.png", width=1000, height=800, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_norm_vs_direction(all_results: Dict, save_path: str):
    """Plot norm R² vs direction R² for different HPs."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("post_attn", "post_ln2"),
        horizontal_spacing=0.12,
    )

    colors = {
        "d_model_sweep": "blue",
        "n_heads_sweep": "green",
        "seq_len_sweep": "orange",
        "vocab_size_sweep": "purple",
    }

    for sweep_name, color in colors.items():
        results = all_results.get(sweep_name, [])
        if not results:
            continue

        for layer_idx, layer in enumerate(["post_attn", "post_ln2"], 1):
            norm_r2s = [r[layer]["norm_r2"] for r in results]
            dir_r2s = [r[layer]["direction_r2"] for r in results]

            fig.add_trace(
                go.Scatter(
                    x=norm_r2s,
                    y=dir_r2s,
                    mode="markers",
                    name=sweep_name.replace("_sweep", "") if layer_idx == 1 else None,
                    showlegend=(layer_idx == 1),
                    marker=dict(size=10, color=color),
                ),
                row=1,
                col=layer_idx,
            )

    # Add diagonal line (norm = direction)
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="norm = direction" if col == 1 else None,
                showlegend=(col == 1),
                line=dict(width=1, color="gray", dash="dash"),
            ),
            row=1,
            col=col,
        )

    fig.update_xaxes(title="Norm R²", range=[0, 1])
    fig.update_yaxes(title="Direction R²", range=[0, 1])

    fig.update_layout(
        title=dict(
            text="Norm vs Direction R² Across Hyperparameters",
            font=dict(size=20, family="Serif"),
        ),
        width=1000,
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    fig.write_image(f"{save_path}.png", width=1000, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def print_summary(all_results: Dict):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("SUMMARY OF HYPERPARAMETER EFFECTS")
    print("=" * 80)

    for sweep_name in [
        "d_model_sweep",
        "n_heads_sweep",
        "seq_len_sweep",
        "vocab_size_sweep",
    ]:
        results = all_results.get(sweep_name, [])
        if not results:
            continue

        param = sweep_name.replace("_sweep", "")
        print(f"\n{param.upper()} SWEEP:")
        print("-" * 60)
        print(
            f"{'Value':<12} {'post_attn R²':<15} {'post_ln2 R²':<15} {'Norm R² (ln2)':<15}"
        )
        print("-" * 60)

        for r in results:
            val = r["config"][param]
            post_attn_r2 = r["post_attn"]["full_r2"]
            post_ln2_r2 = r["post_ln2"]["full_r2"]
            norm_r2 = r["post_ln2"]["norm_r2"]
            print(
                f"{val:<12} {post_attn_r2:<15.4f} {post_ln2_r2:<15.4f} {norm_r2:<15.4f}"
            )


def save_results(all_results: Dict, save_path: str):
    """Save results to JSON."""
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Sweep for Position Encoding"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full sweep (all values)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test (fewer values)"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["d_model", "n_heads", "seq_len", "vocab_size"],
        help="Run single sweep only",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of samples"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    setup_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP: POSITION ENCODING ANALYSIS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Samples: {args.n_samples}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    if args.sweep:
        # Single sweep mode
        print(f"\nRunning single sweep: {args.sweep}")

        values_map = {
            "d_model": QUICK_D_MODEL if args.quick else D_MODEL_VALUES,
            "n_heads": QUICK_N_HEADS if args.quick else N_HEADS_VALUES,
            "seq_len": QUICK_SEQ_LEN if args.quick else SEQ_LEN_VALUES,
            "vocab_size": QUICK_VOCAB_SIZE if args.quick else VOCAB_SIZE_VALUES,
        }

        results = run_sweep(args.sweep, values_map[args.sweep], args.n_samples)
        all_results = {f"{args.sweep}_sweep": results}

        # Plot single sweep
        plot_single_sweep(
            results, args.sweep, str(RESULTS_DIR / f"sweep_{args.sweep}_{timestamp}")
        )

    else:
        # Full grid search
        all_results = run_full_grid_search(quick=args.quick, n_samples=args.n_samples)

        # Generate all plots
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        # Individual sweep plots
        for sweep_name, param in [
            ("d_model_sweep", "d_model"),
            ("n_heads_sweep", "n_heads"),
            ("seq_len_sweep", "seq_len"),
            ("vocab_size_sweep", "vocab_size"),
        ]:
            if all_results.get(sweep_name):
                plot_single_sweep(
                    all_results[sweep_name], param, str(RESULTS_DIR / f"sweep_{param}")
                )

        # Combined plot
        plot_all_sweeps_combined(all_results, str(RESULTS_DIR / "all_sweeps_combined"))
        plot_all_sweeps_combined(
            all_results, str(PLOTS_DIR / "hp_sweep_position_encoding")
        )

        # Grid search heatmap
        if all_results.get("grid_search"):
            plot_grid_search_heatmap(
                all_results["grid_search"],
                str(RESULTS_DIR / "grid_dmodel_seqlen_post_ln2"),
            )
            plot_grid_search_heatmap(
                all_results["grid_search"],
                str(RESULTS_DIR / "grid_dmodel_seqlen_post_attn"),
                layer="post_attn",
            )
            # Paper figure
            plot_grid_search_heatmap(
                all_results["grid_search"],
                str(PLOTS_DIR / "position_encoding_grid_search"),
            )

        # Norm vs Direction plot
        plot_norm_vs_direction(all_results, str(RESULTS_DIR / "norm_vs_direction"))
        plot_norm_vs_direction(all_results, str(PLOTS_DIR / "hp_norm_vs_direction"))

    # Print summary
    print_summary(all_results)

    # Save results
    save_results(all_results, str(RESULTS_DIR / f"hp_sweep_results_{timestamp}.json"))
    save_results(all_results, str(RESULTS_DIR / "hp_sweep_results_latest.json"))

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP COMPLETE")
    print("=" * 60)
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
