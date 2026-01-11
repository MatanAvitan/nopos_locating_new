"""
Single-Sample Position Analysis

This script investigates what property of a SINGLE activation vector encodes position,
addressing the fundamental issue that population statistics cannot be the mechanism
(the model sees only one sample at inference time).

Key Questions:
1. Does the NORM of individual activations correlate with position?
2. Is there a learned DIRECTION w such that w·h_i correlates with position for single samples?
3. Does distance from embedding centroid encode position?
4. What geometric property survives LayerNorm normalization?

Hypotheses Tested:
- H1: Activation norm encodes position (norm grows with position due to averaging)
- H2: A specific direction w (learned via probe) encodes position per-sample
- H3: Cosine similarity to a reference vector encodes position
- H4: The "shape" (direction) of the vector encodes position, not magnitude

Usage:
    python single_sample_position_analysis.py --n_samples 5000 --seq_len 64
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig


# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/single_sample_analysis")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    n_samples: int = 5000
    seq_len: int = 64
    d_model: int = 1024
    n_heads: int = 1
    d_head: int = 1024
    d_mlp: int = 4096
    vocab_size: int = 50257
    batch_size: int = 256
    seed: int = 42


# Hook points to analyze (all should have d_model dimensions)
# Note: We exclude post_mlp (blocks.0.mlp.hook_post) because it has d_mlp dimensions
HOOK_POINTS = {
    "embed": "hook_embed",
    "post_ln1": "blocks.0.ln1.hook_normalized",
    "post_attn": "blocks.0.hook_attn_out",
    "post_attn_residual": "blocks.0.hook_resid_mid",
    "post_ln2": "blocks.0.ln2.hook_normalized",
    # "post_mlp": "blocks.0.mlp.hook_post",  # Excluded: has d_mlp dimensions, not d_model
    "post_mlp_residual": "blocks.0.hook_resid_post",
}


def setup_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(cfg: Config, norm_type: str = "LN"):
    """Create NoPE model."""
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
    model.pos_embed.W_pos.data.zero_()
    model.pos_embed.W_pos.requires_grad = False
    return model


def extract_activations(model, tokens: torch.Tensor, batch_size: int = 256):
    """Extract activations at all hook points."""
    model.eval()
    n_samples = tokens.shape[0]
    activations = {name: [] for name in HOOK_POINTS}

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting"):
            batch = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(
                batch, names_filter=list(HOOK_POINTS.values())
            )

            for name, hook in HOOK_POINTS.items():
                activations[name].append(cache[hook].cpu())

            del cache
            torch.cuda.empty_cache()

    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0).numpy()

    return activations


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 1: Norm encodes position
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_norm_per_sample(activations: dict, cfg: Config):
    """
    For each sample, compute the correlation between position and activation norm.

    This tests whether WITHIN a single sequence, norm correlates with position.
    """
    results = {}

    for name, acts in activations.items():
        # acts: [n_samples, seq_len, d_model]
        norms = np.linalg.norm(acts, axis=2)  # [n_samples, seq_len]

        positions = np.arange(cfg.seq_len)

        # Per-sample correlation
        per_sample_corr = []
        for i in range(cfg.n_samples):
            r, _ = pearsonr(positions, norms[i])
            per_sample_corr.append(r)

        per_sample_corr = np.array(per_sample_corr)

        # Population-level (all samples pooled)
        all_norms = norms.flatten()
        all_positions = np.tile(positions, cfg.n_samples)
        pop_r, pop_p = pearsonr(all_positions, all_norms)

        results[name] = {
            "per_sample_corr_mean": float(np.mean(per_sample_corr)),
            "per_sample_corr_std": float(np.std(per_sample_corr)),
            "per_sample_corr_median": float(np.median(per_sample_corr)),
            "per_sample_positive_frac": float(np.mean(per_sample_corr > 0)),
            "per_sample_strong_frac": float(np.mean(per_sample_corr > 0.5)),
            "population_corr": float(pop_r),
            "population_p_value": float(pop_p),
            "mean_norm_by_position": norms.mean(axis=0).tolist(),
        }

        print(f"\n{name}:")
        print(
            f"  Per-sample correlation: {np.mean(per_sample_corr):.4f} ± {np.std(per_sample_corr):.4f}"
        )
        print(f"  Fraction with positive corr: {np.mean(per_sample_corr > 0):.2%}")
        print(f"  Fraction with r > 0.5: {np.mean(per_sample_corr > 0.5):.2%}")
        print(f"  Population correlation: {pop_r:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 2: Learned direction encodes position
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_learned_direction(activations: dict, cfg: Config):
    """
    Train a linear probe to find direction w, then test if w·h works per-sample.

    Key insight: If the probe works, it finds w such that w·h ≈ position.
    But does this work for INDIVIDUAL samples, or only in expectation?
    """
    results = {}
    positions = np.arange(cfg.seq_len)

    for name, acts in activations.items():
        print(f"\n{name}:")

        # Get actual dimension from data
        n_samples, seq_len, d_act = acts.shape

        # Split into train/test
        n_train = int(n_samples * 0.8)
        train_acts = acts[:n_train]  # [n_train, seq_len, d_act]
        test_acts = acts[n_train:]  # [n_test, seq_len, d_act]

        # Flatten for training
        X_train = train_acts.reshape(-1, d_act)
        y_train = np.tile(positions, n_train)

        # Train probe
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        w = probe.coef_  # This is the "position direction"
        w_normalized = w / np.linalg.norm(w)

        # Test on held-out samples: compute projection w·h for each position
        n_test = n_samples - n_train
        projections = test_acts @ w_normalized  # [n_test, seq_len]

        # Per-sample correlation between projection and position
        per_sample_corr = []
        for i in range(n_test):
            r, _ = pearsonr(positions, projections[i])
            per_sample_corr.append(r)

        per_sample_corr = np.array(per_sample_corr)

        # Population-level
        pop_r, _ = pearsonr(np.tile(positions, n_test), projections.flatten())

        # Also test: can we decode position from projection alone?
        # Simple linear fit: position ≈ a * projection + b
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        X_test_flat = projections.flatten().reshape(-1, 1)
        y_test_flat = np.tile(positions, n_test)
        lr.fit(X_test_flat, y_test_flat)
        y_pred = lr.predict(X_test_flat)
        r2 = r2_score(y_test_flat, y_pred)

        results[name] = {
            "per_sample_corr_mean": float(np.mean(per_sample_corr)),
            "per_sample_corr_std": float(np.std(per_sample_corr)),
            "per_sample_positive_frac": float(np.mean(per_sample_corr > 0)),
            "per_sample_strong_frac": float(np.mean(per_sample_corr > 0.5)),
            "population_corr": float(pop_r),
            "r2_from_projection": float(r2),
            "probe_weight_norm": float(np.linalg.norm(w)),
            "mean_projection_by_position": projections.mean(axis=0).tolist(),
        }

        print(
            f"  Per-sample projection-position corr: {np.mean(per_sample_corr):.4f} ± {np.std(per_sample_corr):.4f}"
        )
        print(f"  Fraction with positive corr: {np.mean(per_sample_corr > 0):.2%}")
        print(f"  Fraction with r > 0.5: {np.mean(per_sample_corr > 0.5):.2%}")
        print(f"  R² from projection alone: {r2:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 3: Distance from centroid encodes position
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_centroid_distance(activations: dict, cfg: Config):
    """
    Test if distance from the embedding centroid encodes position.

    Intuition: Position i averages i+1 embeddings, so it should be closer to
    the embedding centroid as i increases (law of large numbers).
    """
    results = {}

    # Get embedding layer activations (these are the raw token embeddings)
    embed_acts = activations["embed"]  # [n_samples, seq_len, d_model]

    # Compute centroid per sample (mean of all embeddings in the sequence)
    sample_centroids = embed_acts.mean(axis=1, keepdims=True)  # [n_samples, 1, d_model]

    for name, acts in activations.items():
        print(f"\n{name}:")

        # Distance from sample's own centroid
        distances = np.linalg.norm(
            acts - sample_centroids, axis=2
        )  # [n_samples, seq_len]

        positions = np.arange(cfg.seq_len)

        # Per-sample correlation
        per_sample_corr = []
        for i in range(cfg.n_samples):
            r, _ = pearsonr(positions, distances[i])
            per_sample_corr.append(r)

        per_sample_corr = np.array(per_sample_corr)

        # Population-level
        pop_r, _ = pearsonr(np.tile(positions, cfg.n_samples), distances.flatten())

        results[name] = {
            "per_sample_corr_mean": float(np.mean(per_sample_corr)),
            "per_sample_corr_std": float(np.std(per_sample_corr)),
            "per_sample_positive_frac": float(np.mean(per_sample_corr > 0)),
            "population_corr": float(pop_r),
            "mean_distance_by_position": distances.mean(axis=0).tolist(),
        }

        print(
            f"  Per-sample distance-position corr: {np.mean(per_sample_corr):.4f} ± {np.std(per_sample_corr):.4f}"
        )
        print(f"  Population correlation: {pop_r:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 4: Direction (not magnitude) encodes position after LayerNorm
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_direction_encoding(activations: dict, cfg: Config):
    """
    LayerNorm normalizes magnitude but preserves direction.
    Test if the DIRECTION of the vector encodes position.

    Approach: Normalize all vectors to unit norm, then train probe.
    If probe still works, direction encodes position.
    """
    results = {}
    positions = np.arange(cfg.seq_len)

    for name, acts in activations.items():
        print(f"\n{name}:")

        # Get actual dimensions from data
        n_samples, seq_len, d_act = acts.shape

        # Normalize to unit vectors
        norms = np.linalg.norm(acts, axis=2, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        unit_acts = acts / norms  # [n_samples, seq_len, d_act]

        # Split
        n_train = int(n_samples * 0.8)
        train_acts = unit_acts[:n_train]
        test_acts = unit_acts[n_train:]

        # Train probe on unit vectors
        X_train = train_acts.reshape(-1, d_act)
        y_train = np.tile(positions, n_train)

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)

        # Test
        n_test = n_samples - n_train
        X_test = test_acts.reshape(-1, d_act)
        y_test = np.tile(positions, n_test)
        y_pred = probe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)

        # Per-sample accuracy
        y_pred_reshaped = y_pred.reshape(n_test, cfg.seq_len)

        per_sample_corr = []
        for i in range(n_test):
            corr, _ = pearsonr(positions, y_pred_reshaped[i])
            per_sample_corr.append(corr)

        per_sample_corr = np.array(per_sample_corr)

        results[name] = {
            "unit_vector_r2": float(r2),
            "unit_vector_pearson": float(r),
            "per_sample_corr_mean": float(np.mean(per_sample_corr)),
            "per_sample_corr_std": float(np.std(per_sample_corr)),
            "per_sample_positive_frac": float(np.mean(per_sample_corr > 0)),
        }

        print(f"  Unit-vector probe R²: {r2:.4f}")
        print(
            f"  Per-sample corr (predictions vs true): {np.mean(per_sample_corr):.4f} ± {np.std(per_sample_corr):.4f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 5: Cumulative average signature
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_cumulative_average_signature(activations: dict, cfg: Config):
    """
    The attention output at position i is the average of embeddings 0..i.
    This creates a specific geometric signature: h_i = (1/(i+1)) * sum(e_0..e_i)

    Test: Does the relationship between consecutive positions encode position?
    E.g., h_{i+1} - h_i should have a specific structure related to position.
    """
    results = {}

    for name, acts in activations.items():
        print(f"\n{name}:")

        # Compute differences between consecutive positions
        diffs = acts[:, 1:, :] - acts[:, :-1, :]  # [n_samples, seq_len-1, d_model]
        diff_norms = np.linalg.norm(diffs, axis=2)  # [n_samples, seq_len-1]

        positions = np.arange(cfg.seq_len - 1)  # 0 to seq_len-2

        # Per-sample correlation between position and diff norm
        per_sample_corr = []
        for i in range(cfg.n_samples):
            r, _ = pearsonr(positions, diff_norms[i])
            per_sample_corr.append(r)

        per_sample_corr = np.array(per_sample_corr)

        # Theoretical prediction: diff_norm should decrease as 1/i
        # Because h_i = avg(e_0..e_i), so h_{i+1} - h_i = (e_{i+1} - h_i) / (i+2)

        results[name] = {
            "per_sample_corr_mean": float(np.mean(per_sample_corr)),
            "per_sample_corr_std": float(np.std(per_sample_corr)),
            "mean_diff_norm_by_position": diff_norms.mean(axis=0).tolist(),
        }

        print(
            f"  Per-sample diff-norm vs position corr: {np.mean(per_sample_corr):.4f} ± {np.std(per_sample_corr):.4f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL TEST: Can position be decoded from a SINGLE token's activation?
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_single_token_decodability(activations: dict, cfg: Config):
    """
    The ultimate test: Given ONE activation vector h_i from ONE sample,
    can we decode position i?

    This is different from the per-sample correlation tests, which use
    all 64 positions from one sample. Here we test truly single-vector decoding.
    """
    results = {}
    positions = np.arange(cfg.seq_len)

    for name, acts in activations.items():
        print(f"\n{name}:")

        # Get actual dimensions from data
        n_samples, seq_len, d_act = acts.shape

        # Flatten everything
        n_train = int(n_samples * 0.8)
        n_test = n_samples - n_train

        X_train = acts[:n_train].reshape(-1, d_act)
        y_train = np.tile(positions, n_train)

        X_test = acts[n_train:].reshape(-1, d_act)
        y_test = np.tile(positions, n_test)

        # Train probe
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_pred - y_test))
        pearson_r, _ = pearsonr(y_test, y_pred)

        # Accuracy within k positions
        acc_1 = np.mean(np.abs(y_pred - y_test) < 1)
        acc_3 = np.mean(np.abs(y_pred - y_test) < 3)
        acc_5 = np.mean(np.abs(y_pred - y_test) < 5)

        # Exact accuracy
        exact_acc = np.mean(np.round(y_pred) == y_test)

        # Per-position accuracy
        per_pos_mae = []
        for p in positions:
            mask = y_test == p
            per_pos_mae.append(float(np.mean(np.abs(y_pred[mask] - p))))

        results[name] = {
            "r2": float(r2),
            "mae": float(mae),
            "pearson_r": float(pearson_r),
            "accuracy_within_1": float(acc_1),
            "accuracy_within_3": float(acc_3),
            "accuracy_within_5": float(acc_5),
            "exact_accuracy": float(exact_acc),
            "per_position_mae": per_pos_mae,
        }

        print(f"  Single-token decoding R²: {r2:.4f}")
        print(f"  MAE: {mae:.2f} positions")
        print(f"  Accuracy within ±1: {acc_1:.2%}")
        print(f"  Accuracy within ±5: {acc_5:.2%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_per_sample_correlations(
    results_norm: dict, results_direction: dict, results_centroid: dict, save_path: str
):
    """Compare per-sample correlation across different hypotheses."""

    hook_names = list(results_norm.keys())

    fig = go.Figure()

    # Norm correlation
    norm_corrs = [results_norm[h]["per_sample_corr_mean"] for h in hook_names]
    fig.add_trace(
        go.Bar(
            name="Norm",
            x=hook_names,
            y=norm_corrs,
            error_y=dict(
                type="data",
                array=[results_norm[h]["per_sample_corr_std"] for h in hook_names],
            ),
        )
    )

    # Direction (learned) correlation
    dir_corrs = [results_direction[h]["per_sample_corr_mean"] for h in hook_names]
    fig.add_trace(
        go.Bar(
            name="Learned Direction",
            x=hook_names,
            y=dir_corrs,
            error_y=dict(
                type="data",
                array=[results_direction[h]["per_sample_corr_std"] for h in hook_names],
            ),
        )
    )

    # Centroid distance correlation
    cent_corrs = [results_centroid[h]["per_sample_corr_mean"] for h in hook_names]
    fig.add_trace(
        go.Bar(
            name="Centroid Distance",
            x=hook_names,
            y=cent_corrs,
            error_y=dict(
                type="data",
                array=[results_centroid[h]["per_sample_corr_std"] for h in hook_names],
            ),
        )
    )

    fig.update_layout(
        title="Per-Sample Position Correlation by Hypothesis",
        xaxis_title="Activation Point",
        yaxis_title="Mean Per-Sample Correlation",
        barmode="group",
        template="plotly_white",
        width=1000,
        height=500,
    )

    fig.write_image(f"{save_path}.png", width=1000, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")


def plot_single_token_accuracy(results: dict, save_path: str):
    """Plot single-token decoding accuracy by layer."""

    hook_names = list(results.keys())

    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² Score", "MAE (positions)"))

    r2_vals = [results[h]["r2"] for h in hook_names]
    mae_vals = [results[h]["mae"] for h in hook_names]

    fig.add_trace(go.Bar(x=hook_names, y=r2_vals, name="R²"), row=1, col=1)
    fig.add_trace(go.Bar(x=hook_names, y=mae_vals, name="MAE"), row=1, col=2)

    fig.update_layout(
        title="Single-Token Position Decoding Accuracy",
        template="plotly_white",
        width=1000,
        height=400,
        showlegend=False,
    )

    fig.write_image(f"{save_path}.png", width=1000, height=400, scale=2)
    fig.write_image(f"{save_path}.pdf")


def plot_norm_by_position(results: dict, save_path: str):
    """Plot how norm varies with position at each layer."""

    fig = go.Figure()

    for name, data in results.items():
        norms = data["mean_norm_by_position"]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(norms))),
                y=norms,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Mean Activation Norm by Position",
        xaxis_title="Position",
        yaxis_title="Mean Norm",
        template="plotly_white",
        width=800,
        height=500,
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Single-Sample Position Analysis")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    setup_dirs()

    cfg = Config(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        d_model=args.d_model,
        batch_size=args.batch_size,
    )

    print("=" * 70)
    print("SINGLE-SAMPLE POSITION ANALYSIS")
    print("=" * 70)
    print(f"Samples: {cfg.n_samples}")
    print(f"Sequence length: {cfg.seq_len}")
    print(f"Model dimension: {cfg.d_model}")
    print("=" * 70)

    # Generate random tokens
    torch.manual_seed(cfg.seed)
    tokens = torch.randint(
        0, cfg.vocab_size, (cfg.n_samples, cfg.seq_len), device=device
    )

    # Create model and extract activations
    print("\nCreating NoPE model (LayerNorm)...")
    model = create_model(cfg, norm_type="LN")

    print("Extracting activations...")
    activations = extract_activations(model, tokens, cfg.batch_size)

    del model
    torch.cuda.empty_cache()

    all_results = {}

    # ─── Hypothesis 1: Norm encodes position ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Activation norm encodes position")
    print("=" * 70)
    results_norm = analyze_norm_per_sample(activations, cfg)
    all_results["norm_analysis"] = results_norm

    # ─── Hypothesis 2: Learned direction ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Learned direction encodes position")
    print("=" * 70)
    results_direction = analyze_learned_direction(activations, cfg)
    all_results["direction_analysis"] = results_direction

    # ─── Hypothesis 3: Centroid distance ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Distance from centroid encodes position")
    print("=" * 70)
    results_centroid = analyze_centroid_distance(activations, cfg)
    all_results["centroid_analysis"] = results_centroid

    # ─── Hypothesis 4: Direction (unit vectors) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Direction (not magnitude) encodes position")
    print("=" * 70)
    results_unit = analyze_direction_encoding(activations, cfg)
    all_results["unit_vector_analysis"] = results_unit

    # ─── Hypothesis 5: Cumulative average signature ──────────────────────────────
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: Cumulative average signature")
    print("=" * 70)
    results_cumsum = analyze_cumulative_average_signature(activations, cfg)
    all_results["cumsum_analysis"] = results_cumsum

    # ─── Critical test: Single-token decodability ────────────────────────────────
    print("\n" + "=" * 70)
    print("CRITICAL TEST: Single-token position decoding")
    print("=" * 70)
    results_single = analyze_single_token_decodability(activations, cfg)
    all_results["single_token_decoding"] = results_single

    # ─── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Per-Sample Position Encoding")
    print("=" * 70)
    print("\nWhich properties encode position WITHIN a single sample?")
    print("-" * 70)
    print(
        f"{'Layer':<20} {'Norm r':<12} {'Direction r':<12} {'Centroid r':<12} {'Single R²':<12}"
    )
    print("-" * 70)

    for name in HOOK_POINTS:
        norm_r = results_norm[name]["per_sample_corr_mean"]
        dir_r = results_direction[name]["per_sample_corr_mean"]
        cent_r = results_centroid[name]["per_sample_corr_mean"]
        single_r2 = results_single[name]["r2"]
        print(
            f"{name:<20} {norm_r:<12.4f} {dir_r:<12.4f} {cent_r:<12.4f} {single_r2:<12.4f}"
        )

    # ─── Generate plots ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_per_sample_correlations(
        results_norm,
        results_direction,
        results_centroid,
        str(RESULTS_DIR / "per_sample_correlations"),
    )

    plot_single_token_accuracy(
        results_single, str(RESULTS_DIR / "single_token_accuracy")
    )
    plot_norm_by_position(results_norm, str(RESULTS_DIR / "norm_by_position"))

    # ─── Save results ────────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "single_sample_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
