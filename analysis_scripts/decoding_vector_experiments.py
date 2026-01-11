"""
Decoding Vector Experiments

This script validates and extends the decoding vector construction:
    w = W_V · Σ_j LN(E_j)

The decoding vector exploits the approximate orthogonality of random embeddings.
For uniformly sampled tokens with uniform attention, the projection onto w encodes position.

Experiments:
- T4.1: Orthogonality Verification - measure ||E_i · E_j|| for i≠j
- T4.2: Decoding Vector Correlation - corr(w · h_i, i)
- T4.3: Decoding After LN - does it work after LayerNorm?
- T4.4: Decoding Vector Ablation - project out w, measure accuracy drop
- T4.5: Alternative Decoding Vectors - compare different constructions
- T4.6: Vocabulary Size Scaling - how does accuracy depend on vocab size?
- T4.7: Context Length Scaling - length extrapolation

Usage:
    python decoding_vector_experiments.py --n_samples 30000 --seq_len 64
"""

import os

# Only set GPU if not already specified by environment
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/decoding_vector_experiments")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """Configuration for decoding vector experiments."""

    n_samples: int = 30000
    seq_len: int = 64
    d_model: int = 1024
    n_heads: int = 1
    d_head: int = 1024
    d_mlp: int = 4096
    vocab_size: int = 50257
    batch_size: int = 256
    seed: int = 42


def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(cfg: ExperimentConfig, vocab_size: int = None, norm_type: str = "LN"):
    """Create a HookedTransformer model without positional embeddings."""
    if vocab_size is None:
        vocab_size = cfg.vocab_size

    model_cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        d_vocab=vocab_size,
        n_ctx=cfg.seq_len,
        act_fn="gelu",
        normalization_type=norm_type,
        device=device,
    )
    model = HookedTransformer(model_cfg)

    # Zero out positional embeddings
    model.pos_embed.W_pos.data.zero_()
    model.pos_embed.W_pos.requires_grad = False

    return model


def generate_random_tokens(
    n_samples: int, seq_len: int, vocab_size: int, seed: int = 42
):
    """Generate random token sequences."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (n_samples, seq_len), device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT T4.1: ORTHOGONALITY VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def verify_embedding_orthogonality(model, n_pairs: int = 10000):
    """
    T4.1: Verify that random embeddings are approximately orthogonal.

    Computes E_i · E_j / (||E_i|| ||E_j||) for random pairs i≠j.
    For truly orthogonal vectors, this should be 0.

    Returns:
        cosine_similarities: array of pairwise cosine similarities
        statistics: dict with mean, std, max of |cos|
    """
    with torch.no_grad():
        E = model.embed.W_E.detach()  # [vocab_size, d_model]
        vocab_size, d_model = E.shape

        # Normalize embeddings
        E_norm = E / E.norm(dim=1, keepdim=True)

        # Sample random pairs
        torch.manual_seed(42)
        i_indices = torch.randint(0, vocab_size, (n_pairs,), device=device)
        j_indices = torch.randint(0, vocab_size, (n_pairs,), device=device)

        # Ensure i != j
        mask = i_indices == j_indices
        while mask.any():
            j_indices[mask] = torch.randint(0, vocab_size, (mask.sum(),), device=device)
            mask = i_indices == j_indices

        # Compute cosine similarities
        cos_sims = (E_norm[i_indices] * E_norm[j_indices]).sum(dim=1)
        cos_sims_np = cos_sims.cpu().numpy()

    statistics = {
        "mean_abs_cosine": float(np.abs(cos_sims_np).mean()),
        "std_cosine": float(cos_sims_np.std()),
        "max_abs_cosine": float(np.abs(cos_sims_np).max()),
        "expected_for_random": float(
            1 / np.sqrt(d_model)
        ),  # Expected for random vectors
    }

    print("\n[T4.1] Embedding Orthogonality Verification")
    print(f"  Mean |cos(E_i, E_j)|: {statistics['mean_abs_cosine']:.6f}")
    print(f"  Std cos(E_i, E_j): {statistics['std_cosine']:.6f}")
    print(f"  Max |cos(E_i, E_j)|: {statistics['max_abs_cosine']:.6f}")
    print(f"  Expected for random: {statistics['expected_for_random']:.6f}")

    return cos_sims_np, statistics


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT T4.2-T4.3: DECODING VECTOR CONSTRUCTION AND CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_decoding_vector(model, apply_ln: bool = True):
    """
    Compute the decoding vector: w = W_V · Σ_j LN(E_j) or w = W_V · Σ_j E_j

    Args:
        model: HookedTransformer model
        apply_ln: whether to apply LayerNorm before summing

    Returns:
        w: [d_model] decoding vector
    """
    with torch.no_grad():
        E = model.embed.W_E.detach()  # [vocab_size, d_model]

        if apply_ln:
            # Apply LayerNorm
            ln = model.blocks[0].ln1
            E_processed = ln(E)  # [vocab_size, d_model]
        else:
            E_processed = E

        # Sum of processed embeddings
        E_sum = E_processed.sum(dim=0)  # [d_model]

        # Get W_V from attention
        W_V = model.blocks[0].attn.W_V.squeeze(0)  # [d_model, d_head]

        # Decoding vector: w = W_V · E_sum
        # For single-head case: W_V is [d_model, d_head]
        # We want w in d_model space for projecting activations
        # Actually, w = W_V^T · E_sum would give [d_head]
        # But we want to project d_model activations, so use W_V · E_sum

        # More precisely: The value output at position i is V_i = W_V · h_i
        # For uniform attention, the output is (1/(i+1)) Σ_{j≤i} V_j
        # The decoding vector in VALUE space is: w_v = Σ_j W_V · LN(E_j)

        w = W_V.T @ E_sum  # [d_head]

        # For projecting onto full d_model space, we need W_O · w
        W_O = model.blocks[0].attn.W_O.squeeze(0)  # [d_head, d_model]
        w_full = W_O.T @ w  # [d_model]

    return w_full.cpu().numpy()


def compute_decoding_vector_alternative(model, method: str = "sum_embeddings"):
    """
    Compute alternative decoding vectors.

    Methods:
    - "sum_embeddings": w = Σ_j E_j (no LN, no W_V)
    - "sum_ln_embeddings": w = Σ_j LN(E_j) (with LN, no W_V)
    - "full": w = W_O @ W_V @ Σ_j LN(E_j) (full formula)
    - "learned": fit optimal direction from data
    """
    with torch.no_grad():
        E = model.embed.W_E.detach()

        if method == "sum_embeddings":
            w = E.sum(dim=0)
        elif method == "sum_ln_embeddings":
            ln = model.blocks[0].ln1
            E_ln = ln(E)
            w = E_ln.sum(dim=0)
        elif method == "full":
            ln = model.blocks[0].ln1
            E_ln = ln(E)
            E_sum = E_ln.sum(dim=0)
            W_V = model.blocks[0].attn.W_V.squeeze(0)
            W_O = model.blocks[0].attn.W_O.squeeze(0)
            w = W_O @ W_V @ E_sum
        else:
            raise ValueError(f"Unknown method: {method}")

    return w.cpu().numpy()


def test_decoding_vector_correlation(
    model,
    tokens: torch.Tensor,
    decoding_vector: np.ndarray,
    hook_name: str = "blocks.0.hook_resid_mid",
    batch_size: int = 256,
):
    """
    T4.2-T4.3: Test correlation between decoding vector projection and position.

    Returns:
        projections_mean: [seq_len] mean projection at each position
        correlation: Pearson correlation with true position
        per_sample_corrs: correlation for each sample
    """
    model.eval()
    n_samples, seq_len = tokens.shape

    all_projections = []
    w_tensor = torch.tensor(decoding_vector, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_tokens = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(batch_tokens, names_filter=[hook_name])
            acts = cache[hook_name]  # [batch, seq_len, d_model]

            # Project onto decoding vector
            proj = (acts * w_tensor).sum(dim=-1)  # [batch, seq_len]
            all_projections.append(proj.cpu())

            del cache
            torch.cuda.empty_cache()

    all_projections = torch.cat(all_projections, dim=0).numpy()  # [n_samples, seq_len]

    # Mean projection at each position
    projections_mean = all_projections.mean(axis=0)  # [seq_len]

    # Overall correlation
    positions = np.arange(seq_len)
    y_flat = np.tile(positions, n_samples)
    proj_flat = all_projections.flatten()

    correlation, p_value = pearsonr(proj_flat, y_flat)
    spearman_corr, _ = spearmanr(proj_flat, y_flat)

    # Per-sample correlations
    per_sample_corrs = np.array(
        [pearsonr(all_projections[i], positions)[0] for i in range(n_samples)]
    )

    return {
        "projections_mean": projections_mean,
        "correlation": float(correlation),
        "spearman_correlation": float(spearman_corr),
        "p_value": float(p_value),
        "per_sample_corr_mean": float(per_sample_corrs.mean()),
        "per_sample_corr_std": float(per_sample_corrs.std()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT T4.4: DECODING VECTOR ABLATION
# ═══════════════════════════════════════════════════════════════════════════════


def ablate_decoding_vector(
    model,
    tokens: torch.Tensor,
    decoding_vector: np.ndarray,
    hook_name: str = "blocks.0.hook_resid_mid",
    batch_size: int = 256,
):
    """
    T4.4: Project out the decoding vector direction and measure probe accuracy drop.

    Computes:
    1. Baseline probe accuracy on original activations
    2. Ablated probe accuracy after removing decoding vector direction
    """
    from sklearn.linear_model import Ridge

    model.eval()
    n_samples, seq_len = tokens.shape

    w = torch.tensor(decoding_vector, dtype=torch.float32, device=device)
    w_norm = w / w.norm()  # Normalize

    all_acts_original = []
    all_acts_ablated = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting activations"):
            batch_tokens = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(batch_tokens, names_filter=[hook_name])
            acts = cache[hook_name]  # [batch, seq_len, d_model]

            # Original activations
            all_acts_original.append(acts.cpu())

            # Ablated: project out decoding vector direction
            # h_ablated = h - (h · w_norm) * w_norm
            proj_coef = (acts * w_norm).sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]
            acts_ablated = acts - proj_coef * w_norm
            all_acts_ablated.append(acts_ablated.cpu())

            del cache
            torch.cuda.empty_cache()

    acts_original = torch.cat(all_acts_original, dim=0).numpy()
    acts_ablated = torch.cat(all_acts_ablated, dim=0).numpy()

    # Flatten for probing
    d_model = acts_original.shape[-1]
    X_original = acts_original.reshape(-1, d_model)
    X_ablated = acts_ablated.reshape(-1, d_model)
    y = np.tile(np.arange(seq_len), n_samples)

    # Train/test split
    train_size = int(0.8 * len(y))
    X_orig_train, X_orig_test = X_original[:train_size], X_original[train_size:]
    X_abl_train, X_abl_test = X_ablated[:train_size], X_ablated[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train probes
    probe_original = Ridge(alpha=1.0)
    probe_original.fit(X_orig_train, y_train)

    probe_ablated = Ridge(alpha=1.0)
    probe_ablated.fit(X_abl_train, y_train)

    # Evaluate
    pred_original = probe_original.predict(X_orig_test)
    pred_ablated = probe_ablated.predict(X_abl_test)

    r2_original = r2_score(y_test, pred_original)
    r2_ablated = r2_score(y_test, pred_ablated)

    mae_original = mean_absolute_error(y_test, pred_original)
    mae_ablated = mean_absolute_error(y_test, pred_ablated)

    corr_original, _ = pearsonr(y_test, pred_original)
    corr_ablated, _ = pearsonr(y_test, pred_ablated)

    return {
        "original": {
            "r2": float(r2_original),
            "mae": float(mae_original),
            "pearson_r": float(corr_original),
        },
        "ablated": {
            "r2": float(r2_ablated),
            "mae": float(mae_ablated),
            "pearson_r": float(corr_ablated),
        },
        "accuracy_drop": float(r2_original - r2_ablated),
        "relative_drop": float((r2_original - r2_ablated) / r2_original * 100),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT T4.5: COMPARE ALTERNATIVE DECODING VECTORS
# ═══════════════════════════════════════════════════════════════════════════════


def compare_decoding_vectors(model, tokens: torch.Tensor, batch_size: int = 256):
    """
    T4.5: Compare different decoding vector constructions.
    """
    results = {}

    methods = [
        ("full_with_ln", lambda: compute_decoding_vector(model, apply_ln=True)),
        ("full_no_ln", lambda: compute_decoding_vector(model, apply_ln=False)),
        (
            "sum_embeddings",
            lambda: compute_decoding_vector_alternative(model, "sum_embeddings"),
        ),
        (
            "sum_ln_embeddings",
            lambda: compute_decoding_vector_alternative(model, "sum_ln_embeddings"),
        ),
    ]

    print("\n[T4.5] Comparing Decoding Vector Constructions")
    print("-" * 60)

    for name, compute_fn in methods:
        w = compute_fn()

        # Test at multiple hook points
        for hook_name in [
            "blocks.0.hook_attn_out",  # Changed from blocks.0.attn.hook_result
            "blocks.0.hook_resid_mid",
            "blocks.0.ln2.hook_normalized",
        ]:
            hook_short = hook_name.split(".")[-1]
            corr_result = test_decoding_vector_correlation(
                model, tokens, w, hook_name, batch_size
            )

            key = f"{name}_{hook_short}"
            results[key] = {"method": name, "hook": hook_name, **corr_result}

            print(f"  {name} @ {hook_short}: r = {corr_result['correlation']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT T4.6: VOCABULARY SIZE SCALING
# ═══════════════════════════════════════════════════════════════════════════════


def vocab_size_scaling_experiment(
    cfg: ExperimentConfig, vocab_sizes: list = [100, 500, 1000, 5000, 10000, 50000]
):
    """
    T4.6: How does decoding accuracy depend on vocabulary size?
    """
    results = []

    print("\n[T4.6] Vocabulary Size Scaling")
    print("-" * 60)

    for vocab_size in vocab_sizes:
        print(f"\n  Testing vocab_size = {vocab_size}")

        # Create model with this vocab size
        model = create_model(cfg, vocab_size=vocab_size)
        tokens = generate_random_tokens(
            cfg.n_samples // 10, cfg.seq_len, vocab_size, cfg.seed
        )

        # Compute decoding vector and test
        w = compute_decoding_vector(model, apply_ln=True)
        corr_result = test_decoding_vector_correlation(
            model, tokens, w, "blocks.0.hook_resid_mid", cfg.batch_size
        )

        # Also verify orthogonality
        _, ortho_stats = verify_embedding_orthogonality(model, n_pairs=5000)

        results.append(
            {
                "vocab_size": vocab_size,
                "correlation": corr_result["correlation"],
                "per_sample_corr_mean": corr_result["per_sample_corr_mean"],
                "mean_abs_cosine": ortho_stats["mean_abs_cosine"],
            }
        )

        print(f"    Correlation: r = {corr_result['correlation']:.4f}")
        print(f"    Orthogonality: |cos| = {ortho_stats['mean_abs_cosine']:.6f}")

        del model, tokens
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT T4.7: CONTEXT LENGTH SCALING (LENGTH EXTRAPOLATION)
# ═══════════════════════════════════════════════════════════════════════════════


def length_extrapolation_experiment(
    cfg: ExperimentConfig, test_lengths: list = [32, 64, 128]
):
    """
    T4.7: Does the decoding vector work for sequences longer than training?

    Note: Reduced max length from 256 to 128 to avoid OOM on smaller GPUs.
    Uses smaller batch size for longer sequences.
    """
    results = []

    print("\n[T4.7] Length Extrapolation")
    print("-" * 60)

    # Create base model with max length
    max_len = max(test_lengths)
    model_cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        d_vocab=cfg.vocab_size,
        n_ctx=max_len,
        act_fn="gelu",
        normalization_type="LN",
        device=device,
    )
    model = HookedTransformer(model_cfg)
    model.pos_embed.W_pos.data.zero_()
    model.pos_embed.W_pos.requires_grad = False

    # Compute decoding vector (this is fixed)
    w = compute_decoding_vector(model, apply_ln=True)

    for seq_len in test_lengths:
        print(f"\n  Testing seq_len = {seq_len}")

        # Reduce samples and batch size for longer sequences to avoid OOM
        n_test_samples = max(500, cfg.n_samples // (10 * (seq_len // 32)))
        batch_size = max(32, cfg.batch_size // (seq_len // 32))

        tokens = generate_random_tokens(
            n_test_samples, seq_len, cfg.vocab_size, cfg.seed
        )

        # Clear cache before running
        torch.cuda.empty_cache()

        corr_result = test_decoding_vector_correlation(
            model, tokens, w, "blocks.0.hook_resid_mid", batch_size
        )

        results.append(
            {
                "seq_len": seq_len,
                "correlation": corr_result["correlation"],
                "per_sample_corr_mean": corr_result["per_sample_corr_mean"],
            }
        )

        print(f"    Correlation: r = {corr_result['correlation']:.4f}")

        del tokens
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def plot_orthogonality_histogram(cos_sims: np.ndarray, stats: dict, save_path: str):
    """Plot histogram of cosine similarities."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=cos_sims, nbinsx=100, name="Cosine Similarities"))

    # Add vertical lines for expected value
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Orthogonal")
    fig.add_vline(
        x=stats["expected_for_random"],
        line_dash="dot",
        line_color="green",
        annotation_text=f"Expected: {stats['expected_for_random']:.4f}",
    )
    fig.add_vline(x=-stats["expected_for_random"], line_dash="dot", line_color="green")

    fig.update_layout(
        title=dict(
            text="Embedding Orthogonality: Pairwise Cosine Similarities",
            font=dict(size=20, family="Serif"),
        ),
        xaxis=dict(title="cos(E_i, E_j)", title_font=dict(size=16)),
        yaxis=dict(title="Count", title_font=dict(size=16)),
        width=800,
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")


def plot_decoding_correlation(projections_mean: np.ndarray, save_path: str, title: str):
    """Plot decoding vector projection vs position."""
    positions = np.arange(len(projections_mean))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=positions,
            y=projections_mean,
            mode="markers",
            marker=dict(size=8, color="blue", opacity=0.7),
            name="Mean Projection",
        )
    )

    # Fit line
    slope, intercept = np.polyfit(positions, projections_mean, 1)
    fit_line = slope * positions + intercept
    corr, _ = pearsonr(positions, projections_mean)

    fig.add_trace(
        go.Scatter(
            x=positions,
            y=fit_line,
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Fit (r={corr:.4f})",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Serif")),
        xaxis=dict(title="Position", title_font=dict(size=16)),
        yaxis=dict(title="Projection onto Decoding Vector", title_font=dict(size=16)),
        width=800,
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")


def plot_ablation_comparison(ablation_results: dict, save_path: str):
    """Plot comparison of original vs ablated probe accuracy."""
    fig = go.Figure()

    metrics = ["r2", "pearson_r"]
    x_labels = ["R² Score", "Pearson r"]

    fig.add_trace(
        go.Bar(
            name="Original",
            x=x_labels,
            y=[
                ablation_results["original"]["r2"],
                ablation_results["original"]["pearson_r"],
            ],
            marker_color="blue",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Ablated (w removed)",
            x=x_labels,
            y=[
                ablation_results["ablated"]["r2"],
                ablation_results["ablated"]["pearson_r"],
            ],
            marker_color="red",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Decoding Vector Ablation (Drop: {ablation_results['relative_drop']:.1f}%)",
            font=dict(size=20, family="Serif"),
        ),
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title="Score", title_font=dict(size=16)),
        barmode="group",
        width=600,
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=600, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")


def plot_scaling_results(vocab_results: list, length_results: list, save_path: str):
    """Plot vocabulary and length scaling results."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("A) Vocabulary Size Scaling", "B) Sequence Length Scaling"),
        horizontal_spacing=0.12,
    )

    # Vocab scaling
    vocab_sizes = [r["vocab_size"] for r in vocab_results]
    vocab_corrs = [r["correlation"] for r in vocab_results]

    fig.add_trace(
        go.Scatter(
            x=vocab_sizes,
            y=vocab_corrs,
            mode="lines+markers",
            marker=dict(size=10),
            line=dict(width=2, color="blue"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Length scaling
    lengths = [r["seq_len"] for r in length_results]
    length_corrs = [r["correlation"] for r in length_results]

    fig.add_trace(
        go.Scatter(
            x=lengths,
            y=length_corrs,
            mode="lines+markers",
            marker=dict(size=10),
            line=dict(width=2, color="green"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Vocabulary Size", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Sequence Length", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)

    fig.update_layout(
        title=dict(
            text="Decoding Vector Scaling Analysis", font=dict(size=22, family="Serif")
        ),
        width=1000,
        height=450,
        template="plotly_white",
        margin=dict(l=60, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=1000, height=450, scale=2)
    fig.write_image(f"{save_path}.pdf")


def create_summary_figure(ortho_stats, corr_results, ablation_results, save_path: str):
    """Create comprehensive summary figure for paper."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "A) Embedding Orthogonality",
            "B) Decoding Vector Projection",
            "C) Ablation Analysis",
            "D) Method Comparison",
        ),
        specs=[[{}, {}], [{}, {}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # A) Orthogonality - show as text annotation
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="x domain",
        yref="y domain",
        text=f"Mean |cos(E_i, E_j)|: {ortho_stats['mean_abs_cosine']:.4f}<br>"
        + f"Expected (random): {ortho_stats['expected_for_random']:.4f}",
        showarrow=False,
        font=dict(size=14),
        row=1,
        col=1,
    )

    # B) Projection correlation
    positions = np.arange(len(corr_results["projections_mean"]))
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=corr_results["projections_mean"],
            mode="markers",
            marker=dict(size=4, color="blue"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # C) Ablation bars
    fig.add_trace(
        go.Bar(
            x=["Original", "Ablated"],
            y=[ablation_results["original"]["r2"], ablation_results["ablated"]["r2"]],
            marker_color=["blue", "red"],
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Position", row=1, col=2)
    fig.update_yaxes(title_text="Projection", row=1, col=2)
    fig.update_yaxes(title_text="R² Score", row=2, col=1)

    fig.update_layout(
        title=dict(
            text="Decoding Vector Analysis Summary", font=dict(size=22, family="Serif")
        ),
        width=1000,
        height=800,
        template="plotly_white",
        margin=dict(l=60, r=50, t=80, b=60),
    )

    fig.write_image(f"{save_path}.png", width=1000, height=800, scale=2)
    fig.write_image(f"{save_path}.pdf")


def save_results(results: dict, save_path: str):
    """Save results to JSON."""
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            serializable[key] = [
                (
                    {
                        kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in item.items()
                    }
                    if isinstance(item, dict)
                    else item
                )
                for item in value
            ]
        else:
            serializable[key] = value

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Decoding Vector Experiments")
    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    setup_dirs()

    cfg = ExperimentConfig(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        d_model=args.d_model,
        batch_size=args.batch_size,
    )

    print(f"\n{'=' * 60}")
    print("DECODING VECTOR EXPERIMENTS")
    print(f"{'=' * 60}")
    print(f"Samples: {cfg.n_samples}")
    print(f"Sequence length: {cfg.seq_len}")
    print(f"Model dimension: {cfg.d_model}")
    print(f"{'=' * 60}\n")

    all_results = {}

    # Create model and generate tokens
    model = create_model(cfg)
    tokens = generate_random_tokens(
        cfg.n_samples, cfg.seq_len, cfg.vocab_size, cfg.seed
    )

    # T4.1: Orthogonality verification
    cos_sims, ortho_stats = verify_embedding_orthogonality(model)
    all_results["orthogonality"] = ortho_stats

    # T4.2: Compute decoding vector
    print("\n[T4.2] Computing decoding vector...")
    w = compute_decoding_vector(model, apply_ln=True)
    print(f"  Decoding vector norm: {np.linalg.norm(w):.4f}")

    # T4.3: Test at different hook points
    print("\n[T4.3] Testing decoding vector at different layers...")
    for hook_name, hook_label in [
        (
            "blocks.0.hook_attn_out",
            "post_attn",
        ),  # Changed from blocks.0.attn.hook_result
        ("blocks.0.hook_resid_mid", "post_attn_residual"),
        ("blocks.0.ln2.hook_normalized", "post_ln2"),
    ]:
        print(f"\n  Testing at {hook_label}:")
        corr_result = test_decoding_vector_correlation(
            model, tokens, w, hook_name, cfg.batch_size
        )
        all_results[f"correlation_{hook_label}"] = corr_result
        print(f"    Pearson r: {corr_result['correlation']:.4f}")
        print(f"    Per-sample mean r: {corr_result['per_sample_corr_mean']:.4f}")

    # T4.4: Ablation
    print("\n[T4.4] Decoding vector ablation...")
    ablation_results = ablate_decoding_vector(
        model, tokens, w, "blocks.0.hook_resid_mid", cfg.batch_size
    )
    all_results["ablation"] = ablation_results
    print(f"  Original R²: {ablation_results['original']['r2']:.4f}")
    print(f"  Ablated R²: {ablation_results['ablated']['r2']:.4f}")
    print(f"  Accuracy drop: {ablation_results['relative_drop']:.1f}%")

    # T4.5: Compare alternatives
    compare_results = compare_decoding_vectors(model, tokens[:5000], cfg.batch_size)
    all_results["method_comparison"] = compare_results

    del model
    torch.cuda.empty_cache()

    # T4.6: Vocabulary scaling
    vocab_results = vocab_size_scaling_experiment(
        cfg, vocab_sizes=[100, 500, 1000, 5000, 10000, 30000]
    )
    all_results["vocab_scaling"] = vocab_results

    # T4.7: Length extrapolation
    length_results = length_extrapolation_experiment(
        cfg, test_lengths=[32, 64, 128, 256]
    )
    all_results["length_scaling"] = length_results

    # ─── Generate Plots ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_orthogonality_histogram(
        cos_sims, ortho_stats, str(RESULTS_DIR / "orthogonality_histogram")
    )

    plot_decoding_correlation(
        all_results["correlation_post_attn_residual"]["projections_mean"],
        str(RESULTS_DIR / "decoding_correlation"),
        "Decoding Vector: Projection vs Position",
    )

    plot_ablation_comparison(ablation_results, str(RESULTS_DIR / "ablation_comparison"))

    plot_scaling_results(
        vocab_results, length_results, str(PLOTS_DIR / "decoding_vector_scaling")
    )

    # Summary figure
    create_summary_figure(
        ortho_stats,
        all_results["correlation_post_attn_residual"],
        ablation_results,
        str(PLOTS_DIR / "decoding_vector_summary"),
    )

    # ─── Save Results ───────────────────────────────────────────────────────────
    save_results(all_results, str(RESULTS_DIR / "decoding_vector_results.json"))

    print(f"\n{'=' * 60}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Paper figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
