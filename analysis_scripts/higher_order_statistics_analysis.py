"""
Higher-Order Statistics Analysis After LayerNorm

This script investigates HOW positional information survives LayerNorm normalization.
If LN normalizes each sample to zero mean and unit variance, where is position encoded?

Statistics Tested:
- S1: Population Mean - E[h_i] varies by position
- S2: Kurtosis - Fourth moment may vary with position
- S3: Skewness - Third moment may encode position
- S4: Covariance Eigenspectrum - Top eigenvalues may vary systematically
- S5: Specific Directions - Position may be encoded in learned directions
- S6: Pairwise Correlations - Correlation structure may vary by position
- S7: L_p Norms - L1, L∞ norms may differ despite L2 normalization
- S8: Neuron-Specific Activations - Individual neurons may encode position

Usage:
    python higher_order_statistics_analysis.py --n_samples 30000 --seq_len 64
"""

import os

# Only set GPU if not already specified by environment
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np
import torch
from scipy import stats
from scipy.stats import kurtosis, skew, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/higher_order_statistics")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """Configuration for higher-order statistics experiments."""

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


def create_model(cfg: ExperimentConfig, norm_type: str = "LN"):
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

    # Zero out positional embeddings
    model.pos_embed.W_pos.data.zero_()
    model.pos_embed.W_pos.requires_grad = False

    return model


def generate_random_tokens(cfg: ExperimentConfig):
    """Generate random token sequences."""
    torch.manual_seed(cfg.seed)
    return torch.randint(0, cfg.vocab_size, (cfg.n_samples, cfg.seq_len), device=device)


def extract_activations(
    model, tokens: torch.Tensor, hook_name: str, batch_size: int = 256
):
    """Extract activations at a specific hook point."""
    model.eval()
    n_samples = tokens.shape[0]
    activations = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Extracting {hook_name}"):
            batch_tokens = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(batch_tokens, names_filter=[hook_name])
            activations.append(cache[hook_name].detach().cpu())
            del cache
            torch.cuda.empty_cache()

    return torch.cat(activations, dim=0).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTIC COMPUTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_population_mean(activations: np.ndarray):
    """
    S1: Compute population mean at each position.

    Args:
        activations: [n_samples, seq_len, d_model]

    Returns:
        pop_mean: [seq_len, d_model] population mean per position
        mean_norm: [seq_len] L2 norm of population mean
    """
    pop_mean = activations.mean(axis=0)  # [seq_len, d_model]
    mean_norm = np.linalg.norm(pop_mean, axis=1)  # [seq_len]
    return pop_mean, mean_norm


def compute_kurtosis(activations: np.ndarray):
    """
    S2: Compute excess kurtosis at each position.

    Kurtosis measures "tailedness" - how much of variance is due to extreme values.
    For normal distribution, excess kurtosis = 0.

    Returns:
        kurtosis_per_dim: [seq_len, d_model] kurtosis for each dimension at each position
        kurtosis_mean: [seq_len] mean kurtosis across dimensions
    """
    n_samples, seq_len, d_model = activations.shape

    kurtosis_per_dim = np.zeros((seq_len, d_model))
    for pos in tqdm(range(seq_len), desc="Computing kurtosis"):
        for dim in range(d_model):
            kurtosis_per_dim[pos, dim] = kurtosis(activations[:, pos, dim], fisher=True)

    kurtosis_mean = kurtosis_per_dim.mean(axis=1)
    return kurtosis_per_dim, kurtosis_mean


def compute_skewness(activations: np.ndarray):
    """
    S3: Compute skewness at each position.

    Skewness measures asymmetry of the distribution.

    Returns:
        skewness_per_dim: [seq_len, d_model]
        skewness_mean: [seq_len]
    """
    n_samples, seq_len, d_model = activations.shape

    skewness_per_dim = np.zeros((seq_len, d_model))
    for pos in tqdm(range(seq_len), desc="Computing skewness"):
        for dim in range(d_model):
            skewness_per_dim[pos, dim] = skew(activations[:, pos, dim])

    skewness_mean = skewness_per_dim.mean(axis=1)
    return skewness_per_dim, skewness_mean


def compute_covariance_eigenspectrum(activations: np.ndarray, n_components: int = 50):
    """
    S4: Compute covariance eigenspectrum at each position.

    Returns:
        eigenvalues: [seq_len, n_components] top eigenvalues per position
        explained_variance: [seq_len, n_components] explained variance ratio
    """
    n_samples, seq_len, d_model = activations.shape

    eigenvalues = np.zeros((seq_len, n_components))
    explained_variance = np.zeros((seq_len, n_components))

    for pos in tqdm(range(seq_len), desc="Computing eigenspectrum"):
        pos_acts = activations[:, pos, :]  # [n_samples, d_model]

        pca = PCA(n_components=n_components)
        pca.fit(pos_acts)

        eigenvalues[pos] = pca.explained_variance_
        explained_variance[pos] = pca.explained_variance_ratio_

    return eigenvalues, explained_variance


def find_position_encoding_direction(activations: np.ndarray, positions: np.ndarray):
    """
    S5: Find the optimal direction that encodes position.

    Uses linear regression to find w* = argmax_w corr(w·h, position)

    Returns:
        optimal_direction: [d_model] normalized direction
        correlation: float correlation achieved
        projections: [seq_len] mean projection onto direction
    """
    n_samples, seq_len, d_model = activations.shape

    # Flatten for regression
    X = activations.reshape(-1, d_model)  # [n_samples * seq_len, d_model]
    y = np.tile(positions, n_samples)  # [n_samples * seq_len]

    # Fit Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    # Extract optimal direction (normalized)
    optimal_direction = ridge.coef_ / np.linalg.norm(ridge.coef_)

    # Compute correlation
    y_pred = X @ optimal_direction
    correlation, _ = pearsonr(y_pred, y)

    # Mean projection at each position
    projections = np.array(
        [(activations[:, pos, :] @ optimal_direction).mean() for pos in range(seq_len)]
    )

    return optimal_direction, correlation, projections


def compute_pairwise_dimension_correlation(activations: np.ndarray, n_dims: int = 100):
    """
    S6: Compute correlation between dimension pairs at each position.

    Returns:
        corr_matrices: [seq_len, n_dims, n_dims] correlation matrices
        corr_frobenius: [seq_len] Frobenius norm of correlation matrix
    """
    n_samples, seq_len, d_model = activations.shape
    n_dims = min(n_dims, d_model)

    corr_frobenius = np.zeros(seq_len)

    # Just compute summary statistic to save memory
    for pos in tqdm(range(seq_len), desc="Computing pairwise correlations"):
        pos_acts = activations[:, pos, :n_dims]
        corr_matrix = np.corrcoef(pos_acts.T)
        # Frobenius norm (excluding diagonal)
        np.fill_diagonal(corr_matrix, 0)
        corr_frobenius[pos] = np.linalg.norm(corr_matrix, "fro")

    return corr_frobenius


def compute_lp_norms(activations: np.ndarray):
    """
    S7: Compute various Lp norms at each position.

    Even though L2 norm is normalized by LayerNorm, other norms may vary.

    Returns:
        norms: dict mapping p -> [seq_len] mean norm at each position
    """
    n_samples, seq_len, d_model = activations.shape

    norms = {}
    for p_name, p in [("L1", 1), ("L2", 2), ("Linf", np.inf)]:
        mean_norms = np.zeros(seq_len)
        for pos in range(seq_len):
            sample_norms = np.linalg.norm(activations[:, pos, :], ord=p, axis=1)
            mean_norms[pos] = sample_norms.mean()
        norms[p_name] = mean_norms

    return norms


def compute_neuron_position_correlation(activations: np.ndarray, positions: np.ndarray):
    """
    S8: Compute correlation of each neuron's activation with position.

    Returns:
        correlations: [d_model] correlation of each neuron with position
        top_neurons: indices of neurons most correlated with position
    """
    n_samples, seq_len, d_model = activations.shape

    correlations = np.zeros(d_model)
    y = np.tile(positions, n_samples)

    for dim in tqdm(range(d_model), desc="Computing neuron correlations"):
        x = activations[:, :, dim].flatten()
        correlations[dim], _ = pearsonr(x, y)

    top_neurons = np.argsort(np.abs(correlations))[::-1]

    return correlations, top_neurons


def probe_with_statistics(
    activations: np.ndarray,
    positions: np.ndarray,
    kurtosis_per_dim: np.ndarray,
    skewness_per_dim: np.ndarray,
    eigenvalues: np.ndarray,
):
    """
    Test which statistics allow position prediction.

    Trains probes on different feature combinations.
    """
    n_samples, seq_len, d_model = activations.shape

    results = {}

    # Prepare features
    features = {
        "raw_activations": activations.reshape(-1, d_model),
        "kurtosis": np.tile(kurtosis_per_dim, (n_samples, 1, 1)).reshape(-1, d_model),
        "skewness": np.tile(skewness_per_dim, (n_samples, 1, 1)).reshape(-1, d_model),
        "eigenvalues": np.tile(eigenvalues, (n_samples, 1, 1)).reshape(
            -1, eigenvalues.shape[1]
        ),
    }

    y = np.tile(positions, n_samples)
    train_size = int(0.8 * len(y))

    for name, X in features.items():
        # Train/test split
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train probe
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        # Evaluate
        y_pred = ridge.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        corr, _ = pearsonr(y_test, y_pred)

        results[name] = {"r2": float(r2), "pearson_r": float(corr)}
        print(f"  {name}: R² = {r2:.4f}, r = {corr:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def plot_statistics_by_position(data_dict: dict, save_path: str, title: str):
    """Plot multiple statistics vs position."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, (name, values) in enumerate(data_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="lines",
                name=name,
                line=dict(width=2, color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Serif")),
        xaxis=dict(title="Position", title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(title="Value", title_font=dict(size=16), tickfont=dict(size=14)),
        width=800,
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_eigenvalue_heatmap(eigenvalues: np.ndarray, save_path: str):
    """Plot eigenvalue spectrum heatmap by position."""
    fig = px.imshow(
        eigenvalues.T,
        labels=dict(x="Position", y="Eigenvalue Index", color="Eigenvalue"),
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    fig.update_layout(
        title=dict(
            text="Covariance Eigenspectrum by Position",
            font=dict(size=20, family="Serif"),
        ),
        width=800,
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_neuron_correlations(correlations: np.ndarray, top_k: int, save_path: str):
    """Plot histogram of neuron-position correlations."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Correlation Distribution", f"Top {top_k} Neurons"),
        horizontal_spacing=0.15,
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=correlations, nbinsx=50, name="All neurons"), row=1, col=1
    )

    # Top neurons bar chart
    top_indices = np.argsort(np.abs(correlations))[::-1][:top_k]
    fig.add_trace(
        go.Bar(x=list(range(top_k)), y=correlations[top_indices], name="Top neurons"),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Correlation", row=1, col=1)
    fig.update_xaxes(title_text="Neuron Rank", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Correlation with Position", row=1, col=2)

    fig.update_layout(
        title=dict(
            text="Neuron-Position Correlations", font=dict(size=20, family="Serif")
        ),
        width=1000,
        height=400,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=1000, height=400, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def create_summary_figure(
    kurtosis_mean,
    skewness_mean,
    mean_norm,
    lp_norms,
    direction_projections,
    save_path: str,
):
    """Create comprehensive summary figure for paper."""
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "A) Population Mean Norm",
            "B) Kurtosis by Position",
            "C) Skewness by Position",
            "D) Lp Norms",
            "E) Optimal Direction Projection",
            "F) Position Correlation",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    positions = np.arange(len(kurtosis_mean))

    # A) Mean norm
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=mean_norm,
            mode="lines",
            line=dict(width=2, color="blue"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # B) Kurtosis
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=kurtosis_mean,
            mode="lines",
            line=dict(width=2, color="red"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # C) Skewness
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=skewness_mean,
            mode="lines",
            line=dict(width=2, color="green"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # D) Lp norms
    colors = ["purple", "orange", "cyan"]
    for i, (name, values) in enumerate(lp_norms.items()):
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=values,
                mode="lines",
                name=name,
                line=dict(width=2, color=colors[i]),
            ),
            row=2,
            col=1,
        )

    # E) Optimal direction projection
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=direction_projections,
            mode="lines",
            line=dict(width=2, color="magenta"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # F) Position vs projection scatter
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=direction_projections,
            mode="markers",
            marker=dict(size=6, color="magenta"),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # Fit line
    slope, intercept = np.polyfit(positions, direction_projections, 1)
    fit_line = slope * positions + intercept
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=fit_line,
            mode="lines",
            line=dict(width=2, color="black", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # Update axes
    for col in [1, 2, 3]:
        fig.update_xaxes(title_text="Position", row=2, col=col)

    fig.update_layout(
        title=dict(
            text="Higher-Order Statistics Analysis (Post-LayerNorm Activations)",
            font=dict(size=22, family="Serif"),
        ),
        width=1400,
        height=800,
        template="plotly_white",
        legend=dict(x=0.02, y=0.4),
        margin=dict(l=60, r=50, t=100, b=60),
    )

    fig.write_image(f"{save_path}.png", width=1400, height=800, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def save_results(results: dict, save_path: str):
    """Save results to JSON."""
    # Convert numpy arrays to lists
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable[key] = value

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Higher-Order Statistics Analysis")
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
    print("HIGHER-ORDER STATISTICS ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Samples: {cfg.n_samples}")
    print(f"Sequence length: {cfg.seq_len}")
    print(f"Model dimension: {cfg.d_model}")
    print(f"{'=' * 60}\n")

    # Generate tokens and create model
    tokens = generate_random_tokens(cfg)
    model = create_model(cfg, norm_type="LN")

    # Extract post-LN activations (where the paradox occurs)
    hook_name = "blocks.0.ln2.hook_normalized"
    print(f"\nExtracting activations from: {hook_name}")
    activations = extract_activations(model, tokens, hook_name, cfg.batch_size)

    positions = np.arange(cfg.seq_len)

    # ─── Compute All Statistics ─────────────────────────────────────────────────
    results = {}

    print("\n[S1] Computing population mean...")
    pop_mean, mean_norm = compute_population_mean(activations)
    results["mean_norm"] = mean_norm
    corr_mean, _ = pearsonr(positions, mean_norm)
    print(f"  Population mean norm correlation with position: r = {corr_mean:.4f}")

    print("\n[S2] Computing kurtosis...")
    kurtosis_per_dim, kurtosis_mean = compute_kurtosis(activations)
    results["kurtosis_mean"] = kurtosis_mean
    corr_kurt, _ = pearsonr(positions, kurtosis_mean)
    print(f"  Mean kurtosis correlation with position: r = {corr_kurt:.4f}")

    print("\n[S3] Computing skewness...")
    skewness_per_dim, skewness_mean = compute_skewness(activations)
    results["skewness_mean"] = skewness_mean
    corr_skew, _ = pearsonr(positions, skewness_mean)
    print(f"  Mean skewness correlation with position: r = {corr_skew:.4f}")

    print("\n[S4] Computing eigenspectrum...")
    eigenvalues, explained_variance = compute_covariance_eigenspectrum(activations)
    results["eigenvalues"] = eigenvalues
    results["explained_variance"] = explained_variance
    # Top eigenvalue correlation
    corr_eig1, _ = pearsonr(positions, eigenvalues[:, 0])
    print(f"  Top eigenvalue correlation with position: r = {corr_eig1:.4f}")

    print("\n[S5] Finding position-encoding direction...")
    optimal_direction, direction_corr, direction_projections = (
        find_position_encoding_direction(activations, positions)
    )
    results["direction_correlation"] = direction_corr
    results["direction_projections"] = direction_projections
    print(f"  Optimal direction correlation: r = {direction_corr:.4f}")

    print("\n[S6] Computing pairwise correlations...")
    corr_frobenius = compute_pairwise_dimension_correlation(activations)
    results["corr_frobenius"] = corr_frobenius
    corr_frob_pos, _ = pearsonr(positions, corr_frobenius)
    print(f"  Correlation matrix Frobenius norm vs position: r = {corr_frob_pos:.4f}")

    print("\n[S7] Computing Lp norms...")
    lp_norms = compute_lp_norms(activations)
    results["lp_norms"] = lp_norms
    for name, values in lp_norms.items():
        corr, _ = pearsonr(positions, values)
        print(f"  {name} norm correlation with position: r = {corr:.4f}")

    print("\n[S8] Computing neuron correlations...")
    neuron_correlations, top_neurons = compute_neuron_position_correlation(
        activations, positions
    )
    results["neuron_correlations"] = neuron_correlations
    results["top_neurons"] = top_neurons[:100].tolist()
    print(f"  Max neuron correlation: r = {np.abs(neuron_correlations).max():.4f}")
    print(f"  Top 10 neuron correlations: {neuron_correlations[top_neurons[:10]]}")

    # ─── Probe with Different Statistics ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROBING WITH DIFFERENT STATISTICS")
    print("=" * 60)

    probe_results = probe_with_statistics(
        activations, positions, kurtosis_per_dim, skewness_per_dim, eigenvalues
    )
    results["probe_results"] = probe_results

    # ─── Summary Statistics ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: Correlation with Position")
    print("=" * 60)

    summary = {
        "Population Mean Norm": corr_mean,
        "Mean Kurtosis": corr_kurt,
        "Mean Skewness": corr_skew,
        "Top Eigenvalue": corr_eig1,
        "Optimal Direction": direction_corr,
        "Correlation Frobenius": corr_frob_pos,
        "L1 Norm": pearsonr(positions, lp_norms["L1"])[0],
        "L2 Norm": pearsonr(positions, lp_norms["L2"])[0],
        "Linf Norm": pearsonr(positions, lp_norms["Linf"])[0],
    }

    for name, corr in sorted(summary.items(), key=lambda x: -abs(x[1])):
        print(f"  {name:<25}: r = {corr:+.4f}")

    results["summary_correlations"] = summary

    # ─── Generate Plots ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Individual plots
    plot_statistics_by_position(
        {"Kurtosis": kurtosis_mean, "Skewness": skewness_mean},
        str(RESULTS_DIR / "kurtosis_skewness_by_position"),
        "Higher-Order Moments by Position",
    )

    plot_eigenvalue_heatmap(eigenvalues, str(RESULTS_DIR / "eigenvalue_heatmap"))

    plot_neuron_correlations(
        neuron_correlations, 50, str(RESULTS_DIR / "neuron_position_correlations")
    )

    # Summary figure for paper
    create_summary_figure(
        kurtosis_mean,
        skewness_mean,
        mean_norm,
        lp_norms,
        direction_projections,
        str(PLOTS_DIR / "higher_order_statistics_summary"),
    )

    # ─── Save Results ───────────────────────────────────────────────────────────
    save_results(results, str(RESULTS_DIR / "higher_order_statistics_results.json"))

    # Clean up
    del model, activations
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Paper figure saved to: {PLOTS_DIR}/higher_order_statistics_summary.png")


if __name__ == "__main__":
    main()
