"""
Neuron Subgroup Position Encoding Analysis

Hypothesis: After LayerNorm/RMSNorm, position information (originally in variance)
gets "spread" to a sparse combination of neurons. This effect might be stronger
in RMSNorm because:
- LayerNorm: (x - mean) / std → numerator has mean 0
- RMSNorm: x / rms → numerator preserves the original mean structure

Key Questions:
1. Are there specific neurons that consistently correlate with position?
2. Do small subgroups of neurons jointly encode position better than expected?
3. Is this effect stronger in RMSNorm than LayerNorm?
4. Does the "position-encoding neuron set" emerge from variance redistribution?

Methods:
1. Per-neuron correlation with position
2. Sparse probe (L1-regularized) to find minimal neuron set
3. Neuron co-activation patterns by position
4. Comparison of LN vs RMSNorm neuron selectivity

Usage:
    python neuron_subgroup_analysis.py --n_samples 5000 --seq_len 64
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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig


# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/neuron_subgroup_analysis")
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


def setup_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(cfg: Config, norm_type: str = "LN"):
    """Create NoPE model with specified normalization."""
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


def extract_activations(
    model, tokens: torch.Tensor, hook_name: str, batch_size: int = 256
):
    """Extract activations at a specific hook point."""
    model.eval()
    n_samples = tokens.shape[0]
    activations = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Extracting {hook_name}"):
            batch = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(batch, names_filter=[hook_name])
            activations.append(cache[hook_name].cpu())
            del cache
            torch.cuda.empty_cache()

    return torch.cat(activations, dim=0).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Per-Neuron Position Correlation
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_per_neuron_correlation(acts: np.ndarray, cfg: Config):
    """
    Compute correlation between each neuron's activation and position.

    For each neuron d, compute corr(acts[:, :, d].flatten(), positions)
    """
    n_samples, seq_len, d_model = acts.shape
    positions = np.tile(np.arange(seq_len), n_samples)
    acts_flat = acts.reshape(-1, d_model)  # [n_samples * seq_len, d_model]

    correlations = []
    p_values = []

    for d in tqdm(range(d_model), desc="Per-neuron correlation"):
        r, p = pearsonr(acts_flat[:, d], positions)
        correlations.append(r)
        p_values.append(p)

    correlations = np.array(correlations)
    p_values = np.array(p_values)

    # Identify significant neurons (Bonferroni corrected)
    alpha = 0.05 / d_model
    significant_mask = p_values < alpha

    # Sort by absolute correlation
    sorted_idx = np.argsort(np.abs(correlations))[::-1]

    return {
        "correlations": correlations,
        "p_values": p_values,
        "significant_neurons": np.where(significant_mask)[0].tolist(),
        "n_significant": int(np.sum(significant_mask)),
        "top_positive_neurons": sorted_idx[correlations[sorted_idx] > 0][:20].tolist(),
        "top_negative_neurons": sorted_idx[correlations[sorted_idx] < 0][:20].tolist(),
        "mean_abs_corr": float(np.mean(np.abs(correlations))),
        "max_abs_corr": float(np.max(np.abs(correlations))),
        "correlation_histogram": np.histogram(correlations, bins=50)[0].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Per-Sample Per-Neuron Correlation
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_per_sample_neuron_correlation(acts: np.ndarray, cfg: Config):
    """
    For each sample and each neuron, compute correlation with position.
    This reveals neurons that CONSISTENTLY encode position across samples.
    """
    n_samples, seq_len, d_model = acts.shape
    positions = np.arange(seq_len)

    # For each neuron, compute per-sample correlation
    neuron_consistency = []  # Fraction of samples where neuron correlates with position
    neuron_mean_corr = []
    neuron_std_corr = []

    for d in tqdm(range(d_model), desc="Per-sample neuron analysis"):
        sample_corrs = []
        for s in range(n_samples):
            r, _ = pearsonr(acts[s, :, d], positions)
            sample_corrs.append(r)

        sample_corrs = np.array(sample_corrs)
        neuron_consistency.append(np.mean(sample_corrs > 0.3))  # Fraction with r > 0.3
        neuron_mean_corr.append(np.mean(sample_corrs))
        neuron_std_corr.append(np.std(sample_corrs))

    neuron_consistency = np.array(neuron_consistency)
    neuron_mean_corr = np.array(neuron_mean_corr)
    neuron_std_corr = np.array(neuron_std_corr)

    # Find neurons that are consistent across samples
    consistent_neurons = np.where(neuron_consistency > 0.5)[0]

    # Signal-to-noise ratio: mean / std
    snr = np.abs(neuron_mean_corr) / (neuron_std_corr + 1e-8)
    high_snr_neurons = np.where(snr > 2)[0]

    return {
        "neuron_consistency": neuron_consistency.tolist(),
        "neuron_mean_corr": neuron_mean_corr.tolist(),
        "neuron_std_corr": neuron_std_corr.tolist(),
        "consistent_neurons": consistent_neurons.tolist(),
        "n_consistent": len(consistent_neurons),
        "high_snr_neurons": high_snr_neurons.tolist(),
        "n_high_snr": len(high_snr_neurons),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Sparse Probe to Find Minimal Neuron Set
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_sparse_probe(acts: np.ndarray, cfg: Config):
    """
    Train L1-regularized (Lasso) probe to find minimal set of neurons
    that encode position.
    """
    n_samples, seq_len, d_model = acts.shape
    positions = np.arange(seq_len)

    # Split data
    n_train = int(n_samples * 0.8)
    X_train = acts[:n_train].reshape(-1, d_model)
    y_train = np.tile(positions, n_train)
    X_test = acts[n_train:].reshape(-1, d_model)
    y_test = np.tile(positions, n_samples - n_train)

    results = {}

    # Test different sparsity levels
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    for alpha in tqdm(alphas, desc="Sparse probe (Lasso)"):
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_train, y_train)

        y_pred = lasso.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Count non-zero weights
        n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-6)
        nonzero_idx = np.where(np.abs(lasso.coef_) > 1e-6)[0]

        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "r2": float(r2),
            "mae": float(mae),
            "n_nonzero_neurons": int(n_nonzero),
            "nonzero_neurons": nonzero_idx.tolist(),
            "weight_magnitudes": np.abs(lasso.coef_[nonzero_idx]).tolist()
            if len(nonzero_idx) > 0
            else [],
        }

    # Also try ElasticNet (L1 + L2)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000)
    elastic.fit(X_train, y_train)
    y_pred = elastic.predict(X_test)

    n_nonzero = np.sum(np.abs(elastic.coef_) > 1e-6)
    results["elastic_net"] = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_nonzero_neurons": int(n_nonzero),
    }

    # Find optimal: best R² with fewest neurons
    best_tradeoff = None
    best_score = -np.inf
    for key, res in results.items():
        if "alpha" in key:
            # Score: R² - 0.001 * n_neurons (penalize complexity)
            score = res["r2"] - 0.001 * res["n_nonzero_neurons"]
            if score > best_score:
                best_score = score
                best_tradeoff = key

    results["best_tradeoff"] = best_tradeoff

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Neuron Subgroup Joint Encoding
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_neuron_subgroups(acts: np.ndarray, cfg: Config, top_k: int = 50):
    """
    Test if small subgroups of neurons jointly encode position better
    than individual neurons.

    Approach: Take top-k correlated neurons, test all pairs and triplets.
    """
    n_samples, seq_len, d_model = acts.shape
    positions = np.tile(np.arange(seq_len), n_samples)
    acts_flat = acts.reshape(-1, d_model)

    # First, find top-k neurons by absolute correlation
    correlations = []
    for d in range(d_model):
        r, _ = pearsonr(acts_flat[:, d], positions)
        correlations.append(r)
    correlations = np.array(correlations)

    top_neurons = np.argsort(np.abs(correlations))[-top_k:]

    # Test individual neurons
    individual_r2 = {}
    for n in top_neurons:
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(acts_flat[:, n : n + 1], positions)
        y_pred = lr.predict(acts_flat[:, n : n + 1])
        individual_r2[int(n)] = float(r2_score(positions, y_pred))

    # Test pairs (sample 100 random pairs from top neurons)
    np.random.seed(cfg.seed)
    n_pairs = min(100, len(top_neurons) * (len(top_neurons) - 1) // 2)

    pair_results = []
    tested_pairs = set()

    while len(pair_results) < n_pairs:
        i, j = np.random.choice(top_neurons, 2, replace=False)
        pair = tuple(sorted([i, j]))
        if pair in tested_pairs:
            continue
        tested_pairs.add(pair)

        X_pair = acts_flat[:, [i, j]]
        lr = LinearRegression()
        lr.fit(X_pair, positions)
        y_pred = lr.predict(X_pair)
        r2 = r2_score(positions, y_pred)

        # Compare to sum of individual R²
        individual_sum = individual_r2.get(int(i), 0) + individual_r2.get(int(j), 0)

        pair_results.append(
            {
                "neurons": [int(i), int(j)],
                "r2": float(r2),
                "individual_sum_r2": float(individual_sum),
                "synergy": float(r2 - individual_sum),  # Positive = superadditive
            }
        )

    # Sort by synergy
    pair_results.sort(key=lambda x: x["synergy"], reverse=True)

    # Test triplets (sample 50)
    triplet_results = []
    n_triplets = 50

    for _ in range(n_triplets):
        idx = np.random.choice(top_neurons, 3, replace=False)
        X_triplet = acts_flat[:, idx]
        lr = LinearRegression()
        lr.fit(X_triplet, positions)
        y_pred = lr.predict(X_triplet)
        r2 = r2_score(positions, y_pred)

        triplet_results.append(
            {
                "neurons": idx.tolist(),
                "r2": float(r2),
            }
        )

    triplet_results.sort(key=lambda x: x["r2"], reverse=True)

    return {
        "top_neurons": top_neurons.tolist(),
        "individual_r2": individual_r2,
        "best_individual_r2": float(max(individual_r2.values())),
        "pair_results": pair_results[:20],  # Top 20 pairs
        "best_pair_r2": float(pair_results[0]["r2"]) if pair_results else 0,
        "best_pair_synergy": float(pair_results[0]["synergy"]) if pair_results else 0,
        "triplet_results": triplet_results[:10],  # Top 10 triplets
        "best_triplet_r2": float(triplet_results[0]["r2"]) if triplet_results else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Variance Redistribution After Normalization
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_variance_redistribution(
    acts_pre_norm: np.ndarray, acts_post_norm: np.ndarray, cfg: Config
):
    """
    Analyze how variance (which encodes position pre-norm) gets redistributed
    across neurons after normalization.

    Key insight: Before LN, variance ~ 1/(i+1). After LN, variance = 1.
    Where did the position information go?
    """
    n_samples, seq_len, d_model = acts_pre_norm.shape
    positions = np.arange(seq_len)

    results = {}

    # Pre-norm: variance by position
    pre_var_by_pos = np.var(acts_pre_norm, axis=2).mean(axis=0)  # [seq_len]
    pre_var_corr, _ = pearsonr(positions, pre_var_by_pos)

    # Post-norm: variance by position (should be ~constant = 1)
    post_var_by_pos = np.var(acts_post_norm, axis=2).mean(axis=0)
    post_var_corr, _ = pearsonr(positions, post_var_by_pos)

    results["pre_norm_var_by_position"] = pre_var_by_pos.tolist()
    results["post_norm_var_by_position"] = post_var_by_pos.tolist()
    results["pre_norm_var_corr_with_position"] = float(pre_var_corr)
    results["post_norm_var_corr_with_position"] = float(post_var_corr)

    # Per-neuron variance change
    pre_neuron_var = np.var(acts_pre_norm.reshape(-1, d_model), axis=0)
    post_neuron_var = np.var(acts_post_norm.reshape(-1, d_model), axis=0)

    var_change = post_neuron_var / (pre_neuron_var + 1e-8)

    # Neurons that gained variance
    gained_var_neurons = np.where(var_change > 1.5)[0]
    lost_var_neurons = np.where(var_change < 0.5)[0]

    results["n_neurons_gained_variance"] = len(gained_var_neurons)
    results["n_neurons_lost_variance"] = len(lost_var_neurons)
    results["variance_change_mean"] = float(np.mean(var_change))
    results["variance_change_std"] = float(np.std(var_change))

    # Do neurons that gained variance correlate more with position?
    acts_post_flat = acts_post_norm.reshape(-1, d_model)
    all_positions = np.tile(positions, n_samples)

    gained_var_corrs = []
    for n in gained_var_neurons[:50]:  # Sample
        r, _ = pearsonr(acts_post_flat[:, n], all_positions)
        gained_var_corrs.append(abs(r))

    lost_var_corrs = []
    for n in lost_var_neurons[:50]:
        r, _ = pearsonr(acts_post_flat[:, n], all_positions)
        lost_var_corrs.append(abs(r))

    results["gained_var_neurons_mean_corr"] = (
        float(np.mean(gained_var_corrs)) if gained_var_corrs else 0
    )
    results["lost_var_neurons_mean_corr"] = (
        float(np.mean(lost_var_corrs)) if lost_var_corrs else 0
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: LN vs RMSNorm Comparison
# ═══════════════════════════════════════════════════════════════════════════════


def compare_ln_vs_rms(results_ln: dict, results_rms: dict):
    """Compare neuron-level position encoding between LN and RMSNorm."""

    comparison = {}

    # Number of significant neurons
    comparison["ln_n_significant"] = results_ln["per_neuron"]["n_significant"]
    comparison["rms_n_significant"] = results_rms["per_neuron"]["n_significant"]

    # Max correlation
    comparison["ln_max_corr"] = results_ln["per_neuron"]["max_abs_corr"]
    comparison["rms_max_corr"] = results_rms["per_neuron"]["max_abs_corr"]

    # Sparse probe efficiency
    for alpha_key in ["alpha_0.1", "alpha_1.0"]:
        if (
            alpha_key in results_ln["sparse_probe"]
            and alpha_key in results_rms["sparse_probe"]
        ):
            comparison[f"ln_{alpha_key}_r2"] = results_ln["sparse_probe"][alpha_key][
                "r2"
            ]
            comparison[f"rms_{alpha_key}_r2"] = results_rms["sparse_probe"][alpha_key][
                "r2"
            ]
            comparison[f"ln_{alpha_key}_n_neurons"] = results_ln["sparse_probe"][
                alpha_key
            ]["n_nonzero_neurons"]
            comparison[f"rms_{alpha_key}_n_neurons"] = results_rms["sparse_probe"][
                alpha_key
            ]["n_nonzero_neurons"]

    # Consistent neurons (across samples)
    comparison["ln_n_consistent"] = results_ln["per_sample_neuron"]["n_consistent"]
    comparison["rms_n_consistent"] = results_rms["per_sample_neuron"]["n_consistent"]

    # Best subgroup R²
    comparison["ln_best_pair_r2"] = results_ln["subgroups"]["best_pair_r2"]
    comparison["rms_best_pair_r2"] = results_rms["subgroups"]["best_pair_r2"]

    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_neuron_correlations(
    correlations_ln: np.ndarray, correlations_rms: np.ndarray, save_path: str
):
    """Plot distribution of per-neuron correlations for LN vs RMS."""

    fig = make_subplots(rows=1, cols=2, subplot_titles=("LayerNorm", "RMSNorm"))

    fig.add_trace(
        go.Histogram(x=correlations_ln, nbinsx=50, name="LN", marker_color="blue"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=correlations_rms, nbinsx=50, name="RMS", marker_color="red"),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Per-Neuron Position Correlation Distribution",
        template="plotly_white",
        width=1000,
        height=400,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Pearson r with position")
    fig.update_yaxes(title_text="Count")

    fig.write_image(f"{save_path}.png", width=1000, height=400, scale=2)
    fig.write_image(f"{save_path}.pdf")


def plot_sparse_probe_tradeoff(results_ln: dict, results_rms: dict, save_path: str):
    """Plot R² vs number of neurons for sparse probes."""

    fig = go.Figure()

    for name, results, color in [
        ("LayerNorm", results_ln, "blue"),
        ("RMSNorm", results_rms, "red"),
    ]:
        alphas = []
        r2s = []
        n_neurons = []

        for key, val in results.items():
            if key.startswith("alpha_"):
                alphas.append(val["alpha"])
                r2s.append(val["r2"])
                n_neurons.append(val["n_nonzero_neurons"])

        fig.add_trace(
            go.Scatter(
                x=n_neurons,
                y=r2s,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=10),
                text=[f"α={a}" for a in alphas],
                hovertemplate="Neurons: %{x}<br>R²: %{y:.3f}<br>%{text}",
            )
        )

    fig.update_layout(
        title="Sparse Probe: R² vs Number of Neurons",
        xaxis_title="Number of Non-Zero Neurons",
        yaxis_title="R² Score",
        template="plotly_white",
        width=800,
        height=500,
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")


def plot_top_neurons_heatmap(
    acts: np.ndarray, top_neurons: list, cfg: Config, save_path: str, title: str
):
    """Plot activation heatmap of top position-encoding neurons."""

    # Average activation by position for top neurons
    mean_acts = acts.mean(axis=0)  # [seq_len, d_model]
    top_acts = mean_acts[:, top_neurons[:20]]  # [seq_len, 20]

    fig = px.imshow(
        top_acts.T,
        x=list(range(cfg.seq_len)),
        y=[f"N{n}" for n in top_neurons[:20]],
        color_continuous_scale="RdBu_r",
        labels=dict(x="Position", y="Neuron", color="Activation"),
        aspect="auto",
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=800,
        height=400,
    )

    fig.write_image(f"{save_path}.png", width=800, height=400, scale=2)
    fig.write_image(f"{save_path}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Neuron Subgroup Position Analysis")
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
    print("NEURON SUBGROUP POSITION ENCODING ANALYSIS")
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

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYERNORM MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYZING LAYERNORM MODEL")
    print("=" * 70)

    model_ln = create_model(cfg, norm_type="LN")

    # Extract pre-LN2 and post-LN2 activations
    print("\nExtracting pre-LN2 activations...")
    acts_pre_ln2 = extract_activations(
        model_ln, tokens, "blocks.0.hook_resid_mid", cfg.batch_size
    )

    print("Extracting post-LN2 activations...")
    acts_post_ln2 = extract_activations(
        model_ln, tokens, "blocks.0.ln2.hook_normalized", cfg.batch_size
    )

    del model_ln
    torch.cuda.empty_cache()

    results_ln = {}

    print("\n1. Per-neuron correlation analysis...")
    results_ln["per_neuron"] = analyze_per_neuron_correlation(acts_post_ln2, cfg)
    print(f"   Significant neurons: {results_ln['per_neuron']['n_significant']}")
    print(f"   Max |correlation|: {results_ln['per_neuron']['max_abs_corr']:.4f}")

    print("\n2. Per-sample neuron consistency...")
    results_ln["per_sample_neuron"] = analyze_per_sample_neuron_correlation(
        acts_post_ln2, cfg
    )
    print(
        f"   Consistent neurons (>50% samples): {results_ln['per_sample_neuron']['n_consistent']}"
    )

    print("\n3. Sparse probe analysis...")
    results_ln["sparse_probe"] = analyze_sparse_probe(acts_post_ln2, cfg)
    best = results_ln["sparse_probe"]["best_tradeoff"]
    if best:
        print(f"   Best tradeoff: {best}")
        print(f"   R²: {results_ln['sparse_probe'][best]['r2']:.4f}")
        print(f"   Neurons: {results_ln['sparse_probe'][best]['n_nonzero_neurons']}")

    print("\n4. Neuron subgroup analysis...")
    results_ln["subgroups"] = analyze_neuron_subgroups(acts_post_ln2, cfg)
    print(f"   Best individual R²: {results_ln['subgroups']['best_individual_r2']:.4f}")
    print(f"   Best pair R²: {results_ln['subgroups']['best_pair_r2']:.4f}")
    print(f"   Best triplet R²: {results_ln['subgroups']['best_triplet_r2']:.4f}")

    print("\n5. Variance redistribution analysis...")
    results_ln["variance"] = analyze_variance_redistribution(
        acts_pre_ln2, acts_post_ln2, cfg
    )
    print(
        f"   Pre-norm var-position corr: {results_ln['variance']['pre_norm_var_corr_with_position']:.4f}"
    )
    print(
        f"   Post-norm var-position corr: {results_ln['variance']['post_norm_var_corr_with_position']:.4f}"
    )

    correlations_ln = np.array(results_ln["per_neuron"]["correlations"])
    top_neurons_ln = results_ln["per_neuron"]["top_positive_neurons"]

    all_results["LayerNorm"] = results_ln

    del acts_pre_ln2, acts_post_ln2

    # ═══════════════════════════════════════════════════════════════════════════
    # RMSNORM MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYZING RMSNORM MODEL")
    print("=" * 70)

    model_rms = create_model(cfg, norm_type="RMS")

    print("\nExtracting pre-LN2 activations...")
    acts_pre_ln2_rms = extract_activations(
        model_rms, tokens, "blocks.0.hook_resid_mid", cfg.batch_size
    )

    print("Extracting post-LN2 activations...")
    acts_post_ln2_rms = extract_activations(
        model_rms, tokens, "blocks.0.ln2.hook_normalized", cfg.batch_size
    )

    del model_rms
    torch.cuda.empty_cache()

    results_rms = {}

    print("\n1. Per-neuron correlation analysis...")
    results_rms["per_neuron"] = analyze_per_neuron_correlation(acts_post_ln2_rms, cfg)
    print(f"   Significant neurons: {results_rms['per_neuron']['n_significant']}")
    print(f"   Max |correlation|: {results_rms['per_neuron']['max_abs_corr']:.4f}")

    print("\n2. Per-sample neuron consistency...")
    results_rms["per_sample_neuron"] = analyze_per_sample_neuron_correlation(
        acts_post_ln2_rms, cfg
    )
    print(
        f"   Consistent neurons (>50% samples): {results_rms['per_sample_neuron']['n_consistent']}"
    )

    print("\n3. Sparse probe analysis...")
    results_rms["sparse_probe"] = analyze_sparse_probe(acts_post_ln2_rms, cfg)
    best = results_rms["sparse_probe"]["best_tradeoff"]
    if best:
        print(f"   Best tradeoff: {best}")
        print(f"   R²: {results_rms['sparse_probe'][best]['r2']:.4f}")
        print(f"   Neurons: {results_rms['sparse_probe'][best]['n_nonzero_neurons']}")

    print("\n4. Neuron subgroup analysis...")
    results_rms["subgroups"] = analyze_neuron_subgroups(acts_post_ln2_rms, cfg)
    print(
        f"   Best individual R²: {results_rms['subgroups']['best_individual_r2']:.4f}"
    )
    print(f"   Best pair R²: {results_rms['subgroups']['best_pair_r2']:.4f}")
    print(f"   Best triplet R²: {results_rms['subgroups']['best_triplet_r2']:.4f}")

    print("\n5. Variance redistribution analysis...")
    results_rms["variance"] = analyze_variance_redistribution(
        acts_pre_ln2_rms, acts_post_ln2_rms, cfg
    )
    print(
        f"   Pre-norm var-position corr: {results_rms['variance']['pre_norm_var_corr_with_position']:.4f}"
    )
    print(
        f"   Post-norm var-position corr: {results_rms['variance']['post_norm_var_corr_with_position']:.4f}"
    )

    correlations_rms = np.array(results_rms["per_neuron"]["correlations"])
    top_neurons_rms = results_rms["per_neuron"]["top_positive_neurons"]

    all_results["RMSNorm"] = results_rms

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("LN vs RMSNorm COMPARISON")
    print("=" * 70)

    comparison = compare_ln_vs_rms(results_ln, results_rms)
    all_results["comparison"] = comparison

    print(f"\n{'Metric':<40} {'LayerNorm':<15} {'RMSNorm':<15}")
    print("-" * 70)
    print(
        f"{'Significant neurons':<40} {comparison['ln_n_significant']:<15} {comparison['rms_n_significant']:<15}"
    )
    print(
        f"{'Max |correlation|':<40} {comparison['ln_max_corr']:<15.4f} {comparison['rms_max_corr']:<15.4f}"
    )
    print(
        f"{'Consistent neurons':<40} {comparison['ln_n_consistent']:<15} {comparison['rms_n_consistent']:<15}"
    )
    print(
        f"{'Best pair R²':<40} {comparison['ln_best_pair_r2']:<15.4f} {comparison['rms_best_pair_r2']:<15.4f}"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_neuron_correlations(
        correlations_ln,
        correlations_rms,
        str(RESULTS_DIR / "neuron_correlation_distribution"),
    )

    plot_sparse_probe_tradeoff(
        results_ln["sparse_probe"],
        results_rms["sparse_probe"],
        str(RESULTS_DIR / "sparse_probe_tradeoff"),
    )

    # Reload activations for heatmaps (small subset)
    # Skip for now to save memory

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    all_results_json = convert_for_json(all_results)

    with open(RESULTS_DIR / "neuron_subgroup_results.json", "w") as f:
        json.dump(all_results_json, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
