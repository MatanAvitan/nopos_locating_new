"""
Attention Pattern Position Analysis

Investigates how causal attention patterns encode position information.
In NoPE models, the attention pattern itself may be the primary position signal.

Key insight: At position i, the model attends to positions 0..i with causal masking.
The attention pattern (which positions are attended to) inherently encodes position.

Tests:
1. Attention entropy by position
2. Attention pattern statistics (mean, std of weights)
3. Position decoding from attention patterns alone
4. Comparison of uniform vs learned attention patterns

Usage:
    python attention_pattern_analysis.py --n_samples 5000
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, entropy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

RESULTS_DIR = Path("results/attention_pattern_analysis")
device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(d_model=1024, seq_len=64, norm_type="LN"):
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_model,
        n_heads=1,
        d_mlp=4096,
        d_vocab=50257,
        n_ctx=seq_len,
        act_fn="gelu",
        normalization_type=norm_type,
        device=device,
    )
    model = HookedTransformer(cfg)
    model.pos_embed.W_pos.data.zero_()
    model.pos_embed.W_pos.requires_grad = False
    return model


def extract_attention_patterns(model, tokens, batch_size=256):
    """Extract attention patterns [n_samples, n_heads, seq_len, seq_len]."""
    model.eval()
    patterns = []

    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size), desc="Extracting attention"):
            batch = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(
                batch, names_filter=["blocks.0.attn.hook_pattern"]
            )
            patterns.append(cache["blocks.0.attn.hook_pattern"].cpu())
            del cache
            torch.cuda.empty_cache()

    return torch.cat(patterns, dim=0).numpy()


def analyze_attention_entropy(patterns, seq_len):
    """Analyze entropy of attention distribution by query position."""
    # patterns: [n_samples, n_heads, seq_len, seq_len]
    n_samples = patterns.shape[0]

    # For each query position, compute entropy of attention distribution
    entropies_by_pos = []
    theoretical_max_entropy = []

    for q_pos in range(seq_len):
        # Attention at query position q_pos over keys 0..q_pos
        attn_weights = patterns[:, 0, q_pos, : q_pos + 1]  # [n_samples, q_pos+1]

        # Compute entropy for each sample
        sample_entropies = []
        for s in range(n_samples):
            # Add small epsilon to avoid log(0)
            weights = attn_weights[s] + 1e-10
            weights = weights / weights.sum()
            h = entropy(weights)
            sample_entropies.append(h)

        entropies_by_pos.append(np.mean(sample_entropies))
        theoretical_max_entropy.append(
            np.log(q_pos + 1)
        )  # Uniform distribution entropy

    return {
        "entropy_by_position": entropies_by_pos,
        "max_entropy_by_position": theoretical_max_entropy,
        "normalized_entropy": [
            e / m if m > 0 else 0
            for e, m in zip(entropies_by_pos, theoretical_max_entropy)
        ],
    }


def analyze_attention_statistics(patterns, seq_len):
    """Compute statistics of attention patterns by position."""
    n_samples = patterns.shape[0]

    results = {
        "mean_attn_to_first": [],  # Attention to position 0
        "mean_attn_to_self": [],  # Attention to current position
        "attn_std": [],  # Std of attention weights
        "attn_max": [],  # Max attention weight
    }

    for q_pos in range(seq_len):
        attn = patterns[:, 0, q_pos, : q_pos + 1]  # [n_samples, q_pos+1]

        results["mean_attn_to_first"].append(float(attn[:, 0].mean()))
        results["mean_attn_to_self"].append(float(attn[:, -1].mean()))
        results["attn_std"].append(float(attn.std(axis=1).mean()))
        results["attn_max"].append(float(attn.max(axis=1).mean()))

    # Correlations with position
    positions = np.arange(seq_len)
    results["corr_first_with_pos"], _ = pearsonr(
        positions, results["mean_attn_to_first"]
    )
    results["corr_self_with_pos"], _ = pearsonr(positions, results["mean_attn_to_self"])
    results["corr_std_with_pos"], _ = pearsonr(positions, results["attn_std"])

    return results


def decode_position_from_attention(patterns, seq_len):
    """Test if position can be decoded from attention pattern alone."""
    n_samples = patterns.shape[0]
    positions = np.arange(seq_len)

    # Create features from attention patterns
    # For position i, the attention pattern has shape [i+1]
    # Pad to fixed size for probe training

    features = []
    labels = []

    for s in range(n_samples):
        for q_pos in range(seq_len):
            attn = patterns[s, 0, q_pos, : q_pos + 1]
            # Pad to seq_len
            padded = np.zeros(seq_len)
            padded[: q_pos + 1] = attn
            features.append(padded)
            labels.append(q_pos)

    features = np.array(features)
    labels = np.array(labels)

    # Train/test split
    n_total = len(labels)
    n_train = int(n_total * 0.8)

    idx = np.random.permutation(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_pred - y_test))
    r, _ = pearsonr(y_test, y_pred)

    return {
        "r2": float(r2),
        "mae": float(mae),
        "pearson_r": float(r),
    }


def compare_with_uniform_attention(model, tokens, seq_len, batch_size=256):
    """
    Compare model with uniform attention to see how much position info
    comes from attention patterns vs other sources.
    """
    model.eval()

    # Hook to replace attention with uniform
    def uniform_attn_hook(pattern, hook):
        # pattern: [batch, heads, q, k]
        batch, heads, q_len, k_len = pattern.shape
        # Create uniform causal attention
        uniform = torch.zeros_like(pattern)
        for q in range(q_len):
            uniform[:, :, q, : q + 1] = 1.0 / (q + 1)
        return uniform

    # Get activations with learned attention
    acts_learned = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(
                batch, names_filter=["blocks.0.ln2.hook_normalized"]
            )
            acts_learned.append(cache["blocks.0.ln2.hook_normalized"].cpu())
            del cache
    acts_learned = torch.cat(acts_learned, dim=0).numpy()

    # Get activations with uniform attention
    acts_uniform = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i : i + batch_size]
            with model.hooks(
                fwd_hooks=[("blocks.0.attn.hook_pattern", uniform_attn_hook)]
            ):
                _, cache = model.run_with_cache(
                    batch, names_filter=["blocks.0.ln2.hook_normalized"]
                )
            acts_uniform.append(cache["blocks.0.ln2.hook_normalized"].cpu())
            del cache
    acts_uniform = torch.cat(acts_uniform, dim=0).numpy()

    # Compare norm correlations
    positions = np.arange(seq_len)
    n_samples = acts_learned.shape[0]

    # Learned attention norm correlation
    learned_norms = np.linalg.norm(acts_learned, axis=2)
    learned_corrs = [pearsonr(positions, learned_norms[i])[0] for i in range(n_samples)]

    # Uniform attention norm correlation
    uniform_norms = np.linalg.norm(acts_uniform, axis=2)
    uniform_corrs = [pearsonr(positions, uniform_norms[i])[0] for i in range(n_samples)]

    return {
        "learned_mean_corr": float(np.mean(learned_corrs)),
        "learned_std_corr": float(np.std(learned_corrs)),
        "uniform_mean_corr": float(np.mean(uniform_corrs)),
        "uniform_std_corr": float(np.std(uniform_corrs)),
        "learned_mean_norm_by_pos": learned_norms.mean(axis=0).tolist(),
        "uniform_mean_norm_by_pos": uniform_norms.mean(axis=0).tolist(),
    }


def plot_results(entropy_results, stat_results, save_dir):
    """Generate plots."""
    seq_len = len(entropy_results["entropy_by_position"])
    positions = list(range(seq_len))

    # Entropy plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=entropy_results["entropy_by_position"],
            name="Actual Entropy",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=entropy_results["max_entropy_by_position"],
            name="Max Entropy (uniform)",
            mode="lines",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Attention Entropy by Position",
        xaxis_title="Position",
        yaxis_title="Entropy",
        template="plotly_white",
        width=800,
        height=400,
    )
    fig.write_image(f"{save_dir}/attention_entropy.png", scale=2)
    fig.write_image(f"{save_dir}/attention_entropy.pdf")

    # Attention statistics plot
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Attention to First Token",
            "Attention to Self",
            "Attention Std",
            "Attention Max",
        ),
    )

    fig.add_trace(
        go.Scatter(x=positions, y=stat_results["mean_attn_to_first"], mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=positions, y=stat_results["mean_attn_to_self"], mode="lines"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=positions, y=stat_results["attn_std"], mode="lines"), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=positions, y=stat_results["attn_max"], mode="lines"), row=2, col=2
    )

    fig.update_layout(
        title="Attention Statistics by Position",
        template="plotly_white",
        width=900,
        height=600,
        showlegend=False,
    )
    fig.write_image(f"{save_dir}/attention_statistics.png", scale=2)
    fig.write_image(f"{save_dir}/attention_statistics.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=1024)
    args = parser.parse_args()

    setup_dirs()

    print("=" * 70)
    print("ATTENTION PATTERN POSITION ANALYSIS")
    print("=" * 70)

    torch.manual_seed(42)
    tokens = torch.randint(0, 50257, (args.n_samples, args.seq_len), device=device)

    all_results = {}

    for norm_type in ["LN", "RMS"]:
        print(f"\n{'=' * 50}")
        print(f"Analyzing {norm_type} model")
        print("=" * 50)

        model = create_model(args.d_model, args.seq_len, norm_type)

        print("\nExtracting attention patterns...")
        patterns = extract_attention_patterns(model, tokens)

        print("\n1. Attention entropy analysis...")
        entropy_results = analyze_attention_entropy(patterns, args.seq_len)
        print(
            f"   Mean normalized entropy: {np.mean(entropy_results['normalized_entropy']):.4f}"
        )

        print("\n2. Attention statistics...")
        stat_results = analyze_attention_statistics(patterns, args.seq_len)
        print(
            f"   Corr(attn_to_first, position): {stat_results['corr_first_with_pos']:.4f}"
        )
        print(
            f"   Corr(attn_to_self, position): {stat_results['corr_self_with_pos']:.4f}"
        )

        print("\n3. Position decoding from attention...")
        decode_results = decode_position_from_attention(patterns, args.seq_len)
        print(f"   R²: {decode_results['r2']:.4f}")
        print(f"   MAE: {decode_results['mae']:.2f}")

        print("\n4. Uniform vs learned attention comparison...")
        compare_results = compare_with_uniform_attention(model, tokens, args.seq_len)
        print(
            f"   Learned attention norm-pos corr: {compare_results['learned_mean_corr']:.4f}"
        )
        print(
            f"   Uniform attention norm-pos corr: {compare_results['uniform_mean_corr']:.4f}"
        )

        all_results[norm_type] = {
            "entropy": entropy_results,
            "statistics": stat_results,
            "decoding": decode_results,
            "comparison": compare_results,
        }

        del model, patterns
        torch.cuda.empty_cache()

    # Save results
    with open(RESULTS_DIR / "attention_pattern_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate plots
    plot_results(
        all_results["LN"]["entropy"], all_results["LN"]["statistics"], str(RESULTS_DIR)
    )

    print(f"\nResults saved to {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
