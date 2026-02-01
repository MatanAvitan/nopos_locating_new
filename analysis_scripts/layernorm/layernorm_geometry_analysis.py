"""
LayerNorm Geometric Analysis: Why LayerNorm AMPLIFIES Position Signal

Key Question: If LayerNorm normalizes variance to 1, why does position-norm correlation
get STRONGER after LN (from -0.60 to -0.998)?

Hypotheses to test:
1. LN preserves relative variance ordering while normalizing absolute variance
2. Position is encoded in the "shape" (relative neuron activations) not just norm
3. LN's centering step (mean subtraction) reveals position information
4. LN's gain/bias parameters encode position-related transformations

This script provides detailed geometric analysis of LayerNorm's transformation.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

# Add parent directory for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from nanoGPT.model_nope import GPT, GPTConfig

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def create_random_model(n_layer=1, n_head=4, n_embd=256, vocab_size=65, block_size=64):
    """Create a randomly initialized NoPE model."""
    config = GPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=0.0,
        use_positional_embedding=False,
        norm_type="layernorm",
    )
    model = GPT(config)
    model.eval()
    model.to(device)
    return model, config


def manual_layernorm(x, weight, bias, eps=1e-5):
    """Manual LayerNorm to extract intermediate values."""
    # x: (batch, seq, hidden)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_centered = x - mean
    x_normalized = x_centered / torch.sqrt(var + eps)
    output = weight * x_normalized + bias
    return output, mean.squeeze(-1), var.squeeze(-1), x_centered, x_normalized


def analyze_layernorm_geometry(model, n_samples=500, block_size=64):
    """
    Detailed analysis of LayerNorm's geometric transformation.
    """
    results = {"pre_ln": {}, "post_ln": {}, "transformation": {}}

    vocab_size = model.config.vocab_size

    # Generate random input sequences
    tokens = torch.randint(0, vocab_size, (n_samples, block_size), device=device)

    with torch.no_grad():
        # Get embeddings
        tok_emb = model.transformer.wte(tokens)

        # Pass through first layer's attention
        ln1 = model.transformer.h[0].ln_1
        attn = model.transformer.h[0].attn

        # Pre-LN1 (embedding)
        pre_ln1 = tok_emb

        # Through LN1
        post_ln1 = ln1(pre_ln1)

        # Through attention
        post_attn = attn(post_ln1)

        # Residual connection
        post_attn_residual = pre_ln1 + post_attn

        # Through LN2 (pre-MLP)
        ln2 = model.transformer.h[0].ln_2

        # Manual LN2 to get intermediate values
        ln2_out, ln2_mean, ln2_var, ln2_centered, ln2_normalized = manual_layernorm(
            post_attn_residual, ln2.weight, ln2.bias
        )

    # ==== Analysis 1: Variance Redistribution ====
    print("\n=== Analysis 1: Variance Redistribution ===")

    # Pre-LN2 variance per position
    pre_ln2_var = post_attn_residual.var(dim=-1)  # (batch, seq)
    pre_ln2_var_by_pos = pre_ln2_var.mean(dim=0).cpu().numpy()  # (seq,)

    # Post-LN2 variance (should be ~1 due to normalization)
    post_ln2_var = ln2_out.var(dim=-1)
    post_ln2_var_by_pos = post_ln2_var.mean(dim=0).cpu().numpy()

    # Correlation with position
    positions = np.arange(block_size)
    pre_var_corr, _ = stats.pearsonr(positions, pre_ln2_var_by_pos)
    post_var_corr, _ = stats.pearsonr(positions, post_ln2_var_by_pos)

    print(f"Pre-LN2 variance-position correlation: {pre_var_corr:.4f}")
    print(f"Post-LN2 variance-position correlation: {post_var_corr:.4f}")

    results["variance_redistribution"] = {
        "pre_ln_var_pos_corr": float(pre_var_corr),
        "post_ln_var_pos_corr": float(post_var_corr),
        "pre_ln_var_by_pos": pre_ln2_var_by_pos.tolist(),
        "post_ln_var_by_pos": post_ln2_var_by_pos.tolist(),
    }

    # ==== Analysis 2: Norm vs Variance Decomposition ====
    print("\n=== Analysis 2: Norm vs Variance Decomposition ===")

    # L2 norm = sqrt(sum of squared values) = sqrt(n * (var + mean^2))
    # If mean is small, norm ≈ sqrt(n) * std

    pre_ln2_norm = post_attn_residual.norm(dim=-1)  # (batch, seq)
    pre_ln2_mean = post_attn_residual.mean(dim=-1)  # (batch, seq)
    pre_ln2_std = post_attn_residual.std(dim=-1)  # (batch, seq)

    # Mean and std by position
    pre_ln2_norm_by_pos = pre_ln2_norm.mean(dim=0).cpu().numpy()
    pre_ln2_mean_by_pos = pre_ln2_mean.mean(dim=0).cpu().numpy()
    pre_ln2_std_by_pos = pre_ln2_std.mean(dim=0).cpu().numpy()

    norm_corr, _ = stats.pearsonr(positions, pre_ln2_norm_by_pos)
    mean_corr, _ = stats.pearsonr(positions, np.abs(pre_ln2_mean_by_pos))
    std_corr, _ = stats.pearsonr(positions, pre_ln2_std_by_pos)

    print(f"Pre-LN2 norm-position correlation: {norm_corr:.4f}")
    print(f"Pre-LN2 |mean|-position correlation: {mean_corr:.4f}")
    print(f"Pre-LN2 std-position correlation: {std_corr:.4f}")

    results["norm_decomposition"] = {
        "norm_pos_corr": float(norm_corr),
        "mean_pos_corr": float(mean_corr),
        "std_pos_corr": float(std_corr),
        "norm_by_pos": pre_ln2_norm_by_pos.tolist(),
        "mean_by_pos": pre_ln2_mean_by_pos.tolist(),
        "std_by_pos": pre_ln2_std_by_pos.tolist(),
    }

    # ==== Analysis 3: LN Mean Subtraction Effect ====
    print("\n=== Analysis 3: LN Mean Subtraction Effect ===")

    # The mean that LN subtracts
    ln2_mean_by_pos = ln2_mean.mean(dim=0).cpu().numpy()  # (seq,)
    mean_subtracted_corr, _ = stats.pearsonr(positions, ln2_mean_by_pos)

    # After centering but before scaling
    centered_norm = ln2_centered.norm(dim=-1)
    centered_norm_by_pos = centered_norm.mean(dim=0).cpu().numpy()
    centered_norm_corr, _ = stats.pearsonr(positions, centered_norm_by_pos)

    print(f"LN mean subtracted - position correlation: {mean_subtracted_corr:.4f}")
    print(f"Post-centering norm - position correlation: {centered_norm_corr:.4f}")

    results["centering_effect"] = {
        "mean_subtracted_pos_corr": float(mean_subtracted_corr),
        "centered_norm_pos_corr": float(centered_norm_corr),
        "mean_by_pos": ln2_mean_by_pos.tolist(),
        "centered_norm_by_pos": centered_norm_by_pos.tolist(),
    }

    # ==== Analysis 4: Normalized Vector Analysis ====
    print("\n=== Analysis 4: Normalized Vector Analysis ===")

    # After LN normalization (before gain/bias), all vectors have unit variance
    # But do they differ in direction systematically by position?

    normalized_norm = ln2_normalized.norm(dim=-1)  # Should be sqrt(n_embd) for all
    normalized_norm_by_pos = normalized_norm.mean(dim=0).cpu().numpy()
    normalized_norm_corr, _ = stats.pearsonr(positions, normalized_norm_by_pos)

    print(f"Normalized vector norm - position correlation: {normalized_norm_corr:.4f}")
    print(f"Normalized vector norm mean: {normalized_norm_by_pos.mean():.4f}")
    print(f"Expected (sqrt(n_embd)): {np.sqrt(model.config.n_embd):.4f}")

    results["normalized_vectors"] = {
        "norm_pos_corr": float(normalized_norm_corr),
        "mean_norm": float(normalized_norm_by_pos.mean()),
        "expected_norm": float(np.sqrt(model.config.n_embd)),
        "norm_by_pos": normalized_norm_by_pos.tolist(),
    }

    # ==== Analysis 5: Gain/Bias Effect ====
    print("\n=== Analysis 5: LN Gain/Bias Effect ===")

    # LN output = weight * x_normalized + bias
    # Does the weight/bias introduce position-dependent effects?

    # Compare: just normalized vs with gain/bias
    post_ln2_norm = ln2_out.norm(dim=-1)
    post_ln2_norm_by_pos = post_ln2_norm.mean(dim=0).cpu().numpy()
    post_ln2_norm_corr, _ = stats.pearsonr(positions, post_ln2_norm_by_pos)

    # Just with gain (no bias)
    just_gain = ln2.weight * ln2_normalized
    just_gain_norm = just_gain.norm(dim=-1)
    just_gain_norm_by_pos = just_gain_norm.mean(dim=0).detach().cpu().numpy()
    just_gain_norm_corr, _ = stats.pearsonr(positions, just_gain_norm_by_pos)

    print(f"After normalization only - norm-position corr: {normalized_norm_corr:.4f}")
    print(f"After gain only - norm-position corr: {just_gain_norm_corr:.4f}")
    print(f"After gain+bias (full LN) - norm-position corr: {post_ln2_norm_corr:.4f}")

    # Gain statistics
    gain_mean = ln2.weight.mean().item()
    gain_std = ln2.weight.std().item()
    bias_mean = ln2.bias.mean().item()
    bias_std = ln2.bias.std().item()

    print(f"LN gain mean: {gain_mean:.4f}, std: {gain_std:.4f}")
    print(f"LN bias mean: {bias_mean:.4f}, std: {bias_std:.4f}")

    results["gain_bias_effect"] = {
        "normalized_only_corr": float(normalized_norm_corr),
        "gain_only_corr": float(just_gain_norm_corr),
        "full_ln_corr": float(post_ln2_norm_corr),
        "gain_mean": float(gain_mean),
        "gain_std": float(gain_std),
        "bias_mean": float(bias_mean),
        "bias_std": float(bias_std),
        "post_ln_norm_by_pos": post_ln2_norm_by_pos.tolist(),
    }

    # ==== Analysis 6: Per-Sample Correlation Analysis ====
    print("\n=== Analysis 6: Per-Sample Correlation Analysis ===")

    # Compute per-sample correlations for various quantities
    def compute_per_sample_corrs(values, positions):
        """Compute per-sample correlation with position."""
        corrs = []
        for i in range(values.shape[0]):
            sample_vals = values[i].cpu().numpy()
            r, _ = stats.pearsonr(positions, sample_vals)
            corrs.append(r)
        return np.array(corrs)

    pre_var_corrs = compute_per_sample_corrs(pre_ln2_var, positions)
    post_var_corrs = compute_per_sample_corrs(post_ln2_var, positions)
    pre_norm_corrs = compute_per_sample_corrs(pre_ln2_norm, positions)
    post_norm_corrs = compute_per_sample_corrs(post_ln2_norm, positions)

    print(
        f"Pre-LN2 variance-pos per-sample corr: {pre_var_corrs.mean():.4f} ± {pre_var_corrs.std():.4f}"
    )
    print(
        f"Post-LN2 variance-pos per-sample corr: {post_var_corrs.mean():.4f} ± {post_var_corrs.std():.4f}"
    )
    print(
        f"Pre-LN2 norm-pos per-sample corr: {pre_norm_corrs.mean():.4f} ± {pre_norm_corrs.std():.4f}"
    )
    print(
        f"Post-LN2 norm-pos per-sample corr: {post_norm_corrs.mean():.4f} ± {post_norm_corrs.std():.4f}"
    )

    results["per_sample_correlations"] = {
        "pre_ln_var_pos": {
            "mean": float(pre_var_corrs.mean()),
            "std": float(pre_var_corrs.std()),
        },
        "post_ln_var_pos": {
            "mean": float(post_var_corrs.mean()),
            "std": float(post_var_corrs.std()),
        },
        "pre_ln_norm_pos": {
            "mean": float(pre_norm_corrs.mean()),
            "std": float(pre_norm_corrs.std()),
        },
        "post_ln_norm_pos": {
            "mean": float(post_norm_corrs.mean()),
            "std": float(post_norm_corrs.std()),
        },
    }

    # ==== Analysis 7: Direction vs Magnitude After LN ====
    print("\n=== Analysis 7: Direction vs Magnitude After LN ===")

    # Train linear probes on:
    # 1. Full post-LN activations
    # 2. Unit vectors (direction only)
    # 3. Norms only (magnitude only)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    # Flatten for regression
    X_full = ln2_out.reshape(-1, model.config.n_embd).cpu().numpy()
    X_unit = (
        (ln2_out / ln2_out.norm(dim=-1, keepdim=True))
        .reshape(-1, model.config.n_embd)
        .cpu()
        .numpy()
    )
    X_norm = post_ln2_norm.reshape(-1, 1).cpu().numpy()
    y = np.tile(positions, n_samples)

    # Train/test split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    # Full activations
    reg_full = Ridge(alpha=1.0)
    reg_full.fit(X_full[train_idx], y[train_idx])
    r2_full = reg_full.score(X_full[test_idx], y[test_idx])

    # Unit vectors only
    reg_unit = Ridge(alpha=1.0)
    reg_unit.fit(X_unit[train_idx], y[train_idx])
    r2_unit = reg_unit.score(X_unit[test_idx], y[test_idx])

    # Norm only
    reg_norm = Ridge(alpha=1.0)
    reg_norm.fit(X_norm[train_idx], y[train_idx])
    r2_norm = reg_norm.score(X_norm[test_idx], y[test_idx])

    print(f"Full activations R²: {r2_full:.4f}")
    print(f"Unit vectors (direction) R²: {r2_unit:.4f}")
    print(f"Norm (magnitude) R²: {r2_norm:.4f}")

    results["direction_vs_magnitude"] = {
        "full_r2": float(r2_full),
        "direction_only_r2": float(r2_unit),
        "magnitude_only_r2": float(r2_norm),
    }

    # ==== Analysis 8: Theoretical Expected Norm ====
    print("\n=== Analysis 8: Theoretical Expected Norm ===")

    # For uniform attention averaging i+1 random vectors:
    # E[||h_i||] ∝ E[||avg of (i+1) vectors||]
    # For random vectors: ||avg|| ≈ ||original|| / sqrt(i+1)

    # Expected norm ratio: norm(pos_i) / norm(pos_0) ≈ 1/sqrt(i+1)
    expected_ratios = 1.0 / np.sqrt(positions + 1)
    actual_ratios = pre_ln2_norm_by_pos / pre_ln2_norm_by_pos[0]

    ratio_corr, _ = stats.pearsonr(expected_ratios, actual_ratios)

    print(f"Correlation between expected and actual norm ratios: {ratio_corr:.4f}")

    results["theoretical_analysis"] = {
        "expected_norm_ratios": expected_ratios.tolist(),
        "actual_norm_ratios": actual_ratios.tolist(),
        "ratio_correlation": float(ratio_corr),
    }

    return results


def create_plots(results, save_dir):
    """Create visualization plots."""
    os.makedirs(save_dir, exist_ok=True)
    positions = np.arange(len(results["variance_redistribution"]["pre_ln_var_by_pos"]))

    # Plot 1: Variance redistribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(
        positions,
        results["variance_redistribution"]["pre_ln_var_by_pos"],
        label=f"Pre-LN (r={results['variance_redistribution']['pre_ln_var_pos_corr']:.3f})",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Variance")
    ax.set_title("Pre-LayerNorm Variance by Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(
        positions,
        results["variance_redistribution"]["post_ln_var_by_pos"],
        label=f"Post-LN (r={results['variance_redistribution']['post_ln_var_pos_corr']:.3f})",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Variance")
    ax.set_title("Post-LayerNorm Variance by Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "variance_redistribution.png"), dpi=150)
    plt.close()

    # Plot 2: Norm decomposition
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(positions, results["norm_decomposition"]["norm_by_pos"])
    ax.set_xlabel("Position")
    ax.set_ylabel("L2 Norm")
    ax.set_title(
        f"Pre-LN Norm (r={results['norm_decomposition']['norm_pos_corr']:.3f})"
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(positions, results["norm_decomposition"]["mean_by_pos"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean")
    ax.set_title(
        f"Pre-LN Mean (r={results['norm_decomposition']['mean_pos_corr']:.3f})"
    )
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(positions, results["norm_decomposition"]["std_by_pos"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Std Dev")
    ax.set_title(f"Pre-LN Std (r={results['norm_decomposition']['std_pos_corr']:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "norm_decomposition.png"), dpi=150)
    plt.close()

    # Plot 3: Gain/Bias effect
    fig, ax = plt.subplots(figsize=(10, 5))

    # Normalize for comparison
    norm_by_pos = np.array(results["normalized_vectors"]["norm_by_pos"])
    post_ln_norm = np.array(results["gain_bias_effect"]["post_ln_norm_by_pos"])

    ax.plot(
        positions,
        norm_by_pos / norm_by_pos.mean(),
        label=f"After normalization (r={results['gain_bias_effect']['normalized_only_corr']:.3f})",
    )
    ax.plot(
        positions,
        post_ln_norm / post_ln_norm.mean(),
        label=f"After full LN (r={results['gain_bias_effect']['full_ln_corr']:.3f})",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Normalized Norm (relative)")
    ax.set_title("Effect of LN Gain/Bias on Position Encoding")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gain_bias_effect.png"), dpi=150)
    plt.close()

    # Plot 4: Theoretical vs Actual
    fig, ax = plt.subplots(figsize=(10, 5))

    expected = results["theoretical_analysis"]["expected_norm_ratios"]
    actual = results["theoretical_analysis"]["actual_norm_ratios"]

    ax.plot(positions, expected, label="Expected (1/sqrt(i+1))", linestyle="--")
    ax.plot(positions, actual, label="Actual")
    ax.set_xlabel("Position")
    ax.set_ylabel("Norm Ratio (relative to position 0)")
    ax.set_title(
        f"Theoretical vs Actual Norm Decay (r={results['theoretical_analysis']['ratio_correlation']:.3f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "theoretical_vs_actual.png"), dpi=150)
    plt.close()

    # Plot 5: Direction vs Magnitude comparison (bar plot)
    fig, ax = plt.subplots(figsize=(8, 5))

    components = ["Full\nActivations", "Direction\n(Unit Vectors)", "Magnitude\n(Norm)"]
    r2_values = [
        results["direction_vs_magnitude"]["full_r2"],
        results["direction_vs_magnitude"]["direction_only_r2"],
        results["direction_vs_magnitude"]["magnitude_only_r2"],
    ]

    bars = ax.bar(components, r2_values, color=["steelblue", "coral", "green"])
    ax.set_ylabel("R² Score")
    ax.set_title("Position Decoding: Direction vs Magnitude (Post-LN)")
    ax.set_ylim(0, max(r2_values) * 1.2)

    for bar, val in zip(bars, r2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "direction_vs_magnitude.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {save_dir}")


def main():
    print("=" * 70)
    print("LayerNorm Geometric Analysis")
    print("=" * 70)

    # Create model
    print("\nCreating random NoPE model...")
    model, config = create_random_model(
        n_layer=1, n_head=4, n_embd=256, vocab_size=65, block_size=64
    )

    # Run analysis
    print("\nRunning LayerNorm geometry analysis...")
    results = analyze_layernorm_geometry(model, n_samples=500, block_size=64)

    # Save results
    save_dir = Path(__file__).parent.parent / "results" / "layernorm_geometry"
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir / "layernorm_geometry_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_dir / 'layernorm_geometry_results.json'}")

    # Create plots
    create_plots(results, save_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Why LayerNorm Amplifies Position Signal")
    print("=" * 70)

    print(f"""
Key Findings:

1. VARIANCE REDISTRIBUTION:
   - Pre-LN variance-position correlation: {results["variance_redistribution"]["pre_ln_var_pos_corr"]:.4f}
   - Post-LN variance-position correlation: {results["variance_redistribution"]["post_ln_var_pos_corr"]:.4f}
   
2. NORM vs DIRECTION vs MAGNITUDE:
   - Full activations R²: {results["direction_vs_magnitude"]["full_r2"]:.4f}
   - Direction only R²: {results["direction_vs_magnitude"]["direction_only_r2"]:.4f}  
   - Magnitude only R²: {results["direction_vs_magnitude"]["magnitude_only_r2"]:.4f}

3. LN GAIN/BIAS EFFECT:
   - Before gain/bias: norm-pos corr = {results["gain_bias_effect"]["normalized_only_corr"]:.4f}
   - After gain/bias: norm-pos corr = {results["gain_bias_effect"]["full_ln_corr"]:.4f}

4. THEORETICAL FIT:
   - Expected norm ratio (1/sqrt(i+1)) correlation with actual: {results["theoretical_analysis"]["ratio_correlation"]:.4f}

Interpretation:
- LayerNorm normalizes variance to 1, but this doesn't destroy position info
- Position is encoded in the OUTPUT NORM of LN (not input variance)
- The gain/bias parameters {"amplify" if results["gain_bias_effect"]["full_ln_corr"] > results["gain_bias_effect"]["normalized_only_corr"] else "maintain"} the position signal
- Direction vs Magnitude: {"Magnitude dominates" if results["direction_vs_magnitude"]["magnitude_only_r2"] > results["direction_vs_magnitude"]["direction_only_r2"] else "Direction contains info"}
""")


if __name__ == "__main__":
    main()
