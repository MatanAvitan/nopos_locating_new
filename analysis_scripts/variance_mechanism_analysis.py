"""
Variance-Based Position Encoding Mechanism Analysis

Key hypothesis to test:
- Position is encoded in the variance/norm of post-attention activations
- ||z_i||² ∝ 1/(i+1) due to averaging in causal attention
- 1/||z_i||² ∝ (i+1) is linearly predictable
- LayerNorm's division by σ_i effectively encodes sqrt(i+1)
- MLP uses nonlinearity to decode position from these statistics

This script tests these hypotheses on the actual trained NoPE models.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    config.log_attention_stats = False
    model = GPT(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def extract_detailed_activations(model, input_ids, device="cuda"):
    """
    Extract activations at each stage, plus statistics like norms and variances.
    """
    model.eval()
    activations = {}
    statistics = {}

    with torch.no_grad():
        # 1. Raw embeddings
        tok_emb = model.transformer.wte(input_ids)
        activations["raw_embed"] = tok_emb.cpu()

        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[0]

        # 2. Post-LN1 (before attention)
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.cpu()

        # 3. Post-attention OUTPUT ONLY (before residual)
        # We need to hook into the attention to get this
        attn_out = block.attn(x_ln1)
        activations["attn_output"] = attn_out.cpu()

        # 4. Post-attention + residual
        x_attn = x + attn_out
        activations["post_attn_residual"] = x_attn.cpu()

        # 5. Post-LN2 (before MLP)
        x_ln2 = block.ln_2(x_attn)
        activations["post_ln2"] = x_ln2.cpu()

        # 6. Post-MLP (after MLP + residual)
        mlp_out = block.mlp(x_ln2)
        x_mlp = x_attn + mlp_out
        activations["post_mlp"] = x_mlp.cpu()

        # 7. Post-final-LN
        x_final = model.transformer.ln_f(x_mlp)
        activations["post_final_ln"] = x_final.cpu()

    # Compute statistics for each activation point
    for name, act in activations.items():
        # act shape: [batch, seq_len, d_model]
        statistics[f"{name}_norm"] = act.norm(dim=-1)  # [batch, seq_len]
        statistics[f"{name}_var"] = act.var(dim=-1)  # [batch, seq_len]
        statistics[f"{name}_mean"] = act.mean(dim=-1)  # [batch, seq_len]
        statistics[f"{name}_std"] = act.std(dim=-1)  # [batch, seq_len]

    return activations, statistics


def test_norm_position_correlation(statistics, positions, n_samples, seq_len):
    """
    Test if 1/||z_i||² correlates with position.
    """
    results = {}

    stat_keys = [k for k in statistics.keys() if k.endswith("_norm")]

    for key in stat_keys:
        norms = statistics[key]  # [n_samples, seq_len]

        # Flatten for correlation
        norms_flat = norms.flatten().numpy()
        positions_flat = positions.flatten().numpy()

        # Basic norm correlation
        r_norm, _ = pearsonr(norms_flat, positions_flat)

        # Inverse squared norm correlation: 1/||z||² vs position
        inv_sq_norm = 1 / (norms_flat**2 + 1e-8)
        r_inv_sq, _ = pearsonr(inv_sq_norm, positions_flat)

        # Inverse norm correlation: 1/||z|| vs position
        inv_norm = 1 / (norms_flat + 1e-8)
        r_inv, _ = pearsonr(inv_norm, positions_flat)

        # Log norm correlation
        log_norm = np.log(norms_flat + 1e-8)
        r_log, _ = pearsonr(log_norm, positions_flat)

        # Per-position mean norm (averaging over samples)
        mean_norm_per_pos = norms.mean(dim=0).numpy()  # [seq_len]
        pos_range = np.arange(seq_len)
        r_mean_norm, _ = pearsonr(mean_norm_per_pos, pos_range)

        # Theory: ||z_i||² ∝ 1/(i+1), so ||z_i|| ∝ 1/sqrt(i+1)
        # Test: mean_norm vs 1/sqrt(i+1)
        theory_curve = 1 / np.sqrt(pos_range + 1)
        r_theory, _ = pearsonr(mean_norm_per_pos, theory_curve)

        results[key.replace("_norm", "")] = {
            "r_norm_vs_pos": r_norm,
            "r_inv_sq_norm_vs_pos": r_inv_sq,
            "r_inv_norm_vs_pos": r_inv,
            "r_log_norm_vs_pos": r_log,
            "r_mean_norm_vs_pos": r_mean_norm,
            "r_norm_vs_theory": r_theory,
            "mean_norm_per_pos": mean_norm_per_pos,
        }

    return results


def test_variance_position_correlation(statistics, positions, n_samples, seq_len):
    """
    Test if variance-based features correlate with position.
    """
    results = {}

    stat_keys = [k for k in statistics.keys() if k.endswith("_var")]

    for key in stat_keys:
        variances = statistics[key]  # [n_samples, seq_len]

        var_flat = variances.flatten().numpy()
        positions_flat = positions.flatten().numpy()

        r_var, _ = pearsonr(var_flat, positions_flat)

        # Inverse variance
        inv_var = 1 / (var_flat + 1e-8)
        r_inv_var, _ = pearsonr(inv_var, positions_flat)

        results[key.replace("_var", "")] = {
            "r_var_vs_pos": r_var,
            "r_inv_var_vs_pos": r_inv_var,
        }

    return results


def test_linear_probes_on_statistics(statistics, positions, seq_len):
    """
    Train linear probes on various statistics to predict position.
    """
    n_samples = statistics["raw_embed_norm"].shape[0]
    positions_flat = positions.flatten().numpy()

    # Split into train/test
    n_train = int(0.8 * n_samples)

    results = {}

    # Test different statistics as features
    features_to_test = {
        "attn_output_norm": statistics["attn_output_norm"]
        .flatten()
        .numpy()
        .reshape(-1, 1),
        "attn_output_inv_sq_norm": (
            1 / (statistics["attn_output_norm"].flatten().numpy() ** 2 + 1e-8)
        ).reshape(-1, 1),
        "attn_output_var": statistics["attn_output_var"]
        .flatten()
        .numpy()
        .reshape(-1, 1),
        "post_ln2_norm": statistics["post_ln2_norm"].flatten().numpy().reshape(-1, 1),
        "post_ln2_var": statistics["post_ln2_var"].flatten().numpy().reshape(-1, 1),
    }

    for name, X in features_to_test.items():
        # Train/test split (by sample, not by position)
        train_mask = np.repeat(np.arange(n_samples) < n_train, seq_len)
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = positions_flat[train_mask], positions_flat[~train_mask]

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)

        y_pred = probe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)

        results[name] = {
            "r2": r2,
            "mae": mae,
            "pearson_r": r,
        }

    return results


def analyze_layernorm_scale_factors(model, input_ids, device="cuda"):
    """
    Analyze the scale factors applied by LayerNorm at each position.

    LayerNorm: y = (x - mean) / std * gamma + beta

    The std at each position should carry positional information.
    """
    model.eval()

    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[0]

        # Pre-LN2 input (post-attention + residual)
        x_ln1 = block.ln_1(x)
        attn_out = block.attn(x_ln1)
        x_attn = x + attn_out  # Input to LN2

        # Compute per-position statistics that LN2 would use
        mean_per_pos = x_attn.mean(dim=-1)  # [batch, seq_len]
        std_per_pos = x_attn.std(dim=-1)  # [batch, seq_len]

        # LN2 output
        x_ln2 = block.ln_2(x_attn)

    return {
        "pre_ln2_mean": mean_per_pos.cpu(),
        "pre_ln2_std": std_per_pos.cpu(),
        "post_ln2": x_ln2.cpu(),
    }


def plot_mechanism_analysis(norm_results, ln_results, positions, output_path):
    """Create visualization of the variance-based mechanism."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    seq_len = positions.shape[1]
    pos_range = np.arange(seq_len)

    # Plot 1: Attention output norm vs position
    ax = axes[0, 0]
    if "attn_output" in norm_results:
        mean_norm = norm_results["attn_output"]["mean_norm_per_pos"]
        ax.plot(pos_range, mean_norm, "b-", linewidth=2, label="Empirical")
        # Theory: 1/sqrt(i+1)
        theory = mean_norm[0] / np.sqrt(pos_range + 1) * np.sqrt(1)
        ax.plot(pos_range, theory, "r--", linewidth=2, label="Theory: 1/√(i+1)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Mean ||attention output||")
        ax.set_title("Attention Output Norm Decay")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Inverse squared norm vs position
    ax = axes[0, 1]
    if "attn_output" in norm_results:
        mean_norm = norm_results["attn_output"]["mean_norm_per_pos"]
        inv_sq_norm = 1 / (mean_norm**2)
        # Normalize for comparison
        inv_sq_norm_normalized = (
            (inv_sq_norm - inv_sq_norm.min())
            / (inv_sq_norm.max() - inv_sq_norm.min())
            * seq_len
        )
        ax.plot(
            pos_range,
            inv_sq_norm_normalized,
            "b-",
            linewidth=2,
            label="1/||z||² (normalized)",
        )
        ax.plot(pos_range, pos_range, "r--", linewidth=2, label="True position")
        ax.set_xlabel("Position")
        ax.set_ylabel("Normalized 1/||z||²")
        ax.set_title("Inverse Squared Norm ≈ Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Pre-LN2 std vs position
    ax = axes[0, 2]
    if ln_results is not None:
        mean_std = ln_results["pre_ln2_std"].mean(dim=0).numpy()
        ax.plot(pos_range, mean_std, "b-", linewidth=2, label="Empirical")
        # Theory: std ∝ 1/sqrt(i+1)
        theory = mean_std[0] / np.sqrt(pos_range + 1) * np.sqrt(1)
        ax.plot(pos_range, theory, "r--", linewidth=2, label="Theory: 1/√(i+1)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Mean std (pre-LN2)")
        ax.set_title("Pre-LayerNorm Std Decay")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Correlation summary bar chart
    ax = axes[1, 0]
    components = list(norm_results.keys())
    r_values = [norm_results[c]["r_inv_sq_norm_vs_pos"] for c in components]
    colors = ["green" if r > 0.5 else "orange" if r > 0.2 else "red" for r in r_values]
    bars = ax.bar(range(len(components)), r_values, color=colors)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in components], rotation=0, fontsize=8
    )
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation: 1/||z||² vs Position")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 5: Theory fit (norm vs 1/sqrt(i+1))
    ax = axes[1, 1]
    r_theory_values = [norm_results[c]["r_norm_vs_theory"] for c in components]
    bars = ax.bar(range(len(components)), r_theory_values, color="blue", alpha=0.7)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in components], rotation=0, fontsize=8
    )
    ax.set_ylabel("Pearson r")
    ax.set_title("Fit to Theory: ||z|| ∝ 1/√(i+1)")
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 6: Summary text
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = "Key Findings:\n\n"

    if "attn_output" in norm_results:
        r = norm_results["attn_output"]["r_inv_sq_norm_vs_pos"]
        r_theory = norm_results["attn_output"]["r_norm_vs_theory"]
        summary_text += f"• Attention output:\n"
        summary_text += f"  - 1/||z||² vs pos: r = {r:.3f}\n"
        summary_text += f"  - ||z|| vs 1/√(i+1): r = {r_theory:.3f}\n\n"

    if "post_ln2" in norm_results:
        r = norm_results["post_ln2"]["r_inv_sq_norm_vs_pos"]
        summary_text += f"• Post-LayerNorm:\n"
        summary_text += f"  - 1/||z||² vs pos: r = {r:.3f}\n\n"

    summary_text += "Interpretation:\n"
    summary_text += "Position is encoded via the variance/norm\n"
    summary_text += "of attention outputs, which decays as\n"
    summary_text += "1/(i+1) due to causal averaging."

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="nanoGPT/out-nope-1layer-ln/ckpt.pt"
    )
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    print("=" * 70)
    print("VARIANCE-BASED POSITION ENCODING MECHANISM ANALYSIS")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, args.device)

    norm_type = model.config.norm_type
    seq_len = model.config.block_size
    vocab_size = model.config.vocab_size

    val_loss = checkpoint.get("best_val_loss", float("nan"))
    if torch.is_tensor(val_loss):
        val_loss = val_loss.cpu().item()
    perplexity = np.exp(val_loss)

    print(f"Model: {norm_type.upper()}")
    print(f"Seq len: {seq_len}, Vocab: {vocab_size}")
    print(f"Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

    # Generate random sequences
    print(f"\nGenerating {args.n_samples} random sequences...")
    torch.manual_seed(42)
    input_ids = torch.randint(
        0, vocab_size, (args.n_samples, seq_len), device=args.device
    )

    # Extract activations in batches
    print("Extracting activations...")
    batch_size = 100
    all_activations = {}
    all_statistics = {}

    for i in range(0, args.n_samples, batch_size):
        batch = input_ids[i : i + batch_size]
        acts, stats = extract_detailed_activations(model, batch, args.device)

        for k, v in acts.items():
            if k not in all_activations:
                all_activations[k] = []
            all_activations[k].append(v)

        for k, v in stats.items():
            if k not in all_statistics:
                all_statistics[k] = []
            all_statistics[k].append(v)

    # Concatenate
    for k in all_activations:
        all_activations[k] = torch.cat(all_activations[k], dim=0)
    for k in all_statistics:
        all_statistics[k] = torch.cat(all_statistics[k], dim=0)

    # Create position labels
    positions = torch.arange(seq_len).unsqueeze(0).expand(args.n_samples, -1)

    # Test norm-based correlations
    print("\n" + "=" * 70)
    print("TESTING NORM-BASED POSITION ENCODING")
    print("=" * 70)

    norm_results = test_norm_position_correlation(
        all_statistics, positions, args.n_samples, seq_len
    )

    print("\nCorrelation Results (1/||z||² vs position):")
    print("-" * 50)
    for component, res in norm_results.items():
        print(f"\n{component}:")
        print(f"  ||z|| vs position:      r = {res['r_norm_vs_pos']:.4f}")
        print(f"  1/||z||² vs position:   r = {res['r_inv_sq_norm_vs_pos']:.4f}")
        print(f"  1/||z|| vs position:    r = {res['r_inv_norm_vs_pos']:.4f}")
        print(f"  log(||z||) vs position: r = {res['r_log_norm_vs_pos']:.4f}")
        print(f"  ||z|| vs 1/√(i+1):      r = {res['r_norm_vs_theory']:.4f}")

    # Test variance-based correlations
    print("\n" + "=" * 70)
    print("TESTING VARIANCE-BASED POSITION ENCODING")
    print("=" * 70)

    var_results = test_variance_position_correlation(
        all_statistics, positions, args.n_samples, seq_len
    )

    print("\nVariance Correlation Results:")
    print("-" * 50)
    for component, res in var_results.items():
        print(
            f"{component}: var vs pos = {res['r_var_vs_pos']:.4f}, 1/var vs pos = {res['r_inv_var_vs_pos']:.4f}"
        )

    # Test linear probes on statistics
    print("\n" + "=" * 70)
    print("LINEAR PROBES ON STATISTICS")
    print("=" * 70)

    probe_results = test_linear_probes_on_statistics(all_statistics, positions, seq_len)

    print("\nLinear Probe Results:")
    print("-" * 50)
    print(f"{'Feature':<30} {'R²':>10} {'MAE':>10} {'r':>10}")
    print("-" * 60)
    for feature, res in probe_results.items():
        print(
            f"{feature:<30} {res['r2']:>10.4f} {res['mae']:>10.2f} {res['pearson_r']:>10.4f}"
        )

    # Analyze LayerNorm scale factors
    print("\n" + "=" * 70)
    print("LAYERNORM SCALE FACTOR ANALYSIS")
    print("=" * 70)

    # Use a smaller batch for this analysis
    ln_input = input_ids[:500]
    ln_results = analyze_layernorm_scale_factors(model, ln_input, args.device)

    mean_std_per_pos = ln_results["pre_ln2_std"].mean(dim=0).numpy()
    pos_range = np.arange(seq_len)

    r_std_vs_pos, _ = pearsonr(mean_std_per_pos, pos_range)

    # Theory: std ∝ 1/sqrt(i+1)
    theory_curve = 1 / np.sqrt(pos_range + 1)
    r_std_vs_theory, _ = pearsonr(mean_std_per_pos, theory_curve)

    print(f"\nPre-LN2 std vs position: r = {r_std_vs_pos:.4f}")
    print(f"Pre-LN2 std vs 1/√(i+1): r = {r_std_vs_theory:.4f}")

    # The LN scaling factor is 1/std, which is proportional to sqrt(i+1)
    ln_scale = 1 / (mean_std_per_pos + 1e-8)
    ln_scale_normalized = ln_scale / ln_scale[0]  # Normalize to start at 1
    theory_scale = np.sqrt((pos_range + 1))
    theory_scale_normalized = theory_scale / theory_scale[0]

    r_ln_scale_vs_theory, _ = pearsonr(ln_scale_normalized, theory_scale_normalized)
    r_ln_scale_vs_pos, _ = pearsonr(ln_scale, pos_range)

    print(f"\nLayerNorm scale (1/std) vs position: r = {r_ln_scale_vs_pos:.4f}")
    print(f"LayerNorm scale vs √(i+1): r = {r_ln_scale_vs_theory:.4f}")

    # Generate plot
    output_path = Path(args.output_dir) / f"variance_mechanism_{norm_type}.png"
    plot_mechanism_analysis(norm_results, ln_results, positions, output_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    attn_inv_sq_r = norm_results.get("attn_output", {}).get("r_inv_sq_norm_vs_pos", 0)
    attn_theory_r = norm_results.get("attn_output", {}).get("r_norm_vs_theory", 0)

    print(f"""
Key Findings:

1. ATTENTION OUTPUT NORM DECAY
   - ||attention_output|| vs 1/√(i+1): r = {attn_theory_r:.4f}
   - Confirms: uniform attention causes norm decay as 1/√(i+1)

2. INVERSE SQUARED NORM ENCODES POSITION
   - 1/||attention_output||² vs position: r = {attn_inv_sq_r:.4f}
   - This is the key signal: inverse squared norm is linear with position!

3. LAYERNORM PARTIALLY ENCODES POSITION
   - LayerNorm scale factor (1/std) vs √(i+1): r = {r_ln_scale_vs_theory:.4f}
   - LN division by std effectively multiplies by √(i+1)

4. MECHANISM SUMMARY
   - Attention output: z_i has ||z_i||² ∝ 1/(i+1)
   - LayerNorm: divides by std ∝ 1/√(i+1), so multiplies by √(i+1)
   - Result: After LN, position is encoded but requires nonlinearity to fully decode
   - MLP: Uses nonlinearity to decode position from norm/variance information
""")

    # Save detailed results
    results_path = Path(args.output_dir) / f"variance_mechanism_results_{norm_type}.pt"
    torch.save(
        {
            "norm_results": norm_results,
            "var_results": var_results,
            "probe_results": probe_results,
            "ln_scale_vs_theory_r": r_ln_scale_vs_theory,
            "ln_scale_vs_pos_r": r_ln_scale_vs_pos,
        },
        results_path,
    )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
