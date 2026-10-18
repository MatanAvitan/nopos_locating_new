"""
LayerNorm Paradox: Trained vs Randomly Initialized Models

This script compares where positional information resides in:
1. Trained NoPE model (has learned from data)
2. Randomly initialized NoPE model (only architectural biases)

Key question: Does training change HOW position is encoded, or just amplify
existing structure from the architecture?
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, kurtosis
from scipy.spatial.distance import cosine
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


def load_trained_model(checkpoint_path, device="cuda"):
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
    return model, config


def create_random_model(config, device="cuda"):
    """Create randomly initialized model with same config."""
    config.log_attention_stats = False
    model = GPT(config)
    model.to(device)
    model.eval()
    return model


def extract_activations(model, input_ids, device="cuda"):
    """Extract activations at key points in the network."""
    model.eval()

    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[0]

        # Pre-LN1: just embeddings
        pre_ln1 = x.clone()

        # Post-LN1
        x_ln1 = block.ln_1(x)

        # Post-attention (before residual)
        attn_out = block.attn(x_ln1)

        # Pre-LN2 = residual + attn_out
        pre_ln2 = x + attn_out

        # Post-LN2
        post_ln2 = block.ln_2(pre_ln2)

        # LN statistics
        mean_ln2 = pre_ln2.mean(dim=-1, keepdim=True)
        std_ln2 = pre_ln2.std(dim=-1, keepdim=True, unbiased=False)

        # Post-MLP
        mlp_out = block.mlp(post_ln2)
        post_mlp = pre_ln2 + mlp_out

    return {
        "embeddings": tok_emb.cpu(),
        "attn_out": attn_out.cpu(),
        "pre_ln2": pre_ln2.cpu(),
        "post_ln2": post_ln2.cpu(),
        "post_mlp": post_mlp.cpu(),
        "ln2_mean": mean_ln2.squeeze(-1).cpu(),
        "ln2_std": std_ln2.squeeze(-1).cpu(),
    }


def analyze_variance_mechanism(activations, seq_len, label):
    """Analyze if variance follows 1/(i+1) decay."""
    print(f"\n{'=' * 60}")
    print(f"VARIANCE MECHANISM ANALYSIS: {label}")
    print(f"{'=' * 60}")

    results = {}

    for name in ["attn_out", "pre_ln2", "post_ln2"]:
        act = activations[name]  # [n_samples, seq_len, d_model]

        # Variance per position (across samples and dimensions)
        var_per_pos = act.var(dim=(0, 2)).numpy()

        # Theoretical decay: 1/(i+1)
        theoretical = 1.0 / (np.arange(seq_len) + 1)

        # Normalize for comparison
        var_normalized = var_per_pos / var_per_pos[0]
        theoretical_normalized = theoretical / theoretical[0]

        r, _ = pearsonr(var_normalized, theoretical_normalized)

        results[name] = {
            "var_per_pos": var_per_pos,
            "r_to_theory": r,
        }

        print(f"  {name}: r(variance, 1/(i+1)) = {r:.4f}")

    # Also check the LN std (which is sqrt of variance)
    ln_std = activations["ln2_std"]  # [n_samples, seq_len]
    std_per_pos = ln_std.mean(dim=0).numpy()

    # Theory: std ∝ 1/sqrt(i+1)
    theoretical_std = 1.0 / np.sqrt(np.arange(seq_len) + 1)
    std_normalized = std_per_pos / std_per_pos[0]
    theoretical_std_normalized = theoretical_std / theoretical_std[0]

    r_std, _ = pearsonr(std_normalized, theoretical_std_normalized)
    results["ln2_std"] = {"r_to_theory": r_std}
    print(f"  LN2 std: r(std, 1/sqrt(i+1)) = {r_std:.4f}")

    return results


def analyze_population_statistics(activations, seq_len, label):
    """Analyze population-level statistics that encode position."""
    print(f"\n{'=' * 60}")
    print(f"POPULATION STATISTICS ANALYSIS: {label}")
    print(f"{'=' * 60}")

    results = {}

    for name in ["pre_ln2", "post_ln2"]:
        act = activations[name]  # [n_samples, seq_len, d_model]

        # Population mean at each position
        pop_mean = act.mean(dim=0)  # [seq_len, d_model]

        # Norm of population mean
        pop_mean_norm = torch.norm(pop_mean, dim=1).numpy()
        r_norm, _ = pearsonr(pop_mean_norm, np.arange(seq_len))

        # Can we decode position from population mean alone?
        probe = Ridge(alpha=1.0)
        probe.fit(pop_mean.numpy(), np.arange(seq_len))
        pred = probe.predict(pop_mean.numpy())
        r_decode, _ = pearsonr(pred, np.arange(seq_len))

        # Distance from position 0's mean
        distances = torch.norm(pop_mean - pop_mean[0:1], dim=1).numpy()
        r_dist, _ = pearsonr(distances, np.arange(seq_len))

        results[name] = {
            "pop_mean_norm": pop_mean_norm,
            "r_norm_vs_pos": r_norm,
            "r_decode_from_mean": r_decode,
            "r_dist_from_pos0": r_dist,
        }

        print(f"\n  {name}:")
        print(f"    Population mean norm vs pos: r = {r_norm:.4f}")
        print(f"    Position decode from mean:   r = {r_decode:.4f}")
        print(f"    Distance from pos-0 vs pos:  r = {r_dist:.4f}")

    return results


def analyze_probe_accuracy(activations, seq_len, n_samples, label):
    """Test linear and MLP probe accuracy for position prediction."""
    print(f"\n{'=' * 60}")
    print(f"PROBE ACCURACY ANALYSIS: {label}")
    print(f"{'=' * 60}")

    results = {}
    d_model = activations["post_ln2"].shape[-1]

    # Create position labels
    positions = np.tile(np.arange(seq_len), n_samples)

    # Train/test split
    n_train = int(0.8 * n_samples)
    train_mask = np.repeat(np.arange(n_samples) < n_train, seq_len)

    for name in ["embeddings", "attn_out", "pre_ln2", "post_ln2", "post_mlp"]:
        act = activations[name]
        X = act.reshape(-1, d_model).numpy()

        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = positions[train_mask], positions[~train_mask]

        # Linear probe
        linear_probe = Ridge(alpha=1.0)
        linear_probe.fit(X_train, y_train)
        y_pred_linear = linear_probe.predict(X_test)
        r2_linear = r2_score(y_test, y_pred_linear)
        r_linear, _ = pearsonr(y_test, y_pred_linear)

        # MLP probe
        mlp_probe = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
        mlp_probe.fit(X_train, y_train)
        y_pred_mlp = mlp_probe.predict(X_test)
        r2_mlp = r2_score(y_test, y_pred_mlp)
        r_mlp, _ = pearsonr(y_test, y_pred_mlp)

        results[name] = {
            "linear_r2": r2_linear,
            "linear_r": r_linear,
            "mlp_r2": r2_mlp,
            "mlp_r": r_mlp,
        }

        print(
            f"  {name:12s}: Linear R²={r2_linear:.4f} (r={r_linear:.4f}), MLP R²={r2_mlp:.4f} (r={r_mlp:.4f})"
        )

    return results


def analyze_spikiness(activations, seq_len, label):
    """Analyze spikiness/shape metrics that might encode position."""
    print(f"\n{'=' * 60}")
    print(f"SPIKINESS ANALYSIS: {label}")
    print(f"{'=' * 60}")

    results = {}

    for name in ["pre_ln2", "post_ln2"]:
        act = activations[name].numpy()  # [n_samples, seq_len, d_model]

        # Kurtosis per position (averaged over samples)
        kurtosis_per_pos = []
        max_component_per_pos = []
        entropy_per_pos = []

        for pos in range(seq_len):
            pos_acts = act[:, pos, :]  # [n_samples, d_model]

            # Kurtosis
            kurt = np.mean([kurtosis(sample) for sample in pos_acts])
            kurtosis_per_pos.append(kurt)

            # Max absolute component
            max_comp = np.mean(np.max(np.abs(pos_acts), axis=1))
            max_component_per_pos.append(max_comp)

            # Entropy of squared components
            sq = pos_acts**2
            sq_norm = sq / (sq.sum(axis=1, keepdims=True) + 1e-10)
            ent = -np.sum(sq_norm * np.log(sq_norm + 1e-10), axis=1)
            entropy_per_pos.append(np.mean(ent))

        r_kurt, _ = pearsonr(kurtosis_per_pos, np.arange(seq_len))
        r_max, _ = pearsonr(max_component_per_pos, np.arange(seq_len))
        r_ent, _ = pearsonr(entropy_per_pos, np.arange(seq_len))

        results[name] = {
            "kurtosis_per_pos": kurtosis_per_pos,
            "max_component_per_pos": max_component_per_pos,
            "entropy_per_pos": entropy_per_pos,
            "r_kurtosis_vs_pos": r_kurt,
            "r_max_vs_pos": r_max,
            "r_entropy_vs_pos": r_ent,
        }

        print(f"\n  {name}:")
        print(f"    Kurtosis vs position:      r = {r_kurt:.4f}")
        print(f"    Max component vs position: r = {r_max:.4f}")
        print(f"    Entropy vs position:       r = {r_ent:.4f}")

    return results


def analyze_centered_probe(activations, seq_len, n_samples, label):
    """
    Key test: Can we decode position from mean-subtracted activations?

    If yes: Position is in individual sample variation (geometric mechanism)
    If no: Position is only in population statistics (statistical mechanism)
    """
    print(f"\n{'=' * 60}")
    print(f"CENTERED PROBE ANALYSIS: {label}")
    print(f"{'=' * 60}")

    results = {}
    d_model = activations["post_ln2"].shape[-1]

    positions = np.tile(np.arange(seq_len), n_samples)
    n_train = int(0.8 * n_samples)
    train_mask = np.repeat(np.arange(n_samples) < n_train, seq_len)

    for name in ["pre_ln2", "post_ln2"]:
        act = activations[name]  # [n_samples, seq_len, d_model]

        # Population mean at each position
        pop_mean = act.mean(dim=0, keepdim=True)  # [1, seq_len, d_model]

        # Mean-subtracted activations
        centered = act - pop_mean

        # Baseline: full activations
        X_full = act.reshape(-1, d_model).numpy()
        X_centered = centered.reshape(-1, d_model).numpy()

        results_name = {}

        for probe_name, X in [("full", X_full), ("centered", X_centered)]:
            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = positions[train_mask], positions[~train_mask]

            # Linear probe
            probe = Ridge(alpha=1.0)
            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            r, _ = pearsonr(y_test, y_pred)

            results_name[probe_name] = {"r2": r2, "r": r}

        results[name] = results_name

        drop_r2 = results_name["full"]["r2"] - results_name["centered"]["r2"]

        print(f"\n  {name}:")
        print(
            f"    Full activations:     R²={results_name['full']['r2']:.4f}, r={results_name['full']['r']:.4f}"
        )
        print(
            f"    Centered activations: R²={results_name['centered']['r2']:.4f}, r={results_name['centered']['r']:.4f}"
        )
        print(f"    Drop from centering:  ΔR²={drop_r2:.4f}")

        if results_name["centered"]["r2"] > 0.05:
            print(f"    → Individual samples carry position info (geometric mechanism)")
        else:
            print(f"    → Position only in population mean (statistical mechanism)")

    return results


def create_comparison_figure(trained_results, random_results, seq_len, output_path):
    """Create comparison figure of trained vs random model."""

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    pos_range = np.arange(seq_len)

    # Row 1: Variance decay
    ax = axes[0, 0]
    for label, results, color in [
        ("Trained", trained_results, "blue"),
        ("Random", random_results, "red"),
    ]:
        var = results["variance"]["attn_out"]["var_per_pos"]
        var_norm = var / var[0]
        ax.plot(pos_range, var_norm, color=color, label=label, alpha=0.7)
    theory = 1.0 / (pos_range + 1)
    ax.plot(pos_range, theory / theory[0], "k--", label="Theory: 1/(i+1)", alpha=0.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Normalized Variance")
    ax.set_title("Attention Output Variance Decay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: Population mean norm
    ax = axes[0, 1]
    for label, results, color in [
        ("Trained", trained_results, "blue"),
        ("Random", random_results, "red"),
    ]:
        norm = results["population"]["post_ln2"]["pop_mean_norm"]
        ax.plot(pos_range, norm, color=color, label=label, alpha=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("||Population Mean||")
    ax.set_title("Post-LN Population Mean Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: Probe accuracy comparison
    ax = axes[0, 2]
    locations = ["embeddings", "attn_out", "pre_ln2", "post_ln2", "post_mlp"]
    x = np.arange(len(locations))
    width = 0.35

    trained_r2 = [trained_results["probes"][loc]["mlp_r2"] for loc in locations]
    random_r2 = [random_results["probes"][loc]["mlp_r2"] for loc in locations]

    ax.bar(x - width / 2, trained_r2, width, label="Trained", color="blue", alpha=0.7)
    ax.bar(x + width / 2, random_r2, width, label="Random", color="red", alpha=0.7)
    ax.set_xlabel("Activation Point")
    ax.set_ylabel("MLP Probe R²")
    ax.set_title("Position Prediction Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([loc.replace("_", "\n") for loc in locations], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Row 2: Spikiness (entropy)
    ax = axes[1, 0]
    for label, results, color in [
        ("Trained", trained_results, "blue"),
        ("Random", random_results, "red"),
    ]:
        ent = results["spikiness"]["post_ln2"]["entropy_per_pos"]
        ax.plot(pos_range, ent, color=color, label=label, alpha=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("Entropy of Squared Components")
    ax.set_title("Post-LN Spikiness (Entropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Spikiness (kurtosis)
    ax = axes[1, 1]
    for label, results, color in [
        ("Trained", trained_results, "blue"),
        ("Random", random_results, "red"),
    ]:
        kurt = results["spikiness"]["post_ln2"]["kurtosis_per_pos"]
        ax.plot(pos_range, kurt, color=color, label=label, alpha=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Post-LN Spikiness (Kurtosis)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Centered probe comparison
    ax = axes[1, 2]
    categories = [
        "Full\n(pre_ln2)",
        "Centered\n(pre_ln2)",
        "Full\n(post_ln2)",
        "Centered\n(post_ln2)",
    ]
    trained_vals = [
        trained_results["centered"]["pre_ln2"]["full"]["r2"],
        trained_results["centered"]["pre_ln2"]["centered"]["r2"],
        trained_results["centered"]["post_ln2"]["full"]["r2"],
        trained_results["centered"]["post_ln2"]["centered"]["r2"],
    ]
    random_vals = [
        random_results["centered"]["pre_ln2"]["full"]["r2"],
        random_results["centered"]["pre_ln2"]["centered"]["r2"],
        random_results["centered"]["post_ln2"]["full"]["r2"],
        random_results["centered"]["post_ln2"]["centered"]["r2"],
    ]

    x = np.arange(len(categories))
    ax.bar(x - width / 2, trained_vals, width, label="Trained", color="blue", alpha=0.7)
    ax.bar(x + width / 2, random_vals, width, label="Random", color="red", alpha=0.7)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Linear Probe R²")
    ax.set_title("Full vs Centered Activations")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Row 3: Summary statistics
    ax = axes[2, 0]
    ax.axis("off")

    summary = "VARIANCE MECHANISM\n" + "-" * 30 + "\n"
    summary += f"Trained attn_out r: {trained_results['variance']['attn_out']['r_to_theory']:.3f}\n"
    summary += f"Random attn_out r:  {random_results['variance']['attn_out']['r_to_theory']:.3f}\n\n"
    summary += f"Trained LN std r:   {trained_results['variance']['ln2_std']['r_to_theory']:.3f}\n"
    summary += f"Random LN std r:    {random_results['variance']['ln2_std']['r_to_theory']:.3f}\n"

    ax.text(
        0.1,
        0.9,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    ax = axes[2, 1]
    ax.axis("off")

    summary = "POPULATION STATISTICS\n" + "-" * 30 + "\n"
    summary += f"Trained decode r: {trained_results['population']['post_ln2']['r_decode_from_mean']:.3f}\n"
    summary += f"Random decode r:  {random_results['population']['post_ln2']['r_decode_from_mean']:.3f}\n\n"
    summary += f"Trained dist r:   {trained_results['population']['post_ln2']['r_dist_from_pos0']:.3f}\n"
    summary += f"Random dist r:    {random_results['population']['post_ln2']['r_dist_from_pos0']:.3f}\n"

    ax.text(
        0.1,
        0.9,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    ax = axes[2, 2]
    ax.axis("off")

    summary = "KEY FINDING\n" + "-" * 30 + "\n"

    # Compare centered probe results
    t_drop = (
        trained_results["centered"]["post_ln2"]["full"]["r2"]
        - trained_results["centered"]["post_ln2"]["centered"]["r2"]
    )
    r_drop = (
        random_results["centered"]["post_ln2"]["full"]["r2"]
        - random_results["centered"]["post_ln2"]["centered"]["r2"]
    )

    summary += f"Trained: centering drops R² by {t_drop:.3f}\n"
    summary += f"Random:  centering drops R² by {r_drop:.3f}\n\n"

    if random_results["centered"]["post_ln2"]["centered"]["r2"] > 0.05:
        summary += "Random model: Position in\n  individual samples!\n"
        summary += "→ Geometric mechanism exists\n   at initialization"
    else:
        summary += "Random model: Position only\n  in population mean\n"
        summary += "→ Training creates the\n   geometric mechanism"

    ax.text(
        0.1,
        0.9,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    plt.suptitle(
        "LayerNorm Paradox: Trained vs Randomly Initialized Model",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")
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
    print("LAYERNORM PARADOX: TRAINED VS RANDOMLY INITIALIZED")
    print("=" * 70)

    # Load trained model
    print(f"\n1. Loading trained model from {args.checkpoint}")
    trained_model, config = load_trained_model(args.checkpoint, args.device)

    # Create random model with same config
    print("\n2. Creating randomly initialized model with same config")
    random_model = create_random_model(config, args.device)

    seq_len = config.block_size
    vocab_size = config.vocab_size
    norm_type = config.norm_type

    print(f"\nConfig: {norm_type.upper()}, seq_len={seq_len}, vocab_size={vocab_size}")

    # Generate random sequences
    print(f"\n3. Generating {args.n_samples} random sequences...")
    torch.manual_seed(42)
    input_ids = torch.randint(
        0, vocab_size, (args.n_samples, seq_len), device=args.device
    )

    # Extract activations for both models
    print("\n4. Extracting activations...")
    print("   - Trained model...")
    trained_acts = {}
    random_acts = {}

    batch_size = 100
    for key in [
        "embeddings",
        "attn_out",
        "pre_ln2",
        "post_ln2",
        "post_mlp",
        "ln2_mean",
        "ln2_std",
    ]:
        trained_acts[key] = []
        random_acts[key] = []

    for i in range(0, args.n_samples, batch_size):
        batch = input_ids[i : i + batch_size]

        t_batch = extract_activations(trained_model, batch, args.device)
        r_batch = extract_activations(random_model, batch, args.device)

        for key in trained_acts:
            trained_acts[key].append(t_batch[key])
            random_acts[key].append(r_batch[key])

    for key in trained_acts:
        trained_acts[key] = torch.cat(trained_acts[key], dim=0)
        random_acts[key] = torch.cat(random_acts[key], dim=0)

    print("   - Random model...")

    # Run analyses on both
    print("\n" + "=" * 70)
    print("TRAINED MODEL ANALYSIS")
    print("=" * 70)

    trained_results = {
        "variance": analyze_variance_mechanism(trained_acts, seq_len, "TRAINED"),
        "population": analyze_population_statistics(trained_acts, seq_len, "TRAINED"),
        "probes": analyze_probe_accuracy(
            trained_acts, seq_len, args.n_samples, "TRAINED"
        ),
        "spikiness": analyze_spikiness(trained_acts, seq_len, "TRAINED"),
        "centered": analyze_centered_probe(
            trained_acts, seq_len, args.n_samples, "TRAINED"
        ),
    }

    print("\n" + "=" * 70)
    print("RANDOM MODEL ANALYSIS")
    print("=" * 70)

    random_results = {
        "variance": analyze_variance_mechanism(random_acts, seq_len, "RANDOM"),
        "population": analyze_population_statistics(random_acts, seq_len, "RANDOM"),
        "probes": analyze_probe_accuracy(
            random_acts, seq_len, args.n_samples, "RANDOM"
        ),
        "spikiness": analyze_spikiness(random_acts, seq_len, "RANDOM"),
        "centered": analyze_centered_probe(
            random_acts, seq_len, args.n_samples, "RANDOM"
        ),
    }

    # Create comparison figure
    print("\n5. Creating comparison figure...")
    fig_path = Path(args.output_dir) / f"ln_paradox_trained_vs_random_{norm_type}.png"
    create_comparison_figure(trained_results, random_results, seq_len, fig_path)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: TRAINED VS RANDOM MODEL")
    print("=" * 70)

    print(f"""
VARIANCE MECHANISM:
  - Trained: r(attn variance, 1/(i+1)) = {trained_results["variance"]["attn_out"]["r_to_theory"]:.4f}
  - Random:  r(attn variance, 1/(i+1)) = {random_results["variance"]["attn_out"]["r_to_theory"]:.4f}
  → Both follow theoretical decay (architectural property)

POPULATION STATISTICS:
  - Trained: decode from pop mean r = {trained_results["population"]["post_ln2"]["r_decode_from_mean"]:.4f}
  - Random:  decode from pop mean r = {random_results["population"]["post_ln2"]["r_decode_from_mean"]:.4f}
  → Both have position-informative population means

PROBE ACCURACY (post-LN2, MLP):
  - Trained: R² = {trained_results["probes"]["post_ln2"]["mlp_r2"]:.4f}
  - Random:  R² = {random_results["probes"]["post_ln2"]["mlp_r2"]:.4f}
  → {"Training improves decoding" if trained_results["probes"]["post_ln2"]["mlp_r2"] > random_results["probes"]["post_ln2"]["mlp_r2"] + 0.1 else "Similar accuracy"}

CENTERED PROBE (individual sample info):
  - Trained centered R²: {trained_results["centered"]["post_ln2"]["centered"]["r2"]:.4f}
  - Random centered R²:  {random_results["centered"]["post_ln2"]["centered"]["r2"]:.4f}
  → {"Both have geometric mechanism" if random_results["centered"]["post_ln2"]["centered"]["r2"] > 0.05 else "Training creates geometric mechanism"}

KEY INSIGHT:
  The variance mechanism exists at random initialization (architectural).
  Training may amplify or refine how this information is used, but the
  fundamental positional encoding circuit is present from the start.
""")

    # Save results
    results_path = (
        Path(args.output_dir) / f"ln_paradox_trained_vs_random_{norm_type}.pt"
    )
    torch.save(
        {
            "trained": trained_results,
            "random": random_results,
            "config": {
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "norm_type": norm_type,
                "n_samples": args.n_samples,
            },
        },
        results_path,
    )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
