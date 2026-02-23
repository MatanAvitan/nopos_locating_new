"""
Norm Intervention Experiment: Causal Verification that Norm Encodes Position

This script performs causal interventions on activation norms to verify that
norm is the mechanism by which position is encoded in NoPE transformers.

Key Experiments:
1. Norm equalization: Set all positions to same norm, check if position decoding breaks
2. Norm swapping: Swap norms between positions, check if predictions follow norm
3. Norm scaling: Scale norms to simulate different positions
4. Norm isolation: Keep only norm information, zero out direction
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Add parent directory for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "nanoGPT"))

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


def get_activations(model, tokens, layer_name="post_attn_residual"):
    """Get activations at a specific layer."""
    with torch.no_grad():
        # Embedding
        tok_emb = model.transformer.wte(tokens)

        if layer_name == "embed":
            return tok_emb

        # Through first layer
        ln1 = model.transformer.h[0].ln_1
        attn = model.transformer.h[0].attn
        ln2 = model.transformer.h[0].ln_2
        mlp = model.transformer.h[0].mlp

        post_ln1 = ln1(tok_emb)
        if layer_name == "post_ln1":
            return post_ln1

        post_attn = attn(post_ln1)
        if layer_name == "post_attn":
            return post_attn

        post_attn_residual = tok_emb + post_attn
        if layer_name == "post_attn_residual":
            return post_attn_residual

        post_ln2 = ln2(post_attn_residual)
        if layer_name == "post_ln2":
            return post_ln2

        post_mlp = mlp(post_ln2)
        post_mlp_residual = post_attn_residual + post_mlp
        if layer_name == "post_mlp_residual":
            return post_mlp_residual

    raise ValueError(f"Unknown layer: {layer_name}")


def train_position_probe(activations, positions, test_size=0.2):
    """Train a linear probe to predict position from activations."""
    X = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
    y = np.tile(positions, activations.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)

    r2 = probe.score(X_test, y_test)
    predictions = probe.predict(X_test)
    mae = np.mean(np.abs(predictions - y_test))

    return probe, r2, mae


def experiment_norm_equalization(model, n_samples=500, block_size=64):
    """
    Experiment 1: Equalize norms across all positions.
    If norm encodes position, equalizing norms should destroy position information.
    """
    print("\n=== Experiment 1: Norm Equalization ===")
    results = {}

    vocab_size = model.config.vocab_size
    tokens = torch.randint(0, vocab_size, (n_samples, block_size), device=device)
    positions = np.arange(block_size)

    # Get activations at post_attn_residual (where norm is strongest signal)
    activations = get_activations(model, tokens, "post_attn_residual")

    # Train baseline probe
    probe, baseline_r2, baseline_mae = train_position_probe(activations, positions)
    print(f"Baseline R²: {baseline_r2:.4f}, MAE: {baseline_mae:.2f}")

    # Equalize norms to mean norm
    norms = activations.norm(dim=-1, keepdim=True)  # (batch, seq, 1)
    mean_norm = norms.mean()
    equalized = activations * (mean_norm / norms)

    # Evaluate probe on equalized activations
    X_eq = equalized.reshape(-1, equalized.shape[-1]).cpu().numpy()
    y = np.tile(positions, n_samples)
    preds_eq = probe.predict(X_eq)
    r2_eq = 1 - np.sum((preds_eq - y) ** 2) / np.sum((y - y.mean()) ** 2)
    mae_eq = np.mean(np.abs(preds_eq - y))

    print(f"After norm equalization R²: {r2_eq:.4f}, MAE: {mae_eq:.2f}")
    print(
        f"R² drop: {baseline_r2 - r2_eq:.4f} ({100 * (baseline_r2 - r2_eq) / baseline_r2:.1f}% relative)"
    )

    # Train new probe on equalized data
    _, r2_eq_retrain, mae_eq_retrain = train_position_probe(equalized, positions)
    print(f"Retrained on equalized R²: {r2_eq_retrain:.4f}, MAE: {mae_eq_retrain:.2f}")

    results["baseline"] = {"r2": float(baseline_r2), "mae": float(baseline_mae)}
    results["equalized_same_probe"] = {"r2": float(r2_eq), "mae": float(mae_eq)}
    results["equalized_retrained"] = {
        "r2": float(r2_eq_retrain),
        "mae": float(mae_eq_retrain),
    }
    results["r2_drop"] = float(baseline_r2 - r2_eq)
    results["r2_drop_relative"] = (
        float((baseline_r2 - r2_eq) / baseline_r2) if baseline_r2 > 0 else 0
    )

    return results


def experiment_norm_swapping(model, n_samples=500, block_size=64):
    """
    Experiment 2: Swap norms between positions.
    If norm encodes position, probe predictions should follow the swapped norms.
    """
    print("\n=== Experiment 2: Norm Swapping ===")
    results = {}

    vocab_size = model.config.vocab_size
    tokens = torch.randint(0, vocab_size, (n_samples, block_size), device=device)
    positions = np.arange(block_size)

    activations = get_activations(model, tokens, "post_attn_residual")

    # Train baseline probe
    probe, baseline_r2, baseline_mae = train_position_probe(activations, positions)

    # Get norms
    norms = activations.norm(dim=-1, keepdim=True)  # (batch, seq, 1)
    directions = activations / norms

    # Create swapped version: position i gets norm from position (63-i)
    swapped_indices = torch.arange(block_size - 1, -1, -1, device=device)
    swapped_norms = norms[:, swapped_indices, :]
    swapped_activations = directions * swapped_norms

    # Evaluate with swapped norms
    X_swap = (
        swapped_activations.reshape(-1, swapped_activations.shape[-1]).cpu().numpy()
    )
    y_original = np.tile(positions, n_samples)
    y_swapped = np.tile(positions[::-1], n_samples)  # Position that the norm came from

    preds_swap = probe.predict(X_swap)

    # Does the probe predict the original position or the swapped position?
    corr_with_original, _ = stats.pearsonr(preds_swap, y_original)
    corr_with_swapped, _ = stats.pearsonr(preds_swap, y_swapped)

    mae_original = np.mean(np.abs(preds_swap - y_original))
    mae_swapped = np.mean(np.abs(preds_swap - y_swapped))

    print(f"Correlation with original position: {corr_with_original:.4f}")
    print(f"Correlation with swapped position (norm source): {corr_with_swapped:.4f}")
    print(f"MAE to original: {mae_original:.2f}, MAE to swapped: {mae_swapped:.2f}")

    follows_norm = corr_with_swapped > corr_with_original
    print(
        f"Predictions follow {'NORM SOURCE' if follows_norm else 'ORIGINAL POSITION'}"
    )

    results["baseline_r2"] = float(baseline_r2)
    results["corr_with_original"] = float(corr_with_original)
    results["corr_with_swapped"] = float(corr_with_swapped)
    results["mae_to_original"] = float(mae_original)
    results["mae_to_swapped"] = float(mae_swapped)
    results["follows_norm"] = bool(follows_norm)

    return results


def experiment_norm_only(model, n_samples=500, block_size=64):
    """
    Experiment 3: Can we decode position from norm alone?
    Compare norm-only vs direction-only vs full activations.
    """
    print("\n=== Experiment 3: Norm-Only Decoding ===")
    results = {}

    vocab_size = model.config.vocab_size
    tokens = torch.randint(0, vocab_size, (n_samples, block_size), device=device)
    positions = np.arange(block_size)

    layers = [
        "embed",
        "post_ln1",
        "post_attn",
        "post_attn_residual",
        "post_ln2",
        "post_mlp_residual",
    ]

    for layer in layers:
        activations = get_activations(model, tokens, layer)
        norms = activations.norm(dim=-1)  # (batch, seq)
        directions = activations / norms.unsqueeze(-1)  # Unit vectors

        # Full activations
        _, r2_full, _ = train_position_probe(activations, positions)

        # Norm only (reshape for sklearn)
        X_norm = norms.reshape(-1, 1).cpu().numpy()
        y = np.tile(positions, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=42
        )
        probe_norm = Ridge(alpha=1.0)
        probe_norm.fit(X_train, y_train)
        r2_norm = probe_norm.score(X_test, y_test)

        # Direction only
        _, r2_direction, _ = train_position_probe(directions, positions)

        print(
            f"{layer}: Full R²={r2_full:.4f}, Norm R²={r2_norm:.4f}, Direction R²={r2_direction:.4f}"
        )

        results[layer] = {
            "full_r2": float(r2_full),
            "norm_r2": float(r2_norm),
            "direction_r2": float(r2_direction),
        }

    return results


def experiment_targeted_norm_change(model, n_samples=500, block_size=64):
    """
    Experiment 4: Change norm at specific positions and observe effect.
    Make position 32's norm look like position 0's norm.
    """
    print("\n=== Experiment 4: Targeted Norm Change ===")
    results = {}

    vocab_size = model.config.vocab_size
    tokens = torch.randint(0, vocab_size, (n_samples, block_size), device=device)
    positions = np.arange(block_size)

    activations = get_activations(model, tokens, "post_attn_residual")
    probe, _, _ = train_position_probe(activations, positions)

    # Get mean norms by position
    norms = activations.norm(dim=-1)  # (batch, seq)
    mean_norms_by_pos = norms.mean(dim=0)  # (seq,)

    # Test positions
    test_cases = [
        {"source_pos": 0, "target_pos": 32},
        {"source_pos": 0, "target_pos": 63},
        {"source_pos": 32, "target_pos": 0},
        {"source_pos": 63, "target_pos": 0},
    ]

    for case in test_cases:
        source_pos = case["source_pos"]
        target_pos = case["target_pos"]

        # Modify activations: give target_pos the norm of source_pos
        modified = activations.clone()
        target_norms = modified[:, target_pos, :].norm(dim=-1, keepdim=True)
        source_norm = mean_norms_by_pos[source_pos].item()

        # Scale target position to have source position's norm
        scale_factor = source_norm / target_norms
        modified[:, target_pos, :] = modified[:, target_pos, :] * scale_factor

        # Predict for the modified position
        X_target = modified[:, target_pos, :].cpu().numpy()
        preds = probe.predict(X_target)

        mean_pred = preds.mean()
        std_pred = preds.std()

        print(
            f"Position {target_pos} with norm of pos {source_pos}: "
            f"Predicted {mean_pred:.1f} ± {std_pred:.1f} (actual: {target_pos})"
        )

        case["mean_prediction"] = float(mean_pred)
        case["std_prediction"] = float(std_pred)
        case["expected_if_norm_causal"] = float(source_pos)
        case["actual_position"] = float(target_pos)
        case["prediction_closer_to"] = (
            "source"
            if abs(mean_pred - source_pos) < abs(mean_pred - target_pos)
            else "target"
        )

    results["test_cases"] = test_cases
    results["mean_norms_by_pos"] = mean_norms_by_pos.cpu().numpy().tolist()

    return results


def experiment_synthetic_norm_positions(model, n_samples=500, block_size=64):
    """
    Experiment 5: Create synthetic activations with controlled norms.
    Test if synthetic norms produce expected position predictions.
    """
    print("\n=== Experiment 5: Synthetic Norm Positions ===")
    results = {}

    vocab_size = model.config.vocab_size
    tokens = torch.randint(0, vocab_size, (n_samples, block_size), device=device)
    positions = np.arange(block_size)

    activations = get_activations(model, tokens, "post_attn_residual")
    probe, r2, _ = train_position_probe(activations, positions)

    # Get the expected norm profile
    norms = activations.norm(dim=-1)
    mean_norms_by_pos = norms.mean(dim=0).cpu().numpy()

    # Create synthetic activations with specific norm profiles
    directions = activations / norms.unsqueeze(-1)

    # Fit the norm-position relationship
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(positions.reshape(-1, 1), mean_norms_by_pos)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    print(
        f"Norm-position relationship: norm = {intercept:.4f} + {slope:.4f} * position"
    )

    # Predict positions from arbitrary norms
    synthetic_norms = [15.0, 10.0, 5.0, 3.0, 25.0]
    expected_positions = [(n - intercept) / slope for n in synthetic_norms]

    print("\nSynthetic norm -> Expected position mapping:")
    for norm, exp_pos in zip(synthetic_norms, expected_positions):
        print(f"  Norm {norm:.1f} -> Expected position {exp_pos:.1f}")

    # Create activations with these specific norms
    synthetic_results = []
    for target_norm, expected_pos in zip(synthetic_norms, expected_positions):
        # Use directions from random positions, apply target norm
        synthetic = directions * target_norm
        X_syn = synthetic.reshape(-1, synthetic.shape[-1]).cpu().numpy()
        preds = probe.predict(X_syn)

        mean_pred = preds.mean()
        std_pred = preds.std()

        print(
            f"Norm {target_norm:.1f}: Predicted {mean_pred:.1f} ± {std_pred:.1f} (expected: {expected_pos:.1f})"
        )

        synthetic_results.append(
            {
                "target_norm": float(target_norm),
                "expected_position": float(expected_pos),
                "mean_prediction": float(mean_pred),
                "std_prediction": float(std_pred),
            }
        )

    results["norm_position_slope"] = float(slope)
    results["norm_position_intercept"] = float(intercept)
    results["synthetic_results"] = synthetic_results

    return results


def create_plots(all_results, save_dir):
    """Create visualization plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Norm equalization effect
    if "norm_equalization" in all_results:
        eq = all_results["norm_equalization"]
        fig, ax = plt.subplots(figsize=(8, 5))

        conditions = ["Baseline", "Same Probe\n(Equalized)", "Retrained\n(Equalized)"]
        r2_values = [
            eq["baseline"]["r2"],
            eq["equalized_same_probe"]["r2"],
            eq["equalized_retrained"]["r2"],
        ]

        bars = ax.bar(conditions, r2_values, color=["steelblue", "coral", "green"])
        ax.set_ylabel("R² Score")
        ax.set_title("Effect of Norm Equalization on Position Decoding")
        ax.set_ylim(0, max(r2_values) * 1.2 if max(r2_values) > 0 else 0.5)

        for bar, val in zip(bars, r2_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(0.01, bar.get_height() + 0.01),
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "norm_equalization.png"), dpi=150)
        plt.close()

    # Plot 2: Norm-only decoding across layers
    if "norm_only" in all_results:
        norm_only = all_results["norm_only"]
        fig, ax = plt.subplots(figsize=(12, 5))

        layers = list(norm_only.keys())
        x = np.arange(len(layers))
        width = 0.25

        full_r2 = [norm_only[l]["full_r2"] for l in layers]
        norm_r2 = [norm_only[l]["norm_r2"] for l in layers]
        dir_r2 = [norm_only[l]["direction_r2"] for l in layers]

        ax.bar(x - width, full_r2, width, label="Full", color="steelblue")
        ax.bar(x, norm_r2, width, label="Norm Only", color="coral")
        ax.bar(x + width, dir_r2, width, label="Direction Only", color="green")

        ax.set_ylabel("R² Score")
        ax.set_xlabel("Layer")
        ax.set_title("Position Decoding: Norm vs Direction Across Layers")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("_", "\n") for l in layers], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "norm_vs_direction_by_layer.png"), dpi=150)
        plt.close()

    # Plot 3: Norm swapping results
    if "norm_swapping" in all_results:
        swap = all_results["norm_swapping"]
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = [
            "Correlation with\nOriginal Position",
            "Correlation with\nSwapped Position",
        ]
        values = [swap["corr_with_original"], swap["corr_with_swapped"]]
        colors = ["coral", "green"]

        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel("Pearson Correlation")
        ax.set_title("Norm Swapping: Do Predictions Follow Norm?")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

        # Add annotation
        follows = "NORM" if swap["follows_norm"] else "POSITION"
        ax.text(
            0.5,
            0.95,
            f"Predictions follow: {follows}",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "norm_swapping.png"), dpi=150)
        plt.close()

    # Plot 4: Targeted norm change
    if "targeted_norm_change" in all_results:
        targeted = all_results["targeted_norm_change"]
        fig, ax = plt.subplots(figsize=(10, 5))

        cases = targeted["test_cases"]
        labels = [f"Pos {c['target_pos']} → Norm of {c['source_pos']}" for c in cases]
        predictions = [c["mean_prediction"] for c in cases]
        target_pos = [c["actual_position"] for c in cases]
        source_pos = [c["expected_if_norm_causal"] for c in cases]

        x = np.arange(len(labels))
        width = 0.25

        ax.bar(
            x - width,
            target_pos,
            width,
            label="Actual Position",
            color="steelblue",
            alpha=0.7,
        )
        ax.bar(x, predictions, width, label="Predicted Position", color="coral")
        ax.bar(
            x + width,
            source_pos,
            width,
            label="Norm Source Position",
            color="green",
            alpha=0.7,
        )

        ax.set_ylabel("Position")
        ax.set_xlabel("Intervention")
        ax.set_title("Targeted Norm Change: Does Prediction Follow Norm?")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "targeted_norm_change.png"), dpi=150)
        plt.close()

    print(f"Plots saved to {save_dir}")


def main():
    print("=" * 70)
    print("Norm Intervention Experiment")
    print("Causal Verification that Norm Encodes Position")
    print("=" * 70)

    # Create model
    print("\nCreating random NoPE model...")
    model, config = create_random_model(
        n_layer=1, n_head=4, n_embd=256, vocab_size=65, block_size=64
    )

    all_results = {}

    # Run experiments
    all_results["norm_equalization"] = experiment_norm_equalization(model)
    all_results["norm_swapping"] = experiment_norm_swapping(model)
    all_results["norm_only"] = experiment_norm_only(model)
    all_results["targeted_norm_change"] = experiment_targeted_norm_change(model)
    all_results["synthetic_norms"] = experiment_synthetic_norm_positions(model)

    # Save results
    save_dir = Path(__file__).parent.parent / "results" / "norm_intervention"
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir / "norm_intervention_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_dir / 'norm_intervention_results.json'}")

    # Create plots
    create_plots(all_results, save_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Is Norm Causally Related to Position Encoding?")
    print("=" * 70)

    eq = all_results["norm_equalization"]
    swap = all_results["norm_swapping"]

    print(f"""
Key Findings:

1. NORM EQUALIZATION:
   - Baseline R²: {eq["baseline"]["r2"]:.4f}
   - After equalization (same probe) R²: {eq["equalized_same_probe"]["r2"]:.4f}
   - R² drop: {eq["r2_drop"]:.4f} ({eq["r2_drop_relative"] * 100:.1f}% relative)
   - Retrained on equalized R²: {eq["equalized_retrained"]["r2"]:.4f}
   
2. NORM SWAPPING:
   - Correlation with original position: {swap["corr_with_original"]:.4f}
   - Correlation with swapped position: {swap["corr_with_swapped"]:.4f}
   - Predictions follow: {"NORM" if swap["follows_norm"] else "ORIGINAL POSITION"}

3. NORM-ONLY DECODING (post_attn_residual layer):
   - Full activations R²: {all_results["norm_only"]["post_attn_residual"]["full_r2"]:.4f}
   - Norm only R²: {all_results["norm_only"]["post_attn_residual"]["norm_r2"]:.4f}
   - Direction only R²: {all_results["norm_only"]["post_attn_residual"]["direction_r2"]:.4f}

CONCLUSION:
""")

    # Determine conclusion
    if eq["r2_drop_relative"] > 0.5:
        print("  ✓ Norm equalization destroys position info → Norm is CAUSAL")
    elif eq["r2_drop_relative"] > 0.2:
        print(
            "  ~ Norm equalization partially degrades position info → Norm is IMPORTANT"
        )
    else:
        print("  ✗ Norm equalization has little effect → Norm alone is not sufficient")

    if swap["follows_norm"]:
        print("  ✓ Predictions follow swapped norms → Norm is CAUSAL")
    else:
        print("  ✗ Predictions don't follow norms → Direction also matters")

    norm_r2 = all_results["norm_only"]["post_attn_residual"]["norm_r2"]
    full_r2 = all_results["norm_only"]["post_attn_residual"]["full_r2"]
    if norm_r2 > 0.8 * full_r2:
        print("  ✓ Norm alone achieves most of full R² → Norm is PRIMARY mechanism")
    elif norm_r2 > 0.5 * full_r2:
        print("  ~ Norm captures significant info → Norm is MAJOR contributor")
    else:
        print("  ✗ Norm alone is insufficient → Direction also important")


if __name__ == "__main__":
    main()
