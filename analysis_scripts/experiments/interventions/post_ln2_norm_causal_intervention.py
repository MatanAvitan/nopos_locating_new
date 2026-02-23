"""
Post-LN2 Norm Causal Intervention Experiment

Goal: Determine if the language model actually USES the norm-based position encoding
at post-LN2 to predict position, or if it relies on other information.

Experiments:
1. Norm Swapping: Swap norms between position i and position j, see if predictions follow
2. Norm Equalization: Set all norms to the mean, see if position decoding breaks
3. Norm Injection: Give position i the norm from position j, check prediction shift
4. Direction Preservation: Keep direction, modify norm - does prediction change?

We test on both:
- Synthetic uniform tokens (controlled setting)
- Shakespeare data (realistic setting)
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "post_ln2_norm_intervention"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

print(f"Using device: {DEVICE}")


def create_random_model(n_embd=768, n_head=12, n_ctx=64, vocab_size=50257):
    """Create randomly initialized NoPE model (Synthetic-Large config)."""
    config = GPTConfig(
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        block_size=n_ctx,
        vocab_size=vocab_size,
        dropout=0.0,
        bias=False,
        use_positional_embedding=False,
        norm_type="layernorm",
    )
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model, config


def load_trained_model(checkpoint_path):
    """Load trained NoPE model."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 256),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        bias=model_args.get("bias", False),
        use_positional_embedding=False,
        norm_type=model_args.get("norm_type", "layernorm"),
    )

    model = GPT(config)
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(DEVICE)
    return model, config


def get_post_ln2_activations(model, tokens):
    """Get post-LN2 activations with hook."""
    activations = {}

    def hook_fn(module, input, output):
        activations["post_ln2"] = output.detach()

    hook = model.transformer.h[0].ln_2.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(tokens)

    hook.remove()
    return activations["post_ln2"]


def get_activations_at_layers(model, tokens):
    """Get activations at multiple layers for analysis."""
    activations = {}

    with torch.no_grad():
        # Embedding
        tok_emb = model.transformer.wte(tokens)
        activations["embed"] = tok_emb.detach()

        # Through block
        block = model.transformer.h[0]

        # Post LN1
        x_ln1 = block.ln_1(tok_emb)
        activations["post_ln1"] = x_ln1.detach()

        # Post attention
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.detach()

        # Post attention residual
        x_res = tok_emb + attn_out
        activations["post_attn_residual"] = x_res.detach()

        # Post LN2
        x_ln2 = block.ln_2(x_res)
        activations["post_ln2"] = x_ln2.detach()

        # Post MLP
        mlp_out = block.mlp(x_ln2)
        x_mlp_res = x_res + mlp_out
        activations["post_mlp_residual"] = x_mlp_res.detach()

    return activations


def train_norm_probe(norms, positions):
    """Train a linear probe to predict position from norm."""
    X = norms.reshape(-1, 1)
    y = positions.flatten()

    probe = Ridge(alpha=1.0)
    probe.fit(X, y)

    preds = probe.predict(X)
    r2 = r2_score(y, preds)

    return probe, r2


def train_full_probe(activations, positions):
    """Train a linear probe on full activations."""
    X = activations.reshape(-1, activations.shape[-1])
    y = positions.flatten()

    probe = Ridge(alpha=1.0)
    probe.fit(X, y)

    preds = probe.predict(X)
    r2 = r2_score(y, preds)

    return probe, r2


def intervention_norm_equalization(activations, target_norm=None):
    """Set all activations to have the same norm (mean norm by default)."""
    norms = torch.norm(activations, dim=-1, keepdim=True)
    if target_norm is None:
        target_norm = norms.mean()
    directions = activations / (norms + 1e-8)
    return directions * target_norm


def intervention_norm_swap(activations, pos_i, pos_j):
    """Swap norms between positions i and j."""
    norms = torch.norm(activations, dim=-1, keepdim=True)
    directions = activations / (norms + 1e-8)

    # Swap norms
    new_norms = norms.clone()
    new_norms[:, pos_i, :] = norms[:, pos_j, :]
    new_norms[:, pos_j, :] = norms[:, pos_i, :]

    return directions * new_norms


def intervention_norm_shuffle(activations):
    """Randomly shuffle norms across positions (within each sample)."""
    batch_size, seq_len, d_model = activations.shape
    norms = torch.norm(activations, dim=-1, keepdim=True)
    directions = activations / (norms + 1e-8)

    # Shuffle norms across positions for each sample
    shuffled_norms = norms.clone()
    for b in range(batch_size):
        perm = torch.randperm(seq_len)
        shuffled_norms[b] = norms[b, perm]

    return directions * shuffled_norms


def intervention_direction_shuffle(activations):
    """Randomly shuffle directions across positions (keep norms)."""
    batch_size, seq_len, d_model = activations.shape
    norms = torch.norm(activations, dim=-1, keepdim=True)
    directions = activations / (norms + 1e-8)

    # Shuffle directions across positions for each sample
    shuffled_directions = directions.clone()
    for b in range(batch_size):
        perm = torch.randperm(seq_len)
        shuffled_directions[b] = directions[b, perm]

    return shuffled_directions * norms


def run_causal_intervention_experiment(model, tokens, experiment_name="synthetic"):
    """
    Run full causal intervention experiment.

    Returns dict with results for each intervention type.
    """
    print(f"\n{'=' * 60}")
    print(f"CAUSAL INTERVENTION EXPERIMENT: {experiment_name}")
    print(f"{'=' * 60}")

    n_samples, n_ctx = tokens.shape
    positions = np.tile(np.arange(n_ctx), (n_samples, 1))

    # Get baseline activations
    activations = get_activations_at_layers(model, tokens)
    post_ln2 = activations["post_ln2"].cpu().numpy()

    # Compute norms
    norms = np.linalg.norm(post_ln2, axis=-1)  # (batch, seq)

    # === Baseline metrics ===
    print("\n--- Baseline Metrics ---")

    # Norm-position correlation
    flat_norms = norms.flatten()
    flat_pos = positions.flatten()
    norm_pos_corr = np.corrcoef(flat_norms, flat_pos)[0, 1]
    print(f"Norm-position correlation: r = {norm_pos_corr:.4f}")

    # Train probes
    norm_probe, norm_r2 = train_norm_probe(flat_norms, flat_pos)
    full_probe, full_r2 = train_full_probe(post_ln2, positions)

    print(f"Norm-only probe R²: {norm_r2:.4f}")
    print(f"Full activation probe R²: {full_r2:.4f}")

    results = {
        "experiment_name": experiment_name,
        "n_samples": n_samples,
        "n_ctx": n_ctx,
        "baseline": {
            "norm_position_corr": float(norm_pos_corr),
            "norm_probe_r2": float(norm_r2),
            "full_probe_r2": float(full_r2),
        },
        "interventions": {},
    }

    # === Intervention 1: Norm Equalization ===
    print("\n--- Intervention 1: Norm Equalization ---")
    post_ln2_tensor = activations["post_ln2"]
    equalized = intervention_norm_equalization(post_ln2_tensor)
    eq_numpy = equalized.cpu().numpy()

    # Test full probe on equalized activations
    eq_preds = full_probe.predict(eq_numpy.reshape(-1, eq_numpy.shape[-1]))
    eq_r2 = r2_score(flat_pos, eq_preds)

    # Correlation between predictions and true position
    eq_pred_corr = np.corrcoef(eq_preds, flat_pos)[0, 1]

    print(f"Full probe R² after equalization: {eq_r2:.4f} (was {full_r2:.4f})")
    print(f"Prediction-position correlation: {eq_pred_corr:.4f}")
    print(
        f"R² drop: {full_r2 - eq_r2:.4f} ({100 * (full_r2 - eq_r2) / max(full_r2, 1e-6):.1f}%)"
    )

    results["interventions"]["norm_equalization"] = {
        "full_probe_r2": float(eq_r2),
        "pred_position_corr": float(eq_pred_corr),
        "r2_drop": float(full_r2 - eq_r2),
        "r2_drop_percent": float(100 * (full_r2 - eq_r2) / max(full_r2, 1e-6)),
    }

    # === Intervention 2: Norm Shuffle ===
    print("\n--- Intervention 2: Norm Shuffle (randomize norms across positions) ---")
    shuffled_norm = intervention_norm_shuffle(post_ln2_tensor)
    sn_numpy = shuffled_norm.cpu().numpy()

    sn_preds = full_probe.predict(sn_numpy.reshape(-1, sn_numpy.shape[-1]))
    sn_r2 = r2_score(flat_pos, sn_preds)
    sn_pred_corr = np.corrcoef(sn_preds, flat_pos)[0, 1]

    print(f"Full probe R² after norm shuffle: {sn_r2:.4f} (was {full_r2:.4f})")
    print(f"Prediction-position correlation: {sn_pred_corr:.4f}")

    results["interventions"]["norm_shuffle"] = {
        "full_probe_r2": float(sn_r2),
        "pred_position_corr": float(sn_pred_corr),
        "r2_drop": float(full_r2 - sn_r2),
    }

    # === Intervention 3: Direction Shuffle (control) ===
    print(
        "\n--- Intervention 3: Direction Shuffle (keep norms, randomize directions) ---"
    )
    shuffled_dir = intervention_direction_shuffle(post_ln2_tensor)
    sd_numpy = shuffled_dir.cpu().numpy()

    sd_preds = full_probe.predict(sd_numpy.reshape(-1, sd_numpy.shape[-1]))
    sd_r2 = r2_score(flat_pos, sd_preds)
    sd_pred_corr = np.corrcoef(sd_preds, flat_pos)[0, 1]

    # Also test norm probe on direction-shuffled (norms unchanged)
    sd_norms = np.linalg.norm(sd_numpy, axis=-1).flatten()
    sd_norm_preds = norm_probe.predict(sd_norms.reshape(-1, 1))
    sd_norm_r2 = r2_score(flat_pos, sd_norm_preds)

    print(f"Full probe R² after direction shuffle: {sd_r2:.4f} (was {full_r2:.4f})")
    print(f"Norm probe R² (should be unchanged): {sd_norm_r2:.4f} (was {norm_r2:.4f})")
    print(f"Prediction-position correlation: {sd_pred_corr:.4f}")

    results["interventions"]["direction_shuffle"] = {
        "full_probe_r2": float(sd_r2),
        "norm_probe_r2": float(sd_norm_r2),
        "pred_position_corr": float(sd_pred_corr),
    }

    # === Intervention 4: Specific position swap ===
    print("\n--- Intervention 4: Swap norms between position 5 and 55 ---")
    swapped = intervention_norm_swap(post_ln2_tensor, 5, 55)
    sw_numpy = swapped.cpu().numpy()

    # Check predictions at the swapped positions
    sw_preds = full_probe.predict(sw_numpy.reshape(-1, sw_numpy.shape[-1]))
    sw_preds = sw_preds.reshape(n_samples, n_ctx)

    # At position 5, prediction should now be closer to 55 (and vice versa)
    mean_pred_pos5 = sw_preds[:, 5].mean()
    mean_pred_pos55 = sw_preds[:, 55].mean()

    # Baseline predictions at these positions
    baseline_preds = full_probe.predict(post_ln2.reshape(-1, post_ln2.shape[-1]))
    baseline_preds = baseline_preds.reshape(n_samples, n_ctx)
    baseline_pred_pos5 = baseline_preds[:, 5].mean()
    baseline_pred_pos55 = baseline_preds[:, 55].mean()

    print(
        f"Position 5: baseline pred={baseline_pred_pos5:.2f}, after swap pred={mean_pred_pos5:.2f}"
    )
    print(
        f"Position 55: baseline pred={baseline_pred_pos55:.2f}, after swap pred={mean_pred_pos55:.2f}"
    )
    print(f"Prediction shift at pos 5: {mean_pred_pos5 - baseline_pred_pos5:+.2f}")
    print(f"Prediction shift at pos 55: {mean_pred_pos55 - baseline_pred_pos55:+.2f}")

    results["interventions"]["norm_swap_5_55"] = {
        "baseline_pred_pos5": float(baseline_pred_pos5),
        "baseline_pred_pos55": float(baseline_pred_pos55),
        "after_swap_pred_pos5": float(mean_pred_pos5),
        "after_swap_pred_pos55": float(mean_pred_pos55),
        "shift_pos5": float(mean_pred_pos5 - baseline_pred_pos5),
        "shift_pos55": float(mean_pred_pos55 - baseline_pred_pos55),
    }

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Baseline full probe R²: {full_r2:.4f}")
    print(
        f"After norm equalization: {eq_r2:.4f} ({100 * (full_r2 - eq_r2) / max(full_r2, 1e-6):.1f}% drop)"
    )
    print(f"After norm shuffle: {sn_r2:.4f}")
    print(f"After direction shuffle: {sd_r2:.4f}")

    if full_r2 > 0.1 and (full_r2 - eq_r2) > 0.05:
        print(
            "\n✓ CONCLUSION: Norm equalization significantly degrades position prediction,"
        )
        print("  suggesting the model DOES use norm to encode position.")
    elif full_r2 > 0.1 and (full_r2 - sd_r2) > (full_r2 - eq_r2):
        print("\n✓ CONCLUSION: Direction shuffle hurts more than norm equalization,")
        print("  suggesting direction carries more position information than norm.")
    else:
        print("\n? CONCLUSION: Results inconclusive or baseline R² too low.")

    return results


def main():
    print("=" * 70)
    print("POST-LN2 NORM CAUSAL INTERVENTION EXPERIMENT")
    print("=" * 70)

    all_results = {}

    # === Experiment 1: Synthetic Uniform Tokens ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Synthetic Uniform Tokens (Synthetic-Large config)")
    print("=" * 70)

    model, config = create_random_model(
        n_embd=768, n_head=12, n_ctx=64, vocab_size=50257
    )
    n_samples = 500
    tokens = torch.randint(
        0, config.vocab_size, (n_samples, config.block_size), device=DEVICE
    )

    results_synthetic = run_causal_intervention_experiment(
        model, tokens, "synthetic_uniform"
    )
    all_results["synthetic_uniform"] = results_synthetic

    # === Experiment 2: Synthetic with Small Model (matches Table 5 config) ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Synthetic-Small config (matches Table 5)")
    print("=" * 70)

    model_small, config_small = create_random_model(
        n_embd=256, n_head=4, n_ctx=64, vocab_size=1000
    )
    tokens_small = torch.randint(
        0, config_small.vocab_size, (n_samples, config_small.block_size), device=DEVICE
    )

    results_small = run_causal_intervention_experiment(
        model_small, tokens_small, "synthetic_small"
    )
    all_results["synthetic_small"] = results_small

    # === Experiment 3: Trained Model on Shakespeare ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Trained Model (Shakespeare data)")
    print("=" * 70)

    checkpoint_path = (
        Path(__file__).parent.parent.parent.parent / "nanoGPT" / "out-nope-1layer-ln" / "ckpt.pt"
    )
    if checkpoint_path.exists():
        trained_model, trained_config = load_trained_model(checkpoint_path)

        # Load Shakespeare data
        data_path = (
            Path(__file__).parent.parent.parent.parent
            / "nanoGPT"
            / "data"
            / "shakespeare"
            / "train.bin"
        )
        if data_path.exists():
            data = np.memmap(data_path, dtype=np.uint16, mode="r")
            n_ctx = min(trained_config.block_size, 64)  # Use 64 for comparison

            # Sample sequences
            tokens_list = []
            for _ in range(n_samples):
                start_idx = np.random.randint(0, len(data) - n_ctx)
                seq = torch.tensor(
                    data[start_idx : start_idx + n_ctx], dtype=torch.long, device=DEVICE
                )
                tokens_list.append(seq)
            tokens_shakespeare = torch.stack(tokens_list)

            results_shakespeare = run_causal_intervention_experiment(
                trained_model, tokens_shakespeare, "trained_shakespeare"
            )
            all_results["trained_shakespeare"] = results_shakespeare
        else:
            print(f"Shakespeare data not found at {data_path}")
    else:
        print(f"Trained model not found at {checkpoint_path}")

    # === Save results ===
    results_file = RESULTS_DIR / "post_ln2_norm_intervention_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # === Create visualization ===
    create_summary_plot(all_results)

    return all_results


def create_summary_plot(all_results):
    """Create summary visualization of intervention results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (exp_name, results) in enumerate(all_results.items()):
        ax = axes[idx] if idx < 3 else None
        if ax is None:
            continue

        baseline_r2 = results["baseline"]["full_probe_r2"]
        interventions = results["interventions"]

        labels = ["Baseline", "Norm\nEqualized", "Norm\nShuffle", "Direction\nShuffle"]
        values = [
            baseline_r2,
            interventions["norm_equalization"]["full_probe_r2"],
            interventions["norm_shuffle"]["full_probe_r2"],
            interventions["direction_shuffle"]["full_probe_r2"],
        ]
        colors = ["#3498db", "#e74c3c", "#e74c3c", "#2ecc71"]

        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_ylabel("Full Probe R²")
        ax.set_title(
            f"{exp_name}\n(baseline norm R²={results['baseline']['norm_probe_r2']:.2f})"
        )
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                fontsize=10,
            )

    plt.tight_layout()

    plot_path = RESULTS_DIR / "post_ln2_norm_intervention_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.savefig(
        RESULTS_DIR / "post_ln2_norm_intervention_summary.pdf", bbox_inches="tight"
    )
    print(f"Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    results = main()
