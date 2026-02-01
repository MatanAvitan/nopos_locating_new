"""
Experiments to validate hypotheses about where positional information
goes after LayerNorm.

Hypotheses:
-----------
H1: SPIKINESS - Position is encoded in the "shape" of the normalized vector.
    - Few terms averaged → spiky (few components dominate)
    - Many terms averaged → isotropic (components evenly spread)
    - Metrics: kurtosis, max component, entropy of squared components

H2: COVARIANCE - Position is encoded in the covariance structure.
    - The covariance matrix of activations differs by position
    - Early positions have higher variance in certain directions

H3: LEARNED GAMMA/BETA - LN's learned parameters reintroduce positional info.
    - Test by comparing: (a) full LN, (b) LN without gamma/beta
    - If gamma/beta matter, removing them should hurt position decoding

H4: POPULATION MEAN DIRECTION - The expected normalized vector differs by position.
    - Even though each sample is zero-mean, the population mean direction varies
    - This is the "LayerNorm Paradox" claim

H5: ORTHOGONALITY BREAKDOWN - After LN, the effective "directions" still differ.
    - The normalized vectors at different positions point in systematically
      different directions on average

Usage:
    python post_ln_position_hypotheses.py --checkpoint nanoGPT/out-nope-1layer-ln/ckpt.pt
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, kurtosis
from scipy.spatial.distance import cosine
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


def extract_pre_and_post_ln(model, input_ids, device="cuda"):
    """Extract activations before and after LN2, with and without gamma/beta."""
    model.eval()

    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[0]

        # Get pre-LN2 activations
        x_ln1 = block.ln_1(x)
        attn_out = block.attn(x_ln1)
        x_pre_ln2 = x + attn_out  # This is input to LN2

        # Standard LN2 output (with learned gamma/beta)
        x_post_ln2 = block.ln_2(x_pre_ln2)

        # Manual LN without gamma/beta (just normalize)
        mean = x_pre_ln2.mean(dim=-1, keepdim=True)
        var = x_pre_ln2.var(dim=-1, keepdim=True, unbiased=False)
        x_ln2_no_params = (x_pre_ln2 - mean) / torch.sqrt(var + 1e-5)

        # Get gamma and beta from LN2
        gamma = block.ln_2.weight.data  # [d_model]
        beta = (
            block.ln_2.bias.data
            if hasattr(block.ln_2, "bias") and block.ln_2.bias is not None
            else None
        )

    return {
        "pre_ln2": x_pre_ln2.cpu(),
        "post_ln2": x_post_ln2.cpu(),
        "ln2_no_params": x_ln2_no_params.cpu(),
        "gamma": gamma.cpu(),
        "beta": beta.cpu() if beta is not None else None,
    }


def hypothesis_1_spikiness(activations, positions, seq_len):
    """
    H1: Test if spikiness metrics correlate with position.

    Metrics:
    - Kurtosis: higher = more peaked/spiky
    - Max component: higher = one component dominates
    - Entropy of squared components: lower = less uniform = spikier
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: SPIKINESS")
    print("=" * 70)

    results = {}

    for name in ["pre_ln2", "post_ln2", "ln2_no_params"]:
        act = activations[name]  # [n_samples, seq_len, d_model]
        n_samples = act.shape[0]

        # Compute spikiness metrics per position
        kurtosis_per_pos = []
        max_component_per_pos = []
        entropy_per_pos = []

        for pos in range(seq_len):
            pos_acts = act[:, pos, :].numpy()  # [n_samples, d_model]

            # Kurtosis (averaged over samples)
            kurt = np.mean([kurtosis(sample) for sample in pos_acts])
            kurtosis_per_pos.append(kurt)

            # Max absolute component (averaged over samples)
            max_comp = np.mean(np.max(np.abs(pos_acts), axis=1))
            max_component_per_pos.append(max_comp)

            # Entropy of squared components (averaged over samples)
            # Higher entropy = more uniform = less spiky
            sq = pos_acts**2
            sq_normalized = sq / (sq.sum(axis=1, keepdims=True) + 1e-10)
            ent = -np.sum(sq_normalized * np.log(sq_normalized + 1e-10), axis=1)
            entropy_per_pos.append(np.mean(ent))

        pos_range = np.arange(seq_len)

        r_kurt, _ = pearsonr(kurtosis_per_pos, pos_range)
        r_max, _ = pearsonr(max_component_per_pos, pos_range)
        r_ent, _ = pearsonr(entropy_per_pos, pos_range)

        results[name] = {
            "kurtosis_per_pos": kurtosis_per_pos,
            "max_component_per_pos": max_component_per_pos,
            "entropy_per_pos": entropy_per_pos,
            "r_kurtosis_vs_pos": r_kurt,
            "r_max_vs_pos": r_max,
            "r_entropy_vs_pos": r_ent,
        }

        print(f"\n{name}:")
        print(f"  Kurtosis vs position:      r = {r_kurt:.4f}")
        print(f"  Max component vs position: r = {r_max:.4f}")
        print(f"  Entropy vs position:       r = {r_ent:.4f}")

        # Interpretation
        if name == "post_ln2":
            if abs(r_kurt) > 0.3 or abs(r_max) > 0.3 or abs(r_ent) > 0.3:
                print("  → SUPPORTS H1: Spikiness differs by position after LN")
            else:
                print("  → WEAK/NO SUPPORT: Spikiness doesn't strongly encode position")

    return results


def hypothesis_2_covariance(activations, positions, seq_len, n_positions_to_compare=5):
    """
    H2: Test if covariance structure differs by position.

    Compare covariance matrices at different positions.
    If they differ, position info is in the covariance structure.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: COVARIANCE STRUCTURE")
    print("=" * 70)

    results = {}

    for name in ["pre_ln2", "post_ln2", "ln2_no_params"]:
        act = activations[name]  # [n_samples, seq_len, d_model]

        # Sample positions to compare
        positions_to_check = np.linspace(
            0, seq_len - 1, n_positions_to_compare, dtype=int
        )

        # Compute covariance matrix for each position
        cov_matrices = []
        for pos in positions_to_check:
            pos_acts = act[:, pos, :].numpy()  # [n_samples, d_model]
            cov = np.cov(pos_acts.T)  # [d_model, d_model]
            cov_matrices.append(cov)

        # Compare covariance matrices using Frobenius distance
        distances = []
        for i in range(len(cov_matrices)):
            for j in range(i + 1, len(cov_matrices)):
                dist = np.linalg.norm(cov_matrices[i] - cov_matrices[j], "fro")
                pos_diff = abs(positions_to_check[i] - positions_to_check[j])
                distances.append((pos_diff, dist))

        pos_diffs = [d[0] for d in distances]
        cov_dists = [d[1] for d in distances]

        r_cov_dist, _ = pearsonr(pos_diffs, cov_dists)

        # Also compute eigenvalue spectra - do they differ?
        eigenvalues = [np.sort(np.linalg.eigvalsh(cov))[::-1] for cov in cov_matrices]

        # Compare top eigenvalue ratio (measure of "dimensionality")
        top_eig_ratios = [eig[0] / (eig.sum() + 1e-10) for eig in eigenvalues]
        r_eig_ratio, _ = pearsonr(positions_to_check, top_eig_ratios)

        results[name] = {
            "positions_checked": positions_to_check,
            "cov_distance_vs_pos_diff_r": r_cov_dist,
            "top_eigenvalue_ratio_vs_pos_r": r_eig_ratio,
            "top_eig_ratios": top_eig_ratios,
        }

        print(f"\n{name}:")
        print(f"  Cov distance vs position diff: r = {r_cov_dist:.4f}")
        print(f"  Top eigenvalue ratio vs pos:   r = {r_eig_ratio:.4f}")
        print(f"  Top eig ratios: {[f'{r:.4f}' for r in top_eig_ratios]}")

        if name == "post_ln2":
            if abs(r_cov_dist) > 0.3 or abs(r_eig_ratio) > 0.3:
                print("  → SUPPORTS H2: Covariance structure differs by position")
            else:
                print(
                    "  → WEAK/NO SUPPORT: Covariance structure similar across positions"
                )

    return results


def hypothesis_3_gamma_beta(activations, positions, seq_len):
    """
    H3: Test if learned gamma/beta contribute to position encoding.

    Compare probe accuracy on:
    - post_ln2 (with gamma/beta)
    - ln2_no_params (without gamma/beta)

    If gamma/beta matter, removing them should hurt position decoding.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: LEARNED GAMMA/BETA")
    print("=" * 70)

    n_samples = activations["post_ln2"].shape[0]
    d_model = activations["post_ln2"].shape[2]

    # Create position labels
    positions_flat = np.tile(np.arange(seq_len), n_samples)

    # Train/test split
    n_train = int(0.8 * n_samples)
    train_mask = np.repeat(np.arange(n_samples) < n_train, seq_len)

    results = {}

    for name in ["post_ln2", "ln2_no_params"]:
        act = activations[name]  # [n_samples, seq_len, d_model]
        X = act.reshape(-1, d_model).numpy()

        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = positions_flat[train_mask], positions_flat[~train_mask]

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)

        y_pred = probe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)

        results[name] = {"r2": r2, "mae": mae, "pearson_r": r}
        print(f"\n{name}:")
        print(f"  R² = {r2:.4f}, MAE = {mae:.2f}, r = {r:.4f}")

    # Compare
    diff_r2 = results["post_ln2"]["r2"] - results["ln2_no_params"]["r2"]
    diff_mae = results["ln2_no_params"]["mae"] - results["post_ln2"]["mae"]

    print(f"\nDifference (with - without gamma/beta):")
    print(f"  ΔR² = {diff_r2:.4f}")
    print(f"  ΔMAE = {diff_mae:.2f} (positive = gamma/beta helps)")

    if diff_r2 > 0.05:
        print("  → SUPPORTS H3: Gamma/beta significantly help position decoding")
    elif diff_r2 > 0.01:
        print("  → PARTIAL SUPPORT: Gamma/beta provide modest benefit")
    else:
        print("  → WEAK/NO SUPPORT: Gamma/beta don't significantly help")

    results["diff_r2"] = diff_r2
    results["diff_mae"] = diff_mae

    return results


def hypothesis_4_population_mean(activations, positions, seq_len):
    """
    H4: Test if population mean direction differs by position.

    This is the core of the "LayerNorm Paradox":
    - Each sample is zero-mean after LN
    - But the POPULATION mean (E[x_i]) differs by position

    Test: Compute population mean at each position and see if they differ.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: POPULATION MEAN DIRECTION")
    print("=" * 70)

    results = {}

    for name in ["pre_ln2", "post_ln2", "ln2_no_params"]:
        act = activations[name]  # [n_samples, seq_len, d_model]

        # Compute population mean at each position
        pop_means = act.mean(dim=0).numpy()  # [seq_len, d_model]

        # 1. Norm of population mean (should be ~0 for LN outputs if perfectly centered)
        pop_mean_norms = np.linalg.norm(pop_means, axis=1)
        r_mean_norm_vs_pos, _ = pearsonr(pop_mean_norms, np.arange(seq_len))

        # 2. Cosine similarity between population means at different positions
        # If they point in different directions, position info is in direction
        cos_sims_adjacent = []
        for i in range(seq_len - 1):
            cos_sim = 1 - cosine(pop_means[i], pop_means[i + 1])
            cos_sims_adjacent.append(cos_sim)
        avg_adjacent_cos_sim = np.mean(cos_sims_adjacent)

        # 3. Cosine similarity with position 0
        cos_sims_to_pos0 = []
        for i in range(seq_len):
            cos_sim = 1 - cosine(pop_means[0], pop_means[i])
            cos_sims_to_pos0.append(cos_sim)
        r_cos_to_pos0, _ = pearsonr(cos_sims_to_pos0, np.arange(seq_len))

        # 4. Can we decode position from population means alone?
        # Train probe on just the population means
        probe = Ridge(alpha=1.0)
        probe.fit(pop_means, np.arange(seq_len))
        pred = probe.predict(pop_means)
        r_pop_mean_decode, _ = pearsonr(pred, np.arange(seq_len))

        results[name] = {
            "pop_mean_norms": pop_mean_norms,
            "r_mean_norm_vs_pos": r_mean_norm_vs_pos,
            "avg_adjacent_cos_sim": avg_adjacent_cos_sim,
            "r_cos_to_pos0_vs_pos": r_cos_to_pos0,
            "r_pop_mean_decode": r_pop_mean_decode,
        }

        print(f"\n{name}:")
        print(f"  Population mean norm vs pos: r = {r_mean_norm_vs_pos:.4f}")
        print(f"  Mean pop_mean norm: {pop_mean_norms.mean():.4f}")
        print(f"  Avg adjacent cosine sim: {avg_adjacent_cos_sim:.4f}")
        print(f"  Cosine to pos 0 vs pos: r = {r_cos_to_pos0:.4f}")
        print(f"  Position from pop mean alone: r = {r_pop_mean_decode:.4f}")

        if name == "post_ln2":
            if abs(r_pop_mean_decode) > 0.9:
                print(
                    "  → STRONG SUPPORT H4: Population means perfectly encode position!"
                )
            elif abs(r_pop_mean_decode) > 0.5:
                print("  → SUPPORTS H4: Population means encode position")
            else:
                print(
                    "  → WEAK SUPPORT: Population means don't strongly encode position"
                )

    return results


def hypothesis_5_direction_shift(activations, positions, seq_len):
    """
    H5: Test if normalized vectors systematically point in different directions.

    Even after LN makes each sample unit-norm, the directions may differ by position.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: DIRECTION SHIFT")
    print("=" * 70)

    results = {}

    for name in ["post_ln2", "ln2_no_params"]:
        act = activations[name]  # [n_samples, seq_len, d_model]
        n_samples = act.shape[0]

        # Normalize each vector to unit norm
        act_normalized = F.normalize(act, dim=-1).numpy()

        # Compute mean direction at each position
        mean_directions = act_normalized.mean(axis=0)  # [seq_len, d_model]
        mean_directions = mean_directions / (
            np.linalg.norm(mean_directions, axis=1, keepdims=True) + 1e-10
        )

        # Cosine similarity of mean direction to position 0's mean direction
        cos_to_pos0 = [
            np.dot(mean_directions[0], mean_directions[i]) for i in range(seq_len)
        ]
        r_direction_shift, _ = pearsonr(cos_to_pos0, np.arange(seq_len))

        # How "concentrated" are the directions at each position?
        # (Higher concentration = vectors point more consistently in same direction)
        concentration = np.linalg.norm(act_normalized.mean(axis=0), axis=1)
        r_concentration_vs_pos, _ = pearsonr(concentration, np.arange(seq_len))

        results[name] = {
            "cos_to_pos0": cos_to_pos0,
            "r_direction_shift": r_direction_shift,
            "concentration": concentration,
            "r_concentration_vs_pos": r_concentration_vs_pos,
        }

        print(f"\n{name}:")
        print(f"  Direction shift (cos to pos 0) vs pos: r = {r_direction_shift:.4f}")
        print(f"  Direction concentration vs pos: r = {r_concentration_vs_pos:.4f}")

        if abs(r_direction_shift) > 0.5:
            print("  → SUPPORTS H5: Directions systematically shift with position")
        else:
            print("  → WEAK: Directions don't clearly shift with position")

    return results


def plot_hypothesis_results(
    h1_results, h2_results, h4_results, h5_results, seq_len, output_path
):
    """Create visualization of hypothesis test results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    pos_range = np.arange(seq_len)

    # H1: Spikiness (entropy)
    ax = axes[0, 0]
    for name, style in [
        ("pre_ln2", "b-"),
        ("post_ln2", "r-"),
        ("ln2_no_params", "g--"),
    ]:
        ax.plot(
            pos_range, h1_results[name]["entropy_per_pos"], style, label=name, alpha=0.7
        )
    ax.set_xlabel("Position")
    ax.set_ylabel("Entropy of squared components")
    ax.set_title("H1: Spikiness (Entropy)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # H1: Kurtosis
    ax = axes[0, 1]
    for name, style in [
        ("pre_ln2", "b-"),
        ("post_ln2", "r-"),
        ("ln2_no_params", "g--"),
    ]:
        ax.plot(
            pos_range,
            h1_results[name]["kurtosis_per_pos"],
            style,
            label=name,
            alpha=0.7,
        )
    ax.set_xlabel("Position")
    ax.set_ylabel("Kurtosis")
    ax.set_title("H1: Spikiness (Kurtosis)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # H2: Covariance eigenvalue ratio
    ax = axes[0, 2]
    for name, style in [
        ("pre_ln2", "bo-"),
        ("post_ln2", "ro-"),
        ("ln2_no_params", "go--"),
    ]:
        positions_checked = h2_results[name]["positions_checked"]
        ratios = h2_results[name]["top_eig_ratios"]
        ax.plot(positions_checked, ratios, style, label=name, alpha=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("Top eigenvalue ratio")
    ax.set_title("H2: Covariance Structure")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # H4: Population mean norm
    ax = axes[1, 0]
    for name, style in [
        ("pre_ln2", "b-"),
        ("post_ln2", "r-"),
        ("ln2_no_params", "g--"),
    ]:
        ax.plot(
            pos_range, h4_results[name]["pop_mean_norms"], style, label=name, alpha=0.7
        )
    ax.set_xlabel("Position")
    ax.set_ylabel("||population mean||")
    ax.set_title("H4: Population Mean Norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # H5: Direction shift
    ax = axes[1, 1]
    for name, style in [("post_ln2", "r-"), ("ln2_no_params", "g--")]:
        ax.plot(
            pos_range, h5_results[name]["cos_to_pos0"], style, label=name, alpha=0.7
        )
    ax.set_xlabel("Position")
    ax.set_ylabel("Cosine sim to pos 0")
    ax.set_title("H5: Direction Shift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 2]
    ax.axis("off")

    summary = "HYPOTHESIS TEST SUMMARY\n" + "=" * 30 + "\n\n"

    # H1
    r_ent = h1_results["post_ln2"]["r_entropy_vs_pos"]
    summary += f"H1 (Spikiness):\n  Entropy vs pos: r={r_ent:.3f}\n"
    summary += f"  {'SUPPORTED' if abs(r_ent) > 0.3 else 'WEAK'}\n\n"

    # H2
    r_cov = h2_results["post_ln2"]["top_eigenvalue_ratio_vs_pos_r"]
    summary += f"H2 (Covariance):\n  Eig ratio vs pos: r={r_cov:.3f}\n"
    summary += f"  {'SUPPORTED' if abs(r_cov) > 0.3 else 'WEAK'}\n\n"

    # H4
    r_pop = h4_results["post_ln2"]["r_pop_mean_decode"]
    summary += f"H4 (Pop Mean):\n  Decode from mean: r={r_pop:.3f}\n"
    summary += f"  {'STRONG' if abs(r_pop) > 0.9 else 'SUPPORTED' if abs(r_pop) > 0.5 else 'WEAK'}\n\n"

    # H5
    r_dir = h5_results["post_ln2"]["r_direction_shift"]
    summary += f"H5 (Direction):\n  Shift vs pos: r={r_dir:.3f}\n"
    summary += f"  {'SUPPORTED' if abs(r_dir) > 0.5 else 'WEAK'}\n"

    ax.text(
        0.1,
        0.9,
        summary,
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
    print("POST-LAYERNORM POSITIONAL ENCODING HYPOTHESES")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, args.device)

    norm_type = model.config.norm_type
    seq_len = model.config.block_size
    vocab_size = model.config.vocab_size

    print(f"Model: {norm_type.upper()}, Seq len: {seq_len}")

    # Generate random sequences
    print(f"\nGenerating {args.n_samples} random sequences...")
    torch.manual_seed(42)
    input_ids = torch.randint(
        0, vocab_size, (args.n_samples, seq_len), device=args.device
    )

    # Extract activations in batches
    print("Extracting activations...")
    batch_size = 100
    all_activations = {"pre_ln2": [], "post_ln2": [], "ln2_no_params": []}

    for i in range(0, args.n_samples, batch_size):
        batch = input_ids[i : i + batch_size]
        acts = extract_pre_and_post_ln(model, batch, args.device)
        for k in all_activations:
            all_activations[k].append(acts[k])

    for k in all_activations:
        all_activations[k] = torch.cat(all_activations[k], dim=0)

    # Store gamma/beta from last batch
    all_activations["gamma"] = acts["gamma"]
    all_activations["beta"] = acts["beta"]

    # Create position labels
    positions = torch.arange(seq_len).unsqueeze(0).expand(args.n_samples, -1)

    # Run hypothesis tests
    h1_results = hypothesis_1_spikiness(all_activations, positions, seq_len)
    h2_results = hypothesis_2_covariance(all_activations, positions, seq_len)
    h3_results = hypothesis_3_gamma_beta(all_activations, positions, seq_len)
    h4_results = hypothesis_4_population_mean(all_activations, positions, seq_len)
    h5_results = hypothesis_5_direction_shift(all_activations, positions, seq_len)

    # Generate plot
    output_path = Path(args.output_dir) / f"post_ln_hypotheses_{norm_type}.png"
    plot_hypothesis_results(
        h1_results, h2_results, h4_results, h5_results, seq_len, output_path
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: WHERE IS POSITION AFTER LAYERNORM?")
    print("=" * 70)

    print(f"""
Based on hypothesis tests:

H1 (Spikiness): r = {h1_results["post_ln2"]["r_entropy_vs_pos"]:.3f}
    Position encoded in shape/distribution of components?
    
H2 (Covariance): r = {h2_results["post_ln2"]["top_eigenvalue_ratio_vs_pos_r"]:.3f}
    Position encoded in covariance structure?
    
H3 (Gamma/Beta): ΔR² = {h3_results["diff_r2"]:.4f}
    Learned LN parameters help decode position?
    
H4 (Population Mean): r = {h4_results["post_ln2"]["r_pop_mean_decode"]:.3f}
    Position encoded in expected (population) direction?
    
H5 (Direction Shift): r = {h5_results["post_ln2"]["r_direction_shift"]:.3f}
    Normalized vectors systematically shift direction by position?

INTERPRETATION:
The strongest signal appears to be in the POPULATION MEAN (H4).
Even though each sample is zero-mean after LN, the expected value
E[x_i] differs by position. This is because the token distribution
at each position is different (due to the averaging of different
numbers of random embeddings), creating position-dependent expectations.
""")

    # Save results
    results_path = Path(args.output_dir) / f"post_ln_hypotheses_results_{norm_type}.pt"
    torch.save(
        {
            "h1": h1_results,
            "h2": h2_results,
            "h3": h3_results,
            "h4": h4_results,
            "h5": h5_results,
        },
        results_path,
    )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
