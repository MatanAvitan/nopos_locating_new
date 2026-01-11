"""
Direction vs Norm Independence Analysis

Key question: Are direction and norm independently encoding position, or are they
correlated manifestations of the same underlying mechanism?

Experiments:
1. Direction isolation: Keep direction, set all norms to constant
2. Norm isolation: Keep norms, randomize directions (shuffle neurons within position)
3. Cross-decoding: Train probe on direction-only, test on norm-only and vice versa
4. Orthogonalization: Remove norm information from direction and test prediction
5. PCA analysis: What directions encode position? Are they aligned with norm?
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy import stats

# Configuration
N_SAMPLES = 500
N_CTX = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "direction_norm_independence"


def create_random_model():
    """Create randomly initialized NoPE model."""
    config = GPTConfig(
        n_layer=1,
        n_head=4,
        n_embd=256,
        block_size=N_CTX,
        vocab_size=1000,
        dropout=0.0,
        use_positional_embedding=False,
        norm_type="layernorm",
    )
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model


def get_activations_with_cache(model, tokens):
    """Get activations at different layers."""
    activations = {}

    # Embedding
    tok_emb = model.transformer.wte(tokens)
    activations["embed"] = tok_emb.detach()

    # First block
    block = model.transformer.h[0]

    # Post LN1
    x = block.ln_1(tok_emb)
    activations["post_ln1"] = x.detach()

    # Post attention
    attn_out = block.attn(x)[0]
    activations["post_attn"] = attn_out.detach()

    # Post attention residual
    x = tok_emb + attn_out
    activations["post_attn_residual"] = x.detach()

    # Post LN2
    x_ln2 = block.ln_2(x)
    activations["post_ln2"] = x_ln2.detach()

    # Post MLP residual
    mlp_out = block.mlp(x_ln2)
    x = x + mlp_out
    activations["post_mlp_residual"] = x.detach()

    return activations


def extract_direction_and_norm(activations):
    """Separate activations into direction (unit vectors) and norm (scalars)."""
    norms = torch.norm(activations, dim=-1, keepdim=True)
    directions = activations / (norms + 1e-8)
    return directions, norms.squeeze(-1)


def train_ridge_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train ridge regression probe and return R² on test set."""
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return r2, probe, y_pred


def experiment_1_direction_isolation(model, layer="post_ln2"):
    """
    Experiment 1: Keep direction, set constant norm

    If direction encoding is independent of norm, position should still be decodable.
    """
    print("\n=== Experiment 1: Direction Isolation ===")

    all_directions = []
    all_norms = []
    all_positions = []

    for _ in range(N_SAMPLES):
        tokens = torch.randint(0, 1000, (1, N_CTX), device=DEVICE)
        activations = get_activations_with_cache(model, tokens)
        act = activations[layer][0]  # (seq_len, d_model)

        directions, norms = extract_direction_and_norm(act)
        all_directions.append(directions.cpu().numpy())
        all_norms.append(norms.cpu().numpy())
        all_positions.append(np.arange(N_CTX))

    # Flatten
    directions = np.vstack(all_directions)  # (N_SAMPLES * N_CTX, d_model)
    norms = np.concatenate(all_norms)  # (N_SAMPLES * N_CTX,)
    positions = np.concatenate(all_positions)  # (N_SAMPLES * N_CTX,)

    # Split train/test
    n_train = int(0.8 * len(positions))
    idx = np.random.permutation(len(positions))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    results = {}

    # Baseline: Full activations
    full_act = directions * norms[:, np.newaxis]
    r2_full, _, _ = train_ridge_probe(
        full_act[train_idx],
        positions[train_idx],
        full_act[test_idx],
        positions[test_idx],
    )
    results["full_activations_r2"] = r2_full
    print(f"  Full activations R²: {r2_full:.4f}")

    # Direction only (constant norm = 1)
    r2_dir, _, _ = train_ridge_probe(
        directions[train_idx],
        positions[train_idx],
        directions[test_idx],
        positions[test_idx],
    )
    results["direction_only_r2"] = r2_dir
    print(f"  Direction only (norm=1) R²: {r2_dir:.4f}")

    # Norm only (1D feature)
    r2_norm, _, _ = train_ridge_probe(
        norms[train_idx].reshape(-1, 1),
        positions[train_idx],
        norms[test_idx].reshape(-1, 1),
        positions[test_idx],
    )
    results["norm_only_r2"] = r2_norm
    print(f"  Norm only R²: {r2_norm:.4f}")

    # Direction with random (shuffled) norms
    shuffled_norms = norms.copy()
    np.random.shuffle(shuffled_norms)
    dir_random_norm = directions * shuffled_norms[:, np.newaxis]
    r2_random_norm, _, _ = train_ridge_probe(
        dir_random_norm[train_idx],
        positions[train_idx],
        dir_random_norm[test_idx],
        positions[test_idx],
    )
    results["direction_random_norm_r2"] = r2_random_norm
    print(f"  Direction with random norm R²: {r2_random_norm:.4f}")

    # Direction with mean norm (constant but realistic)
    mean_norm = np.mean(norms)
    dir_mean_norm = directions * mean_norm
    r2_mean_norm, _, _ = train_ridge_probe(
        dir_mean_norm[train_idx],
        positions[train_idx],
        dir_mean_norm[test_idx],
        positions[test_idx],
    )
    results["direction_mean_norm_r2"] = r2_mean_norm
    print(f"  Direction with mean norm R²: {r2_mean_norm:.4f}")

    return results, directions, norms, positions, train_idx, test_idx


def experiment_2_cross_decoding(directions, norms, positions, train_idx, test_idx):
    """
    Experiment 2: Cross-decoding between direction and norm

    Train on direction, test on norm-constructed features and vice versa.
    If they encode position the same way, cross-decoding should work.
    """
    print("\n=== Experiment 2: Cross-Decoding ===")

    results = {}

    # Train probe on direction
    probe_dir = Ridge(alpha=1.0)
    probe_dir.fit(directions[train_idx], positions[train_idx])

    # Train probe on norm
    probe_norm = Ridge(alpha=1.0)
    probe_norm.fit(norms[train_idx].reshape(-1, 1), positions[train_idx])

    # Same-domain predictions
    pred_dir_dir = probe_dir.predict(directions[test_idx])
    pred_norm_norm = probe_norm.predict(norms[test_idx].reshape(-1, 1))

    r2_dir_dir = 1 - np.sum((positions[test_idx] - pred_dir_dir) ** 2) / np.sum(
        (positions[test_idx] - np.mean(positions[test_idx])) ** 2
    )
    r2_norm_norm = 1 - np.sum((positions[test_idx] - pred_norm_norm) ** 2) / np.sum(
        (positions[test_idx] - np.mean(positions[test_idx])) ** 2
    )

    results["same_domain_direction_r2"] = r2_dir_dir
    results["same_domain_norm_r2"] = r2_norm_norm

    print(f"  Same-domain direction R²: {r2_dir_dir:.4f}")
    print(f"  Same-domain norm R²: {r2_norm_norm:.4f}")

    # Cross-domain: Can direction probe's weights predict from norm?
    # This only makes sense if we look at correlation of predictions
    corr_predictions = np.corrcoef(pred_dir_dir, pred_norm_norm)[0, 1]
    results["prediction_correlation"] = corr_predictions
    print(f"  Correlation of direction vs norm predictions: {corr_predictions:.4f}")

    return results


def experiment_3_orthogonalize_norm(directions, norms, positions, train_idx, test_idx):
    """
    Experiment 3: Remove norm information from directions

    Project out the "norm direction" from the activation space.
    If direction encoding is truly independent, this shouldn't hurt performance much.
    """
    print("\n=== Experiment 3: Orthogonalize Norm from Direction ===")

    results = {}

    # Full activations
    full_act = directions * norms[:, np.newaxis]

    # Find the direction that correlates with norm
    # Method 1: Use norms as regression target for activations
    probe = Ridge(alpha=1.0)
    probe.fit(full_act[train_idx], norms[train_idx])
    norm_direction = probe.coef_ / (np.linalg.norm(probe.coef_) + 1e-8)  # Unit vector

    # Project out norm direction from all activations
    proj = np.outer(norm_direction, norm_direction)  # Projection matrix
    full_act_orthogonal = full_act - full_act @ proj

    # Also project out from directions
    directions_orthogonal = directions - directions @ proj

    # Test position decoding after removing norm direction
    r2_full_orth, _, _ = train_ridge_probe(
        full_act_orthogonal[train_idx],
        positions[train_idx],
        full_act_orthogonal[test_idx],
        positions[test_idx],
    )
    results["full_orthogonalized_r2"] = r2_full_orth
    print(f"  Full activations after removing norm direction R²: {r2_full_orth:.4f}")

    r2_dir_orth, _, _ = train_ridge_probe(
        directions_orthogonal[train_idx],
        positions[train_idx],
        directions_orthogonal[test_idx],
        positions[test_idx],
    )
    results["direction_orthogonalized_r2"] = r2_dir_orth
    print(f"  Directions after removing norm direction R²: {r2_dir_orth:.4f}")

    # How much variance did we remove?
    var_original = np.var(full_act)
    var_orthogonal = np.var(full_act_orthogonal)
    var_removed_frac = 1 - var_orthogonal / var_original
    results["variance_removed_fraction"] = var_removed_frac
    print(f"  Fraction of variance removed: {var_removed_frac:.4f}")

    return results


def experiment_4_pca_analysis(directions, norms, positions, train_idx, test_idx):
    """
    Experiment 4: PCA analysis of position encoding

    What principal components encode position?
    How are they related to norm?
    """
    print("\n=== Experiment 4: PCA Analysis ===")

    results = {}

    # Full activations
    full_act = directions * norms[:, np.newaxis]

    # PCA on training set
    pca = PCA(n_components=min(50, full_act.shape[1]))
    full_act_pca = pca.fit_transform(full_act)

    # Correlation of each PC with position and norm
    pc_pos_corrs = []
    pc_norm_corrs = []

    for i in range(min(20, full_act_pca.shape[1])):
        corr_pos = np.corrcoef(full_act_pca[:, i], positions)[0, 1]
        corr_norm = np.corrcoef(full_act_pca[:, i], norms)[0, 1]
        pc_pos_corrs.append(corr_pos)
        pc_norm_corrs.append(corr_norm)

    results["pc_position_correlations"] = pc_pos_corrs
    results["pc_norm_correlations"] = pc_norm_corrs
    results["explained_variance_ratio"] = pca.explained_variance_ratio_[:20].tolist()

    print(f"  Top 5 PCs position correlations: {pc_pos_corrs[:5]}")
    print(f"  Top 5 PCs norm correlations: {pc_norm_corrs[:5]}")

    # Find the PC most correlated with position
    best_pos_pc = np.argmax(np.abs(pc_pos_corrs))
    best_norm_pc = np.argmax(np.abs(pc_norm_corrs))

    results["best_position_pc"] = int(best_pos_pc)
    results["best_norm_pc"] = int(best_norm_pc)
    results["best_position_pc_corr"] = pc_pos_corrs[best_pos_pc]
    results["best_norm_pc_corr"] = pc_norm_corrs[best_norm_pc]

    print(
        f"  Best PC for position: PC{best_pos_pc} (r={pc_pos_corrs[best_pos_pc]:.4f})"
    )
    print(f"  Best PC for norm: PC{best_norm_pc} (r={pc_norm_corrs[best_norm_pc]:.4f})")

    # Are position and norm PCs the same?
    if best_pos_pc == best_norm_pc:
        print(f"  -> Position and norm are encoded by the SAME PC!")
    else:
        print(f"  -> Position and norm are encoded by DIFFERENT PCs")
        # Correlation between the two best PCs
        corr_best_pcs = np.corrcoef(
            full_act_pca[:, best_pos_pc], full_act_pca[:, best_norm_pc]
        )[0, 1]
        results["correlation_best_pcs"] = corr_best_pcs
        print(f"  -> Correlation between best position/norm PCs: {corr_best_pcs:.4f}")

    # How much position information is in top K PCs?
    for k in [1, 5, 10, 20]:
        if k <= full_act_pca.shape[1]:
            r2_k, _, _ = train_ridge_probe(
                full_act_pca[train_idx, :k],
                positions[train_idx],
                full_act_pca[test_idx, :k],
                positions[test_idx],
            )
            results[f"top_{k}_pcs_r2"] = r2_k
            print(f"  Position R² with top {k} PCs: {r2_k:.4f}")

    return results


def experiment_5_position_specific_analysis(model, layer="post_ln2"):
    """
    Experiment 5: Analyze activations at specific positions

    Look at how activations cluster by position.
    """
    print("\n=== Experiment 5: Position-Specific Analysis ===")

    results = {}

    # Collect activations by position
    activations_by_pos = {i: [] for i in range(N_CTX)}

    for _ in range(N_SAMPLES):
        tokens = torch.randint(0, 1000, (1, N_CTX), device=DEVICE)
        activations = get_activations_with_cache(model, tokens)
        act = activations[layer][0].cpu().numpy()

        for i in range(N_CTX):
            activations_by_pos[i].append(act[i])

    # Stack by position
    for i in range(N_CTX):
        activations_by_pos[i] = np.stack(activations_by_pos[i])  # (N_SAMPLES, d_model)

    # Compute mean and std for each position
    mean_by_pos = np.stack(
        [np.mean(activations_by_pos[i], axis=0) for i in range(N_CTX)]
    )
    std_by_pos = np.stack([np.std(activations_by_pos[i], axis=0) for i in range(N_CTX)])

    # Mean activation norm by position
    mean_norm_by_pos = [
        np.mean(np.linalg.norm(activations_by_pos[i], axis=1)) for i in range(N_CTX)
    ]
    results["mean_norm_by_position"] = mean_norm_by_pos

    # Variance of activations by position (how consistent is each position?)
    var_by_pos = [np.mean(np.var(activations_by_pos[i], axis=0)) for i in range(N_CTX)]
    results["variance_by_position"] = var_by_pos

    # Cosine similarity between mean activations of adjacent positions
    cos_sim_adjacent = []
    for i in range(N_CTX - 1):
        cos_sim = np.dot(mean_by_pos[i], mean_by_pos[i + 1]) / (
            np.linalg.norm(mean_by_pos[i]) * np.linalg.norm(mean_by_pos[i + 1]) + 1e-8
        )
        cos_sim_adjacent.append(cos_sim)
    results["cosine_similarity_adjacent"] = cos_sim_adjacent

    print(
        f"  Mean norm range: {min(mean_norm_by_pos):.4f} to {max(mean_norm_by_pos):.4f}"
    )
    print(f"  Variance range: {min(var_by_pos):.6f} to {max(var_by_pos):.6f}")
    print(f"  Mean adjacent cosine similarity: {np.mean(cos_sim_adjacent):.4f}")

    # Can we decode position from mean activation alone?
    # (i.e., is there a consistent "prototype" for each position?)
    prototype_based_accuracy = 0
    for i in range(N_CTX):
        # For each sample at position i, find the closest mean
        for sample_act in activations_by_pos[i][:100]:  # Use subset for speed
            distances = [
                np.linalg.norm(sample_act - mean_by_pos[j]) for j in range(N_CTX)
            ]
            predicted_pos = np.argmin(distances)
            if predicted_pos == i:
                prototype_based_accuracy += 1

    prototype_based_accuracy /= N_CTX * 100
    results["prototype_classification_accuracy"] = prototype_based_accuracy
    print(f"  Prototype-based classification accuracy: {prototype_based_accuracy:.4f}")

    return results


def experiment_6_layer_comparison(model):
    """
    Experiment 6: Compare direction vs norm importance across layers
    """
    print("\n=== Experiment 6: Layer-by-Layer Direction vs Norm ===")

    results = {}
    layers = [
        "embed",
        "post_ln1",
        "post_attn",
        "post_attn_residual",
        "post_ln2",
        "post_mlp_residual",
    ]

    for layer in layers:
        all_directions = []
        all_norms = []
        all_positions = []

        for _ in range(N_SAMPLES):
            tokens = torch.randint(0, 1000, (1, N_CTX), device=DEVICE)
            activations = get_activations_with_cache(model, tokens)
            act = activations[layer][0]

            directions, norms = extract_direction_and_norm(act)
            all_directions.append(directions.cpu().numpy())
            all_norms.append(norms.cpu().numpy())
            all_positions.append(np.arange(N_CTX))

        directions = np.vstack(all_directions)
        norms = np.concatenate(all_norms)
        positions = np.concatenate(all_positions)

        n_train = int(0.8 * len(positions))
        idx = np.random.permutation(len(positions))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        # Direction only
        r2_dir, _, _ = train_ridge_probe(
            directions[train_idx],
            positions[train_idx],
            directions[test_idx],
            positions[test_idx],
        )

        # Norm only
        r2_norm, _, _ = train_ridge_probe(
            norms[train_idx].reshape(-1, 1),
            positions[train_idx],
            norms[test_idx].reshape(-1, 1),
            positions[test_idx],
        )

        # Full
        full_act = directions * norms[:, np.newaxis]
        r2_full, _, _ = train_ridge_probe(
            full_act[train_idx],
            positions[train_idx],
            full_act[test_idx],
            positions[test_idx],
        )

        results[layer] = {
            "direction_r2": r2_dir,
            "norm_r2": r2_norm,
            "full_r2": r2_full,
            "direction_dominance": r2_dir / (r2_dir + r2_norm + 1e-8),
        }

        print(
            f"  {layer}: dir R²={r2_dir:.4f}, norm R²={r2_norm:.4f}, full R²={r2_full:.4f}"
        )

    return results


def main():
    print("=" * 60)
    print("Direction vs Norm Independence Analysis")
    print("=" * 60)

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    print(f"\nUsing device: {DEVICE}")
    model = create_random_model()
    print("Created random NoPE model")

    all_results = {}

    # Run experiments
    # Experiment 1: Direction isolation
    results_1, directions, norms, positions, train_idx, test_idx = (
        experiment_1_direction_isolation(model, layer="post_ln2")
    )
    all_results["direction_isolation"] = results_1

    # Experiment 2: Cross-decoding
    results_2 = experiment_2_cross_decoding(
        directions, norms, positions, train_idx, test_idx
    )
    all_results["cross_decoding"] = results_2

    # Experiment 3: Orthogonalize norm
    results_3 = experiment_3_orthogonalize_norm(
        directions, norms, positions, train_idx, test_idx
    )
    all_results["orthogonalize_norm"] = results_3

    # Experiment 4: PCA analysis
    results_4 = experiment_4_pca_analysis(
        directions, norms, positions, train_idx, test_idx
    )
    all_results["pca_analysis"] = results_4

    # Experiment 5: Position-specific analysis
    results_5 = experiment_5_position_specific_analysis(model, layer="post_ln2")
    all_results["position_specific"] = results_5

    # Experiment 6: Layer comparison
    results_6 = experiment_6_layer_comparison(model)
    all_results["layer_comparison"] = results_6

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Direction vs Norm Independence")
    print("=" * 60)

    print("\n1. Direction Isolation (post_ln2 layer):")
    print(f"   - Direction only R²: {results_1['direction_only_r2']:.4f}")
    print(f"   - Norm only R²: {results_1['norm_only_r2']:.4f}")
    print(
        f"   -> Direction carries {results_1['direction_only_r2'] / results_1['full_activations_r2'] * 100:.1f}% of position info"
    )
    print(
        f"   -> Norm carries {results_1['norm_only_r2'] / results_1['full_activations_r2'] * 100:.1f}% of position info"
    )

    print("\n2. After Orthogonalizing Norm Direction:")
    print(
        f"   - R² dropped from {results_1['full_activations_r2']:.4f} to {results_3['full_orthogonalized_r2']:.4f}"
    )
    print(f"   - Variance removed: {results_3['variance_removed_fraction'] * 100:.2f}%")

    print("\n3. PCA Analysis:")
    print(
        f"   - Best PC for position: PC{results_4['best_position_pc']} (r={results_4['best_position_pc_corr']:.4f})"
    )
    print(
        f"   - Best PC for norm: PC{results_4['best_norm_pc']} (r={results_4['best_norm_pc_corr']:.4f})"
    )

    # Conclusions
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    dir_r2 = results_1["direction_only_r2"]
    norm_r2 = results_1["norm_only_r2"]

    if dir_r2 > norm_r2 * 1.5:
        print("-> Direction is DOMINANT for position encoding")
    elif norm_r2 > dir_r2 * 1.5:
        print("-> Norm is DOMINANT for position encoding")
    else:
        print("-> Direction and Norm are ROUGHLY EQUAL for position encoding")

    if results_4["best_position_pc"] == results_4["best_norm_pc"]:
        print("-> Position and Norm are encoded in the SAME principal component")
        print("   (They are manifestations of the same underlying structure)")
    else:
        print("-> Position and Norm are encoded in DIFFERENT principal components")
        print("   (They provide partially independent information)")

    # Save results
    with open(RESULTS_DIR / "direction_norm_independence_results.json", "w") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        json.dump(convert_to_serializable(all_results), f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
