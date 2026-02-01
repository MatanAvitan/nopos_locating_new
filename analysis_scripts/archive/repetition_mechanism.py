"""
Repetition Mechanism Analysis for Position Encoding in NoPE Transformers

Hypothesis: Token repetition creates a "dominant direction" in activation space after causal
attention. Even though LayerNorm destroys magnitude/norm information, this dominant direction
survives. The "sharpness" of this dominant direction encodes how many repetitions occurred,
which correlates with position (later positions → more repetitions statistically).

Experiments:
1. Controlled Repetition Rate: Vary vocab size to control repetition rate
2. Dominant Direction Sharpness: Measure participation ratio, energy concentration, kurtosis
3. Repetition Count → Activation Structure: Direct correlation analysis
4. Before vs After LN2: Compare position information survival
5. Natural Language vs Random Tokens: Compare repetition structure

Author: Analysis script for NoPE position encoding paper
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.stats import pearsonr, spearmanr, kurtosis
from scipy.linalg import svd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "repetition_mechanism"
PLOTS_DIR = Path(__file__).parent.parent / "overleaf" / "nopos---claude-version" / "plots"


def create_random_nope_model(
    n_layer=1,
    n_head=12,
    n_embd=768,
    block_size=128,
    vocab_size=50257,
    seed=42,
):
    """Create a randomly initialized NoPE GPT-2 model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = GPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=0.0,
        bias=False,
        use_positional_embedding=False,
        norm_type="layernorm",
        log_attention_stats=False,
    )

    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model


def get_activations_hook(model, tokens):
    """
    Extract activations at key points in the first transformer block.

    Returns dict with:
    - embed: Token embeddings
    - post_attn: Attention output (before residual)
    - post_attn_residual: After attention + residual (input to LN2)
    - post_ln2: After LN2 (before MLP)
    """
    activations = {}

    with torch.no_grad():
        # Token embeddings
        tok_emb = model.transformer.wte(tokens)
        activations["embed"] = tok_emb.clone()

        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[0]

        # Post LN1
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.clone()

        # Attention output (before residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # Post attention + residual (input to LN2)
        x_attn = x + attn_out
        activations["post_attn_residual"] = x_attn.clone()

        # Post LN2
        if not block.skip_ln2:
            x_ln2 = block.ln_2(x_attn)
            activations["post_ln2"] = x_ln2.clone()

    return activations


def generate_controlled_sequences(n_samples, seq_len, effective_vocab_size, full_vocab_size=50257):
    """
    Generate sequences with controlled repetition rate by limiting effective vocabulary.

    Args:
        n_samples: Number of sequences to generate
        seq_len: Sequence length
        effective_vocab_size: Number of unique tokens to use (smaller = more repetition)
        full_vocab_size: Full vocabulary size of the model

    Returns:
        tokens: (n_samples, seq_len) tensor
    """
    # Sample from a restricted vocabulary
    effective_vocab = torch.arange(effective_vocab_size)

    # Generate random indices into the effective vocab
    indices = torch.randint(0, effective_vocab_size, (n_samples, seq_len))

    return indices.to(DEVICE)


def compute_participation_ratio(activations):
    """
    Compute participation ratio (effective dimensionality) of activations.

    PR = (Σλᵢ)² / Σλᵢ²

    Low PR = one dominant direction, High PR = spread across dimensions

    Args:
        activations: (batch, seq_len, d_model) tensor

    Returns:
        pr: (batch, seq_len) tensor of participation ratios
    """
    batch, seq_len, d_model = activations.shape

    # For each position in each sample, compute PR
    pr = torch.zeros(batch, seq_len)

    for b in range(batch):
        for j in range(seq_len):
            vec = activations[b, j, :]  # (d_model,)

            # Compute squared components
            vec_sq = vec ** 2

            # PR = (sum of squared values)^2 / sum of fourth powers
            # This is equivalent to (||v||²)² / ||v||⁴ for a vector
            # But we want dimensionality, so we use SVD approach

            # For a single vector, PR is not well-defined
            # Instead, compute based on component distribution
            vec_normalized = vec_sq / (vec_sq.sum() + 1e-10)

            # Effective dimensionality = 1 / Σpᵢ² (inverse Herfindahl index)
            pr[b, j] = 1.0 / (vec_normalized ** 2).sum()

    return pr


def compute_participation_ratio_batch(activations):
    """
    Batched computation of participation ratio.

    Uses the squared component distribution interpretation:
    PR = 1 / Σ(vᵢ²/||v||²)²

    Low PR = peaked distribution = dominant direction
    High PR = flat distribution = spread across dimensions
    """
    # activations: (batch, seq_len, d_model)

    # Squared components
    vec_sq = activations ** 2  # (batch, seq_len, d_model)

    # Normalize to get probability distribution
    vec_sq_sum = vec_sq.sum(dim=-1, keepdim=True) + 1e-10
    p = vec_sq / vec_sq_sum  # (batch, seq_len, d_model)

    # Effective dimensionality = 1 / Σpᵢ²
    p_sq = p ** 2
    pr = 1.0 / (p_sq.sum(dim=-1) + 1e-10)  # (batch, seq_len)

    return pr


def compute_top_k_energy(activations, k_values=[1, 5, 10]):
    """
    Compute fraction of energy in top-k components after SVD.

    For each position, we compute what fraction of ||activation||² is
    explained by the top-k singular values when considering the batch
    dimension as samples.

    Args:
        activations: (batch, seq_len, d_model) tensor
        k_values: list of k values to compute

    Returns:
        results: dict mapping k -> (seq_len,) array of energy fractions
    """
    batch, seq_len, d_model = activations.shape

    results = {}
    for k in k_values:
        results[k] = np.zeros(seq_len)

    # For each position, compute SVD across batch
    for j in range(seq_len):
        # Get all activations at position j: (batch, d_model)
        X = activations[:, j, :].cpu().numpy()

        # Center the data
        X_centered = X - X.mean(axis=0)

        # SVD
        try:
            U, s, Vt = svd(X_centered, full_matrices=False)

            # Total variance = sum of squared singular values
            total_var = (s ** 2).sum()

            for k in k_values:
                top_k_var = (s[:k] ** 2).sum()
                results[k][j] = top_k_var / (total_var + 1e-10)
        except:
            for k in k_values:
                results[k][j] = np.nan

    return results


def compute_kurtosis_batch(activations):
    """
    Compute kurtosis of activation components at each position.

    High kurtosis = peaked distribution = dominant direction

    Args:
        activations: (batch, seq_len, d_model) tensor

    Returns:
        kurt: (batch, seq_len) tensor
    """
    # Use scipy's kurtosis (Fisher's definition, excess kurtosis)
    act_np = activations.cpu().numpy()
    batch, seq_len, d_model = act_np.shape

    kurt = np.zeros((batch, seq_len))
    for b in range(batch):
        for j in range(seq_len):
            kurt[b, j] = kurtosis(act_np[b, j, :], fisher=True)

    return torch.from_numpy(kurt)


def compute_repetition_count(tokens):
    """
    For each position j, compute the count of the most-repeated token up to that position.

    Args:
        tokens: (batch, seq_len) tensor

    Returns:
        max_rep_count: (batch, seq_len) tensor
    """
    batch, seq_len = tokens.shape
    max_rep_count = torch.zeros(batch, seq_len, dtype=torch.long)

    tokens_np = tokens.cpu().numpy()

    for b in range(batch):
        for j in range(seq_len):
            # Count tokens up to and including position j
            token_counts = Counter(tokens_np[b, :j+1])
            max_rep_count[b, j] = max(token_counts.values())

    return max_rep_count


def compute_dominant_token_projection(activations, tokens, model):
    """
    For positions where we know which token repeated most, compute projection
    onto that token's value vector direction.

    projection = activation · v_repeated / ||activation||

    Args:
        activations: (batch, seq_len, d_model) tensor (post-attention)
        tokens: (batch, seq_len) tensor
        model: The GPT model (to get W_v and embeddings)

    Returns:
        projections: (batch, seq_len) tensor
    """
    batch, seq_len, d_model = activations.shape
    projections = torch.zeros(batch, seq_len)

    tokens_np = tokens.cpu().numpy()

    with torch.no_grad():
        block = model.transformer.h[0]
        # W_v is part of c_attn: [n_embd, 3*n_embd] -> last third is V
        c_attn_weight = block.attn.c_attn.weight  # (3*n_embd, n_embd)
        W_v = c_attn_weight[2*d_model:, :]  # (n_embd, n_embd) - V projection

        embedding_weight = model.transformer.wte.weight  # (vocab_size, n_embd)

        for b in range(batch):
            for j in range(seq_len):
                # Find most repeated token up to position j
                token_counts = Counter(tokens_np[b, :j+1])
                most_common_token = token_counts.most_common(1)[0][0]

                # Get the value vector for this token
                # v = W_v @ e_token
                e_token = embedding_weight[most_common_token]  # (n_embd,)

                # Apply LN1 to embedding (as model does)
                e_ln = block.ln_1(e_token.unsqueeze(0).unsqueeze(0)).squeeze()

                # Get value vector
                v_token = W_v @ e_ln  # (n_embd,)

                # Compute projection
                act = activations[b, j, :]
                proj = torch.dot(act, v_token) / (act.norm() + 1e-10)
                projections[b, j] = proj.abs().item()

    return projections


def experiment_1_controlled_repetition(model, n_samples=500, seq_len=128):
    """
    Experiment 1: Controlled Repetition Rate

    Create synthetic sequences with controlled repetition by varying vocab size.
    Measure sharpness metrics at each position for each condition.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Controlled Repetition Rate")
    print("="*70)

    # Repetition conditions: (name, effective_vocab_size)
    # With seq_len=128:
    # - vocab=128: ~0% forced repetition (all unique possible)
    # - vocab=96: ~25% positions will see repetition
    # - vocab=64: ~50% must repeat
    # - vocab=32: ~75% must repeat
    # - vocab=13: ~90% must repeat
    conditions = [
        ("0% forced", seq_len),      # Can be all unique
        ("~25% rep", 96),
        ("~50% rep", 64),
        ("~75% rep", 32),
        ("~90% rep", 13),
    ]

    results = {}

    for name, vocab_size in conditions:
        print(f"\nCondition: {name} (vocab={vocab_size})")

        # Generate sequences
        tokens = generate_controlled_sequences(n_samples, seq_len, vocab_size)

        # Get activations
        activations = get_activations_hook(model, tokens)

        # Compute metrics for post-attention and post-ln2
        metrics = {}
        for layer in ["post_attn", "post_attn_residual", "post_ln2"]:
            if layer not in activations:
                continue

            act = activations[layer]

            # Participation ratio
            pr = compute_participation_ratio_batch(act)
            metrics[f"{layer}_pr_mean"] = pr.mean(dim=0).cpu().numpy()
            metrics[f"{layer}_pr_std"] = pr.std(dim=0).cpu().numpy()

            # Kurtosis
            kurt = compute_kurtosis_batch(act)
            metrics[f"{layer}_kurtosis_mean"] = kurt.mean(dim=0).cpu().numpy()
            metrics[f"{layer}_kurtosis_std"] = kurt.std(dim=0).cpu().numpy()

            # Norm
            norms = act.norm(dim=-1)
            metrics[f"{layer}_norm_mean"] = norms.mean(dim=0).cpu().numpy()
            metrics[f"{layer}_norm_std"] = norms.std(dim=0).cpu().numpy()

        # Compute actual repetition counts
        rep_counts = compute_repetition_count(tokens)
        metrics["rep_count_mean"] = rep_counts.float().mean(dim=0).cpu().numpy()
        metrics["rep_count_std"] = rep_counts.float().std(dim=0).cpu().numpy()

        results[name] = metrics

        # Print summary
        if "post_attn_pr_mean" in metrics:
            print(f"  Post-attn PR: {metrics['post_attn_pr_mean'].mean():.2f} (mean across positions)")
        if "post_ln2_pr_mean" in metrics:
            print(f"  Post-LN2 PR: {metrics['post_ln2_pr_mean'].mean():.2f}")
        print(f"  Mean repetition count: {metrics['rep_count_mean'].mean():.2f}")

    return results


def experiment_2_sharpness_metrics(model, n_samples=500, seq_len=128, effective_vocab=32):
    """
    Experiment 2: Measure all sharpness metrics in detail

    For a moderate repetition condition, compute:
    - Participation Ratio
    - Top-k Energy Concentration
    - Projection onto repeated token's direction
    - Kurtosis
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Dominant Direction Sharpness Metrics")
    print("="*70)

    # Generate sequences with moderate repetition
    tokens = generate_controlled_sequences(n_samples, seq_len, effective_vocab)

    # Get activations
    activations = get_activations_hook(model, tokens)

    results = {}

    for layer in ["post_attn", "post_attn_residual", "post_ln2"]:
        if layer not in activations:
            continue

        print(f"\nLayer: {layer}")
        act = activations[layer]

        # 1. Participation Ratio
        pr = compute_participation_ratio_batch(act)
        results[f"{layer}_pr"] = pr.cpu().numpy()
        print(f"  Participation Ratio: mean={pr.mean():.2f}, std={pr.std():.2f}")

        # 2. Top-k Energy Concentration
        top_k_energy = compute_top_k_energy(act, k_values=[1, 5, 10, 50])
        for k, energy in top_k_energy.items():
            results[f"{layer}_top{k}_energy"] = energy
        print(f"  Top-1 Energy: {top_k_energy[1].mean():.4f}")
        print(f"  Top-5 Energy: {top_k_energy[5].mean():.4f}")
        print(f"  Top-10 Energy: {top_k_energy[10].mean():.4f}")

        # 3. Kurtosis
        kurt = compute_kurtosis_batch(act)
        results[f"{layer}_kurtosis"] = kurt.cpu().numpy()
        print(f"  Kurtosis: mean={kurt.mean():.2f}, std={kurt.std():.2f}")

        # 4. Norm (for comparison)
        norms = act.norm(dim=-1)
        results[f"{layer}_norm"] = norms.cpu().numpy()

    # 5. Projection onto repeated token's direction (only for post_attn)
    if "post_attn" in activations:
        print("\nComputing projection onto dominant token direction...")
        projections = compute_dominant_token_projection(
            activations["post_attn"], tokens, model
        )
        results["post_attn_projection"] = projections.cpu().numpy()
        print(f"  Mean projection: {projections.mean():.4f}")

    # Also store positions and repetition counts
    positions = np.arange(seq_len)
    results["positions"] = positions

    rep_counts = compute_repetition_count(tokens)
    results["rep_counts"] = rep_counts.numpy()

    return results


def experiment_3_repetition_vs_structure(model, n_samples=1000, seq_len=128):
    """
    Experiment 3: Direct correlation between repetition count and activation structure

    For each position j:
    1. Count max repetitions up to j
    2. Measure sharpness metrics
    3. Compute correlation
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Repetition Count → Activation Structure")
    print("="*70)

    # Use moderate vocab for interesting repetition patterns
    effective_vocab = 32
    tokens = generate_controlled_sequences(n_samples, seq_len, effective_vocab)

    # Get activations
    activations = get_activations_hook(model, tokens)

    # Compute repetition counts
    rep_counts = compute_repetition_count(tokens)
    rep_counts_flat = rep_counts.flatten().float().cpu().numpy()
    positions_flat = np.tile(np.arange(seq_len), n_samples)

    results = {
        "correlations": {},
        "per_position": {},
    }

    for layer in ["post_attn", "post_attn_residual", "post_ln2"]:
        if layer not in activations:
            continue

        print(f"\nLayer: {layer}")
        act = activations[layer]

        # Participation Ratio
        pr = compute_participation_ratio_batch(act)
        pr_flat = pr.flatten().cpu().numpy()

        # Inverse PR (sharpness) - lower PR = sharper
        inv_pr = 1.0 / (pr_flat + 1e-10)

        # Kurtosis
        kurt = compute_kurtosis_batch(act)
        kurt_flat = kurt.flatten().cpu().numpy()

        # Norm
        norms = act.norm(dim=-1).flatten().cpu().numpy()

        # Correlations with repetition count
        r_pr, p_pr = pearsonr(rep_counts_flat, pr_flat)
        r_inv_pr, p_inv_pr = pearsonr(rep_counts_flat, inv_pr)
        r_kurt, p_kurt = pearsonr(rep_counts_flat, kurt_flat)
        r_norm, p_norm = pearsonr(rep_counts_flat, norms)

        results["correlations"][layer] = {
            "pr_vs_rep": (r_pr, p_pr),
            "inv_pr_vs_rep": (r_inv_pr, p_inv_pr),
            "kurtosis_vs_rep": (r_kurt, p_kurt),
            "norm_vs_rep": (r_norm, p_norm),
        }

        print(f"  PR vs RepCount: r={r_pr:.4f}, p={p_pr:.2e}")
        print(f"  1/PR vs RepCount: r={r_inv_pr:.4f}, p={p_inv_pr:.2e}")
        print(f"  Kurtosis vs RepCount: r={r_kurt:.4f}, p={p_kurt:.2e}")
        print(f"  Norm vs RepCount: r={r_norm:.4f}, p={p_norm:.2e}")

        # Also correlate with position
        r_pr_pos, _ = pearsonr(positions_flat, pr_flat)
        r_kurt_pos, _ = pearsonr(positions_flat, kurt_flat)
        r_norm_pos, _ = pearsonr(positions_flat, norms)

        print(f"  PR vs Position: r={r_pr_pos:.4f}")
        print(f"  Kurtosis vs Position: r={r_kurt_pos:.4f}")
        print(f"  Norm vs Position: r={r_norm_pos:.4f}")

        # Per-position averages
        pr_per_pos = pr.mean(dim=0).cpu().numpy()
        kurt_per_pos = kurt.mean(dim=0).cpu().numpy()
        norm_per_pos = act.norm(dim=-1).mean(dim=0).cpu().numpy()

        results["per_position"][layer] = {
            "pr": pr_per_pos,
            "kurtosis": kurt_per_pos,
            "norm": norm_per_pos,
        }

    rep_count_per_pos = rep_counts.float().mean(dim=0).cpu().numpy()
    results["per_position"]["rep_count"] = rep_count_per_pos

    return results


def experiment_4_before_after_ln2(model, n_samples=500, seq_len=128, effective_vocab=32):
    """
    Experiment 4: Compare position information before vs after LN2

    Key hypothesis: Norm information is destroyed by LN2, but dominant direction
    sharpness survives.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Before vs After LayerNorm2")
    print("="*70)

    tokens = generate_controlled_sequences(n_samples, seq_len, effective_vocab)
    activations = get_activations_hook(model, tokens)

    positions = torch.arange(seq_len).float()
    positions_flat = positions.repeat(n_samples).numpy()

    results = {
        "norm": {},
        "direction_sharpness": {},
        "linear_probe_r2": {},
    }

    for layer in ["post_attn_residual", "post_ln2"]:
        if layer not in activations:
            continue

        print(f"\nLayer: {layer}")
        act = activations[layer]

        # Norm analysis
        norms = act.norm(dim=-1).flatten().cpu().numpy()
        r_norm, _ = pearsonr(positions_flat, norms)

        # Norm after LN should be nearly constant
        norm_std = np.std(norms)
        results["norm"][layer] = {
            "r_vs_position": r_norm,
            "std": norm_std,
            "mean": np.mean(norms),
        }
        print(f"  Norm: mean={np.mean(norms):.4f}, std={norm_std:.4f}")
        print(f"  Norm vs Position: r={r_norm:.4f}")

        # Direction sharpness (Participation Ratio)
        pr = compute_participation_ratio_batch(act)
        pr_flat = pr.flatten().cpu().numpy()
        r_pr, _ = pearsonr(positions_flat, pr_flat)

        # Kurtosis
        kurt = compute_kurtosis_batch(act)
        kurt_flat = kurt.flatten().cpu().numpy()
        r_kurt, _ = pearsonr(positions_flat, kurt_flat)

        results["direction_sharpness"][layer] = {
            "pr_r_vs_position": r_pr,
            "pr_mean": np.mean(pr_flat),
            "kurt_r_vs_position": r_kurt,
            "kurt_mean": np.mean(kurt_flat),
        }
        print(f"  PR vs Position: r={r_pr:.4f}")
        print(f"  Kurtosis vs Position: r={r_kurt:.4f}")

        # Linear probe on full activations vs position
        act_flat = act.reshape(-1, act.shape[-1]).cpu().numpy()

        n_train = int(0.8 * len(positions_flat))
        idx = np.random.permutation(len(positions_flat))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        probe = Ridge(alpha=1.0)
        probe.fit(act_flat[train_idx], positions_flat[train_idx])
        pred = probe.predict(act_flat[test_idx])
        r2 = r2_score(positions_flat[test_idx], pred)
        r_probe, _ = pearsonr(positions_flat[test_idx], pred)

        results["linear_probe_r2"][layer] = {
            "r2": r2,
            "pearson_r": r_probe,
        }
        print(f"  Linear Probe: R²={r2:.4f}, r={r_probe:.4f}")

    return results


def experiment_5_natural_vs_random(model, n_samples=500, seq_len=128):
    """
    Experiment 5: Natural Language vs Random Tokens

    Compare:
    - Natural language (simulated with Zipf distribution)
    - Random tokens (uniform distribution)

    Natural language has more repetition at later positions.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Natural Language vs Random Tokens")
    print("="*70)

    vocab_size = model.config.vocab_size

    results = {
        "natural": {},
        "random": {},
    }

    # Generate "natural" language-like sequences with Zipf distribution
    # Zipf: P(rank k) ∝ 1/k^s, typical s ≈ 1.0
    print("\nGenerating natural-like sequences (Zipf distribution)...")
    s = 1.0
    ranks = np.arange(1, min(vocab_size, 10000) + 1)
    probs = 1.0 / (ranks ** s)
    probs = probs / probs.sum()

    natural_tokens = np.random.choice(
        len(probs),
        size=(n_samples, seq_len),
        p=probs
    )
    natural_tokens = torch.from_numpy(natural_tokens).long().to(DEVICE)

    # Generate random tokens (uniform)
    print("Generating random sequences (uniform distribution)...")
    random_tokens = torch.randint(0, vocab_size, (n_samples, seq_len), device=DEVICE)

    for name, tokens in [("natural", natural_tokens), ("random", random_tokens)]:
        print(f"\n{name.upper()} tokens:")

        activations = get_activations_hook(model, tokens)
        rep_counts = compute_repetition_count(tokens)

        # Mean repetition per position
        rep_mean = rep_counts.float().mean(dim=0).cpu().numpy()
        print(f"  Mean max repetition at pos 0: {rep_mean[0]:.2f}")
        print(f"  Mean max repetition at pos {seq_len-1}: {rep_mean[-1]:.2f}")

        positions = np.arange(seq_len)
        positions_flat = np.tile(positions, n_samples)

        for layer in ["post_attn_residual", "post_ln2"]:
            if layer not in activations:
                continue

            act = activations[layer]

            # Sharpness metrics
            pr = compute_participation_ratio_batch(act)
            pr_mean = pr.mean(dim=0).cpu().numpy()

            kurt = compute_kurtosis_batch(act)
            kurt_mean = kurt.mean(dim=0).cpu().numpy()

            norm = act.norm(dim=-1)
            norm_mean = norm.mean(dim=0).cpu().numpy()

            # Correlations with position
            r_pr, _ = pearsonr(positions, pr_mean)
            r_kurt, _ = pearsonr(positions, kurt_mean)
            r_norm, _ = pearsonr(positions, norm_mean)

            results[name][layer] = {
                "pr_mean_per_pos": pr_mean,
                "kurt_mean_per_pos": kurt_mean,
                "norm_mean_per_pos": norm_mean,
                "r_pr_vs_pos": r_pr,
                "r_kurt_vs_pos": r_kurt,
                "r_norm_vs_pos": r_norm,
            }

            print(f"  {layer}:")
            print(f"    PR vs Position: r={r_pr:.4f}")
            print(f"    Kurtosis vs Position: r={r_kurt:.4f}")
            print(f"    Norm vs Position: r={r_norm:.4f}")

        results[name]["rep_count_per_pos"] = rep_mean

    return results


def plot_experiment_1(results, output_path):
    """Plot Experiment 1: Controlled repetition rate results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    conditions = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(conditions)))

    seq_len = len(results[conditions[0]]["rep_count_mean"])
    positions = np.arange(seq_len)

    # Plot 1: Repetition count over positions
    ax = axes[0, 0]
    for i, cond in enumerate(conditions):
        ax.plot(positions, results[cond]["rep_count_mean"],
                color=colors[i], label=cond, linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Max Repetition Count")
    ax.set_title("Repetition Count by Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Participation Ratio (post-attention)
    ax = axes[0, 1]
    for i, cond in enumerate(conditions):
        if "post_attn_pr_mean" in results[cond]:
            ax.plot(positions, results[cond]["post_attn_pr_mean"],
                    color=colors[i], label=cond, linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("PR (Post-Attention)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Participation Ratio (post-LN2)
    ax = axes[0, 2]
    for i, cond in enumerate(conditions):
        if "post_ln2_pr_mean" in results[cond]:
            ax.plot(positions, results[cond]["post_ln2_pr_mean"],
                    color=colors[i], label=cond, linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("PR (Post-LN2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Kurtosis (post-attention)
    ax = axes[1, 0]
    for i, cond in enumerate(conditions):
        if "post_attn_kurtosis_mean" in results[cond]:
            ax.plot(positions, results[cond]["post_attn_kurtosis_mean"],
                    color=colors[i], label=cond, linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Kurtosis (Post-Attention)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Kurtosis (post-LN2)
    ax = axes[1, 1]
    for i, cond in enumerate(conditions):
        if "post_ln2_kurtosis_mean" in results[cond]:
            ax.plot(positions, results[cond]["post_ln2_kurtosis_mean"],
                    color=colors[i], label=cond, linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Kurtosis (Post-LN2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: Norm (post-attention)
    ax = axes[1, 2]
    for i, cond in enumerate(conditions):
        if "post_attn_norm_mean" in results[cond]:
            ax.plot(positions, results[cond]["post_attn_norm_mean"],
                    color=colors[i], label=cond, linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Norm")
    ax.set_title("Norm (Post-Attention)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_experiment_4(results, output_path):
    """Plot Experiment 4: Before vs After LN2."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layers = ["post_attn_residual", "post_ln2"]
    layer_labels = ["Pre-LN2", "Post-LN2"]
    colors = ["blue", "orange"]

    # Plot 1: Norm vs Position correlation
    ax = axes[0]
    values = [results["norm"][l]["r_vs_position"] for l in layers if l in results["norm"]]
    bars = ax.bar(layer_labels[:len(values)], values, color=colors[:len(values)])
    ax.set_ylabel("Pearson r")
    ax.set_title("Norm vs Position Correlation")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Direction sharpness (PR) vs Position correlation
    ax = axes[1]
    values = [results["direction_sharpness"][l]["pr_r_vs_position"]
              for l in layers if l in results["direction_sharpness"]]
    bars = ax.bar(layer_labels[:len(values)], values, color=colors[:len(values)])
    ax.set_ylabel("Pearson r")
    ax.set_title("Participation Ratio vs Position")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Linear Probe R²
    ax = axes[2]
    values = [results["linear_probe_r2"][l]["r2"]
              for l in layers if l in results["linear_probe_r2"]]
    bars = ax.bar(layer_labels[:len(values)], values, color=colors[:len(values)])
    ax.set_ylabel("R²")
    ax.set_title("Linear Probe Performance")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_experiment_5(results, output_path, seq_len=128):
    """Plot Experiment 5: Natural vs Random comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    positions = np.arange(seq_len)

    # Row 1: Natural language
    # Plot 1a: Repetition count
    ax = axes[0, 0]
    ax.plot(positions, results["natural"]["rep_count_per_pos"], 'b-', linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Max Rep Count")
    ax.set_title("Natural: Repetition Count")
    ax.grid(True, alpha=0.3)

    # Plot 1b: PR post-LN2
    ax = axes[0, 1]
    if "post_ln2" in results["natural"]:
        ax.plot(positions, results["natural"]["post_ln2"]["pr_mean_per_pos"],
                'b-', linewidth=2)
        r = results["natural"]["post_ln2"]["r_pr_vs_pos"]
        ax.set_title(f"Natural: PR (r={r:.3f})")
    ax.set_xlabel("Position")
    ax.set_ylabel("Participation Ratio")
    ax.grid(True, alpha=0.3)

    # Plot 1c: Norm post-LN2
    ax = axes[0, 2]
    if "post_ln2" in results["natural"]:
        ax.plot(positions, results["natural"]["post_ln2"]["norm_mean_per_pos"],
                'b-', linewidth=2)
        r = results["natural"]["post_ln2"]["r_norm_vs_pos"]
        ax.set_title(f"Natural: Norm (r={r:.3f})")
    ax.set_xlabel("Position")
    ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)

    # Row 2: Random tokens
    ax = axes[1, 0]
    ax.plot(positions, results["random"]["rep_count_per_pos"], 'r-', linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Max Rep Count")
    ax.set_title("Random: Repetition Count")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "post_ln2" in results["random"]:
        ax.plot(positions, results["random"]["post_ln2"]["pr_mean_per_pos"],
                'r-', linewidth=2)
        r = results["random"]["post_ln2"]["r_pr_vs_pos"]
        ax.set_title(f"Random: PR (r={r:.3f})")
    ax.set_xlabel("Position")
    ax.set_ylabel("Participation Ratio")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if "post_ln2" in results["random"]:
        ax.plot(positions, results["random"]["post_ln2"]["norm_mean_per_pos"],
                'r-', linewidth=2)
        r = results["random"]["post_ln2"]["r_norm_vs_pos"]
        ax.set_title(f"Random: Norm (r={r:.3f})")
    ax.set_xlabel("Position")
    ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(all_results):
    """Create a summary table of all correlations."""

    print("\n" + "="*70)
    print("SUMMARY TABLE: Correlation with Position")
    print("="*70)

    print("\n{:<30} {:<15} {:<15} {:<15}".format(
        "Metric", "Before LN2", "After LN2", "Survives LN?"
    ))
    print("-" * 75)

    # From Experiment 4
    if "exp4" in all_results:
        exp4 = all_results["exp4"]

        # Norm
        norm_pre = exp4["norm"].get("post_attn_residual", {}).get("r_vs_position", np.nan)
        norm_post = exp4["norm"].get("post_ln2", {}).get("r_vs_position", np.nan)
        survives = "NO" if abs(norm_post) < 0.1 else "YES"
        print("{:<30} {:<15.4f} {:<15.4f} {:<15}".format(
            "Norm", norm_pre, norm_post, survives
        ))

        # PR
        pr_pre = exp4["direction_sharpness"].get("post_attn_residual", {}).get("pr_r_vs_position", np.nan)
        pr_post = exp4["direction_sharpness"].get("post_ln2", {}).get("pr_r_vs_position", np.nan)
        survives = "YES" if abs(pr_post) > 0.1 else "NO"
        print("{:<30} {:<15.4f} {:<15.4f} {:<15}".format(
            "Participation Ratio", pr_pre, pr_post, survives
        ))

        # Kurtosis
        kurt_pre = exp4["direction_sharpness"].get("post_attn_residual", {}).get("kurt_r_vs_position", np.nan)
        kurt_post = exp4["direction_sharpness"].get("post_ln2", {}).get("kurt_r_vs_position", np.nan)
        survives = "YES" if abs(kurt_post) > 0.1 else "NO"
        print("{:<30} {:<15.4f} {:<15.4f} {:<15}".format(
            "Kurtosis", kurt_pre, kurt_post, survives
        ))

        # Linear Probe R²
        probe_pre = exp4["linear_probe_r2"].get("post_attn_residual", {}).get("r2", np.nan)
        probe_post = exp4["linear_probe_r2"].get("post_ln2", {}).get("r2", np.nan)
        survives = "YES" if probe_post > 0.1 else "NO"
        print("{:<30} {:<15.4f} {:<15.4f} {:<15}".format(
            "Linear Probe R²", probe_pre, probe_post, survives
        ))

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("REPETITION MECHANISM ANALYSIS FOR NoPE TRANSFORMERS")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print(f"Samples: {args.n_samples}")
    print(f"Sequence length: {args.seq_len}")

    # Create model
    print("\nInitializing randomly initialized NoPE GPT-2 model...")
    model = create_random_nope_model(
        n_layer=1,
        n_head=12,
        n_embd=768,
        block_size=args.seq_len,
        vocab_size=50257,
        seed=args.seed,
    )

    all_results = {}

    # Experiment 1: Controlled Repetition Rate
    results_1 = experiment_1_controlled_repetition(
        model, n_samples=args.n_samples, seq_len=args.seq_len
    )
    all_results["exp1"] = results_1
    plot_experiment_1(results_1, RESULTS_DIR / "exp1_controlled_repetition.png")

    # Experiment 2: Sharpness Metrics
    results_2 = experiment_2_sharpness_metrics(
        model, n_samples=args.n_samples, seq_len=args.seq_len, effective_vocab=32
    )
    all_results["exp2"] = results_2

    # Experiment 3: Repetition → Structure
    results_3 = experiment_3_repetition_vs_structure(
        model, n_samples=args.n_samples * 2, seq_len=args.seq_len
    )
    all_results["exp3"] = results_3

    # Experiment 4: Before vs After LN2
    results_4 = experiment_4_before_after_ln2(
        model, n_samples=args.n_samples, seq_len=args.seq_len, effective_vocab=32
    )
    all_results["exp4"] = results_4
    plot_experiment_4(results_4, RESULTS_DIR / "exp4_before_after_ln2.png")

    # Experiment 5: Natural vs Random
    results_5 = experiment_5_natural_vs_random(
        model, n_samples=args.n_samples, seq_len=args.seq_len
    )
    all_results["exp5"] = results_5
    plot_experiment_5(results_5, RESULTS_DIR / "exp5_natural_vs_random.png", args.seq_len)

    # Summary
    create_summary_table(all_results)

    # Save results (convert numpy arrays for JSON serialization)
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(x) for x in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    results_path = RESULTS_DIR / "repetition_mechanism_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Copy plot to overleaf
    import shutil
    for plot_name in ["exp1_controlled_repetition.png", "exp4_before_after_ln2.png",
                      "exp5_natural_vs_random.png"]:
        src = RESULTS_DIR / plot_name
        dst = PLOTS_DIR / f"repetition_{plot_name}"
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied to: {dst}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
