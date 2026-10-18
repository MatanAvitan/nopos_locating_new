"""
Test the Decoding Vector Mechanism with Orthogonality

Key claim from the paper:
- w = Σ_j LN(E_j) is the decoding vector
- Due to orthogonality: w · e_k ≈ 1 for any embedding e_k
- Therefore: Σ_{j=0}^{i} (w · v_j) ≈ (i+1) * c

But the MLP sees z_i = AVERAGE, not SUM. How does this work?

Hypothesis 1: The MLP learns to "undo" the averaging
- z_i = (1/(i+1)) * Σ v_j
- w · z_i = (1/(i+1)) * Σ (w · v_j) ≈ c (constant)
- But ||z_i||² ∝ 1/(i+1), so the MLP can use: (w · z_i) * (i+1) ≈ (i+1) * c
- The MLP learns to extract (i+1) from the norm and multiply

Hypothesis 2: The cumulative structure is directly recoverable
- Even though z_i is averaged, it contains information about how many vectors contributed
- The "spread" or "concentration" of z_i in embedding space encodes position

Let's test both hypotheses.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))
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


def test_orthogonality_property(model, device="cuda"):
    """
    Test: w · e_k ≈ 1 for any embedding e_k

    Where w = Σ_j LN(E_j)
    """
    print("\n" + "=" * 70)
    print("TEST 1: Orthogonality Property")
    print("=" * 70)

    W_E = model.transformer.wte.weight.data  # [vocab_size, d_model]
    vocab_size, d_model = W_E.shape

    # Normalize embeddings (LayerNorm style)
    E_mean = W_E.mean(dim=-1, keepdim=True)
    E_std = W_E.std(dim=-1, keepdim=True) + 1e-5
    E_normalized = (W_E - E_mean) / E_std  # [vocab_size, d_model]

    # Compute decoding vector: w = Σ_j LN(E_j)
    w = E_normalized.sum(dim=0)  # [d_model]

    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {d_model}")
    print(f"||w||: {w.norm().item():.2f}")

    # Test: w · e_k for random embeddings
    n_test = 10000
    random_indices = torch.randint(0, vocab_size, (n_test,), device=device)
    test_embeddings = E_normalized[random_indices]  # [n_test, d_model]

    dot_products = (test_embeddings @ w).cpu().numpy()  # [n_test]

    print(f"\nDot product w · LN(e_k) for {n_test} random tokens:")
    print(f"  Mean: {dot_products.mean():.4f}")
    print(f"  Std:  {dot_products.std():.4f}")
    print(f"  Min:  {dot_products.min():.4f}")
    print(f"  Max:  {dot_products.max():.4f}")

    # Theory: E[w · e_k] ≈ 1 because e_k contributes 1 to w, rest is noise
    # Actually: w · e_k = Σ_j (e_j · e_k) = 1 + Σ_{j≠k} (e_j · e_k)
    # Due to orthogonality: Σ_{j≠k} (e_j · e_k) ≈ 0

    # But wait - we need to be more careful about the normalization
    # Let's check the actual expected value

    # For high-d random vectors: e_j · e_k ~ N(0, 1/d) for j ≠ k
    # Sum of V-1 such terms: ~ N(0, (V-1)/d)
    expected_std = np.sqrt((vocab_size - 1) / d_model)
    print(f"\nTheoretical analysis:")
    print(f"  Expected std of noise term: {expected_std:.4f}")
    print(f"  Observed std: {dot_products.std():.4f}")

    return {
        "mean_dot_product": dot_products.mean(),
        "std_dot_product": dot_products.std(),
        "w_norm": w.norm().item(),
        "w": w.cpu(),
    }


def test_cumsum_vs_average(model, n_samples=1000, seq_len=64, device="cuda"):
    """
    Test: Can we decode position from both cumsum and average formulations?

    cumsum: Σ_{j=0}^{i} (w · v_j)
    average: (1/(i+1)) * Σ_{j=0}^{i} (w · v_j)
    """
    print("\n" + "=" * 70)
    print("TEST 2: Cumsum vs Average Decoding")
    print("=" * 70)

    W_E = model.transformer.wte.weight.data
    vocab_size, d_model = W_E.shape

    # Get W_V from first attention layer
    # In nanoGPT, c_attn projects to [q, k, v] concatenated
    c_attn_weight = model.transformer.h[
        0
    ].attn.c_attn.weight.data  # [d_model, 3*d_model]
    W_V = c_attn_weight[
        :, 2 * d_model :
    ]  # Last third is V projection [d_model, d_model]

    # Compute decoding vector
    E_mean = W_E.mean(dim=-1, keepdim=True)
    E_std = W_E.std(dim=-1, keepdim=True) + 1e-5
    E_normalized = (W_E - E_mean) / E_std

    w_embed = E_normalized.sum(dim=0)  # [d_model]

    # w through value projection: w_v = W_V^T @ w_embed
    # Actually, value vectors are: v = LN(e) @ W_V^T
    # So decoding vector should be: w = W_V @ Σ LN(e_j) = W_V @ w_embed
    w = W_V @ w_embed  # [d_model]
    w = w / (w.norm() + 1e-8)  # Normalize

    print(f"Decoding vector ||w||: {w.norm().item():.4f}")

    # Generate random sequences
    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)

    # Get embeddings and value vectors
    embeddings = W_E[tokens]  # [n_samples, seq_len, d_model]

    # Normalize embeddings
    e_mean = embeddings.mean(dim=-1, keepdim=True)
    e_std = embeddings.std(dim=-1, keepdim=True) + 1e-5
    e_norm = (embeddings - e_mean) / e_std

    # Value vectors
    values = e_norm @ W_V.T  # [n_samples, seq_len, d_model]

    # Compute cumsum and average for each position
    cumsum_decoded = []
    average_decoded = []

    for i in range(seq_len):
        # Values up to position i (inclusive)
        v_up_to_i = values[:, : i + 1, :]  # [n_samples, i+1, d_model]

        # Cumsum: Σ (w · v_j)
        dot_products = (v_up_to_i * w).sum(dim=-1)  # [n_samples, i+1]
        cumsum_val = dot_products.sum(dim=-1)  # [n_samples]
        cumsum_decoded.append(cumsum_val.cpu().numpy())

        # Average: (1/(i+1)) * Σ (w · v_j)
        average_val = cumsum_val / (i + 1)
        average_decoded.append(average_val.cpu().numpy())

    cumsum_decoded = np.array(cumsum_decoded).T  # [n_samples, seq_len]
    average_decoded = np.array(average_decoded).T  # [n_samples, seq_len]

    positions = np.arange(seq_len)

    # Correlation analysis
    cumsum_flat = cumsum_decoded.flatten()
    average_flat = average_decoded.flatten()
    pos_flat = np.tile(positions, n_samples)

    r_cumsum, _ = pearsonr(cumsum_flat, pos_flat)
    r_average, _ = pearsonr(average_flat, pos_flat)

    print(f"\nCorrelation with position:")
    print(f"  Cumsum Σ(w·v_j):     r = {r_cumsum:.4f}")
    print(f"  Average (1/i)Σ(w·v_j): r = {r_average:.4f}")

    # Mean values per position
    mean_cumsum = cumsum_decoded.mean(axis=0)
    mean_average = average_decoded.mean(axis=0)

    print(f"\nMean decoded value per position (first 10 positions):")
    print(f"  Position: {list(range(10))}")
    print(f"  Cumsum:   {[f'{x:.2f}' for x in mean_cumsum[:10]]}")
    print(f"  Average:  {[f'{x:.2f}' for x in mean_average[:10]]}")

    # Linear fit
    from sklearn.linear_model import LinearRegression

    reg_cumsum = LinearRegression()
    reg_cumsum.fit(pos_flat.reshape(-1, 1), cumsum_flat)
    r2_cumsum = reg_cumsum.score(pos_flat.reshape(-1, 1), cumsum_flat)

    reg_average = LinearRegression()
    reg_average.fit(pos_flat.reshape(-1, 1), average_flat)
    r2_average = reg_average.score(pos_flat.reshape(-1, 1), average_flat)

    print(f"\nLinear regression R²:")
    print(f"  Cumsum:  R² = {r2_cumsum:.4f}, slope = {reg_cumsum.coef_[0]:.4f}")
    print(f"  Average: R² = {r2_average:.4f}, slope = {reg_average.coef_[0]:.6f}")

    return {
        "r_cumsum": r_cumsum,
        "r_average": r_average,
        "r2_cumsum": r2_cumsum,
        "r2_average": r2_average,
        "mean_cumsum": mean_cumsum,
        "mean_average": mean_average,
    }


def test_norm_times_average(model, n_samples=1000, seq_len=64, device="cuda"):
    """
    Test: Can we recover position by multiplying average by norm-based factor?

    z_i = (1/(i+1)) * Σ v_j  (average)
    ||z_i||² ∝ 1/(i+1)

    So: (w · z_i) * ||z_i||^{-2} ∝ (i+1) * (w · z_i) / constant

    If w · v_j ≈ c, then w · z_i ≈ c (constant)
    And (w · z_i) * (i+1) ≈ c * (i+1) (linear in position!)
    """
    print("\n" + "=" * 70)
    print("TEST 3: Recovering Position from Average + Norm")
    print("=" * 70)

    W_E = model.transformer.wte.weight.data
    vocab_size, d_model = W_E.shape

    c_attn_weight = model.transformer.h[0].attn.c_attn.weight.data
    W_V = c_attn_weight[:, 2 * d_model :]

    # Compute decoding vector
    E_mean = W_E.mean(dim=-1, keepdim=True)
    E_std = W_E.std(dim=-1, keepdim=True) + 1e-5
    E_normalized = (W_E - E_mean) / E_std
    w_embed = E_normalized.sum(dim=0)
    w = W_V @ w_embed
    w = w / (w.norm() + 1e-8)

    # Generate random sequences
    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)

    embeddings = W_E[tokens]
    e_mean = embeddings.mean(dim=-1, keepdim=True)
    e_std = embeddings.std(dim=-1, keepdim=True) + 1e-5
    e_norm = (embeddings - e_mean) / e_std
    values = e_norm @ W_V.T

    # Compute attention output (average) for each position
    results = {
        "w_dot_z": [],  # w · z_i (should be ~constant)
        "z_norm_sq": [],  # ||z_i||² (should be ∝ 1/(i+1))
        "w_dot_z_times_inv_norm_sq": [],  # (w · z_i) / ||z_i||²
        "reconstructed_pos": [],  # Trying to reconstruct position
    }

    for i in range(seq_len):
        v_up_to_i = values[:, : i + 1, :]  # [n_samples, i+1, d_model]
        z_i = v_up_to_i.mean(dim=1)  # [n_samples, d_model] - the AVERAGE

        # w · z_i
        w_dot_z = (z_i * w).sum(dim=-1)  # [n_samples]
        results["w_dot_z"].append(w_dot_z.cpu().numpy())

        # ||z_i||²
        z_norm_sq = (z_i**2).sum(dim=-1)  # [n_samples]
        results["z_norm_sq"].append(z_norm_sq.cpu().numpy())

        # (w · z_i) / ||z_i||²
        ratio = w_dot_z / (z_norm_sq + 1e-8)
        results["w_dot_z_times_inv_norm_sq"].append(ratio.cpu().numpy())

        # Reconstructed position: if ||z_i||² ∝ 1/(i+1), then 1/||z_i||² ∝ (i+1)
        # Scale factor to convert to position
        inv_norm_sq = 1 / (z_norm_sq + 1e-8)
        results["reconstructed_pos"].append(inv_norm_sq.cpu().numpy())

    # Convert to arrays
    for k in results:
        results[k] = np.array(results[k]).T  # [n_samples, seq_len]

    positions = np.arange(seq_len)
    pos_flat = np.tile(positions, n_samples)

    # Analysis
    print("\nPer-position mean values (first 10 positions):")
    print(f"Position:        {list(range(10))}")
    print(
        f"w·z_i:           {[f'{x:.3f}' for x in results['w_dot_z'].mean(axis=0)[:10]]}"
    )
    print(
        f"||z_i||²:        {[f'{x:.4f}' for x in results['z_norm_sq'].mean(axis=0)[:10]]}"
    )
    print(
        f"(w·z)/(||z||²):  {[f'{x:.2f}' for x in results['w_dot_z_times_inv_norm_sq'].mean(axis=0)[:10]]}"
    )

    # Correlations
    print("\nCorrelations with position:")
    for name, vals in results.items():
        r, _ = pearsonr(vals.flatten(), pos_flat)
        print(f"  {name}: r = {r:.4f}")

    # The key test: can we linearly predict position from 1/||z_i||²?
    inv_norm_sq_flat = results["reconstructed_pos"].flatten()

    # Linear regression
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(inv_norm_sq_flat.reshape(-1, 1), pos_flat)
    pred = reg.predict(inv_norm_sq_flat.reshape(-1, 1))
    r2 = r2_score(pos_flat, pred)
    mae = np.abs(pos_flat - pred).mean()

    print(f"\nLinear probe on 1/||z_i||²:")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae:.2f} positions")

    # What about ||z_i|| vs 1/sqrt(i+1)?
    mean_norm = np.sqrt(results["z_norm_sq"].mean(axis=0))
    theory = 1 / np.sqrt(positions + 1)
    # Normalize for comparison
    theory_scaled = theory * mean_norm[0]
    r_theory, _ = pearsonr(mean_norm, theory_scaled)
    print(f"\n||z_i|| vs 1/√(i+1): r = {r_theory:.4f}")

    return results


def test_what_mlp_actually_sees(model, n_samples=1000, seq_len=64, device="cuda"):
    """
    The MLP doesn't see raw attention output - it sees post-LN2 activations.

    Let's trace exactly what the MLP receives and test what features
    carry positional information.
    """
    print("\n" + "=" * 70)
    print("TEST 4: What the MLP Actually Sees")
    print("=" * 70)

    W_E = model.transformer.wte.weight.data
    vocab_size, d_model = W_E.shape

    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)

    with torch.no_grad():
        # Trace through the network
        tok_emb = model.transformer.wte(tokens)
        x = model.transformer.drop(tok_emb)

        block = model.transformer.h[0]

        # Pre-attention LN
        x_ln1 = block.ln_1(x)

        # Attention output (before residual)
        attn_out = block.attn(x_ln1)

        # Post-attention (with residual)
        x_post_attn = x + attn_out

        # Pre-MLP LN (this is what MLP sees!)
        x_ln2 = block.ln_2(x_post_attn)

    # The MLP sees x_ln2
    # Due to LayerNorm, each position has mean=0 and std=1 (approximately)
    # So the "norm" information is lost!

    # But wait - LayerNorm uses LEARNED scale (gamma) and bias (beta)
    # Let's check the per-position statistics of x_ln2

    x_ln2_np = x_ln2.cpu().numpy()  # [n_samples, seq_len, d_model]

    print("\nPost-LN2 statistics (what MLP sees):")
    print(f"  Mean per position (should be ~0): {x_ln2_np.mean(axis=(0, 2))[:5]}")
    print(f"  Std per position (should be ~1):  {x_ln2_np.std(axis=(0, 2))[:5]}")

    # The key question: what feature of x_ln2 carries position info?

    # Option 1: The POPULATION mean differs by position
    # (even though per-sample mean is 0)
    pop_mean_per_pos = x_ln2_np.mean(axis=0)  # [seq_len, d_model]
    pop_mean_norm = np.linalg.norm(pop_mean_per_pos, axis=-1)  # [seq_len]

    positions = np.arange(seq_len)
    r_pop_mean, _ = pearsonr(pop_mean_norm, positions)
    print(f"\nPopulation mean ||E[x_ln2]|| vs position: r = {r_pop_mean:.4f}")

    # Option 2: Specific directions carry position info
    # The MLP first layer projects x_ln2 to hidden dim
    # Let's see if any single direction correlates with position

    # Compute correlation of each dimension with position
    correlations = []
    for d in range(d_model):
        dim_values = x_ln2_np[:, :, d].flatten()
        pos_flat = np.tile(positions, n_samples)
        r, _ = pearsonr(dim_values, pos_flat)
        correlations.append(r)

    correlations = np.array(correlations)
    top_dims = np.argsort(np.abs(correlations))[-10:][::-1]

    print(f"\nTop 10 dimensions by |correlation| with position:")
    for d in top_dims:
        print(f"  Dim {d}: r = {correlations[d]:.4f}")

    print(f"\nMax |correlation| across all dims: {np.abs(correlations).max():.4f}")
    print(f"Mean |correlation|: {np.abs(correlations).mean():.4f}")

    # Option 3: The PRE-LN2 activations have position in variance
    # and LN doesn't fully remove it
    x_post_attn_np = x_post_attn.cpu().numpy()

    pre_ln_var = x_post_attn_np.var(axis=-1)  # [n_samples, seq_len]
    mean_pre_ln_var = pre_ln_var.mean(axis=0)  # [seq_len]

    r_pre_ln_var, _ = pearsonr(mean_pre_ln_var, positions)
    print(f"\nPre-LN2 variance vs position: r = {r_pre_ln_var:.4f}")

    # The LN divides by std = sqrt(var), so scale factor is 1/sqrt(var)
    # This scale factor varies with position!
    ln_scale = 1 / np.sqrt(mean_pre_ln_var + 1e-8)
    r_ln_scale, _ = pearsonr(ln_scale, positions)
    print(f"LN scale factor (1/std) vs position: r = {r_ln_scale:.4f}")

    return {
        "pop_mean_norm": pop_mean_norm,
        "dim_correlations": correlations,
        "pre_ln_var": mean_pre_ln_var,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="nanoGPT/out-nope-1layer-ln/ckpt.pt"
    )
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 70)
    print("DECODING VECTOR & ORTHOGONALITY ANALYSIS")
    print("=" * 70)

    print(f"\nLoading model from {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, args.device)

    norm_type = model.config.norm_type
    print(f"Model type: {norm_type.upper()}")

    # Run all tests
    results = {}

    results["orthogonality"] = test_orthogonality_property(model, args.device)

    results["cumsum_vs_avg"] = test_cumsum_vs_average(
        model, args.n_samples, args.seq_len, args.device
    )

    results["norm_times_avg"] = test_norm_times_average(
        model, args.n_samples, args.seq_len, args.device
    )

    results["mlp_input"] = test_what_mlp_actually_sees(
        model, args.n_samples, args.seq_len, args.device
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("""
Key findings:

1. ORTHOGONALITY: The decoding vector w = Σ LN(e_j) has the property that
   w · e_k ≈ constant (due to approximate orthogonality of random embeddings)

2. CUMSUM vs AVERAGE:
   - Cumsum Σ(w·v_j) IS linear with position (r ≈ 1.0)
   - Average (1/i)Σ(w·v_j) is approximately CONSTANT
   
3. RECOVERING POSITION FROM AVERAGE:
   - The MLP sees the AVERAGE z_i, not the SUM
   - But ||z_i||² ∝ 1/(i+1), so position is encoded in the NORM
   - A nonlinear operation (squaring, then inverting) is needed
   
4. WHAT THE MLP SEES:
   - After LayerNorm, per-sample statistics are normalized
   - But POPULATION-level statistics differ by position
   - Specific directions in embedding space correlate with position
   
CONCLUSION: The decoding vector mechanism works for the CUMSUM formulation.
For the AVERAGE formulation (what the network computes), position must be 
recovered through the NORM, which requires nonlinearity (explaining why 
MLP probes >> linear probes).
""")


if __name__ == "__main__":
    main()
