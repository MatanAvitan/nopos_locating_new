"""
Long Context Position Encoding Analysis - LayerNorm vs RMSNorm

Key questions to answer:
1. How does post-normalization norm variation scale with context length?
2. Is the 1/√(i+1) theoretical prediction robust at 8K context?
3. Does the decoding vector hyperplane work at long contexts?
4. Does RMSNorm (no mean centering) preserve positional info better than LayerNorm?
5. What is the actual mechanism - norm, direction, or a specific hyperplane?

This script tests context lengths from 64 to 8192 with both norm types.
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
from scipy import stats

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "long_context_analysis"


def create_model(n_ctx, n_embd=256, n_layer=1, n_head=4, norm_type="layernorm"):
    """Create randomly initialized NoPE model with specified context length and norm type."""
    config = GPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=n_ctx,
        vocab_size=1000,
        dropout=0.0,
        use_positional_embedding=False,
        norm_type=norm_type,
    )
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model, config


def get_activations(model, tokens):
    """Get activations at key layers."""
    activations = {}

    tok_emb = model.transformer.wte(tokens)
    activations["embed"] = tok_emb.detach()

    block = model.transformer.h[0]

    # Post LN1/RMSNorm1 (input to attention)
    x = block.ln_1(tok_emb)
    activations["post_norm1"] = x.detach()

    # Post attention (before residual)
    attn_out = block.attn(x)[0]
    activations["post_attn"] = attn_out.detach()

    # Post attention residual
    x = tok_emb + attn_out
    activations["post_attn_residual"] = x.detach()

    # Post LN2/RMSNorm2
    x_ln2 = block.ln_2(x)
    activations["post_norm2"] = x_ln2.detach()

    return activations


def compute_decoding_vector(model, norm_type="layernorm"):
    """
    Compute the theoretical decoding vector: w = W_V · Σ Norm(E_j)
    """
    E = model.transformer.wte.weight.detach()  # (vocab_size, d_model)
    norm1 = model.transformer.h[0].ln_1

    if norm_type == "layernorm":
        # LN(E) = (E - mean) / std * gamma + beta
        E_centered = E - E.mean(dim=-1, keepdim=True)
        E_std = E.std(dim=-1, keepdim=True)
        E_norm = E_centered / (E_std + 1e-5) * norm1.weight + norm1.bias
    else:
        # RMSNorm(E) = E / rms * gamma (no mean centering, no bias)
        rms = torch.sqrt(torch.mean(E**2, dim=-1, keepdim=True))
        E_norm = E / (rms + 1e-5) * norm1.weight

    sum_norm_E = E_norm.sum(dim=0)  # (d_model,)

    attn = model.transformer.h[0].attn
    n_embd = model.config.n_embd
    W_V = attn.c_attn.weight[:, 2 * n_embd :].detach()

    w = W_V @ sum_norm_E
    w = w / (torch.norm(w) + 1e-8)

    return w


def analyze_context_length(n_ctx, norm_type="layernorm", n_samples=200, n_embd=256):
    """
    Comprehensive analysis for a specific context length and norm type.
    """
    print(f"\n{'=' * 60}")
    print(f"CONTEXT LENGTH: {n_ctx}, NORM TYPE: {norm_type.upper()}")
    print(f"{'=' * 60}")

    model, config = create_model(n_ctx, n_embd=n_embd, norm_type=norm_type)
    decoding_vector = compute_decoding_vector(model, norm_type)

    results = {
        "n_ctx": n_ctx,
        "norm_type": norm_type,
        "n_embd": n_embd,
        "n_samples": n_samples,
    }

    # Collect activations
    all_post_attn = []
    all_post_norm2 = []
    all_positions = []

    for _ in range(n_samples):
        tokens = torch.randint(0, 1000, (1, n_ctx), device=DEVICE)
        acts = get_activations(model, tokens)

        all_post_attn.append(acts["post_attn"][0].cpu().numpy())
        all_post_norm2.append(acts["post_norm2"][0].cpu().numpy())
        all_positions.append(np.arange(n_ctx))

    post_attn = np.vstack(all_post_attn)
    post_norm2 = np.vstack(all_post_norm2)
    positions = np.concatenate(all_positions)

    # 1. NORM ANALYSIS
    print("\n1. NORM ANALYSIS")

    post_attn_norms = np.linalg.norm(post_attn, axis=1)
    post_norm2_norms = np.linalg.norm(post_norm2, axis=1)

    corr_attn_norm = np.corrcoef(post_attn_norms, positions)[0, 1]
    corr_norm2_norm = np.corrcoef(post_norm2_norms, positions)[0, 1]

    results["post_attn_norm_corr"] = corr_attn_norm
    results["post_norm2_norm_corr"] = corr_norm2_norm

    print(f"  Post-attention norm-position correlation: {corr_attn_norm:.4f}")
    print(f"  Post-{norm_type} norm-position correlation: {corr_norm2_norm:.4f}")

    # Mean norm by position
    mean_norm_by_pos_attn = np.array(
        [post_attn_norms[positions == i].mean() for i in range(n_ctx)]
    )
    mean_norm_by_pos_norm2 = np.array(
        [post_norm2_norms[positions == i].mean() for i in range(n_ctx)]
    )

    # Theoretical prediction: 1/sqrt(i+1) for post_attn
    theoretical = 1 / np.sqrt(np.arange(1, n_ctx + 1))
    theoretical_scaled = theoretical * mean_norm_by_pos_attn[0]

    corr_with_theory = np.corrcoef(mean_norm_by_pos_attn, theoretical_scaled)[0, 1]
    results["norm_theory_correlation"] = corr_with_theory
    print(f"  Correlation with 1/√(i+1) theory: {corr_with_theory:.4f}")

    # Norm range
    norm_range_attn = mean_norm_by_pos_attn.max() - mean_norm_by_pos_attn.min()
    norm_range_norm2 = mean_norm_by_pos_norm2.max() - mean_norm_by_pos_norm2.min()

    results["post_attn_norm_range"] = norm_range_attn
    results["post_norm2_norm_range"] = norm_range_norm2
    results["post_attn_norm_mean"] = mean_norm_by_pos_attn.mean()
    results["post_norm2_norm_mean"] = mean_norm_by_pos_norm2.mean()

    print(
        f"  Post-attention norm range: {norm_range_attn:.4f} (mean={mean_norm_by_pos_attn.mean():.2f})"
    )
    print(
        f"  Post-{norm_type} norm range: {norm_range_norm2:.4f} (mean={mean_norm_by_pos_norm2.mean():.2f})"
    )
    print(
        f"  Post-{norm_type} relative variation: {norm_range_norm2 / mean_norm_by_pos_norm2.mean() * 100:.4f}%"
    )

    # 2. DECODING VECTOR ANALYSIS
    print("\n2. DECODING VECTOR HYPERPLANE")

    w = decoding_vector.cpu().numpy()

    proj_attn = post_attn @ w
    proj_norm2 = post_norm2 @ w

    corr_proj_attn = np.corrcoef(proj_attn, positions)[0, 1]
    corr_proj_norm2 = np.corrcoef(proj_norm2, positions)[0, 1]

    results["decoding_vector_corr_post_attn"] = corr_proj_attn
    results["decoding_vector_corr_post_norm2"] = corr_proj_norm2

    print(f"  Decoding vector correlation (post-attn): {corr_proj_attn:.4f}")
    print(f"  Decoding vector correlation (post-{norm_type}): {corr_proj_norm2:.4f}")

    # 3. DIRECTION VS NORM PROBE
    print("\n3. DIRECTION VS NORM (Linear Probe R²)")

    n_train = int(0.8 * len(positions))
    idx = np.random.permutation(len(positions))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    post_attn_dir = post_attn / (
        np.linalg.norm(post_attn, axis=1, keepdims=True) + 1e-8
    )
    post_norm2_dir = post_norm2 / (
        np.linalg.norm(post_norm2, axis=1, keepdims=True) + 1e-8
    )

    def fit_probe(X_train, y_train, X_test, y_test):
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return max(0, 1 - ss_res / ss_tot)

    # Post-attention
    r2_attn_full = fit_probe(
        post_attn[train_idx],
        positions[train_idx],
        post_attn[test_idx],
        positions[test_idx],
    )
    r2_attn_dir = fit_probe(
        post_attn_dir[train_idx],
        positions[train_idx],
        post_attn_dir[test_idx],
        positions[test_idx],
    )
    r2_attn_norm = fit_probe(
        post_attn_norms[train_idx].reshape(-1, 1),
        positions[train_idx],
        post_attn_norms[test_idx].reshape(-1, 1),
        positions[test_idx],
    )

    # Post-norm2
    r2_norm2_full = fit_probe(
        post_norm2[train_idx],
        positions[train_idx],
        post_norm2[test_idx],
        positions[test_idx],
    )
    r2_norm2_dir = fit_probe(
        post_norm2_dir[train_idx],
        positions[train_idx],
        post_norm2_dir[test_idx],
        positions[test_idx],
    )
    r2_norm2_norm = fit_probe(
        post_norm2_norms[train_idx].reshape(-1, 1),
        positions[train_idx],
        post_norm2_norms[test_idx].reshape(-1, 1),
        positions[test_idx],
    )

    results["post_attn_full_r2"] = r2_attn_full
    results["post_attn_direction_r2"] = r2_attn_dir
    results["post_attn_norm_r2"] = r2_attn_norm
    results["post_norm2_full_r2"] = r2_norm2_full
    results["post_norm2_direction_r2"] = r2_norm2_dir
    results["post_norm2_norm_r2"] = r2_norm2_norm

    print(
        f"  Post-attention: full={r2_attn_full:.4f}, direction={r2_attn_dir:.4f}, norm={r2_attn_norm:.4f}"
    )
    print(
        f"  Post-{norm_type}: full={r2_norm2_full:.4f}, direction={r2_norm2_dir:.4f}, norm={r2_norm2_norm:.4f}"
    )

    # 4. POSITION DISCRIMINATION BY RANGE
    print("\n4. POSITION DISCRIMINATION BY RANGE")

    def discrimination_score(start, end):
        mask = (positions >= start) & (positions < end)
        if mask.sum() < 100:
            return np.nan

        local_pos = positions[mask]
        local_act = post_norm2[mask]

        n_train_local = int(0.8 * len(local_pos))
        idx_local = np.random.permutation(len(local_pos))

        probe = Ridge(alpha=1.0)
        probe.fit(
            local_act[idx_local[:n_train_local]], local_pos[idx_local[:n_train_local]]
        )
        y_pred = probe.predict(local_act[idx_local[n_train_local:]])
        y_true = local_pos[idx_local[n_train_local:]]

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return max(0, 1 - ss_res / ss_tot)

    ranges = [
        (0, n_ctx // 4),
        (n_ctx // 4, n_ctx // 2),
        (n_ctx // 2, 3 * n_ctx // 4),
        (3 * n_ctx // 4, n_ctx),
    ]
    range_scores = {}

    for start, end in ranges:
        score = discrimination_score(start, end)
        range_name = f"{start}-{end}"
        range_scores[range_name] = score
        print(f"  Range {range_name}: R² = {score:.4f}")

    results["range_discrimination"] = range_scores

    # 5. Late position discrimination
    print("\n5. LATE POSITION DISCRIMINATION")

    if n_ctx >= 200:
        late_start = n_ctx - 100
        late_score = discrimination_score(late_start, n_ctx)
        results["late_positions_r2"] = late_score
        print(f"  Positions {late_start}-{n_ctx}: R² = {late_score:.4f}")

        late_norm_diff = mean_norm_by_pos_norm2[late_start] - mean_norm_by_pos_norm2[-1]
        results["late_norm_difference"] = late_norm_diff
        print(
            f"  Norm difference (pos {late_start} vs {n_ctx - 1}): {late_norm_diff:.6f}"
        )

    # Store data for plotting
    results["mean_norm_by_pos_attn"] = mean_norm_by_pos_attn.tolist()
    results["mean_norm_by_pos_norm2"] = mean_norm_by_pos_norm2.tolist()

    return results


def main():
    print("=" * 70)
    print("LONG CONTEXT POSITION ENCODING: LAYERNORM vs RMSNORM")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Test multiple context lengths with both norm types
    context_lengths = [64, 256, 512, 1024, 2048, 4096, 8192]
    norm_types = ["layernorm", "rmsnorm"]

    all_results = {}

    for norm_type in norm_types:
        all_results[norm_type] = {}

        for n_ctx in context_lengths:
            # Reduce samples for longer contexts
            n_samples = max(50, 500 // (n_ctx // 64))

            try:
                # Reset seed for fair comparison
                torch.manual_seed(42)
                np.random.seed(42)

                results = analyze_context_length(
                    n_ctx, norm_type=norm_type, n_samples=n_samples
                )
                all_results[norm_type][n_ctx] = results
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"  OOM at context length {n_ctx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                raise

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY: LAYERNORM vs RMSNORM SCALING")
    print("=" * 70)

    for norm_type in norm_types:
        print(f"\n### {norm_type.upper()} ###")
        print(
            "\n| Context | Post-Attn Norm R² | Post-Norm Norm R² | Theory Corr | DV Corr |"
        )
        print(
            "|---------|-------------------|-------------------|-------------|---------|"
        )

        for n_ctx, res in all_results[norm_type].items():
            print(
                f"| {n_ctx:7d} | {res['post_attn_norm_r2']:17.4f} | {res['post_norm2_norm_r2']:17.4f} | "
                f"{res['norm_theory_correlation']:11.4f} | {res['decoding_vector_corr_post_norm2']:7.4f} |"
            )

    # Direction vs Norm comparison
    print("\n" + "=" * 70)
    print("DIRECTION vs NORM ACROSS NORM TYPES")
    print("=" * 70)

    for norm_type in norm_types:
        print(f"\n### {norm_type.upper()} ###")
        print(
            "\n| Context | Dir R² (attn) | Norm R² (attn) | Dir R² (norm) | Norm R² (norm) |"
        )
        print(
            "|---------|---------------|----------------|---------------|----------------|"
        )

        for n_ctx, res in all_results[norm_type].items():
            print(
                f"| {n_ctx:7d} | {res['post_attn_direction_r2']:13.4f} | {res['post_attn_norm_r2']:14.4f} | "
                f"{res['post_norm2_direction_r2']:13.4f} | {res['post_norm2_norm_r2']:14.4f} |"
            )

    # KEY COMPARISON: LN vs RMS at 8K
    print("\n" + "=" * 70)
    print("KEY COMPARISON: LAYERNORM vs RMSNORM")
    print("=" * 70)

    if 8192 in all_results.get("layernorm", {}) and 8192 in all_results.get(
        "rmsnorm", {}
    ):
        ln = all_results["layernorm"][8192]
        rms = all_results["rmsnorm"][8192]

        print("\nAt 8K context:")
        print(f"                      LayerNorm    RMSNorm")
        print(
            f"  Post-norm2 full R²: {ln['post_norm2_full_r2']:10.4f}   {rms['post_norm2_full_r2']:.4f}"
        )
        print(
            f"  Direction R²:       {ln['post_norm2_direction_r2']:10.4f}   {rms['post_norm2_direction_r2']:.4f}"
        )
        print(
            f"  Norm R²:            {ln['post_norm2_norm_r2']:10.4f}   {rms['post_norm2_norm_r2']:.4f}"
        )
        print(
            f"  Decoding vector r:  {ln['decoding_vector_corr_post_norm2']:10.4f}   {rms['decoding_vector_corr_post_norm2']:.4f}"
        )
        print(
            f"  Norm variation %:   {ln['post_norm2_norm_range'] / ln['post_norm2_norm_mean'] * 100:10.4f}   {rms['post_norm2_norm_range'] / rms['post_norm2_norm_mean'] * 100:.4f}"
        )

        if ln["post_norm2_full_r2"] < rms["post_norm2_full_r2"]:
            print("\n  -> RMSNorm preserves position info BETTER than LayerNorm!")
        else:
            print("\n  -> LayerNorm preserves position info better or equal to RMSNorm")

    # Save results
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(RESULTS_DIR / "long_context_ln_vs_rms_results.json", "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
