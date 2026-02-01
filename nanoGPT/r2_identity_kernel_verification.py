"""
R2 Identity-Kernel Verification: Empirical Tests for the Paper Claims

This script verifies the "identity-kernel + subspace projection" theory for the R2 regime:
  - R2: Only Attn2 (and head) trainable, everything else frozen

The theory claims:
  1. Block 2 attention learns near-identity kernel: alpha_{i,j} ≈ delta_{ij}
  2. This makes attention a positionwise linear map: Attn2(x)_i ≈ A @ x_i
  3. The matrix A = W_O @ W_V acts as a subspace projection
  4. Rows of A aligned with w_dec (position-decoding direction) write position into coordinates
  5. These coordinates become near-perfect position predictors after Attn2

Tests:
  1. Near-Identity Kernel Verification - measure diagonal mass per head
  2. Compute Effective Linear Map A = W_O @ W_V
  3. Fit w_dec and Measure Row Alignment
  4. Verify Coordinate Concentration
  5. Decompose Local vs Non-Local Contributions

Usage:
    CUDA_VISIBLE_DEVICES=0 python r2_identity_kernel_verification.py [--wandb]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# Optional wandb
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ICML style settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "text.usetex": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_model(regime, device="cuda"):
    """Load a trained model for a specific regime."""
    checkpoint_path = f"out-2layer-mechanism/{regime}/best_ckpt.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]

    model_config_fields = [
        "block_size",
        "vocab_size",
        "n_embd",
        "n_head",
        "dropout",
        "norm_type",
    ]
    filtered_config = {
        k: config_dict[k] for k in model_config_fields if k in config_dict
    }
    filtered_config["bias"] = True
    filtered_config["use_regression"] = True

    config = TwoLayerMechanismConfig(**filtered_config)
    model = TwoLayerMechanismModel(config)

    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, config


def get_data(n_samples, block_size, device="cuda"):
    """Load OpenWebText data."""
    data_path = "data/openwebtext/train.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (n_samples,))
    tokens = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    return tokens.to(device)


def compute_forward_pass(model, tokens, device="cuda"):
    """
    Compute forward pass and return all intermediate activations.

    Returns dict with:
        - e: embeddings [B, T, D]
        - ln1_r1: LN(r1) input to Block 2 attention [B, T, D]
        - r1: Block 1 output (before LN) [B, T, D]
        - attn_weights2: Block 2 attention weights [B, H, T, T]
        - attn_out2: Block 2 attention output (after W_O) [B, T, D]
        - r2_attn: r1 + attn_out2 [B, T, D]
    """
    config = model.config
    B, T = tokens.shape
    D = config.n_embd
    n_head = config.n_head
    head_dim = D // n_head

    with torch.no_grad():
        # Embeddings
        e = model.wte(tokens)

        # Block 1 (frozen)
        ln1_out = model.block1.ln_1(e)

        attn1 = model.block1.attn
        qkv1 = attn1.c_attn(ln1_out)
        q1, k1, v1 = qkv1.split(D, dim=2)
        q1 = q1.view(B, T, n_head, head_dim).transpose(1, 2)
        k1 = k1.view(B, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B, T, n_head, head_dim).transpose(1, 2)

        scores1 = (q1 @ k1.transpose(-2, -1)) / np.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores1 = scores1.masked_fill(causal_mask, float("-inf"))
        attn_weights1 = F.softmax(scores1, dim=-1)

        attn_out1 = (attn_weights1 @ v1).transpose(1, 2).reshape(B, T, D)
        attn_out1 = attn1.c_proj(attn_out1)
        r1_attn = e + attn_out1

        ln2_out_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_out_b1)
        r1 = r1_attn + mlp_out1

        # Block 2 (attention trained)
        ln1_r1 = model.block2.ln_1(r1)  # Input to Attn2

        attn2 = model.block2.attn
        qkv2 = attn2.c_attn(ln1_r1)
        q2, k2, v2 = qkv2.split(D, dim=2)
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        scores2 = (q2 @ k2.transpose(-2, -1)) / np.sqrt(head_dim)
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))
        attn_weights2 = F.softmax(scores2, dim=-1)

        # Attention output
        attn_out2_preproj = (attn_weights2 @ v2).transpose(1, 2).reshape(B, T, D)
        attn_out2 = attn2.c_proj(attn_out2_preproj)

        r2_attn = r1 + attn_out2

        # Also compute final output
        ln2_out_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_out_b2)
        r2 = r2_attn + mlp_out2
        final_ln = model.ln_f(r2)

    return {
        "e": e,
        "r1": r1,
        "ln1_r1": ln1_r1,
        "attn_weights2": attn_weights2,
        "attn_out2": attn_out2,
        "r2_attn": r2_attn,
        "r2": r2,
        "final_ln": final_ln,
        "v2": v2,  # [B, H, T, head_dim]
        "attn_out2_preproj": attn_out2_preproj,
    }


# =============================================================================
# TEST 1: Near-Identity Kernel Verification
# =============================================================================


def test_near_identity_kernel(attn_weights, threshold=0.5):
    """
    Test 1: Verify that Block 2 attention learns near-identity kernel.

    For each head h:
        diagonal_mass[h] = mean_over_positions(alpha_{i,i}^h)
        off_diagonal_mass[h] = mean_over_positions(sum_{j<i} alpha_{i,j}^h)

    Returns:
        dict with per-head diagonal mass and summary statistics
    """
    # attn_weights: [B, H, T, T]
    B, H, T, _ = attn_weights.shape

    # Average over batch
    attn_avg = attn_weights.mean(dim=0)  # [H, T, T]

    results = {
        "per_head": {},
        "summary": {},
    }

    diagonal_masses = []
    off_diagonal_masses = []

    for h in range(H):
        # Diagonal mass: mean of alpha_{i,i} for i > 0 (position 0 has no diagonal)
        diag_weights = torch.diag(attn_avg[h])  # [T]
        diagonal_mass = diag_weights[1:].mean().item()  # Exclude position 0

        # Off-diagonal mass: for each position i, sum_{j<i} alpha_{i,j}
        # This is 1 - alpha_{i,i} for i > 0
        off_diag_mass = 1.0 - diagonal_mass

        diagonal_masses.append(diagonal_mass)
        off_diagonal_masses.append(off_diag_mass)

        results["per_head"][f"head_{h}"] = {
            "diagonal_mass": diagonal_mass,
            "off_diagonal_mass": off_diag_mass,
            "is_identity_like": diagonal_mass > threshold,
        }

    # Summary
    n_identity_like = sum(1 for d in diagonal_masses if d > threshold)

    results["summary"] = {
        "mean_diagonal_mass": np.mean(diagonal_masses),
        "std_diagonal_mass": np.std(diagonal_masses),
        "min_diagonal_mass": np.min(diagonal_masses),
        "max_diagonal_mass": np.max(diagonal_masses),
        "n_identity_like_heads": n_identity_like,
        "pct_identity_like_heads": 100 * n_identity_like / H,
        "threshold_used": threshold,
    }

    return results, diagonal_masses


def plot_diagonal_mass(diagonal_masses, save_dir):
    """Plot diagonal mass per head."""
    fig, ax = plt.subplots(figsize=(6, 4))

    n_heads = len(diagonal_masses)
    x = np.arange(n_heads)

    colors = ["#1f77b4" if d > 0.5 else "#d62728" for d in diagonal_masses]
    ax.bar(x, diagonal_masses, color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(
        y=0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)"
    )
    ax.axhline(y=1.0, color="lightgray", linestyle=":", linewidth=0.5)

    ax.set_xlabel("Head Index")
    ax.set_ylabel("Diagonal Mass (Self-Attention)")
    ax.set_title("Test 1: Near-Identity Kernel Verification")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.legend(loc="upper right")

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"{save_dir}/test1_diagonal_mass.{fmt}", dpi=300, bbox_inches="tight"
        )
    plt.close()

    return f"{save_dir}/test1_diagonal_mass.png"


# =============================================================================
# TEST 2: Compute Effective Linear Map A = W_O @ W_V
# =============================================================================


def test_effective_linear_map(model):
    """
    Test 2: Compute the effective linear map A = W_O @ W_V.

    For multi-head attention, the combined transformation is:
        A = W_O @ [W_V^1; W_V^2; ...; W_V^H]  where each W_V^h is [head_dim, D]

    But since W_V is applied per-head and W_O reassembles:
        For input x: attn_out = W_O @ concat(W_V^h @ x for all h)

    We compute the full D x D matrix A where: attn_out_i ≈ A @ x_i (under identity kernel)
    """
    config = model.config
    D = config.n_embd
    n_head = config.n_head
    head_dim = D // n_head

    attn2 = model.block2.attn

    # W_qkv: [3*D, D] - rows are [Q; K; V]
    W_qkv = attn2.c_attn.weight  # [3*D, D]
    W_V = W_qkv[2 * D :, :]  # [D, D]

    # W_O: [D, D]
    W_O = attn2.c_proj.weight  # [D, D]

    # Effective linear map: output = W_O @ W_V @ x
    # But attention reshapes: W_V produces [B, T, D], then views as [B, T, H, d] -> [B, H, T, d]
    # After attention (identity kernel): still [B, H, T, d], then reshape to [B, T, D]
    # Then W_O: [D, D] @ [D] -> [D]

    # So A = W_O @ W_V is correct for the combined transformation
    A = W_O @ W_V  # [D, D]

    # Analyze A
    results = {
        "A_shape": list(A.shape),
        "A_norm": A.norm().item(),
        "A_frobenius": A.norm("fro").item(),
        "A_rank_approx": torch.linalg.matrix_rank(A.float()).item(),
    }

    # SVD analysis
    U, S, Vh = torch.linalg.svd(A.float())
    results["top_10_singular_values"] = S[:10].tolist()
    results["singular_value_ratio_1_10"] = (
        (S[0] / S[9]).item() if S[9] > 0 else float("inf")
    )

    # Effective rank (using entropy of normalized singular values)
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()
    results["effective_rank"] = effective_rank

    return A, results


def plot_matrix_A(A, save_dir):
    """Plot the effective linear map A = W_O @ W_V."""
    A_np = A.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Full matrix
    im0 = axes[0].imshow(
        A_np,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.abs(A_np).max(),
        vmax=np.abs(A_np).max(),
    )
    axes[0].set_title("A = W_O @ W_V")
    axes[0].set_xlabel("Input Dimension")
    axes[0].set_ylabel("Output Dimension")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # SVD spectrum
    U, S, Vh = np.linalg.svd(A_np)
    axes[1].semilogy(S[:50], "o-", markersize=3)
    axes[1].set_xlabel("Singular Value Index")
    axes[1].set_ylabel("Singular Value (log scale)")
    axes[1].set_title("SVD Spectrum of A")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{save_dir}/test2_matrix_A.{fmt}", dpi=300, bbox_inches="tight")
    plt.close()

    return f"{save_dir}/test2_matrix_A.png"


# =============================================================================
# TEST 3: Fit w_dec and Measure Row Alignment
# =============================================================================


def test_w_dec_alignment(ln1_r1, positions, A, device="cuda"):
    """
    Test 3: Fit w_dec probe on LN(r1) and measure alignment with rows of A.

    1. Train linear probe: pred_i = w_dec^T @ LN(r1)_i + b
    2. For each row a_k of A: alignment[k] = cos(a_k, w_dec)
    3. Report top-k coordinates by |alignment|
    """
    # ln1_r1: [B, T, D], positions: [B*T]
    B, T, D = ln1_r1.shape

    # Flatten for regression
    X = ln1_r1.reshape(-1, D)  # [B*T, D]
    y = positions.float()  # [B*T]

    # Train linear probe using Ridge regression (sklearn for stability)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    probe = Ridge(alpha=0.01)
    probe.fit(X_np, y_np)

    w_dec = torch.tensor(probe.coef_, dtype=torch.float32, device=device)  # [D]
    b_dec = probe.intercept_

    # Probe performance
    y_pred = probe.predict(X_np)
    r2 = 1 - np.sum((y_np - y_pred) ** 2) / np.sum((y_np - y_np.mean()) ** 2)
    mae = np.mean(np.abs(y_np - y_pred))

    # Normalize w_dec for cosine similarity
    w_dec_norm = w_dec / (w_dec.norm() + 1e-8)

    # Compute alignment of each row of A with w_dec
    A_rows = A  # [D, D] - each row A[k, :] is an output coordinate
    A_rows_norm = A_rows / (A_rows.norm(dim=1, keepdim=True) + 1e-8)

    alignments = (A_rows_norm @ w_dec_norm).detach().cpu().numpy()  # [D]

    # Top coordinates by |alignment|
    sorted_idx = np.argsort(-np.abs(alignments))

    results = {
        "probe_r2": r2,
        "probe_mae": mae,
        "w_dec_norm": w_dec.norm().item(),
        "top_20_aligned_coords": [
            {
                "coord": int(sorted_idx[i]),
                "alignment": float(alignments[sorted_idx[i]]),
                "abs_alignment": float(np.abs(alignments[sorted_idx[i]])),
            }
            for i in range(20)
        ],
        "mean_abs_alignment": float(np.mean(np.abs(alignments))),
        "max_abs_alignment": float(np.max(np.abs(alignments))),
    }

    return w_dec, alignments, results


def plot_alignment_histogram(alignments, save_dir):
    """Plot histogram of row alignments with w_dec."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram of alignments
    axes[0].hist(alignments, bins=50, edgecolor="black", linewidth=0.5)
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Cosine Alignment with w_dec")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Test 3: Row Alignment Distribution")

    # Sorted absolute alignments
    sorted_abs = np.sort(np.abs(alignments))[::-1]
    axes[1].bar(range(50), sorted_abs[:50], edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Coordinate Rank (by |alignment|)")
    axes[1].set_ylabel("|Alignment with w_dec|")
    axes[1].set_title("Top 50 Aligned Coordinates")

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{save_dir}/test3_alignment.{fmt}", dpi=300, bbox_inches="tight")
    plt.close()

    return f"{save_dir}/test3_alignment.png"


# =============================================================================
# TEST 4: Verify Coordinate Concentration
# =============================================================================


def test_coordinate_concentration(
    ln1_r1, r2_attn, positions, alignments, device="cuda"
):
    """
    Test 4: Verify that high-alignment coordinates become near-perfect after Attn2.

    For each coordinate k:
        before_r2[k] = R²(LN(r1)[:, k], positions)
        after_r2[k] = R²(r2_attn[:, k], positions)

    Report which coords become near-perfect (R² > 0.9) after Attn2.
    Verify these are the high-alignment coords from Test 3.
    """
    B, T, D = ln1_r1.shape

    # Flatten
    ln1_r1_flat = ln1_r1.reshape(-1, D)  # [B*T, D]
    r2_attn_flat = r2_attn.reshape(-1, D)  # [B*T, D]
    y = positions.float().cpu().numpy()

    before_r2 = np.zeros(D)
    after_r2 = np.zeros(D)

    ss_tot = np.sum((y - y.mean()) ** 2)

    for k in range(D):
        # Before Attn2
        x_before = ln1_r1_flat[:, k].cpu().numpy()
        # Simple linear regression
        slope = np.cov(x_before, y)[0, 1] / (np.var(x_before) + 1e-8)
        intercept = y.mean() - slope * x_before.mean()
        y_pred = slope * x_before + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        before_r2[k] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # After Attn2
        x_after = r2_attn_flat[:, k].cpu().numpy()
        slope = np.cov(x_after, y)[0, 1] / (np.var(x_after) + 1e-8)
        intercept = y.mean() - slope * x_after.mean()
        y_pred = slope * x_after + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        after_r2[k] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Find coordinates that improved significantly
    improvement = after_r2 - before_r2

    # High-alignment coords (top 20 by |alignment|)
    top_aligned = np.argsort(-np.abs(alignments))[:20]

    results = {
        "before_r2_mean": float(np.mean(before_r2)),
        "after_r2_mean": float(np.mean(after_r2)),
        "n_coords_r2_above_0.9_before": int(np.sum(before_r2 > 0.9)),
        "n_coords_r2_above_0.9_after": int(np.sum(after_r2 > 0.9)),
        "top_improved_coords": [
            {
                "coord": int(np.argsort(-improvement)[i]),
                "before_r2": float(before_r2[np.argsort(-improvement)[i]]),
                "after_r2": float(after_r2[np.argsort(-improvement)[i]]),
                "improvement": float(improvement[np.argsort(-improvement)[i]]),
                "alignment": float(alignments[np.argsort(-improvement)[i]]),
            }
            for i in range(10)
        ],
        "alignment_vs_improvement_corr": float(
            np.corrcoef(np.abs(alignments), improvement)[0, 1]
        ),
    }

    # Check if top-aligned coords are also top-improved
    top_improved = np.argsort(-improvement)[:20]
    overlap = len(set(top_aligned) & set(top_improved))
    results["overlap_top20_aligned_vs_improved"] = overlap

    return before_r2, after_r2, results


def plot_coordinate_r2(before_r2, after_r2, alignments, save_dir):
    """Plot per-coordinate R² before and after Attn2."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Scatter: before vs after R²
    axes[0].scatter(before_r2, after_r2, alpha=0.3, s=10)
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=1, label="y=x")
    axes[0].set_xlabel("R² Before Attn2")
    axes[0].set_ylabel("R² After Attn2")
    axes[0].set_title("Coordinate-wise R² Change")
    axes[0].legend(loc="lower right")
    axes[0].set_xlim(-0.1, 1.1)
    axes[0].set_ylim(-0.1, 1.1)

    # Color by alignment
    improvement = after_r2 - before_r2
    sc = axes[1].scatter(
        np.abs(alignments), improvement, c=after_r2, cmap="viridis", alpha=0.5, s=10
    )
    axes[1].set_xlabel("|Alignment with w_dec|")
    axes[1].set_ylabel("R² Improvement")
    axes[1].set_title("Alignment vs Improvement")
    plt.colorbar(sc, ax=axes[1], label="Final R²")

    # Top 50 coordinates sorted by final R²
    sorted_idx = np.argsort(-after_r2)[:50]
    x = np.arange(50)
    axes[2].bar(x - 0.2, before_r2[sorted_idx], 0.4, label="Before Attn2", alpha=0.7)
    axes[2].bar(x + 0.2, after_r2[sorted_idx], 0.4, label="After Attn2", alpha=0.7)
    axes[2].set_xlabel("Coordinate Rank (by final R²)")
    axes[2].set_ylabel("R²")
    axes[2].set_title("Top 50 Coordinates by Final R²")
    axes[2].legend(loc="upper right")
    axes[2].set_xticks(x[::5])

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"{save_dir}/test4_coordinate_r2.{fmt}", dpi=300, bbox_inches="tight"
        )
    plt.close()

    return f"{save_dir}/test4_coordinate_r2.png"


# =============================================================================
# TEST 5: Decompose Local vs Non-Local
# =============================================================================


def test_local_vs_nonlocal(ln1_r1, attn_out2, A, device="cuda"):
    """
    Test 5: Decompose Attn2 output into local (A @ x_i) vs non-local (mixing residual).

    local_contribution = A @ LN(r1)_i  (what we'd get with perfect identity kernel)
    actual_attn_output = Attn2(x)_i
    eta_i = actual - local  (non-local mixing residual)

    Report:
        ||eta_i|| / ||actual||  (should be small if identity kernel)
        R²(local, actual)  (should be high)
    """
    B, T, D = ln1_r1.shape

    # Compute local contribution: A @ x_i for each position
    # ln1_r1: [B, T, D], A: [D, D]
    # local: [B, T, D]
    local_contribution = torch.einsum("btd,od->bto", ln1_r1, A)

    # Actual attention output (after W_O)
    actual = attn_out2  # [B, T, D]

    # Non-local residual
    eta = actual - local_contribution

    # Compute statistics
    local_norms = local_contribution.norm(dim=-1)  # [B, T]
    actual_norms = actual.norm(dim=-1)  # [B, T]
    eta_norms = eta.norm(dim=-1)  # [B, T]

    # Relative residual: ||eta|| / ||actual||
    relative_residual = eta_norms / (actual_norms + 1e-8)

    # R² between local and actual (flattened)
    local_flat = local_contribution.reshape(-1, D)
    actual_flat = actual.reshape(-1, D)

    # Per-dimension R²
    r2_per_dim = []
    for d in range(D):
        l = local_flat[:, d].detach().cpu().numpy()
        a = actual_flat[:, d].detach().cpu().numpy()
        ss_res = np.sum((a - l) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_per_dim.append(r2)

    # Overall cosine similarity
    cos_sim = F.cosine_similarity(local_flat, actual_flat, dim=-1).mean().item()

    results = {
        "mean_relative_residual": relative_residual.mean().item(),
        "std_relative_residual": relative_residual.std().item(),
        "median_relative_residual": relative_residual.median().item(),
        "max_relative_residual": relative_residual.max().item(),
        "mean_r2_per_dim": float(np.mean(r2_per_dim)),
        "min_r2_per_dim": float(np.min(r2_per_dim)),
        "mean_cosine_similarity": cos_sim,
        "local_norm_mean": local_norms.mean().item(),
        "actual_norm_mean": actual_norms.mean().item(),
        "eta_norm_mean": eta_norms.mean().item(),
    }

    # Per-position analysis
    per_pos_relative = relative_residual.detach().mean(dim=0).cpu().numpy()  # [T]
    results["per_position_relative_residual"] = per_pos_relative.tolist()

    return local_contribution.detach(), eta.detach(), results


def plot_local_vs_nonlocal(local_contribution, actual, eta, save_dir):
    """Plot local vs non-local decomposition."""
    B, T, D = local_contribution.shape

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Per-position relative residual
    local_norms = local_contribution.norm(dim=-1).mean(dim=0).detach().cpu().numpy()
    actual_norms = actual.norm(dim=-1).mean(dim=0).detach().cpu().numpy()
    eta_norms = eta.norm(dim=-1).mean(dim=0).detach().cpu().numpy()
    relative = eta_norms / (actual_norms + 1e-8)

    positions = np.arange(T)
    axes[0].plot(positions, relative, "b-", linewidth=1)
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("||eta|| / ||actual||")
    axes[0].set_title("Relative Non-Local Residual vs Position")
    axes[0].set_ylim(0, max(0.5, relative.max() * 1.1))

    # Norm comparison
    axes[1].plot(positions, local_norms, "g-", label="Local (A @ x)", linewidth=1)
    axes[1].plot(positions, actual_norms, "b-", label="Actual", linewidth=1)
    axes[1].plot(positions, eta_norms, "r-", label="Residual (eta)", linewidth=1)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Norm")
    axes[1].set_title("Component Norms vs Position")
    axes[1].legend(loc="upper right")

    # Scatter: local vs actual (sample positions)
    sample_idx = [0, 32, 64, 96, 127]
    local_flat = local_contribution[0, sample_idx, :].detach().cpu().numpy().flatten()
    actual_flat = actual[0, sample_idx, :].detach().cpu().numpy().flatten()

    axes[2].scatter(local_flat, actual_flat, alpha=0.1, s=1)
    lims = [
        min(local_flat.min(), actual_flat.min()),
        max(local_flat.max(), actual_flat.max()),
    ]
    axes[2].plot(lims, lims, "r--", linewidth=1)
    axes[2].set_xlabel("Local Prediction (A @ x)")
    axes[2].set_ylabel("Actual Attn Output")
    axes[2].set_title("Local vs Actual (Sample Positions)")

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"{save_dir}/test5_local_vs_nonlocal.{fmt}", dpi=300, bbox_inches="tight"
        )
    plt.close()

    return f"{save_dir}/test5_local_vs_nonlocal.png"


# =============================================================================
# Combined Summary Figure
# =============================================================================


def create_summary_figure(all_results, save_dir):
    """Create a single summary figure for the paper."""
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Diagonal mass per head
    ax1 = fig.add_subplot(gs[0, 0])
    diagonal_masses = [
        all_results["test1"]["per_head"][f"head_{h}"]["diagonal_mass"]
        for h in range(12)
    ]
    colors = ["#2ecc71" if d > 0.5 else "#e74c3c" for d in diagonal_masses]
    ax1.bar(range(12), diagonal_masses, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Diagonal Mass")
    ax1.set_title("(a) Near-Identity Kernel")
    ax1.set_ylim(0, 1.05)

    # 2. Row alignment distribution
    ax2 = fig.add_subplot(gs[0, 1])
    alignments = all_results["alignments"]
    ax2.hist(alignments, bins=40, edgecolor="black", linewidth=0.3, color="steelblue")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("cos(a_k, w_dec)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Row Alignment with w_dec")

    # 3. Coordinate R² improvement
    ax3 = fig.add_subplot(gs[0, 2])
    before_r2 = all_results["before_r2"]
    after_r2 = all_results["after_r2"]
    ax3.scatter(before_r2, after_r2, alpha=0.3, s=10, c="steelblue")
    ax3.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax3.set_xlabel("R² Before Attn2")
    ax3.set_ylabel("R² After Attn2")
    ax3.set_title("(c) Coordinate Concentration")
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)

    # 4. Local vs actual decomposition
    ax4 = fig.add_subplot(gs[1, 0])
    per_pos_residual = all_results["test5"]["per_position_relative_residual"]
    ax4.plot(per_pos_residual, "b-", linewidth=1)
    ax4.set_xlabel("Position")
    ax4.set_ylabel("||eta|| / ||actual||")
    ax4.set_title("(d) Non-Local Residual")
    ax4.set_ylim(0, max(0.3, max(per_pos_residual) * 1.1))

    # 5. Key metrics table
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")

    metrics_text = f"""
    R2 Identity-Kernel Verification Summary
    =======================================
    
    Test 1: Near-Identity Kernel
      - Heads with diagonal mass > 0.5: {all_results["test1"]["summary"]["n_identity_like_heads"]}/12 ({all_results["test1"]["summary"]["pct_identity_like_heads"]:.1f}%)
      - Mean diagonal mass: {all_results["test1"]["summary"]["mean_diagonal_mass"]:.3f}
    
    Test 2: Effective Linear Map A
      - Rank of A: {all_results["test2"]["A_rank_approx"]}
      - Effective rank: {all_results["test2"]["effective_rank"]:.1f}
    
    Test 3: w_dec Alignment
      - Probe R²: {all_results["test3"]["probe_r2"]:.4f}
      - Max |alignment|: {all_results["test3"]["max_abs_alignment"]:.3f}
    
    Test 4: Coordinate Concentration
      - Coords with R² > 0.9 (before): {all_results["test4"]["n_coords_r2_above_0.9_before"]}
      - Coords with R² > 0.9 (after): {all_results["test4"]["n_coords_r2_above_0.9_after"]}
      - Alignment-Improvement correlation: {all_results["test4"]["alignment_vs_improvement_corr"]:.3f}
    
    Test 5: Local vs Non-Local
      - Mean relative residual: {all_results["test5"]["mean_relative_residual"]:.4f}
      - Mean cosine(local, actual): {all_results["test5"]["mean_cosine_similarity"]:.4f}
    """

    ax5.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax5.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        "R2 Regime: Identity-Kernel + Subspace Projection Verification",
        fontsize=12,
        y=0.98,
    )

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"{save_dir}/r2_verification_summary.{fmt}", dpi=300, bbox_inches="tight"
        )
    plt.close()

    return f"{save_dir}/r2_verification_summary.png"


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="R2 Identity-Kernel Verification")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument(
        "--n_samples", type=int, default=256, help="Number of sequences"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup output directory
    save_dir = "out-2layer-mechanism/r2_identity_kernel"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)

    # Initialize W&B
    if args.wandb and HAS_WANDB:
        wandb.init(
            project="nope-2layer-mechanism",
            name="r2-identity-kernel-verification",
            config=vars(args),
        )

    print("=" * 80)
    print("R2 IDENTITY-KERNEL VERIFICATION")
    print("=" * 80)

    # Load model
    print("\n[1/6] Loading R2 model...")
    model, config = load_model("R2", device)

    # Load data
    print("\n[2/6] Loading data...")
    tokens = get_data(args.n_samples, config.block_size, device)
    print(f"  Loaded {tokens.shape[0]} sequences of length {tokens.shape[1]}")

    # Compute forward pass
    print("\n[3/6] Computing forward pass...")
    activations = compute_forward_pass(model, tokens, device)

    # Create position targets
    T = config.block_size
    positions = torch.arange(T, device=device).repeat(args.n_samples)  # [B*T]

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    all_results = {}

    # Test 1: Near-Identity Kernel
    print("\n" + "=" * 60)
    print("TEST 1: Near-Identity Kernel Verification")
    print("=" * 60)
    test1_results, diagonal_masses = test_near_identity_kernel(
        activations["attn_weights2"]
    )
    all_results["test1"] = test1_results
    print(f"  Mean diagonal mass: {test1_results['summary']['mean_diagonal_mass']:.4f}")
    print(
        f"  Identity-like heads (>0.5): {test1_results['summary']['n_identity_like_heads']}/12"
    )
    plot1_path = plot_diagonal_mass(diagonal_masses, f"{save_dir}/plots")

    # Test 2: Effective Linear Map
    print("\n" + "=" * 60)
    print("TEST 2: Compute Effective Linear Map A = W_O @ W_V")
    print("=" * 60)
    A, test2_results = test_effective_linear_map(model)
    all_results["test2"] = test2_results
    print(f"  A shape: {test2_results['A_shape']}")
    print(f"  A rank: {test2_results['A_rank_approx']}")
    print(f"  Effective rank: {test2_results['effective_rank']:.2f}")
    plot2_path = plot_matrix_A(A, f"{save_dir}/plots")

    # Test 3: w_dec Alignment
    print("\n" + "=" * 60)
    print("TEST 3: Fit w_dec and Measure Row Alignment")
    print("=" * 60)
    w_dec, alignments, test3_results = test_w_dec_alignment(
        activations["ln1_r1"], positions, A, device
    )
    all_results["test3"] = test3_results
    all_results["alignments"] = alignments.tolist()
    print(f"  Probe R²: {test3_results['probe_r2']:.4f}")
    print(f"  Max |alignment|: {test3_results['max_abs_alignment']:.4f}")
    print(f"  Top aligned coord: {test3_results['top_20_aligned_coords'][0]}")
    plot3_path = plot_alignment_histogram(alignments, f"{save_dir}/plots")

    # Test 4: Coordinate Concentration
    print("\n" + "=" * 60)
    print("TEST 4: Verify Coordinate Concentration")
    print("=" * 60)
    before_r2, after_r2, test4_results = test_coordinate_concentration(
        activations["ln1_r1"], activations["r2_attn"], positions, alignments, device
    )
    all_results["test4"] = test4_results
    all_results["before_r2"] = before_r2.tolist()
    all_results["after_r2"] = after_r2.tolist()
    print(
        f"  Coords with R² > 0.9 before: {test4_results['n_coords_r2_above_0.9_before']}"
    )
    print(
        f"  Coords with R² > 0.9 after: {test4_results['n_coords_r2_above_0.9_after']}"
    )
    print(
        f"  Alignment-Improvement corr: {test4_results['alignment_vs_improvement_corr']:.4f}"
    )
    plot4_path = plot_coordinate_r2(
        before_r2, after_r2, alignments, f"{save_dir}/plots"
    )

    # Test 5: Local vs Non-Local
    print("\n" + "=" * 60)
    print("TEST 5: Decompose Local vs Non-Local")
    print("=" * 60)
    local_contribution, eta, test5_results = test_local_vs_nonlocal(
        activations["ln1_r1"], activations["attn_out2"], A, device
    )
    all_results["test5"] = test5_results
    print(f"  Mean relative residual: {test5_results['mean_relative_residual']:.4f}")
    print(
        f"  Mean cosine(local, actual): {test5_results['mean_cosine_similarity']:.4f}"
    )
    plot5_path = plot_local_vs_nonlocal(
        local_contribution, activations["attn_out2"], eta, f"{save_dir}/plots"
    )

    # Create summary figure
    print("\n" + "=" * 60)
    print("Creating Summary Figure...")
    print("=" * 60)
    summary_path = create_summary_figure(all_results, f"{save_dir}/plots")

    # Save results
    results_path = f"{save_dir}/r2_identity_kernel_results.json"

    # Convert numpy arrays to lists for JSON
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    json_results = convert_to_serializable(
        {
            "test1": all_results["test1"],
            "test2": all_results["test2"],
            "test3": all_results["test3"],
            "test4": all_results["test4"],
            "test5": all_results["test5"],
        }
    )

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Log to W&B
    if args.wandb and HAS_WANDB:
        wandb.log(
            {
                "test1/mean_diagonal_mass": test1_results["summary"][
                    "mean_diagonal_mass"
                ],
                "test1/n_identity_like_heads": test1_results["summary"][
                    "n_identity_like_heads"
                ],
                "test2/A_rank": test2_results["A_rank_approx"],
                "test2/effective_rank": test2_results["effective_rank"],
                "test3/probe_r2": test3_results["probe_r2"],
                "test3/max_abs_alignment": test3_results["max_abs_alignment"],
                "test4/n_coords_above_0.9_before": test4_results[
                    "n_coords_r2_above_0.9_before"
                ],
                "test4/n_coords_above_0.9_after": test4_results[
                    "n_coords_r2_above_0.9_after"
                ],
                "test4/alignment_improvement_corr": test4_results[
                    "alignment_vs_improvement_corr"
                ],
                "test5/mean_relative_residual": test5_results["mean_relative_residual"],
                "test5/mean_cosine_similarity": test5_results["mean_cosine_similarity"],
            }
        )

        # Log images
        wandb.log(
            {
                "plots/test1_diagonal_mass": wandb.Image(plot1_path),
                "plots/test2_matrix_A": wandb.Image(plot2_path),
                "plots/test3_alignment": wandb.Image(plot3_path),
                "plots/test4_coordinate_r2": wandb.Image(plot4_path),
                "plots/test5_local_vs_nonlocal": wandb.Image(plot5_path),
                "plots/summary": wandb.Image(summary_path),
            }
        )

        wandb.finish()

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {save_dir}/")
    print(f"  - r2_identity_kernel_results.json")
    print(f"  - plots/test1_diagonal_mass.png")
    print(f"  - plots/test2_matrix_A.png")
    print(f"  - plots/test3_alignment.png")
    print(f"  - plots/test4_coordinate_r2.png")
    print(f"  - plots/test5_local_vs_nonlocal.png")
    print(f"  - plots/r2_verification_summary.png")


if __name__ == "__main__":
    main()
