"""
R2 Subspace Analysis: Low-Rank Non-Local Extractor Verification

This script implements two key tests to verify the "low-rank subspace projection" theory:

Test 6: SVD Component-wise R²
  - Compute SVD of B = W_O @ W_V
  - For each singular component m, compute R² of (U_r^T o_i)_m vs position
  - Prediction: Top few components carry most positional R²

Test 7: Subspace Ablation Sweep
  - For r in [10, 20, 50, 70, 100, ...]:
    - Compute ablated output: o_i^{ablated} = (I - U_r U_r^T) o_i
    - Measure position decoding R² with ablated outputs
  - Prediction: Removing top subspace kills decoding

Theory (from discussion):
  The R2 mechanism works via two-stage factorization:
  1. Sequence-space extraction: z_i = sum_j alpha_{i,j} C x_j (low-dim feature)
  2. Model-space alignment: o_i ≈ U z_i (project back via learned basis)

Usage:
    CUDA_VISIBLE_DEVICES=0 python r2_subspace_analysis.py [--model_dir DIR] [--wandb]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def load_model(model_dir, regime, device="cuda"):
    """Load a trained model for a specific regime."""
    checkpoint_path = f"{model_dir}/{regime}/best_ckpt.pt"

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


def compute_forward_pass_with_attn_output(model, tokens, device="cuda"):
    """
    Compute forward pass and return attention outputs for subspace analysis.

    Returns:
        o_i: Block 2 attention output (after W_O projection) [B, T, D]
        r2_attn: Residual after Attn2 = r1 + o_i [B, T, D]
        final_ln: Final layer norm output [B, T, D]
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

        # Block 2 (attention trained in R2)
        ln1_r1 = model.block2.ln_1(r1)

        attn2 = model.block2.attn
        qkv2 = attn2.c_attn(ln1_r1)
        q2, k2, v2 = qkv2.split(D, dim=2)
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        scores2 = (q2 @ k2.transpose(-2, -1)) / np.sqrt(head_dim)
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))
        attn_weights2 = F.softmax(scores2, dim=-1)

        # o_i = Attn2 output (after W_O)
        attn_out2_preproj = (attn_weights2 @ v2).transpose(1, 2).reshape(B, T, D)
        o_i = attn2.c_proj(
            attn_out2_preproj
        )  # This is the key output for subspace analysis

        r2_attn = r1 + o_i

        # Continue through rest of model
        ln2_out_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_out_b2)
        r2 = r2_attn + mlp_out2
        final_ln = model.ln_f(r2)

    return {
        "o_i": o_i,  # Block 2 attention output [B, T, D]
        "r1": r1,
        "r2_attn": r2_attn,
        "final_ln": final_ln,
        "ln1_r1": ln1_r1,
    }


def get_B_matrix(model):
    """
    Get the effective linear map B = W_O @ W_V from Block 2 attention.
    """
    config = model.config
    D = config.n_embd

    attn2 = model.block2.attn
    W_qkv = attn2.c_attn.weight  # [3*D, D]
    W_V = W_qkv[2 * D :, :]  # [D, D]
    W_O = attn2.c_proj.weight  # [D, D]

    B = W_O @ W_V  # [D, D]
    return B.detach()  # Detach to avoid grad issues


def compute_position_r2(X, positions):
    """Compute R² for position regression from features X."""
    # X: [N, d], positions: [N]
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
    y_np = positions.detach().cpu().numpy() if torch.is_tensor(positions) else positions

    # Handle 1D case
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)

    probe = Ridge(alpha=0.01)
    probe.fit(X_np, y_np)
    y_pred = probe.predict(X_np)

    ss_res = np.sum((y_np - y_pred) ** 2)
    ss_tot = np.sum((y_np - y_np.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return r2


# =============================================================================
# TEST 6: SVD Component-wise R²
# =============================================================================


def test_svd_component_r2(o_i, B, positions, device="cuda"):
    """
    Test 6: Project Attn2 outputs onto top singular vectors of B.

    For each singular component m, compute R² of (U^T o_i)_m vs position.

    Args:
        o_i: Attention output [B, T, D]
        B: Effective linear map W_O @ W_V [D, D]
        positions: Position indices [B*T]

    Returns:
        dict with per-component R² and summary stats
    """
    B_size, T, D = o_i.shape

    # Compute SVD of B
    B_float = B.float()
    U, S, Vh = torch.linalg.svd(B_float)

    # U: [D, D] - left singular vectors (columns)
    # S: [D] - singular values
    # Vh: [D, D] - right singular vectors (rows)

    # Flatten o_i for projection
    o_flat = o_i.reshape(-1, D).float()  # [B*T, D]

    # Project onto each left singular vector
    # projections[m] = U[:, m]^T @ o_i = o_i @ U[:, m]
    projections = (
        o_flat @ U
    )  # [B*T, D] - each column is projection onto that singular vector

    # Compute R² for each component
    component_r2 = []
    positions_np = positions.cpu().numpy()

    for m in range(min(D, 200)):  # Compute for first 200 components
        proj_m = projections[:, m].cpu().numpy()
        r2 = compute_position_r2(proj_m.reshape(-1, 1), positions_np)
        component_r2.append(r2)

    # Summary statistics
    component_r2 = np.array(component_r2)
    cumulative_r2 = []

    # Compute cumulative R² (using top-k components together)
    for k in [1, 5, 10, 20, 50, 100]:
        if k <= len(component_r2):
            top_k_proj = projections[:, :k].cpu().numpy()
            r2_k = compute_position_r2(top_k_proj, positions_np)
            cumulative_r2.append({"k": k, "r2": r2_k})

    results = {
        "per_component_r2": component_r2.tolist(),
        "singular_values": S[:100].cpu().numpy().tolist(),
        "cumulative_r2": cumulative_r2,
        "top_10_component_r2": component_r2[:10].tolist(),
        "max_single_component_r2": float(np.max(component_r2)),
        "argmax_component": int(np.argmax(component_r2)),
    }

    return results, U, S


def plot_svd_component_r2(results, S, save_dir):
    """Plot SVD component-wise R² analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    component_r2 = np.array(results["per_component_r2"])
    singular_values = np.array(results["singular_values"])

    # (a) Per-component R²
    ax = axes[0, 0]
    ax.bar(
        range(min(50, len(component_r2))),
        component_r2[:50],
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("Singular Component Index")
    ax.set_ylabel("Position R²")
    ax.set_title("(a) Per-Component Position R²")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)

    # (b) Singular value spectrum
    ax = axes[0, 1]
    ax.semilogy(range(len(singular_values)), singular_values, "o-", markersize=2)
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Singular Value (log)")
    ax.set_title("(b) Singular Value Spectrum of B")
    ax.grid(True, alpha=0.3)

    # (c) Cumulative R² vs number of components
    ax = axes[1, 0]
    cum_r2 = results["cumulative_r2"]
    ks = [c["k"] for c in cum_r2]
    r2s = [c["r2"] for c in cum_r2]
    ax.plot(ks, r2s, "o-", color="steelblue", markersize=8, linewidth=2)
    ax.set_xlabel("Number of Top Components (k)")
    ax.set_ylabel("Cumulative Position R²")
    ax.set_title("(c) Cumulative R² vs Components")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # (d) Component R² vs Singular Value
    ax = axes[1, 1]
    n_plot = min(100, len(component_r2), len(singular_values))
    sc = ax.scatter(
        singular_values[:n_plot],
        component_r2[:n_plot],
        c=range(n_plot),
        cmap="viridis",
        s=20,
        alpha=0.7,
    )
    ax.set_xlabel("Singular Value")
    ax.set_ylabel("Component Position R²")
    ax.set_title("(d) R² vs Singular Value")
    ax.set_xscale("log")
    plt.colorbar(sc, ax=ax, label="Component Index")

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"{save_dir}/test6_svd_component_r2.{fmt}", dpi=300, bbox_inches="tight"
        )
    plt.close()

    return f"{save_dir}/test6_svd_component_r2.png"


# =============================================================================
# TEST 7: Subspace Ablation Sweep
# =============================================================================


def test_subspace_ablation(o_i, r1, model, positions, U, r_values=None, device="cuda"):
    """
    Test 7: Subspace ablation sweep.

    For each r in r_values:
        - Compute U_r = U[:, :r] (top-r left singular vectors)
        - Ablate: o_i^{ablated} = (I - U_r U_r^T) o_i
        - Compute position R² from final output with ablated attention

    Args:
        o_i: Original attention output [B, T, D]
        r1: Block 1 output [B, T, D]
        model: The model (for MLP2 and final LN)
        positions: Position indices [B*T]
        U: Left singular vectors of B [D, D]
        r_values: List of subspace dimensions to ablate

    Returns:
        dict with R² for each ablation level
    """
    B_size, T, D = o_i.shape

    if r_values is None:
        r_values = [1, 2, 5, 10, 20, 50, 70, 100, 150, 200, 300, 500]

    # Filter r_values to valid range
    r_values = [r for r in r_values if r <= D]

    results = {
        "r_values": r_values,
        "original_r2": None,
        "ablated_r2": [],
        "retained_r2": [],  # R² when keeping ONLY top-r (opposite of ablation)
    }

    positions_np = positions.cpu().numpy()

    with torch.no_grad():
        # Original R² (no ablation)
        r2_attn_orig = r1 + o_i
        ln2_out = model.block2.ln_2(r2_attn_orig)
        mlp_out2 = model.block2.mlp(ln2_out)
        r2_orig = r2_attn_orig + mlp_out2
        final_orig = model.ln_f(r2_orig)

        final_flat_orig = final_orig.reshape(-1, D).cpu().numpy()
        original_r2 = compute_position_r2(final_flat_orig, positions_np)
        results["original_r2"] = original_r2

        # Ablation sweep
        U_float = U.float()
        o_i_float = o_i.float()

        for r in r_values:
            # Top-r singular vectors
            U_r = U_float[:, :r]  # [D, r]

            # Projector onto top-r subspace: P_r = U_r @ U_r^T
            # Ablated output: o_ablated = (I - P_r) @ o_i = o_i - U_r @ (U_r^T @ o_i)

            # o_i: [B, T, D], need to project each position
            o_flat = o_i_float.reshape(-1, D)  # [B*T, D]

            # Project: U_r^T @ o_i^T -> [r, B*T], then U_r @ ... -> [D, B*T]
            proj_coeff = o_flat @ U_r  # [B*T, r]
            proj = proj_coeff @ U_r.T  # [B*T, D]

            o_ablated = (o_flat - proj).reshape(B_size, T, D)  # Remove top-r subspace
            o_retained = proj.reshape(B_size, T, D)  # Keep only top-r subspace

            # Compute R² with ablated output
            r2_attn_abl = r1 + o_ablated.to(r1.dtype)
            ln2_out_abl = model.block2.ln_2(r2_attn_abl)
            mlp_out2_abl = model.block2.mlp(ln2_out_abl)
            r2_abl = r2_attn_abl + mlp_out2_abl
            final_abl = model.ln_f(r2_abl)

            final_flat_abl = final_abl.reshape(-1, D).cpu().numpy()
            ablated_r2 = compute_position_r2(final_flat_abl, positions_np)
            results["ablated_r2"].append({"r": r, "r2": ablated_r2})

            # Compute R² with retained-only output
            r2_attn_ret = r1 + o_retained.to(r1.dtype)
            ln2_out_ret = model.block2.ln_2(r2_attn_ret)
            mlp_out2_ret = model.block2.mlp(ln2_out_ret)
            r2_ret = r2_attn_ret + mlp_out2_ret
            final_ret = model.ln_f(r2_ret)

            final_flat_ret = final_ret.reshape(-1, D).cpu().numpy()
            retained_r2 = compute_position_r2(final_flat_ret, positions_np)
            results["retained_r2"].append({"r": r, "r2": retained_r2})

    return results


def plot_subspace_ablation(results, save_dir):
    """Plot subspace ablation sweep results."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    original_r2 = results["original_r2"]
    ablated = results["ablated_r2"]
    retained = results["retained_r2"]

    rs_abl = [a["r"] for a in ablated]
    r2s_abl = [a["r2"] for a in ablated]

    rs_ret = [a["r"] for a in retained]
    r2s_ret = [a["r2"] for a in retained]

    # (a) Ablation: R² after removing top-r subspace
    ax = axes[0]
    ax.axhline(
        y=original_r2,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Original R²={original_r2:.4f}",
    )
    ax.plot(
        rs_abl,
        r2s_abl,
        "o-",
        color="#e74c3c",
        markersize=8,
        linewidth=2,
        label="After removing top-r",
    )
    ax.set_xlabel("Subspace Dimension (r)")
    ax.set_ylabel("Position R²")
    ax.set_title("(a) Ablation: Remove Top-r Subspace")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (b) Retention: R² keeping only top-r subspace
    ax = axes[1]
    ax.axhline(
        y=original_r2,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Original R²={original_r2:.4f}",
    )
    ax.plot(
        rs_ret,
        r2s_ret,
        "s-",
        color="#2ecc71",
        markersize=8,
        linewidth=2,
        label="Keeping only top-r",
    )
    ax.set_xlabel("Subspace Dimension (r)")
    ax.set_ylabel("Position R²")
    ax.set_title("(b) Retention: Keep Only Top-r Subspace")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"{save_dir}/test7_subspace_ablation.{fmt}", dpi=300, bbox_inches="tight"
        )
    plt.close()

    return f"{save_dir}/test7_subspace_ablation.png"


# =============================================================================
# Combined Summary Figure
# =============================================================================


def create_subspace_summary(test6_results, test7_results, model_name, save_dir):
    """Create a combined summary figure for subspace analysis."""
    fig = plt.figure(figsize=(12, 8))

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # (a) Top component R²
    ax1 = fig.add_subplot(gs[0, 0])
    component_r2 = np.array(test6_results["per_component_r2"][:50])
    ax1.bar(
        range(len(component_r2)),
        component_r2,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
    )
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Singular Component Index")
    ax1.set_ylabel("Position R²")
    ax1.set_title("(a) Per-Component Position R²")

    # (b) Cumulative R²
    ax2 = fig.add_subplot(gs[0, 1])
    cum_r2 = test6_results["cumulative_r2"]
    ks = [c["k"] for c in cum_r2]
    r2s = [c["r2"] for c in cum_r2]
    ax2.plot(ks, r2s, "o-", color="steelblue", markersize=8, linewidth=2)
    ax2.set_xlabel("Number of Top Components (k)")
    ax2.set_ylabel("Cumulative Position R²")
    ax2.set_title("(b) Cumulative R² vs Components")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    # (c) Ablation curve
    ax3 = fig.add_subplot(gs[1, 0])
    original_r2 = test7_results["original_r2"]
    ablated = test7_results["ablated_r2"]
    rs_abl = [a["r"] for a in ablated]
    r2s_abl = [a["r2"] for a in ablated]
    ax3.axhline(
        y=original_r2,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Original={original_r2:.3f}",
    )
    ax3.plot(
        rs_abl,
        r2s_abl,
        "o-",
        color="#e74c3c",
        markersize=6,
        linewidth=2,
        label="After ablation",
    )
    ax3.set_xlabel("Ablated Subspace Dim (r)")
    ax3.set_ylabel("Position R²")
    ax3.set_title("(c) Ablation: Remove Top-r")
    ax3.set_xscale("log")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # (d) Retention curve
    ax4 = fig.add_subplot(gs[1, 1])
    retained = test7_results["retained_r2"]
    rs_ret = [a["r"] for a in retained]
    r2s_ret = [a["r2"] for a in retained]
    ax4.axhline(
        y=original_r2,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Original={original_r2:.3f}",
    )
    ax4.plot(
        rs_ret,
        r2s_ret,
        "s-",
        color="#2ecc71",
        markersize=6,
        linewidth=2,
        label="Keeping only top-r",
    )
    ax4.set_xlabel("Retained Subspace Dim (r)")
    ax4.set_ylabel("Position R²")
    ax4.set_title("(d) Retention: Keep Only Top-r")
    ax4.set_xscale("log")
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Subspace Analysis: {model_name}", fontsize=12, y=0.98)

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"{save_dir}/subspace_summary_{model_name}.{fmt}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()

    return f"{save_dir}/subspace_summary_{model_name}.png"


# =============================================================================
# Main
# =============================================================================


def run_analysis(model_dir, regime, n_samples, device, save_dir, use_wandb=False):
    """Run full subspace analysis for a single model."""

    print(f"\n{'=' * 60}")
    print(f"SUBSPACE ANALYSIS: {model_dir}/{regime}")
    print(f"{'=' * 60}")

    # Load model
    print("\n[1/5] Loading model...")
    model, config = load_model(model_dir, regime, device)
    model_name = f"{os.path.basename(model_dir)}_{regime}"

    # Get model info
    n_head = config.n_head
    n_embd = config.n_embd
    print(f"  Model: n_embd={n_embd}, n_head={n_head}")

    # Load data
    print("\n[2/5] Loading data...")
    tokens = get_data(n_samples, config.block_size, device)
    T = config.block_size
    positions = torch.arange(T, device=device).repeat(n_samples)
    print(f"  Loaded {n_samples} sequences of length {T}")

    # Forward pass
    print("\n[3/5] Computing forward pass...")
    activations = compute_forward_pass_with_attn_output(model, tokens, device)
    o_i = activations["o_i"]
    r1 = activations["r1"]

    # Get B matrix
    B = get_B_matrix(model)
    print(f"  B matrix shape: {B.shape}")

    # Create output directory
    regime_save_dir = f"{save_dir}/{model_name}"
    os.makedirs(regime_save_dir, exist_ok=True)

    # Test 6: SVD Component R²
    print("\n[4/5] Test 6: SVD Component-wise R²...")
    test6_results, U, S = test_svd_component_r2(o_i, B, positions, device)
    print(
        f"  Max single-component R²: {test6_results['max_single_component_r2']:.4f} (component {test6_results['argmax_component']})"
    )
    print(f"  Top-10 cumulative R²: {test6_results['cumulative_r2'][2]['r2']:.4f}")
    plot6_path = plot_svd_component_r2(test6_results, S, regime_save_dir)

    # Test 7: Subspace Ablation
    print("\n[5/5] Test 7: Subspace Ablation Sweep...")
    test7_results = test_subspace_ablation(o_i, r1, model, positions, U, device=device)
    print(f"  Original R²: {test7_results['original_r2']:.4f}")

    # Find where ablation drops R² by 50%
    original = test7_results["original_r2"]
    for abl in test7_results["ablated_r2"]:
        if abl["r2"] < original * 0.5:
            print(f"  R² drops below 50% when removing top-{abl['r']} subspace")
            break

    plot7_path = plot_subspace_ablation(test7_results, regime_save_dir)

    # Summary figure
    summary_path = create_subspace_summary(
        test6_results, test7_results, model_name, regime_save_dir
    )

    # Save results
    all_results = {
        "model_name": model_name,
        "n_head": n_head,
        "n_embd": n_embd,
        "test6": test6_results,
        "test7": test7_results,
    }

    results_path = f"{regime_save_dir}/subspace_analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to: {results_path}")

    # W&B logging
    if use_wandb and HAS_WANDB:
        wandb.log(
            {
                f"{model_name}/test6_max_component_r2": test6_results[
                    "max_single_component_r2"
                ],
                f"{model_name}/test7_original_r2": test7_results["original_r2"],
                f"{model_name}/plots/test6_svd": wandb.Image(plot6_path),
                f"{model_name}/plots/test7_ablation": wandb.Image(plot7_path),
                f"{model_name}/plots/summary": wandb.Image(summary_path),
            }
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="R2 Subspace Analysis")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="out-2layer-mechanism",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="R2",
        help="Regime to analyze (R0, R2, etc.) or 'all' for all available",
    )
    parser.add_argument(
        "--n_samples", type=int, default=256, help="Number of sequences"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup output directory
    save_dir = f"{args.model_dir}/subspace_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize W&B
    if args.wandb and HAS_WANDB:
        wandb.init(
            project="nope-2layer-mechanism",
            name=f"subspace-analysis-{os.path.basename(args.model_dir)}",
            config=vars(args),
        )

    # Determine regimes to analyze
    if args.regime == "all":
        # Find all available regimes
        regimes = []
        for d in os.listdir(args.model_dir):
            ckpt_path = os.path.join(args.model_dir, d, "best_ckpt.pt")
            if os.path.exists(ckpt_path):
                regimes.append(d)
        print(f"Found regimes: {regimes}")
    else:
        regimes = [args.regime]

    # Run analysis for each regime
    all_results = {}
    for regime in regimes:
        try:
            results = run_analysis(
                args.model_dir, regime, args.n_samples, device, save_dir, args.wandb
            )
            all_results[regime] = results
        except Exception as e:
            print(f"Error analyzing {regime}: {e}")
            continue

    # Finish W&B
    if args.wandb and HAS_WANDB:
        wandb.finish()

    print("\n" + "=" * 60)
    print("SUBSPACE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {save_dir}/")


if __name__ == "__main__":
    main()
