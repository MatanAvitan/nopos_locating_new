"""
SVD Component Analysis: What's Special About the Top-5 Position-Encoding Components?

This script investigates the geometric structure of the SVD components that encode position.

Key Questions:
1. What do the left/right singular vectors look like? (sparse/dense, aligned with dims?)
2. How do projections onto each component vary with position? (linear, log, power law?)
3. Why do some high-σ components have low position R²?
4. Do components encode different "aspects" of position (like Fourier frequencies)?
5. Is the position subspace similar across R0 and R2?

Usage:
    CUDA_VISIBLE_DEVICES=0 python svd_component_analysis.py
"""

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
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "text.usetex": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_model(model_dir, regime, device="cuda"):
    """Load a trained model."""
    checkpoint_path = f"{model_dir}/{regime}/best_ckpt.pt"
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
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, config


def get_B_matrix_and_svd(model):
    """Get B = W_O @ W_V and its SVD."""
    config = model.config
    D = config.n_embd

    attn2 = model.block2.attn
    W_qkv = attn2.c_attn.weight  # [3*D, D]
    W_V = W_qkv[2 * D :, :].detach()  # [D, D]
    W_O = attn2.c_proj.weight.detach()  # [D, D]

    B = W_O @ W_V  # [D, D]

    # SVD: B = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(B.float())

    return B, U, S, Vh


def get_data(n_samples, block_size, device="cuda"):
    """Load OpenWebText data."""
    data_path = "data/openwebtext/train.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (n_samples,))
    tokens = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    return tokens.to(device)


def compute_attention_output(model, tokens, device="cuda"):
    """Compute Block 2 attention output o_i for each position."""
    config = model.config
    B_size, T = tokens.shape
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
        q1 = q1.view(B_size, T, n_head, head_dim).transpose(1, 2)
        k1 = k1.view(B_size, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B_size, T, n_head, head_dim).transpose(1, 2)

        scores1 = (q1 @ k1.transpose(-2, -1)) / np.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores1 = scores1.masked_fill(causal_mask, float("-inf"))
        attn_weights1 = F.softmax(scores1, dim=-1)

        attn_out1 = (attn_weights1 @ v1).transpose(1, 2).reshape(B_size, T, D)
        attn_out1 = attn1.c_proj(attn_out1)
        r1_attn = e + attn_out1

        ln2_out_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_out_b1)
        r1 = r1_attn + mlp_out1

        # Block 2 attention
        ln1_r1 = model.block2.ln_1(r1)
        attn2 = model.block2.attn
        qkv2 = attn2.c_attn(ln1_r1)
        q2, k2, v2 = qkv2.split(D, dim=2)
        q2 = q2.view(B_size, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B_size, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B_size, T, n_head, head_dim).transpose(1, 2)

        scores2 = (q2 @ k2.transpose(-2, -1)) / np.sqrt(head_dim)
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))
        attn_weights2 = F.softmax(scores2, dim=-1)

        # o_i = Attn2 output (after W_O)
        attn_out2_preproj = (attn_weights2 @ v2).transpose(1, 2).reshape(B_size, T, D)
        o_i = attn2.c_proj(attn_out2_preproj)

    return o_i, attn_weights2


# =============================================================================
# PHASE 1: Visualize SVD Components
# =============================================================================


def phase1_visualize_svd_components(U, S, Vh, save_dir):
    """Visualize the structure of top SVD components."""
    print("\n[Phase 1] Visualizing SVD components...")

    n_top = 10  # Analyze top 10 components
    D = U.shape[0]

    U_np = U.cpu().numpy()
    S_np = S.cpu().numpy()
    Vh_np = Vh.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # (a) Singular value spectrum
    ax = axes[0, 0]
    ax.semilogy(S_np[:100], "o-", markersize=3)
    ax.axvline(x=5, color="red", linestyle="--", alpha=0.7, label="Top-5 boundary")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title("(a) Singular Value Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Heatmap of top-10 left singular vectors
    ax = axes[0, 1]
    # Show subset of dimensions for visibility
    dim_subset = np.linspace(0, D - 1, 100).astype(int)
    im = ax.imshow(
        U_np[dim_subset, :n_top].T, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1
    )
    ax.set_xlabel("Embedding Dimension (sampled)")
    ax.set_ylabel("SVD Component")
    ax.set_title("(b) Left Singular Vectors U[:, 0:10]")
    plt.colorbar(im, ax=ax)

    # (c) Histogram of U values for top-5 vs rest
    ax = axes[0, 2]
    ax.hist(
        U_np[:, :5].flatten(),
        bins=50,
        alpha=0.7,
        label="Top-5 components",
        density=True,
    )
    ax.hist(
        U_np[:, 50:55].flatten(),
        bins=50,
        alpha=0.7,
        label="Components 50-55",
        density=True,
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("(c) Distribution of U Values")
    ax.legend()

    # (d) Sparsity analysis: fraction of "large" values per component
    ax = axes[1, 0]
    threshold = 0.05  # Consider values > 0.05 as "active"
    sparsity = [(np.abs(U_np[:, m]) > threshold).mean() for m in range(min(50, D))]
    ax.bar(range(len(sparsity)), sparsity)
    ax.set_xlabel("SVD Component")
    ax.set_ylabel("Fraction of dims > 0.05")
    ax.set_title("(d) Sparsity of U Columns")
    ax.axhline(
        y=np.mean(sparsity),
        color="red",
        linestyle="--",
        label=f"Mean={np.mean(sparsity):.3f}",
    )
    ax.legend()

    # (e) Top dimensions for each of the top-5 components
    ax = axes[1, 1]
    for m in range(5):
        top_dims = np.argsort(np.abs(U_np[:, m]))[-10:]  # Top 10 dims
        ax.scatter([m] * 10, top_dims, alpha=0.6, s=20)
    ax.set_xlabel("SVD Component")
    ax.set_ylabel("Top Embedding Dimensions")
    ax.set_title("(e) Which Dims Are Active per Component")

    # (f) Correlation between U columns (are top components orthogonal?)
    ax = axes[1, 2]
    corr_matrix = np.corrcoef(U_np[:, :n_top].T)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Component")
    ax.set_ylabel("Component")
    ax.set_title("(f) Correlation Between U Columns")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase1_svd_structure.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/phase1_svd_structure.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Additional analysis: print statistics
    print(f"  Singular values (top 10): {S_np[:10].round(2)}")
    print(
        f"  U column norms (should be 1): {np.linalg.norm(U_np[:, :5], axis=0).round(4)}"
    )
    print(f"  Mean sparsity (top-5): {np.mean(sparsity[:5]):.4f}")
    print(f"  Mean sparsity (all): {np.mean(sparsity):.4f}")

    return {"singular_values": S_np[:20].tolist(), "sparsity": sparsity[:20]}


# =============================================================================
# PHASE 2: Projection Analysis
# =============================================================================


def phase2_projection_analysis(o_i, U, S, save_dir, n_components=10):
    """Analyze how projections onto SVD components vary with position."""
    print("\n[Phase 2] Analyzing projections onto SVD components...")

    B_size, T, D = o_i.shape
    U_np = U.cpu().numpy()
    o_np = o_i.cpu().numpy()

    # Compute projections: proj[b, t, m] = U[:, m].T @ o[b, t, :]
    o_flat = o_np.reshape(-1, D)  # [B*T, D]
    projections = o_flat @ U_np[:, :n_components]  # [B*T, n_components]
    projections = projections.reshape(B_size, T, n_components)  # [B, T, n_components]

    # Compute mean and std over batch for each position
    mean_proj = projections.mean(axis=0)  # [T, n_components]
    std_proj = projections.std(axis=0)  # [T, n_components]

    positions = np.arange(T)

    # Fit functional forms
    def linear_func(x, a, b):
        return a * x + b

    def log_func(x, a, b):
        return a * np.log(x + 1) + b

    def sqrt_func(x, a, b):
        return a * np.sqrt(x + 1) + b

    def inv_sqrt_func(x, a, b):
        return a / np.sqrt(x + 1) + b

    fit_results = []

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for m in range(n_components):
        ax = axes[m // 5, m % 5]

        y = mean_proj[:, m]
        yerr = std_proj[:, m]

        # Plot mean with error bands
        ax.plot(positions, y, "b-", linewidth=1.5, label="Mean projection")
        ax.fill_between(positions, y - yerr, y + yerr, alpha=0.2)

        # Fit different functions
        fits = {}
        try:
            popt_lin, _ = curve_fit(linear_func, positions, y)
            fits["linear"] = {
                "params": popt_lin,
                "r2": 1
                - np.sum((y - linear_func(positions, *popt_lin)) ** 2)
                / np.sum((y - y.mean()) ** 2),
            }
        except:
            fits["linear"] = {"params": [0, 0], "r2": 0}

        try:
            popt_log, _ = curve_fit(log_func, positions, y)
            fits["log"] = {
                "params": popt_log,
                "r2": 1
                - np.sum((y - log_func(positions, *popt_log)) ** 2)
                / np.sum((y - y.mean()) ** 2),
            }
        except:
            fits["log"] = {"params": [0, 0], "r2": 0}

        try:
            popt_sqrt, _ = curve_fit(sqrt_func, positions, y)
            fits["sqrt"] = {
                "params": popt_sqrt,
                "r2": 1
                - np.sum((y - sqrt_func(positions, *popt_sqrt)) ** 2)
                / np.sum((y - y.mean()) ** 2),
            }
        except:
            fits["sqrt"] = {"params": [0, 0], "r2": 0}

        try:
            popt_inv, _ = curve_fit(inv_sqrt_func, positions, y)
            fits["inv_sqrt"] = {
                "params": popt_inv,
                "r2": 1
                - np.sum((y - inv_sqrt_func(positions, *popt_inv)) ** 2)
                / np.sum((y - y.mean()) ** 2),
            }
        except:
            fits["inv_sqrt"] = {"params": [0, 0], "r2": 0}

        # Find best fit
        best_fit = max(fits.items(), key=lambda x: x[1]["r2"])
        fit_results.append(
            {
                "component": m,
                "best_fit": best_fit[0],
                "best_r2": best_fit[1]["r2"],
                "all_fits": {k: {"r2": v["r2"]} for k, v in fits.items()},
            }
        )

        # Plot best fit
        if best_fit[0] == "linear":
            ax.plot(
                positions,
                linear_func(positions, *best_fit[1]["params"]),
                "r--",
                alpha=0.7,
            )
        elif best_fit[0] == "log":
            ax.plot(
                positions, log_func(positions, *best_fit[1]["params"]), "r--", alpha=0.7
            )
        elif best_fit[0] == "sqrt":
            ax.plot(
                positions,
                sqrt_func(positions, *best_fit[1]["params"]),
                "r--",
                alpha=0.7,
            )
        elif best_fit[0] == "inv_sqrt":
            ax.plot(
                positions,
                inv_sqrt_func(positions, *best_fit[1]["params"]),
                "r--",
                alpha=0.7,
            )

        ax.set_xlabel("Position")
        ax.set_ylabel("Projection")
        ax.set_title(f"Comp {m}: {best_fit[0]} (R²={best_fit[1]['r2']:.3f})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase2_projections.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/phase2_projections.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary
    print("  Best fits per component:")
    for r in fit_results:
        print(
            f"    Component {r['component']}: {r['best_fit']} (R²={r['best_r2']:.4f})"
        )

    # Additional plot: correlation matrix of projections across positions
    fig, ax = plt.subplots(figsize=(8, 6))
    proj_flat = projections.reshape(-1, n_components)  # [B*T, n_comp]
    corr = np.corrcoef(proj_flat.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("SVD Component")
    ax.set_ylabel("SVD Component")
    ax.set_title("Correlation Between Component Projections")
    plt.colorbar(im, ax=ax)
    for i in range(n_components):
        for j in range(n_components):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/phase2_projection_correlation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    return fit_results, mean_proj, std_proj


# =============================================================================
# PHASE 3: Why Do Some High-σ Components Have Low Position R²?
# =============================================================================


def phase3_component_analysis(o_i, U, S, tokens, save_dir, n_components=10):
    """Analyze what the non-position components encode."""
    print("\n[Phase 3] Analyzing what non-position components encode...")

    B_size, T, D = o_i.shape
    U_np = U.cpu().numpy()
    S_np = S.cpu().numpy()
    o_np = o_i.cpu().numpy()
    tokens_np = tokens.cpu().numpy()

    # Compute projections
    o_flat = o_np.reshape(-1, D)
    projections = o_flat @ U_np[:, :n_components]
    projections = projections.reshape(B_size, T, n_components)

    positions = np.tile(np.arange(T), B_size)
    tokens_flat = tokens_np.flatten()

    results = []

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for m in range(n_components):
        ax = axes[m // 5, m % 5]

        proj_m = projections[:, :, m].flatten()

        # Position R² (linear probe)
        probe = Ridge(alpha=0.01)
        probe.fit(positions.reshape(-1, 1), proj_m)
        pos_r2 = probe.score(positions.reshape(-1, 1), proj_m)

        # Token variance: how much variance is explained by token identity?
        # Group by token and compute between-group vs within-group variance
        unique_tokens = np.unique(tokens_flat)
        token_means = {
            t: proj_m[tokens_flat == t].mean()
            for t in unique_tokens
            if (tokens_flat == t).sum() > 5
        }

        # Compute variance explained by token
        overall_mean = proj_m.mean()
        ss_total = np.sum((proj_m - overall_mean) ** 2)

        ss_between = 0
        for t, mean_t in token_means.items():
            n_t = (tokens_flat == t).sum()
            ss_between += n_t * (mean_t - overall_mean) ** 2

        token_r2 = ss_between / ss_total if ss_total > 0 else 0

        # Variance by position (how consistent are projections at each position?)
        pos_variance = []
        for pos in range(T):
            pos_variance.append(projections[:, pos, m].var())
        mean_pos_var = np.mean(pos_variance)

        results.append(
            {
                "component": m,
                "singular_value": float(S_np[m]),
                "position_r2": float(pos_r2),
                "token_r2": float(token_r2),
                "mean_pos_variance": float(mean_pos_var),
            }
        )

        # Scatter plot: projection vs position, colored by a sample of tokens
        sample_idx = np.random.choice(
            len(proj_m), min(5000, len(proj_m)), replace=False
        )
        scatter = ax.scatter(
            positions[sample_idx],
            proj_m[sample_idx],
            c=tokens_flat[sample_idx] % 20,
            cmap="tab20",
            alpha=0.3,
            s=1,
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("Projection")
        ax.set_title(
            f"C{m}: σ={S_np[m]:.1f}, pos_R²={pos_r2:.2f}, tok_R²={token_r2:.2f}"
        )

    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/phase3_component_variance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{save_dir}/phase3_component_variance.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_components)
    width = 0.35

    pos_r2s = [r["position_r2"] for r in results]
    tok_r2s = [r["token_r2"] for r in results]

    ax.bar(x - width / 2, pos_r2s, width, label="Position R²", color="steelblue")
    ax.bar(x + width / 2, tok_r2s, width, label="Token R²", color="coral")
    ax.set_xlabel("SVD Component")
    ax.set_ylabel("R²")
    ax.set_title("Position vs Token Variance Explained by Each Component")
    ax.legend()
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase3_pos_vs_token_r2.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/phase3_pos_vs_token_r2.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Component analysis:")
    print(f"  {'Comp':<6} {'σ':<8} {'Pos R²':<10} {'Token R²':<10}")
    print("  " + "-" * 34)
    for r in results:
        print(
            f"  {r['component']:<6} {r['singular_value']:<8.2f} {r['position_r2']:<10.4f} {r['token_r2']:<10.4f}"
        )

    return results


# =============================================================================
# PHASE 4: Compare R0 vs R2 Position Subspaces
# =============================================================================


def phase4_compare_subspaces(model_dir, device, save_dir):
    """Compare the position subspaces learned by R0 and R2."""
    print("\n[Phase 4] Comparing R0 vs R2 position subspaces...")

    # Load both models
    model_r0, _ = load_model(model_dir, "R0", device)
    model_r2, _ = load_model(model_dir, "R2", device)

    # Get SVD for both
    _, U_r0, S_r0, _ = get_B_matrix_and_svd(model_r0)
    _, U_r2, S_r2, _ = get_B_matrix_and_svd(model_r2)

    U_r0_np = U_r0.cpu().numpy()
    U_r2_np = U_r2.cpu().numpy()
    S_r0_np = S_r0.cpu().numpy()
    S_r2_np = S_r2.cpu().numpy()

    # Compute alignment between top-k subspaces
    # Principal angles between subspaces
    def subspace_alignment(U1, U2, k):
        """Compute alignment between top-k subspaces."""
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]

        # Compute U1_k.T @ U2_k and take SVD
        M = U1_k.T @ U2_k
        _, s, _ = np.linalg.svd(M)

        # s contains cos(principal angles)
        # Mean alignment = mean(s)
        return s, s.mean()

    alignments = []
    for k in [1, 2, 5, 10, 20, 50]:
        cos_angles, mean_align = subspace_alignment(U_r0_np, U_r2_np, k)
        alignments.append(
            {
                "k": k,
                "mean_alignment": float(mean_align),
                "min_alignment": float(cos_angles.min()),
                "cos_angles": cos_angles.tolist(),
            }
        )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Singular value comparison
    ax = axes[0]
    ax.semilogy(S_r0_np[:50], "o-", markersize=3, label="R0 (full train)")
    ax.semilogy(S_r2_np[:50], "s-", markersize=3, label="R2 (attn only)")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("(a) Singular Value Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Column-wise alignment
    ax = axes[1]
    col_align = [np.abs(U_r0_np[:, m] @ U_r2_np[:, m]) for m in range(50)]
    ax.bar(range(50), col_align)
    ax.set_xlabel("Component Index")
    ax.set_ylabel("|U_R0[:,m] · U_R2[:,m]|")
    ax.set_title("(b) Per-Column Alignment")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7)

    # (c) Subspace alignment vs k
    ax = axes[2]
    ks = [a["k"] for a in alignments]
    means = [a["mean_alignment"] for a in alignments]
    mins = [a["min_alignment"] for a in alignments]
    ax.plot(ks, means, "o-", label="Mean alignment")
    ax.plot(ks, mins, "s--", label="Min alignment")
    ax.set_xlabel("Subspace Dimension k")
    ax.set_ylabel("Alignment (cos principal angle)")
    ax.set_title("(c) Top-k Subspace Alignment R0↔R2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase4_r0_vs_r2.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/phase4_r0_vs_r2.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Subspace alignments (R0 vs R2):")
    for a in alignments:
        print(
            f"    k={a['k']}: mean={a['mean_alignment']:.4f}, min={a['min_alignment']:.4f}"
        )

    return alignments


# =============================================================================
# MAIN
# =============================================================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_dir = "out-2layer-mechanism"
    regime = "R0"  # Analyze the fully trained model
    n_samples = 256

    # Output directory
    save_dir = f"{model_dir}/svd_component_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model: {model_dir}/{regime}")
    model, config = load_model(model_dir, regime, device)
    print(f"  n_embd={config.n_embd}, n_head={config.n_head}")

    # Get B matrix and SVD
    print("\nComputing B = W_O @ W_V and SVD...")
    B, U, S, Vh = get_B_matrix_and_svd(model)
    print(f"  B shape: {B.shape}")
    print(f"  Top-5 singular values: {S[:5].cpu().numpy().round(2)}")

    # Load data and compute attention outputs
    print(f"\nLoading {n_samples} sequences...")
    tokens = get_data(n_samples, config.block_size, device)

    print("Computing attention outputs...")
    o_i, attn_weights = compute_attention_output(model, tokens, device)
    print(f"  o_i shape: {o_i.shape}")

    # Phase 1: Visualize SVD components
    phase1_results = phase1_visualize_svd_components(U, S, Vh, save_dir)

    # Phase 2: Projection analysis
    phase2_results, mean_proj, std_proj = phase2_projection_analysis(
        o_i, U, S, save_dir
    )

    # Phase 3: What do non-position components encode?
    phase3_results = phase3_component_analysis(o_i, U, S, tokens, save_dir)

    # Phase 4: Compare R0 vs R2
    phase4_results = phase4_compare_subspaces(model_dir, device, save_dir)

    # Save all results
    all_results = {
        "model": f"{model_dir}/{regime}",
        "phase1": phase1_results,
        "phase2": phase2_results,
        "phase3": phase3_results,
        "phase4": phase4_results,
    }

    with open(f"{save_dir}/svd_component_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SVD COMPONENT ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
