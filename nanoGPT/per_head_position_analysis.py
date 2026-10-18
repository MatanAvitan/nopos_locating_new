"""
Per-Head Position Analysis: Which Attention Heads Encode Position?

This script investigates how individual attention heads contribute to position encoding.

Key Questions:
1. Do all 12 heads contribute equally to position encoding, or do some specialize?
2. What are the attention patterns of high-contribution vs low-contribution heads?
3. Do individual heads learn different aspects of position (similar to how SVD components
   encode sqrt(i), log(i), etc.)?
4. Is there a relationship between head attention pattern and position contribution?

Usage:
    CUDA_VISIBLE_DEVICES=0 python per_head_position_analysis.py
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
from tqdm import tqdm

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


def get_per_head_outputs(model, tokens):
    """
    Get the output of each attention head separately.

    Returns:
        head_outputs: [B, T, H, D] tensor of per-head outputs (before output projection aggregation)
        attn_weights: [B, H, T, T] attention weights
    """
    B, T = tokens.shape
    config = model.config
    D = config.n_embd
    H = config.n_head
    head_dim = D // H

    # Forward through embedding and block1
    with torch.no_grad():
        emb = model.wte(tokens)  # Token embeddings (NoPE - no positional)
        emb = model.drop(emb)
        r1_out = model.block1(emb)  # Block returns just the output, no tuple

        # Get the pre-attention LayerNorm output
        x = model.block2.ln_1(r1_out)  # [B, T, D]

        # Get Q, K, V
        attn = model.block2.attn
        qkv = attn.c_attn(x)  # [B, T, 3*D]
        q, k, v = qkv.split(D, dim=2)

        # Reshape to per-head
        q = q.view(B, T, H, head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        k = k.view(B, T, H, head_dim).transpose(1, 2)
        v = v.view(B, T, H, head_dim).transpose(1, 2)

        # Compute attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        causal_mask = torch.triu(
            torch.ones(T, T, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)  # [B, H, T, T]

        # Per-head output (before W_O)
        head_out = att @ v  # [B, H, T, head_dim]

        # Get W_O per head
        W_O = attn.c_proj.weight  # [D, D]
        W_O_heads = W_O.view(D, H, head_dim)  # [D, H, head_dim]

        # Compute per-head contribution to residual
        # head_out: [B, H, T, head_dim]
        # W_O_heads: [D, H, head_dim]
        # We want: [B, T, H, D] where each head's contribution is head_out @ W_O_head

        head_contributions = torch.zeros(B, T, H, D, device=tokens.device)
        for h in range(H):
            # head_out[:, h, :, :] is [B, T, head_dim]
            # W_O_heads[:, h, :] is [D, head_dim]
            # Result: [B, T, head_dim] @ [head_dim, D] = [B, T, D]
            head_contributions[:, :, h, :] = head_out[:, h, :, :] @ W_O_heads[:, h, :].T

        # Also get the full attention output for comparison
        full_attn_out = head_contributions.sum(dim=2)  # [B, T, D]
        if attn.c_proj.bias is not None:
            full_attn_out = full_attn_out + attn.c_proj.bias

        return head_contributions, att, full_attn_out


def analyze_head_position_contribution(
    model, n_sequences=500, seq_len=128, device="cuda"
):
    """
    Analyze how each head's output contributes to position encoding.

    For each head, we:
    1. Compute the head's output contribution to the residual stream
    2. Train a linear probe to predict position from just that head's output
    3. Measure the position R² achieved by each head
    """
    config = model.config
    H = config.n_head
    D = config.n_embd

    print(f"\n{'=' * 60}")
    print("Per-Head Position Contribution Analysis")
    print(f"{'=' * 60}")

    # Generate data
    tokens = torch.randint(0, config.vocab_size, (n_sequences, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).float()

    # Get per-head outputs
    head_contributions, attn_weights, full_attn_out = get_per_head_outputs(
        model, tokens
    )

    # Move to CPU for analysis
    head_contributions = head_contributions.cpu().numpy()  # [B, T, H, D]
    attn_weights = attn_weights.cpu().numpy()  # [B, H, T, T]
    full_attn_out = full_attn_out.cpu().numpy()  # [B, T, D]
    positions_np = positions.cpu().numpy()

    results = {"per_head": [], "summary": {}}

    # Analyze each head
    print("\nPer-Head Position R²:")
    print("-" * 50)

    head_r2_scores = []
    head_attn_stats = []

    for h in range(H):
        # Get this head's contribution across all sequences and positions
        # Shape: [n_sequences * seq_len, D]
        head_h_out = head_contributions[:, :, h, :].reshape(-1, D)
        y = np.tile(positions_np, n_sequences)

        # Train linear probe
        probe = Ridge(alpha=1.0)
        probe.fit(head_h_out, y)
        pred = probe.predict(head_h_out)

        # Compute R²
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot

        head_r2_scores.append(r2)

        # Analyze attention pattern for this head
        mean_attn = attn_weights[:, h, :, :].mean(axis=0)  # [T, T]

        # Compute attention statistics
        diag_mass = np.mean([mean_attn[i, i] for i in range(seq_len)])
        bos_mass = np.mean(mean_attn[:, 0])  # How much attention to position 0 (BOS)

        # Compute "locality" - how concentrated is attention near the diagonal?
        locality_scores = []
        for i in range(seq_len):
            if i > 0:
                prefix_attn = mean_attn[i, : i + 1]
                weights = np.arange(i + 1)[::-1]  # Distance from current position
                weighted_dist = np.sum(prefix_attn * weights) / max(
                    np.sum(prefix_attn), 1e-8
                )
                locality_scores.append(weighted_dist)
        mean_locality = np.mean(locality_scores) if locality_scores else 0

        head_attn_stats.append(
            {
                "diag_mass": float(diag_mass),
                "bos_mass": float(bos_mass),
                "mean_locality": float(mean_locality),
            }
        )

        results["per_head"].append(
            {
                "head": h,
                "position_r2": float(r2),
                "diag_mass": float(diag_mass),
                "bos_mass": float(bos_mass),
                "mean_locality": float(mean_locality),
            }
        )

        print(
            f"  Head {h:2d}: R²={r2:.4f}, diag_mass={diag_mass:.3f}, bos_mass={bos_mass:.3f}"
        )

    # Also compute R² for full attention output (all heads combined)
    full_out_flat = full_attn_out.reshape(-1, D)
    y = np.tile(positions_np, n_sequences)
    probe_full = Ridge(alpha=1.0)
    probe_full.fit(full_out_flat, y)
    pred_full = probe_full.predict(full_out_flat)
    ss_res_full = np.sum((y - pred_full) ** 2)
    r2_full = 1 - ss_res_full / ss_tot

    print(f"\n  Full Attn (all heads): R²={r2_full:.4f}")

    # Summary statistics
    head_r2_arr = np.array(head_r2_scores)
    results["summary"] = {
        "full_attn_r2": float(r2_full),
        "mean_per_head_r2": float(np.mean(head_r2_arr)),
        "max_per_head_r2": float(np.max(head_r2_arr)),
        "min_per_head_r2": float(np.min(head_r2_arr)),
        "best_head": int(np.argmax(head_r2_arr)),
        "worst_head": int(np.argmin(head_r2_arr)),
    }

    print(f"\nSummary:")
    print(
        f"  Best head: {results['summary']['best_head']} (R²={results['summary']['max_per_head_r2']:.4f})"
    )
    print(
        f"  Worst head: {results['summary']['worst_head']} (R²={results['summary']['min_per_head_r2']:.4f})"
    )
    print(f"  Mean per-head R²: {results['summary']['mean_per_head_r2']:.4f}")

    return results, head_contributions, attn_weights


def analyze_head_ablation(model, n_sequences=500, seq_len=128, device="cuda"):
    """
    Ablation study: What happens when we remove each head?

    This measures the causal importance of each head for position encoding.
    """
    config = model.config
    H = config.n_head
    D = config.n_embd

    print(f"\n{'=' * 60}")
    print("Head Ablation Study")
    print(f"{'=' * 60}")

    # Generate data
    tokens = torch.randint(0, config.vocab_size, (n_sequences, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).float()
    positions_np = positions.cpu().numpy()
    y = np.tile(positions_np, n_sequences)

    # Get full per-head outputs
    head_contributions, _, full_attn_out = get_per_head_outputs(model, tokens)
    head_contributions = head_contributions.cpu().numpy()  # [B, T, H, D]

    # Baseline: all heads
    full_out = head_contributions.sum(axis=2)  # [B, T, D]
    full_out_flat = full_out.reshape(-1, D)
    probe_baseline = Ridge(alpha=1.0)
    probe_baseline.fit(full_out_flat, y)
    pred_baseline = probe_baseline.predict(full_out_flat)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res_baseline = np.sum((y - pred_baseline) ** 2)
    r2_baseline = 1 - ss_res_baseline / ss_tot

    print(f"\nBaseline (all heads): R²={r2_baseline:.4f}")
    print("\nR² after removing each head:")
    print("-" * 50)

    ablation_results = []

    for h in range(H):
        # Remove head h
        ablated = head_contributions.copy()
        ablated[:, :, h, :] = 0
        ablated_sum = ablated.sum(axis=2).reshape(-1, D)

        probe_abl = Ridge(alpha=1.0)
        probe_abl.fit(ablated_sum, y)
        pred_abl = probe_abl.predict(ablated_sum)
        ss_res_abl = np.sum((y - pred_abl) ** 2)
        r2_abl = 1 - ss_res_abl / ss_tot

        delta_r2 = r2_baseline - r2_abl  # Positive = head was important

        ablation_results.append(
            {"head": h, "r2_after_removal": float(r2_abl), "delta_r2": float(delta_r2)}
        )

        importance = (
            "HIGH" if delta_r2 > 0.05 else ("MEDIUM" if delta_r2 > 0.01 else "LOW")
        )
        print(
            f"  Remove head {h:2d}: R²={r2_abl:.4f} (Δ={delta_r2:+.4f}) [{importance}]"
        )

    return ablation_results, r2_baseline


def analyze_head_functional_forms(model, n_sequences=500, seq_len=128, device="cuda"):
    """
    For each head, what functional form best describes how its output varies with position?

    Similar to the SVD component analysis, but per-head.
    """
    config = model.config
    H = config.n_head
    D = config.n_embd

    print(f"\n{'=' * 60}")
    print("Per-Head Functional Form Analysis")
    print(f"{'=' * 60}")

    # Generate data
    tokens = torch.randint(0, config.vocab_size, (n_sequences, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).float()
    positions_np = positions.cpu().numpy()

    # Get per-head outputs
    head_contributions, _, _ = get_per_head_outputs(model, tokens)
    head_contributions = head_contributions.cpu().numpy()  # [B, T, H, D]

    # For each head, project onto its top SVD direction and fit functional forms
    functional_forms = {
        "linear": lambda x, a, b: a * x + b,
        "log": lambda x, a, b: a * np.log(x + 1) + b,
        "sqrt": lambda x, a, b: a * np.sqrt(x + 1) + b,
        "inv_sqrt": lambda x, a, b: a / np.sqrt(x + 1) + b,
    }

    results = []

    print("\nHead functional forms (projection onto top singular direction):")
    print("-" * 70)

    for h in range(H):
        # Get head h output: [B, T, D]
        head_h_out = head_contributions[:, :, h, :]

        # Average over sequences to get [T, D]
        mean_head_out = head_h_out.mean(axis=0)

        # SVD of mean_head_out
        U, S, Vh = np.linalg.svd(mean_head_out, full_matrices=False)

        # Project onto top direction: [T,]
        proj = mean_head_out @ Vh[0]

        # Fit functional forms
        fit_results = {}
        for name, func in functional_forms.items():
            try:
                popt, _ = curve_fit(func, positions_np, proj, maxfev=5000)
                pred = func(positions_np, *popt)
                ss_res = np.sum((proj - pred) ** 2)
                ss_tot = np.sum((proj - np.mean(proj)) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-8)
                fit_results[name] = float(r2)
            except Exception:
                fit_results[name] = 0.0

        best_form = max(fit_results, key=fit_results.get)
        best_r2 = fit_results[best_form]

        # Also compute overall position R² for this projection
        # (How well does the projection correlate with position across all samples?)
        all_proj = head_h_out @ Vh[0]  # [B, T]
        y = np.tile(positions_np, (n_sequences, 1))
        corr = np.corrcoef(all_proj.flatten(), y.flatten())[0, 1]

        results.append(
            {
                "head": h,
                "best_form": best_form,
                "best_r2": float(best_r2),
                "all_fits": fit_results,
                "position_correlation": float(corr),
                "top_singular_value": float(S[0]),
            }
        )

        print(
            f"  Head {h:2d}: {best_form:8s} (R²={best_r2:.3f}), pos_corr={corr:.3f}, σ_top={S[0]:.2f}"
        )

    return results


def create_visualizations(results, head_contributions, attn_weights, save_dir):
    """Create comprehensive visualizations."""

    per_head = results["per_head_contribution"]["per_head"]
    ablation = results["ablation"]
    functional = results["functional_forms"]
    H = len(per_head)

    # Figure 1: Per-head position R² bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Position R² per head
    ax = axes[0]
    head_r2 = [h["position_r2"] for h in per_head]
    colors = plt.cm.viridis(np.array(head_r2) / max(head_r2))
    bars = ax.bar(range(H), head_r2, color=colors)
    ax.set_xlabel("Head")
    ax.set_ylabel("Position R²")
    ax.set_title("(A) Per-Head Position R²")
    ax.set_xticks(range(H))

    # Panel B: Ablation importance
    ax = axes[1]
    delta_r2 = [h["delta_r2"] for h in ablation]
    colors = ["green" if d > 0.01 else "orange" if d > 0 else "red" for d in delta_r2]
    ax.bar(range(H), delta_r2, color=colors)
    ax.set_xlabel("Head")
    ax.set_ylabel("ΔR² (importance)")
    ax.set_title("(B) Ablation Importance")
    ax.set_xticks(range(H))
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Panel C: Attention statistics
    ax = axes[2]
    diag_mass = [h["diag_mass"] for h in per_head]
    bos_mass = [h["bos_mass"] for h in per_head]
    x = np.arange(H)
    width = 0.35
    ax.bar(x - width / 2, diag_mass, width, label="Diagonal mass", alpha=0.8)
    ax.bar(x + width / 2, bos_mass, width, label="BOS mass", alpha=0.8)
    ax.set_xlabel("Head")
    ax.set_ylabel("Attention mass")
    ax.set_title("(C) Attention Patterns")
    ax.set_xticks(x)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/per_head_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Functional forms per head
    fig, ax = plt.subplots(figsize=(10, 5))

    forms = ["linear", "sqrt", "log", "inv_sqrt"]
    form_colors = {
        "linear": "blue",
        "sqrt": "green",
        "log": "orange",
        "inv_sqrt": "red",
    }

    x = np.arange(H)
    width = 0.2

    for i, form in enumerate(forms):
        r2_vals = [h["all_fits"][form] for h in functional]
        ax.bar(
            x + i * width,
            r2_vals,
            width,
            label=form,
            color=form_colors[form],
            alpha=0.7,
        )

    ax.set_xlabel("Head")
    ax.set_ylabel("Fit R²")
    ax.set_title("Functional Form Fit R² per Head")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(range(H))
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/per_head_functional_forms.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Figure 3: Attention heatmaps for top and bottom heads
    head_r2 = [h["position_r2"] for h in per_head]
    sorted_heads = np.argsort(head_r2)
    top_heads = sorted_heads[-3:][::-1]  # Top 3
    bottom_heads = sorted_heads[:3]  # Bottom 3

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Mean attention across samples for each head
    mean_attn = attn_weights.mean(axis=0)  # [H, T, T]

    for i, h in enumerate(top_heads):
        ax = axes[0, i]
        im = ax.imshow(mean_attn[h, :64, :64], aspect="auto", cmap="Blues")
        ax.set_title(f"Top: Head {h} (R²={head_r2[h]:.3f})")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for i, h in enumerate(bottom_heads):
        ax = axes[1, i]
        im = ax.imshow(mean_attn[h, :64, :64], aspect="auto", cmap="Blues")
        ax.set_title(f"Bottom: Head {h} (R²={head_r2[h]:.3f})")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Attention Patterns: High vs Low Position-Encoding Heads")
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/attention_patterns_top_bottom.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Figure 4: Position R² vs attention statistics scatter
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    diag_mass = [h["diag_mass"] for h in per_head]
    ax.scatter(diag_mass, head_r2, s=100, alpha=0.7)
    for i in range(H):
        ax.annotate(str(i), (diag_mass[i], head_r2[i]), fontsize=8)
    ax.set_xlabel("Diagonal attention mass")
    ax.set_ylabel("Position R²")
    ax.set_title("(A) Position R² vs Diagonal Mass")

    ax = axes[1]
    bos_mass = [h["bos_mass"] for h in per_head]
    ax.scatter(bos_mass, head_r2, s=100, alpha=0.7)
    for i in range(H):
        ax.annotate(str(i), (bos_mass[i], head_r2[i]), fontsize=8)
    ax.set_xlabel("BOS attention mass")
    ax.set_ylabel("Position R²")
    ax.set_title("(B) Position R² vs BOS Mass")

    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/position_r2_vs_attention.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"\nVisualizations saved to {save_dir}/")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_dir = "out-2layer-mechanism"
    regime = "R0"  # Start with fully-trained model

    # Create output directory
    save_dir = f"{model_dir}/per_head_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    print(f"\nLoading {regime} model...")
    model, config = load_model(model_dir, regime, device)
    print(f"  n_head={config.n_head}, n_embd={config.n_embd}")

    # Run analyses
    results = {}

    # 1. Per-head position contribution
    contribution_results, head_contributions, attn_weights = (
        analyze_head_position_contribution(
            model, n_sequences=200, seq_len=128, device=device
        )
    )
    results["per_head_contribution"] = contribution_results

    # 2. Ablation study
    ablation_results, baseline_r2 = analyze_head_ablation(
        model, n_sequences=200, seq_len=128, device=device
    )
    results["ablation"] = ablation_results
    results["baseline_r2"] = baseline_r2

    # 3. Functional form analysis
    functional_results = analyze_head_functional_forms(
        model, n_sequences=200, seq_len=128, device=device
    )
    results["functional_forms"] = functional_results

    # Save results
    results_path = f"{save_dir}/per_head_analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Create visualizations
    create_visualizations(results, head_contributions, attn_weights, save_dir)

    # Also run for R2 model for comparison
    print(f"\n{'=' * 60}")
    print("Running analysis for R2 model...")
    print(f"{'=' * 60}")

    model_r2, config_r2 = load_model(model_dir, "R2", device)

    contribution_r2, head_contrib_r2, attn_r2 = analyze_head_position_contribution(
        model_r2, n_sequences=200, seq_len=128, device=device
    )
    ablation_r2, baseline_r2_r2 = analyze_head_ablation(
        model_r2, n_sequences=200, seq_len=128, device=device
    )
    functional_r2 = analyze_head_functional_forms(
        model_r2, n_sequences=200, seq_len=128, device=device
    )

    results_r2 = {
        "per_head_contribution": contribution_r2,
        "ablation": ablation_r2,
        "baseline_r2": baseline_r2_r2,
        "functional_forms": functional_r2,
    }

    # Save R2 results
    save_dir_r2 = f"{model_dir}/per_head_analysis_R2"
    os.makedirs(save_dir_r2, exist_ok=True)

    with open(f"{save_dir_r2}/per_head_analysis_results.json", "w") as f:
        json.dump(results_r2, f, indent=2)

    create_visualizations(results_r2, head_contrib_r2, attn_r2, save_dir_r2)

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("R0 vs R2 Comparison")
    print(f"{'=' * 60}")

    r0_head_r2 = [
        h["position_r2"] for h in results["per_head_contribution"]["per_head"]
    ]
    r2_head_r2 = [
        h["position_r2"] for h in results_r2["per_head_contribution"]["per_head"]
    ]

    print("\nPer-head Position R²:")
    print(f"{'Head':<6} {'R0':<10} {'R2':<10} {'Diff':<10}")
    print("-" * 40)
    for h in range(config.n_head):
        diff = r0_head_r2[h] - r2_head_r2[h]
        print(f"{h:<6} {r0_head_r2[h]:<10.4f} {r2_head_r2[h]:<10.4f} {diff:+.4f}")

    print(f"\nR0 mean: {np.mean(r0_head_r2):.4f}, R2 mean: {np.mean(r2_head_r2):.4f}")


if __name__ == "__main__":
    main()
