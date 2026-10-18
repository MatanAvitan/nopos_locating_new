"""
Comprehensive Analysis for 2-Layer Mechanism Experiments

This script implements Experiments 2-6 from the mechanism dissection spec:
- Experiment 2: Coefficient Template-Fit Analysis (H1 vs H2 vs H3)
- Experiment 3: Invariance & Anti-Shortcut Tests (rules out H3)
- Experiment 4: Attention Pattern Characterization (H1 vs H2)
- Experiment 5: Head-Level Causal Mediation
- Experiment 6: LN Sensitivity & Magnitude Diagnostics (H4)

All experiments use:
- WandB project: nope-2layer-mechanism
- Comprehensive logging of hypotheses, metrics, and diagnostic info
- Loaded model from Experiment 1 training

Usage:
    python analyze_2layer_mechanism.py --checkpoint out-2layer-mechanism/R1/best_ckpt.pt --wandb
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import minimize_scalar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model_2layer_mechanism import (
    TwoLayerMechanismModel,
    TwoLayerMechanismConfig,
)


# =============================================================================
# Hypothesis Reference
# =============================================================================

HYPOTHESES = {
    "H1": "Iterated-Averaging / Harmonic-Profile",
    "H2": "Learned Prefix Kernel",
    "H3": "Token-ID Leakage / Shortcut",
    "H4": "Magnitude / Variance-Decay",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_owt_data(data_dir: str = "data/openwebtext"):
    """Load OpenWebText data."""
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Get a batch of sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device)


def compute_token_frequencies(data: np.ndarray, vocab_size: int = 50304) -> np.ndarray:
    """Compute token frequency distribution."""
    # Sample a portion for efficiency
    sample_size = min(10_000_000, len(data))
    sample = data[:sample_size]

    freqs = np.zeros(vocab_size)
    unique, counts = np.unique(sample, return_counts=True)
    freqs[unique] = counts
    freqs = freqs / freqs.sum()
    return freqs


# =============================================================================
# Experiment 2: Coefficient Template-Fit Analysis
# =============================================================================


def compute_harmonic_template(i: int) -> np.ndarray:
    """
    Compute harmonic tail template for position i.
    h_{i,k} ∝ Σ_{t=k..i} 1/t = H_i - H_{k-1}
    """
    template = np.zeros(i)
    H = np.cumsum(1.0 / np.arange(1, i + 1))  # Harmonic numbers
    for k in range(1, i + 1):
        # h_{i,k} = H_i - H_{k-1}
        H_k_minus_1 = 0 if k == 1 else H[k - 2]
        template[k - 1] = H[i - 1] - H_k_minus_1
    # Normalize
    template = template / (np.linalg.norm(template) + 1e-10)
    return template


def compute_uniform_template(i: int) -> np.ndarray:
    """Uniform template: all coefficients equal."""
    template = np.ones(i) / np.sqrt(i)
    return template


def compute_power_law_template(i: int, alpha: float) -> np.ndarray:
    """Power-law template: p_{i,k}(α) ∝ (i-k+1)^α."""
    template = np.array([(i - k + 1) ** alpha for k in range(1, i + 1)])
    template = template / (np.linalg.norm(template) + 1e-10)
    return template


def compute_exponential_template(i: int, beta: float) -> np.ndarray:
    """Exponential template: q_{i,k}(β) ∝ exp(-β(i-k))."""
    template = np.array([np.exp(-beta * (i - k)) for k in range(1, i + 1)])
    template = template / (np.linalg.norm(template) + 1e-10)
    return template


def ridge_regression_coefficients(
    z: torch.Tensor, E: torch.Tensor, lambda_reg: float = 1e-3
) -> torch.Tensor:
    """
    Compute ridge regression coefficients: ĉ = (E^T E + λI)^{-1} E^T z

    Args:
        z: representation vector [d]
        E: prefix embeddings [d, i] where columns are e_1, ..., e_i
        lambda_reg: regularization parameter

    Returns:
        coefficients [i]
    """
    d, i = E.shape

    # (E^T E + λI)^{-1} E^T z
    EtE = E.T @ E  # [i, i]
    reg = lambda_reg * torch.trace(EtE) / i * torch.eye(i, device=E.device)

    try:
        coeffs = torch.linalg.solve(EtE + reg, E.T @ z)
    except:
        # Fallback to pseudo-inverse
        coeffs = torch.linalg.lstsq(E.T, z).solution[:i]

    return coeffs


def fit_power_law_alpha(coeffs_abs: np.ndarray, i: int) -> Tuple[float, float]:
    """Fit optimal alpha for power-law template."""

    def neg_corr(alpha):
        template = compute_power_law_template(i, alpha)
        corr, _ = stats.spearmanr(coeffs_abs[: len(template)], template)
        return -corr if not np.isnan(corr) else 0

    result = minimize_scalar(neg_corr, bounds=(-2, 2), method="bounded")
    return result.x, -result.fun


def fit_exponential_beta(coeffs_abs: np.ndarray, i: int) -> Tuple[float, float]:
    """Fit optimal beta for exponential template."""

    def neg_corr(beta):
        template = compute_exponential_template(i, beta)
        corr, _ = stats.spearmanr(coeffs_abs[: len(template)], template)
        return -corr if not np.isnan(corr) else 0

    result = minimize_scalar(neg_corr, bounds=(0.01, 5), method="bounded")
    return result.x, -result.fun


@torch.no_grad()
def run_experiment2_coefficient_analysis(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    device: str,
    n_samples: int = 100,
    batch_size: int = 32,
    positions_to_analyze: List[int] = None,
) -> Dict:
    """
    Experiment 2: Coefficient Template-Fit Analysis

    Project representations onto prefix-token span and compare to templates.
    """
    model.eval()
    config = model.config

    if positions_to_analyze is None:
        # Analyze positions 4, 8, 16, 32, 64, 96, 128 (or max)
        positions_to_analyze = [
            p for p in [4, 8, 16, 32, 64, 96, 128] if p < config.block_size
        ]

    results = {
        "tap_points": ["block2_attn", "block2_post_attn", "block2_out"],
        "positions": positions_to_analyze,
        "template_correlations": {},
        "best_fit_params": {},
    }

    # Collect coefficients for each tap point
    for tap_name in results["tap_points"]:
        results["template_correlations"][tap_name] = {
            "uniform": {pos: [] for pos in positions_to_analyze},
            "harmonic": {pos: [] for pos in positions_to_analyze},
            "power_law": {pos: [] for pos in positions_to_analyze},
            "exponential": {pos: [] for pos in positions_to_analyze},
        }
        results["best_fit_params"][tap_name] = {
            "alpha": {pos: [] for pos in positions_to_analyze},
            "beta": {pos: [] for pos in positions_to_analyze},
        }

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        # Get batch
        x = get_batch(data, batch_size, config.block_size, device)

        # Forward pass with taps
        with torch.no_grad():
            _, _ = model(x, capture_taps=True)

        taps = model.get_all_taps()
        embeddings = taps["embeddings"]  # [B, T, d]

        for tap_name in results["tap_points"]:
            z_all = taps[tap_name]  # [B, T, d]

            for pos in positions_to_analyze:
                # Get representation at position pos
                z = z_all[:, pos, :]  # [B, d]

                # Get prefix embeddings [e_1, ..., e_{pos}]
                E = embeddings[:, : pos + 1, :].transpose(1, 2)  # [B, d, pos+1]

                for b in range(z.shape[0]):
                    z_b = z[b]  # [d]
                    E_b = E[b]  # [d, pos+1]

                    # Compute coefficients
                    coeffs = ridge_regression_coefficients(z_b, E_b)
                    coeffs_abs = torch.abs(coeffs).cpu().numpy()

                    i = pos + 1  # Number of prefix tokens

                    # Compare to templates
                    if i >= 2:
                        uniform_template = compute_uniform_template(i)
                        harmonic_template = compute_harmonic_template(i)

                        # Spearman correlation with absolute coefficients
                        uniform_corr, _ = stats.spearmanr(coeffs_abs, uniform_template)
                        harmonic_corr, _ = stats.spearmanr(
                            coeffs_abs, harmonic_template
                        )

                        # Fit parametric templates
                        alpha, power_corr = fit_power_law_alpha(coeffs_abs, i)
                        beta, exp_corr = fit_exponential_beta(coeffs_abs, i)

                        results["template_correlations"][tap_name]["uniform"][
                            pos
                        ].append(uniform_corr if not np.isnan(uniform_corr) else 0)
                        results["template_correlations"][tap_name]["harmonic"][
                            pos
                        ].append(harmonic_corr if not np.isnan(harmonic_corr) else 0)
                        results["template_correlations"][tap_name]["power_law"][
                            pos
                        ].append(power_corr if not np.isnan(power_corr) else 0)
                        results["template_correlations"][tap_name]["exponential"][
                            pos
                        ].append(exp_corr if not np.isnan(exp_corr) else 0)
                        results["best_fit_params"][tap_name]["alpha"][pos].append(alpha)
                        results["best_fit_params"][tap_name]["beta"][pos].append(beta)

    # Aggregate results
    summary = {
        "template_correlations_mean": {},
        "template_correlations_std": {},
        "best_template_per_position": {},
        "hypothesis_support": {},
    }

    for tap_name in results["tap_points"]:
        summary["template_correlations_mean"][tap_name] = {}
        summary["template_correlations_std"][tap_name] = {}
        summary["best_template_per_position"][tap_name] = {}

        for pos in positions_to_analyze:
            means = {}
            for template in ["uniform", "harmonic", "power_law", "exponential"]:
                vals = results["template_correlations"][tap_name][template][pos]
                means[template] = np.mean(vals) if vals else 0

            summary["template_correlations_mean"][tap_name][pos] = means
            summary["best_template_per_position"][tap_name][pos] = max(
                means, key=means.get
            )

    # Hypothesis interpretation
    # H1: harmonic should win at block2 tap points
    # H2: power_law or exponential should win
    block2_out_harmonic = np.mean(
        [
            summary["template_correlations_mean"]["block2_out"][pos]["harmonic"]
            for pos in positions_to_analyze
        ]
    )
    block2_out_power = np.mean(
        [
            summary["template_correlations_mean"]["block2_out"][pos]["power_law"]
            for pos in positions_to_analyze
        ]
    )

    if block2_out_harmonic > block2_out_power + 0.1:
        summary["hypothesis_support"]["H1"] = True
        summary["hypothesis_support"]["H2"] = False
    elif block2_out_power > block2_out_harmonic + 0.1:
        summary["hypothesis_support"]["H1"] = False
        summary["hypothesis_support"]["H2"] = True
    else:
        summary["hypothesis_support"]["H1"] = "inconclusive"
        summary["hypothesis_support"]["H2"] = "inconclusive"

    results["summary"] = summary
    return results


# =============================================================================
# Experiment 3: Invariance & Anti-Shortcut Tests
# =============================================================================


@torch.no_grad()
def run_experiment3_invariance_tests(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    token_freqs: np.ndarray,
    device: str,
    n_samples: int = 500,
    batch_size: int = 32,
) -> Dict:
    """
    Experiment 3: Test invariance to rule out H3 (shortcut).

    Perturbations:
    1. Token shuffle within sequence
    2. Prefix permutation
    3. Replace current token
    4. Frequency-matched swaps
    """
    model.eval()
    config = model.config

    results = {
        "baseline": {"mae": [], "r2": []},
        "token_shuffle": {"mae": [], "r2": []},
        "prefix_permutation": {"mae": [], "r2": []},
        "replace_current": {"mae": [], "r2": []},
        "freq_matched_swap": {"mae": [], "r2": []},
    }

    # Create frequency-matched token groups
    n_groups = 100
    freq_bins = np.percentile(
        token_freqs[token_freqs > 0], np.linspace(0, 100, n_groups + 1)
    )
    token_groups = [[] for _ in range(n_groups)]
    for tok in range(len(token_freqs)):
        if token_freqs[tok] > 0:
            for g in range(n_groups):
                if token_freqs[tok] <= freq_bins[g + 1]:
                    token_groups[g].append(tok)
                    break

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        x = get_batch(data, batch_size, config.block_size, device)
        targets = (
            torch.arange(config.block_size)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(device)
        )

        # 1. Baseline
        output, loss = model(x, targets, capture_taps=False)
        preds = output.squeeze(-1)
        baseline_mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        ss_tot = ((targets.float() - targets.float().mean()) ** 2).sum()
        baseline_r2 = (1 - ss_res / ss_tot).item()
        results["baseline"]["mae"].append(baseline_mae)
        results["baseline"]["r2"].append(baseline_r2)

        # 2. Token shuffle (permute token IDs, keep positions)
        x_shuffled = x.clone()
        for b in range(batch_size):
            perm = torch.randperm(config.block_size)
            x_shuffled[b] = x[b, perm]
        output, _ = model(x_shuffled, targets, capture_taps=False)
        preds = output.squeeze(-1)
        shuffle_mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        shuffle_r2 = (1 - ss_res / ss_tot).item()
        results["token_shuffle"]["mae"].append(shuffle_mae)
        results["token_shuffle"]["r2"].append(shuffle_r2)

        # 3. Prefix permutation (for each position, permute its prefix)
        x_prefix_perm = x.clone()
        for b in range(batch_size):
            # Permute prefix of last position
            half_pos = config.block_size // 2
            perm = torch.randperm(half_pos)
            x_prefix_perm[b, :half_pos] = x[b, perm]
        output, _ = model(x_prefix_perm, targets, capture_taps=False)
        preds = output.squeeze(-1)
        prefix_mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        prefix_r2 = (1 - ss_res / ss_tot).item()
        results["prefix_permutation"]["mae"].append(prefix_mae)
        results["prefix_permutation"]["r2"].append(prefix_r2)

        # 4. Replace current token with random token
        x_replaced = x.clone()
        random_tokens = torch.randint(0, config.vocab_size, x.shape, device=device)
        # Replace every other position
        for pos in range(0, config.block_size, 2):
            x_replaced[:, pos] = random_tokens[:, pos]
        output, _ = model(x_replaced, targets, capture_taps=False)
        preds = output.squeeze(-1)
        replace_mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        replace_r2 = (1 - ss_res / ss_tot).item()
        results["replace_current"]["mae"].append(replace_mae)
        results["replace_current"]["r2"].append(replace_r2)

        # 5. Frequency-matched swap
        x_freq_swap = x.clone()
        for b in range(batch_size):
            for pos in range(config.block_size):
                tok = x[b, pos].item()
                # Find frequency group
                tok_freq = token_freqs[tok]
                for g in range(n_groups):
                    if tok_freq <= freq_bins[g + 1] and len(token_groups[g]) > 1:
                        # Pick random token from same group
                        alternatives = [t for t in token_groups[g] if t != tok]
                        if alternatives:
                            x_freq_swap[b, pos] = np.random.choice(alternatives)
                        break
        output, _ = model(x_freq_swap, targets, capture_taps=False)
        preds = output.squeeze(-1)
        freq_mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        freq_r2 = (1 - ss_res / ss_tot).item()
        results["freq_matched_swap"]["mae"].append(freq_mae)
        results["freq_matched_swap"]["r2"].append(freq_r2)

    # Aggregate results
    summary = {}
    for perturbation in results:
        summary[perturbation] = {
            "mae_mean": np.mean(results[perturbation]["mae"]),
            "mae_std": np.std(results[perturbation]["mae"]),
            "r2_mean": np.mean(results[perturbation]["r2"]),
            "r2_std": np.std(results[perturbation]["r2"]),
        }

    # Hypothesis interpretation
    baseline_r2 = summary["baseline"]["r2_mean"]

    # H3 check: if replacing current token causes large drop, H3 supported
    replace_drop = baseline_r2 - summary["replace_current"]["r2_mean"]
    freq_swap_drop = baseline_r2 - summary["freq_matched_swap"]["r2_mean"]

    summary["hypothesis_support"] = {}
    if replace_drop > 0.3 or freq_swap_drop > 0.3:
        summary["hypothesis_support"]["H3"] = True
        summary["hypothesis_support"]["reason"] = (
            f"Large performance drop when replacing tokens (replace: {replace_drop:.3f}, freq_swap: {freq_swap_drop:.3f})"
        )
    else:
        summary["hypothesis_support"]["H3"] = False
        summary["hypothesis_support"]["reason"] = (
            f"Model robust to token replacement (replace: {replace_drop:.3f}, freq_swap: {freq_swap_drop:.3f})"
        )

    return {"raw": results, "summary": summary}


# =============================================================================
# Experiment 4: Attention Pattern Characterization
# =============================================================================


@torch.no_grad()
def run_experiment4_attention_analysis(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    device: str,
    n_samples: int = 100,
    batch_size: int = 32,
) -> Dict:
    """
    Experiment 4: Characterize attention patterns in Block 2.

    Metrics:
    - Uniformity (KL to uniform)
    - Entropy
    - Kernel shape (attention vs relative distance)
    """
    model.eval()
    config = model.config

    results = {
        "block1": {"kl_divergence": [], "entropy": [], "kernel_shape": []},
        "block2": {"kl_divergence": [], "entropy": [], "kernel_shape": []},
    }

    n_batches = (n_samples + batch_size - 1) // batch_size

    all_attn1 = []
    all_attn2 = []

    for batch_idx in range(n_batches):
        x = get_batch(data, batch_size, config.block_size, device)

        # Forward pass
        _, _ = model(x, capture_taps=True)
        attn1, attn2 = model.get_attention_weights()

        all_attn1.append(attn1.cpu())
        all_attn2.append(attn2.cpu())

    all_attn1 = torch.cat(all_attn1, dim=0)  # [N, n_head, T, T]
    all_attn2 = torch.cat(all_attn2, dim=0)

    n_head = all_attn1.shape[1]
    T = all_attn1.shape[2]

    # Compute metrics per head
    for block_name, attn in [("block1", all_attn1), ("block2", all_attn2)]:
        head_metrics = []

        for h in range(n_head):
            head_attn = attn[:, h, :, :]  # [N, T, T]

            # KL divergence to uniform (per position, averaged)
            kl_per_pos = []
            entropy_per_pos = []

            for pos in range(1, T):
                # Uniform distribution over prefix [0..pos]
                uniform = torch.ones(pos + 1) / (pos + 1)

                # Average attention at this position
                attn_at_pos = head_attn[:, pos, : pos + 1].mean(dim=0)  # [pos+1]
                attn_at_pos = attn_at_pos.clamp(min=1e-10)
                attn_at_pos = attn_at_pos / attn_at_pos.sum()

                # KL divergence
                kl = (attn_at_pos * (attn_at_pos.log() - uniform.log())).sum().item()
                kl_per_pos.append(kl)

                # Entropy
                entropy = -(attn_at_pos * attn_at_pos.log()).sum().item()
                max_entropy = np.log(pos + 1)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                entropy_per_pos.append(normalized_entropy)

            results[block_name]["kl_divergence"].append(
                {
                    "head": h,
                    "mean_kl": np.mean(kl_per_pos),
                    "per_position": kl_per_pos,
                }
            )
            results[block_name]["entropy"].append(
                {
                    "head": h,
                    "mean_normalized_entropy": np.mean(entropy_per_pos),
                    "per_position": entropy_per_pos,
                }
            )

        # Kernel shape: average attention as function of relative distance
        kernel_shape = np.zeros(T)
        counts = np.zeros(T)

        for pos in range(T):
            for rel_dist in range(pos + 1):
                # rel_dist = pos - j where j is the attended position
                j = pos - rel_dist
                kernel_shape[rel_dist] += attn[:, :, pos, j].mean().item()
                counts[rel_dist] += 1

        kernel_shape = kernel_shape / (counts + 1e-10)
        results[block_name]["kernel_shape"] = kernel_shape.tolist()

    # Summary statistics
    summary = {
        "block1": {
            "mean_kl": np.mean(
                [h["mean_kl"] for h in results["block1"]["kl_divergence"]]
            ),
            "mean_entropy": np.mean(
                [h["mean_normalized_entropy"] for h in results["block1"]["entropy"]]
            ),
            "near_uniform_heads": sum(
                1 for h in results["block1"]["kl_divergence"] if h["mean_kl"] < 0.1
            ),
        },
        "block2": {
            "mean_kl": np.mean(
                [h["mean_kl"] for h in results["block2"]["kl_divergence"]]
            ),
            "mean_entropy": np.mean(
                [h["mean_normalized_entropy"] for h in results["block2"]["entropy"]]
            ),
            "near_uniform_heads": sum(
                1 for h in results["block2"]["kl_divergence"] if h["mean_kl"] < 0.1
            ),
        },
    }

    # Hypothesis interpretation
    # H1: Block2 attention should be near-uniform (high entropy, low KL)
    # H2: Block2 attention should show structured kernel
    if summary["block2"]["mean_kl"] < 0.2 and summary["block2"]["mean_entropy"] > 0.8:
        summary["hypothesis_support"] = {
            "H1": True,
            "H2": False,
            "reason": "Block2 attention is near-uniform",
        }
    elif summary["block2"]["mean_kl"] > 0.5:
        summary["hypothesis_support"] = {
            "H1": False,
            "H2": True,
            "reason": "Block2 attention has structured pattern",
        }
    else:
        summary["hypothesis_support"] = {
            "H1": "partial",
            "H2": "partial",
            "reason": "Mixed attention patterns",
        }

    results["summary"] = summary
    return results


# =============================================================================
# Experiment 5: Head-Level Causal Mediation
# =============================================================================


@torch.no_grad()
def run_experiment5_causal_mediation(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    device: str,
    n_samples: int = 200,
    batch_size: int = 32,
) -> Dict:
    """
    Experiment 5: Identify which heads are necessary for position encoding.

    For each head in block2, ablate (zero out) and measure performance drop.
    """
    model.eval()
    config = model.config

    results = {
        "baseline": {"mae": None, "r2": None},
        "head_ablations": [],
    }

    # Get baseline performance
    baseline_maes = []
    baseline_r2s = []

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        x = get_batch(data, batch_size, config.block_size, device)
        targets = (
            torch.arange(config.block_size)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(device)
        )

        output, _ = model(x, targets, capture_taps=False)
        preds = output.squeeze(-1)

        mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        ss_tot = ((targets.float() - targets.float().mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()

        baseline_maes.append(mae)
        baseline_r2s.append(r2)

    results["baseline"]["mae"] = np.mean(baseline_maes)
    results["baseline"]["r2"] = np.mean(baseline_r2s)

    # Ablate each head
    n_head = config.n_head
    head_dim = config.n_embd // n_head

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Store original projection weights
    original_c_proj = raw_model.block2.attn.c_proj.weight.clone()

    for h in range(n_head):
        # Zero out this head's contribution
        with torch.no_grad():
            # c_proj weight is [n_embd, n_embd]
            # Each head contributes head_dim columns
            start_col = h * head_dim
            end_col = (h + 1) * head_dim

            ablated_weight = original_c_proj.clone()
            ablated_weight[:, start_col:end_col] = 0
            raw_model.block2.attn.c_proj.weight.copy_(ablated_weight)

        # Evaluate
        ablated_maes = []
        ablated_r2s = []

        for batch_idx in range(n_batches):
            x = get_batch(data, batch_size, config.block_size, device)
            targets = (
                torch.arange(config.block_size)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .to(device)
            )

            output, _ = model(x, targets, capture_taps=False)
            preds = output.squeeze(-1)

            mae = (preds - targets.float()).abs().mean().item()
            ss_res = ((targets.float() - preds) ** 2).sum()
            ss_tot = ((targets.float() - targets.float().mean()) ** 2).sum()
            r2 = (1 - ss_res / ss_tot).item()

            ablated_maes.append(mae)
            ablated_r2s.append(r2)

        delta_mae = np.mean(ablated_maes) - results["baseline"]["mae"]
        delta_r2 = results["baseline"]["r2"] - np.mean(ablated_r2s)

        results["head_ablations"].append(
            {
                "head": h,
                "ablated_mae": np.mean(ablated_maes),
                "ablated_r2": np.mean(ablated_r2s),
                "delta_mae": delta_mae,
                "delta_r2": delta_r2,
            }
        )

        # Restore original weights
        raw_model.block2.attn.c_proj.weight.copy_(original_c_proj)

    # Find position heads (large causal effect)
    position_heads = [
        h["head"]
        for h in results["head_ablations"]
        if h["delta_r2"] > 0.1  # R² drop > 0.1 is significant
    ]

    results["summary"] = {
        "baseline_mae": results["baseline"]["mae"],
        "baseline_r2": results["baseline"]["r2"],
        "position_heads": position_heads,
        "n_important_heads": len(position_heads),
        "max_delta_r2": max(h["delta_r2"] for h in results["head_ablations"]),
        "mean_delta_r2": np.mean([h["delta_r2"] for h in results["head_ablations"]]),
    }

    return results


# =============================================================================
# Experiment 6: LN Sensitivity & Magnitude Diagnostics
# =============================================================================


@torch.no_grad()
def run_experiment6_ln_sensitivity(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    device: str,
    n_samples: int = 500,
    batch_size: int = 32,
) -> Dict:
    """
    Experiment 6: Test magnitude/variance hypotheses (H4).

    Tests:
    1. Norm-only probe: Can norms alone predict position?
    2. LN ablation: Remove LN4 affine parameters
    3. Diagnostic scalars: ||r||, ||x||, ||a|| correlations with position
    """
    model.eval()
    config = model.config

    results = {
        "norm_probes": {},
        "ln_ablation": {},
        "diagnostic_scalars": {},
    }

    # Collect norms at each tap point
    tap_norms = {
        "block1_out": [],
        "block2_ln1": [],
        "block2_attn": [],
        "block2_post_attn": [],
        "block2_out": [],
    }
    positions = []

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        x = get_batch(data, batch_size, config.block_size, device)

        # Forward pass with taps
        _, _ = model(x, capture_taps=True)
        taps = model.get_all_taps()

        for tap_name in tap_norms:
            norms = torch.norm(taps[tap_name], dim=-1)  # [B, T]
            tap_norms[tap_name].append(norms.cpu())

        pos = torch.arange(config.block_size).unsqueeze(0).expand(batch_size, -1)
        positions.append(pos)

    # Concatenate
    for tap_name in tap_norms:
        tap_norms[tap_name] = torch.cat(tap_norms[tap_name], dim=0).numpy()  # [N, T]
    positions = torch.cat(positions, dim=0).numpy()

    # 1. Norm-only linear probe
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    for tap_name in tap_norms:
        norms = tap_norms[tap_name].flatten().reshape(-1, 1)
        pos_flat = positions.flatten()

        # Train simple ridge regression
        probe = Ridge(alpha=1.0)
        probe.fit(norms, pos_flat)
        preds = probe.predict(norms)

        r2 = r2_score(pos_flat, preds)
        corr = np.corrcoef(norms.flatten(), pos_flat)[0, 1]

        results["norm_probes"][tap_name] = {
            "r2": r2,
            "correlation": corr,
        }

    # 2. LN ablation: set ln_2 affine to identity
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Store original LN parameters
    original_ln2_weight = raw_model.block2.ln_2.weight.clone()
    original_ln2_bias = (
        raw_model.block2.ln_2.bias.clone()
        if raw_model.block2.ln_2.bias is not None
        else None
    )

    # Set to identity (weight=1, bias=0)
    with torch.no_grad():
        raw_model.block2.ln_2.weight.fill_(1.0)
        if raw_model.block2.ln_2.bias is not None:
            raw_model.block2.ln_2.bias.fill_(0.0)

    # Evaluate
    ablated_maes = []
    ablated_r2s = []

    for batch_idx in range(min(n_batches, 10)):
        x = get_batch(data, batch_size, config.block_size, device)
        targets = (
            torch.arange(config.block_size)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(device)
        )

        output, _ = model(x, targets, capture_taps=False)
        preds = output.squeeze(-1)

        mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        ss_tot = ((targets.float() - targets.float().mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()

        ablated_maes.append(mae)
        ablated_r2s.append(r2)

    results["ln_ablation"] = {
        "ablated_mae": np.mean(ablated_maes),
        "ablated_r2": np.mean(ablated_r2s),
    }

    # Restore original parameters
    with torch.no_grad():
        raw_model.block2.ln_2.weight.copy_(original_ln2_weight)
        if original_ln2_bias is not None:
            raw_model.block2.ln_2.bias.copy_(original_ln2_bias)

    # Get baseline for comparison
    baseline_maes = []
    baseline_r2s = []

    for batch_idx in range(min(n_batches, 10)):
        x = get_batch(data, batch_size, config.block_size, device)
        targets = (
            torch.arange(config.block_size)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(device)
        )

        output, _ = model(x, targets, capture_taps=False)
        preds = output.squeeze(-1)

        mae = (preds - targets.float()).abs().mean().item()
        ss_res = ((targets.float() - preds) ** 2).sum()
        ss_tot = ((targets.float() - targets.float().mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()

        baseline_maes.append(mae)
        baseline_r2s.append(r2)

    results["ln_ablation"]["baseline_mae"] = np.mean(baseline_maes)
    results["ln_ablation"]["baseline_r2"] = np.mean(baseline_r2s)
    results["ln_ablation"]["delta_r2"] = (
        results["ln_ablation"]["baseline_r2"] - results["ln_ablation"]["ablated_r2"]
    )

    # 3. Diagnostic scalars summary
    for tap_name in tap_norms:
        norms = tap_norms[tap_name]
        # Mean norm per position
        mean_norms_per_pos = norms.mean(axis=0)
        corr_with_pos = np.corrcoef(mean_norms_per_pos, np.arange(config.block_size))[
            0, 1
        ]

        results["diagnostic_scalars"][tap_name] = {
            "mean_norm_per_position": mean_norms_per_pos.tolist(),
            "norm_position_correlation": corr_with_pos,
        }

    # Summary and hypothesis interpretation
    summary = {
        "norm_probe_r2_max": max(v["r2"] for v in results["norm_probes"].values()),
        "ln_ablation_r2_drop": results["ln_ablation"]["delta_r2"],
        "block2_out_norm_corr": results["diagnostic_scalars"]["block2_out"][
            "norm_position_correlation"
        ],
    }

    # H4: magnitude/variance mechanism
    if summary["norm_probe_r2_max"] > 0.5 and abs(summary["ln_ablation_r2_drop"]) > 0.3:
        summary["hypothesis_support"] = {
            "H4": True,
            "reason": f"Norms strongly predictive (R²={summary['norm_probe_r2_max']:.3f}) and LN ablation causes large drop ({summary['ln_ablation_r2_drop']:.3f})",
        }
    else:
        summary["hypothesis_support"] = {
            "H4": False,
            "reason": f"Norms weakly predictive (R²={summary['norm_probe_r2_max']:.3f}) or LN ablation has small effect ({summary['ln_ablation_r2_drop']:.3f})",
        }

    results["summary"] = summary
    return results


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def load_model_from_checkpoint(
    checkpoint_path: str, device: str
) -> TwoLayerMechanismModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint.get("config", {})
    model_config = TwoLayerMechanismConfig(
        block_size=config_dict.get("block_size", 128),
        vocab_size=config_dict.get("vocab_size", 50304),
        n_embd=config_dict.get("n_embd", 768),
        n_head=config_dict.get("n_head", 12),
        dropout=0.0,
        norm_type=config_dict.get("norm_type", "layernorm"),
        use_regression=True,
    )

    model = TwoLayerMechanismModel(model_config)

    # Load state dict
    state_dict = checkpoint["model"]
    # Handle compiled model prefix
    unwrapped = {}
    for key, value in state_dict.items():
        new_key = key[10:] if key.startswith("_orig_mod.") else key
        unwrapped[new_key] = value

    model.load_state_dict(unwrapped, strict=False)
    model.to(device)
    model.eval()

    return model


def run_all_analyses(
    checkpoint_path: str,
    device: str = "cuda",
    wandb_log: bool = True,
    out_dir: str = "analysis_results",
) -> Dict:
    """Run all experiments 2-6."""

    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load data
    print("Loading data...")
    train_data, val_data = load_owt_data()
    token_freqs = compute_token_frequencies(train_data)

    # Initialize WandB
    if wandb_log:
        import wandb

        regime = Path(checkpoint_path).parent.name
        wandb.init(
            project="nope-2layer-mechanism",
            name=f"analysis_{regime}",
            config={
                "checkpoint": checkpoint_path,
                "regime": regime,
            },
            tags=["analysis", regime, "exp2-6"],
        )

        wandb.run.summary["hypotheses"] = HYPOTHESES

    all_results = {}

    # Experiment 2: Coefficient Template-Fit
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Coefficient Template-Fit Analysis")
    print("=" * 60)
    results_exp2 = run_experiment2_coefficient_analysis(
        model, val_data, device, n_samples=100
    )
    all_results["experiment2"] = results_exp2

    if wandb_log:
        import wandb

        wandb.log(
            {
                "exp2/hypothesis_support_H1": results_exp2["summary"][
                    "hypothesis_support"
                ].get("H1", "N/A"),
                "exp2/hypothesis_support_H2": results_exp2["summary"][
                    "hypothesis_support"
                ].get("H2", "N/A"),
            }
        )

    print(
        f"  H1 support: {results_exp2['summary']['hypothesis_support'].get('H1', 'N/A')}"
    )
    print(
        f"  H2 support: {results_exp2['summary']['hypothesis_support'].get('H2', 'N/A')}"
    )

    # Experiment 3: Invariance Tests
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Invariance & Anti-Shortcut Tests")
    print("=" * 60)
    results_exp3 = run_experiment3_invariance_tests(
        model, val_data, token_freqs, device, n_samples=200
    )
    all_results["experiment3"] = results_exp3

    if wandb_log:
        wandb.log(
            {
                "exp3/baseline_r2": results_exp3["summary"]["baseline"]["r2_mean"],
                "exp3/token_shuffle_r2": results_exp3["summary"]["token_shuffle"][
                    "r2_mean"
                ],
                "exp3/replace_current_r2": results_exp3["summary"]["replace_current"][
                    "r2_mean"
                ],
                "exp3/hypothesis_support_H3": results_exp3["summary"][
                    "hypothesis_support"
                ]["H3"],
            }
        )

    print(f"  Baseline R²: {results_exp3['summary']['baseline']['r2_mean']:.4f}")
    print(
        f"  Token shuffle R²: {results_exp3['summary']['token_shuffle']['r2_mean']:.4f}"
    )
    print(
        f"  Replace current R²: {results_exp3['summary']['replace_current']['r2_mean']:.4f}"
    )
    print(f"  H3 support: {results_exp3['summary']['hypothesis_support']['H3']}")

    # Experiment 4: Attention Pattern Analysis
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Attention Pattern Characterization")
    print("=" * 60)
    results_exp4 = run_experiment4_attention_analysis(
        model, val_data, device, n_samples=100
    )
    all_results["experiment4"] = results_exp4

    if wandb_log:
        wandb.log(
            {
                "exp4/block2_mean_kl": results_exp4["summary"]["block2"]["mean_kl"],
                "exp4/block2_mean_entropy": results_exp4["summary"]["block2"][
                    "mean_entropy"
                ],
                "exp4/block2_near_uniform_heads": results_exp4["summary"]["block2"][
                    "near_uniform_heads"
                ],
            }
        )

    print(f"  Block2 mean KL: {results_exp4['summary']['block2']['mean_kl']:.4f}")
    print(
        f"  Block2 mean entropy: {results_exp4['summary']['block2']['mean_entropy']:.4f}"
    )
    print(
        f"  Block2 near-uniform heads: {results_exp4['summary']['block2']['near_uniform_heads']}"
    )

    # Experiment 5: Causal Mediation
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Head-Level Causal Mediation")
    print("=" * 60)
    results_exp5 = run_experiment5_causal_mediation(
        model, val_data, device, n_samples=100
    )
    all_results["experiment5"] = results_exp5

    if wandb_log:
        wandb.log(
            {
                "exp5/n_position_heads": results_exp5["summary"]["n_important_heads"],
                "exp5/max_delta_r2": results_exp5["summary"]["max_delta_r2"],
                "exp5/position_heads": results_exp5["summary"]["position_heads"],
            }
        )

    print(f"  Position heads: {results_exp5['summary']['position_heads']}")
    print(f"  Max ΔR²: {results_exp5['summary']['max_delta_r2']:.4f}")

    # Experiment 6: LN Sensitivity
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: LN Sensitivity & Magnitude Diagnostics")
    print("=" * 60)
    results_exp6 = run_experiment6_ln_sensitivity(
        model, val_data, device, n_samples=200
    )
    all_results["experiment6"] = results_exp6

    if wandb_log:
        wandb.log(
            {
                "exp6/norm_probe_r2_max": results_exp6["summary"]["norm_probe_r2_max"],
                "exp6/ln_ablation_r2_drop": results_exp6["summary"][
                    "ln_ablation_r2_drop"
                ],
                "exp6/hypothesis_support_H4": results_exp6["summary"][
                    "hypothesis_support"
                ]["H4"],
            }
        )

    print(f"  Max norm probe R²: {results_exp6['summary']['norm_probe_r2_max']:.4f}")
    print(
        f"  LN ablation R² drop: {results_exp6['summary']['ln_ablation_r2_drop']:.4f}"
    )
    print(f"  H4 support: {results_exp6['summary']['hypothesis_support']['H4']}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL HYPOTHESIS SUMMARY")
    print("=" * 80)

    summary = {
        "H1_support": results_exp2["summary"]["hypothesis_support"].get("H1", "N/A"),
        "H2_support": results_exp2["summary"]["hypothesis_support"].get("H2", "N/A"),
        "H3_support": results_exp3["summary"]["hypothesis_support"]["H3"],
        "H4_support": results_exp6["summary"]["hypothesis_support"]["H4"],
    }

    for h, support in summary.items():
        print(f"  {h}: {support}")

    if wandb_log:
        wandb.run.summary["final_hypothesis_summary"] = summary
        wandb.finish()

    # Save results
    results_path = os.path.join(out_dir, "analysis_results.json")

    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="2-Layer Mechanism Analysis")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--out_dir", type=str, default="analysis_results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_all_analyses(
        checkpoint_path=args.checkpoint,
        device=args.device,
        wandb_log=args.wandb,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
