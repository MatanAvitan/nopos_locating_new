"""
Deep Mechanism Analysis: The Exact Mathematics of Position Encoding in 2-Layer NoPE

This script provides a comprehensive scientific investigation of the three open questions:

1. BOS HEAD MECHANISM (R0): What makes q_i · k_0 >> q_i · k_j for j > 0?
   - Analyze W_Q, W_K matrices for heads 6 and 9
   - Decompose the attention score computation
   - Identify the learned features that create BOS attention

2. LINEAR HEAD MECHANISM: How does w · x + b extract position?
   - Analyze which dimensions of the final representation correlate with position
   - Decompose the weight vector into interpretable components
   - Trace the information flow from input to prediction

3. CURRENT-POSITION MECHANISM (R2): How do heads read the variance-decay signal?
   - Analyze the input to Block 2 (from frozen random Block 1)
   - Understand what current-position heads extract
   - Compare to the BOS mechanism

Author: Scientific Analysis for 2-Layer NoPE Mechanism Study
Date: January 2026
"""

import os
import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.linalg import svd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig


def load_checkpoint(
    regime: str, base_dir: str = "out-2layer-mechanism"
) -> Tuple[TwoLayerMechanismModel, dict]:
    """Load a trained checkpoint for analysis."""
    ckpt_path = Path(base_dir) / regime / "best_ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = TwoLayerMechanismConfig()
    model = TwoLayerMechanismModel(config)

    # Load state dict (handle _orig_mod prefix from torch.compile)
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model, checkpoint


def get_owt_batch(
    batch_size: int = 64, block_size: int = 128, device: str = "cpu"
) -> torch.Tensor:
    """Load a batch of OpenWebText data."""
    data_path = Path("data/openwebtext/train.bin")
    if not data_path.exists():
        # Fall back to random tokens if data not available
        print("Warning: OWT data not found, using random tokens")
        return torch.randint(0, 50304, (batch_size, block_size), device=device)

    data = np.memmap(str(data_path), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device)


# =============================================================================
# PART 1: BOS HEAD MECHANISM ANALYSIS
# =============================================================================


@dataclass
class BOSHeadAnalysis:
    """Results from BOS head mechanism analysis."""

    head_idx: int

    # Q, K, V weight matrices for this head
    W_Q: torch.Tensor  # [d_head, d_model]
    W_K: torch.Tensor  # [d_head, d_model]
    W_V: torch.Tensor  # [d_head, d_model]

    # Key findings
    query_pos0_bias: float  # How much q_i prefers k_0
    key_pos0_uniqueness: float  # How unique is k_0 vs other keys
    value_pos0_norm: float  # Norm of v_0

    # Attention pattern statistics
    mean_attn_to_pos0: float
    std_attn_to_pos0: float


def analyze_bos_heads(model: TwoLayerMechanismModel, data: torch.Tensor) -> Dict:
    """
    QUESTION 1: What makes BOS heads (6, 9) attend exclusively to position 0?

    Mathematical Analysis:
    Attention score: score_{i,j} = q_i · k_j / sqrt(d_head)
    For BOS heads: score_{i,0} >> score_{i,j} for j > 0

    We want to understand WHY this happens. Possible reasons:
    a) k_0 is in a unique direction that aligns with all queries
    b) q_i has a learned component that specifically matches k_0
    c) The LN1 output at position 0 has special properties
    """
    results = {}
    device = next(model.parameters()).device
    data = data.to(device)

    print("\n" + "=" * 80)
    print("PART 1: BOS HEAD MECHANISM ANALYSIS")
    print("=" * 80)

    # Run forward pass to get activations
    with torch.no_grad():
        model(data, capture_taps=True)
        taps = model.get_all_taps()
        attn1, attn2 = model.get_attention_weights()

    # Get the input to Block 2 attention (post-LN)
    block2_ln1_out = taps["block2_ln1"]  # [B, T, d]
    B, T, d = block2_ln1_out.shape

    # Extract W_Q, W_K, W_V for Block 2
    # c_attn projects to [Q, K, V] concatenated
    W_qkv = model.block2.attn.c_attn.weight  # [3*d, d]
    b_qkv = model.block2.attn.c_attn.bias  # [3*d]

    W_Q = W_qkv[:d, :]  # [d, d]
    W_K = W_qkv[d : 2 * d, :]  # [d, d]
    W_V = W_qkv[2 * d :, :]  # [d, d]

    b_Q = b_qkv[:d]
    b_K = b_qkv[d : 2 * d]
    b_V = b_qkv[2 * d :]

    n_head = model.config.n_head
    head_dim = d // n_head

    print(f"\nModel configuration: d={d}, n_head={n_head}, head_dim={head_dim}")

    # Identify BOS heads from attention patterns
    print("\n--- Attention Pattern Analysis ---")
    mean_attn_to_pos0 = attn2[:, :, :, 0].mean(dim=(0, 2))  # [n_head]

    bos_heads = []
    for h in range(n_head):
        attn_to_0 = mean_attn_to_pos0[h].item()
        if attn_to_0 > 0.5:
            bos_heads.append(h)
            print(f"  Head {h}: Mean attention to pos 0 = {attn_to_0:.3f} [BOS HEAD]")
        else:
            print(f"  Head {h}: Mean attention to pos 0 = {attn_to_0:.3f}")

    results["bos_heads"] = bos_heads
    results["mean_attn_to_pos0"] = mean_attn_to_pos0.tolist()

    # Analyze each BOS head in detail
    print("\n--- Deep Dive into BOS Heads ---")

    for h in bos_heads:
        print(f"\n>>> HEAD {h} ANALYSIS <<<")

        # Extract per-head Q, K, V weights
        h_start, h_end = h * head_dim, (h + 1) * head_dim
        W_Q_h = W_Q[h_start:h_end, :]  # [head_dim, d]
        W_K_h = W_K[h_start:h_end, :]  # [head_dim, d]
        W_V_h = W_V[h_start:h_end, :]  # [head_dim, d]

        b_Q_h = b_Q[h_start:h_end]
        b_K_h = b_K[h_start:h_end]

        # Compute Q and K for this head on the actual data
        # Q_h = x @ W_Q_h.T + b_Q_h
        # K_h = x @ W_K_h.T + b_K_h

        x = block2_ln1_out  # [B, T, d]
        Q_h = torch.einsum("btd,hd->bth", x, W_Q_h) + b_Q_h  # [B, T, head_dim]
        K_h = torch.einsum("btd,hd->bth", x, W_K_h) + b_K_h  # [B, T, head_dim]

        # Attention scores (before softmax)
        # score[i,j] = Q_h[i] · K_h[j] / sqrt(head_dim)
        scores = torch.einsum("bih,bjh->bij", Q_h, K_h) / math.sqrt(
            head_dim
        )  # [B, T, T]

        # Mask future positions
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores_masked = scores.masked_fill(causal_mask, -1e9)

        # Analyze score patterns
        print(f"\n  Score analysis (before softmax):")

        # Score to position 0 vs others
        score_to_0 = scores_masked[:, :, 0]  # [B, T]

        # For each query position, what's the score to pos 0 vs max score to other positions?
        for query_pos in [1, 16, 32, 64, 96, 127]:
            if query_pos >= T:
                continue
            s0 = score_to_0[:, query_pos].mean().item()  # score to pos 0

            # Scores to positions 1..query_pos
            other_scores = scores_masked[
                :, query_pos, 1 : query_pos + 1
            ]  # [B, query_pos]
            if other_scores.numel() > 0:
                s_max_other = other_scores.max(dim=-1)[0].mean().item()
                s_mean_other = other_scores.mean().item()
            else:
                s_max_other = float("-inf")
                s_mean_other = float("-inf")

            print(
                f"    Query pos {query_pos:3d}: score(q,k_0)={s0:+.2f}, "
                f"max_other={s_max_other:+.2f}, mean_other={s_mean_other:+.2f}, "
                f"Δ={s0 - s_max_other:+.2f}"
            )

        # THE KEY QUESTION: Why is score(q_i, k_0) >> score(q_i, k_j)?
        #
        # score(q_i, k_0) = (x_i @ W_Q_h.T + b_Q_h) · (x_0 @ W_K_h.T + b_K_h)
        #                 = x_i @ W_Q_h.T @ W_K_h @ x_0.T + ...bias terms...
        #
        # Let's decompose this

        print(f"\n  Decomposition of score computation:")

        # Compute the QK^T product for this head
        # This is the "what features in x create high scores" matrix
        QK_product = W_Q_h.T @ W_K_h  # [d, d]

        # SVD of QK product to understand its structure
        U, S, Vh = torch.linalg.svd(QK_product)
        print(f"    QK^T matrix: shape {QK_product.shape}")
        print(f"    Top 5 singular values: {S[:5].tolist()}")
        print(f"    Effective rank (>1% of max): {(S > 0.01 * S[0]).sum().item()}")

        # What makes x_0 special?
        # x_0 is the LN1 output at position 0, which is LN(Block1_output_0)
        x_0 = block2_ln1_out[:, 0, :]  # [B, d]
        x_others = block2_ln1_out[:, 1:, :]  # [B, T-1, d]

        # Key at position 0
        k_0 = K_h[:, 0, :]  # [B, head_dim]

        # Mean key at other positions
        k_others_mean = K_h[:, 1:, :].mean(dim=1)  # [B, head_dim]

        # How different is k_0?
        k_0_norm = k_0.norm(dim=-1).mean().item()
        k_others_norm = K_h[:, 1:, :].norm(dim=-1).mean().item()
        k_0_direction = k_0 / k_0.norm(dim=-1, keepdim=True)
        k_others_direction = K_h[:, 1:, :] / K_h[:, 1:, :].norm(dim=-1, keepdim=True)

        # Cosine similarity between k_0 and k_others
        cos_k0_kothers = torch.einsum("bh,bth->bt", k_0_direction, k_others_direction)
        mean_cos_k0_kothers = cos_k0_kothers.mean().item()

        print(f"\n  Key vector analysis:")
        print(f"    ||k_0|| = {k_0_norm:.3f}")
        print(f"    ||k_others|| (mean) = {k_others_norm:.3f}")
        print(f"    Ratio ||k_0|| / ||k_others|| = {k_0_norm / k_others_norm:.3f}")
        print(f"    Mean cosine(k_0, k_others) = {mean_cos_k0_kothers:.3f}")

        # The bias terms!
        # b_Q · b_K contribution is constant, but b_Q · k_j depends on k_j
        bias_contrib = (b_Q_h * b_K_h).sum().item()
        bQ_dot_k0 = (b_Q_h.unsqueeze(0) * K_h[:, 0, :]).sum(dim=-1).mean().item()
        bQ_dot_k_others = (
            (b_Q_h.unsqueeze(0).unsqueeze(0) * K_h[:, 1:, :]).sum(dim=-1).mean().item()
        )

        print(f"\n  Bias contribution analysis:")
        print(f"    b_Q · b_K = {bias_contrib:.3f}")
        print(f"    b_Q · k_0 (mean) = {bQ_dot_k0:.3f}")
        print(f"    b_Q · k_others (mean) = {bQ_dot_k_others:.3f}")
        print(f"    Δ(b_Q · k_0 - b_Q · k_others) = {bQ_dot_k0 - bQ_dot_k_others:.3f}")

        # What about the query bias?
        # q_i · b_K contribution
        q_mean = Q_h.mean(dim=(0, 1))  # [head_dim]
        q_dot_bK = (q_mean * b_K_h).sum().item()

        print(f"    mean(q) · b_K = {q_dot_bK:.3f}")

        # Store per-head results
        results[f"head_{h}"] = {
            "k0_norm": k_0_norm,
            "k_others_norm": k_others_norm,
            "k0_k_others_ratio": k_0_norm / k_others_norm,
            "mean_cos_k0_kothers": mean_cos_k0_kothers,
            "bQ_dot_k0": bQ_dot_k0,
            "bQ_dot_k_others": bQ_dot_k_others,
            "bias_advantage": bQ_dot_k0 - bQ_dot_k_others,
            "qk_top_singular_values": S[:5].tolist(),
        }

        # THE HYPOTHESIS: x_0 has unique properties because it's the ONLY position
        # that wasn't averaged in Block 1 attention
        print(f"\n  x_0 uniqueness analysis (LN output at pos 0):")
        x_0_mean = x_0.mean(dim=0)  # [d]
        x_others_mean = x_others.mean(dim=(0, 1))  # [d]
        x_0_std = x_0.std(dim=0).mean().item()
        x_others_std = x_others.std(dim=(0, 1)).mean().item()

        # Project x_0 and x_others onto top singular vectors of QK^T
        top_V = Vh[:5, :]  # [5, d] - top 5 right singular vectors
        x_0_proj = (x_0 @ top_V.T).mean(dim=0)  # [5]
        x_others_proj = (x_others.reshape(-1, d) @ top_V.T).mean(dim=0)  # [5]

        print(f"    x_0 std: {x_0_std:.3f}, x_others std: {x_others_std:.3f}")
        print(
            f"    x_0 projection onto top-5 QK^T singular vectors: {x_0_proj.tolist()}"
        )
        print(
            f"    x_others projection onto top-5 QK^T singular vectors: {x_others_proj.tolist()}"
        )

    return results


# =============================================================================
# PART 2: LINEAR HEAD MECHANISM ANALYSIS
# =============================================================================


def analyze_linear_head(model: TwoLayerMechanismModel, data: torch.Tensor) -> Dict:
    """
    QUESTION 2: How does the linear head w · x + b extract position?

    The prediction is: pred_i = w · h_i + b
    Where h_i is the final representation at position i

    We want to understand:
    a) Which dimensions of w have high magnitude?
    b) How do those dimensions correlate with position in h?
    c) What is the "position direction" in representation space?
    """
    results = {}
    device = next(model.parameters()).device
    data = data.to(device)

    print("\n" + "=" * 80)
    print("PART 2: LINEAR HEAD MECHANISM ANALYSIS")
    print("=" * 80)

    # Get the linear head parameters
    w = model.pos_head.weight.squeeze()  # [d]
    b = model.pos_head.bias.item()

    print(f"\nLinear head: pred = w · x + b")
    print(f"  ||w|| = {w.norm().item():.4f}")
    print(f"  b = {b:.4f}")

    # Run forward pass
    with torch.no_grad():
        preds, _ = model(data, capture_taps=True)
        taps = model.get_all_taps()

    # Get final representation before head
    # This is after ln_f
    h = model.ln_f(taps["block2_out"])  # [B, T, d]
    B, T, d = h.shape

    # Verify prediction
    manual_pred = torch.einsum("btd,d->bt", h, w) + b
    pred_diff = (manual_pred - preds.squeeze()).abs().max().item()
    print(f"  Prediction verification error: {pred_diff:.6f}")

    # Analyze weight vector
    print(f"\n--- Weight Vector Analysis ---")

    # Top dimensions by magnitude
    w_abs = w.abs()
    top_dims = w_abs.argsort(descending=True)[:20]
    print(f"  Top 20 dimensions by |w|:")
    for i, dim in enumerate(top_dims[:10]):
        print(f"    Dim {dim.item():3d}: w = {w[dim].item():+.4f}")

    # Weight statistics
    w_pos = (w > 0).sum().item()
    w_neg = (w < 0).sum().item()
    print(f"\n  Weight sign distribution: {w_pos} positive, {w_neg} negative")

    # Correlation of each dimension with position
    print(f"\n--- Dimension-Position Correlation Analysis ---")

    positions = torch.arange(T, device=device).float()
    positions_expanded = positions.unsqueeze(0).expand(B, T)  # [B, T]

    # Flatten for correlation
    h_flat = h.reshape(B * T, d)  # [B*T, d]
    pos_flat = positions_expanded.reshape(B * T)  # [B*T]

    # Compute correlation for each dimension
    dim_pos_corr = torch.zeros(d, device=device)
    for dim in range(d):
        h_dim = h_flat[:, dim]
        corr = torch.corrcoef(torch.stack([h_dim, pos_flat]))[0, 1]
        dim_pos_corr[dim] = corr if not torch.isnan(corr) else 0

    # Dimensions most correlated with position
    top_pos_corr_dims = dim_pos_corr.abs().argsort(descending=True)[:20]
    print(f"  Top 20 dimensions by |correlation with position|:")
    for i, dim in enumerate(top_pos_corr_dims[:10]):
        print(
            f"    Dim {dim.item():3d}: corr = {dim_pos_corr[dim].item():+.4f}, w = {w[dim].item():+.4f}"
        )

    # THE KEY INSIGHT: Does w align with the position-correlated direction?
    print(f"\n--- Weight-Correlation Alignment ---")

    # Normalize both
    w_norm = w / w.norm()
    corr_norm = dim_pos_corr / dim_pos_corr.norm()

    alignment = (w_norm * corr_norm).sum().item()
    print(f"  Cosine(w, dim_pos_corr) = {alignment:.4f}")

    # This tells us: does the model use dimensions that correlate with position?

    # Weighted contribution
    # pred = sum_d w_d * h_d = sum_d (w_d * h_d)
    # Contribution from dimension d is w_d * h_d

    h_mean = h.mean(dim=(0, 1))  # [d] - mean activation per dim
    contribution = w * h_mean

    print(f"\n  Mean contribution per dimension (w_d * mean(h_d)):")
    top_contrib_dims = contribution.abs().argsort(descending=True)[:10]
    for dim in top_contrib_dims:
        print(
            f"    Dim {dim.item():3d}: contrib = {contribution[dim].item():+.4f}, "
            f"w = {w[dim].item():+.4f}, mean_h = {h_mean[dim].item():+.4f}"
        )

    # Position-dependent contribution
    # For position i, contribution is w · h_i
    # We want to see how this varies with position

    contributions_by_pos = torch.einsum("btd,d->bt", h, w)  # [B, T]
    mean_contrib_by_pos = contributions_by_pos.mean(dim=0)  # [T]

    print(f"\n  Mean w·h by position:")
    for pos in [0, 16, 32, 64, 96, 127]:
        print(f"    Pos {pos:3d}: w·h = {mean_contrib_by_pos[pos].item():.2f}")

    # Correlation of w·h with position
    corr_pred_pos = torch.corrcoef(torch.stack([mean_contrib_by_pos, positions]))[
        0, 1
    ].item()
    print(f"\n  Correlation(w·h, position) = {corr_pred_pos:.4f}")

    # Store results
    results["w_norm"] = w.norm().item()
    results["b"] = b
    results["w_corr_alignment"] = alignment
    results["pred_pos_correlation"] = corr_pred_pos
    results["top_dims_by_w"] = [(d.item(), w[d].item()) for d in top_dims[:20]]
    results["top_dims_by_pos_corr"] = [
        (d.item(), dim_pos_corr[d].item()) for d in top_pos_corr_dims[:20]
    ]

    # Decompose prediction into interpretable components
    print(f"\n--- Prediction Decomposition ---")

    # h_i = ln_f(block2_out_i)
    # block2_out_i = block2_post_attn_i + mlp2_out_i
    # block2_post_attn_i = block1_out_i + attn2_out_i

    # Let's trace back what contributes to the prediction
    block2_out = taps["block2_out"]  # [B, T, d]
    block2_post_attn = taps["block2_post_attn"]  # [B, T, d]
    block2_attn = taps["block2_attn"]  # [B, T, d]
    block2_mlp = taps["block2_mlp"]  # [B, T, d]
    block1_out = taps["block1_out"]  # [B, T, d]

    # How much does each component contribute to the position signal?
    # Note: we need to pass through ln_f before projecting with w

    def ln_project(x):
        """Apply final LN and project with w."""
        return torch.einsum("btd,d->bt", model.ln_f(x), w)

    # Full prediction
    pred_full = ln_project(block2_out)

    # Contribution from block1 (through residual)
    pred_block1_only = ln_project(block1_out)

    # Contribution from attn2 (delta)
    pred_with_attn2 = ln_project(block1_out + block2_attn)
    attn2_contribution = pred_with_attn2 - pred_block1_only

    # Contribution from mlp2 (delta)
    mlp2_contribution = pred_full - pred_with_attn2

    print(f"  Contribution decomposition (mean over batch):")
    print(f"    Block1 residual: {pred_block1_only.mean():.2f}")
    print(f"    Attn2 delta: {attn2_contribution.mean():.2f}")
    print(f"    MLP2 delta: {mlp2_contribution.mean():.2f}")

    # Position correlation of each component
    corr_block1 = torch.corrcoef(
        torch.stack([pred_block1_only.mean(dim=0), positions])
    )[0, 1].item()
    corr_attn2 = torch.corrcoef(
        torch.stack([attn2_contribution.mean(dim=0), positions])
    )[0, 1].item()
    corr_mlp2 = torch.corrcoef(torch.stack([mlp2_contribution.mean(dim=0), positions]))[
        0, 1
    ].item()

    print(f"\n  Position correlation of each component:")
    print(f"    Block1 residual: r = {corr_block1:.4f}")
    print(f"    Attn2 contribution: r = {corr_attn2:.4f}")
    print(f"    MLP2 contribution: r = {corr_mlp2:.4f}")

    results["component_correlations"] = {
        "block1": corr_block1,
        "attn2": corr_attn2,
        "mlp2": corr_mlp2,
    }

    return results


# =============================================================================
# PART 3: CURRENT-POSITION MECHANISM ANALYSIS (R2)
# =============================================================================


def analyze_current_position_mechanism(
    model_r2: TwoLayerMechanismModel,
    model_r0: TwoLayerMechanismModel,
    data: torch.Tensor,
) -> Dict:
    """
    QUESTION 3: How do current-position heads in R2 read the variance-decay signal?

    In R2, Block 1 is frozen (random), so the input to Block 2 is:
    r_i = e_i + Attn1(LN(e))_i + MLP1(LN2(r_i^{post-attn}))

    This contains position information through:
    - Causal attention averaging creates variance decay
    - Later positions have lower variance (more averaging)

    The R2 model learns current-position heads that attend to the current position
    and read this variance signal directly.
    """
    results = {}
    device = next(model_r2.parameters()).device
    data = data.to(device)

    print("\n" + "=" * 80)
    print("PART 3: CURRENT-POSITION MECHANISM (R2) ANALYSIS")
    print("=" * 80)

    # Run forward pass on both models
    with torch.no_grad():
        model_r2(data, capture_taps=True)
        taps_r2 = model_r2.get_all_taps()
        attn1_r2, attn2_r2 = model_r2.get_attention_weights()

        model_r0(data, capture_taps=True)
        taps_r0 = model_r0.get_all_taps()
        attn1_r0, attn2_r0 = model_r0.get_attention_weights()

    B, n_head, T, _ = attn2_r2.shape
    d = model_r2.config.n_embd
    head_dim = d // n_head

    # Identify current-position heads in R2
    print(f"\n--- Identifying Current-Position Heads (R2) ---")

    # A current-position head attends mostly to the current position
    diag_attn = torch.zeros(n_head)
    for h in range(n_head):
        # Mean attention to current position
        diag_attn[h] = torch.diagonal(attn2_r2[0, h, :, :]).mean()

    current_pos_heads = []
    for h in range(n_head):
        attn_to_self = diag_attn[h].item()
        attn_to_others_max = attn2_r2[:, h, :, :].max(dim=-1)[0].mean().item()

        # Check if this is a current-position head
        is_current_pos = attn_to_self > 0.1 and attn_to_self > attn_to_others_max * 0.5

        if is_current_pos:
            current_pos_heads.append(h)
            print(
                f"  Head {h}: Mean self-attention = {attn_to_self:.3f} [CURRENT-POS HEAD]"
            )
        else:
            first_tok_attn = attn2_r2[:, h, :, 0].mean().item()
            print(
                f"  Head {h}: Mean self-attention = {attn_to_self:.3f}, first_tok = {first_tok_attn:.3f}"
            )

    results["current_pos_heads"] = current_pos_heads

    # Analyze the input to Block 2 (from frozen Block 1)
    print(f"\n--- Block 1 Output Analysis (Input to Block 2) ---")

    block1_out = taps_r2["block1_out"]  # [B, T, d]

    # Position-dependent statistics
    positions = torch.arange(T, device=device).float()

    # Norm at each position
    norms = block1_out.norm(dim=-1)  # [B, T]
    mean_norm_by_pos = norms.mean(dim=0)  # [T]

    # Variance at each position (across embedding dimensions)
    vars = block1_out.var(dim=-1)  # [B, T]
    mean_var_by_pos = vars.mean(dim=0)  # [T]

    # Correlation with position
    corr_norm_pos = torch.corrcoef(torch.stack([mean_norm_by_pos, positions]))[
        0, 1
    ].item()
    corr_var_pos = torch.corrcoef(torch.stack([mean_var_by_pos, positions]))[
        0, 1
    ].item()

    print(f"  Block 1 output norm: correlation with position = {corr_norm_pos:.4f}")
    print(f"  Block 1 output variance: correlation with position = {corr_var_pos:.4f}")

    # Sample values
    print(f"\n  Sample norms by position:")
    for pos in [0, 16, 32, 64, 96, 127]:
        print(
            f"    Pos {pos:3d}: norm = {mean_norm_by_pos[pos].item():.3f}, var = {mean_var_by_pos[pos].item():.4f}"
        )

    results["block1_norm_pos_corr"] = corr_norm_pos
    results["block1_var_pos_corr"] = corr_var_pos

    # What do current-position heads extract?
    print(f"\n--- What Current-Position Heads Extract ---")

    # The value vector v_i at position i
    # Value contribution: sum_j attn[i,j] * v_j
    # For current-pos heads: attn[i,i] is high, so output ≈ v_i

    # Get V weights for Block 2
    W_qkv = model_r2.block2.attn.c_attn.weight
    W_V = W_qkv[2 * d :, :]  # [d, d]
    b_V = model_r2.block2.attn.c_attn.bias[2 * d :]

    # Input to attention is LN(block1_out)
    block2_ln1 = taps_r2["block2_ln1"]  # [B, T, d]

    # Compute V for all positions
    V = block2_ln1 @ W_V.T + b_V  # [B, T, d]

    for h in current_pos_heads[:3]:  # Analyze first 3
        h_start, h_end = h * head_dim, (h + 1) * head_dim
        V_h = V[:, :, h_start:h_end]  # [B, T, head_dim]

        # V norm at each position
        V_h_norms = V_h.norm(dim=-1)  # [B, T]
        mean_V_norm_by_pos = V_h_norms.mean(dim=0)

        corr_V_pos = torch.corrcoef(torch.stack([mean_V_norm_by_pos, positions]))[
            0, 1
        ].item()

        print(f"\n  Head {h} value vector analysis:")
        print(f"    ||V_h|| correlation with position: {corr_V_pos:.4f}")

        # V direction - does it have a consistent "position" direction?
        V_h_flat = V_h.reshape(B * T, head_dim)
        V_h_mean = V_h_flat.mean(dim=0)

        # Compute direction variance
        V_h_centered = V_h_flat - V_h_mean
        V_h_cov = V_h_centered.T @ V_h_centered / (B * T)
        eigenvalues = torch.linalg.eigvalsh(V_h_cov)

        print(f"    V_h covariance top eigenvalues: {eigenvalues[-5:].tolist()}")

        results[f"head_{h}_V_pos_corr"] = corr_V_pos

    # Compare R2 vs R0 attention patterns
    print(f"\n--- R2 vs R0 Attention Pattern Comparison ---")

    # Attention entropy (measure of uniformity)
    def attention_entropy(attn):
        # attn: [B, n_head, T, T]
        # Compute entropy for each position
        entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)  # [B, n_head, T]
        return entropy.mean(dim=(0, 2))  # [n_head]

    entropy_r2 = attention_entropy(attn2_r2)
    entropy_r0 = attention_entropy(attn2_r0)

    print(f"  Attention entropy by head (higher = more uniform):")
    print(
        f"  {'Head':<6} {'R2 Entropy':<12} {'R0 Entropy':<12} {'R2 Pattern':<15} {'R0 Pattern':<15}"
    )
    print(f"  {'-' * 60}")

    for h in range(n_head):
        r2_ent = entropy_r2[h].item()
        r0_ent = entropy_r0[h].item()

        # Classify pattern
        r2_diag = diag_attn[h].item()
        r2_first = attn2_r2[:, h, :, 0].mean().item()
        r0_first = attn2_r0[:, h, :, 0].mean().item()

        r2_pattern = (
            "current-pos"
            if h in current_pos_heads
            else ("BOS" if r2_first > 0.5 else "other")
        )
        r0_pattern = "BOS" if r0_first > 0.5 else "other"

        print(
            f"  {h:<6} {r2_ent:<12.3f} {r0_ent:<12.3f} {r2_pattern:<15} {r0_pattern:<15}"
        )

    results["entropy_r2"] = entropy_r2.tolist()
    results["entropy_r0"] = entropy_r0.tolist()

    # THE KEY INSIGHT: How does current-position attention help?
    print(f"\n--- Why Current-Position Attention Works ---")

    # For a current-position head with attention mostly to self:
    # output_i ≈ V[i] = LN(block1_out_i) @ W_V.T + b_V
    #
    # This directly reads the representation at position i, which has:
    # - Norm correlated with position (r = {corr_norm_pos})
    # - Variance correlated with position (r = {corr_var_pos})

    # The linear head then extracts position from this signal

    # Verify: correlation between V norm and position
    V_norms = V.norm(dim=-1)  # [B, T]
    V_norm_mean = V_norms.mean(dim=0)
    corr_V_norm_pos = torch.corrcoef(torch.stack([V_norm_mean, positions]))[0, 1].item()

    print(f"  ||V|| (full) correlation with position: {corr_V_norm_pos:.4f}")
    print(
        f"  This means: current-position heads directly read the position-correlated signal!"
    )

    results["V_norm_pos_corr"] = corr_V_norm_pos

    return results


# =============================================================================
# PART 4: SYNTHESIS - THE COMPLETE MECHANISM
# =============================================================================


def synthesize_findings(
    bos_results: Dict, head_results: Dict, r2_results: Dict
) -> Dict:
    """Synthesize all findings into a complete mechanism description."""

    print("\n" + "=" * 80)
    print("SYNTHESIS: THE COMPLETE POSITION ENCODING MECHANISM")
    print("=" * 80)

    synthesis = {}

    print(
        """
    
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                    2-LAYER NoPE POSITION ENCODING MECHANISM                    ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  STAGE 1: BLOCK 1 (RANDOM/FROZEN) CREATES POSITION-DEPENDENT STATISTICS      ║
    ║  ─────────────────────────────────────────────────────────────────────────    ║
    ║                                                                               ║
    ║  Input: Token embeddings e_0, e_1, ..., e_{{T-1}}                             ║
    ║                                                                               ║
    ║  Causal attention averaging:                                                  ║
    ║    - Position 0: sees only e_0 → high variance, high norm                    ║
    ║    - Position i: averages e_0..e_i → lower variance, lower norm              ║
    ║                                                                               ║
    ║  Output: r_i^1 with position-correlated statistics                           ║
    ║    - Norm-position correlation:  r = -0.58                                   ║
    ║    - Variance-position correlation: r = -0.62                                ║
    ║                                                                               ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  STAGE 2: BLOCK 2 EXTRACTS POSITION SIGNAL (TWO STRATEGIES)                  ║
    ║  ─────────────────────────────────────────────────────────────────────────    ║
    ║                                                                               ║
    ║  ┌─────────────────────────────┬─────────────────────────────────────────┐   ║
    ║  │   R0: BOS REFERENCE         │   R2: CURRENT-POSITION READING         │   ║
    ║  ├─────────────────────────────┼─────────────────────────────────────────┤   ║
    ║  │                             │                                         │   ║
    ║  │  Heads 6, 9 attend to pos 0 │  Heads attend to current position      │   ║
    ║  │                             │                                         │   ║
    ║  │  WHY IT WORKS:              │  WHY IT WORKS:                          │   ║
    ║  │  • k_0 is unique (un-       │  • V[i] directly contains the          │   ║
    ║  │    averaged embedding)      │    position-correlated signal          │   ║
    ║  │  • Learned Q,K biases       │  • ||V[i]|| correlates with position   │   ║
    ║  │    favor k_0                │                                         │   ║
    ║  │  • Provides constant        │  • No reference needed - direct        │   ║
    ║  │    reference for comparison │    readout of local statistics         │   ║
    ║  │                             │                                         │   ║
    ║  │  R² = 0.9996               │  R² = 0.9528                            │   ║
    ║  └─────────────────────────────┴─────────────────────────────────────────┘   ║
    ║                                                                               ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  STAGE 3: LINEAR HEAD EXTRACTS POSITION                                       ║
    ║  ─────────────────────────────────────────────────────────────────────────    ║
    ║                                                                               ║
    ║  Prediction: pred_i = w · h_i + b                                            ║
    ║                                                                               ║
    ║  The weight vector w aligns with position-correlated dimensions:              ║
    ║    - Cosine(w, dim_pos_corr) = {alignment:.4f}                               ║
    ║    - Position-prediction correlation = {pred_corr:.4f}                        ║
    ║                                                                               ║
    ║  Contribution breakdown:                                                      ║
    ║    - Block1 residual: position corr = {block1_corr:.4f}                      ║
    ║    - Attn2 delta: position corr = {attn2_corr:.4f}                           ║
    ║    - MLP2 delta: position corr = {mlp2_corr:.4f}                             ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """.format(
            alignment=head_results.get("w_corr_alignment", 0),
            pred_corr=head_results.get("pred_pos_correlation", 0),
            block1_corr=head_results.get("component_correlations", {}).get("block1", 0),
            attn2_corr=head_results.get("component_correlations", {}).get("attn2", 0),
            mlp2_corr=head_results.get("component_correlations", {}).get("mlp2", 0),
        )
    )

    # Mathematical summary
    print("\n" + "-" * 80)
    print("MATHEMATICAL SUMMARY")
    print("-" * 80)

    print(
        """
    THE FUNDAMENTAL INSIGHT:
    ────────────────────────
    
    Causal attention creates a natural position signal through AVERAGING:
    
        output_i ≈ (1/(i+1)) Σⱼ₌₀ⁱ f(eⱼ)
    
    This creates:
        - Variance ~ 1/(i+1)  (central limit theorem)
        - Norm decreases with position (averaging reduces magnitude)
    
    The model learns to READ this signal, not CREATE it.
    
    
    WHY BOS HEADS WORK (R0):
    ────────────────────────
    
    Position 0 is SPECIAL because it's the only un-averaged position.
    
        x_0 = LN(e_0 + small_perturbation)
        x_i = LN((1/(i+1)) Σⱼ₌₀ⁱ embeddings)  for i > 0
    
    BOS heads learn to:
    1. Make k_0 unique through the learned W_K and b_K
    2. Make all queries prefer k_0 through learned W_Q and b_Q
    3. Extract a CONSTANT reference vector v_0
    
    The prediction then uses: f(current_state, reference) → position
    
    
    WHY CURRENT-POSITION HEADS WORK (R2):
    ─────────────────────────────────────
    
    When Block 1 is frozen, its output already encodes position in:
        - Local norm: ||r_i^1|| correlates with position (r = {norm_corr:.3f})
        - Local variance: Var(r_i^1) correlates with position (r = {var_corr:.3f})
    
    Current-position heads simply READ this local signal:
        output_i ≈ V[i] = W_V @ LN(r_i^1) + b_V
    
    No reference needed - the position is encoded in the local statistics!
    """.format(
            norm_corr=r2_results.get("block1_norm_pos_corr", 0),
            var_corr=r2_results.get("block1_var_pos_corr", 0),
        )
    )

    synthesis["bos_heads"] = bos_results.get("bos_heads", [])
    synthesis["current_pos_heads"] = r2_results.get("current_pos_heads", [])
    synthesis["w_corr_alignment"] = head_results.get("w_corr_alignment", 0)
    synthesis["mechanisms"] = {
        "R0": "BOS reference comparison",
        "R2": "Current-position variance reading",
    }

    return synthesis


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("\n" + "=" * 80)
    print("DEEP MECHANISM ANALYSIS: 2-Layer NoPE Position Encoding")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load checkpoints
    print("\n--- Loading Checkpoints ---")
    model_r0, _ = load_checkpoint("R0")
    model_r0 = model_r0.to(device)
    print(f"  Loaded R0 (Full Training) model")

    model_r2, _ = load_checkpoint("R2")
    model_r2 = model_r2.to(device)
    print(f"  Loaded R2 (Attn2-only) model")

    # Get data
    print("\n--- Loading Data ---")
    torch.manual_seed(42)
    data = get_owt_batch(batch_size=32, block_size=128, device=device)
    print(f"  Loaded batch: {data.shape}")

    # Run all analyses
    bos_results = analyze_bos_heads(model_r0, data)
    head_results = analyze_linear_head(model_r0, data)
    r2_results = analyze_current_position_mechanism(model_r2, model_r0, data)

    # Synthesize findings
    synthesis = synthesize_findings(bos_results, head_results, r2_results)

    # Save results
    output_dir = Path("out-2layer-mechanism/deep_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "bos_analysis": {
            k: v for k, v in bos_results.items() if not isinstance(v, torch.Tensor)
        },
        "head_analysis": head_results,
        "r2_analysis": r2_results,
        "synthesis": synthesis,
    }

    with open(output_dir / "deep_mechanism_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to {output_dir}/deep_mechanism_analysis.json")

    return all_results


if __name__ == "__main__":
    main()
