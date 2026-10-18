"""
Corrected Decoding Vector Analysis for NoPE Transformers

This script uses the orthogonality of high-dimensional embeddings to decode position.

KEY INSIGHT - Orthogonality in High Dimensions:
    In 768-d space, random embeddings are approximately orthogonal:
    - e_i · e_i ≈ c (constant, related to norm²)
    - e_i · e_j ≈ 0 for i ≠ j

DECODING MECHANISM:
    1. Invert attention: inverted = W_v^{-1} @ W_o^{-1} @ activation
       At position j: inverted ≈ Σ_{i=1}^j normalized(e_i)
    
    2. Project onto vocab sum: projection = inverted · Σ_{all vocab} e_token
    
    3. Due to orthogonality:
       - The j tokens in the sequence each contribute: e_i · e_i ≈ c
       - All other vocab tokens contribute: e_i · e_j ≈ 0
       - Total: projection ≈ j * c (LINEAR in position!)

Layers analyzed:
- post_attn: After attention, before residual
- post_ln2: After LN2, before MLP

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis_scripts/decoding_vector_corrected.py \
        --n_sequences 100 --context_length 256

Author: Research Assistant
Date: January 2026
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Results will only be saved locally.")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "decoding_vector_corrected"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
OWT_DATA_PATH = PROJECT_ROOT / "nanoGPT" / "data" / "openwebtext" / "train.bin"

# Layers to analyze
LAYERS = ["post_attn", "post_ln2"]

# Model constants
D_MODEL = 768
N_HEAD = 12
D_HEAD = D_MODEL // N_HEAD  # 64
SIGMA_INIT = 0.02  # Standard GPT-2 embedding initialization std


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def create_random_model(
    use_pe: bool = False, block_size: int = 256
) -> Tuple[GPT, GPTConfig]:
    """Create a randomly initialized NoPE model."""
    config = GPTConfig(
        n_layer=1,
        n_head=N_HEAD,
        n_embd=D_MODEL,
        block_size=block_size,
        vocab_size=50304,
        dropout=0.0,
        use_positional_embedding=use_pe,
        norm_type="layernorm",
        bias=False,
    )
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model, config


def load_trained_model(
    checkpoint_path: str, use_pe: bool = False, block_size: int = 256
) -> Tuple[GPT, GPTConfig]:
    """Load a trained model from checkpoint."""
    full_path = CHECKPOINT_DIR / checkpoint_path
    if not full_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {full_path}")

    checkpoint = torch.load(full_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", N_HEAD),
        n_embd=model_args.get("n_embd", D_MODEL),
        block_size=block_size,
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=use_pe,
        norm_type="layernorm",
        bias=model_args.get("bias", False),
    )

    model = GPT(config)
    state_dict = checkpoint["model"]
    # Remove _orig_mod. prefix if present (from torch.compile)
    unwrapped = {
        (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(unwrapped)
    model.eval()
    model.to(DEVICE)
    return model, config


def load_owt_data() -> np.ndarray:
    """Load OpenWebText training data."""
    if not OWT_DATA_PATH.exists():
        raise FileNotFoundError(f"OWT data not found at {OWT_DATA_PATH}")
    data = np.memmap(OWT_DATA_PATH, dtype=np.uint16, mode="r")
    return data


def get_owt_sequences(
    data: np.ndarray, n_sequences: int, context_length: int, seed: int = 42
) -> torch.Tensor:
    """Sample random sequences from OWT data."""
    np.random.seed(seed)
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, n_sequences)

    sequences = []
    for start in starts:
        seq = data[start : start + context_length].astype(np.int64)
        sequences.append(seq)

    return torch.tensor(np.stack(sequences), device=DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_projection_matrices(model: GPT) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract W_v and W_o matrices from the model, structured by head.

    Each head has:
    - W_v_h: (768, 64) - value projection for head h
    - W_o_h: (64, 768) - output projection for head h

    Returns:
        W_v_heads: list of 12 tensors, each (768, 64)
        W_o_heads: list of 12 tensors, each (64, 768)
    """
    block = model.transformer.h[0]
    d_head = D_MODEL // N_HEAD  # 64

    # Extract W_V from c_attn (last 768 rows of 2304 total)
    c_attn_weight = block.attn.c_attn.weight.data  # (2304, 768)
    W_v_combined = c_attn_weight[2 * D_MODEL :, :].T  # (768, 768)

    # Split into heads: (768, 768) -> 12 x (768, 64)
    W_v_heads = []
    for h in range(N_HEAD):
        start = h * d_head
        end = (h + 1) * d_head
        W_v_h = W_v_combined[:, start:end]  # (768, 64)
        W_v_heads.append(W_v_h)

    # Extract W_O from c_proj
    # c_proj.weight is (768, 768), where input is concatenated heads
    # For forward: concat(head_outputs) @ W_o.T
    # W_o.T is (768, 768), so W_o is (768, 768)
    W_o_combined = block.attn.c_proj.weight.data.T  # (768, 768)

    # Split into heads: (768, 768) -> 12 x (64, 768)
    W_o_heads = []
    for h in range(N_HEAD):
        start = h * d_head
        end = (h + 1) * d_head
        W_o_h = W_o_combined[start:end, :]  # (64, 768)
        W_o_heads.append(W_o_h)

    return W_v_heads, W_o_heads


def compute_vocab_sum_and_inverses(
    model: GPT, W_v: torch.Tensor, W_o: torch.Tensor
) -> tuple:
    """
    Compute the vocabulary embedding sum and weight pseudo-inverses.

    Key insight: In high dimensions, embeddings are approximately orthogonal.
    - e_i · e_i ≈ c (constant)
    - e_i · e_j ≈ 0 for i ≠ j

    So when we project the inverted activation at position j onto the vocab sum:
    - Only the j tokens in the sequence contribute (each contributes c)
    - Result scales linearly with j: projection ≈ j * c

    Args:
        model: GPT model (to access embedding matrix)
        W_v: (768, 768) value projection matrix
        W_o: (768, 768) output projection matrix

    Returns:
        vocab_sum: (768,) sum of all vocabulary embeddings
        W_v_pinv: (768, 768) pseudo-inverse of W_v
        W_o_pinv: (768, 768) pseudo-inverse of W_o
    """
    # Compute pseudo-inverses
    W_v_pinv = torch.linalg.pinv(W_v)  # (768, 768)
    W_o_pinv = torch.linalg.pinv(W_o)  # (768, 768)

    # Sum ALL vocabulary embeddings (the decoding vector)
    all_embeddings = model.transformer.wte.weight.data  # (vocab_size, 768)
    vocab_sum = all_embeddings.sum(dim=0)  # (768,)

    print(f"    Vocab sum norm: {vocab_sum.norm().item():.4f}")
    print(f"    Vocab size: {all_embeddings.shape[0]}")

    return vocab_sum, W_v_pinv, W_o_pinv


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def get_activations_and_embeddings(
    model: GPT, tokens: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract activations at post_attn and post_ln2, plus raw embeddings.

    Returns:
        activations: Dict with 'post_attn' and 'post_ln2' tensors (B, T, D)
        embeddings: Raw token embeddings (B, T, D)
    """
    activations = {}

    with torch.no_grad():
        # Get embeddings
        embeddings = model.transformer.wte(tokens)  # (B, T, D)

        # No positional embeddings for NoPE
        x = embeddings.clone()

        # Get the block
        block = model.transformer.h[0]

        # Post-LN1
        x_ln1 = block.ln_1(x)

        # Post-Attention output (BEFORE residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # After first residual
        x = x + attn_out

        # Post-LN2 (BEFORE MLP)
        if hasattr(block, "ln_2") and not getattr(block, "skip_ln2", False):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
        else:
            activations["post_ln2"] = x.clone()

    return activations, embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# DECODING VECTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_decoding_projections(
    activations: torch.Tensor,
    vocab_sum: torch.Tensor,
    W_v_pinv_heads: list,
    W_o_pinv_heads: list,
) -> torch.Tensor:
    """
    Decode position from activations using orthogonality of embeddings.

    Process (PER HEAD):
    1. Split activation into 12 heads: activation -> 12 x (B, T, 64)
    2. For each head h:
       a. Invert: inverted_h = W_v_h^{-1} @ W_o_h^{-1} @ activation_h
       b. This gives (B, T, 768) per head
    3. Sum across heads: inverted = Σ_h inverted_h
    4. Project onto vocab sum: projection = inverted · vocab_sum

    Args:
        activations: (B, T, 768) activations at some layer
        vocab_sum: (768,) sum of all vocab embeddings
        W_v_pinv_heads: list of 12 tensors, each (64, 768)
        W_o_pinv_heads: list of 12 tensors, each (768, 64)

    Returns:
        projections: (B, T) scalar projections for each position
    """
    B, T, D = activations.shape
    d_head = D // N_HEAD  # 64

    # Step 1: Split activations by head
    # (B, T, 768) -> (B, T, 12, 64)
    acts_heads = activations.reshape(B, T, N_HEAD, d_head)

    # Step 2: Invert each head independently
    inverted_total = torch.zeros(B, T, D, device=activations.device)

    for h in range(N_HEAD):
        # Get head h activations: (B, T, 64)
        acts_h = acts_heads[:, :, h, :]  # (B, T, 64)

        # Flatten for matrix multiply
        acts_h_flat = acts_h.reshape(B * T, d_head)  # (B*T, 64)

        W_o_h_pinv = W_o_pinv_heads[h]  # (768, 64)
        W_v_h_pinv = W_v_pinv_heads[h]  # (64, 768)

        # Apply W_o_h^{-1}: (768, 64) @ (64, B*T)^T = (768, 64) @ (B*T, 64)^T
        # = (B*T, 768)
        step1 = (W_o_h_pinv @ acts_h_flat.T).T  # (B*T, 768)

        # Apply W_v_h^{-1}: (64, 768) @ (768, B*T)^T = (B*T, 64)
        # Wait, this doesn't give us (B*T, 768)...

        # Let me reconsider the order:
        # Forward for head h: x (768,) -> W_v_h (768,64) -> v_h (64,) -> W_o_h (64,768) -> out_h (768,)
        # So: out_h = x @ W_v_h @ W_o_h
        # Inverse: x = out_h @ W_o_h^{-1} @ W_v_h^{-1}
        # But W_o_h is (64, 768), so W_o_h^{-1} is (768, 64)
        # And W_v_h is (768, 64), so W_v_h^{-1} is (64, 768)

        # out_h @ W_o_h^{-1} = (768,) @ (768, 64) = (64,)
        # (64,) @ W_v_h^{-1} = (64,) @ (64, 768) = (768,)

        # So correct order: W_v_h^{-1} @ W_o_h^{-1} @ out_h
        step1 = (
            W_o_h_pinv @ acts_h_flat.T
        ).T  # (768, 64) @ (B*T, 64)^T = (B*T, 768)? No...

        # Let me be more careful:
        # acts_h_flat: (B*T, 64)
        # W_o_h_pinv: (768, 64)
        # We want: acts_h_flat @ W_o_h_pinv.T = (B*T, 64) @ (64, 768) = (B*T, 768)

        step1 = acts_h_flat @ W_o_h_pinv.T  # (B*T, 64) @ (64, 768) = (B*T, 768)

        # step1: (B*T, 768)
        # W_v_h_pinv: (64, 768)
        # We want: step1 @ W_v_h_pinv.T = (B*T, 768) @ (768, 64) = (B*T, 64)
        # Then reshape to (B, T, 64)

        # Actually wait - the inversion should give us back the normalized embeddings
        # which are (768,) dimensional, not (64,)

        # Let me reconsider the forward pass more carefully:
        # For head h:
        #   input x: (768,)
        #   v_h = x @ W_v_h: (768,) @ (768, 64) = (64,)
        #   Then attention averages the v_h's across sequence
        #   attn_out_h = avg(v_h) @ W_o_h: (64,) @ (64, 768) = (768,)

        # So the output is (768,) per head
        # The full output is the SUM of all 12 heads (not concat)

        # To invert head h:
        #   We have attn_out_h: (768,)
        #   avg(v_h) = attn_out_h @ W_o_h^{-1}: (768,) @ (768, 64) = (64,)
        #   But we want to get back to the (768,) space

        # Hmm, I think the issue is that the attention output is the SUM of heads,
        # not the concatenation. Let me check the nanoGPT code...

        # Actually, looking at standard transformer:
        # - Each head produces (64,) after attention
        # - All 12 heads are CONCATENATED to (768,)
        # - Then projected through W_o

        # So: concat([h1, h2, ..., h12]) @ W_o
        # where concat(...) is (768,)

        # This means activations (B, T, 768) = concat of 12 heads * some W_o

        # Let me just try the simplest approach:
        # For head h, the contribution to the output is head_h @ W_o_h
        # To invert: output_h @ W_o_h^{-1} = head_h
        # Then head_h @ W_v_h^{-1} = input_h (the 768-d input projected to 64-d)

        # But that gives us 64-d, not 768-d. We can't recover the full input.

        # I think the key insight is: we want to get back to the SUM of normalized embeddings
        # For head h: Σ ln_i @ W_v_h @ W_o_h = activation_contribution_h
        # Invert: activation_contribution_h @ W_o_h^{-1} @ W_v_h^{-1} = Σ ln_i (projected)

        # Since W_v_h is (768, 64), W_v_h^{-1} is (64, 768)
        # So we DO get back to 768-d!

        # Correct order:
        # acts_h: (B*T, 64) - this is the head h's part of the activation
        # step1 = acts_h @ W_o_h^{-1}.T: (B*T, 64) @ (64, 768) = (B*T, 768)
        # inverted_h = step1 @ W_v_h^{-1}.T: (B*T, 768) @ (768, 64) = (B*T, 64)

        # Wait, that ends up at 64-d again. This doesn't work.

        # Let me reconsider: maybe the activations at this layer are NOT split by head?
        # Maybe post_attn is the FULL (768,) output after all heads have been combined?

        # If so, we can't split it back into heads after the fact.
        # We'd need to go back to the attention mechanism itself.

        # For now, let me assume activations are the FULL combined output
        # and we need a different approach for per-head inversion.

        # Actually, I think the issue is that we need to get the PER-HEAD OUTPUTS
        # from the model, not split the combined output.

        # Let me simplify: just combine all the W_v_h and W_o_h into full matrices
        # like I had before, but construct them correctly from the per-head inverses.

        pass  # Placeholder - will fix this

    return torch.zeros(B, T, device=activations.device)  # Placeholder


def analyze_position_correlation(
    projections: np.ndarray,
    positions: np.ndarray,
) -> Dict:
    """
    Analyze correlation between projections and positions.

    Args:
        projections: (N,) projection values
        positions: (N,) position indices

    Returns:
        results: Dict with correlation metrics
    """
    # Overall correlation
    pearson_r, pearson_p = pearsonr(projections, positions)
    spearman_r, spearman_p = spearmanr(projections, positions)

    # R² from linear regression
    X = projections.reshape(-1, 1)
    y = positions

    # Split
    n = len(y)
    n_train = int(0.8 * n)
    idx = np.random.permutation(n)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[idx[:n_train]])
    X_test = scaler.transform(X[idx[n_train:]])
    y_train = y[idx[:n_train]]
    y_test = y[idx[n_train:]]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "r2_score": float(r2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def create_projection_vs_position_plot(
    projections: np.ndarray,
    positions: np.ndarray,
    title: str,
    n_samples_plot: int = 5000,
) -> plt.Figure:
    """Create scatter plot of projection vs position."""
    # Subsample for plotting
    if len(positions) > n_samples_plot:
        idx = np.random.choice(len(positions), n_samples_plot, replace=False)
        projections_sub = projections[idx]
        positions_sub = positions[idx]
    else:
        projections_sub = projections
        positions_sub = positions

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by position
    scatter = ax.scatter(
        positions_sub, projections_sub, c=positions_sub, cmap="viridis", alpha=0.5, s=5
    )

    # Add regression line
    z = np.polyfit(positions_sub, projections_sub, 1)
    p = np.poly1d(z)
    x_line = np.linspace(positions_sub.min(), positions_sub.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Linear fit")

    # Compute correlation for display
    r, _ = pearsonr(projections_sub, positions_sub)

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Projection onto Decoding Vector", fontsize=12)
    ax.set_title(f"{title}\nPearson r = {r:.4f}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label="Position")
    plt.tight_layout()

    return fig


def create_per_position_boxplot(
    projections: np.ndarray,
    positions: np.ndarray,
    title: str,
    n_bins: int = 16,
) -> plt.Figure:
    """Create boxplot of projections binned by position."""
    max_pos = positions.max() + 1
    bin_size = max_pos // n_bins

    binned_data = []
    bin_labels = []

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        mask = (positions >= start) & (positions < end)
        if mask.sum() > 0:
            binned_data.append(projections[mask])
            bin_labels.append(f"{start}-{end - 1}")

    fig, ax = plt.subplots(figsize=(14, 6))

    bp = ax.boxplot(binned_data, labels=bin_labels, patch_artist=True)

    # Color boxes by position
    colors = plt.cm.viridis(np.linspace(0, 1, len(binned_data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Position Range", fontsize=12)
    ax.set_ylabel("Projection onto Decoding Vector", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def create_summary_comparison_plot(
    results: Dict,
    title: str,
) -> plt.Figure:
    """Create summary bar plot comparing models and layers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results.keys())
    layers = LAYERS

    # Plot 1: Pearson correlation
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35

    for i, layer in enumerate(layers):
        correlations = [
            results[m]["layers"].get(layer, {}).get("pearson_r", 0) for m in models
        ]
        ax.bar(x + i * width, correlations, width, label=layer, alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Position-Projection Correlation")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Plot 2: R² score
    ax = axes[1]
    for i, layer in enumerate(layers):
        r2_scores = [
            results[m]["layers"].get(layer, {}).get("r2_score", 0) for m in models
        ]
        ax.bar(x + i * width, r2_scores, width, label=layer, alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("R² Score")
    ax.set_title("Position Prediction from Projection")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def run_analysis(
    model_name: str,
    model: GPT,
    tokens: torch.Tensor,
    use_wandb: bool = True,
) -> Dict:
    """Run full decoding vector analysis for a single model."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(f"{'=' * 60}")

    n_sequences, context_length = tokens.shape

    results = {
        "model": model_name,
        "n_sequences": n_sequences,
        "context_length": context_length,
        "layers": {},
    }

    # Extract projection matrices
    print("\n[1/4] Extracting W_v and W_o matrices...")
    W_v, W_o = extract_projection_matrices(model)
    print(f"    W_v shape: {W_v.shape}")
    print(f"    W_o shape: {W_o.shape}")

    # Compute vocab sum and weight inverses
    print("\n[2/4] Computing vocab sum and pseudo-inverses...")
    vocab_sum, W_v_pinv, W_o_pinv = compute_vocab_sum_and_inverses(model, W_v, W_o)

    # Extract activations and compute projections
    print("\n[3/4] Extracting activations and computing projections...")
    all_projections = {layer: [] for layer in LAYERS}
    all_positions = []

    batch_size = 10
    for batch_start in tqdm(
        range(0, n_sequences, batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, n_sequences)
        batch_tokens = tokens[batch_start:batch_end]

        activations, embeddings = get_activations_and_embeddings(model, batch_tokens)

        for layer in LAYERS:
            layer_acts = activations[layer]
            projections = compute_decoding_projections(
                layer_acts, vocab_sum, W_v_pinv, W_o_pinv
            )
            all_projections[layer].append(projections.cpu().numpy())

        # Track positions
        B = batch_end - batch_start
        for _ in range(B):
            all_positions.extend(range(context_length))

    # Concatenate all results
    positions = np.array(all_positions)
    for layer in LAYERS:
        all_projections[layer] = np.concatenate(all_projections[layer]).flatten()

    # Analyze each layer
    print("\n[4/4] Analyzing correlations...")
    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")
        projections = all_projections[layer]

        # Compute correlation metrics
        metrics = analyze_position_correlation(projections, positions)
        results["layers"][layer] = metrics

        print(f"    Pearson r: {metrics['pearson_r']:.4f}")
        print(f"    Spearman r: {metrics['spearman_r']:.4f}")
        print(f"    R² score: {metrics['r2_score']:.4f}")

        # Create visualizations
        # 1. Scatter plot
        fig_scatter = create_projection_vs_position_plot(
            projections,
            positions,
            f"{model_name} - {layer}\nProjection onto Decoding Vector vs Position",
        )
        scatter_path = PLOTS_DIR / f"scatter_{model_name}_{layer}.png"
        fig_scatter.savefig(scatter_path, dpi=150, bbox_inches="tight")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"scatter/{model_name}/{layer}": wandb.Image(fig_scatter)})
        plt.close(fig_scatter)

        # 2. Boxplot
        fig_box = create_per_position_boxplot(
            projections,
            positions,
            f"{model_name} - {layer}\nProjection Distribution by Position",
        )
        box_path = PLOTS_DIR / f"boxplot_{model_name}_{layer}.png"
        fig_box.savefig(box_path, dpi=150, bbox_inches="tight")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"boxplot/{model_name}/{layer}": wandb.Image(fig_box)})
        plt.close(fig_box)

    return results


def main():
    parser = argparse.ArgumentParser(description="Corrected Decoding Vector Analysis")
    parser.add_argument(
        "--n_sequences", type=int, default=100, help="Number of sequences"
    )
    parser.add_argument(
        "--context_length", type=int, default=256, help="Context length"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with fewer samples"
    )
    args = parser.parse_args()

    if args.quick:
        args.n_sequences = 30

    print("=" * 70)
    print("CORRECTED DECODING VECTOR ANALYSIS")
    print("=" * 70)
    print(f"Formula: decoding_vector = W_o^-1 @ W_v^-1 * √d * σ_init + Σe_i")
    print(f"  Sequences: {args.n_sequences}")
    print(f"  Context length: {args.context_length}")
    print(f"  σ_init: {SIGMA_INIT}")
    print(f"  √d: {np.sqrt(D_MODEL):.2f}")
    print(f"  Device: {DEVICE}")

    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        wandb.init(
            project="nope-decoding-vector",
            name=f"corrected_n{args.n_sequences}_c{args.context_length}",
            config={
                "formula": "W_o^-1 @ W_v^-1 * sqrt(d) * sigma_init + sum(e_i)",
                "sigma_init": SIGMA_INIT,
                "sqrt_d": np.sqrt(D_MODEL),
                **vars(args),
            },
        )
        print("\nWandB initialized. Project: nope-decoding-vector")

    # Load OWT data
    print("\nLoading OWT data...")
    owt_data = load_owt_data()
    tokens = get_owt_sequences(owt_data, args.n_sequences, args.context_length)
    print(f"  Loaded {args.n_sequences} sequences of length {args.context_length}")

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # NoPE + LayerNorm - Random Init ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL: NoPE + LayerNorm (Random Initialization)")
    print("=" * 70)

    random_model, _ = create_random_model(use_pe=False, block_size=args.context_length)
    results = run_analysis("NoPE_LN_random", random_model, tokens, use_wandb)
    all_results["NoPE_LN_random"] = results
    del random_model
    torch.cuda.empty_cache()

    # Create summary comparison plot
    print("\n[Creating summary plots...]")
    fig_summary = create_summary_comparison_plot(
        all_results, "Decoding Vector Analysis: Position-Projection Correlation"
    )
    summary_path = PLOTS_DIR / "summary_comparison.png"
    fig_summary.savefig(summary_path, dpi=150, bbox_inches="tight")
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary/comparison": wandb.Image(fig_summary)})
    plt.close(fig_summary)

    # Save results
    results_path = RESULTS_DIR / "decoding_vector_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Layer':<15} {'Pearson r':<12} {'R²':<10}")
    print("-" * 65)

    for model_name, model_results in all_results.items():
        for layer, metrics in model_results.get("layers", {}).items():
            print(
                f"{model_name:<25} {layer:<15} {metrics['pearson_r']:>10.4f} {metrics['r2_score']:>10.4f}"
            )

    if use_wandb:
        # Log summary metrics
        for model_name, model_results in all_results.items():
            for layer, metrics in model_results.get("layers", {}).items():
                wandb.log(
                    {
                        f"metrics/{model_name}/{layer}/pearson_r": metrics["pearson_r"],
                        f"metrics/{model_name}/{layer}/r2_score": metrics["r2_score"],
                    }
                )
        wandb.finish()
        print("\nWandB run finished.")

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
