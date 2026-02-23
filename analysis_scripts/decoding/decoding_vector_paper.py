"""
Decoding Vector Analysis - Paper Formula Implementation

This script implements the EXACT formula from the paper:
    w = W_V · Σ_{j=1}^{N} LN(E_j)

Where:
- Sum over all N tokens in the SEQUENCE (not vocab!)
- Apply LayerNorm to each embedding E_j
- Then transform by W_V

The decoded position at i is:
    decoded(i) = Σ_{j=1}^{i} (w · v_j) ≈ i · c

where v_j = W_V · LN(E_j) are the value vectors.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis_scripts/decoding_vector_paper.py --quick
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "decoding_vector_paper"
PLOTS_DIR = RESULTS_DIR / "plots"
OWT_DATA_PATH = PROJECT_ROOT / "nanoGPT" / "data" / "openwebtext" / "train.bin"

D_MODEL = 768
N_HEAD = 12


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


def extract_W_v(model: GPT) -> torch.Tensor:
    """Extract W_V matrix (768, 768)."""
    block = model.transformer.h[0]
    c_attn_weight = block.attn.c_attn.weight.data  # (2304, 768)
    W_v = c_attn_weight[2 * D_MODEL :, :].T  # (768, 768)
    return W_v


def compute_decoding_vector_paper(
    model: GPT, tokens: torch.Tensor, W_v: torch.Tensor
) -> torch.Tensor:
    """
    Compute the decoding vector using the PAPER formula:
        w = W_V · Σ_{j=1}^{N} LN(E_j)

    Args:
        model: GPT model
        tokens: (T,) token indices for a single sequence
        W_v: (768, 768) value projection matrix

    Returns:
        w: (768,) decoding vector
    """
    with torch.no_grad():
        # Get embeddings for this sequence
        embeddings = model.transformer.wte(tokens)  # (T, 768)

        # Apply LayerNorm to each embedding
        block = model.transformer.h[0]
        ln_embeddings = block.ln_1(embeddings.unsqueeze(0)).squeeze(0)  # (T, 768)

        # Sum normalized embeddings: Σ_{j=1}^{N} LN(E_j)
        sum_ln_embeddings = ln_embeddings.sum(dim=0)  # (768,)

        # Transform by W_V: w = W_V · Σ LN(E_j)
        w = W_v @ sum_ln_embeddings  # (768,)

    return w


def compute_value_vectors(
    model: GPT, tokens: torch.Tensor, W_v: torch.Tensor
) -> torch.Tensor:
    """
    Compute value vectors: v_j = W_V · LN(E_j)

    Args:
        model: GPT model
        tokens: (T,) token indices
        W_v: (768, 768) value projection matrix

    Returns:
        v: (T, 768) value vectors
    """
    with torch.no_grad():
        embeddings = model.transformer.wte(tokens)  # (T, 768)
        block = model.transformer.h[0]
        ln_embeddings = block.ln_1(embeddings.unsqueeze(0)).squeeze(0)  # (T, 768)

        # v_j = W_V · LN(E_j)
        v = ln_embeddings @ W_v.T  # (T, 768)

    return v


def decode_positions_paper(w: torch.Tensor, v: torch.Tensor) -> np.ndarray:
    """
    Decode positions using the paper formula:
        decoded(i) = Σ_{j=1}^{i} (w · v_j)

    Args:
        w: (768,) decoding vector
        v: (T, 768) value vectors

    Returns:
        decoded: (T,) decoded positions
    """
    T = v.shape[0]

    # Compute w · v_j for each j
    contributions = (v @ w).cpu().numpy()  # (T,)

    # Cumulative sum: decoded(i) = Σ_{j=1}^{i} (w · v_j)
    decoded = np.cumsum(contributions)  # (T,)

    return decoded


def analyze_position_correlation(decoded: np.ndarray, positions: np.ndarray) -> Dict:
    """Analyze correlation between decoded and true positions."""
    pearson_r, pearson_p = pearsonr(decoded, positions)
    spearman_r, spearman_p = spearmanr(decoded, positions)

    X = decoded.reshape(-1, 1)
    y = positions

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


def create_scatter_plot(
    decoded: np.ndarray,
    positions: np.ndarray,
    title: str,
):
    """Create scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        positions, decoded, c=positions, cmap="viridis", alpha=0.6, s=10
    )

    # Linear fit
    z = np.polyfit(positions, decoded, 1)
    p = np.poly1d(z)
    x_line = np.linspace(positions.min(), positions.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Linear fit")

    r, _ = pearsonr(decoded, positions)

    ax.set_xlabel("True Position", fontsize=12)
    ax.set_ylabel("Decoded Position (cumsum of w·v_j)", fontsize=12)
    ax.set_title(f"{title}\nPearson r = {r:.4f}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label="Position")
    plt.tight_layout()

    return fig


def run_analysis(model_name: str, model: GPT, tokens: torch.Tensor) -> Dict:
    """Run full decoding analysis."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(f"{'=' * 60}")

    n_sequences, context_length = tokens.shape
    results = {
        "model": model_name,
        "n_sequences": n_sequences,
        "context_length": context_length,
    }

    # Extract W_V
    print("\n[1/3] Extracting W_V matrix...")
    W_v = extract_W_v(model)
    print(f"    W_V shape: {W_v.shape}")

    # Decode each sequence
    print("\n[2/3] Decoding positions for each sequence...")
    all_decoded = []
    all_positions = []

    for seq_idx in tqdm(range(n_sequences), desc="Processing sequences"):
        seq_tokens = tokens[seq_idx]  # (T,)

        # Compute decoding vector for this sequence
        w = compute_decoding_vector_paper(model, seq_tokens, W_v)

        # Compute value vectors
        v = compute_value_vectors(model, seq_tokens, W_v)

        # Decode positions
        decoded = decode_positions_paper(w, v)

        all_decoded.extend(decoded)
        all_positions.extend(range(context_length))

    all_decoded = np.array(all_decoded)
    all_positions = np.array(all_positions)

    # Analyze
    print("\n[3/3] Analyzing correlations...")
    metrics = analyze_position_correlation(all_decoded, all_positions)
    results.update(metrics)

    print(f"    Pearson r: {metrics['pearson_r']:.4f}")
    print(f"    Spearman r: {metrics['spearman_r']:.4f}")
    print(f"    R² score: {metrics['r2_score']:.4f}")

    # Plot
    fig = create_scatter_plot(all_decoded, all_positions, model_name)
    plot_path = PLOTS_DIR / f"scatter_{model_name}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sequences", type=int, default=100)
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_sequences = 30
        args.context_length = 64

    print("=" * 70)
    print("DECODING VECTOR ANALYSIS - PAPER FORMULA")
    print("=" * 70)
    print(f"Formula: w = W_V · Σ_{{j=1}}^{{N}} LN(E_j)")
    print(f"  Sequences: {args.n_sequences}")
    print(f"  Context: {args.context_length}")
    print(f"  Device: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("\nLoading OWT data...")
    owt_data = load_owt_data()
    tokens = get_owt_sequences(owt_data, args.n_sequences, args.context_length)

    # Run analysis
    print("\n" + "=" * 70)
    print("MODEL: NoPE + LayerNorm (Random Initialization)")
    print("=" * 70)

    random_model, _ = create_random_model(use_pe=False, block_size=args.context_length)
    results = run_analysis("NoPE_LN_random", random_model, tokens)

    # Save
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump({"NoPE_LN_random": results}, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Pearson r:  {results['pearson_r']:>8.4f}")
    print(f"Spearman r: {results['spearman_r']:>8.4f}")
    print(f"R² score:   {results['r2_score']:>8.4f}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
