"""
Per-Head Decoding Vector Analysis for NoPE Transformers

This script correctly handles the multi-head structure by:
1. Inverting each head's W_v and W_o independently
2. Constructing the combined inverse from per-head inverses
3. Using the orthogonality of embeddings to decode position

KEY INSIGHT: W_o and W_v are block-structured (12 heads), and we must
invert each block separately, then combine.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis_scripts/decoding_vector_per_head.py --quick
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List
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

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "decoding_vector_per_head"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
OWT_DATA_PATH = PROJECT_ROOT / "nanoGPT" / "data" / "openwebtext" / "train.bin"

D_MODEL = 768
N_HEAD = 12
D_HEAD = D_MODEL // N_HEAD  # 64


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


def extract_head_matrices(model: GPT) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extract per-head W_v and W_o matrices.

    Returns:
        W_v_heads: list of 12 tensors, each (768, 64)
        W_o_heads: list of 12 tensors, each (64, 768)
    """
    block = model.transformer.h[0]

    # Extract W_V from c_attn
    c_attn_weight = block.attn.c_attn.weight.data  # (2304, 768)
    W_v_combined = c_attn_weight[2 * D_MODEL :, :].T  # (768, 768)

    # Split into heads
    W_v_heads = []
    for h in range(N_HEAD):
        start = h * D_HEAD
        end = (h + 1) * D_HEAD
        W_v_h = W_v_combined[:, start:end]  # (768, 64)
        W_v_heads.append(W_v_h)

    # Extract W_O from c_proj
    W_o_combined = block.attn.c_proj.weight.data.T  # (768, 768)

    # Split into heads
    W_o_heads = []
    for h in range(N_HEAD):
        start = h * D_HEAD
        end = (h + 1) * D_HEAD
        W_o_h = W_o_combined[start:end, :]  # (64, 768)
        W_o_heads.append(W_o_h)

    return W_v_heads, W_o_heads


def construct_combined_inverse(
    W_v_heads: List[torch.Tensor], W_o_heads: List[torch.Tensor]
) -> torch.Tensor:
    """
    Construct the combined inverse W_combined^{-1} from per-head inverses.

    For multi-head attention:
        Forward: concat([v_1, v_2, ..., v_12]) @ W_o -> output (768,)
        where v_h = input @ W_v_h (64-dimensional)

    To invert, we compute each head's inverse separately, then stack.

    Returns:
        W_combined_inv: (768, 768) - the combined inverse matrix
    """
    print(f"\n    Constructing combined inverse from {N_HEAD} heads...")

    # For each head, compute: W_v_h^{-1} @ W_o_h^{-1}
    head_inverses = []

    for h in range(N_HEAD):
        W_v_h = W_v_heads[h]  # (768, 64)
        W_o_h = W_o_heads[h]  # (64, 768)

        # Pseudo-inverses
        W_v_h_pinv = torch.linalg.pinv(W_v_h)  # (64, 768)
        W_o_h_pinv = torch.linalg.pinv(W_o_h)  # (768, 64)

        # Combined inverse for this head: W_v_h^{-1} @ W_o_h^{-1}
        # (64, 768) @ (768, 64) = (64, 64)
        head_inv = W_v_h_pinv @ W_o_h_pinv  # (64, 64)

        head_inverses.append(head_inv)

        if h == 0:
            print(f"      Head 0 inverse shape: {head_inv.shape}")

    # Stack into block-diagonal matrix
    # Each head contributes a (64, 64) block
    W_combined_inv = torch.zeros(D_MODEL, D_MODEL, device=W_v_heads[0].device)

    for h in range(N_HEAD):
        start = h * D_HEAD
        end = (h + 1) * D_HEAD
        W_combined_inv[start:end, start:end] = head_inverses[h]

    print(f"      Combined inverse shape: {W_combined_inv.shape}")
    print(f"      Sparsity: {(W_combined_inv == 0).float().mean().item():.2%}")

    return W_combined_inv


def get_activations_and_embeddings(
    model: GPT, tokens: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract activations at post_attn and post_ln2.

    Returns:
        activations: Dict with 'post_attn' and 'post_ln2' tensors (B, T, D)
        embeddings: Raw token embeddings (B, T, D)
    """
    activations = {}

    with torch.no_grad():
        embeddings = model.transformer.wte(tokens)  # (B, T, D)
        x = embeddings.clone()

        block = model.transformer.h[0]

        # Post-LN1
        x_ln1 = block.ln_1(x)

        # Post-Attention (BEFORE residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # After residual
        x = x + attn_out

        # Post-LN2
        if hasattr(block, "ln_2") and not getattr(block, "skip_ln2", False):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
        else:
            activations["post_ln2"] = x.clone()

    return activations, embeddings


def compute_decoding_projections(
    activations: torch.Tensor,
    vocab_sum: torch.Tensor,
    W_combined_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Decode position using the combined inverse matrix.

    Process:
    1. Invert: inverted = activations @ W_combined_inv.T
    2. Project onto vocab sum: projection = inverted · vocab_sum

    Args:
        activations: (B, T, 768) activations
        vocab_sum: (768,) sum of all vocab embeddings
        W_combined_inv: (768, 768) combined inverse matrix

    Returns:
        projections: (B, T) scalar projections
    """
    B, T, D = activations.shape

    # Invert: (B, T, 768) @ (768, 768) = (B, T, 768)
    inverted = activations @ W_combined_inv.T

    # Project onto vocab sum
    vocab_sum_norm = vocab_sum / (vocab_sum.norm() + 1e-8)
    projections = inverted @ vocab_sum_norm  # (B, T)

    return projections


def analyze_position_correlation(values: np.ndarray, positions: np.ndarray) -> Dict:
    """Analyze correlation between values and positions."""
    pearson_r, pearson_p = pearsonr(values, positions)
    spearman_r, spearman_p = spearmanr(values, positions)

    X = values.reshape(-1, 1)
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
    values: np.ndarray,
    positions: np.ndarray,
    title: str,
    n_samples_plot: int = 5000,
):
    """Create scatter plot."""
    if len(positions) > n_samples_plot:
        idx = np.random.choice(len(positions), n_samples_plot, replace=False)
        values_sub = values[idx]
        positions_sub = positions[idx]
    else:
        values_sub = values
        positions_sub = positions

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        positions_sub, values_sub, c=positions_sub, cmap="viridis", alpha=0.5, s=5
    )

    z = np.polyfit(positions_sub, values_sub, 1)
    p = np.poly1d(z)
    x_line = np.linspace(positions_sub.min(), positions_sub.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Linear fit")

    r, _ = pearsonr(values_sub, positions_sub)

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Projection onto Vocab Sum", fontsize=12)
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
        "layers": {},
    }

    # Extract per-head matrices
    print("\n[1/4] Extracting per-head W_v and W_o matrices...")
    W_v_heads, W_o_heads = extract_head_matrices(model)
    print(f"    {N_HEAD} heads, each W_v: (768, 64), W_o: (64, 768)")

    # Construct combined inverse
    print("\n[2/4] Constructing combined inverse matrix...")
    W_combined_inv = construct_combined_inverse(W_v_heads, W_o_heads)

    # Get vocab sum
    all_embeddings = model.transformer.wte.weight.data
    vocab_sum = all_embeddings.sum(dim=0)
    print(f"    Vocab sum norm: {vocab_sum.norm().item():.4f}")

    # Extract activations
    print("\n[3/4] Extracting activations...")
    all_projections = {"post_attn": [], "post_ln2": []}
    all_positions = []

    batch_size = 10
    for batch_start in tqdm(
        range(0, n_sequences, batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, n_sequences)
        batch_tokens = tokens[batch_start:batch_end]

        activations, _ = get_activations_and_embeddings(model, batch_tokens)

        for layer in ["post_attn", "post_ln2"]:
            projections = compute_decoding_projections(
                activations[layer], vocab_sum, W_combined_inv
            )
            all_projections[layer].append(projections.cpu().numpy())

        B = batch_end - batch_start
        for _ in range(B):
            all_positions.extend(range(context_length))

    positions = np.array(all_positions)
    for layer in ["post_attn", "post_ln2"]:
        all_projections[layer] = np.concatenate(all_projections[layer]).flatten()

    # Analyze
    print("\n[4/4] Analyzing correlations...")
    for layer in ["post_attn", "post_ln2"]:
        print(f"\n--- Layer: {layer} ---")
        projections = all_projections[layer]

        metrics = analyze_position_correlation(projections, positions)
        results["layers"][layer] = metrics

        print(f"    Pearson r: {metrics['pearson_r']:.4f}")
        print(f"    R² score: {metrics['r2_score']:.4f}")

        # Plot
        fig = create_scatter_plot(projections, positions, f"{model_name} - {layer}")
        plot_path = PLOTS_DIR / f"scatter_{model_name}_{layer}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sequences", type=int, default=100)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_sequences = 30

    print("=" * 70)
    print("PER-HEAD DECODING VECTOR ANALYSIS")
    print("=" * 70)
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
    for layer, metrics in results["layers"].items():
        print(
            f"{layer:15} Pearson r: {metrics['pearson_r']:>8.4f}  R²: {metrics['r2_score']:>8.4f}"
        )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
