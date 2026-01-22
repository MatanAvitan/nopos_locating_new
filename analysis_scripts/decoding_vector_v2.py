"""
Decoding Vector Analysis V2 - Direct Inversion Approach

Instead of projecting onto a "decoding vector", we directly invert the attention
transformation to recover the normalized embedding sum, then measure its norm.

Forward pass (per position j):
    ln_i = (e_i - E[e_i]) / std(e_i)  (LayerNorm)
    v_i = ln_i @ W_v                   (Value projection, per head)
    attn_out_j = Σ_{i=1}^j v_i @ W_o   (Causal sum + output projection)

Inverse:
    W_v^{-1} @ W_o^{-1} @ attn_out_j = Σ_{i=1}^j ln_i

The norm of this sum should decrease with position (more averaging).

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis_scripts/decoding_vector_v2.py --quick
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

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "decoding_vector_v2"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
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


def extract_projection_matrices(model: GPT) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract W_v and W_o matrices from the model.

    Returns:
        W_v: (768, 768) - V projection matrix
        W_o: (768, 768) - O projection matrix
    """
    block = model.transformer.h[0]

    # Extract W_V from c_attn (last 768 rows of 2304 total)
    c_attn_weight = block.attn.c_attn.weight.data  # (2304, 768)
    W_v = c_attn_weight[2 * D_MODEL :, :].T  # (768, 768)

    # Extract W_O from c_proj
    W_o = block.attn.c_proj.weight.data.T  # (768, 768)

    return W_v, W_o


def get_attention_outputs(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """
    Extract attention outputs (post_attn, before residual).

    Returns:
        attn_out: (B, T, D) attention outputs
    """
    with torch.no_grad():
        # Get embeddings
        embeddings = model.transformer.wte(tokens)  # (B, T, D)
        x = embeddings.clone()

        # Get the block
        block = model.transformer.h[0]

        # Post-LN1
        x_ln1 = block.ln_1(x)

        # Post-Attention output (BEFORE residual)
        attn_out = block.attn(x_ln1)

    return attn_out


def invert_attention(
    attn_out: torch.Tensor, W_v: torch.Tensor, W_o: torch.Tensor
) -> torch.Tensor:
    """
    Invert the attention transformation to recover the normalized embedding sum.

    Forward: Σ ln_i @ W_v @ W_o = attn_out
    Inverse: W_v^{-1} @ W_o^{-1} @ attn_out = Σ ln_i

    Args:
        attn_out: (B, T, D) attention outputs
        W_v: (D, D) value projection
        W_o: (D, D) output projection

    Returns:
        inverted: (B, T, D) inverted activations
    """
    # Compute pseudo-inverses
    W_v_pinv = torch.linalg.pinv(W_v)  # (D, D)
    W_o_pinv = torch.linalg.pinv(W_o)  # (D, D)

    # Apply inverses: W_V^{-1} @ W_O^{-1} @ attn_out
    # First: W_O^{-1} @ attn_out
    B, T, D = attn_out.shape
    attn_flat = attn_out.reshape(B * T, D)  # (B*T, D)

    step1 = (W_o_pinv @ attn_flat.T).T  # (D, B*T) -> (B*T, D)
    step2 = (W_v_pinv @ step1.T).T  # (D, B*T) -> (B*T, D)

    inverted = step2.reshape(B, T, D)  # (B, T, D)

    return inverted


def analyze_position_correlation(values: np.ndarray, positions: np.ndarray) -> Dict:
    """
    Analyze correlation between values and positions.

    Args:
        values: (N,) values
        positions: (N,) position indices

    Returns:
        results: Dict with correlation metrics
    """
    # Overall correlation
    pearson_r, pearson_p = pearsonr(values, positions)
    spearman_r, spearman_p = spearmanr(values, positions)

    # R² from linear regression
    X = values.reshape(-1, 1)
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


def create_scatter_plot(
    values: np.ndarray,
    positions: np.ndarray,
    title: str,
    ylabel: str,
    n_samples_plot: int = 5000,
) -> plt.Figure:
    """Create scatter plot of values vs position."""
    # Subsample for plotting
    if len(positions) > n_samples_plot:
        idx = np.random.choice(len(positions), n_samples_plot, replace=False)
        values_sub = values[idx]
        positions_sub = positions[idx]
    else:
        values_sub = values
        positions_sub = positions

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by position
    scatter = ax.scatter(
        positions_sub, values_sub, c=positions_sub, cmap="viridis", alpha=0.5, s=5
    )

    # Add regression line
    z = np.polyfit(positions_sub, values_sub, 1)
    p = np.poly1d(z)
    x_line = np.linspace(positions_sub.min(), positions_sub.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Linear fit")

    # Compute correlation for display
    r, _ = pearsonr(values_sub, positions_sub)

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\nPearson r = {r:.4f}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label="Position")
    plt.tight_layout()

    return fig


def run_analysis(
    model_name: str,
    model: GPT,
    tokens: torch.Tensor,
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
    }

    # Extract projection matrices
    print("\n[1/3] Extracting W_v and W_o matrices...")
    W_v, W_o = extract_projection_matrices(model)
    print(f"    W_v shape: {W_v.shape}")
    print(f"    W_o shape: {W_o.shape}")

    # Extract attention outputs and invert
    print("\n[2/3] Extracting and inverting attention outputs...")
    all_norms = []
    all_positions = []

    batch_size = 10
    for batch_start in tqdm(
        range(0, n_sequences, batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, n_sequences)
        batch_tokens = tokens[batch_start:batch_end]

        # Get attention outputs
        attn_out = get_attention_outputs(model, batch_tokens)

        # Invert to get normalized embedding sums
        inverted = invert_attention(attn_out, W_v, W_o)

        # Compute norms
        norms = torch.norm(inverted, dim=2)  # (B, T)
        all_norms.append(norms.cpu().numpy())

        # Track positions
        B = batch_end - batch_start
        for _ in range(B):
            all_positions.extend(range(context_length))

    # Concatenate all results
    all_norms = np.concatenate(all_norms).flatten()
    positions = np.array(all_positions)

    # Analyze correlations
    print("\n[3/3] Analyzing correlations...")
    metrics = analyze_position_correlation(all_norms, positions)
    results.update(metrics)

    print(f"    Pearson r: {metrics['pearson_r']:.4f}")
    print(f"    Spearman r: {metrics['spearman_r']:.4f}")
    print(f"    R² score: {metrics['r2_score']:.4f}")

    # Create visualization
    fig = create_scatter_plot(
        all_norms,
        positions,
        f"{model_name} - Inverted Norm vs Position",
        "Norm of Inverted Activation",
    )
    plot_path = PLOTS_DIR / f"scatter_{model_name}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser(description="Decoding Vector Analysis V2")
    parser.add_argument(
        "--n_sequences", type=int, default=100, help="Number of sequences"
    )
    parser.add_argument(
        "--context_length", type=int, default=256, help="Context length"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with fewer samples"
    )
    args = parser.parse_args()

    if args.quick:
        args.n_sequences = 30

    print("=" * 70)
    print("DECODING VECTOR ANALYSIS V2 - Direct Inversion")
    print("=" * 70)
    print(f"  Sequences: {args.n_sequences}")
    print(f"  Context length: {args.context_length}")
    print(f"  Device: {DEVICE}")

    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load OWT data
    print("\nLoading OWT data...")
    owt_data = load_owt_data()
    tokens = get_owt_sequences(owt_data, args.n_sequences, args.context_length)
    print(f"  Loaded {args.n_sequences} sequences of length {args.context_length}")

    all_results = {}

    # NoPE + LayerNorm - Random Init
    print("\n" + "=" * 70)
    print("MODEL: NoPE + LayerNorm (Random Initialization)")
    print("=" * 70)

    random_model, _ = create_random_model(use_pe=False, block_size=args.context_length)
    results = run_analysis("NoPE_LN_random", random_model, tokens)
    all_results["NoPE_LN_random"] = results
    del random_model
    torch.cuda.empty_cache()

    # Baseline + PE - Random Init (for comparison)
    print("\n" + "=" * 70)
    print("MODEL: Baseline + PE (Random Initialization)")
    print("=" * 70)

    baseline_model, _ = create_random_model(use_pe=True, block_size=args.context_length)
    results = run_analysis("Baseline_PE_random", baseline_model, tokens)
    all_results["Baseline_PE_random"] = results
    del baseline_model
    torch.cuda.empty_cache()

    # Save results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Pearson r':<12} {'R²':<10}")
    print("-" * 50)

    for model_name, model_results in all_results.items():
        print(
            f"{model_name:<25} {model_results['pearson_r']:>10.4f} {model_results['r2_score']:>10.4f}"
        )

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
