"""
Token-Position Correlation Analysis in Natural Language

This script analyzes how token distributions vary with position in natural language,
providing evidence for why NoPE transformers can leverage position-correlated token statistics.

Experiments:
- T1.1: Token Frequency by Position - P(token | position) heatmap
- T1.2: Token Entropy by Position - H(token | position) vs position
- T1.3: Token Transition Probabilities - P(token_i | token_{i-1}, position)
- T1.4: Top-k Token Concentration - coverage by top-k tokens per position
- T1.5: Position-Specific Vocabulary - tokens overrepresented at specific positions

Usage:
    python token_position_correlation_natural_language.py --dataset wikitext --max_seq_len 256
"""

import os

# Only set GPU if not already specified by environment
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from scipy.stats import entropy
from scipy.special import rel_entr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/token_position_correlation")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
HF_CACHE_DIR = "/home/nlp/matan_avitan/cache_dir"


def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_tokenize_dataset(
    dataset_name: str, split: str, tokenizer, max_seq_len: int, max_samples: int = None
):
    """Load dataset and tokenize into fixed-length sequences."""
    print(f"Loading {dataset_name} ({split})...")

    if dataset_name == "wikitext":
        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split=split, cache_dir=HF_CACHE_DIR
        )
        text_column = "text"
    elif dataset_name == "wikitext2":
        ds = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split=split, cache_dir=HF_CACHE_DIR
        )
        text_column = "text"
    elif dataset_name == "openwebtext":
        ds = load_dataset("openwebtext", split="train", cache_dir=HF_CACHE_DIR)
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Concatenate all text and tokenize
    print("Tokenizing...")
    all_text = " ".join([ex[text_column] for ex in ds if ex[text_column].strip()])
    tokens = tokenizer.encode(all_text, add_special_tokens=False)

    # Split into fixed-length sequences
    n_seqs = len(tokens) // max_seq_len
    if max_samples:
        n_seqs = min(n_seqs, max_samples)

    tokens = tokens[: n_seqs * max_seq_len]
    sequences = np.array(tokens).reshape(n_seqs, max_seq_len)

    print(f"Created {n_seqs} sequences of length {max_seq_len}")
    return sequences


def compute_token_frequency_by_position(sequences: np.ndarray, vocab_size: int):
    """
    T1.1: Compute P(token | position) for each position.

    Returns:
        freq_matrix: [seq_len, vocab_size] matrix of P(token | position)
        counts_matrix: [seq_len, vocab_size] raw counts
    """
    n_seqs, seq_len = sequences.shape
    counts_matrix = np.zeros((seq_len, vocab_size), dtype=np.int64)

    print("Computing token frequency by position...")
    for pos in tqdm(range(seq_len)):
        for token in sequences[:, pos]:
            counts_matrix[pos, token] += 1

    # Normalize to get probabilities
    freq_matrix = counts_matrix / counts_matrix.sum(axis=1, keepdims=True)

    return freq_matrix, counts_matrix


def compute_entropy_by_position(freq_matrix: np.ndarray):
    """
    T1.2: Compute H(token | position) for each position.

    Returns:
        entropies: [seq_len] array of entropy values
    """
    print("Computing entropy by position...")
    entropies = np.array(
        [entropy(freq_matrix[pos]) for pos in range(freq_matrix.shape[0])]
    )
    return entropies


def compute_transition_probabilities(
    sequences: np.ndarray, vocab_size: int, top_k: int = 100
):
    """
    T1.3: Compute P(token_i | token_{i-1}, position) for top-k tokens.

    Returns:
        transition_entropy: [seq_len-1] entropy of transition distributions by position
    """
    n_seqs, seq_len = sequences.shape

    print("Computing transition probabilities...")
    # For each position, compute entropy of P(token | prev_token)
    transition_entropies = []

    for pos in tqdm(range(1, seq_len)):
        # Count (prev_token, curr_token) pairs at this position
        trans_counts = defaultdict(Counter)
        for seq in sequences:
            prev_token = seq[pos - 1]
            curr_token = seq[pos]
            trans_counts[prev_token][curr_token] += 1

        # Compute average conditional entropy H(curr | prev)
        total_entropy = 0
        total_weight = 0
        for prev_token, curr_counts in trans_counts.items():
            total = sum(curr_counts.values())
            probs = np.array(list(curr_counts.values())) / total
            total_entropy += total * entropy(probs)
            total_weight += total

        transition_entropies.append(
            total_entropy / total_weight if total_weight > 0 else 0
        )

    return np.array(transition_entropies)


def compute_topk_concentration(
    freq_matrix: np.ndarray, k_values: list = [10, 50, 100, 500]
):
    """
    T1.4: Compute fraction of tokens covered by top-k most common tokens per position.

    Returns:
        concentration: dict mapping k -> [seq_len] coverage array
    """
    print("Computing top-k concentration...")
    seq_len = freq_matrix.shape[0]
    concentration = {}

    for k in k_values:
        coverage = np.array(
            [np.sort(freq_matrix[pos])[-k:].sum() for pos in range(seq_len)]
        )
        concentration[k] = coverage

    return concentration


def find_position_specific_tokens(
    freq_matrix: np.ndarray, tokenizer, n_positions: int = 10, n_tokens: int = 20
):
    """
    T1.5: Find tokens that are statistically overrepresented at specific positions.

    Uses log-likelihood ratio compared to marginal distribution.

    Returns:
        position_tokens: dict mapping position -> list of (token, token_str, llr_score)
    """
    print("Finding position-specific tokens...")
    seq_len, vocab_size = freq_matrix.shape

    # Compute marginal token distribution (average across positions)
    marginal = freq_matrix.mean(axis=0)

    # For each position, compute log-likelihood ratio vs marginal
    position_tokens = {}

    # Focus on specific positions of interest
    positions_of_interest = [0, 1, 2, 3, 4] + list(
        range(seq_len - 5, seq_len)
    )  # Start and end
    positions_of_interest = [p for p in positions_of_interest if p < seq_len]

    for pos in positions_of_interest:
        # Compute KL divergence contribution for each token
        # D_KL = sum_t P(t|pos) * log(P(t|pos) / P(t))
        pos_dist = freq_matrix[pos]

        # Log-likelihood ratio: log(P(t|pos) / P(t))
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = np.log(pos_dist / (marginal + 1e-10) + 1e-10)
            llr = np.nan_to_num(llr, nan=0, posinf=0, neginf=0)

        # Get top tokens by LLR (overrepresented at this position)
        top_indices = np.argsort(llr)[-n_tokens:][::-1]

        position_tokens[pos] = [
            (int(idx), tokenizer.decode([idx]), float(llr[idx]), float(pos_dist[idx]))
            for idx in top_indices
            if pos_dist[idx] > 1e-5  # Only significant tokens
        ]

    return position_tokens


def compute_position_distinguishability(freq_matrix: np.ndarray):
    """
    Compute how distinguishable each position is from others using KL divergence.

    Returns:
        kl_matrix: [seq_len, seq_len] pairwise KL divergences
        avg_kl: [seq_len] average KL from each position to others
    """
    print("Computing position distinguishability...")
    seq_len = freq_matrix.shape[0]
    kl_matrix = np.zeros((seq_len, seq_len))

    for i in tqdm(range(seq_len)):
        for j in range(seq_len):
            if i != j:
                # KL(P_i || P_j)
                kl_matrix[i, j] = np.sum(
                    rel_entr(freq_matrix[i] + 1e-10, freq_matrix[j] + 1e-10)
                )

    avg_kl = kl_matrix.sum(axis=1) / (seq_len - 1)

    return kl_matrix, avg_kl


def plot_token_frequency_heatmap(
    freq_matrix: np.ndarray, tokenizer, save_path: str, top_k: int = 100
):
    """Plot heatmap of P(token | position) for top-k tokens."""
    # Get top-k most common tokens overall
    marginal = freq_matrix.mean(axis=0)
    top_tokens = np.argsort(marginal)[-top_k:][::-1]

    # Extract submatrix
    submatrix = freq_matrix[:, top_tokens].T  # [top_k, seq_len]

    # Get token labels
    token_labels = [tokenizer.decode([t]).replace("\n", "\\n")[:15] for t in top_tokens]

    fig = px.imshow(
        submatrix,
        labels=dict(x="Position", y="Token", color="P(token|position)"),
        x=list(range(freq_matrix.shape[0])),
        y=token_labels,
        color_continuous_scale="Blues",
        aspect="auto",
        template="plotly_white",
    )

    fig.update_layout(
        title=dict(
            text=f"Token Distribution by Position (Top {top_k} tokens)",
            font=dict(size=20, family="Serif"),
        ),
        width=1200,
        height=800,
        margin=dict(l=100, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=1200, height=800, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_entropy_by_position(entropies: np.ndarray, save_path: str):
    """Plot entropy H(token | position) vs position."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(entropies))),
            y=entropies,
            mode="lines",
            line=dict(width=2, color="blue"),
            name="Entropy",
        )
    )

    fig.update_layout(
        title=dict(
            text="Token Entropy by Position in Natural Language",
            font=dict(size=20, family="Serif"),
        ),
        xaxis=dict(title="Position", title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(
            title="H(token | position)",
            title_font=dict(size=16),
            tickfont=dict(size=14),
        ),
        width=800,
        height=500,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_topk_concentration(concentration: dict, save_path: str):
    """Plot top-k token concentration by position."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, (k, coverage) in enumerate(concentration.items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(coverage))),
                y=coverage,
                mode="lines",
                line=dict(width=2, color=colors[i % len(colors)]),
                name=f"Top-{k}",
            )
        )

    fig.update_layout(
        title=dict(
            text="Token Concentration by Position", font=dict(size=20, family="Serif")
        ),
        xaxis=dict(title="Position", title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(
            title="Cumulative Probability",
            title_font=dict(size=16),
            tickfont=dict(size=14),
        ),
        width=800,
        height=500,
        template="plotly_white",
        legend=dict(x=0.8, y=0.2),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_kl_divergence_heatmap(kl_matrix: np.ndarray, save_path: str):
    """Plot pairwise KL divergence between positions."""
    fig = px.imshow(
        kl_matrix,
        labels=dict(x="Position j", y="Position i", color="KL(P_i || P_j)"),
        color_continuous_scale="Viridis",
        aspect="equal",
        template="plotly_white",
    )

    fig.update_layout(
        title=dict(
            text="Position Distinguishability (KL Divergence)",
            font=dict(size=20, family="Serif"),
        ),
        width=700,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=700, height=600, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def plot_transition_entropy(transition_entropies: np.ndarray, save_path: str):
    """Plot transition entropy by position."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(transition_entropies) + 1)),
            y=transition_entropies,
            mode="lines",
            line=dict(width=2, color="green"),
            name="Transition Entropy",
        )
    )

    fig.update_layout(
        title=dict(
            text="Token Transition Entropy by Position",
            font=dict(size=20, family="Serif"),
        ),
        xaxis=dict(title="Position", title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(
            title="H(token_i | token_{i-1})",
            title_font=dict(size=16),
            tickfont=dict(size=14),
        ),
        width=800,
        height=500,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def create_summary_figure(
    entropies, concentration, avg_kl, transition_entropies, save_path: str
):
    """Create a 2x2 summary figure with all main results."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "A) Token Entropy by Position",
            "B) Top-k Token Concentration",
            "C) Position Distinguishability (Avg KL)",
            "D) Transition Entropy",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # A) Entropy
    fig.add_trace(
        go.Scatter(
            x=list(range(len(entropies))),
            y=entropies,
            mode="lines",
            line=dict(width=2, color="blue"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # B) Top-k concentration
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (k, coverage) in enumerate(concentration.items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(coverage))),
                y=coverage,
                mode="lines",
                line=dict(width=2, color=colors[i % len(colors)]),
                name=f"Top-{k}",
            ),
            row=1,
            col=2,
        )

    # C) Avg KL divergence
    fig.add_trace(
        go.Scatter(
            x=list(range(len(avg_kl))),
            y=avg_kl,
            mode="lines",
            line=dict(width=2, color="purple"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # D) Transition entropy
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(transition_entropies) + 1)),
            y=transition_entropies,
            mode="lines",
            line=dict(width=2, color="green"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Position", row=1, col=1)
    fig.update_xaxes(title_text="Position", row=1, col=2)
    fig.update_xaxes(title_text="Position", row=2, col=1)
    fig.update_xaxes(title_text="Position", row=2, col=2)

    fig.update_yaxes(title_text="Entropy (nats)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Prob.", row=1, col=2)
    fig.update_yaxes(title_text="Avg KL Divergence", row=2, col=1)
    fig.update_yaxes(title_text="Transition Entropy", row=2, col=2)

    fig.update_layout(
        title=dict(
            text="Token-Position Correlation in Natural Language (WikiText-103)",
            font=dict(size=22, family="Serif"),
        ),
        width=1200,
        height=900,
        template="plotly_white",
        legend=dict(x=0.85, y=0.95),
        margin=dict(l=60, r=50, t=100, b=50),
    )

    fig.write_image(f"{save_path}.png", width=1200, height=900, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}.png and .pdf")


def save_results(results: dict, save_path: str):
    """Save numerical results to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable[key] = value

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Token-Position Correlation Analysis")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "wikitext2", "openwebtext"],
    )
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=50000)
    args = parser.parse_args()

    setup_dirs()

    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    # Load and tokenize dataset
    sequences = load_and_tokenize_dataset(
        args.dataset, "train", tokenizer, args.max_seq_len, args.max_samples
    )

    print(f"\n{'=' * 60}")
    print("RUNNING TOKEN-POSITION CORRELATION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Sequences: {sequences.shape[0]}")
    print(f"Sequence length: {sequences.shape[1]}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"{'=' * 60}\n")

    # T1.1: Token frequency by position
    print("\n[T1.1] Computing token frequency by position...")
    freq_matrix, counts_matrix = compute_token_frequency_by_position(
        sequences, vocab_size
    )

    # T1.2: Entropy by position
    print("\n[T1.2] Computing entropy by position...")
    entropies = compute_entropy_by_position(freq_matrix)

    # T1.3: Transition probabilities
    print("\n[T1.3] Computing transition probabilities...")
    transition_entropies = compute_transition_probabilities(sequences, vocab_size)

    # T1.4: Top-k concentration
    print("\n[T1.4] Computing top-k concentration...")
    concentration = compute_topk_concentration(freq_matrix)

    # T1.5: Position-specific tokens
    print("\n[T1.5] Finding position-specific tokens...")
    position_tokens = find_position_specific_tokens(freq_matrix, tokenizer)

    # Position distinguishability
    print("\n[Extra] Computing position distinguishability...")
    kl_matrix, avg_kl = compute_position_distinguishability(freq_matrix)

    # ─── Print Summary Statistics ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    print(f"Entropy range: {entropies.min():.2f} - {entropies.max():.2f} nats")
    print(f"Entropy mean: {entropies.mean():.2f} nats")
    print(f"Entropy std: {entropies.std():.2f} nats")
    print(f"Avg KL divergence: {avg_kl.mean():.4f}")
    print(f"Max KL divergence: {kl_matrix.max():.4f}")

    print("\nTop-k concentration at position 0 vs position 128:")
    for k, coverage in concentration.items():
        print(f"  Top-{k}: pos 0 = {coverage[0]:.3f}, pos 128 = {coverage[128]:.3f}")

    print("\nPosition-specific tokens (examples):")
    for pos in [0, 1, args.max_seq_len - 1]:
        if pos in position_tokens:
            print(f"\n  Position {pos}:")
            for token_id, token_str, llr, prob in position_tokens[pos][:5]:
                print(f"    '{token_str}' (LLR={llr:.2f}, P={prob:.4f})")

    # ─── Generate Plots ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("GENERATING PLOTS")
    print(f"{'=' * 60}")

    # Individual plots
    plot_token_frequency_heatmap(
        freq_matrix, tokenizer, str(RESULTS_DIR / "token_frequency_heatmap")
    )
    plot_entropy_by_position(entropies, str(RESULTS_DIR / "entropy_by_position"))
    plot_topk_concentration(concentration, str(RESULTS_DIR / "topk_concentration"))
    plot_kl_divergence_heatmap(kl_matrix, str(RESULTS_DIR / "kl_divergence_heatmap"))
    plot_transition_entropy(
        transition_entropies, str(RESULTS_DIR / "transition_entropy")
    )

    # Summary figure for paper
    create_summary_figure(
        entropies,
        concentration,
        avg_kl,
        transition_entropies,
        str(PLOTS_DIR / "token_position_correlation_summary"),
    )

    # ─── Save Results ───────────────────────────────────────────────────────────
    results = {
        "dataset": args.dataset,
        "n_sequences": int(sequences.shape[0]),
        "seq_len": int(sequences.shape[1]),
        "vocab_size": vocab_size,
        "entropies": entropies,
        "entropy_stats": {
            "min": float(entropies.min()),
            "max": float(entropies.max()),
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
        },
        "concentration": {str(k): v for k, v in concentration.items()},
        "avg_kl_divergence": avg_kl,
        "transition_entropies": transition_entropies,
        "position_specific_tokens": {str(k): v for k, v in position_tokens.items()},
    }
    save_results(results, str(RESULTS_DIR / "token_position_correlation_results.json"))

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Paper figure saved to: {PLOTS_DIR}/token_position_correlation_summary.png")


if __name__ == "__main__":
    main()
