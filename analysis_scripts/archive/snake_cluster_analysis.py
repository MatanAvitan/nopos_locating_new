"""
24-Snake Cluster Analysis for NoPE Transformers

This script analyzes the 24 snake-like clusters visible in t-SNE visualizations
of post-attention and post-LN2 activations in NoPE transformers.

Tasks:
1. Reproduce 24-snake t-SNE visualization using OWT text
2. Extract per-cluster statistics (mean, std, norm of 768-dim activations)
3. Sequence coherence analysis (do positions from same sequence stay in same snake?)
4. Within-cluster linear probe (is position linearly decodable within each snake?)

Layers analyzed:
- post_attn: After attention, before residual add
- post_ln2: After LN2, before MLP

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis_scripts/snake_cluster_analysis.py \
        --n_sequences 100 --context_length 256 --wandb

Author: Research Assistant
Date: January 2026
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
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
RESULTS_DIR = PROJECT_ROOT / "results" / "snake_cluster_analysis"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
OWT_DATA_PATH = PROJECT_ROOT / "nanoGPT" / "data" / "openwebtext" / "train.bin"

# Layers to analyze (per user request)
LAYERS = ["post_attn", "post_ln2"]

LAYER_NAMES = {
    "post_attn": "Post-Attention (Before Residual)",
    "post_ln2": "Post-LN2 (Before MLP)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def create_random_model(
    use_pe: bool = False, block_size: int = 256
) -> Tuple[GPT, GPTConfig]:
    """Create a randomly initialized NoPE model."""
    config = GPTConfig(
        n_layer=1,
        n_head=12,
        n_embd=768,
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
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
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
# ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def get_layer_activations(
    model: GPT, tokens: torch.Tensor, layer_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Extract activations at post_attn and post_ln2 layers.

    Architecture flow:
        x -> LN1 -> Attention -> (+x) -> LN2 -> MLP -> (+x)
                    ^                    ^
                    post_attn            post_ln2
    """
    activations = {}

    with torch.no_grad():
        # Embedding
        tok_emb = model.transformer.wte(tokens)
        if hasattr(model.transformer, "wpe") and model.config.use_positional_embedding:
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        # Get the specified block
        block = model.transformer.h[layer_idx]

        # Post-LN1
        x_ln1 = block.ln_1(x)

        # Post-Attention output (BEFORE adding residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # After first residual connection
        x = x + attn_out

        # Post-LN2 (BEFORE MLP)
        if hasattr(block, "ln_2") and not getattr(block, "skip_ln2", False):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
        else:
            activations["post_ln2"] = x.clone()

    return activations


def collect_all_activations(
    model: GPT, tokens: torch.Tensor, batch_size: int = 10
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Collect activations for all sequences.

    Returns:
        activations: Dict[layer_name, (N*T, D) array]
        positions: (N*T,) array of position indices
        sequence_ids: (N*T,) array of sequence indices
    """
    n_sequences, context_length = tokens.shape
    all_activations = {layer: [] for layer in LAYERS}
    all_positions = []
    all_seq_ids = []

    for batch_start in tqdm(
        range(0, n_sequences, batch_size), desc="Extracting activations"
    ):
        batch_end = min(batch_start + batch_size, n_sequences)
        batch_tokens = tokens[batch_start:batch_end]

        acts = get_layer_activations(model, batch_tokens)

        for layer in LAYERS:
            if layer in acts:
                # Shape: (batch, T, D) -> flatten to (batch*T, D)
                layer_acts = acts[layer].cpu().numpy()
                batch_size_actual = layer_acts.shape[0]
                layer_acts = layer_acts.reshape(-1, layer_acts.shape[-1])
                all_activations[layer].append(layer_acts)

        # Track positions and sequence IDs
        for seq_idx in range(batch_start, batch_end):
            all_positions.extend(range(context_length))
            all_seq_ids.extend([seq_idx] * context_length)

    # Stack all batches
    for layer in LAYERS:
        if all_activations[layer]:
            all_activations[layer] = np.vstack(all_activations[layer])

    positions = np.array(all_positions)
    sequence_ids = np.array(all_seq_ids)

    return all_activations, positions, sequence_ids


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: T-SNE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def create_tsne_visualization(
    activations: np.ndarray,
    positions: np.ndarray,
    title: str,
    n_samples_tsne: int = 5000,
    perplexity: int = 30,
    n_position_groups: int = 8,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create t-SNE visualization colored by position groups.

    Returns:
        fig: matplotlib Figure
        embeddings: (n_samples_tsne, 2) t-SNE coordinates
    """
    # Subsample if needed
    if len(positions) > n_samples_tsne:
        idx = np.random.choice(len(positions), n_samples_tsne, replace=False)
        activations_sub = activations[idx]
        positions_sub = positions[idx]
    else:
        activations_sub = activations
        positions_sub = positions
        idx = np.arange(len(positions))

    # Bin positions for coloring
    max_pos = positions_sub.max() + 1
    group_size = max_pos / n_position_groups
    position_groups = np.clip(
        (positions_sub / group_size).astype(int), 0, n_position_groups - 1
    )

    # Run t-SNE
    print(f"    Running t-SNE on {len(positions_sub)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    embeddings = tsne.fit_transform(activations_sub)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color map
    cmap = plt.cm.get_cmap("viridis", n_position_groups)

    # Plot each position group
    for group in range(n_position_groups):
        mask = position_groups == group
        if mask.sum() > 0:
            start_pos = int(group * group_size)
            end_pos = int((group + 1) * group_size) - 1
            label = f"pos {start_pos}-{end_pos}"
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[cmap(group)],
                label=label,
                alpha=0.6,
                s=8,
            )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: CLUSTER IDENTIFICATION AND STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════


def identify_clusters(
    embeddings_2d: np.ndarray, n_clusters: int = 24, method: str = "kmeans"
) -> np.ndarray:
    """
    Identify clusters in 2D t-SNE space.

    Args:
        embeddings_2d: (N, 2) t-SNE embeddings
        n_clusters: Number of clusters (for K-Means)
        method: "kmeans" or "dbscan"

    Returns:
        cluster_labels: (N,) cluster assignments
    """
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_2d)
    elif method == "dbscan":
        # Estimate eps based on data spread
        std = embeddings_2d.std()
        dbscan = DBSCAN(eps=std * 0.15, min_samples=10)
        labels = dbscan.fit_predict(embeddings_2d)
    else:
        raise ValueError(f"Unknown method: {method}")

    return labels


def compute_cluster_statistics(
    activations: np.ndarray,
    cluster_labels: np.ndarray,
    positions: np.ndarray,
) -> Dict[int, Dict]:
    """
    Compute statistics for each cluster.

    For each cluster, compute:
    - Mean vector of original 768-dim activations
    - Std of original activations (per-dimension and overall)
    - Mean norm of original activations
    - Position distribution (which positions fall into each cluster)
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_stats = {}

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise cluster from DBSCAN
            continue

        mask = cluster_labels == cluster_id
        cluster_acts = activations[mask]
        cluster_pos = positions[mask]

        # Compute statistics on original 768-dim activations
        mean_vector = cluster_acts.mean(axis=0)
        std_vector = cluster_acts.std(axis=0)
        overall_std = cluster_acts.std()

        # Norms
        norms = np.linalg.norm(cluster_acts, axis=1)
        mean_norm = norms.mean()
        std_norm = norms.std()

        # Position distribution
        pos_counts = np.bincount(cluster_pos, minlength=positions.max() + 1)
        dominant_positions = np.argsort(pos_counts)[-5:][::-1]  # Top 5 positions

        # Position range
        min_pos = cluster_pos.min()
        max_pos = cluster_pos.max()
        mean_pos = cluster_pos.mean()

        cluster_stats[int(cluster_id)] = {
            "n_samples": int(mask.sum()),
            "mean_vector": mean_vector.tolist(),  # For JSON serialization
            "std_vector_mean": float(std_vector.mean()),
            "std_vector_std": float(std_vector.std()),
            "overall_std": float(overall_std),
            "mean_norm": float(mean_norm),
            "std_norm": float(std_norm),
            "min_position": int(min_pos),
            "max_position": int(max_pos),
            "mean_position": float(mean_pos),
            "dominant_positions": dominant_positions.tolist(),
            "position_range": int(max_pos - min_pos),
        }

    return cluster_stats


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: SEQUENCE COHERENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_sequence_coherence(
    cluster_labels: np.ndarray,
    sequence_ids: np.ndarray,
    positions: np.ndarray,
) -> Dict:
    """
    Analyze if positions from the same sequence stay in the same cluster.

    For each sequence:
    - Count how many unique clusters its positions span
    - Compute the "coherence" score (1 = all positions in same cluster)

    Returns:
        coherence_results: Dict with per-sequence and aggregate stats
    """
    unique_sequences = np.unique(sequence_ids)

    per_sequence_stats = []
    for seq_id in unique_sequences:
        mask = sequence_ids == seq_id
        seq_clusters = cluster_labels[mask]
        seq_positions = positions[mask]

        # Remove noise labels (-1)
        valid_mask = seq_clusters != -1
        seq_clusters_valid = seq_clusters[valid_mask]

        if len(seq_clusters_valid) == 0:
            continue

        unique_clusters = np.unique(seq_clusters_valid)
        n_clusters_used = len(unique_clusters)

        # Dominant cluster (most common)
        cluster_counts = np.bincount(seq_clusters_valid)
        dominant_cluster = np.argmax(cluster_counts)
        dominant_fraction = cluster_counts[dominant_cluster] / len(seq_clusters_valid)

        per_sequence_stats.append(
            {
                "sequence_id": int(seq_id),
                "n_clusters_used": int(n_clusters_used),
                "dominant_cluster": int(dominant_cluster),
                "dominant_fraction": float(dominant_fraction),
                "n_positions": int(len(seq_clusters_valid)),
            }
        )

    # Aggregate statistics
    n_clusters_list = [s["n_clusters_used"] for s in per_sequence_stats]
    dominant_fractions = [s["dominant_fraction"] for s in per_sequence_stats]

    return {
        "per_sequence": per_sequence_stats,
        "aggregate": {
            "mean_clusters_per_sequence": float(np.mean(n_clusters_list)),
            "std_clusters_per_sequence": float(np.std(n_clusters_list)),
            "median_clusters_per_sequence": float(np.median(n_clusters_list)),
            "min_clusters_per_sequence": int(np.min(n_clusters_list)),
            "max_clusters_per_sequence": int(np.max(n_clusters_list)),
            "mean_dominant_fraction": float(np.mean(dominant_fractions)),
            "std_dominant_fraction": float(np.std(dominant_fractions)),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4: WITHIN-CLUSTER LINEAR PROBE
# ═══════════════════════════════════════════════════════════════════════════════


def train_within_cluster_probes(
    activations: np.ndarray,
    cluster_labels: np.ndarray,
    positions: np.ndarray,
    min_samples_per_cluster: int = 50,
) -> Dict[int, Dict]:
    """
    Train linear probes to predict position within each cluster.

    For each cluster:
    - Train a Ridge regression on the 768-dim activations to predict position
    - Compute R² score
    - If position is linearly decodable within each snake, R² should be high
    """
    unique_clusters = np.unique(cluster_labels)
    probe_results = {}

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue

        mask = cluster_labels == cluster_id
        cluster_acts = activations[mask]
        cluster_pos = positions[mask]

        if len(cluster_pos) < min_samples_per_cluster:
            probe_results[int(cluster_id)] = {
                "status": "skipped",
                "reason": f"Too few samples ({len(cluster_pos)} < {min_samples_per_cluster})",
                "n_samples": int(len(cluster_pos)),
            }
            continue

        # Split into train/test
        n_samples = len(cluster_pos)
        n_train = int(0.8 * n_samples)

        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        X_train, X_test = cluster_acts[train_idx], cluster_acts[test_idx]
        y_train, y_test = cluster_pos[train_idx], cluster_pos[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)

        # Predict and evaluate
        y_pred = ridge.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)

        # Correlation
        corr, _ = pearsonr(y_test, y_pred)

        # MAE
        mae = np.abs(y_test - y_pred).mean()

        probe_results[int(cluster_id)] = {
            "status": "success",
            "n_samples": int(len(cluster_pos)),
            "n_train": int(n_train),
            "n_test": int(n_samples - n_train),
            "r2_score": float(r2),
            "correlation": float(corr),
            "mae": float(mae),
            "position_range": [int(cluster_pos.min()), int(cluster_pos.max())],
        }

    return probe_results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def create_cluster_statistics_figure(
    cluster_stats: Dict[int, Dict],
    title: str,
) -> plt.Figure:
    """Create a summary figure of cluster statistics."""
    clusters = sorted(cluster_stats.keys())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Samples per cluster
    ax = axes[0, 0]
    n_samples = [cluster_stats[c]["n_samples"] for c in clusters]
    ax.bar(range(n_clusters), n_samples, color="steelblue", alpha=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Samples per Cluster")
    ax.set_xticks(range(0, n_clusters, max(1, n_clusters // 10)))

    # 2. Mean position per cluster
    ax = axes[0, 1]
    mean_pos = [cluster_stats[c]["mean_position"] for c in clusters]
    colors = plt.cm.viridis(np.array(mean_pos) / max(mean_pos))
    ax.bar(range(n_clusters), mean_pos, color=colors, alpha=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean Position")
    ax.set_title("Mean Position per Cluster")
    ax.set_xticks(range(0, n_clusters, max(1, n_clusters // 10)))

    # 3. Mean norm per cluster
    ax = axes[1, 0]
    mean_norms = [cluster_stats[c]["mean_norm"] for c in clusters]
    ax.bar(range(n_clusters), mean_norms, color="coral", alpha=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean Activation Norm")
    ax.set_title("Mean Norm per Cluster")
    ax.set_xticks(range(0, n_clusters, max(1, n_clusters // 10)))

    # 4. Position range per cluster
    ax = axes[1, 1]
    pos_ranges = [cluster_stats[c]["position_range"] for c in clusters]
    ax.bar(range(n_clusters), pos_ranges, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Position Range (max - min)")
    ax.set_title("Position Spread per Cluster")
    ax.set_xticks(range(0, n_clusters, max(1, n_clusters // 10)))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def create_coherence_histogram(
    coherence_results: Dict,
    title: str,
) -> plt.Figure:
    """Create histogram of sequence coherence."""
    per_seq = coherence_results["per_sequence"]
    n_clusters_list = [s["n_clusters_used"] for s in per_seq]
    dominant_fractions = [s["dominant_fraction"] for s in per_seq]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Histogram of clusters per sequence
    ax = axes[0]
    ax.hist(
        n_clusters_list,
        bins=range(1, max(n_clusters_list) + 2),
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
    )
    ax.axvline(
        np.mean(n_clusters_list),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(n_clusters_list):.1f}",
    )
    ax.set_xlabel("Number of Clusters per Sequence")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Diversity per Sequence")
    ax.legend()

    # 2. Histogram of dominant cluster fraction
    ax = axes[1]
    ax.hist(dominant_fractions, bins=20, color="coral", alpha=0.8, edgecolor="black")
    ax.axvline(
        np.mean(dominant_fractions),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(dominant_fractions):.2f}",
    )
    ax.set_xlabel("Fraction in Dominant Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Sequence Concentration in Dominant Cluster")
    ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def create_probe_results_figure(
    probe_results: Dict[int, Dict],
    title: str,
) -> plt.Figure:
    """Create summary figure of within-cluster probe results."""
    # Filter successful probes
    successful = {
        k: v for k, v in probe_results.items() if v.get("status") == "success"
    }
    clusters = sorted(successful.keys())

    if len(clusters) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No successful probes", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. R² per cluster
    ax = axes[0]
    r2_scores = [successful[c]["r2_score"] for c in clusters]
    colors = [
        "green" if r2 > 0.5 else "orange" if r2 > 0.2 else "red" for r2 in r2_scores
    ]
    ax.bar(range(len(clusters)), r2_scores, color=colors, alpha=0.8)
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="R²=0.5")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("R² Score")
    ax.set_title("Position Decodability Within Clusters")
    ax.set_ylim(-0.1, 1.0)
    ax.legend()

    # 2. Correlation per cluster
    ax = axes[1]
    correlations = [successful[c]["correlation"] for c in clusters]
    ax.bar(range(len(clusters)), correlations, color="steelblue", alpha=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Correlation (pred vs true)")
    ax.set_title("Position Prediction Correlation")
    ax.set_ylim(-0.1, 1.0)

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
    n_clusters: int = 24,
) -> Dict:
    """
    Run full analysis pipeline for a single model.
    """
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(f"{'=' * 60}")

    results = {
        "model": model_name,
        "layers": {},
    }

    # Collect activations
    print("\n[1/4] Extracting activations...")
    activations, positions, sequence_ids = collect_all_activations(model, tokens)

    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")
        layer_acts = activations[layer]

        layer_results = {}

        # Task 1: t-SNE visualization
        print("[2/4] Creating t-SNE visualization...")
        fig_tsne, embeddings_2d = create_tsne_visualization(
            layer_acts,
            positions,
            title=f"{model_name} - {LAYER_NAMES[layer]}\nColored by Position Group",
            n_samples_tsne=5000,
        )

        # Save t-SNE plot
        tsne_path = (
            PLOTS_DIR
            / f"tsne_{model_name.replace(' ', '_').replace('+', '')}_{layer}.png"
        )
        fig_tsne.savefig(tsne_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {tsne_path}")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"tsne/{model_name}/{layer}": wandb.Image(fig_tsne)})
        plt.close(fig_tsne)

        # Identify clusters on t-SNE subsample
        print(f"[3/4] Identifying {n_clusters} clusters...")

        # For cluster analysis, we need to subsample to match t-SNE
        n_samples_cluster = min(5000, len(positions))
        cluster_idx = np.random.choice(len(positions), n_samples_cluster, replace=False)
        cluster_labels = identify_clusters(
            embeddings_2d, n_clusters=n_clusters, method="kmeans"
        )

        # Map cluster labels back to the subsample
        cluster_acts_sub = layer_acts[cluster_idx]
        cluster_pos_sub = positions[cluster_idx]
        cluster_seq_sub = sequence_ids[cluster_idx]

        # Task 2: Cluster statistics
        print("    Computing cluster statistics...")
        cluster_stats = compute_cluster_statistics(
            cluster_acts_sub, cluster_labels, cluster_pos_sub
        )
        layer_results["cluster_stats"] = {
            k: {
                kk: vv for kk, vv in v.items() if kk != "mean_vector"
            }  # Exclude large vectors for summary
            for k, v in cluster_stats.items()
        }

        # Save cluster stats figure
        fig_stats = create_cluster_statistics_figure(
            cluster_stats,
            title=f"{model_name} - {LAYER_NAMES[layer]} - Cluster Statistics",
        )
        stats_path = (
            PLOTS_DIR
            / f"cluster_stats_{model_name.replace(' ', '_').replace('+', '')}_{layer}.png"
        )
        fig_stats.savefig(stats_path, dpi=150, bbox_inches="tight")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"cluster_stats/{model_name}/{layer}": wandb.Image(fig_stats)})
        plt.close(fig_stats)

        # Task 3: Sequence coherence
        print("    Analyzing sequence coherence...")
        coherence_results = analyze_sequence_coherence(
            cluster_labels, cluster_seq_sub, cluster_pos_sub
        )
        layer_results["coherence"] = coherence_results["aggregate"]

        # Save coherence figure
        fig_coherence = create_coherence_histogram(
            coherence_results,
            title=f"{model_name} - {LAYER_NAMES[layer]} - Sequence Coherence",
        )
        coherence_path = (
            PLOTS_DIR
            / f"coherence_{model_name.replace(' ', '_').replace('+', '')}_{layer}.png"
        )
        fig_coherence.savefig(coherence_path, dpi=150, bbox_inches="tight")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"coherence/{model_name}/{layer}": wandb.Image(fig_coherence)})
        plt.close(fig_coherence)

        # Task 4: Within-cluster probes
        print("    Training within-cluster linear probes...")
        probe_results = train_within_cluster_probes(
            cluster_acts_sub, cluster_labels, cluster_pos_sub
        )

        # Summarize probe results
        successful_probes = [
            v for v in probe_results.values() if v.get("status") == "success"
        ]
        if successful_probes:
            mean_r2 = np.mean([p["r2_score"] for p in successful_probes])
            mean_corr = np.mean([p["correlation"] for p in successful_probes])
            layer_results["within_cluster_probes"] = {
                "n_clusters_probed": len(successful_probes),
                "mean_r2": float(mean_r2),
                "mean_correlation": float(mean_corr),
                "per_cluster": probe_results,
            }
        else:
            layer_results["within_cluster_probes"] = {
                "n_clusters_probed": 0,
                "mean_r2": 0.0,
                "mean_correlation": 0.0,
            }

        # Save probe results figure
        fig_probes = create_probe_results_figure(
            probe_results,
            title=f"{model_name} - {LAYER_NAMES[layer]} - Within-Cluster Position Probes",
        )
        probes_path = (
            PLOTS_DIR
            / f"probes_{model_name.replace(' ', '_').replace('+', '')}_{layer}.png"
        )
        fig_probes.savefig(probes_path, dpi=150, bbox_inches="tight")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"probes/{model_name}/{layer}": wandb.Image(fig_probes)})
        plt.close(fig_probes)

        results["layers"][layer] = layer_results

        # Print summary
        print(f"\n    Summary for {layer}:")
        print(f"      Clusters identified: {len(cluster_stats)}")
        print(
            f"      Coherence: {layer_results['coherence']['mean_clusters_per_sequence']:.1f} clusters/sequence"
        )
        print(
            f"      Dominant fraction: {layer_results['coherence']['mean_dominant_fraction']:.2f}"
        )
        if layer_results["within_cluster_probes"]["n_clusters_probed"] > 0:
            print(
                f"      Within-cluster probe R²: {layer_results['within_cluster_probes']['mean_r2']:.3f}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="24-Snake Cluster Analysis for NoPE")
    parser.add_argument(
        "--n_sequences", type=int, default=100, help="Number of sequences"
    )
    parser.add_argument(
        "--context_length", type=int, default=256, help="Context length"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=24, help="Number of clusters to identify"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with fewer samples"
    )
    args = parser.parse_args()

    if args.quick:
        args.n_sequences = 30
        args.n_clusters = 16

    print("=" * 70)
    print("24-SNAKE CLUSTER ANALYSIS FOR NoPE TRANSFORMERS")
    print("=" * 70)
    print(f"  Sequences: {args.n_sequences}")
    print(f"  Context length: {args.context_length}")
    print(f"  Clusters: {args.n_clusters}")
    print(f"  WandB: {args.wandb}")
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
            project="nope-snake-clusters",
            name=f"snake_analysis_n{args.n_sequences}_c{args.context_length}",
            config=vars(args),
        )
        print("\nWandB initialized. Project: nope-snake-clusters")

    # Load OWT data
    print("\nLoading OWT data...")
    owt_data = load_owt_data()
    tokens = get_owt_sequences(owt_data, args.n_sequences, args.context_length)
    print(f"  Loaded {args.n_sequences} sequences of length {args.context_length}")

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # NoPE + LayerNorm - Random Init
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL: NoPE + LayerNorm (Random Initialization)")
    print("=" * 70)

    random_model, _ = create_random_model(use_pe=False, block_size=args.context_length)
    results = run_analysis(
        "NoPE_LN_random",
        random_model,
        tokens,
        use_wandb=use_wandb,
        n_clusters=args.n_clusters,
    )
    all_results["NoPE_LN_random"] = results
    del random_model
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════════
    # NoPE + LayerNorm - Trained
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL: NoPE + LayerNorm (Trained)")
    print("=" * 70)

    try:
        trained_model, _ = load_trained_model(
            "out-nope-owt-ln/ckpt.pt", use_pe=False, block_size=args.context_length
        )
        results = run_analysis(
            "NoPE_LN_trained",
            trained_model,
            tokens,
            use_wandb=use_wandb,
            n_clusters=args.n_clusters,
        )
        all_results["NoPE_LN_trained"] = results
        del trained_model
        torch.cuda.empty_cache()
    except FileNotFoundError as e:
        print(f"  Skipping trained model: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Baseline + PE - Random Init (for comparison)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL: Baseline + PE (Random Initialization)")
    print("=" * 70)

    baseline_model, _ = create_random_model(use_pe=True, block_size=args.context_length)
    results = run_analysis(
        "Baseline_PE_random",
        baseline_model,
        tokens,
        use_wandb=use_wandb,
        n_clusters=args.n_clusters,
    )
    all_results["Baseline_PE_random"] = results
    del baseline_model
    torch.cuda.empty_cache()

    # Save results
    results_path = RESULTS_DIR / "snake_cluster_results.json"

    # Remove large vectors for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {
                k: clean_for_json(v)
                for k, v in obj.items()
                if k != "mean_vector" and k != "per_cluster"
            }
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(results_path, "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        for layer, layer_results in model_results.get("layers", {}).items():
            coherence = layer_results.get("coherence", {})
            probes = layer_results.get("within_cluster_probes", {})
            print(f"  {layer}:")
            print(
                f"    Clusters/sequence: {coherence.get('mean_clusters_per_sequence', 0):.1f}"
            )
            print(
                f"    Dominant fraction: {coherence.get('mean_dominant_fraction', 0):.2f}"
            )
            print(f"    Within-cluster R²: {probes.get('mean_r2', 0):.3f}")

    if use_wandb:
        wandb.finish()
        print("\nWandB run finished.")

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
