"""
Comprehensive Decoding Vector Ablation and Cluster Analysis

This script implements 4 major experiments:
1. Decoding Vector Correlation at All Layers
2. Ablation of Decoding Vector Direction (accuracy drop measurement)
3. 24-Cluster Snake Analysis with OWT Samples
4. Sample-Level Kurtosis after LayerNorm

All results are logged to WandB for tracking and visualization.

Models analyzed:
- NoPE + LayerNorm (standard NoPE transformer)
- Baseline + PE (standard transformer with positional embeddings)

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis_scripts/decoding_vector_ablation_comprehensive.py \
        --n_sequences 1000 \
        --context_length 512 \
        --wandb

Author: Research Assistant
Date: January 2026
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.stats import pearsonr, kurtosis, spearmanr
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))
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
RESULTS_DIR = PROJECT_ROOT / "results" / "decoding_ablation_comprehensive"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
OWT_DATA_PATH = PROJECT_ROOT / "nanoGPT" / "data" / "openwebtext" / "train.bin"


@dataclass
class ExperimentConfig:
    """Configuration for a model experiment."""

    name: str
    short_name: str
    checkpoint_path: str
    use_positional_embedding: bool
    use_batchnorm_ln2: bool = False
    skip_ln2: bool = False


# Only NoPE + LayerNorm and Baseline + PE (no BatchNorm or No-LN2 variants)
EXPERIMENTS = [
    ExperimentConfig(
        name="NoPE + LayerNorm",
        short_name="NoPE_LN",
        checkpoint_path="out-nope-owt-ln/ckpt.pt",
        use_positional_embedding=False,
    ),
    ExperimentConfig(
        name="Baseline + PE",
        short_name="Baseline_PE",
        checkpoint_path="out-baseline-owt-pe/ckpt.pt",
        use_positional_embedding=True,
    ),
]

# Layers to analyze
LAYERS = [
    "embed",
    "post_ln1",
    "post_attn",
    "pre_ln2",  # This is x + attn_out (after residual, before LN2) - THE KEY LAYER FOR 24 SNAKES
    "post_ln2",
    "post_mlp_residual",
]

LAYER_NAMES = {
    "embed": "Embedding",
    "post_ln1": "Post-LN1",
    "post_attn": "Post-Attn",
    "pre_ln2": "Pre-LN2",  # After attention residual, before LN2 - shows 24 snakes
    "post_ln2": "Post-LN2",
    "post_mlp_residual": "MLP+Res",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def create_random_model(
    exp: ExperimentConfig, block_size: int = 512
) -> Tuple[GPT, GPTConfig]:
    """Create a randomly initialized model."""
    config = GPTConfig(
        n_layer=1,
        n_head=12,
        n_embd=768,
        block_size=block_size,
        vocab_size=50304,
        dropout=0.0,
        use_positional_embedding=exp.use_positional_embedding,
        norm_type="layernorm",
        bias=False,
        skip_ln2=exp.skip_ln2,
        use_batchnorm_ln2=exp.use_batchnorm_ln2,
    )
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model, config


def load_trained_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Load a trained model from checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / exp.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 512),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=exp.use_positional_embedding,
        norm_type=model_args.get("norm_type", "layernorm"),
        bias=model_args.get("bias", False),
        skip_ln2=exp.skip_ln2,
        use_batchnorm_ln2=exp.use_batchnorm_ln2,
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


def load_owt_tokens(n_sequences: int, context_length: int) -> torch.Tensor:
    """Load tokens from OpenWebText train.bin."""
    if not OWT_DATA_PATH.exists():
        raise FileNotFoundError(f"OWT data not found: {OWT_DATA_PATH}")

    # Memory-map the data
    data = np.memmap(OWT_DATA_PATH, dtype=np.uint16, mode="r")
    total_tokens = len(data)

    # Sample random starting positions
    max_start = total_tokens - context_length
    np.random.seed(42)
    starts = np.random.randint(0, max_start, size=n_sequences)

    # Extract sequences
    tokens = np.zeros((n_sequences, context_length), dtype=np.int64)
    for i, start in enumerate(starts):
        tokens[i] = data[start : start + context_length].astype(np.int64)

    return torch.tensor(tokens, device=DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def get_activations_at_layers(
    model: GPT,
    tokens: torch.Tensor,
    skip_ln2: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Extract activations at all key layers.

    IMPORTANT: pre_ln2 is the layer that shows 24 snake clusters in t-SNE!
    This is x + attn_out (after attention residual, before LN2).

    Returns dict mapping layer name to activation tensor [batch, seq_len, d_model].
    """
    activations = {}

    with torch.no_grad():
        # Token embeddings
        tok_emb = model.transformer.wte(tokens)

        # Add positional embeddings if available
        if hasattr(model.transformer, "wpe") and model.config.use_positional_embedding:
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        activations["embed"] = x.clone()

        block = model.transformer.h[0]

        # Post LN1
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.clone()

        # Post attention (before residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # Pre-LN2: After attention residual, BEFORE LN2 - THIS SHOWS 24 SNAKES!
        x = x + attn_out
        activations["pre_ln2"] = x.clone()

        # Post LN2 (if not skipped)
        if not skip_ln2 and hasattr(block, "ln_2"):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.clone()
            mlp_input = x

        # Post MLP residual
        mlp_out = block.mlp(mlp_input)
        x = x + mlp_out
        activations["post_mlp_residual"] = x.clone()

    return activations


# ═══════════════════════════════════════════════════════════════════════════════
# DECODING VECTOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_decoding_vectors(model: GPT) -> Dict[str, torch.Tensor]:
    """
    Compute layer-appropriate decoding vectors.

    The decoding vector formula:
    - embed, post_ln1: w = Σ_j E_j (sum of embeddings)
    - post_attn and beyond: w = W_O @ W_V @ Σ_j LN(E_j)
    """
    with torch.no_grad():
        # Get token embeddings
        E = model.transformer.wte.weight.detach()  # [vocab_size, n_embd]

        # Sum of raw embeddings
        sum_E = E.sum(dim=0)  # [n_embd]

        # Apply LN1 to embeddings
        ln1 = model.transformer.h[0].ln_1
        E_centered = E - E.mean(dim=-1, keepdim=True)
        E_std = E.std(dim=-1, keepdim=True)
        E_ln = E_centered / (E_std + 1e-5)
        E_ln = E_ln * ln1.weight
        if hasattr(ln1, "bias") and ln1.bias is not None:
            E_ln = E_ln + ln1.bias

        sum_ln_E = E_ln.sum(dim=0)  # [n_embd]

        # Get W_V and W_O from attention
        attn = model.transformer.h[0].attn
        n_embd = model.config.n_embd

        W_V = attn.c_attn.weight[2 * n_embd :, :].detach()  # [n_embd, n_embd]
        W_O = attn.c_proj.weight.detach()  # [n_embd, n_embd]

        # Post-attention decoding vector: w = W_O @ W_V @ sum_ln_E
        w_post_attn = W_O @ W_V @ sum_ln_E

        def normalize(v):
            return v / (torch.norm(v) + 1e-8)

        return {
            "embed": normalize(sum_E),
            "post_ln1": normalize(sum_ln_E),
            "post_attn": normalize(w_post_attn),
            "pre_ln2": normalize(w_post_attn),
            "post_ln2": normalize(w_post_attn),
            "post_mlp_residual": normalize(w_post_attn),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: DECODING VECTOR AT ALL LAYERS
# ═══════════════════════════════════════════════════════════════════════════════


def exp1_decoding_vector_correlation(
    model: GPT,
    config: GPTConfig,
    tokens: torch.Tensor,
    exp_config: ExperimentConfig,
    init_type: str,
) -> Dict:
    """
    Test decoding vector correlation at each layer.

    Returns dict with correlation metrics per layer.
    """
    print(f"\n  [Exp1] Decoding vector correlation: {exp_config.name} ({init_type})")

    n_sequences, ctx = tokens.shape

    # Compute decoding vectors
    decoding_vectors = compute_decoding_vectors(model)

    # Get activations
    results = {"model": exp_config.name, "init_type": init_type, "layers": {}}

    # Process in batches
    batch_size = 32
    all_projections = {layer: [] for layer in LAYERS}

    for i in range(0, n_sequences, batch_size):
        batch_tokens = tokens[i : i + batch_size]
        activations = get_activations_at_layers(
            model, batch_tokens, skip_ln2=exp_config.skip_ln2
        )

        for layer in LAYERS:
            if layer in decoding_vectors:
                w = decoding_vectors[layer]
                proj = (activations[layer] * w).sum(dim=-1)  # [batch, ctx]
                all_projections[layer].append(proj.cpu().numpy())

    # Stack projections
    for layer in LAYERS:
        if all_projections[layer]:
            all_projections[layer] = np.vstack(
                all_projections[layer]
            )  # [n_sequences, ctx]

    # Compute statistics
    positions = np.arange(ctx)

    for layer in LAYERS:
        if layer not in all_projections or len(all_projections[layer]) == 0:
            continue

        projs = all_projections[layer]  # [n_sequences, ctx]

        # Mean projection at each position
        mean_proj = projs.mean(axis=0)
        std_proj = projs.std(axis=0)

        # Overall correlation (flatten all samples)
        proj_flat = projs.flatten()
        pos_flat = np.tile(positions, n_sequences)
        overall_corr, overall_p = pearsonr(proj_flat, pos_flat)

        # Per-sample correlations
        per_sample_corrs = [
            pearsonr(projs[i], positions)[0] for i in range(n_sequences)
        ]
        per_sample_mean = np.mean(per_sample_corrs)
        per_sample_std = np.std(per_sample_corrs)

        # Mean projection correlation (key metric)
        mean_corr, _ = pearsonr(mean_proj, positions)

        results["layers"][layer] = {
            "overall_correlation": float(overall_corr),
            "per_sample_corr_mean": float(per_sample_mean),
            "per_sample_corr_std": float(per_sample_std),
            "mean_projection_correlation": float(mean_corr),
        }

        print(
            f"    {LAYER_NAMES[layer]:>12}: overall_r={overall_corr:.3f}, per_sample_r={per_sample_mean:.3f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: ABLATION - PROJECT OUT DECODING DIRECTION
# ═══════════════════════════════════════════════════════════════════════════════


def train_position_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """Train a linear probe for position regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    y_pred = ridge.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_test, y_pred)

    return float(r2), float(corr)


def exp2_ablation_accuracy_drop(
    model: GPT,
    config: GPTConfig,
    tokens: torch.Tensor,
    exp_config: ExperimentConfig,
    init_type: str,
) -> Dict:
    """
    Measure accuracy drop when projecting out decoding vector direction.

    Ablation: h' = h - (h·w/||w||²) * w
    """
    print(f"\n  [Exp2] Ablation accuracy drop: {exp_config.name} ({init_type})")

    n_sequences, ctx = tokens.shape

    # Compute decoding vectors
    decoding_vectors = compute_decoding_vectors(model)

    # Get activations
    batch_size = 32
    all_activations = {layer: [] for layer in LAYERS}

    for i in range(0, n_sequences, batch_size):
        batch_tokens = tokens[i : i + batch_size]
        activations = get_activations_at_layers(
            model, batch_tokens, skip_ln2=exp_config.skip_ln2
        )
        for layer in LAYERS:
            all_activations[layer].append(activations[layer].cpu().numpy())

    for layer in LAYERS:
        all_activations[layer] = np.vstack(
            all_activations[layer]
        )  # [n_seq, ctx, d_model]

    # Prepare positions
    positions = np.tile(np.arange(ctx), n_sequences)

    results = {"model": exp_config.name, "init_type": init_type, "layers": {}}

    for layer in LAYERS:
        if layer not in decoding_vectors:
            continue

        acts = all_activations[layer]  # [n_seq, ctx, d_model]
        X = acts.reshape(-1, acts.shape[-1])  # [n_seq * ctx, d_model]
        y = positions

        # Train/test split
        n_train = int(0.8 * len(y))
        idx = np.random.permutation(len(y))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Original accuracy
        r2_orig, corr_orig = train_position_probe(X_train, y_train, X_test, y_test)

        # Ablated accuracy
        w = decoding_vectors[layer].cpu().numpy()
        w_norm = w / (np.linalg.norm(w) + 1e-8)

        # Project out decoding direction: h' = h - (h·w)w
        X_train_ablated = X_train - np.outer(X_train @ w_norm, w_norm)
        X_test_ablated = X_test - np.outer(X_test @ w_norm, w_norm)

        r2_ablated, corr_ablated = train_position_probe(
            X_train_ablated, y_train, X_test_ablated, y_test
        )

        accuracy_drop = r2_orig - r2_ablated
        relative_drop = accuracy_drop / (abs(r2_orig) + 1e-8) * 100

        results["layers"][layer] = {
            "r2_original": r2_orig,
            "r2_ablated": r2_ablated,
            "accuracy_drop": accuracy_drop,
            "relative_drop_pct": relative_drop,
            "corr_original": corr_orig,
            "corr_ablated": corr_ablated,
        }

        print(
            f"    {LAYER_NAMES[layer]:>12}: R²_orig={r2_orig:.3f}, R²_ablated={r2_ablated:.3f}, drop={relative_drop:.1f}%"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: 24-CLUSTER SNAKE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def exp3_cluster_snake_analysis(
    model: GPT,
    config: GPTConfig,
    tokens: torch.Tensor,
    exp_config: ExperimentConfig,
    init_type: str,
    n_clusters: int = 24,
    layer: str = "pre_ln2",  # KEY: Use pre_ln2 for 24 snakes!
    use_wandb: bool = False,
) -> Dict:
    """
    Analyze the 24 snake-like clusters in t-SNE visualization.

    IMPORTANT: Use "pre_ln2" layer (after attention residual, before LN2) to see the 24 snakes.

    Key questions:
    1. Do all positions from the same sequence stay in the same snake?
    2. What determines cluster membership?
    3. Can we predict position with linear probe within each cluster?

    For each cluster, extract:
    - Mean vector of original high-dim activations
    - Std of original high-dim activations
    - Mean norm of original high-dim activations
    """
    print(
        f"\n  [Exp3] 24-Cluster Snake Analysis: {exp_config.name} ({init_type}) @ {layer}"
    )

    n_sequences, ctx = tokens.shape

    # Get activations at target layer
    batch_size = 32
    all_activations = []

    for i in range(0, n_sequences, batch_size):
        batch_tokens = tokens[i : i + batch_size]
        activations = get_activations_at_layers(
            model, batch_tokens, skip_ln2=exp_config.skip_ln2
        )
        all_activations.append(activations[layer].cpu().numpy())

    activations = np.vstack(all_activations)  # [n_seq, ctx, d_model]
    n_seq, ctx, d_model = activations.shape

    # Flatten: each token-position is a point
    X = activations.reshape(-1, d_model)  # [n_seq * ctx, d_model]
    positions = np.tile(np.arange(ctx), n_seq)
    sequence_ids = np.repeat(np.arange(n_seq), ctx)

    # Subsample for t-SNE (max 10K points for speed)
    max_points = min(10000, len(X))
    subsample_idx = np.random.choice(len(X), max_points, replace=False)
    X_sub = X[subsample_idx]
    pos_sub = positions[subsample_idx]
    seq_sub = sequence_ids[subsample_idx]

    print(f"    Running t-SNE on {max_points} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    embeddings = tsne.fit_transform(X_sub)

    # Cluster using K-means
    print(f"    Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # --- Analysis 1: Sequence Coherence ---
    print("    Analyzing sequence coherence...")
    unique_seqs = np.unique(seq_sub)

    clusters_per_sequence = []
    for seq_id in unique_seqs:
        mask = seq_sub == seq_id
        unique_clusters = np.unique(cluster_labels[mask])
        clusters_per_sequence.append(len(unique_clusters))

    coherence_stats = {
        "mean_clusters_per_sequence": float(np.mean(clusters_per_sequence)),
        "median_clusters_per_sequence": float(np.median(clusters_per_sequence)),
        "std_clusters_per_sequence": float(np.std(clusters_per_sequence)),
        "max_clusters_per_sequence": int(np.max(clusters_per_sequence)),
        "min_clusters_per_sequence": int(np.min(clusters_per_sequence)),
        "pct_single_cluster": float(
            np.mean(np.array(clusters_per_sequence) == 1) * 100
        ),
    }
    print(
        f"    Sequence coherence: mean={coherence_stats['mean_clusters_per_sequence']:.1f} clusters/sequence"
    )
    print(
        f"    {coherence_stats['pct_single_cluster']:.1f}% of sequences stay in a single cluster"
    )

    # --- Analysis 2: Per-Cluster Statistics (mean, std, norm of original high-dim vectors) ---
    print("    Computing per-cluster statistics...")
    cluster_stats = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_vecs = X_sub[mask]  # [n_points_in_cluster, d_model]
            cluster_positions = pos_sub[mask]

            mean_vec = cluster_vecs.mean(axis=0)
            std_vec = cluster_vecs.std(axis=0)
            norms = np.linalg.norm(cluster_vecs, axis=1)

            # Position statistics within cluster
            pos_mean = cluster_positions.mean()
            pos_std = cluster_positions.std()

            cluster_stats[c] = {
                "size": int(mask.sum()),
                "mean_vector_norm": float(np.linalg.norm(mean_vec)),
                "mean_norm": float(norms.mean()),
                "std_norm": float(norms.std()),
                "mean_std_across_dims": float(std_vec.mean()),
                "position_mean": float(pos_mean),
                "position_std": float(pos_std),
            }

    # --- Analysis 3: Per-Cluster Linear Probing ---
    print("    Training per-cluster linear probes...")
    cluster_probe_results = {}

    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() < 100:  # Skip small clusters
            continue

        X_cluster = X_sub[mask]
        y_cluster = pos_sub[mask]

        # Train/test split within cluster
        n_train = int(0.8 * len(y_cluster))
        idx = np.random.permutation(len(y_cluster))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        if len(test_idx) < 10:
            continue

        X_train = X_cluster[train_idx]
        X_test = X_cluster[test_idx]
        y_train = y_cluster[train_idx]
        y_test = y_cluster[test_idx]

        r2, corr = train_position_probe(X_train, y_train, X_test, y_test)

        cluster_probe_results[c] = {
            "size": int(mask.sum()),
            "r2": r2,
            "correlation": corr,
        }

    # Average per-cluster R²
    if cluster_probe_results:
        avg_r2 = np.mean([v["r2"] for v in cluster_probe_results.values()])
        avg_corr = np.mean([v["correlation"] for v in cluster_probe_results.values()])
        print(f"    Per-cluster probe: avg R²={avg_r2:.3f}, avg corr={avg_corr:.3f}")
    else:
        avg_r2, avg_corr = 0.0, 0.0

    # --- Create Visualization (COLOR BY POSITION, not by cluster) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Color by position bucket
    n_buckets = 8
    bucket_size = ctx / n_buckets
    pos_buckets = np.clip((pos_sub / bucket_size).astype(int), 0, n_buckets - 1)

    # Plot 1: t-SNE colored by position (TOP LEFT)
    cmap = plt.cm.get_cmap("viridis", n_buckets)
    for bucket in range(n_buckets):
        mask = pos_buckets == bucket
        if mask.sum() > 0:
            start_pos = int(bucket * bucket_size)
            end_pos = int((bucket + 1) * bucket_size) - 1
            label = f"pos {start_pos}-{end_pos}"
            axes[0, 0].scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[cmap(bucket)],
                label=label,
                alpha=0.6,
                s=8,
            )
    axes[0, 0].set_title(
        f"t-SNE colored by position\n{exp_config.name} ({init_type}) @ {layer}",
        fontsize=12,
    )
    axes[0, 0].set_xlabel("t-SNE 1")
    axes[0, 0].set_ylabel("t-SNE 2")
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: t-SNE with cluster boundaries (TOP RIGHT)
    scatter2 = axes[0, 1].scatter(
        embeddings[:, 0], embeddings[:, 1], c=pos_sub, cmap="viridis", alpha=0.5, s=5
    )
    plt.colorbar(scatter2, ax=axes[0, 1], label="Position")
    # Draw cluster centroids
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            centroid = embeddings[mask].mean(axis=0)
            axes[0, 1].annotate(
                str(c),
                centroid,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                color="red",
            )
    axes[0, 1].set_title(f"t-SNE with cluster labels (K={n_clusters})")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")

    # Plot 3: Sequence coherence histogram (BOTTOM LEFT)
    axes[1, 0].hist(
        clusters_per_sequence,
        bins=range(1, max(clusters_per_sequence) + 2),
        edgecolor="black",
        alpha=0.7,
    )
    axes[1, 0].set_title("Sequence Coherence\n(clusters per sequence)")
    axes[1, 0].set_xlabel("Number of clusters")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].axvline(
        coherence_stats["mean_clusters_per_sequence"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean={coherence_stats['mean_clusters_per_sequence']:.1f}",
    )
    axes[1, 0].legend()

    # Plot 4: Per-cluster statistics (BOTTOM RIGHT)
    if cluster_stats:
        cluster_ids = sorted(cluster_stats.keys())
        sizes = [cluster_stats[c]["size"] for c in cluster_ids]
        mean_norms = [cluster_stats[c]["mean_norm"] for c in cluster_ids]
        pos_means = [cluster_stats[c]["position_mean"] for c in cluster_ids]

        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        bars = ax4.bar(range(len(cluster_ids)), sizes, alpha=0.6, label="Cluster size")
        ax4_twin.plot(range(len(cluster_ids)), mean_norms, "ro-", label="Mean norm")

        ax4.set_xlabel("Cluster ID")
        ax4.set_ylabel("Cluster Size", color="blue")
        ax4_twin.set_ylabel("Mean Norm", color="red")
        ax4.set_title("Per-Cluster Statistics")
        ax4.legend(loc="upper left")
        ax4_twin.legend(loc="upper right")

    plt.tight_layout()

    # Save figure
    fig_path = (
        PLOTS_DIR / f"cluster_analysis_{exp_config.short_name}_{init_type}_{layer}.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fig_path}")

    # Log to wandb if available (as image, not scalar)
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(
            {
                f"exp3/{exp_config.short_name}_{init_type}_{layer}": wandb.Image(
                    str(fig_path)
                ),
            }
        )

    return {
        "model": exp_config.name,
        "init_type": init_type,
        "layer": layer,
        "n_clusters": n_clusters,
        "coherence_stats": coherence_stats,
        "cluster_stats": cluster_stats,
        "cluster_probe_results": cluster_probe_results,
        "avg_per_cluster_r2": avg_r2,
        "avg_per_cluster_corr": avg_corr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: SAMPLE-LEVEL KURTOSIS
# ═══════════════════════════════════════════════════════════════════════════════


def exp4_sample_level_kurtosis(
    model: GPT,
    config: GPTConfig,
    tokens: torch.Tensor,
    exp_config: ExperimentConfig,
    init_type: str,
    layer: str = "post_ln2",
    use_wandb: bool = False,
) -> Dict:
    """
    Compute kurtosis at the individual sample level (across dimensions).

    For each sample s and position i:
        kurtosis_sample[s, i] = kurtosis(activations[s, i, :])  # across d_model dims
    """
    print(f"\n  [Exp4] Sample-level kurtosis: {exp_config.name} ({init_type})")

    n_sequences, ctx = tokens.shape

    # Get activations
    batch_size = 32
    all_activations = []

    for i in range(0, n_sequences, batch_size):
        batch_tokens = tokens[i : i + batch_size]
        activations = get_activations_at_layers(
            model, batch_tokens, skip_ln2=exp_config.skip_ln2
        )
        all_activations.append(activations[layer].cpu().numpy())

    activations = np.vstack(all_activations)  # [n_seq, ctx, d_model]
    n_seq, ctx, d_model = activations.shape

    # Compute kurtosis per sample per position (across dimensions) - VECTORIZED
    print("    Computing sample-level kurtosis (vectorized)...")

    # Vectorized kurtosis computation: kurtosis = E[(X-mu)^4] / E[(X-mu)^2]^2 - 3
    # Reshape to [n_seq * ctx, d_model] for efficient computation
    acts_flat = activations.reshape(n_seq * ctx, d_model)

    # Center the data (per sample)
    mean_per_sample = acts_flat.mean(axis=1, keepdims=True)
    centered = acts_flat - mean_per_sample

    # Compute moments
    var_per_sample = np.mean(centered**2, axis=1)
    fourth_moment = np.mean(centered**4, axis=1)

    # Excess kurtosis (Fisher definition)
    sample_kurtosis_flat = fourth_moment / (var_per_sample**2 + 1e-8) - 3
    sample_kurtosis = sample_kurtosis_flat.reshape(n_seq, ctx)

    # Statistics
    mean_kurtosis_by_pos = sample_kurtosis.mean(axis=0)  # [ctx]
    std_kurtosis_by_pos = sample_kurtosis.std(axis=0)

    # Correlation with position
    positions = np.arange(ctx)
    kurtosis_flat = sample_kurtosis.flatten()
    pos_flat = np.tile(positions, n_seq)

    overall_corr, _ = pearsonr(kurtosis_flat, pos_flat)
    mean_corr, _ = pearsonr(mean_kurtosis_by_pos, positions)

    print(f"    Overall kurtosis-position corr: r={overall_corr:.4f}")
    print(f"    Mean kurtosis-position corr: r={mean_corr:.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Mean kurtosis by position
    axes[0].fill_between(
        positions,
        mean_kurtosis_by_pos - std_kurtosis_by_pos,
        mean_kurtosis_by_pos + std_kurtosis_by_pos,
        alpha=0.3,
    )
    axes[0].plot(positions, mean_kurtosis_by_pos, linewidth=2)
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Kurtosis")
    axes[0].set_title(f"Mean Kurtosis by Position\nr={mean_corr:.3f}")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Kurtosis distribution
    axes[1].hist(kurtosis_flat, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Kurtosis")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Kurtosis Distribution")
    axes[1].axvline(
        np.mean(kurtosis_flat),
        color="red",
        linestyle="--",
        label=f"Mean={np.mean(kurtosis_flat):.2f}",
    )
    axes[1].legend()

    # Plot 3: Kurtosis heatmap (sample of sequences)
    n_show = min(50, n_seq)
    im = axes[2].imshow(sample_kurtosis[:n_show, :], aspect="auto", cmap="viridis")
    axes[2].set_xlabel("Position")
    axes[2].set_ylabel("Sequence")
    axes[2].set_title(f"Kurtosis Heatmap (first {n_show} sequences)")
    plt.colorbar(im, ax=axes[2], label="Kurtosis")

    plt.suptitle(
        f"Sample-Level Kurtosis: {exp_config.name} ({init_type}) @ {layer}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save
    fig_path = PLOTS_DIR / f"kurtosis_{exp_config.short_name}_{init_type}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fig_path}")

    # Log to wandb (as image, not scalar)
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(
            {
                f"exp4/{exp_config.short_name}_{init_type}": wandb.Image(str(fig_path)),
            }
        )

    return {
        "model": exp_config.name,
        "init_type": init_type,
        "layer": layer,
        "overall_kurtosis_position_corr": float(overall_corr),
        "mean_kurtosis_position_corr": float(mean_corr),
        "mean_kurtosis": float(np.mean(kurtosis_flat)),
        "std_kurtosis": float(np.std(kurtosis_flat)),
        "mean_kurtosis_by_position": mean_kurtosis_by_pos.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE CREATION
# ═══════════════════════════════════════════════════════════════════════════════


def create_summary_figure(all_results: Dict, use_wandb: bool = False):
    """Create comprehensive summary figure for all experiments."""

    # Create a multi-panel figure
    fig = plt.figure(figsize=(20, 16))

    # Layout: 4 rows
    # Row 1: Decoding vector correlation by layer (random vs trained)
    # Row 2: Ablation accuracy drop
    # Row 3: Cluster coherence comparison
    # Row 4: Kurtosis comparison

    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    exp_names = [e.name for e in EXPERIMENTS]
    colors = {"random": "#3498db", "trained": "#e74c3c"}

    # --- Row 1: Decoding Vector Correlation ---
    for col, init_type in enumerate(["random", "trained"]):
        ax = fig.add_subplot(gs[0, col])

        for exp_name in exp_names:
            key = f"{exp_name}_{init_type}"
            if key in all_results.get("exp1_decoding_vector", {}):
                data = all_results["exp1_decoding_vector"][key]
                layers = list(data["layers"].keys())
                corrs = [data["layers"][l]["overall_correlation"] for l in layers]
                layer_labels = [LAYER_NAMES.get(l, l) for l in layers]

                ax.plot(
                    range(len(layers)),
                    corrs,
                    "o-",
                    label=exp_name,
                    linewidth=2,
                    markersize=8,
                )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Correlation")
        ax.set_title(
            f"Exp1: Decoding Vector Correlation ({init_type})", fontweight="bold"
        )
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- Row 2: Ablation Accuracy Drop ---
    for col, init_type in enumerate(["random", "trained"]):
        ax = fig.add_subplot(gs[1, col])

        for exp_name in exp_names:
            key = f"{exp_name}_{init_type}"
            if key in all_results.get("exp2_ablation", {}):
                data = all_results["exp2_ablation"][key]
                layers = list(data["layers"].keys())
                drops = [data["layers"][l]["accuracy_drop"] for l in layers]
                layer_labels = [LAYER_NAMES.get(l, l) for l in layers]

                ax.bar(
                    np.arange(len(layers))
                    + (0.2 if exp_name == exp_names[0] else -0.2),
                    drops,
                    width=0.35,
                    label=exp_name,
                    alpha=0.8,
                )

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("R² Drop (orig - ablated)")
        ax.set_title(f"Exp2: Ablation Accuracy Drop ({init_type})", fontweight="bold")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # --- Row 3: Cluster Coherence ---
    ax = fig.add_subplot(gs[2, :])

    x_positions = []
    x_labels = []
    coherence_values = []
    bar_colors = []

    idx = 0
    for exp_name in exp_names:
        for init_type in ["random", "trained"]:
            key = f"{exp_name}_{init_type}"
            if key in all_results.get("exp3_clusters", {}):
                data = all_results["exp3_clusters"][key]
                x_positions.append(idx)
                x_labels.append(f"{exp_name[:10]}\n({init_type})")
                coherence_values.append(data["coherence_stats"]["pct_single_cluster"])
                bar_colors.append(colors[init_type])
                idx += 1

    if coherence_values:
        ax.bar(
            x_positions,
            coherence_values,
            color=bar_colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("% Sequences in Single Cluster")
        ax.set_title(
            "Exp3: Sequence Coherence (higher = more structure)", fontweight="bold"
        )
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")

    # --- Row 4: Kurtosis Correlation ---
    ax = fig.add_subplot(gs[3, :])

    x_positions = []
    x_labels = []
    kurtosis_corrs = []
    bar_colors = []

    idx = 0
    for exp_name in exp_names:
        for init_type in ["random", "trained"]:
            key = f"{exp_name}_{init_type}"
            if key in all_results.get("exp4_kurtosis", {}):
                data = all_results["exp4_kurtosis"][key]
                x_positions.append(idx)
                x_labels.append(f"{exp_name[:10]}\n({init_type})")
                kurtosis_corrs.append(data["mean_kurtosis_position_corr"])
                bar_colors.append(colors[init_type])
                idx += 1

    if kurtosis_corrs:
        ax.bar(
            x_positions, kurtosis_corrs, color=bar_colors, alpha=0.8, edgecolor="black"
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Kurtosis-Position Correlation")
        ax.set_title(
            "Exp4: Sample-Level Kurtosis Correlation with Position", fontweight="bold"
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Comprehensive Decoding Vector Ablation Analysis\n(NoPE+LN vs Baseline+PE)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save
    fig_path = PLOTS_DIR / "comprehensive_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary figure: {fig_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary/comprehensive_figure": wandb.Image(str(fig_path))})

    return fig_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Decoding Vector Ablation Analysis"
    )
    parser.add_argument(
        "--n_sequences", type=int, default=1000, help="Number of OWT sequences"
    )
    parser.add_argument(
        "--context_length", type=int, default=512, help="Context length"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=24,
        help="Number of clusters for snake analysis",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to WandB")
    parser.add_argument("--exp1", action="store_true", help="Run experiment 1 only")
    parser.add_argument("--exp2", action="store_true", help="Run experiment 2 only")
    parser.add_argument("--exp3", action="store_true", help="Run experiment 3 only")
    parser.add_argument("--exp4", action="store_true", help="Run experiment 4 only")
    args = parser.parse_args()

    # If no specific experiment selected, run all
    run_all = not (args.exp1 or args.exp2 or args.exp3 or args.exp4)

    print("=" * 80)
    print("COMPREHENSIVE DECODING VECTOR ABLATION ANALYSIS")
    print("Models: NoPE + LayerNorm, Baseline + PE (only)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Sequences: {args.n_sequences}")
    print(f"Context length: {args.context_length}")
    print(f"WandB: {args.wandb}")
    print("=" * 80)

    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize WandB
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="nope-decoding-ablation",
            name=f"comprehensive_analysis_{args.n_sequences}seq",
            config={
                "n_sequences": args.n_sequences,
                "context_length": args.context_length,
                "n_clusters": args.n_clusters,
                "experiments": [e.name for e in EXPERIMENTS],
            },
        )
        print("\nWandB initialized. Project: nope-decoding-ablation")

    # Load OWT tokens
    print(f"\nLoading {args.n_sequences} sequences from OWT...")
    owt_tokens = load_owt_tokens(args.n_sequences, args.context_length)
    print(f"Loaded tokens: {owt_tokens.shape}")

    # Also prepare random tokens for comparison
    random_tokens = torch.randint(
        0, 50304, (args.n_sequences, args.context_length), device=DEVICE
    )

    # Results storage
    all_results = {
        "exp1_decoding_vector": {},
        "exp2_ablation": {},
        "exp3_clusters": {},
        "exp4_kurtosis": {},
    }

    # ─── Run Experiments ─────────────────────────────────────────────────────────

    for exp in EXPERIMENTS:
        print(f"\n{'#' * 80}")
        print(f"# {exp.name}")
        print(f"{'#' * 80}")

        for init_type in ["random", "trained"]:
            result_key = f"{exp.name}_{init_type}"

            # Load model
            try:
                if init_type == "random":
                    model, config = create_random_model(exp, args.context_length)
                    tokens = random_tokens  # Use random tokens for random model
                else:
                    model, config = load_trained_model(exp)
                    tokens = owt_tokens  # Use OWT tokens for trained model
            except FileNotFoundError as e:
                print(f"  Skipping {init_type}: {e}")
                continue

            # Select token subset
            tokens_subset = tokens[: args.n_sequences]

            # Experiment 1: Decoding Vector Correlation
            if run_all or args.exp1:
                result = exp1_decoding_vector_correlation(
                    model, config, tokens_subset, exp, init_type
                )
                all_results["exp1_decoding_vector"][result_key] = result

            # Experiment 2: Ablation
            if run_all or args.exp2:
                result = exp2_ablation_accuracy_drop(
                    model, config, tokens_subset, exp, init_type
                )
                all_results["exp2_ablation"][result_key] = result

            # Experiment 3: Cluster Analysis (use pre_ln2 for 24 snakes!)
            if run_all or args.exp3:
                result = exp3_cluster_snake_analysis(
                    model,
                    config,
                    tokens_subset,
                    exp,
                    init_type,
                    n_clusters=args.n_clusters,
                    layer="pre_ln2",  # KEY: Use pre_ln2 to see 24 snakes!
                    use_wandb=use_wandb,
                )
                all_results["exp3_clusters"][result_key] = result

            # Experiment 4: Sample-Level Kurtosis
            if run_all or args.exp4:
                result = exp4_sample_level_kurtosis(
                    model,
                    config,
                    tokens_subset,
                    exp,
                    init_type,
                    layer="post_ln2",
                    use_wandb=use_wandb,
                )
                all_results["exp4_kurtosis"][result_key] = result

            # Clean up
            del model
            torch.cuda.empty_cache()

    # ─── Create Summary Figure ───────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("CREATING SUMMARY FIGURE")
    print("=" * 80)
    create_summary_figure(all_results, use_wandb)

    # ─── Save Results ────────────────────────────────────────────────────────────

    results_path = RESULTS_DIR / "comprehensive_results.json"

    # Convert to JSON-serializable format
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ─── Print Summary ───────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for exp_name in [e.name for e in EXPERIMENTS]:
        print(f"\n--- {exp_name} ---")
        for init_type in ["random", "trained"]:
            key = f"{exp_name}_{init_type}"

            if key in all_results["exp3_clusters"]:
                data = all_results["exp3_clusters"][key]
                print(
                    f"  {init_type}: {data['coherence_stats']['pct_single_cluster']:.1f}% single-cluster sequences"
                )

            if key in all_results["exp4_kurtosis"]:
                data = all_results["exp4_kurtosis"][key]
                print(
                    f"  {init_type}: kurtosis-position r={data['mean_kurtosis_position_corr']:.3f}"
                )

    # Finish WandB
    if use_wandb:
        # Log summary figure
        wandb.finish()
        print("\nWandB run finished.")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results: {results_path}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
