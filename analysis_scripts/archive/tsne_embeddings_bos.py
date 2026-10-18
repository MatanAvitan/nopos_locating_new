"""
t-SNE Visualization of Trained Token Embeddings

Creates an ICML-quality figure showing t-SNE projection of token embeddings
from the trained 2-layer position regressor, with BOS token highlighted.

Usage:
    python analysis_scripts/tsne_embeddings_bos.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tiktoken

# Add nanoGPT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig


# ICML style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# BOS token ID (GPT-2 EOT token)
BOS_TOKEN_ID = 50256


def load_checkpoint_embeddings(checkpoint_path: str) -> np.ndarray:
    """Load trained token embeddings from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract embedding weights
    state_dict = checkpoint['model']
    embeddings = state_dict['wte.weight'].numpy()

    print(f"Embedding shape: {embeddings.shape}")
    return embeddings, checkpoint.get('config', {})


def run_tsne(embeddings: np.ndarray, n_components: int = 2, perplexity: int = 30,
             random_state: int = 42, n_samples: int = 5000) -> tuple:
    """Run t-SNE on embeddings, sampling for efficiency."""
    vocab_size = embeddings.shape[0]

    # Sample tokens for visualization (too many tokens for full t-SNE)
    # Always include BOS token
    np.random.seed(random_state)

    # Sample indices, ensuring BOS is included
    sample_indices = np.random.choice(vocab_size, size=min(n_samples, vocab_size), replace=False)
    if BOS_TOKEN_ID not in sample_indices:
        sample_indices[-1] = BOS_TOKEN_ID

    sample_embeddings = embeddings[sample_indices]

    print(f"Running t-SNE on {len(sample_indices)} tokens...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state,
                n_iter=1000, learning_rate='auto', init='pca')
    tsne_result = tsne.fit_transform(sample_embeddings)

    return tsne_result, sample_indices


def create_icml_tsne_plot(tsne_result: np.ndarray, sample_indices: np.ndarray,
                           output_dir: str, model_name: str = "R0"):
    """Create ICML-quality t-SNE plot with BOS token highlighted."""

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    # Find BOS token in samples
    bos_idx = np.where(sample_indices == BOS_TOKEN_ID)[0][0]

    # Plot all tokens except BOS
    non_bos_mask = np.arange(len(sample_indices)) != bos_idx
    ax.scatter(tsne_result[non_bos_mask, 0], tsne_result[non_bos_mask, 1],
               s=3, alpha=0.4, c='#4C72B0', label='Regular tokens', rasterized=True)

    # Plot BOS token prominently
    ax.scatter(tsne_result[bos_idx, 0], tsne_result[bos_idx, 1],
               s=200, c='#C44E52', marker='*', edgecolors='black', linewidths=0.5,
               label='BOS token', zorder=10)

    # Add annotation for BOS
    ax.annotate('BOS\n(token 50256)',
                xy=(tsne_result[bos_idx, 0], tsne_result[bos_idx, 1]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#C44E52', alpha=0.9))

    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title(f'Token Embeddings ({model_name} Trained)', fontweight='bold')

    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'tsne_embeddings_{model_name}.png')
    pdf_path = os.path.join(output_dir, f'tsne_embeddings_{model_name}.pdf')

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.close()

    return png_path, pdf_path


def compute_bos_distance_stats(embeddings: np.ndarray) -> dict:
    """Compute statistics about BOS token's distance from other embeddings."""
    bos_emb = embeddings[BOS_TOKEN_ID]

    # Compute distances to all other tokens
    distances = np.linalg.norm(embeddings - bos_emb, axis=1)

    # Exclude BOS itself
    other_distances = np.delete(distances, BOS_TOKEN_ID)

    stats = {
        'bos_norm': float(np.linalg.norm(bos_emb)),
        'mean_distance_from_bos': float(np.mean(other_distances)),
        'std_distance_from_bos': float(np.std(other_distances)),
        'min_distance_from_bos': float(np.min(other_distances)),
        'max_distance_from_bos': float(np.max(other_distances)),
        'median_distance_from_bos': float(np.median(other_distances)),
    }

    # Compute mean norm of regular tokens for comparison
    regular_norms = np.linalg.norm(embeddings[:BOS_TOKEN_ID], axis=1)
    stats['mean_regular_token_norm'] = float(np.mean(regular_norms))
    stats['std_regular_token_norm'] = float(np.std(regular_norms))

    return stats


def main():
    # Paths
    checkpoint_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt'
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/tsne_embeddings'

    # Load embeddings
    embeddings, config = load_checkpoint_embeddings(checkpoint_path)

    # Compute BOS distance statistics
    print("\n" + "=" * 50)
    print("BOS Token Distance Statistics")
    print("=" * 50)
    stats = compute_bos_distance_stats(embeddings)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 50 + "\n")

    # Run t-SNE
    tsne_result, sample_indices = run_tsne(embeddings, n_samples=5000)

    # Create ICML-quality plot
    png_path, pdf_path = create_icml_tsne_plot(tsne_result, sample_indices, output_dir)

    print("\nDone!")
    print(f"Output: {output_dir}")

    return {
        'stats': stats,
        'png_path': png_path,
        'pdf_path': pdf_path
    }


if __name__ == '__main__':
    main()
