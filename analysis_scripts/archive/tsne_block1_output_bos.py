"""
t-SNE Visualization of Block 1 Output Representations

Creates an ICML-quality figure showing t-SNE projection of representations
after block 1 (before block 2 attention), with BOS token highlighted.

Usage:
    python analysis_scripts/tsne_block1_output_bos.py
"""

import os
import sys
import numpy as np
import torch
from contextlib import nullcontext
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint.get('config', {})

    # Create model config - use fixed block_size=128 as that's what was trained
    model_config = TwoLayerMechanismConfig(
        block_size=128,  # Fixed - training used 128
        vocab_size=config_dict.get('vocab_size', 50304),
        n_embd=config_dict.get('n_embd', 768),
        n_head=config_dict.get('n_head', 12),
        dropout=0.0,
        norm_type=config_dict.get('norm_type', 'layernorm'),
        use_regression=True,
    )

    model = TwoLayerMechanismModel(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, model_config


def load_data(data_path: str):
    """Load memory-mapped data."""
    return np.memmap(data_path, dtype=np.uint16, mode='r')


def get_block1_representations(model, data, n_samples: int = 1000,
                                block_size: int = 128, device: str = 'cuda'):
    """
    Run data through model and collect block 1 output representations.

    Returns:
        representations: [n_samples * block_size, n_embd] array
        token_ids: [n_samples * block_size] array of token IDs
        positions: [n_samples * block_size] array of position indices
    """
    all_representations = []
    all_token_ids = []
    all_positions = []

    batch_size = 32  # Reduced for memory
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"Collecting block 1 representations from {n_samples} sequences...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            # Sample random sequences
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            if current_batch_size <= 0:
                break

            ix = np.random.randint(0, len(data) - block_size, size=current_batch_size)
            x = torch.stack([
                torch.from_numpy(data[i:i + block_size].astype(np.int64))
                for i in ix
            ]).to(device)

            # Forward through embeddings and block 1
            tok_emb = model.wte(x)  # [B, T, C]
            h = model.drop(tok_emb)

            # Pass through block 1 with tap capture
            block1_out = model.block1(h, capture_taps=True)

            # Get the block output (after residual)
            representations = block1_out.cpu().numpy()  # [B, T, C]

            # Flatten and store
            B, T, C = representations.shape
            all_representations.append(representations.reshape(-1, C))
            all_token_ids.append(x.cpu().numpy().reshape(-1))
            all_positions.append(
                np.tile(np.arange(T), B)
            )

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} sequences...")

    representations = np.concatenate(all_representations, axis=0)
    token_ids = np.concatenate(all_token_ids, axis=0)
    positions = np.concatenate(all_positions, axis=0)

    print(f"Collected {len(representations)} representations")
    return representations, token_ids, positions


def run_tsne_on_representations(representations: np.ndarray, token_ids: np.ndarray,
                                 positions: np.ndarray, n_samples: int = 10000,
                                 perplexity: int = 50, random_state: int = 42):
    """Run t-SNE on sampled representations."""

    total_points = len(representations)

    # Find BOS token indices
    bos_indices = np.where(token_ids == BOS_TOKEN_ID)[0]
    print(f"Found {len(bos_indices)} BOS token occurrences out of {total_points} total")

    # Sample points, ensuring we include BOS tokens
    np.random.seed(random_state)

    if len(bos_indices) > 0:
        # Include all BOS tokens (up to a limit)
        max_bos = min(len(bos_indices), n_samples // 10)  # At most 10% BOS
        bos_sample = np.random.choice(bos_indices, size=max_bos, replace=False)

        # Sample remaining from non-BOS
        non_bos_indices = np.where(token_ids != BOS_TOKEN_ID)[0]
        n_non_bos = n_samples - len(bos_sample)
        non_bos_sample = np.random.choice(non_bos_indices, size=min(n_non_bos, len(non_bos_indices)), replace=False)

        sample_indices = np.concatenate([bos_sample, non_bos_sample])
    else:
        sample_indices = np.random.choice(total_points, size=min(n_samples, total_points), replace=False)

    sample_reps = representations[sample_indices]
    sample_tokens = token_ids[sample_indices]
    sample_positions = positions[sample_indices]

    print(f"Running t-SNE on {len(sample_indices)} points (including {np.sum(sample_tokens == BOS_TOKEN_ID)} BOS tokens)...")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000, learning_rate='auto', init='pca')
    tsne_result = tsne.fit_transform(sample_reps)

    return tsne_result, sample_tokens, sample_positions


def create_icml_tsne_plot(tsne_result: np.ndarray, token_ids: np.ndarray,
                          positions: np.ndarray, output_dir: str):
    """Create ICML-quality t-SNE plot with BOS token highlighted."""

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    # Separate BOS and non-BOS
    bos_mask = token_ids == BOS_TOKEN_ID
    non_bos_mask = ~bos_mask

    n_bos = np.sum(bos_mask)
    n_non_bos = np.sum(non_bos_mask)

    # Plot non-BOS tokens
    scatter = ax.scatter(tsne_result[non_bos_mask, 0], tsne_result[non_bos_mask, 1],
                         s=3, alpha=0.3, c='#4C72B0', label=f'Regular tokens (n={n_non_bos})',
                         rasterized=True)

    # Plot BOS tokens prominently
    if n_bos > 0:
        ax.scatter(tsne_result[bos_mask, 0], tsne_result[bos_mask, 1],
                   s=50, c='#C44E52', marker='*', edgecolors='black', linewidths=0.3,
                   label=f'BOS tokens (n={n_bos})', zorder=10, alpha=0.9)

        # Add annotation pointing to BOS cluster centroid
        bos_centroid = tsne_result[bos_mask].mean(axis=0)
        ax.annotate('BOS tokens\n(ID 50256)',
                    xy=(bos_centroid[0], bos_centroid[1]),
                    xytext=(20, 20), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#C44E52', alpha=0.9))

    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title('Block 1 Output Representations', fontweight='bold')

    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', markerscale=1.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'tsne_block1_output_bos.png')
    pdf_path = os.path.join(output_dir, 'tsne_block1_output_bos.pdf')

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.close()

    return png_path, pdf_path


def compute_bos_representation_stats(representations: np.ndarray, token_ids: np.ndarray):
    """Compute statistics comparing BOS representations to others."""
    bos_mask = token_ids == BOS_TOKEN_ID
    non_bos_mask = ~bos_mask

    if np.sum(bos_mask) == 0:
        return {"error": "No BOS tokens found"}

    bos_reps = representations[bos_mask]
    non_bos_reps = representations[non_bos_mask]

    # Norms
    bos_norms = np.linalg.norm(bos_reps, axis=1)
    non_bos_norms = np.linalg.norm(non_bos_reps, axis=1)

    # Mean representations
    bos_mean = bos_reps.mean(axis=0)
    non_bos_mean = non_bos_reps.mean(axis=0)

    # Distance between means
    mean_distance = np.linalg.norm(bos_mean - non_bos_mean)

    # Cosine similarity of means
    cos_sim = np.dot(bos_mean, non_bos_mean) / (np.linalg.norm(bos_mean) * np.linalg.norm(non_bos_mean))

    stats = {
        'n_bos': int(np.sum(bos_mask)),
        'n_non_bos': int(np.sum(non_bos_mask)),
        'bos_mean_norm': float(np.mean(bos_norms)),
        'bos_std_norm': float(np.std(bos_norms)),
        'non_bos_mean_norm': float(np.mean(non_bos_norms)),
        'non_bos_std_norm': float(np.std(non_bos_norms)),
        'mean_representation_distance': float(mean_distance),
        'mean_representation_cosine_sim': float(cos_sim),
    }

    return stats


def main():
    # Paths
    checkpoint_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt'
    data_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext/val.bin'
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/tsne_embeddings'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)

    # Load data
    data = load_data(data_path)
    print(f"Loaded {len(data):,} tokens from validation data")

    # Get block 1 representations
    representations, token_ids, positions = get_block1_representations(
        model, data, n_samples=500, block_size=config.block_size, device=device
    )

    # Compute statistics
    print("\n" + "=" * 60)
    print("BOS Token Representation Statistics (Block 1 Output)")
    print("=" * 60)
    stats = compute_bos_representation_stats(representations, token_ids)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # Run t-SNE
    tsne_result, sample_tokens, sample_positions = run_tsne_on_representations(
        representations, token_ids, positions, n_samples=8000, perplexity=50
    )

    # Create ICML-quality plot
    png_path, pdf_path = create_icml_tsne_plot(
        tsne_result, sample_tokens, sample_positions, output_dir
    )

    print("\nDone!")
    print(f"Output: {output_dir}")

    return {
        'stats': stats,
        'png_path': png_path,
        'pdf_path': pdf_path
    }


if __name__ == '__main__':
    main()
