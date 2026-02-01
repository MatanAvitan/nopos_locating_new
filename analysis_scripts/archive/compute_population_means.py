"""
Compute Population Means for Position Decoding Intervention Experiments

This script computes μ_i = E_samples[LN(h_i)] for each position i,
which is used in intervention experiments to test whether the model
uses population-level statistics for position decoding.

Usage:
    python compute_population_means.py --setting synthetic
    python compute_population_means.py --setting natural_language
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ARTIFACTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Default settings
DEFAULT_N_SAMPLES = 5000
DEFAULT_N_CTX = 64
DEFAULT_D_MODEL = 1024
DEFAULT_D_VOCAB = 5000


def create_synthetic_model(d_model=DEFAULT_D_MODEL, d_vocab=DEFAULT_D_VOCAB, n_ctx=DEFAULT_N_CTX):
    """Create a single-layer transformer with frozen embeddings and attention."""
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_model,  # Single head
        n_heads=1,
        d_mlp=4 * d_model,
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        act_fn='relu',
        normalization_type='LN',
        device=DEVICE
    )
    model = HookedTransformer(cfg)

    # Deactivate positional embeddings
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False

    # Freeze embeddings and attention
    model.embed.W_E.requires_grad = False
    model.blocks[0].attn.W_Q.requires_grad = False
    model.blocks[0].attn.W_K.requires_grad = False
    model.blocks[0].attn.W_V.requires_grad = False
    model.blocks[0].attn.W_O.requires_grad = False

    return model


def compute_population_means(model, n_samples=DEFAULT_N_SAMPLES, n_ctx=DEFAULT_N_CTX,
                             d_vocab=DEFAULT_D_VOCAB, batch_size=64):
    """
    Compute μ_i = E_samples[LN(h_i)] for each position i.

    Args:
        model: HookedTransformer model
        n_samples: Number of samples to average over
        n_ctx: Context length
        d_vocab: Vocabulary size for random token generation
        batch_size: Batch size for efficient computation

    Returns:
        pop_means: [n_ctx, d_model] - population mean for each position
        pop_stds: [n_ctx, d_model] - population std for each position
    """
    hook_name = 'blocks.0.ln2.hook_normalized'
    all_acts = []

    model.eval()
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Computing population means"):
            current_batch_size = min(batch_size, n_samples - len(all_acts) * batch_size)
            if current_batch_size <= 0:
                break

            tokens = torch.randint(0, d_vocab, (current_batch_size, n_ctx), device=DEVICE)
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name].detach().cpu()  # [batch, n_ctx, d_model]
            all_acts.append(acts)

            del cache
            torch.cuda.empty_cache()

    all_acts = torch.cat(all_acts, dim=0)[:n_samples]  # [n_samples, n_ctx, d_model]

    pop_means = all_acts.mean(dim=0)  # [n_ctx, d_model]
    pop_stds = all_acts.std(dim=0)    # [n_ctx, d_model]

    return pop_means, pop_stds


def analyze_population_means(pop_means, pop_stds):
    """
    Analyze the computed population means to verify positional structure.

    Returns:
        dict with analysis results
    """
    from scipy.stats import pearsonr, spearmanr

    n_ctx, d_model = pop_means.shape
    positions = np.arange(n_ctx)

    # Compute correlation of each dimension with position
    dim_correlations = []
    for d in range(d_model):
        corr, _ = pearsonr(positions, pop_means[:, d].numpy())
        dim_correlations.append(corr)

    dim_correlations = np.array(dim_correlations)

    # Mean across dimensions for each position
    mean_per_position = pop_means.mean(dim=1).numpy()  # [n_ctx]

    # Overall correlation of mean pattern with position
    overall_corr, p_val = pearsonr(positions, mean_per_position)
    spearman_r, _ = spearmanr(positions, mean_per_position)

    # Count dimensions with strong correlation
    n_strong_pos = (dim_correlations > 0.5).sum()
    n_strong_neg = (dim_correlations < -0.5).sum()

    results = {
        'overall_pearson_r': overall_corr,
        'overall_spearman_r': spearman_r,
        'p_value': p_val,
        'n_dims_strong_pos_corr': n_strong_pos,
        'n_dims_strong_neg_corr': n_strong_neg,
        'mean_abs_dim_corr': np.abs(dim_correlations).mean(),
        'max_dim_corr': dim_correlations.max(),
        'min_dim_corr': dim_correlations.min(),
        'dim_correlations': dim_correlations
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute population means for intervention experiments')
    parser.add_argument('--setting', type=str, default='synthetic',
                        choices=['synthetic', 'natural_language'],
                        help='Data setting: synthetic (uniform tokens) or natural_language')
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES,
                        help='Number of samples to average over')
    parser.add_argument('--d_model', type=int, default=DEFAULT_D_MODEL,
                        help='Model dimension')
    parser.add_argument('--d_vocab', type=int, default=DEFAULT_D_VOCAB,
                        help='Vocabulary size')
    parser.add_argument('--n_ctx', type=int, default=DEFAULT_N_CTX,
                        help='Context length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(f"Computing Population Means - Setting: {args.setting}")
    print("=" * 70)
    print(f"  n_samples: {args.n_samples}")
    print(f"  d_model: {args.d_model}")
    print(f"  d_vocab: {args.d_vocab}")
    print(f"  n_ctx: {args.n_ctx}")
    print(f"  device: {DEVICE}")

    # Create model
    print("\nCreating model...")
    if args.setting == 'synthetic':
        model = create_synthetic_model(
            d_model=args.d_model,
            d_vocab=args.d_vocab,
            n_ctx=args.n_ctx
        )
    else:
        # For natural language, we would load a pretrained model
        # For now, use the same architecture but note this should be extended
        print("  Note: Using synthetic model architecture for natural language setting")
        print("  TODO: Load pretrained model for natural language experiments")
        model = create_synthetic_model(
            d_model=args.d_model,
            d_vocab=args.d_vocab,
            n_ctx=args.n_ctx
        )

    # Compute population means
    print("\nComputing population means...")
    pop_means, pop_stds = compute_population_means(
        model,
        n_samples=args.n_samples,
        n_ctx=args.n_ctx,
        d_vocab=args.d_vocab
    )

    print(f"\nPopulation means shape: {pop_means.shape}")
    print(f"Population stds shape: {pop_stds.shape}")

    # Analyze
    print("\nAnalyzing population means...")
    results = analyze_population_means(pop_means, pop_stds)

    print("\n" + "=" * 70)
    print("Analysis Results")
    print("=" * 70)
    print(f"  Overall Pearson r with position: {results['overall_pearson_r']:.4f} (p={results['p_value']:.2e})")
    print(f"  Overall Spearman r with position: {results['overall_spearman_r']:.4f}")
    print(f"  Dimensions with strong positive correlation (>0.5): {results['n_dims_strong_pos_corr']}")
    print(f"  Dimensions with strong negative correlation (<-0.5): {results['n_dims_strong_neg_corr']}")
    print(f"  Mean absolute dimension correlation: {results['mean_abs_dim_corr']:.4f}")
    print(f"  Max dimension correlation: {results['max_dim_corr']:.4f}")
    print(f"  Min dimension correlation: {results['min_dim_corr']:.4f}")

    # Save results
    save_path = ARTIFACTS_DIR / f"population_means_{args.setting}.pt"
    save_data = {
        'pop_means': pop_means,
        'pop_stds': pop_stds,
        'analysis': results,
        'config': {
            'setting': args.setting,
            'n_samples': args.n_samples,
            'd_model': args.d_model,
            'd_vocab': args.d_vocab,
            'n_ctx': args.n_ctx,
            'seed': args.seed
        }
    }
    torch.save(save_data, save_path)
    print(f"\nSaved to: {save_path}")

    return pop_means, pop_stds, results


if __name__ == "__main__":
    main()
