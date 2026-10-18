"""
Sample Convergence Analysis
Demonstrates pattern emergence with increasing sample sizes for Figure 6.
Shows how positional patterns emerge at the population level as sample size increases.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader, TensorDataset
from utils import device
import sys
import glob
from scipy.stats import pearsonr

SAMPLE_SIZES = [10, 50, 100, 250, 500, 1000, 2000]
HOOK_NAME = 'blocks.0.ln2.hook_normalized'
N_CTX = 64


def find_best_trained_model():
    """
    Find the best trained model from Phase 2 experiments.

    Returns:
        Path to model checkpoint
    """
    # Look for natural language models first (preferred)
    model_dirs = glob.glob('models/*natural*')

    if not model_dirs:
        # Fallback to synthetic models
        model_dirs = glob.glob('models/*synthetic*')

    if not model_dirs:
        raise FileNotFoundError("No trained models found. Please run Phase 2 experiments first.")

    # Find the most recent model with best results
    best_model = None
    best_acc = 0.0

    for model_dir in model_dirs:
        results_file = Path(model_dir) / 'results.json'
        if results_file.exists():
            import json
            with open(results_file) as f:
                results = json.load(f)
                acc = results.get('accuracy', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_dir

    if best_model is None:
        # Just use most recent
        best_model = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)

    # Find checkpoint file
    ckpt_files = list(Path(best_model).glob('*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {best_model}")

    print(f"Using model: {best_model}")
    print(f"Checkpoint: {ckpt_files[0]}")
    print(f"Best accuracy: {best_acc:.4f}")

    return ckpt_files[0]


def load_model_from_checkpoint(ckpt_path):
    """
    Load HookedTransformer from PyTorch Lightning checkpoint.

    Args:
        ckpt_path: Path to checkpoint

    Returns:
        HookedTransformer model
    """
    from nopos_lit_model import NoposLitTransformer

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract config from checkpoint or reconstruct
    # Most models use these defaults
    from transformer_lens import HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=2048,  # Natural language default
        d_head=2048,
        n_heads=1,
        d_mlp=8192,
        d_vocab=50257,  # GPT-2 tokenizer
        n_ctx=N_CTX,
        act_fn='relu',
        normalization_type='LN',
        device=device
    )

    # Try to create model and load state dict
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer(cfg)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Could not load with config, trying alternatives: {e}")
        # Try loading with different d_model sizes
        for d_model in [1024, 2048, 512]:
            try:
                cfg.d_model = d_model
                cfg.d_head = d_model
                cfg.d_mlp = d_model * 4
                model = HookedTransformer(cfg)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.to(device)
                model.eval()
                print(f"Successfully loaded with d_model={d_model}")
                return model
            except:
                continue

        raise RuntimeError("Could not load model from checkpoint")


def extract_activations_for_samples(model, n_samples, vocab_size=50257):
    """
    Extract LayerNorm activations for N samples.

    Args:
        model: HookedTransformer
        n_samples: Number of samples to collect
        vocab_size: Vocabulary size

    Returns:
        Tensor of activations [n_samples, n_ctx, d_model]
    """
    model.eval()
    activations = []

    batch_size = min(100, n_samples)
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)

            # Generate random tokens
            tokens = torch.randint(0, vocab_size, (current_batch_size, N_CTX)).to(device)

            # Run with cache to extract activations
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])

            # Extract activations
            acts = cache[HOOK_NAME].detach().cpu()  # [batch, n_ctx, d_model]
            activations.append(acts)

    return torch.cat(activations, dim=0)[:n_samples]  # Ensure exact count


def analyze_sample_convergence(model, vocab_size=50257):
    """
    Analyze how positional patterns emerge with increasing samples.

    Args:
        model: Trained HookedTransformer
        vocab_size: Vocabulary size

    Returns:
        Dict with analysis results
    """
    results = {
        'sample_sizes': SAMPLE_SIZES,
        'correlations': [],
        'pattern_strengths': [],
        'activations_by_size': {}
    }

    print("\n" + "="*60)
    print("SAMPLE CONVERGENCE ANALYSIS")
    print("="*60)
    print(f"Model: {model}")
    print(f"Hook: {HOOK_NAME}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print("="*60 + "\n")

    for n_samples in SAMPLE_SIZES:
        print(f"\nAnalyzing {n_samples} samples...")

        # Extract activations
        activations = extract_activations_for_samples(model, n_samples, vocab_size)
        # [n_samples, n_ctx, d_model]

        # Compute population average
        pop_avg = activations.mean(dim=0)  # [n_ctx, d_model]

        # Compute position-wise mean across dimensions
        pos_pattern = pop_avg.mean(dim=1).numpy()  # [n_ctx]

        # Calculate correlation with position index (monotonicity)
        positions = np.arange(N_CTX)
        correlation, p_value = pearsonr(positions, pos_pattern)

        # Pattern strength (variance of position pattern)
        pattern_strength = pos_pattern.var()

        results['correlations'].append(float(correlation))
        results['pattern_strengths'].append(float(pattern_strength))
        results['activations_by_size'][n_samples] = activations

        print(f"  Correlation: {correlation:.4f} (p={p_value:.6f})")
        print(f"  Pattern strength: {pattern_strength:.6f}")
        print(f"  Pattern range: [{pos_pattern.min():.4f}, {pos_pattern.max():.4f}]")

    print("\n" + "="*60)
    print("CONVERGENCE SUMMARY")
    print("="*60)
    for i, n_samples in enumerate(SAMPLE_SIZES):
        corr = results['correlations'][i]
        strength = results['pattern_strengths'][i]
        print(f"  {n_samples:4d} samples: corr={corr:+.4f}, strength={strength:.6f}")

    return results


def main():
    """
    Main execution function.
    """
    print("Starting Sample Convergence Analysis...")

    # Find and load trained model
    try:
        ckpt_path = find_best_trained_model()
        model = load_model_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        print("\nPlease ensure Phase 2 experiments have been run first.")
        print("Expected model locations: models/*natural* or models/*synthetic*")
        import traceback
        traceback.print_exc()
        return

    # Determine vocab size from model
    vocab_size = model.cfg.d_vocab
    print(f"Model vocabulary size: {vocab_size}")

    # Run analysis
    try:
        results = analyze_sample_convergence(model, vocab_size)

        # Save results
        output_file = Path('results/sample_convergence_data.pkl')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n✓ Results saved to: {output_file}")

        # Also save JSON summary (without full activations)
        import json
        summary = {
            'sample_sizes': results['sample_sizes'],
            'correlations': results['correlations'],
            'pattern_strengths': results['pattern_strengths'],
            'vocab_size': vocab_size,
            'n_ctx': N_CTX
        }

        with open(output_file.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Summary saved to: {output_file.with_suffix('.json')}")

    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
