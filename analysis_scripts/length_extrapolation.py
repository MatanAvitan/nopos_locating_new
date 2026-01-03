"""
Length Extrapolation Experiment
Tests generalization to sequences longer than training (64 → 128 tokens).
Evaluates if the implicit positional encoding mechanism can extrapolate to unseen positions.
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformer_lens import HookedTransformer, HookedTransformerConfig
from torch.utils.data import DataLoader, TensorDataset
from utils import device
import sys
sys.path.append('..')
from evaluation import evaluate
import glob


TRAIN_LENGTH = 64
TEST_LENGTHS = [64, 80, 96, 112, 128]


def find_best_trained_model():
    """Find the best trained model from Phase 2 experiments."""
    model_dirs = glob.glob('models/*synthetic*w_ln*')

    if not model_dirs:
        raise FileNotFoundError("No trained models found. Please run Phase 2 experiments first.")

    best_model = None
    best_acc = 0.0

    for model_dir in model_dirs:
        results_file = Path(model_dir) / 'results.json'
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                acc = results.get('accuracy', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_dir

    if best_model is None:
        best_model = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)

    ckpt_files = list(Path(best_model).glob('*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {best_model}")

    print(f"Using model: {best_model}")
    print(f"Best accuracy: {best_acc:.4f}")

    return ckpt_files[0], best_acc


def load_model_and_extend_context(ckpt_path, new_n_ctx):
    """
    Load model and extend context length.

    Args:
        ckpt_path: Path to checkpoint
        new_n_ctx: New context length

    Returns:
        Extended model
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Create extended config
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=1024,
        d_head=1024,
        n_heads=1,
        d_mlp=4096,
        d_vocab=5000,  # Default synthetic vocab
        n_ctx=new_n_ctx,  # Extended context
        act_fn='relu',
        normalization_type='LN',
        device=device
    )

    # Create model with extended context
    model = HookedTransformer(cfg)

    # Load weights (careful with position embeddings)
    try:
        state_dict = checkpoint['state_dict']

        # Remove Lightning wrapper prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_key] = v

        # Load, allowing position embedding size mismatch
        model.load_state_dict(new_state_dict, strict=False)

        # Ensure positional embeddings are deactivated
        model.pos_embed.W_pos.data[:] = 0.0
        model.pos_embed.W_pos.requires_grad = False

        model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def evaluate_at_length(model, test_length, vocab_size=5000, n_samples=10000):
    """
    Evaluate model on sequences of given length.

    Args:
        model: HookedTransformer
        test_length: Sequence length to test
        vocab_size: Vocabulary size
        n_samples: Number of test samples

    Returns:
        Dict with results
    """
    print(f"\nEvaluating at length {test_length}...")

    # Generate test data
    test_tokens = torch.randint(0, vocab_size, (n_samples, test_length))
    test_labels = torch.arange(test_length).expand(n_samples, -1)

    test_dataset = TensorDataset(test_tokens, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=4)

    # Evaluate
    model.eval()
    total_correct = 0
    total_tokens = 0
    per_position_correct = torch.zeros(test_length)
    per_position_total = torch.zeros(test_length)

    with torch.no_grad():
        for batch_tokens, batch_labels in test_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_tokens)
            predictions = logits.argmax(dim=-1)

            # Overall accuracy
            correct = (predictions == batch_labels)
            total_correct += correct.sum().item()
            total_tokens += batch_labels.numel()

            # Per-position accuracy
            for pos in range(test_length):
                per_position_correct[pos] += correct[:, pos].sum().item()
                per_position_total[pos] += batch_labels.size(0)

    overall_accuracy = total_correct / total_tokens
    per_position_accuracy = (per_position_correct / per_position_total).numpy()

    print(f"  Overall accuracy: {overall_accuracy:.4f}")
    print(f"  Accuracy on seen positions (0-{TRAIN_LENGTH-1}): {per_position_accuracy[:TRAIN_LENGTH].mean():.4f}")
    if test_length > TRAIN_LENGTH:
        print(f"  Accuracy on unseen positions ({TRAIN_LENGTH}-{test_length-1}): {per_position_accuracy[TRAIN_LENGTH:].mean():.4f}")

    return {
        'test_length': test_length,
        'overall_accuracy': float(overall_accuracy),
        'per_position_accuracy': per_position_accuracy.tolist(),
        'seen_positions_accuracy': float(per_position_accuracy[:TRAIN_LENGTH].mean()),
        'unseen_positions_accuracy': float(per_position_accuracy[TRAIN_LENGTH:].mean()) if test_length > TRAIN_LENGTH else None
    }


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("LENGTH EXTRAPOLATION EXPERIMENT")
    print("="*60)
    print(f"Train length: {TRAIN_LENGTH}")
    print(f"Test lengths: {TEST_LENGTHS}")
    print("="*60 + "\n")

    # Find and load model
    try:
        ckpt_path, train_acc = find_best_trained_model()
    except Exception as e:
        print(f"ERROR: Could not find model: {e}")
        print("\nPlease run Phase 2 experiments first.")
        return

    all_results = []

    for test_length in TEST_LENGTHS:
        try:
            # Load model with extended context
            model = load_model_and_extend_context(ckpt_path, test_length)

            # Evaluate
            result = evaluate_at_length(model, test_length)
            all_results.append(result)

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR at length {test_length}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    output_file = Path('results/length_extrapolation_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'train_length': TRAIN_LENGTH,
        'test_lengths': TEST_LENGTHS,
        'train_accuracy': train_acc,
        'results': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print("EXTRAPOLATION SUMMARY")
    print("="*60)
    for result in all_results:
        length = result['test_length']
        overall = result['overall_accuracy']
        seen = result['seen_positions_accuracy']
        unseen = result.get('unseen_positions_accuracy')

        print(f"Length {length:3d}: overall={overall:.4f}, seen={seen:.4f}", end="")
        if unseen is not None:
            print(f", unseen={unseen:.4f}")
        else:
            print()

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
