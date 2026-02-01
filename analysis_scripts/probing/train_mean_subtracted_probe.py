"""
Experiment B4: Train Probe on Mean-Subtracted Activations

This is the PRIORITY experiment that definitively tests whether Mechanism 1
(individual sample features) alone is sufficient for position decoding,
or whether Mechanism 2 (population mean) is also necessary.

Key Question:
If we subtract the population mean μ_i from each position's activation h_i,
can a probe still predict position from the residual r_i = h_i - μ_i?

Interpretation:
- If residual probe accuracy ≈ baseline → Mechanism 1 is sufficient
- If residual probe accuracy << baseline → Mechanism 2 (population mean) is necessary

Usage:
    python train_mean_subtracted_probe.py --setting synthetic
    python train_mean_subtracted_probe.py --setting synthetic --skip_baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr

from transformer_lens import HookedTransformer, HookedTransformerConfig

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ARTIFACTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/artifacts")
PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")

# Default settings
DEFAULT_N_CTX = 64
DEFAULT_D_MODEL = 1024
DEFAULT_D_VOCAB = 5000
DEFAULT_N_TRAIN_SAMPLES = 10000
DEFAULT_N_TEST_SAMPLES = 2000
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3


class PositionProbe(nn.Module):
    """MLP probe for position prediction."""

    def __init__(self, d_model, n_ctx, hidden_mult=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_mult * d_model),
            nn.ReLU(),
            nn.Linear(hidden_mult * d_model, n_ctx)
        )

    def forward(self, x):
        return self.mlp(x)


def create_synthetic_model(d_model, d_vocab, n_ctx):
    """Create a single-layer transformer with frozen embeddings and attention."""
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_model,
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

    return model


def generate_activations(model, n_samples, n_ctx, d_vocab, pop_means=None, batch_size=64):
    """
    Generate activations from the model.

    Args:
        model: HookedTransformer model
        n_samples: Number of samples to generate
        n_ctx: Context length
        d_vocab: Vocabulary size
        pop_means: If provided, subtract population means (for residual activations)
        batch_size: Batch size for generation

    Returns:
        activations: [n_samples * n_ctx, d_model]
        positions: [n_samples * n_ctx]
    """
    hook_name = 'blocks.0.ln2.hook_normalized'
    all_acts = []
    all_positions = []

    model.eval()
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Generating activations"):
            current_batch_size = min(batch_size, n_samples - len(all_acts))
            if current_batch_size <= 0:
                break

            tokens = torch.randint(0, d_vocab, (current_batch_size, n_ctx), device=DEVICE)
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name].detach()  # [batch, n_ctx, d_model]

            # Subtract population means if provided (for residual activations)
            if pop_means is not None:
                acts = acts - pop_means.to(DEVICE).unsqueeze(0)

            # Flatten: [batch, n_ctx, d_model] -> [batch * n_ctx, d_model]
            acts_flat = acts.view(-1, acts.shape[-1]).cpu()
            positions = torch.arange(n_ctx).repeat(current_batch_size).cpu()

            all_acts.append(acts_flat)
            all_positions.append(positions)

            del cache
            torch.cuda.empty_cache()

    activations = torch.cat(all_acts, dim=0)
    positions = torch.cat(all_positions, dim=0)

    return activations, positions


def train_probe(train_acts, train_positions, d_model, n_ctx,
                epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH_SIZE):
    """
    Train a position probe on activations.

    Returns:
        probe: Trained PositionProbe
        train_history: List of (epoch, loss, accuracy) tuples
    """
    probe = PositionProbe(d_model, n_ctx).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Create data loader
    dataset = torch.utils.data.TensorDataset(train_acts, train_positions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_history = []

    for epoch in range(epochs):
        probe.train()
        total_loss = 0
        correct = 0
        total = 0

        for acts, positions in loader:
            acts = acts.to(DEVICE)
            positions = positions.to(DEVICE)

            optimizer.zero_grad()
            logits = probe(acts)
            loss = criterion(logits, positions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * acts.shape[0]
            preds = logits.argmax(dim=-1)
            correct += (preds == positions).sum().item()
            total += acts.shape[0]

        scheduler.step()

        avg_loss = total_loss / total
        accuracy = correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

        train_history.append((epoch + 1, avg_loss, accuracy))

    return probe, train_history


def evaluate_probe(probe, test_acts, test_positions, n_ctx):
    """
    Evaluate a trained probe.

    Returns:
        dict with accuracy, per-position accuracy, and correlation metrics
    """
    probe.eval()
    batch_size = 1024

    dataset = torch.utils.data.TensorDataset(test_acts, test_positions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for acts, positions in loader:
            acts = acts.to(DEVICE)
            logits = probe(acts)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_targets.append(positions)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Overall accuracy
    accuracy = (all_preds == all_targets).mean()

    # Per-position accuracy
    per_pos_accuracy = np.zeros(n_ctx)
    for pos in range(n_ctx):
        mask = all_targets == pos
        if mask.sum() > 0:
            per_pos_accuracy[pos] = (all_preds[mask] == pos).mean()

    # Correlation between predictions and targets
    pearson_r, p_val = pearsonr(all_preds, all_targets)

    # Mean Absolute Error
    mae = np.abs(all_preds - all_targets).mean()

    return {
        'accuracy': accuracy,
        'per_position_accuracy': per_pos_accuracy,
        'pearson_r': pearson_r,
        'p_value': p_val,
        'mae': mae,
        'predictions': all_preds,
        'targets': all_targets
    }


def run_experiment(args):
    """Run the B4 experiment: compare baseline vs residual probe."""

    print("=" * 70)
    print("Experiment B4: Train Probe on Mean-Subtracted Activations")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  d_model: {args.d_model}")
    print(f"  d_vocab: {args.d_vocab}")
    print(f"  n_ctx: {args.n_ctx}")
    print(f"  n_train_samples: {args.n_train}")
    print(f"  n_test_samples: {args.n_test}")
    print(f"  epochs: {args.epochs}")
    print(f"  device: {DEVICE}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create model
    print("\n[Step 1] Creating model...")
    model = create_synthetic_model(args.d_model, args.d_vocab, args.n_ctx)

    # Load or compute population means
    print("\n[Step 2] Loading population means...")
    pop_means_path = ARTIFACTS_DIR / f"population_means_{args.setting}.pt"

    if pop_means_path.exists():
        data = torch.load(pop_means_path)
        pop_means = data['pop_means']
        print(f"  Loaded from: {pop_means_path}")
    else:
        print(f"  Population means not found. Computing now...")
        from compute_population_means import compute_population_means
        pop_means, _ = compute_population_means(
            model, n_samples=5000, n_ctx=args.n_ctx, d_vocab=args.d_vocab
        )
        # Save for future use
        torch.save({'pop_means': pop_means}, pop_means_path)
        print(f"  Saved to: {pop_means_path}")

    results = {}

    # =========================================================================
    # Baseline Probe (on raw activations)
    # =========================================================================
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("[Baseline] Training probe on RAW activations")
        print("=" * 70)

        print("\n  Generating training activations...")
        train_acts, train_pos = generate_activations(
            model, args.n_train, args.n_ctx, args.d_vocab, pop_means=None
        )

        print(f"  Training data shape: {train_acts.shape}")
        print("\n  Training probe...")
        baseline_probe, baseline_history = train_probe(
            train_acts, train_pos, args.d_model, args.n_ctx, epochs=args.epochs
        )

        print("\n  Generating test activations...")
        test_acts, test_pos = generate_activations(
            model, args.n_test, args.n_ctx, args.d_vocab, pop_means=None
        )

        print("\n  Evaluating...")
        baseline_results = evaluate_probe(baseline_probe, test_acts, test_pos, args.n_ctx)

        print(f"\n  [Baseline Results]")
        print(f"    Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"    Pearson r: {baseline_results['pearson_r']:.4f}")
        print(f"    MAE: {baseline_results['mae']:.2f}")

        results['baseline'] = baseline_results
        results['baseline_probe'] = baseline_probe
        results['baseline_history'] = baseline_history

    # =========================================================================
    # Residual Probe (on mean-subtracted activations)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Residual] Training probe on MEAN-SUBTRACTED activations (h - μ)")
    print("=" * 70)

    print("\n  Generating training activations with mean subtraction...")
    train_acts_residual, train_pos_residual = generate_activations(
        model, args.n_train, args.n_ctx, args.d_vocab, pop_means=pop_means
    )

    print(f"  Training data shape: {train_acts_residual.shape}")
    print("\n  Training probe...")
    residual_probe, residual_history = train_probe(
        train_acts_residual, train_pos_residual, args.d_model, args.n_ctx, epochs=args.epochs
    )

    print("\n  Generating test activations with mean subtraction...")
    test_acts_residual, test_pos_residual = generate_activations(
        model, args.n_test, args.n_ctx, args.d_vocab, pop_means=pop_means
    )

    print("\n  Evaluating...")
    residual_results = evaluate_probe(residual_probe, test_acts_residual, test_pos_residual, args.n_ctx)

    print(f"\n  [Residual Results]")
    print(f"    Accuracy: {residual_results['accuracy']:.4f}")
    print(f"    Pearson r: {residual_results['pearson_r']:.4f}")
    print(f"    MAE: {residual_results['mae']:.2f}")

    results['residual'] = residual_results
    results['residual_probe'] = residual_probe
    results['residual_history'] = residual_history

    # =========================================================================
    # Summary and Interpretation
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT B4 SUMMARY")
    print("=" * 70)

    if 'baseline' in results:
        baseline_acc = results['baseline']['accuracy']
        residual_acc = results['residual']['accuracy']
        relative_drop = (baseline_acc - residual_acc) / baseline_acc * 100

        print(f"\n  Baseline probe accuracy:  {baseline_acc:.4f}")
        print(f"  Residual probe accuracy:  {residual_acc:.4f}")
        print(f"  Relative accuracy drop:   {relative_drop:.1f}%")

        print("\n  INTERPRETATION:")
        if relative_drop < 5:
            print("    ✓ Residual probe ≈ Baseline")
            print("    → Mechanism 1 (individual sample features) is SUFFICIENT")
            print("    → Population mean (Mechanism 2) is NOT necessary for decoding")
        elif relative_drop < 20:
            print("    ~ Residual probe slightly worse than Baseline")
            print("    → Both mechanisms contribute, but Mechanism 1 is primary")
        else:
            print("    ✗ Residual probe << Baseline")
            print("    → Mechanism 2 (population mean) is NECESSARY")
            print("    → The model relies on population statistics for position decoding")
    else:
        print(f"\n  Residual probe accuracy: {results['residual']['accuracy']:.4f}")
        print("  (Run without --skip_baseline to see comparison)")

    # Save results
    save_path = ARTIFACTS_DIR / f"experiment_b4_{args.setting}.pt"
    torch.save({
        'results': {k: v for k, v in results.items() if not k.endswith('_probe')},
        'config': vars(args)
    }, save_path)
    print(f"\n  Results saved to: {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment B4: Mean-Subtracted Probe Training')
    parser.add_argument('--setting', type=str, default='synthetic',
                        choices=['synthetic', 'natural_language'])
    parser.add_argument('--d_model', type=int, default=DEFAULT_D_MODEL)
    parser.add_argument('--d_vocab', type=int, default=DEFAULT_D_VOCAB)
    parser.add_argument('--n_ctx', type=int, default=DEFAULT_N_CTX)
    parser.add_argument('--n_train', type=int, default=DEFAULT_N_TRAIN_SAMPLES)
    parser.add_argument('--n_test', type=int, default=DEFAULT_N_TEST_SAMPLES)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline probe training')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
