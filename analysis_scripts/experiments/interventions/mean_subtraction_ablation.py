"""
Experiment B1: Position-Specific Mean Subtraction Ablation

This experiment tests whether a trained probe relies on the population mean
by subtracting it at test time and measuring the accuracy drop.

Hypothesis:
If the MLP learned to use population mean for position decoding,
removing it at test time should significantly hurt predictions.

Procedure:
1. Train a baseline probe on raw activations
2. At test time, subtract population mean: h'_i = h_i - μ_i
3. Measure accuracy drop compared to baseline

Also includes:
- B2: Cross-position mean patching (replace h_i with μ_j)
- B3: Mean-only prediction (use μ_i directly without sample variation)

Usage:
    python mean_subtraction_ablation.py --setting synthetic
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer, HookedTransformerConfig

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ARTIFACTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/artifacts")
PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")

# Default settings
DEFAULT_N_CTX = 64
DEFAULT_D_MODEL = 1024
DEFAULT_D_VOCAB = 5000
DEFAULT_N_TRAIN = 10000
DEFAULT_N_TEST = 2000


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


def create_model(d_model, d_vocab, n_ctx):
    """Create a single-layer transformer."""
    cfg = HookedTransformerConfig(
        n_layers=1, d_model=d_model, d_head=d_model, n_heads=1,
        d_mlp=4 * d_model, d_vocab=d_vocab, n_ctx=n_ctx,
        act_fn='relu', normalization_type='LN', device=DEVICE
    )
    model = HookedTransformer(cfg)
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False
    return model


def get_activations(model, tokens, hook_name='blocks.0.ln2.hook_normalized'):
    """Get post-LN activations for given tokens."""
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(DEVICE), names_filter=[hook_name])
        return cache[hook_name].detach()


def train_baseline_probe(model, n_samples, n_ctx, d_vocab, d_model, epochs=100):
    """Train a probe on raw activations."""
    print("  Training baseline probe...")
    probe = PositionProbe(d_model, n_ctx).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    n_batches = n_samples // batch_size

    for epoch in range(epochs):
        probe.train()
        total_loss = 0
        correct = 0
        total = 0

        for _ in range(n_batches):
            tokens = torch.randint(0, d_vocab, (batch_size, n_ctx), device=DEVICE)
            acts = get_activations(model, tokens)  # [batch, n_ctx, d_model]
            acts_flat = acts.view(-1, d_model)
            positions = torch.arange(n_ctx, device=DEVICE).repeat(batch_size)

            optimizer.zero_grad()
            logits = probe(acts_flat)
            loss = criterion(logits, positions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == positions).sum().item()
            total += positions.numel()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Loss={total_loss/n_batches:.4f}, Acc={correct/total:.4f}")

    return probe


def evaluate_probe(probe, activations, n_ctx):
    """Evaluate probe on activations. Returns accuracy and predictions."""
    probe.eval()
    with torch.no_grad():
        batch_size = activations.shape[0]
        acts_flat = activations.view(-1, activations.shape[-1])  # [B*N, D]
        positions = torch.arange(n_ctx, device=DEVICE).repeat(batch_size)

        logits = probe(acts_flat)
        preds = logits.argmax(dim=-1)

        accuracy = (preds == positions).float().mean().item()

    return accuracy, preds.cpu(), positions.cpu()


def experiment_b1_mean_subtraction(model, probe, pop_means, n_test, n_ctx, d_vocab):
    """
    B1: Subtract population mean at test time and measure accuracy drop.
    """
    print("\n[Experiment B1] Mean Subtraction at Test Time")

    # Generate test data
    batch_size = 64
    n_batches = n_test // batch_size

    baseline_correct = 0
    ablated_correct = 0
    total = 0

    for _ in tqdm(range(n_batches), desc="  Testing"):
        tokens = torch.randint(0, d_vocab, (batch_size, n_ctx), device=DEVICE)
        acts = get_activations(model, tokens)  # [batch, n_ctx, d_model]

        # Baseline: raw activations
        baseline_acc, _, _ = evaluate_probe(probe, acts, n_ctx)
        baseline_correct += baseline_acc * batch_size * n_ctx

        # Ablated: subtract population mean
        acts_ablated = acts - pop_means.to(DEVICE).unsqueeze(0)
        ablated_acc, _, _ = evaluate_probe(probe, acts_ablated, n_ctx)
        ablated_correct += ablated_acc * batch_size * n_ctx

        total += batch_size * n_ctx

    baseline_accuracy = baseline_correct / total
    ablated_accuracy = ablated_correct / total
    relative_drop = (baseline_accuracy - ablated_accuracy) / baseline_accuracy * 100

    print(f"\n  Baseline accuracy:    {baseline_accuracy:.4f}")
    print(f"  After mean ablation:  {ablated_accuracy:.4f}")
    print(f"  Relative drop:        {relative_drop:.1f}%")

    return {
        'baseline_accuracy': baseline_accuracy,
        'ablated_accuracy': ablated_accuracy,
        'relative_drop': relative_drop
    }


def experiment_b2_cross_position_patching(model, probe, pop_means, n_test, n_ctx, d_vocab):
    """
    B2: Replace h_i with μ_j (mean of a different position) and see if prediction shifts.
    """
    print("\n[Experiment B2] Cross-Position Mean Patching")

    batch_size = 64
    n_batches = n_test // batch_size

    # Track how often prediction shifts toward patched position
    shifts_toward_patch = 0
    total_patches = 0

    for _ in tqdm(range(n_batches), desc="  Patching"):
        tokens = torch.randint(0, d_vocab, (batch_size, n_ctx), device=DEVICE)
        acts = get_activations(model, tokens)  # [batch, n_ctx, d_model]

        # For each sample, randomly select a position to patch
        for b in range(batch_size):
            # Original position and a different target position
            orig_pos = np.random.randint(0, n_ctx)
            target_pos = np.random.randint(0, n_ctx)
            while target_pos == orig_pos:
                target_pos = np.random.randint(0, n_ctx)

            # Original prediction
            orig_act = acts[b:b+1, orig_pos:orig_pos+1, :]  # [1, 1, d_model]
            with torch.no_grad():
                orig_pred = probe(orig_act.view(1, -1)).argmax(dim=-1).item()

            # Patched: replace with target position's population mean
            patched_act = pop_means[target_pos].to(DEVICE).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                patched_pred = probe(patched_act.view(1, -1)).argmax(dim=-1).item()

            # Does prediction shift toward target?
            orig_dist_to_target = abs(orig_pred - target_pos)
            patched_dist_to_target = abs(patched_pred - target_pos)

            if patched_dist_to_target < orig_dist_to_target:
                shifts_toward_patch += 1
            total_patches += 1

    shift_rate = shifts_toward_patch / total_patches

    print(f"\n  Patches that shifted toward target: {shifts_toward_patch}/{total_patches}")
    print(f"  Shift rate: {shift_rate:.4f} ({shift_rate*100:.1f}%)")

    return {
        'shifts_toward_patch': shifts_toward_patch,
        'total_patches': total_patches,
        'shift_rate': shift_rate
    }


def experiment_b3_mean_only_prediction(probe, pop_means, n_ctx):
    """
    B3: Use only population means (no sample variation) for prediction.
    If population mean carries position info, this should work well.
    """
    print("\n[Experiment B3] Mean-Only Prediction")

    probe.eval()
    with torch.no_grad():
        # Just pass the population means through the probe
        logits = probe(pop_means.to(DEVICE))  # [n_ctx, n_ctx]
        preds = logits.argmax(dim=-1).cpu()

    positions = torch.arange(n_ctx)
    accuracy = (preds == positions).float().mean().item()
    pearson_r, p_val = pearsonr(preds.numpy(), positions.numpy())

    print(f"\n  Mean-only accuracy: {accuracy:.4f}")
    print(f"  Pearson r: {pearson_r:.4f} (p={p_val:.2e})")

    return {
        'accuracy': accuracy,
        'pearson_r': pearson_r,
        'predictions': preds.numpy()
    }


def run_experiments(args):
    """Run all B-series experiments."""

    print("=" * 70)
    print("Mean Subtraction Ablation Experiments (B1, B2, B3)")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create model
    print("\n[Step 1] Creating model...")
    model = create_model(args.d_model, args.d_vocab, args.n_ctx)

    # Load population means
    print("\n[Step 2] Loading population means...")
    pop_means_path = ARTIFACTS_DIR / f"population_means_{args.setting}.pt"

    if not pop_means_path.exists():
        print(f"  ERROR: Population means not found at {pop_means_path}")
        print("  Run compute_population_means.py first.")
        return

    data = torch.load(pop_means_path)
    pop_means = data['pop_means']
    print(f"  Loaded from: {pop_means_path}")

    # Train baseline probe
    print("\n[Step 3] Training baseline probe...")
    probe = train_baseline_probe(
        model, args.n_train, args.n_ctx, args.d_vocab, args.d_model, epochs=100
    )

    # Run experiments
    results = {}

    results['B1'] = experiment_b1_mean_subtraction(
        model, probe, pop_means, args.n_test, args.n_ctx, args.d_vocab
    )

    results['B2'] = experiment_b2_cross_position_patching(
        model, probe, pop_means, args.n_test, args.n_ctx, args.d_vocab
    )

    results['B3'] = experiment_b3_mean_only_prediction(probe, pop_means, args.n_ctx)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n[B1] Mean Subtraction:")
    print(f"  Baseline: {results['B1']['baseline_accuracy']:.4f}")
    print(f"  Ablated:  {results['B1']['ablated_accuracy']:.4f}")
    print(f"  Drop:     {results['B1']['relative_drop']:.1f}%")

    print("\n[B2] Cross-Position Patching:")
    print(f"  Shift rate: {results['B2']['shift_rate']:.4f}")

    print("\n[B3] Mean-Only Prediction:")
    print(f"  Accuracy: {results['B3']['accuracy']:.4f}")

    print("\nINTERPRETATION:")
    if results['B1']['relative_drop'] > 20:
        print("  → B1 shows significant drop: Population mean IS used by the probe")
    else:
        print("  → B1 shows small drop: Probe does NOT heavily rely on population mean")

    if results['B2']['shift_rate'] > 0.7:
        print("  → B2 shows high shift rate: Predictions follow population mean structure")
    else:
        print("  → B2 shows low shift rate: Individual sample features dominate")

    if results['B3']['accuracy'] > 0.8:
        print("  → B3 shows high accuracy: Population means alone predict position well")
    else:
        print("  → B3 shows low accuracy: Individual sample variation is necessary")

    # Save results
    save_path = ARTIFACTS_DIR / f"experiment_b_series_{args.setting}.pt"
    torch.save({'results': results, 'config': vars(args)}, save_path)
    print(f"\nResults saved to: {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Mean Subtraction Ablation Experiments')
    parser.add_argument('--setting', type=str, default='synthetic',
                        choices=['synthetic', 'natural_language'])
    parser.add_argument('--d_model', type=int, default=DEFAULT_D_MODEL)
    parser.add_argument('--d_vocab', type=int, default=DEFAULT_D_VOCAB)
    parser.add_argument('--n_ctx', type=int, default=DEFAULT_N_CTX)
    parser.add_argument('--n_train', type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument('--n_test', type=int, default=DEFAULT_N_TEST)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiments(args)


if __name__ == "__main__":
    main()
