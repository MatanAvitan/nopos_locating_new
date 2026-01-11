"""
Experiment A1: Decoding Vector Direction Ablation

This experiment tests whether the model uses the decoding vector direction
for position prediction by projecting it out and measuring accuracy drop.

Decoding Vector Formula:
    w = W_V · Σ_j LN(E_j)

Ablation:
    h' = h - (h·w/||w||²) * w  (project out the w direction)

If the model uses this direction for position decoding, projecting it out
should significantly hurt position prediction accuracy.

Also includes:
- A2: Value Vector Corruption (add noise to value vectors)
- A3: Orthogonality Ablation (test if orthogonality is necessary)

Usage:
    python decoding_vector_ablation.py --setting synthetic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def normalize_embedding(e, eps=1e-5):
    """Normalize embedding across neuron dimension (LayerNorm-style)."""
    if e.dim() == 1:
        mean = e.mean()
        std = e.std(unbiased=False) + eps
        return (e - mean) / std
    else:
        mean = e.mean(dim=-1, keepdim=True)
        std = e.std(dim=-1, keepdim=True, unbiased=False) + eps
        return (e - mean) / std


def compute_decoding_vector(W_E, W_V):
    """
    Compute the decoding vector: w = W_V · Σ_j LN(E_j)

    Args:
        W_E: [vocab_size, d_model] - Embedding matrix
        W_V: [d_model, d_model] - Value projection matrix

    Returns:
        w: [d_model] - Normalized decoding vector
    """
    # Normalize each embedding
    e_normalized = normalize_embedding(W_E)  # [vocab, d_model]

    # Sum all normalized embeddings
    sum_normalized = e_normalized.sum(dim=0)  # [d_model]

    # Apply W_V
    w = sum_normalized @ W_V.T  # [d_model]

    # Normalize for stability
    w = w / (w.norm() + 1e-8)

    return w


def ablate_direction(h, direction):
    """
    Project out a direction from activations.

    h' = h - (h·d/||d||²) * d

    Args:
        h: [..., d_model] - Activations
        direction: [d_model] - Direction to project out

    Returns:
        h_ablated: [..., d_model] - Activations with direction removed
    """
    d_norm = direction / (direction.norm() + 1e-8)

    # Project h onto direction
    proj_scalar = (h @ d_norm).unsqueeze(-1)  # [..., 1]
    proj = proj_scalar * d_norm  # [..., d_model]

    # Remove projection
    h_ablated = h - proj

    return h_ablated


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


def train_probe(model, n_samples, n_ctx, d_vocab, d_model, epochs=100):
    """Train a probe on raw activations."""
    print("  Training probe...")
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
            acts = get_activations(model, tokens)
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
    """Evaluate probe on activations."""
    probe.eval()
    with torch.no_grad():
        batch_size = activations.shape[0]
        acts_flat = activations.view(-1, activations.shape[-1])
        positions = torch.arange(n_ctx, device=DEVICE).repeat(batch_size)

        logits = probe(acts_flat)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == positions).float().mean().item()

    return accuracy


def experiment_a1_decoding_vector_ablation(model, probe, n_test, n_ctx, d_vocab, d_model):
    """
    A1: Project out the decoding vector direction and measure accuracy drop.
    """
    print("\n[Experiment A1] Decoding Vector Direction Ablation")

    # Compute decoding vector
    W_E = model.embed.W_E.data
    W_V = model.blocks[0].attn.W_V.data.squeeze(0)  # Remove head dimension

    decoding_vector = compute_decoding_vector(W_E, W_V)
    print(f"  Decoding vector norm: {decoding_vector.norm().item():.4f}")

    # Generate test data
    batch_size = 64
    n_batches = n_test // batch_size

    baseline_acc_sum = 0
    ablated_acc_sum = 0
    total = 0

    for _ in tqdm(range(n_batches), desc="  Testing"):
        tokens = torch.randint(0, d_vocab, (batch_size, n_ctx), device=DEVICE)
        acts = get_activations(model, tokens)

        # Baseline accuracy
        baseline_acc = evaluate_probe(probe, acts, n_ctx)
        baseline_acc_sum += baseline_acc * batch_size

        # Ablated: project out decoding vector
        acts_ablated = ablate_direction(acts, decoding_vector)
        ablated_acc = evaluate_probe(probe, acts_ablated, n_ctx)
        ablated_acc_sum += ablated_acc * batch_size

        total += batch_size

    baseline_accuracy = baseline_acc_sum / total
    ablated_accuracy = ablated_acc_sum / total
    relative_drop = (baseline_accuracy - ablated_accuracy) / baseline_accuracy * 100

    print(f"\n  Baseline accuracy:    {baseline_accuracy:.4f}")
    print(f"  After vector ablation:{ablated_accuracy:.4f}")
    print(f"  Relative drop:        {relative_drop:.1f}%")

    return {
        'baseline_accuracy': baseline_accuracy,
        'ablated_accuracy': ablated_accuracy,
        'relative_drop': relative_drop,
        'decoding_vector_norm': decoding_vector.norm().item()
    }


def experiment_a1_multi_direction_ablation(model, probe, n_test, n_ctx, d_vocab, d_model, n_random=5):
    """
    Ablate decoding vector vs random directions to show specificity.
    """
    print("\n[Experiment A1b] Comparison: Decoding Vector vs Random Directions")

    W_E = model.embed.W_E.data
    W_V = model.blocks[0].attn.W_V.data.squeeze(0)

    # Compute decoding vector
    decoding_vector = compute_decoding_vector(W_E, W_V)

    # Generate random directions
    random_directions = [torch.randn(d_model, device=DEVICE) for _ in range(n_random)]
    random_directions = [d / d.norm() for d in random_directions]

    batch_size = 64
    n_batches = n_test // batch_size

    results = {'decoding': [], 'random': [[] for _ in range(n_random)]}

    for batch_idx in tqdm(range(n_batches), desc="  Testing"):
        tokens = torch.randint(0, d_vocab, (batch_size, n_ctx), device=DEVICE)
        acts = get_activations(model, tokens)

        # Baseline
        baseline_acc = evaluate_probe(probe, acts, n_ctx)

        # Ablate decoding vector
        acts_ablated = ablate_direction(acts, decoding_vector)
        decoding_ablated_acc = evaluate_probe(probe, acts_ablated, n_ctx)
        results['decoding'].append(baseline_acc - decoding_ablated_acc)

        # Ablate random directions
        for i, rand_dir in enumerate(random_directions):
            acts_rand_ablated = ablate_direction(acts, rand_dir)
            rand_ablated_acc = evaluate_probe(probe, acts_rand_ablated, n_ctx)
            results['random'][i].append(baseline_acc - rand_ablated_acc)

    decoding_drop = np.mean(results['decoding'])
    random_drops = [np.mean(r) for r in results['random']]
    avg_random_drop = np.mean(random_drops)

    print(f"\n  Accuracy drop from decoding vector ablation: {decoding_drop:.4f}")
    print(f"  Accuracy drop from random direction ablation: {avg_random_drop:.4f} (avg of {n_random})")
    print(f"  Ratio (decoding/random): {decoding_drop/avg_random_drop:.2f}x")

    return {
        'decoding_drop': decoding_drop,
        'random_drops': random_drops,
        'avg_random_drop': avg_random_drop,
        'ratio': decoding_drop / avg_random_drop if avg_random_drop > 0 else float('inf')
    }


def experiment_a2_value_vector_corruption(model, n_test, n_ctx, d_vocab, d_model, noise_levels=None):
    """
    A2: Add noise to value vectors and measure counting mechanism degradation.
    """
    print("\n[Experiment A2] Value Vector Corruption")

    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

    W_E = model.embed.W_E.data
    W_V = model.blocks[0].attn.W_V.data.squeeze(0)

    results = []

    for noise_scale in noise_levels:
        correlations = []

        for _ in tqdm(range(n_test // 100), desc=f"  Noise={noise_scale:.1f}", leave=False):
            # Generate random sequence
            tokens = torch.randint(0, d_vocab, (n_ctx,), device=DEVICE)
            e = W_E[tokens]  # [n_ctx, d_model]

            # Normalize and compute value vectors
            e_norm = normalize_embedding(e)
            v = e_norm @ W_V.T  # [n_ctx, d_model]

            # Add noise to value vectors
            if noise_scale > 0:
                noise = torch.randn_like(v) * noise_scale * v.std()
                v_noisy = v + noise
            else:
                v_noisy = v

            # Compute attention outputs (uniform attention)
            cumsum_v = torch.cumsum(v_noisy, dim=0)
            positions_tensor = torch.arange(1, n_ctx + 1, device=DEVICE).float().unsqueeze(1)
            z = cumsum_v / positions_tensor

            # Count positive contributions for each position
            decoded = []
            for pos_idx in range(n_ctx):
                z_i = z[pos_idx]
                v_contributing = v_noisy[:pos_idx + 1]
                dots = (v_contributing * z_i).sum(dim=1)
                positive_count = (dots > 0).sum().item()
                decoded.append(positive_count)

            # Correlation with true positions
            true_pos = np.arange(1, n_ctx + 1)
            corr, _ = pearsonr(decoded, true_pos)
            correlations.append(corr)

        avg_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        results.append({
            'noise_scale': noise_scale,
            'correlation': avg_corr,
            'std': std_corr
        })
        print(f"  Noise {noise_scale:.1f}: Correlation = {avg_corr:.4f} ± {std_corr:.4f}")

    return results


def experiment_a3_orthogonality_check(model, n_test, n_ctx, d_vocab, d_model):
    """
    A3: Check how orthogonal the embeddings are and its effect on counting.
    """
    print("\n[Experiment A3] Embedding Orthogonality Analysis")

    W_E = model.embed.W_E.data

    # Sample pairs of embeddings and compute cosine similarity
    n_pairs = 10000
    idx1 = torch.randint(0, d_vocab, (n_pairs,), device=DEVICE)
    idx2 = torch.randint(0, d_vocab, (n_pairs,), device=DEVICE)

    # Exclude same indices
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    e1 = W_E[idx1]  # [n_pairs, d_model]
    e2 = W_E[idx2]  # [n_pairs, d_model]

    # Cosine similarity
    cos_sim = F.cosine_similarity(e1, e2, dim=1)
    avg_cos_sim = cos_sim.abs().mean().item()
    std_cos_sim = cos_sim.std().item()

    print(f"\n  Number of pairs tested: {len(idx1)}")
    print(f"  Average |cosine similarity|: {avg_cos_sim:.6f}")
    print(f"  Std of cosine similarity: {std_cos_sim:.6f}")
    print(f"  Expected for random (d={d_model}): ~{1/np.sqrt(d_model):.6f}")

    # How close to orthogonal?
    expected_random = 1 / np.sqrt(d_model)
    orthogonality_ratio = avg_cos_sim / expected_random

    print(f"\n  Ratio (actual/expected): {orthogonality_ratio:.2f}")
    if orthogonality_ratio < 1.5:
        print("  → Embeddings are approximately orthogonal (as expected for high-d random vectors)")
    else:
        print("  → Embeddings show some structure beyond random")

    return {
        'avg_abs_cos_sim': avg_cos_sim,
        'std_cos_sim': std_cos_sim,
        'expected_random': expected_random,
        'orthogonality_ratio': orthogonality_ratio
    }


def run_experiments(args):
    """Run all A-series experiments."""

    print("=" * 70)
    print("Decoding Vector Ablation Experiments (A1, A2, A3)")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create model
    print("\n[Step 1] Creating model...")
    model = create_model(args.d_model, args.d_vocab, args.n_ctx)

    # Train probe
    print("\n[Step 2] Training probe...")
    probe = train_probe(model, args.n_train, args.n_ctx, args.d_vocab, args.d_model, epochs=100)

    # Run experiments
    results = {}

    results['A1'] = experiment_a1_decoding_vector_ablation(
        model, probe, args.n_test, args.n_ctx, args.d_vocab, args.d_model
    )

    results['A1b'] = experiment_a1_multi_direction_ablation(
        model, probe, args.n_test, args.n_ctx, args.d_vocab, args.d_model
    )

    results['A2'] = experiment_a2_value_vector_corruption(
        model, args.n_test, args.n_ctx, args.d_vocab, args.d_model
    )

    results['A3'] = experiment_a3_orthogonality_check(
        model, args.n_test, args.n_ctx, args.d_vocab, args.d_model
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n[A1] Decoding Vector Ablation:")
    print(f"  Baseline: {results['A1']['baseline_accuracy']:.4f}")
    print(f"  Ablated:  {results['A1']['ablated_accuracy']:.4f}")
    print(f"  Drop:     {results['A1']['relative_drop']:.1f}%")

    print("\n[A1b] Decoding vs Random Direction Ablation:")
    print(f"  Decoding vector drop: {results['A1b']['decoding_drop']:.4f}")
    print(f"  Random direction drop: {results['A1b']['avg_random_drop']:.4f}")
    print(f"  Ratio: {results['A1b']['ratio']:.2f}x")

    print("\n[A2] Value Vector Corruption:")
    for r in results['A2']:
        print(f"  Noise {r['noise_scale']:.1f}: Correlation = {r['correlation']:.4f}")

    print("\n[A3] Orthogonality:")
    print(f"  Avg |cos_sim|: {results['A3']['avg_abs_cos_sim']:.6f}")
    print(f"  Expected random: {results['A3']['expected_random']:.6f}")

    print("\nINTERPRETATION:")
    if results['A1']['relative_drop'] > 20:
        print("  → A1: Decoding vector direction IS used for position prediction")
    else:
        print("  → A1: Decoding vector direction may not be critical")

    if results['A1b']['ratio'] > 2:
        print("  → A1b: Decoding vector is SPECIFICALLY important (not just any direction)")

    # Save results
    save_path = ARTIFACTS_DIR / f"experiment_a_series_{args.setting}.pt"
    torch.save({'results': results, 'config': vars(args)}, save_path)
    print(f"\nResults saved to: {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Decoding Vector Ablation Experiments')
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
