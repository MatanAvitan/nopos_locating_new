"""
Train BOS@80 Model From Scratch with Attention Pattern Logging.

Tracks attention to position 0 over training to see when it emerges.
"""

import os
import sys
import time
import math
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
})

BOS_TOKEN_ID = 50256
BOS_POSITION = 80


def load_data(data_path):
    return np.memmap(data_path, dtype=np.uint16, mode='r')


def get_batch_with_bos(data, batch_size, block_size, bos_position, device):
    tokens_needed = block_size - 1
    ix = np.random.randint(0, len(data) - tokens_needed, size=batch_size)

    sequences = []
    for i in ix:
        before_bos = data[i:i + bos_position].astype(np.int64)
        after_bos = data[i + bos_position:i + tokens_needed].astype(np.int64)
        seq = np.concatenate([before_bos, [BOS_TOKEN_ID], after_bos])
        sequences.append(torch.from_numpy(seq))

    x = torch.stack(sequences).to(device)
    pos_targets = torch.arange(block_size, device=device).unsqueeze(0).expand(batch_size, -1)
    return x, pos_targets


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def compute_attention_stats(model, data, device, n_batches=5):
    """Compute attention to position 0 and BOS position for all heads."""
    model.eval()
    batch_size = 32
    block_size = 128

    attn1_accum = torch.zeros(12, device=device)
    attn2_accum_pos0 = torch.zeros(12, device=device)
    attn2_accum_bos = torch.zeros(12, device=device)

    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = get_batch_with_bos(data, batch_size, block_size, BOS_POSITION, device)
            _ = model(x, capture_taps=True)
            attn1, attn2 = model.get_attention_weights()

            # Block 1: attention to position 0
            for h in range(12):
                attn1_accum[h] += attn1[:, h, :, 0].mean()

            # Block 2: attention to position 0 and BOS (position 80)
            for h in range(12):
                attn2_accum_pos0[h] += attn2[:, h, :, 0].mean()
                # Only queries >= 80 can see BOS
                attn2_accum_bos[h] += attn2[:, h, 80:, 80].mean()

    model.train()

    return {
        'block1_to_pos0': (attn1_accum / n_batches).cpu().numpy(),
        'block2_to_pos0': (attn2_accum_pos0 / n_batches).cpu().numpy(),
        'block2_to_bos': (attn2_accum_bos / n_batches).cpu().numpy(),
    }


def main():
    # Config
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/training_dynamics'
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Training params (shorter for quick experiment)
    max_iters = 2000
    batch_size = 32
    block_size = 128
    learning_rate = 1e-3
    min_lr = 1e-4
    warmup_iters = 100
    lr_decay_iters = 2000
    weight_decay = 0.01
    log_interval = 50
    eval_interval = 100

    # Data
    data_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext/train.bin'
    data = load_data(data_path)
    print(f"Loaded {len(data):,} tokens")

    # Model
    model_config = TwoLayerMechanismConfig(
        block_size=block_size,
        vocab_size=50304,
        n_embd=768,
        n_head=12,
        dropout=0.0,
        norm_type='layernorm',
        use_regression=True,
    )

    torch.manual_seed(42)
    model = TwoLayerMechanismModel(model_config)
    model.to(device)
    model.apply_regime_R0()  # Full training

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in device else 'cpu'
    )

    # Logging
    history = {
        'iter': [],
        'loss': [],
        'lr': [],
        'block2_head7_pos0': [],  # Track Head 7 specifically
        'block2_head9_bos': [],   # Track Head 9 (BOS head)
        'all_attn_stats': [],
    }

    print("\n" + "=" * 70)
    print("Training BOS@80 Model with Attention Logging")
    print("=" * 70)

    # Initial stats
    print("\nInitial attention stats (before training):")
    init_stats = compute_attention_stats(model, data, device)
    print(f"  Head 7 attention to pos 0: {init_stats['block2_to_pos0'][7]:.4f}")
    print(f"  Head 9 attention to BOS:   {init_stats['block2_to_bos'][9]:.4f}")

    # Training loop
    model.train()
    t0 = time.time()

    for iter_num in range(max_iters):
        # Learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        x, pos_targets = get_batch_with_bos(data, batch_size, block_size, BOS_POSITION, device)

        # Forward
        preds, loss = model(x, targets=pos_targets, capture_taps=False)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt*1000:.0f}ms")

        # Eval attention stats
        if iter_num % eval_interval == 0:
            stats = compute_attention_stats(model, data, device)
            history['iter'].append(iter_num)
            history['loss'].append(loss.item())
            history['lr'].append(lr)
            history['block2_head7_pos0'].append(float(stats['block2_to_pos0'][7]))
            history['block2_head9_bos'].append(float(stats['block2_to_bos'][9]))
            history['all_attn_stats'].append({
                'block1_to_pos0': stats['block1_to_pos0'].tolist(),
                'block2_to_pos0': stats['block2_to_pos0'].tolist(),
                'block2_to_bos': stats['block2_to_bos'].tolist(),
            })

    # Final stats
    print("\nFinal attention stats (after training):")
    final_stats = compute_attention_stats(model, data, device)
    print(f"  Head 7 attention to pos 0: {final_stats['block2_to_pos0'][7]:.4f}")
    print(f"  Head 9 attention to BOS:   {final_stats['block2_to_bos'][9]:.4f}")

    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # =========================================================================
    # Visualization
    # =========================================================================
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: Loss curve
    ax = axes[0, 0]
    ax.plot(history['iter'], history['loss'], 'b-', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.set_yscale('log')

    # Plot 2: Head 7 attention to position 0
    ax = axes[0, 1]
    ax.plot(history['iter'], history['block2_head7_pos0'], 'r-', linewidth=2, label='Head 7 → Pos 0')
    ax.plot(history['iter'], history['block2_head9_bos'], 'g-', linewidth=2, label='Head 9 → BOS')
    ax.axhline(y=init_stats['block2_to_pos0'][7], color='r', linestyle='--', alpha=0.5, label='Head 7 at init')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Attention')
    ax.set_title('Attention Pattern Emergence', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Plot 3: All heads attention to position 0 over time
    ax = axes[1, 0]
    n_evals = len(history['all_attn_stats'])
    for h in range(12):
        attn_over_time = [history['all_attn_stats'][i]['block2_to_pos0'][h] for i in range(n_evals)]
        alpha = 1.0 if h in [7, 2, 3] else 0.3
        lw = 2 if h == 7 else 1
        label = f'Head {h}' if h == 7 else None
        ax.plot(history['iter'], attn_over_time, linewidth=lw, alpha=alpha, label=label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Attention to Position 0')
    ax.set_title('Block 2: All Heads → Position 0', fontweight='bold')
    ax.legend()

    # Plot 4: All heads attention to BOS over time
    ax = axes[1, 1]
    for h in range(12):
        attn_over_time = [history['all_attn_stats'][i]['block2_to_bos'][h] for i in range(n_evals)]
        alpha = 1.0 if h == 9 else 0.3
        lw = 2 if h == 9 else 1
        label = f'Head {h}' if h == 9 else None
        ax.plot(history['iter'], attn_over_time, linewidth=lw, alpha=alpha, label=label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Attention to BOS (pos 80)')
    ax.set_title('Block 2: All Heads → BOS Position', fontweight='bold')
    ax.legend()

    fig.suptitle('Training Dynamics: How Position-0 and BOS Attention Emerge', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, 'training_dynamics.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'training_dynamics.pdf'), bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'training_dynamics.png')}")
    plt.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Training Dynamics")
    print("=" * 70)

    # Find when attention crosses thresholds
    for thresh in [0.5, 0.9]:
        for name, vals in [('Head 7 → Pos 0', history['block2_head7_pos0']),
                           ('Head 9 → BOS', history['block2_head9_bos'])]:
            crossed = [i for i, v in enumerate(vals) if v > thresh]
            if crossed:
                iter_crossed = history['iter'][crossed[0]]
                print(f"  {name} crosses {thresh:.0%} at iteration {iter_crossed}")
            else:
                print(f"  {name} never crosses {thresh:.0%}")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
