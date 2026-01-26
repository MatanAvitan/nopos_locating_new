"""
Train BOS@80 Model From Scratch (Full 20k iterations) with:
1. Attention pattern logging
2. Gradient flow per position logging

This tracks when position-0 attention emerges and how gradient flows evolve.
"""

import os
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
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


class GradientCaptureAttention(torch.nn.Module):
    """Wrapper that captures gradients on attention scores."""

    def __init__(self, original_attn):
        super().__init__()
        self.original = original_attn
        self.captured_grads = None
        self.last_attention_weights = None

    def forward(self, x, return_attn_weights=False):
        B, T, C = x.size()
        n_head = self.original.n_head
        head_dim = C // n_head

        q, k, v = self.original.c_attn(x).split(self.original.n_embd, dim=2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Register hook to capture gradient
        if att.requires_grad:
            att.register_hook(self._save_gradient)

        self.last_attention_weights = att.detach()
        self.original.last_attention_weights = att.detach()

        att_dropped = self.original.attn_dropout(att)
        y = att_dropped @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.original.resid_dropout(self.original.c_proj(y))

        return y

    def _save_gradient(self, grad):
        self.captured_grads = grad.detach().clone()


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
    """Compute attention to position 0 and BOS for all heads."""
    model.eval()
    batch_size = 32
    block_size = 128

    attn2_pos0 = torch.zeros(12, device=device)
    attn2_bos = torch.zeros(12, device=device)

    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = get_batch_with_bos(data, batch_size, block_size, BOS_POSITION, device)
            _ = model(x, capture_taps=True)
            _, attn2 = model.get_attention_weights()

            for h in range(12):
                attn2_pos0[h] += attn2[:, h, :, 0].mean()
                attn2_bos[h] += attn2[:, h, 80:, 80].mean()

    model.train()
    return {
        'block2_to_pos0': (attn2_pos0 / n_batches).cpu().numpy(),
        'block2_to_bos': (attn2_bos / n_batches).cpu().numpy(),
    }


def compute_gradient_stats(model, data, device, block_size=128):
    """
    Compute gradient magnitude per key position during one forward-backward pass.
    Returns gradient accumulated over all queries for each key position.
    """
    model.train()
    batch_size = 32

    x, pos_targets = get_batch_with_bos(data, batch_size, block_size, BOS_POSITION, device)

    model.zero_grad()
    preds, loss = model(x, targets=pos_targets)
    loss.backward()

    # Extract gradients from Block 2 attention
    grad2 = model.block2.attn.captured_grads  # [B, H, T_q, T_k]

    if grad2 is None:
        return None

    # Sum gradient magnitude per key position (over batch, head, query)
    grad_by_pos = grad2.abs().sum(dim=(0, 1, 2)).cpu().numpy()  # [T_k]

    # Per-head gradient to position 0 and position 80
    grad_to_pos0 = grad2[:, :, :, 0].abs().sum(dim=(0, 2)).cpu().numpy()  # [H]
    grad_to_bos = grad2[:, :, 80:, 80].abs().sum(dim=(0, 2)).cpu().numpy()  # [H]

    return {
        'grad_by_position': grad_by_pos,
        'grad_to_pos0_by_head': grad_to_pos0,
        'grad_to_bos_by_head': grad_to_bos,
    }


def main():
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/full_training_dynamics'
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Training params - FULL 20k iterations
    max_iters = 20000
    batch_size = 32
    block_size = 128
    learning_rate = 1e-3
    min_lr = 1e-4
    warmup_iters = 200
    lr_decay_iters = 20000
    weight_decay = 0.01
    log_interval = 100
    eval_interval = 500  # Log attention and gradients every 500 steps

    # Data
    data_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext/train.bin'
    data = load_data(data_path)
    print(f"Loaded {len(data):,} tokens")

    # Model with gradient capture wrappers
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

    # Wrap attention modules for gradient capture
    model.block1.attn = GradientCaptureAttention(model.block1.attn)
    model.block2.attn = GradientCaptureAttention(model.block2.attn)

    model.to(device)
    model.apply_regime_R0()

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
        'attn_head7_pos0': [],
        'attn_head9_bos': [],
        'all_heads_pos0': [],
        'all_heads_bos': [],
        'grad_by_position': [],
        'grad_head7_pos0': [],
        'grad_head9_bos': [],
    }

    print("\n" + "=" * 70)
    print(f"Training BOS@80 Model for {max_iters} iterations with gradient logging")
    print("=" * 70)

    # Initial stats
    init_stats = compute_attention_stats(model, data, device)
    print(f"\nInitial attention - Head 7 → pos 0: {init_stats['block2_to_pos0'][7]:.4f}")
    print(f"Initial attention - Head 9 → BOS:   {init_stats['block2_to_bos'][9]:.4f}")

    # Training loop
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for iter_num in range(max_iters + 1):
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

        running_loss += loss.item()

        # Log
        if iter_num % log_interval == 0 and iter_num > 0:
            avg_loss = running_loss / log_interval
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt*1000/log_interval:.0f}ms/iter")
            running_loss = 0.0

        # Detailed eval
        if iter_num % eval_interval == 0:
            # Attention stats
            attn_stats = compute_attention_stats(model, data, device)

            # Gradient stats
            grad_stats = compute_gradient_stats(model, data, device, block_size)

            # Record
            history['iter'].append(iter_num)
            history['loss'].append(loss.item())
            history['attn_head7_pos0'].append(float(attn_stats['block2_to_pos0'][7]))
            history['attn_head9_bos'].append(float(attn_stats['block2_to_bos'][9]))
            history['all_heads_pos0'].append(attn_stats['block2_to_pos0'].tolist())
            history['all_heads_bos'].append(attn_stats['block2_to_bos'].tolist())

            if grad_stats:
                history['grad_by_position'].append(grad_stats['grad_by_position'].tolist())
                history['grad_head7_pos0'].append(float(grad_stats['grad_to_pos0_by_head'][7]))
                history['grad_head9_bos'].append(float(grad_stats['grad_to_bos_by_head'][9]))
            else:
                history['grad_by_position'].append(None)
                history['grad_head7_pos0'].append(None)
                history['grad_head9_bos'].append(None)

            # Print progress
            print(f"  [EVAL] Head7→pos0: {attn_stats['block2_to_pos0'][7]:.4f}, "
                  f"Head9→BOS: {attn_stats['block2_to_bos'][9]:.4f}")

    # Save history
    print("\nSaving training history...")
    with open(os.path.join(output_dir, 'full_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # =========================================================================
    # Create comprehensive visualizations
    # =========================================================================
    print("Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Plot 1: Loss curve
    ax = axes[0, 0]
    ax.plot(history['iter'], history['loss'], 'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.set_yscale('log')

    # Plot 2: Attention emergence
    ax = axes[0, 1]
    ax.plot(history['iter'], history['attn_head7_pos0'], 'r-', linewidth=2, label='Head 7 → Pos 0')
    ax.plot(history['iter'], history['attn_head9_bos'], 'g-', linewidth=2, label='Head 9 → BOS')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Attention')
    ax.set_title('Attention Pattern Emergence', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Plot 3: All heads to position 0
    ax = axes[0, 2]
    for h in range(12):
        attn_over_time = [history['all_heads_pos0'][i][h] for i in range(len(history['iter']))]
        alpha = 1.0 if h == 7 else 0.3
        lw = 2 if h == 7 else 0.5
        color = 'red' if h == 7 else 'blue'
        ax.plot(history['iter'], attn_over_time, color=color, linewidth=lw, alpha=alpha)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Attention to Position 0')
    ax.set_title('All Heads → Position 0', fontweight='bold')
    ax.set_ylim(0, 1.05)

    # Plot 4: Gradient flow by position over training
    ax = axes[1, 0]
    n_snapshots = len([g for g in history['grad_by_position'] if g is not None])
    if n_snapshots > 0:
        # Take snapshots at different training stages
        valid_indices = [i for i, g in enumerate(history['grad_by_position']) if g is not None]
        snapshot_indices = [valid_indices[j] for j in [0, len(valid_indices)//3, 2*len(valid_indices)//3, -1]]

        positions = np.arange(block_size)
        for idx in snapshot_indices:
            grad = np.array(history['grad_by_position'][idx])
            grad_norm = grad / grad.max() if grad.max() > 0 else grad
            iter_num = history['iter'][idx]
            ax.plot(positions, grad_norm, label=f'iter {iter_num}', alpha=0.8)

        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=80, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Normalized Gradient')
        ax.set_title('Gradient by Position (snapshots)', fontweight='bold')
        ax.legend(fontsize=7)

    # Plot 5: Gradient to pos 0 vs BOS over training
    ax = axes[1, 1]
    valid_grad7 = [(history['iter'][i], history['grad_head7_pos0'][i])
                   for i in range(len(history['iter'])) if history['grad_head7_pos0'][i] is not None]
    valid_grad9 = [(history['iter'][i], history['grad_head9_bos'][i])
                   for i in range(len(history['iter'])) if history['grad_head9_bos'][i] is not None]

    if valid_grad7:
        iters7, grads7 = zip(*valid_grad7)
        ax.plot(iters7, grads7, 'r-', linewidth=1, label='Head 7 grad → pos 0')
    if valid_grad9:
        iters9, grads9 = zip(*valid_grad9)
        ax.plot(iters9, grads9, 'g-', linewidth=1, label='Head 9 grad → BOS')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Head-Specific Gradients Over Training', fontweight='bold')
    ax.legend()

    # Plot 6: Attention vs Gradient correlation
    ax = axes[1, 2]
    if valid_grad7 and len(valid_grad7) > 5:
        # Plot attention vs gradient for Head 7
        attn7 = [history['attn_head7_pos0'][i] for i in range(len(history['iter']))
                 if history['grad_head7_pos0'][i] is not None]
        grad7 = [history['grad_head7_pos0'][i] for i in range(len(history['iter']))
                 if history['grad_head7_pos0'][i] is not None]

        colors = np.linspace(0, 1, len(attn7))
        scatter = ax.scatter(grad7, attn7, c=colors, cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Training Progress')
        ax.set_xlabel('Gradient to Position 0')
        ax.set_ylabel('Attention to Position 0')
        ax.set_title('Head 7: Attention vs Gradient', fontweight='bold')

    fig.suptitle('Full Training Dynamics (20k iterations)', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, 'full_training_dynamics.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'full_training_dynamics.pdf'), bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'full_training_dynamics.png')}")
    plt.close()

    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING DYNAMICS SUMMARY")
    print("=" * 70)

    # Find when attention crosses thresholds
    for thresh in [0.5, 0.9, 0.99]:
        for name, vals in [('Head 7 → Pos 0', history['attn_head7_pos0']),
                           ('Head 9 → BOS', history['attn_head9_bos'])]:
            crossed = [i for i, v in enumerate(vals) if v > thresh]
            if crossed:
                iter_crossed = history['iter'][crossed[0]]
                print(f"  {name} crosses {thresh:.0%} at iteration {iter_crossed}")
            else:
                print(f"  {name} never crosses {thresh:.0%}")

    print(f"\nFinal attention - Head 7 → pos 0: {history['attn_head7_pos0'][-1]:.4f}")
    print(f"Final attention - Head 9 → BOS:   {history['attn_head9_bos'][-1]:.4f}")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
