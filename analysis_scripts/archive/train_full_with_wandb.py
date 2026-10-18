"""
Train BOS@80 Model with Comprehensive WandB Logging.

Tracks ALL 12 heads and logs:
- Training loss
- Per-head attention to position 0 (all 12 heads)
- Per-head attention to BOS position (all 12 heads)
- Attention maps as images
- Gradient norm per position per head
- Identifies which heads specialize during training
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
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style for local plots
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


def compute_all_attention_stats(model, data, device, n_batches=5, block_size=128):
    """Compute attention statistics for ALL heads."""
    model.eval()
    batch_size = 32
    n_heads = 12

    # Accumulators for all heads
    block1_to_pos0 = torch.zeros(n_heads, device=device)
    block2_to_pos0 = torch.zeros(n_heads, device=device)
    block2_to_bos = torch.zeros(n_heads, device=device)

    # Store one attention map for visualization
    sample_attn1 = None
    sample_attn2 = None

    with torch.no_grad():
        for batch_idx in range(n_batches):
            x, _ = get_batch_with_bos(data, batch_size, block_size, BOS_POSITION, device)
            _ = model(x, capture_taps=True)
            attn1, attn2 = model.get_attention_weights()

            # Store first batch attention maps for visualization
            if batch_idx == 0:
                sample_attn1 = attn1[0].cpu().numpy()  # [n_head, T, T]
                sample_attn2 = attn2[0].cpu().numpy()

            # Accumulate stats for all heads
            for h in range(n_heads):
                block1_to_pos0[h] += attn1[:, h, :, 0].mean()
                block2_to_pos0[h] += attn2[:, h, :, 0].mean()
                # Only queries >= 80 can see BOS
                block2_to_bos[h] += attn2[:, h, 80:, 80].mean()

    model.train()

    return {
        'block1_to_pos0': (block1_to_pos0 / n_batches).cpu().numpy(),
        'block2_to_pos0': (block2_to_pos0 / n_batches).cpu().numpy(),
        'block2_to_bos': (block2_to_bos / n_batches).cpu().numpy(),
        'sample_attn1': sample_attn1,
        'sample_attn2': sample_attn2,
    }


def compute_gradient_stats_all_heads(model, data, device, block_size=128):
    """Compute gradient statistics for ALL heads and positions."""
    model.train()
    batch_size = 32
    n_heads = 12

    x, pos_targets = get_batch_with_bos(data, batch_size, block_size, BOS_POSITION, device)

    model.zero_grad()
    preds, loss = model(x, targets=pos_targets)
    loss.backward()

    # Extract gradients from Block 2 attention
    grad2 = model.block2.attn.captured_grads  # [B, H, T_q, T_k]

    if grad2 is None:
        return None

    # Gradient norm per position (summed over batch and query)
    # Shape: [n_heads, T_k]
    grad_per_head_pos = grad2.abs().sum(dim=(0, 2)).cpu().numpy()

    # Total gradient per position (summed over all heads too)
    grad_by_position = grad2.abs().sum(dim=(0, 1, 2)).cpu().numpy()

    # Per-head gradient to position 0
    grad_to_pos0 = grad2[:, :, :, 0].abs().sum(dim=(0, 2)).cpu().numpy()

    # Per-head gradient to BOS position
    grad_to_bos = grad2[:, :, 80:, 80].abs().sum(dim=(0, 2)).cpu().numpy()

    return {
        'grad_per_head_pos': grad_per_head_pos,  # [n_heads, T]
        'grad_by_position': grad_by_position,     # [T]
        'grad_to_pos0': grad_to_pos0,             # [n_heads]
        'grad_to_bos': grad_to_bos,               # [n_heads]
    }


def create_attention_heatmap_figure(attn, title, bos_position=80):
    """Create a figure with attention maps for all heads."""
    n_heads = attn.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap='Blues', aspect='auto', vmin=0)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axvline(x=bos_position, color='green', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_title(f'Head {h}', fontsize=8)
        ax.set_xticks([0, bos_position, 127])
        ax.set_yticks([0, bos_position, 127])
        ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def create_gradient_heatmap_figure(grad_per_head_pos, title):
    """Create heatmap of gradient per head and position."""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grad_per_head_pos, aspect='auto', cmap='Reds')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Head Index')
    ax.set_title(title, fontweight='bold')
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.7, linewidth=1)
    plt.colorbar(im, ax=ax, label='Gradient Magnitude')
    plt.tight_layout()
    return fig


def main():
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/full_training_wandb'
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Training params
    max_iters = 20000
    batch_size = 32
    block_size = 128
    learning_rate = 1e-3
    min_lr = 1e-4
    warmup_iters = 200
    lr_decay_iters = 20000
    weight_decay = 0.01
    log_interval = 100
    eval_interval = 500
    n_heads = 12

    # Initialize WandB
    wandb.init(
        project="nope-2layer-mechanism-bos80",
        name="training_dynamics_all_heads",
        config={
            "max_iters": max_iters,
            "batch_size": batch_size,
            "block_size": block_size,
            "learning_rate": learning_rate,
            "bos_position": BOS_POSITION,
            "n_heads": n_heads,
        },
        tags=["training_dynamics", "gradient_analysis", "all_heads"],
    )

    # Data
    data_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext/train.bin'
    data = load_data(data_path)
    print(f"Loaded {len(data):,} tokens")

    # Model with gradient capture
    model_config = TwoLayerMechanismConfig(
        block_size=block_size,
        vocab_size=50304,
        n_embd=768,
        n_head=n_heads,
        dropout=0.0,
        norm_type='layernorm',
        use_regression=True,
    )

    torch.manual_seed(42)
    model = TwoLayerMechanismModel(model_config)

    # Wrap attention for gradient capture
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

    # History for local backup
    history = {
        'iter': [],
        'loss': [],
        'block2_to_pos0': [],  # [n_evals, n_heads]
        'block2_to_bos': [],
        'grad_to_pos0': [],
        'grad_to_bos': [],
    }

    print("\n" + "=" * 70)
    print(f"Training BOS@80 with WandB logging - ALL {n_heads} heads tracked")
    print("=" * 70)

    # Initial stats
    init_stats = compute_all_attention_stats(model, data, device)
    print("\nInitial attention to position 0 (all heads):")
    for h in range(n_heads):
        print(f"  Head {h:2d}: {init_stats['block2_to_pos0'][h]:.4f}")

    # Log initial attention maps
    fig1 = create_attention_heatmap_figure(init_stats['sample_attn2'], 'Block 2 Attention Maps (Init)')
    wandb.log({"attention_maps/block2_init": wandb.Image(fig1)})
    plt.close(fig1)

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

        # Basic logging
        if iter_num % log_interval == 0 and iter_num > 0:
            avg_loss = running_loss / log_interval
            wandb.log({
                "train/loss": avg_loss,
                "train/lr": lr,
                "train/iter": iter_num,
            })

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt*1000/log_interval:.0f}ms/iter")
            running_loss = 0.0

        # Detailed evaluation
        if iter_num % eval_interval == 0:
            # Attention stats for ALL heads
            attn_stats = compute_all_attention_stats(model, data, device)

            # Gradient stats for ALL heads
            grad_stats = compute_gradient_stats_all_heads(model, data, device, block_size)

            # Log to WandB - per-head attention to position 0
            for h in range(n_heads):
                wandb.log({
                    f"attention/head{h}_to_pos0": attn_stats['block2_to_pos0'][h],
                    f"attention/head{h}_to_bos": attn_stats['block2_to_bos'][h],
                    "train/iter": iter_num,
                })

            # Log gradient stats
            if grad_stats:
                for h in range(n_heads):
                    wandb.log({
                        f"gradient/head{h}_to_pos0": grad_stats['grad_to_pos0'][h],
                        f"gradient/head{h}_to_bos": grad_stats['grad_to_bos'][h],
                        "train/iter": iter_num,
                    })

                # Log gradient by position (aggregate)
                wandb.log({
                    "gradient/total_to_pos0": grad_stats['grad_by_position'][0],
                    "gradient/total_to_bos": grad_stats['grad_by_position'][80],
                    "gradient/ratio_pos0_bos": grad_stats['grad_by_position'][0] / (grad_stats['grad_by_position'][80] + 1e-8),
                    "train/iter": iter_num,
                })

            # Log attention maps periodically
            if iter_num % (eval_interval * 4) == 0:  # Every 2000 iterations
                fig2 = create_attention_heatmap_figure(
                    attn_stats['sample_attn2'],
                    f'Block 2 Attention Maps (iter {iter_num})'
                )
                wandb.log({f"attention_maps/block2_iter{iter_num}": wandb.Image(fig2)})
                plt.close(fig2)

                if grad_stats:
                    fig3 = create_gradient_heatmap_figure(
                        grad_stats['grad_per_head_pos'],
                        f'Gradient per Head/Position (iter {iter_num})'
                    )
                    wandb.log({f"gradient_maps/iter{iter_num}": wandb.Image(fig3)})
                    plt.close(fig3)

            # Local history backup
            history['iter'].append(iter_num)
            history['loss'].append(loss.item())
            history['block2_to_pos0'].append(attn_stats['block2_to_pos0'].tolist())
            history['block2_to_bos'].append(attn_stats['block2_to_bos'].tolist())
            if grad_stats:
                history['grad_to_pos0'].append(grad_stats['grad_to_pos0'].tolist())
                history['grad_to_bos'].append(grad_stats['grad_to_bos'].tolist())

            # Print summary
            max_pos0_head = np.argmax(attn_stats['block2_to_pos0'])
            max_bos_head = np.argmax(attn_stats['block2_to_bos'])
            print(f"  [EVAL] Max pos0: Head {max_pos0_head} ({attn_stats['block2_to_pos0'][max_pos0_head]:.4f}), "
                  f"Max BOS: Head {max_bos_head} ({attn_stats['block2_to_bos'][max_bos_head]:.4f})")

    # =========================================================================
    # Final Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS: Which heads specialized?")
    print("=" * 70)

    final_stats = compute_all_attention_stats(model, data, device)

    # Identify specialized heads
    pos0_heads = []
    bos_heads = []

    print("\nFinal attention to position 0 (Block 2):")
    for h in range(n_heads):
        val = final_stats['block2_to_pos0'][h]
        marker = ""
        if val > 0.5:
            pos0_heads.append(h)
            marker = " <-- POSITION-0 HEAD"
        print(f"  Head {h:2d}: {val:.4f}{marker}")

    print("\nFinal attention to BOS position (Block 2):")
    for h in range(n_heads):
        val = final_stats['block2_to_bos'][h]
        marker = ""
        if val > 0.5:
            bos_heads.append(h)
            marker = " <-- BOS HEAD"
        print(f"  Head {h:2d}: {val:.4f}{marker}")

    # Log summary to WandB
    wandb.run.summary["specialized_pos0_heads"] = pos0_heads
    wandb.run.summary["specialized_bos_heads"] = bos_heads
    wandb.run.summary["max_pos0_attention"] = float(np.max(final_stats['block2_to_pos0']))
    wandb.run.summary["max_bos_attention"] = float(np.max(final_stats['block2_to_bos']))

    # Find when each head crossed 50% threshold
    for h in range(n_heads):
        attn_history = [history['block2_to_pos0'][i][h] for i in range(len(history['iter']))]
        crossed = [i for i, v in enumerate(attn_history) if v > 0.5]
        if crossed:
            iter_crossed = history['iter'][crossed[0]]
            wandb.run.summary[f"head{h}_crossed_50pct_pos0_at"] = iter_crossed
            print(f"Head {h} crossed 50% attention to pos 0 at iteration {iter_crossed}")

    # Save local history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Final attention maps
    fig_final = create_attention_heatmap_figure(final_stats['sample_attn2'], 'Block 2 Attention Maps (Final)')
    wandb.log({"attention_maps/block2_final": wandb.Image(fig_final)})
    fig_final.savefig(os.path.join(output_dir, 'final_attention_maps.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_final)

    # Create summary heatmap: heads × iterations
    n_evals = len(history['iter'])
    attn_matrix = np.array(history['block2_to_pos0'])  # [n_evals, n_heads]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(attn_matrix.T, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Head Index')
    ax.set_title('Attention to Position 0 Over Training (All Heads)', fontweight='bold')
    ax.set_xticks(np.arange(0, n_evals, 5))
    ax.set_xticklabels([history['iter'][i] for i in range(0, n_evals, 5)], rotation=45)
    ax.set_yticks(range(n_heads))
    plt.colorbar(im, ax=ax, label='Attention to Position 0')
    plt.tight_layout()

    wandb.log({"summary/attention_heatmap_all_heads": wandb.Image(fig)})
    fig.savefig(os.path.join(output_dir, 'attention_heatmap_all_heads.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll outputs saved to: {output_dir}")
    print("WandB run complete!")

    wandb.finish()


if __name__ == '__main__':
    main()
