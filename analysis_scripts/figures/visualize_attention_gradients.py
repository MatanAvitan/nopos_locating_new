"""
Visualize Gradient Flow Through Attention to Different Key Positions.

This script empirically tests whether position 0 receives more gradient signal
during training due to causal attention visibility asymmetry.

The hypothesis is:
- Position j is visible to queries j, j+1, ..., L-1
- So position 0 gets gradient from L queries, position 80 from L-80 queries
- This creates an optimization bias toward attending to position 0

We test this by:
1. Loading the BOS@80 trained model
2. Running forward/backward with gradient hooks on attention
3. Measuring gradient magnitude per key position
4. Comparing to the predicted (L-j) scaling
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats

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
    """
    Wrapper for CausalSelfAttention that captures gradients on attention scores.
    """

    def __init__(self, original_attn):
        super().__init__()
        self.original = original_attn
        self.captured_grads = None
        self.attention_scores = None

    def forward(self, x, return_attn_weights=False):
        B, T, C = x.size()
        n_head = self.original.n_head
        head_dim = C // n_head

        # Calculate Q, K, V
        q, k, v = self.original.c_attn(x).split(self.original.n_embd, dim=2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Store reference for gradient capture (don't detach!)
        self.attention_scores = att

        # Register hook to capture gradient
        if att.requires_grad:
            att.register_hook(self._save_gradient)

        # Store for analysis (detached)
        self.original.last_attention_weights = att.detach()

        # Continue forward
        att_dropped = self.original.attn_dropout(att)
        y = att_dropped @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.original.resid_dropout(self.original.c_proj(y))

        return y

    def _save_gradient(self, grad):
        self.captured_grads = grad.detach().clone()


def load_model_with_grad_capture(checkpoint_path, device='cuda'):
    """Load model and wrap attention modules for gradient capture."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = TwoLayerMechanismConfig(
        block_size=128,
        vocab_size=50304,
        n_embd=768,
        n_head=12,
        dropout=0.0,
        norm_type='layernorm',
        use_regression=True,
    )

    model = TwoLayerMechanismModel(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Wrap attention modules
    wrapped_attn1 = GradientCaptureAttention(model.block1.attn)
    wrapped_attn2 = GradientCaptureAttention(model.block2.attn)

    # Replace in model
    model.block1.attn = wrapped_attn1
    model.block2.attn = wrapped_attn2

    model.train()  # Enable gradient computation
    return model


def load_data(data_path):
    return np.memmap(data_path, dtype=np.uint16, mode='r')


def get_batch_with_bos(data, batch_size, block_size, bos_position, device):
    """Get batch with BOS token at specified position."""
    tokens_needed = block_size - 1
    ix = np.random.randint(0, len(data) - tokens_needed, size=batch_size)

    sequences = []
    for i in ix:
        before_bos = data[i:i + bos_position].astype(np.int64)
        after_bos = data[i + bos_position:i + tokens_needed].astype(np.int64)
        seq = np.concatenate([before_bos, [BOS_TOKEN_ID], after_bos])
        sequences.append(torch.from_numpy(seq))

    return torch.stack(sequences).to(device)


def compute_gradient_by_position(grad_tensor, n_heads=12):
    """
    Compute gradient magnitude for each key position.

    Args:
        grad_tensor: [B, n_head, T_query, T_key] gradient tensor

    Returns:
        grad_by_position: [T_key] total gradient magnitude per key position
        grad_by_head_position: [n_head, T_key] gradient per head per position
    """
    B, H, T_q, T_k = grad_tensor.shape

    # Total gradient magnitude per key position (summed over batch, head, query)
    grad_by_position = grad_tensor.abs().sum(dim=(0, 1, 2)).cpu().numpy()

    # Per-head breakdown
    grad_by_head_position = grad_tensor.abs().sum(dim=(0, 2)).cpu().numpy()  # [H, T_k]

    return grad_by_position, grad_by_head_position


def analyze_gradient_vs_visibility(grad_by_position, L):
    """
    Compare actual gradient to predicted visibility scaling.

    If the visibility hypothesis is correct:
    - Gradient to position j should scale as (L - j), the number of queries that can see it
    """
    positions = np.arange(L)
    visibility = L - positions  # How many queries can see each position

    # Normalize both to [0, 1] for comparison
    grad_normalized = grad_by_position / grad_by_position.max()
    visibility_normalized = visibility / visibility.max()

    # Compute correlation
    correlation, p_value = stats.pearsonr(grad_by_position, visibility)

    return {
        'correlation': correlation,
        'p_value': p_value,
        'grad_normalized': grad_normalized,
        'visibility_normalized': visibility_normalized,
        'visibility': visibility,
    }


def create_gradient_visualization(grad_by_position, grad_by_head_position,
                                   analysis_results, block_name, output_dir):
    """Create comprehensive gradient visualization."""
    L = len(grad_by_position)
    positions = np.arange(L)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: Gradient magnitude by key position
    ax = axes[0, 0]
    ax.bar(positions, grad_by_position, color='#4C72B0', alpha=0.8, width=1.0)
    ax.axvline(x=0, color='#C44E52', linestyle='--', linewidth=2, label='Position 0')
    ax.axvline(x=BOS_POSITION, color='#55A868', linestyle='--', linewidth=2, label=f'BOS (pos {BOS_POSITION})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Total Gradient Magnitude')
    ax.set_title(f'{block_name}: Gradient to Each Key Position', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, L - 0.5)

    # Plot 2: Actual gradient vs visibility prediction
    ax = axes[0, 1]
    visibility = analysis_results['visibility']
    ax.scatter(visibility, grad_by_position, alpha=0.5, s=10, c='#4C72B0')

    # Fit line
    slope, intercept, r_value, _, _ = stats.linregress(visibility, grad_by_position)
    x_line = np.linspace(visibility.min(), visibility.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f'Linear fit (r={analysis_results["correlation"]:.3f})')

    ax.set_xlabel('Visibility Count (L - j)')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient vs Causal Visibility', fontweight='bold')
    ax.legend()

    # Add annotation
    corr_text = f'Pearson r = {analysis_results["correlation"]:.4f}\np = {analysis_results["p_value"]:.2e}'
    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Normalized comparison
    ax = axes[1, 0]
    ax.plot(positions, analysis_results['grad_normalized'], 'b-', linewidth=2,
            label='Actual gradient (normalized)')
    ax.plot(positions, analysis_results['visibility_normalized'], 'r--', linewidth=2,
            label='Predicted: (L-j)/L')
    ax.axvline(x=0, color='#C44E52', linestyle=':', alpha=0.5)
    ax.axvline(x=BOS_POSITION, color='#55A868', linestyle=':', alpha=0.5)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Actual vs Predicted Gradient Pattern', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, L - 0.5)

    # Plot 4: Per-head gradient heatmap
    ax = axes[1, 1]
    im = ax.imshow(grad_by_head_position, aspect='auto', cmap='Blues')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Head Index')
    ax.set_title('Gradient by Head and Position', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Gradient Magnitude')

    # Mark special positions
    for pos, color, label in [(0, '#C44E52', 'Pos 0'), (BOS_POSITION, '#55A868', 'BOS')]:
        ax.axvline(x=pos, color=color, linestyle='--', linewidth=1, alpha=0.7)

    fig.suptitle(f'Gradient Flow Analysis: {block_name}', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, f'gradient_analysis_{block_name.lower().replace(" ", "_")}.png')
    pdf_path = os.path.join(output_dir, f'gradient_analysis_{block_name.lower().replace(" ", "_")}.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")

    plt.close()


def main():
    # Paths
    bos80_ckpt = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism-bos80/R0/best_ckpt.pt'
    data_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext/val.bin'
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/gradient_analysis'
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model with gradient capture
    print("\nLoading BOS@80 model with gradient capture wrappers...")
    model = load_model_with_grad_capture(bos80_ckpt, device)

    # Load data
    data = load_data(data_path)
    print(f"Loaded {len(data):,} tokens")

    # Accumulate gradients over multiple batches
    n_batches = 20
    batch_size = 32
    block_size = 128

    np.random.seed(42)
    torch.manual_seed(42)

    # Accumulators
    total_grad_block1 = torch.zeros(block_size, device=device)
    total_grad_block2 = torch.zeros(block_size, device=device)
    head_grad_block1 = torch.zeros(12, block_size, device=device)
    head_grad_block2 = torch.zeros(12, block_size, device=device)

    print(f"\nRunning {n_batches} batches of {batch_size} sequences...")

    for batch_idx in range(n_batches):
        # Get batch
        x = get_batch_with_bos(data, batch_size, block_size, BOS_POSITION, device)
        pos_targets = torch.arange(block_size, device=device).unsqueeze(0).expand(batch_size, -1).float()

        # Forward pass
        model.zero_grad()
        preds, loss = model(x, targets=pos_targets.long())

        # Backward pass
        loss.backward()

        # Collect gradients
        if model.block1.attn.captured_grads is not None:
            grad1 = model.block1.attn.captured_grads  # [B, H, T, T]
            total_grad_block1 += grad1.abs().sum(dim=(0, 1, 2))
            head_grad_block1 += grad1.abs().sum(dim=(0, 2))

        if model.block2.attn.captured_grads is not None:
            grad2 = model.block2.attn.captured_grads
            total_grad_block2 += grad2.abs().sum(dim=(0, 1, 2))
            head_grad_block2 += grad2.abs().sum(dim=(0, 2))

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{n_batches} done")

    # Convert to numpy
    grad_block1 = total_grad_block1.cpu().numpy()
    grad_block2 = total_grad_block2.cpu().numpy()
    head_grad_block1 = head_grad_block1.cpu().numpy()
    head_grad_block2 = head_grad_block2.cpu().numpy()

    print("\n" + "=" * 70)
    print("GRADIENT FLOW ANALYSIS RESULTS")
    print("=" * 70)

    # Analyze Block 1
    print("\n--- Block 1 Analysis ---")
    analysis1 = analyze_gradient_vs_visibility(grad_block1, block_size)
    print(f"Correlation with visibility (L-j): r = {analysis1['correlation']:.4f} (p = {analysis1['p_value']:.2e})")
    print(f"Position 0 gradient: {grad_block1[0]:.2f}")
    print(f"Position 80 (BOS) gradient: {grad_block1[80]:.2f}")
    print(f"Ratio (pos 0 / pos 80): {grad_block1[0] / grad_block1[80]:.2f}x")

    # Analyze Block 2
    print("\n--- Block 2 Analysis ---")
    analysis2 = analyze_gradient_vs_visibility(grad_block2, block_size)
    print(f"Correlation with visibility (L-j): r = {analysis2['correlation']:.4f} (p = {analysis2['p_value']:.2e})")
    print(f"Position 0 gradient: {grad_block2[0]:.2f}")
    print(f"Position 80 (BOS) gradient: {grad_block2[80]:.2f}")
    print(f"Ratio (pos 0 / pos 80): {grad_block2[0] / grad_block2[80]:.2f}x")

    # Per-head analysis for Block 2 (where we see position-0 attention)
    print("\n--- Block 2 Per-Head Gradient to Position 0 ---")
    for h in range(12):
        grad_to_0 = head_grad_block2[h, 0]
        grad_to_80 = head_grad_block2[h, 80]
        ratio = grad_to_0 / grad_to_80 if grad_to_80 > 0 else float('inf')
        marker = ""
        if ratio > 2.0:
            marker = " <-- HIGH RATIO"
        print(f"Head {h:2d}: grad[0]={grad_to_0:10.2f}, grad[80]={grad_to_80:10.2f}, ratio={ratio:6.2f}{marker}")

    # Theoretical prediction
    print("\n--- Theoretical Prediction ---")
    print(f"If visibility is the mechanism:")
    print(f"  Position 0: visible to {block_size} queries")
    print(f"  Position 80: visible to {block_size - 80} queries")
    print(f"  Predicted ratio: {block_size / (block_size - 80):.2f}x")
    print(f"  Actual ratio (Block 2): {grad_block2[0] / grad_block2[80]:.2f}x")

    print("\n" + "=" * 70)

    # Create visualizations
    print("\nCreating visualizations...")
    create_gradient_visualization(grad_block1, head_grad_block1, analysis1, "Block 1", output_dir)
    create_gradient_visualization(grad_block2, head_grad_block2, analysis2, "Block 2", output_dir)

    # Create combined summary figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    positions = np.arange(block_size)

    for ax, grad, analysis, block_name in [
        (axes[0], grad_block1, analysis1, "Block 1"),
        (axes[1], grad_block2, analysis2, "Block 2")
    ]:
        # Normalize for comparison
        grad_norm = grad / grad.max()
        visibility_norm = (block_size - positions) / block_size

        ax.fill_between(positions, 0, grad_norm, alpha=0.3, color='#4C72B0', label='Actual gradient')
        ax.plot(positions, grad_norm, 'b-', linewidth=2)
        ax.plot(positions, visibility_norm, 'r--', linewidth=2, label='Predicted (L-j)/L')

        ax.axvline(x=0, color='#C44E52', linestyle=':', alpha=0.7)
        ax.axvline(x=BOS_POSITION, color='#55A868', linestyle=':', alpha=0.7)

        ax.set_xlabel('Key Position')
        ax.set_ylabel('Normalized Gradient')
        ax.set_title(f'{block_name}\n(r = {analysis["correlation"]:.3f})', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-0.5, block_size - 0.5)
        ax.set_ylim(0, 1.1)

    fig.suptitle('Gradient Flow Follows Causal Visibility Pattern', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    summary_path = os.path.join(output_dir, 'gradient_visibility_summary.png')
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'gradient_visibility_summary.pdf'), bbox_inches='tight')
    print(f"Saved summary: {summary_path}")
    plt.close()

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
