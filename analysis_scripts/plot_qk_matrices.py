"""
Plot W_Q, W_K, and W_Q @ W_K.T for BOS heads.

Compares:
- Original vanilla R0 experiment: BOS heads (identified previously)
- BOS@80 experiment: Head 7 (attends to pos 0) and Head 9 (attends to pos 80)
"""

import os
import sys
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


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
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
    model.eval()
    return model


def extract_qk_for_head(model, block_idx: int, head_idx: int):
    """
    Extract W_Q, W_K, and biases for a specific head.

    The c_attn weight has shape [3 * n_embd, n_embd] for Q, K, V concatenated.
    """
    if block_idx == 0:
        attn = model.block1.attn
    else:
        attn = model.block2.attn

    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    # c_attn.weight: [3*n_embd, n_embd] - projects input to Q, K, V
    # Split into Q, K, V sections
    W = attn.c_attn.weight.data  # [3*768, 768]
    W_Q_all = W[:n_embd, :]      # [768, 768]
    W_K_all = W[n_embd:2*n_embd, :]  # [768, 768]

    # Extract for specific head
    # Each head has head_dim rows
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    W_Q = W_Q_all[start:end, :]  # [64, 768]
    W_K = W_K_all[start:end, :]  # [64, 768]

    # Biases (if present)
    b_Q, b_K = None, None
    if attn.c_attn.bias is not None:
        b = attn.c_attn.bias.data
        b_Q = b[start:end]
        b_K = b[n_embd + start:n_embd + end]

    return W_Q.numpy(), W_K.numpy(), b_Q, b_K


def plot_qk_analysis(W_Q, W_K, b_Q, b_K, title: str, output_path: str):
    """
    Create a figure with:
    1. W_Q heatmap
    2. W_K heatmap
    3. W_Q @ W_K.T heatmap
    4. Singular values of W_Q @ W_K.T
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # W_Q @ W_K.T
    QK = W_Q @ W_K.T  # [64, 64]

    # Add bias outer product if biases exist
    if b_Q is not None and b_K is not None:
        # The attention score includes: q @ k.T where q = x @ W_Q.T + b_Q
        # So full term is (x @ W_Q.T + b_Q) @ (x @ W_K.T + b_K).T
        # The bias contribution to the "constant" part would be b_Q @ b_K.T
        # But for the learned pattern, W_Q @ W_K.T is the main term
        pass

    # Plot W_Q
    ax = axes[0, 0]
    vmax = max(abs(W_Q.min()), abs(W_Q.max()))
    im1 = ax.imshow(W_Q, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title('W_Q [head_dim × n_embd]')
    ax.set_xlabel('Input dimension')
    ax.set_ylabel('Head dimension')
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Plot W_K
    ax = axes[0, 1]
    vmax = max(abs(W_K.min()), abs(W_K.max()))
    im2 = ax.imshow(W_K, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title('W_K [head_dim × n_embd]')
    ax.set_xlabel('Input dimension')
    ax.set_ylabel('Head dimension')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Plot W_Q @ W_K.T
    ax = axes[1, 0]
    vmax = max(abs(QK.min()), abs(QK.max()))
    im3 = ax.imshow(QK, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title('W_Q @ W_K.T [head_dim × head_dim]')
    ax.set_xlabel('Key head dimension')
    ax.set_ylabel('Query head dimension')
    plt.colorbar(im3, ax=ax, shrink=0.8)

    # Singular value decomposition of W_Q @ W_K.T
    ax = axes[1, 1]
    U, S, Vh = np.linalg.svd(QK)
    ax.bar(range(len(S)), S, color='#4C72B0', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value')
    ax.set_title(f'SVD of W_Q @ W_K.T (rank ≈ {np.sum(S > 0.01 * S[0])})')
    ax.set_xlim(-0.5, len(S) - 0.5)

    # Add stats
    ax.text(0.95, 0.95, f'||QK||_F = {np.linalg.norm(QK):.2f}\nmax = {QK.max():.3f}\nmin = {QK.min():.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    fig.savefig(output_path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_path + '.pdf', bbox_inches='tight')
    print(f"Saved: {output_path}.png/pdf")
    plt.close()


def main():
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/qk_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Load both models
    print("Loading vanilla R0 model...")
    vanilla_ckpt = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt'
    vanilla_model = load_model(vanilla_ckpt)

    print("Loading BOS@80 R0 model...")
    bos80_ckpt = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism-bos80/R0/best_ckpt.pt'
    bos80_model = load_model(bos80_ckpt)

    # From previous analysis, vanilla R0 BOS heads in Block 2 were heads 6 and 9
    # Let me check which heads we identified - I'll plot several key heads

    print("\n" + "=" * 70)
    print("Plotting QK matrices for key attention heads")
    print("=" * 70)

    # =========================================================================
    # Vanilla R0 - Block 2 heads
    # =========================================================================
    print("\n--- Vanilla R0 Block 2 ---")

    # Plot all Block 2 heads for vanilla to identify patterns
    for head_idx in range(12):
        W_Q, W_K, b_Q, b_K = extract_qk_for_head(vanilla_model, block_idx=1, head_idx=head_idx)
        QK = W_Q @ W_K.T
        print(f"Head {head_idx}: ||W_Q@W_K.T||_F = {np.linalg.norm(QK):.3f}, "
              f"max = {QK.max():.3f}, trace = {np.trace(QK):.3f}")

    # Plot specific heads for vanilla
    for head_idx in [6, 9]:  # Known BOS heads from previous analysis
        W_Q, W_K, b_Q, b_K = extract_qk_for_head(vanilla_model, block_idx=1, head_idx=head_idx)
        plot_qk_analysis(W_Q, W_K, b_Q, b_K,
                        f'Vanilla R0 - Block 2 Head {head_idx} (BOS head)',
                        os.path.join(output_dir, f'vanilla_block2_head{head_idx}'))

    # =========================================================================
    # BOS@80 R0 - Block 2 heads
    # =========================================================================
    print("\n--- BOS@80 R0 Block 2 ---")

    for head_idx in range(12):
        W_Q, W_K, b_Q, b_K = extract_qk_for_head(bos80_model, block_idx=1, head_idx=head_idx)
        QK = W_Q @ W_K.T
        print(f"Head {head_idx}: ||W_Q@W_K.T||_F = {np.linalg.norm(QK):.3f}, "
              f"max = {QK.max():.3f}, trace = {np.trace(QK):.3f}")

    # Head 7: attends to position 0 (96.8%)
    W_Q, W_K, b_Q, b_K = extract_qk_for_head(bos80_model, block_idx=1, head_idx=7)
    plot_qk_analysis(W_Q, W_K, b_Q, b_K,
                    'BOS@80 R0 - Block 2 Head 7 (Position-0 head, 96.8% attn to pos 0)',
                    os.path.join(output_dir, 'bos80_block2_head7_pos0'))

    # Head 9: attends to position 80/BOS (97.9%)
    W_Q, W_K, b_Q, b_K = extract_qk_for_head(bos80_model, block_idx=1, head_idx=9)
    plot_qk_analysis(W_Q, W_K, b_Q, b_K,
                    'BOS@80 R0 - Block 2 Head 9 (BOS head, 97.9% attn to pos 80)',
                    os.path.join(output_dir, 'bos80_block2_head9_bos'))

    # Head 2: also attends to position 0 (71.9%)
    W_Q, W_K, b_Q, b_K = extract_qk_for_head(bos80_model, block_idx=1, head_idx=2)
    plot_qk_analysis(W_Q, W_K, b_Q, b_K,
                    'BOS@80 R0 - Block 2 Head 2 (71.9% attn to pos 0)',
                    os.path.join(output_dir, 'bos80_block2_head2_pos0'))

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
