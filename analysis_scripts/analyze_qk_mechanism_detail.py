"""
Detailed Analysis of QK Mechanism for Position-0 Attention.

Investigates how the model can attend to position 0 with ~100% attention
even though the token at position 0 constantly changes.

Tests multiple hypotheses:
1. Bias terms dominate the attention score
2. W_K projects all tokens to similar vectors
3. Check attention at initialization (before training)
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

BOS_TOKEN_ID = 50256


def load_model(checkpoint_path, device='cpu'):
    """Load trained model."""
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


def create_fresh_model(device='cpu'):
    """Create a fresh model with Xavier initialization (no training)."""
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
    model.eval()
    return model


def extract_qk_params(model, block_idx, head_idx):
    """
    Extract W_Q, W_K, b_Q, b_K for a specific head.
    """
    block = model.block1 if block_idx == 0 else model.block2
    attn = block.attn

    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    # c_attn weight: [3*n_embd, n_embd] for Q, K, V concatenated
    W = attn.c_attn.weight.data  # [3*768, 768]
    W_Q_all = W[:n_embd, :]
    W_K_all = W[n_embd:2*n_embd, :]

    # Extract for specific head
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    W_Q = W_Q_all[start:end, :]  # [64, 768]
    W_K = W_K_all[start:end, :]  # [64, 768]

    # Biases
    b_Q, b_K = None, None
    if attn.c_attn.bias is not None:
        b = attn.c_attn.bias.data
        b_Q = b[start:end]
        b_K = b[n_embd + start:n_embd + end]

    return W_Q, W_K, b_Q, b_K


def analyze_bias_contribution(W_Q, W_K, b_Q, b_K, embeddings):
    """
    Analyze how much bias terms contribute to attention scores.

    Full attention score (before softmax):
    score = (W_Q @ x_i + b_Q) · (W_K @ x_j + b_K) / sqrt(d)
          = x_i.T @ W_Q.T @ W_K @ x_j + b_Q · (W_K @ x_j) + (W_Q @ x_i) · b_K + b_Q · b_K

    We analyze the magnitude of each term.
    """
    head_dim = W_Q.shape[0]
    scale = 1.0 / np.sqrt(head_dim)

    # Sample some embeddings
    n_samples = min(1000, embeddings.shape[0])
    idx = np.random.choice(embeddings.shape[0], n_samples, replace=False)
    E = embeddings[idx]  # [n_samples, n_embd]

    # Compute Q and K for all samples
    Q = E @ W_Q.T  # [n_samples, head_dim]
    K = E @ W_K.T  # [n_samples, head_dim]

    # Term 1: x_i.T @ W_Q.T @ W_K @ x_j (token-token interaction)
    # Sample pairs
    QK_scores = Q @ K.T  # [n_samples, n_samples]
    term1_mean = np.abs(QK_scores).mean() * scale
    term1_std = np.abs(QK_scores).std() * scale

    # Term 2: b_Q · (W_K @ x_j) = b_Q · k_j
    if b_Q is not None:
        term2_values = K @ b_Q  # [n_samples]
        term2_mean = np.abs(term2_values).mean() * scale
        term2_std = np.abs(term2_values).std() * scale
    else:
        term2_mean, term2_std = 0, 0

    # Term 3: (W_Q @ x_i) · b_K = q_i · b_K
    if b_K is not None:
        term3_values = Q @ b_K  # [n_samples]
        term3_mean = np.abs(term3_values).mean() * scale
        term3_std = np.abs(term3_values).std() * scale
    else:
        term3_mean, term3_std = 0, 0

    # Term 4: b_Q · b_K (constant term)
    if b_Q is not None and b_K is not None:
        term4 = float(np.dot(b_Q, b_K)) * scale
    else:
        term4 = 0

    return {
        'term1_qk': (term1_mean, term1_std),
        'term2_bq_k': (term2_mean, term2_std),
        'term3_q_bk': (term3_mean, term3_std),
        'term4_bq_bk': term4,
        'bias_norms': (np.linalg.norm(b_Q) if b_Q is not None else 0,
                       np.linalg.norm(b_K) if b_K is not None else 0),
    }


def analyze_key_variance(W_K, embeddings):
    """
    Analyze how much key vectors vary across different tokens.

    If W_K projects all tokens to similar vectors, variance is low.
    """
    # Compute K for all tokens
    K = embeddings @ W_K.T  # [vocab_size, head_dim]

    # Per-dimension variance
    var_per_dim = np.var(K, axis=0)

    # Overall variance
    total_var = np.var(K)

    # Norm statistics
    norms = np.linalg.norm(K, axis=1)

    # Mean key vector and distances from mean
    mean_K = K.mean(axis=0)
    dist_from_mean = np.linalg.norm(K - mean_K, axis=1)

    return {
        'total_variance': total_var,
        'mean_var_per_dim': var_per_dim.mean(),
        'norm_mean': norms.mean(),
        'norm_std': norms.std(),
        'dist_from_mean_mean': dist_from_mean.mean(),
        'dist_from_mean_std': dist_from_mean.std(),
        'K_all': K,
    }


def compute_attention_at_init(model, device='cpu'):
    """
    Compute attention patterns for a freshly initialized model.

    Uses random token sequences to see if position 0 gets special attention at init.
    """
    model.eval()
    model.to(device)

    # Generate random sequences
    batch_size = 64
    block_size = 128
    vocab_size = model.config.vocab_size

    # Random tokens (excluding special tokens > 50256)
    x = torch.randint(0, 50256, (batch_size, block_size), device=device)

    with torch.no_grad():
        # Forward pass
        _ = model(x, capture_taps=True)
        attn1, attn2 = model.get_attention_weights()

    return attn1.cpu().numpy(), attn2.cpu().numpy()


def main():
    output_dir = '/home/nlp/matan_avitan/git/nopos_locating_new/results/qk_mechanism_analysis'
    os.makedirs(output_dir, exist_ok=True)

    device = 'cpu'  # CPU is fine for this analysis

    # =========================================================================
    # Load trained BOS@80 model
    # =========================================================================
    print("=" * 70)
    print("ANALYSIS 1: Trained BOS@80 Model - Bias Terms and Key Variance")
    print("=" * 70)

    bos80_ckpt = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism-bos80/R0/best_ckpt.pt'
    trained_model = load_model(bos80_ckpt, device)

    # Get embeddings
    embeddings = trained_model.wte.weight.data.numpy()  # [vocab_size, n_embd]
    print(f"Embeddings shape: {embeddings.shape}")

    # Analyze Head 7 (position-0 attention head) in Block 2
    print("\n--- Block 2, Head 7 (Position-0 Attention Head) ---")
    W_Q, W_K, b_Q, b_K = extract_qk_params(trained_model, block_idx=1, head_idx=7)

    # Convert to numpy
    W_Q = W_Q.numpy()
    W_K = W_K.numpy()
    b_Q = b_Q.numpy() if b_Q is not None else None
    b_K = b_K.numpy() if b_K is not None else None

    # Bias analysis
    bias_analysis = analyze_bias_contribution(W_Q, W_K, b_Q, b_K, embeddings)
    print("\nBias Term Analysis:")
    print(f"  Term 1 (q·k, token-dependent): mean={bias_analysis['term1_qk'][0]:.4f}, std={bias_analysis['term1_qk'][1]:.4f}")
    print(f"  Term 2 (b_Q·k):                mean={bias_analysis['term2_bq_k'][0]:.4f}, std={bias_analysis['term2_bq_k'][1]:.4f}")
    print(f"  Term 3 (q·b_K):                mean={bias_analysis['term3_q_bk'][0]:.4f}, std={bias_analysis['term3_q_bk'][1]:.4f}")
    print(f"  Term 4 (b_Q·b_K, constant):    {bias_analysis['term4_bq_bk']:.4f}")
    print(f"  Bias norms: ||b_Q||={bias_analysis['bias_norms'][0]:.4f}, ||b_K||={bias_analysis['bias_norms'][1]:.4f}")

    # Key variance analysis
    key_analysis = analyze_key_variance(W_K, embeddings)
    print("\nKey Variance Analysis:")
    print(f"  Total variance of keys: {key_analysis['total_variance']:.6f}")
    print(f"  Mean variance per dim:  {key_analysis['mean_var_per_dim']:.6f}")
    print(f"  Key norm: mean={key_analysis['norm_mean']:.4f}, std={key_analysis['norm_std']:.4f}")
    print(f"  Distance from mean key: mean={key_analysis['dist_from_mean_mean']:.4f}, std={key_analysis['dist_from_mean_std']:.4f}")

    # Compare to Head 9 (BOS attention head)
    print("\n--- Block 2, Head 9 (BOS Token Attention Head) ---")
    W_Q9, W_K9, b_Q9, b_K9 = extract_qk_params(trained_model, block_idx=1, head_idx=9)
    W_Q9, W_K9 = W_Q9.numpy(), W_K9.numpy()
    b_Q9 = b_Q9.numpy() if b_Q9 is not None else None
    b_K9 = b_K9.numpy() if b_K9 is not None else None

    bias_analysis9 = analyze_bias_contribution(W_Q9, W_K9, b_Q9, b_K9, embeddings)
    print(f"  Term 4 (b_Q·b_K, constant): {bias_analysis9['term4_bq_bk']:.4f}")

    key_analysis9 = analyze_key_variance(W_K9, embeddings)
    print(f"  Total variance of keys: {key_analysis9['total_variance']:.6f}")

    # =========================================================================
    # Check attention at initialization
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Attention Patterns at Initialization (No Training)")
    print("=" * 70)

    fresh_model = create_fresh_model(device)
    attn1_init, attn2_init = compute_attention_at_init(fresh_model, device)

    print("\nAttention to position 0 at initialization:")
    print("\nBlock 1:")
    for h in range(12):
        attn_to_0 = attn1_init[:, h, :, 0].mean()
        print(f"  Head {h:2d}: {attn_to_0:.4f}")

    print("\nBlock 2:")
    for h in range(12):
        attn_to_0 = attn2_init[:, h, :, 0].mean()
        print(f"  Head {h:2d}: {attn_to_0:.4f}")

    # Compare to trained model
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Attention at Init vs After Training")
    print("=" * 70)

    # Get attention from trained model on same random data
    trained_model.eval()
    x = torch.randint(0, 50256, (64, 128))
    with torch.no_grad():
        _ = trained_model(x, capture_taps=True)
        attn1_trained, attn2_trained = trained_model.get_attention_weights()
    attn1_trained = attn1_trained.numpy()
    attn2_trained = attn2_trained.numpy()

    print("\nBlock 2 - Attention to Position 0:")
    print(f"{'Head':<6} {'At Init':<12} {'After Training':<15} {'Change':<10}")
    print("-" * 45)
    for h in range(12):
        init_attn = attn2_init[:, h, :, 0].mean()
        trained_attn = attn2_trained[:, h, :, 0].mean()
        change = trained_attn - init_attn
        marker = " <-- INCREASED" if change > 0.1 else ""
        print(f"{h:<6} {init_attn:<12.4f} {trained_attn:<15.4f} {change:+.4f}{marker}")

    # =========================================================================
    # Visualization
    # =========================================================================
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: Bias term magnitudes
    ax = axes[0, 0]
    terms = ['q·k\n(token)', 'b_Q·k', 'q·b_K', 'b_Q·b_K\n(const)']
    values = [
        bias_analysis['term1_qk'][0],
        bias_analysis['term2_bq_k'][0],
        bias_analysis['term3_q_bk'][0],
        abs(bias_analysis['term4_bq_bk']),
    ]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    ax.bar(terms, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean Magnitude')
    ax.set_title('Head 7: Attention Score Components', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Key vector variance comparison
    ax = axes[0, 1]
    heads = list(range(12))
    variances = []
    for h in heads:
        W_Q_h, W_K_h, _, _ = extract_qk_params(trained_model, block_idx=1, head_idx=h)
        kv = analyze_key_variance(W_K_h.numpy(), embeddings)
        variances.append(kv['total_variance'])

    bars = ax.bar(heads, variances, color='#4C72B0', edgecolor='black', linewidth=0.5)
    bars[7].set_color('#C44E52')  # Highlight Head 7
    bars[9].set_color('#55A868')  # Highlight Head 9
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Key Vector Variance')
    ax.set_title('Block 2: Key Vector Variance by Head', fontweight='bold')
    ax.legend([bars[7], bars[9]], ['Head 7 (pos-0)', 'Head 9 (BOS)'], loc='upper right')

    # Plot 3: Attention to position 0 at init
    ax = axes[1, 0]
    init_attn_to_0 = [attn2_init[:, h, :, 0].mean() for h in range(12)]
    ax.bar(range(12), init_attn_to_0, color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax.axhline(y=1/128, color='red', linestyle='--', label='Uniform (1/128)', linewidth=2)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Mean Attention to Position 0')
    ax.set_title('Block 2 at Initialization: Attention to Pos 0', fontweight='bold')
    ax.legend()

    # Plot 4: Attention change after training
    ax = axes[1, 1]
    trained_attn_to_0 = [attn2_trained[:, h, :, 0].mean() for h in range(12)]
    width = 0.35
    x = np.arange(12)
    ax.bar(x - width/2, init_attn_to_0, width, label='At Init', color='#4C72B0', alpha=0.7)
    ax.bar(x + width/2, trained_attn_to_0, width, label='After Training', color='#C44E52', alpha=0.7)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Mean Attention to Position 0')
    ax.set_title('Block 2: Position-0 Attention Before/After Training', fontweight='bold')
    ax.legend()
    ax.set_xticks(x)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'qk_mechanism_analysis.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'qk_mechanism_analysis.pdf'), bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'qk_mechanism_analysis.png')}")
    plt.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Hypothesis 1 (Bias Terms Dominate):
  - Constant term b_Q·b_K = {bias_analysis['term4_bq_bk']:.4f}
  - Token-dependent term q·k = {bias_analysis['term1_qk'][0]:.4f} (mean)
  - {'YES: Bias dominates' if abs(bias_analysis['term4_bq_bk']) > bias_analysis['term1_qk'][0] else 'NO: Token term is larger'}

Hypothesis 2 (Low Key Variance):
  - Key variance for Head 7: {key_analysis['total_variance']:.6f}
  - Key variance for Head 9: {key_analysis9['total_variance']:.6f}
  - {'YES: Head 7 has lower variance' if key_analysis['total_variance'] < key_analysis9['total_variance'] else 'NO: Head 7 does NOT have lower variance'}

Hypothesis 5 (Initialization):
  - At init, attention to pos 0 ≈ {np.mean(init_attn_to_0):.4f} (expected for uniform: {1/128:.4f})
  - After training, Head 7 attention to pos 0 = {trained_attn_to_0[7]:.4f}
  - {'Position 0 attention EMERGES during training' if trained_attn_to_0[7] > 0.5 else 'Position 0 attention already present at init'}
""")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
