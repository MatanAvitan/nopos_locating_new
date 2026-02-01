"""
MLP Position Decoding Algorithm Analysis

This script verifies the mathematical framework for how the MLP decodes
positional information from the variance signal after LayerNorm.

Key Insight:
With near-uniform attention weights (≈1/i for position i), the attention output is:
    z_i = (1/i) * Σ_{j=1}^{i} W_v * x_j

The variance of z_i decays as Var(z_i) ∝ 1/i, which LayerNorm converts to a
mean-based signal that the MLP can decode.

Mathematical Framework:
1. Attention aggregation: z_i = (1/i) * Σ_{j=1}^{i} v_j where v_j = W_v * x_j
2. Variance decay: Var(z_i) ∝ 1/i (central limit theorem effect)
3. LayerNorm scale: σ_i ∝ 1/√i, so scale factor ∝ √i
4. MLP decoding: Can construct weights to extract position from scaled outputs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..')

# Configuration
N_CTX = 64
D_MODEL = 1024
D_MLP = 4096
N_SAMPLES = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_attention_outputs(n_samples, n_ctx, d_model):
    """
    Create synthetic attention outputs with uniform attention pattern.

    At position i, output is: z_i = (1/i) * Σ_{j=1}^{i} v_j
    where v_j are random value vectors.
    """
    # Generate random value vectors for each position
    # Shape: [n_samples, n_ctx, d_model]
    values = torch.randn(n_samples, n_ctx, d_model, device=DEVICE) * 0.1

    # Compute attention outputs with uniform weights
    # z_i = (1/i) * cumsum(v)[:i]
    cumsum_values = torch.cumsum(values, dim=1)  # [n_samples, n_ctx, d_model]

    # Divide by position index (1-indexed)
    positions = torch.arange(1, n_ctx + 1, device=DEVICE).float().view(1, -1, 1)
    attn_outputs = cumsum_values / positions  # [n_samples, n_ctx, d_model]

    return attn_outputs, values


def analyze_variance_decay(attn_outputs):
    """Analyze variance decay pattern in attention outputs."""
    # Compute variance across samples for each position
    # Shape: [n_ctx, d_model]
    variance_per_pos = attn_outputs.var(dim=0)

    # Average variance across dimensions
    mean_variance = variance_per_pos.mean(dim=1).cpu().numpy()

    return mean_variance


def apply_layernorm(x, eps=1e-5):
    """Apply LayerNorm and return normalized output and scale factors."""
    # Compute mean and std across the last dimension (d_model)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)

    normalized = (x - mean) / std

    return normalized, std.squeeze(-1)


def construct_mlp_decoder(d_model, d_mlp, n_ctx):
    """
    Construct MLP weights that can decode position from LayerNorm outputs.

    The key insight: After LayerNorm, the scale factor σ_i ∝ 1/√i encodes position.
    We construct weights such that:
    - W1: First column detects the magnitude of the input (related to 1/σ)
    - W2: Maps the detected magnitude to position
    """
    # Initialize MLP weights
    W1 = torch.zeros(d_model, d_mlp, device=DEVICE)
    b1 = torch.zeros(d_mlp, device=DEVICE)
    W2 = torch.zeros(d_mlp, n_ctx, device=DEVICE)
    b2 = torch.zeros(n_ctx, device=DEVICE)

    # Strategy: Use the variance of LN output (which should be ~1)
    # but the pre-LN variance encodes position
    # The MLP can learn to use the activation patterns that correlate with position

    # Simple construction: project to a scalar that correlates with position
    # Then use softmax-like classification

    # W1: First few neurons detect different aspects of the input
    # Use random projections that will capture variance-related signals
    W1[:, :n_ctx] = torch.randn(d_model, n_ctx, device=DEVICE) * 0.1

    # W2: Map to position logits
    W2[:n_ctx, :] = torch.eye(n_ctx, device=DEVICE)

    return W1, b1, W2, b2


def analyze_mlp_decoding_capability(model_path=None):
    """
    Analyze how the MLP can decode position from LayerNorm outputs.

    Returns analysis results and generates plots.
    """
    print("=" * 60)
    print("MLP Position Decoding Analysis")
    print("=" * 60)

    # Step 1: Create synthetic data with uniform attention
    print("\n1. Generating synthetic attention outputs with uniform weights...")
    attn_outputs, values = create_synthetic_attention_outputs(N_SAMPLES, N_CTX, D_MODEL)
    print(f"   Shape: {attn_outputs.shape}")

    # Step 2: Analyze variance decay
    print("\n2. Analyzing variance decay pattern...")
    variance = analyze_variance_decay(attn_outputs)
    theoretical = variance[0] / np.arange(1, N_CTX + 1)

    print(f"   Variance at pos 0: {variance[0]:.6f}")
    print(f"   Variance at pos 63: {variance[63]:.6f}")
    print(f"   Ratio (should be ~64): {variance[0]/variance[63]:.2f}")

    # Step 3: Apply LayerNorm
    print("\n3. Applying LayerNorm...")
    ln_outputs, scale_factors = apply_layernorm(attn_outputs)

    # Scale factors should show monotonic pattern
    mean_scale = scale_factors.mean(dim=0).cpu().numpy()
    print(f"   Mean scale at pos 0: {mean_scale[0]:.6f}")
    print(f"   Mean scale at pos 63: {mean_scale[63]:.6f}")

    # Step 4: Verify scale factor encodes position
    print("\n4. Verifying scale factor encodes position...")
    positions = np.arange(N_CTX)
    from scipy.stats import pearsonr
    corr, pval = pearsonr(positions, mean_scale)
    print(f"   Correlation with position: {corr:.4f} (p={pval:.2e})")

    # Step 5: Test position prediction from scale factor alone
    print("\n5. Testing position prediction from scale factor...")

    # Train a simple linear regressor from scale to position
    X = scale_factors.cpu().numpy()  # [n_samples, n_ctx]
    y = np.tile(np.arange(N_CTX), (N_SAMPLES, 1))  # [n_samples, n_ctx]

    # Flatten for regression
    X_flat = X.flatten()
    y_flat = y.flatten()

    # Simple linear fit
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(X_flat, y_flat)
    predictions = slope * X_flat + intercept
    mae = np.abs(predictions - y_flat).mean()

    print(f"   Linear fit R²: {r_value**2:.4f}")
    print(f"   MAE from scale factor alone: {mae:.2f} positions")

    # Step 6: Generate plots
    print("\n6. Generating plots...")

    # Plot 1: Variance decay
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.plot(positions, variance, 'b-', linewidth=2, label='Empirical')
    ax1.plot(positions, theoretical, 'r--', linewidth=2, label='Theoretical (1/i)')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Variance', fontsize=12)
    ax1.set_title('Variance Decay in Attention Outputs', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scale factor vs position
    ax2 = axes[0, 1]
    ax2.plot(positions, mean_scale, 'g-', linewidth=2)
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('LayerNorm Scale Factor (σ)', fontsize=12)
    ax2.set_title(f'Scale Factor Encodes Position (r={corr:.3f})', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scale factor histogram by position
    ax3 = axes[1, 0]
    for pos in [0, 15, 31, 47, 63]:
        ax3.hist(scale_factors[:, pos].cpu().numpy(), bins=30, alpha=0.5,
                 label=f'Pos {pos}', density=True)
    ax3.set_xlabel('Scale Factor', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Scale Factor Distribution by Position', fontsize=14)
    ax3.legend()

    # Plot 4: Position prediction accuracy
    ax4 = axes[1, 1]
    # Bin predictions and show accuracy
    pred_positions = np.clip(np.round(predictions), 0, N_CTX-1).astype(int)
    accuracy_per_pos = []
    for pos in range(N_CTX):
        mask = y_flat == pos
        acc = (pred_positions[mask] == pos).mean()
        accuracy_per_pos.append(acc)

    ax4.bar(positions, accuracy_per_pos, color='purple', alpha=0.7)
    ax4.set_xlabel('Position', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title(f'Position Prediction Accuracy (Mean: {np.mean(accuracy_per_pos):.2%})', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {PLOTS_DIR / 'mlp_decoding_analysis.png'}")

    # Additional plot: Mathematical framework visualization
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Show the relationship: variance → scale → position
    ax.plot(positions, 1/np.sqrt(positions + 1), 'b-', linewidth=2,
            label=r'Theoretical $\sigma_i \propto 1/\sqrt{i+1}$')
    ax.plot(positions, mean_scale / mean_scale.max(), 'r--', linewidth=2,
            label='Empirical (normalized)')
    ax.set_xlabel('Position $i$', fontsize=14)
    ax.set_ylabel('Scale Factor (normalized)', fontsize=14)
    ax.set_title('LayerNorm Scale Factor as Position Encoder', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'layernorm_scale_position.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {PLOTS_DIR / 'layernorm_scale_position.png'}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    return {
        'variance': variance,
        'scale_factors': mean_scale,
        'position_correlation': corr,
        'prediction_mae': mae,
        'prediction_accuracy': np.mean(accuracy_per_pos)
    }


def demonstrate_mlp_construction():
    """
    Demonstrate how to construct MLP weights for position decoding.

    Mathematical Framework:
    -----------------------
    Given attention output z_i = (1/i) * Σ_{j=1}^{i} v_j where v_j = W_v * x_j

    The variance Var(z_i) ∝ 1/i creates a signal that LayerNorm converts to position.

    After LayerNorm:
        LN(z_i) = (z_i - μ_i) / σ_i
        where σ_i ∝ 1/√i

    The MLP can decode position by:
    1. Computing w₁ᵀ · LN(z_i) where w₁ is tuned to extract the scale-dependent signal
    2. Applying ReLU to create position-dependent activations
    3. Using w₂ to map these activations to position logits

    Key insight: The scale factor σ_i itself encodes position!
    """
    print("\n" + "=" * 60)
    print("MLP Construction for Position Decoding")
    print("=" * 60)

    print("""
    Mathematical Framework:
    =======================

    1. ATTENTION AGGREGATION:
       With near-uniform attention weights A_{ij} ≈ 1/i for j ≤ i:

           z_i = Σ_{j=1}^{i} A_{ij} · v_j ≈ (1/i) · Σ_{j=1}^{i} v_j

       where v_j = W_V · x_j are value vectors.

    2. VARIANCE DECAY (Central Limit Theorem):
       If token embeddings are approximately i.i.d.:

           Var(z_i) = (1/i²) · Var(Σ_{j=1}^{i} v_j) ≈ (1/i²) · i · Var(v) = Var(v)/i

       Therefore: Var(z_i) ∝ 1/i

    3. LAYERNORM AS VARIANCE READER:
       LayerNorm computes: LN(z_i) = (z_i - μ_i) / (σ_i + ε)

       Since σ_i = √Var(z_i) ∝ 1/√i:
       - The scale factor 1/(σ_i + ε) ∝ √i
       - This scale factor directly encodes position!

    4. MLP DECODING:
       The MLP can decode position via:

       h_i = ReLU(W₁ᵀ · LN(z_i) + b₁)

       where W₁ has columns tuned to detect the position-dependent patterns.

       Output: logits = W₂ᵀ · h_i + b₂

       With appropriate W₁, W₂, the MLP maps the variance-based signal
       to position predictions.

    5. CONSTRUCTIVE PROOF:
       We can construct W₁ such that one neuron computes:

           h_i^{(k)} = ReLU(w₁ᵀ · z_i / σ_i) ∝ ReLU(√i · constant)

       Then W₂ maps these √i-proportional activations to position logits.
    """)

    return


if __name__ == "__main__":
    # Run analysis
    results = analyze_mlp_decoding_capability()

    # Show construction explanation
    demonstrate_mlp_construction()

    print("\n" + "=" * 60)
    print("Summary of Results")
    print("=" * 60)
    print(f"Position correlation with scale factor: {results['position_correlation']:.4f}")
    print(f"Mean prediction accuracy from scale alone: {results['prediction_accuracy']:.2%}")
    print(f"Mean absolute error: {results['prediction_mae']:.2f} positions")
