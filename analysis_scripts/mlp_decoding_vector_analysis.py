"""
MLP Position Decoding via Token Embedding Orthogonality

Mathematical Framework:
=======================
1. Attention output at position i with uniform weights:
   z_i = (1/i) * Σ_{j=1}^{i} v_j   where v_j = W_V * x_j

2. Key insight: Xavier-initialized embeddings are approximately orthogonal in high dimensions
   - v_k · v_j ≈ 0 when k ≠ j (orthogonality)
   - v_k · v_k = ||v_k||² > 0 (self-similarity)

3. Decoding vector construction: w_k = v_k for each token k

4. Dot product for position decoding:
   w_k · z_i = (1/i) * Σ_{j=1}^{i} v_k · v_j

   - If token k appears at some position j ≤ i: contributes ||v_k||² / i
   - If token k doesn't appear: contributes ≈ 0

5. Summing over all tokens that appear: Σ_k (w_k · z_i) ≈ i * (||v||²/i) = ||v||²
   But more importantly: the number of positive contributions = position i
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
D_VOCAB = 5000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def xavier_init(shape, gain=1.0):
    """Xavier initialization - naturally produces approximately orthogonal vectors in high dimensions."""
    fan_in, fan_out = shape[0], shape[1]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return torch.randn(*shape, device=DEVICE) * std


def verify_orthogonality(embeddings):
    """Verify that Xavier-initialized embeddings are approximately orthogonal."""
    # Normalize embeddings
    normed = embeddings / embeddings.norm(dim=1, keepdim=True)

    # Compute pairwise dot products
    dot_products = normed @ normed.T

    # Extract off-diagonal elements
    mask = ~torch.eye(dot_products.shape[0], dtype=bool, device=DEVICE)
    off_diagonal = dot_products[mask]

    return {
        'mean_off_diagonal': off_diagonal.mean().item(),
        'std_off_diagonal': off_diagonal.std().item(),
        'max_off_diagonal': off_diagonal.abs().max().item(),
        'diagonal_mean': dot_products.diag().mean().item()
    }


def run_decoding_experiment():
    """
    Run the MLP position decoding experiment.

    Key experiment:
    - Create token embeddings with Xavier initialization
    - Compute value vectors v_j = W_V * x_j
    - Compute attention outputs z_i = (1/i) * Σ_{j=1}^{i} v_j
    - Compute dot products v_k · z_i for decoding
    - Verify that sum of positive contributions ≈ position i
    """
    print("=" * 70)
    print("MLP Position Decoding via Token Embedding Orthogonality")
    print("=" * 70)

    # Step 1: Initialize embeddings with Xavier (naturally approximately orthogonal)
    print("\n1. Initializing token embeddings with Xavier initialization...")
    W_E = xavier_init((D_VOCAB, D_MODEL))
    W_V = xavier_init((D_MODEL, D_MODEL))

    # Verify orthogonality
    print("   Verifying orthogonality of embeddings...")
    ortho_stats = verify_orthogonality(W_E[:1000])  # Check first 1000 tokens
    print(f"   Mean off-diagonal dot product: {ortho_stats['mean_off_diagonal']:.6f}")
    print(f"   Std off-diagonal dot product: {ortho_stats['std_off_diagonal']:.6f}")
    print(f"   Max |off-diagonal|: {ortho_stats['max_off_diagonal']:.6f}")
    print(f"   Mean diagonal (self-similarity): {ortho_stats['diagonal_mean']:.6f}")

    # Step 2: Generate a random sequence of tokens
    print("\n2. Generating random token sequence...")
    tokens = torch.randint(0, D_VOCAB, (N_CTX,), device=DEVICE)
    print(f"   Sequence length: {N_CTX}")
    print(f"   Unique tokens: {len(tokens.unique())}")

    # Step 3: Compute value vectors v_j = W_V * x_j
    print("\n3. Computing value vectors v_j = W_V @ x_j...")
    x = W_E[tokens]  # [N_CTX, D_MODEL]
    v = x @ W_V.T    # [N_CTX, D_MODEL] - value vectors

    # Compute norms of value vectors
    v_norms = v.norm(dim=1)
    print(f"   Mean ||v_j||: {v_norms.mean().item():.4f}")
    print(f"   Std ||v_j||: {v_norms.std().item():.4f}")

    # Step 4: Compute attention outputs z_i = (1/i) * Σ_{j=1}^{i} v_j
    print("\n4. Computing attention outputs with uniform weights...")
    cumsum_v = torch.cumsum(v, dim=0)  # [N_CTX, D_MODEL]
    positions = torch.arange(1, N_CTX + 1, device=DEVICE).float().unsqueeze(1)
    z = cumsum_v / positions  # [N_CTX, D_MODEL] - attention outputs

    # Step 5: Compute dot products for decoding
    print("\n5. Computing decoding dot products...")

    # For each position i, compute v_k · z_i for all tokens k that appeared up to position i
    # This is the key decoding mechanism

    # Store results for plotting
    results = {
        'positions': [],
        'decoded_positions': [],
        'individual_contributions': {},
        'contribution_matrices': []
    }

    # Analyze specific positions (0-indexed)
    positions_to_analyze = [4, 9, 19, 31, 49, 62]  # Corresponds to positions 5, 10, 20, 32, 50, 63

    for pos_idx in positions_to_analyze:
        # Tokens that appeared up to position pos_idx
        tokens_up_to_pos = tokens[:pos_idx + 1]

        # Value vectors for those tokens
        v_contributing = v[:pos_idx + 1]  # [pos_idx+1, D_MODEL]

        # Attention output at this position
        z_i = z[pos_idx]  # [D_MODEL]

        # Compute dot products: v_j · z_i for each contributing token
        dot_products = (v_contributing * z_i).sum(dim=1)  # [pos_idx+1]

        # Store results
        results['positions'].append(pos_idx + 1)  # 1-indexed position
        results['individual_contributions'][pos_idx + 1] = dot_products.cpu().numpy()

        # The decoded position estimate: sum of dot products normalized
        # Since each contributing token gives approximately ||v||²/i contribution,
        # and there are i tokens, the sum should be approximately ||v||²
        # But the COUNT of positive contributions = position
        positive_contributions = (dot_products > 0).sum().item()
        results['decoded_positions'].append(positive_contributions)

        print(f"\n   Position {pos_idx + 1}:")
        print(f"     Tokens contributing: {pos_idx + 1}")
        print(f"     Positive dot products: {positive_contributions}")
        print(f"     Mean dot product: {dot_products.mean().item():.6f}")
        print(f"     Sum of dot products: {dot_products.sum().item():.4f}")

    # Step 6: Full position decoding for all positions
    print("\n6. Computing full position decoding curve...")
    decoded_all = []
    sum_dot_products_all = []

    for pos_idx in range(N_CTX):
        v_contributing = v[:pos_idx + 1]
        z_i = z[pos_idx]
        dot_products = (v_contributing * z_i).sum(dim=1)

        # Count positive contributions
        positive_count = (dot_products > 0).sum().item()
        decoded_all.append(positive_count)
        sum_dot_products_all.append(dot_products.sum().item())

    results['all_decoded'] = decoded_all
    results['all_sums'] = sum_dot_products_all

    # Compute correlation
    from scipy.stats import pearsonr, spearmanr
    true_positions = np.arange(1, N_CTX + 1)
    pearson_r, _ = pearsonr(true_positions, decoded_all)
    spearman_r, _ = spearmanr(true_positions, decoded_all)

    print(f"\n   Pearson correlation: {pearson_r:.4f}")
    print(f"   Spearman correlation: {spearman_r:.4f}")

    results['pearson'] = pearson_r
    results['spearman'] = spearman_r

    return results, v, z, tokens


def generate_plots(results, v, z, tokens):
    """Generate visualization plots for the decoding analysis."""
    print("\n7. Generating plots...")

    # Plot 1: Individual dot product contributions for several positions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    positions_to_plot = [5, 10, 20, 32, 50, 63]  # 1-indexed for display

    for idx, pos in enumerate(positions_to_plot):
        ax = axes[idx]
        contributions = results['individual_contributions'][pos]  # stored with 1-indexed keys

        # Create bar plot
        x = np.arange(len(contributions))
        colors = ['green' if c > 0 else 'red' for c in contributions]
        ax.bar(x, contributions, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Token Index j', fontsize=10)
        ax.set_ylabel('$v_j \\cdot z_i$', fontsize=10)
        ax.set_title(f'Position i={pos}\n(+: {(np.array(contributions) > 0).sum()}, true: {pos})', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Individual Dot Product Contributions $v_j \\cdot z_i$ at Different Positions', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: mlp_decoding_contributions.png")

    # Plot 2: Decoded position vs true position
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    true_positions = np.arange(1, N_CTX + 1)

    # Left: Count-based decoding
    ax1.scatter(true_positions, results['all_decoded'], alpha=0.7, s=50, c='blue')
    ax1.plot([0, N_CTX], [0, N_CTX], 'r--', linewidth=2, label='Perfect decoding')
    ax1.set_xlabel('True Position $i$', fontsize=12)
    ax1.set_ylabel('Decoded Position (count of +ve dot products)', fontsize=12)
    ax1.set_title(f'Position Decoding via Orthogonality\n(Pearson r = {results["pearson"]:.4f})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, N_CTX + 1)
    ax1.set_ylim(0, N_CTX + 1)

    # Right: Sum-based decoding
    ax2.scatter(true_positions, results['all_sums'], alpha=0.7, s=50, c='green')
    ax2.set_xlabel('True Position $i$', fontsize=12)
    ax2.set_ylabel('Sum of Dot Products $\\sum_j v_j \\cdot z_i$', fontsize=12)
    ax2.set_title('Sum of Contributions vs Position', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_vs_position.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: mlp_decoding_vs_position.png")

    # Plot 3: Orthogonality verification - dot product matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sample subset of value vectors for visualization
    n_sample = min(20, N_CTX)
    v_sample = v[:n_sample]
    v_normed = v_sample / v_sample.norm(dim=1, keepdim=True)
    dot_matrix = (v_normed @ v_normed.T).cpu().numpy()

    im1 = ax1.imshow(dot_matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    plt.colorbar(im1, ax=ax1, label='Dot Product')
    ax1.set_xlabel('Token j', fontsize=12)
    ax1.set_ylabel('Token k', fontsize=12)
    ax1.set_title('Normalized Dot Products $\\hat{v}_k \\cdot \\hat{v}_j$\n(Diagonal ≈ 1, Off-diagonal ≈ 0)', fontsize=12)

    # Histogram of off-diagonal elements
    mask = ~np.eye(n_sample, dtype=bool)
    off_diag = dot_matrix[mask]
    ax2.hist(off_diag, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linewidth=2, linestyle='--')
    ax2.set_xlabel('Dot Product Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Off-Diagonal Distribution\n(mean={off_diag.mean():.4f}, std={off_diag.std():.4f})', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_orthogonality_verification.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: mlp_orthogonality_verification.png")

    # Plot 4: Mathematical framework summary
    fig = plt.figure(figsize=(12, 8))

    # Main decoding plot
    ax_main = fig.add_subplot(111)
    ax_main.scatter(true_positions, results['all_decoded'], alpha=0.8, s=60, c='blue', label='Decoded')
    ax_main.plot([0, N_CTX], [0, N_CTX], 'r--', linewidth=2, label='y = x (perfect)')

    # Add error bars showing deviation
    errors = np.array(results['all_decoded']) - true_positions
    ax_main.fill_between(true_positions, true_positions + errors, true_positions,
                          alpha=0.2, color='blue', label='Deviation')

    ax_main.set_xlabel('True Position $i$', fontsize=14)
    ax_main.set_ylabel('Decoded Position', fontsize=14)
    ax_main.set_title('MLP Position Decoding via Token Embedding Orthogonality\n' +
                      r'$z_i = \frac{1}{i}\sum_{j=1}^{i} v_j$, decode by counting $\{j : v_j \cdot z_i > 0\}$',
                      fontsize=14)
    ax_main.legend(fontsize=11, loc='upper left')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, N_CTX + 2)
    ax_main.set_ylim(0, N_CTX + 2)

    # Add text box with statistics
    textstr = f'Pearson r = {results["pearson"]:.4f}\nSpearman ρ = {results["spearman"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=12,
                  verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mlp_decoding_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: mlp_decoding_summary.png")


def run_multiple_sequences():
    """Run decoding on multiple sequences to get average performance."""
    print("\n" + "=" * 70)
    print("Running Multi-Sequence Analysis")
    print("=" * 70)

    n_sequences = 100
    all_correlations = []
    all_mae = []

    # Initialize once
    W_E = xavier_init((D_VOCAB, D_MODEL))
    W_V = xavier_init((D_MODEL, D_MODEL))

    for seq_idx in range(n_sequences):
        # Generate random sequence
        tokens = torch.randint(0, D_VOCAB, (N_CTX,), device=DEVICE)

        # Compute value vectors
        x = W_E[tokens]
        v = x @ W_V.T

        # Compute attention outputs
        cumsum_v = torch.cumsum(v, dim=0)
        positions = torch.arange(1, N_CTX + 1, device=DEVICE).float().unsqueeze(1)
        z = cumsum_v / positions

        # Decode all positions
        decoded = []
        for pos_idx in range(N_CTX):
            v_contributing = v[:pos_idx + 1]
            z_i = z[pos_idx]
            dot_products = (v_contributing * z_i).sum(dim=1)
            positive_count = (dot_products > 0).sum().item()
            decoded.append(positive_count)

        # Compute metrics
        true_positions = np.arange(1, N_CTX + 1)
        from scipy.stats import pearsonr
        corr, _ = pearsonr(true_positions, decoded)
        mae = np.abs(np.array(decoded) - true_positions).mean()

        all_correlations.append(corr)
        all_mae.append(mae)

    print(f"\nResults over {n_sequences} sequences:")
    print(f"  Mean Pearson correlation: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}")
    print(f"  Mean MAE: {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f} positions")

    return all_correlations, all_mae


def print_mathematical_framework():
    """Print the mathematical framework explanation."""
    print("\n" + "=" * 70)
    print("MATHEMATICAL FRAMEWORK")
    print("=" * 70)
    print("""
    MLP Position Decoding via Token Embedding Orthogonality
    =======================================================

    1. ATTENTION OUTPUT WITH UNIFORM WEIGHTS:
       At position i, the attention output is:

           z_i = (1/i) × Σ_{j=1}^{i} v_j

       where v_j = W_V × x_j are value vectors for each token.

    2. KEY PROPERTY: APPROXIMATE ORTHOGONALITY
       Xavier-initialized embeddings in high dimensions are approximately orthogonal:

           v_k · v_j ≈ 0    when k ≠ j (different tokens)
           v_k · v_k = ||v_k||² > 0    (self-similarity)

    3. DECODING MECHANISM:
       Construct decoding vector w_k = v_k for each token k.

       The dot product w_k · z_i reveals position:

           w_k · z_i = v_k · [(1/i) × Σ_{j=1}^{i} v_j]
                     = (1/i) × Σ_{j=1}^{i} (v_k · v_j)

       By orthogonality:
           - If token k appears at position j ≤ i: contributes ||v_k||² / i > 0
           - If token k doesn't appear at j ≤ i: contributes ≈ 0

    4. POSITION RECOVERY:
       Count the number of positive dot products:

           decoded_position = |{j : v_j · z_i > 0}| ≈ i

       Each of the i tokens that contributed to z_i gives a positive dot product,
       while non-contributing tokens give approximately zero.

    5. MLP IMPLEMENTATION:
       The MLP can implement this by:
       - W_1: columns contain v_k for each token k (detecting each token)
       - ReLU: keeps only positive contributions
       - W_2: sums up the activations → gives position count
    """)


if __name__ == "__main__":
    # Print framework
    print_mathematical_framework()

    # Run main experiment
    results, v, z, tokens = run_decoding_experiment()

    # Generate plots
    generate_plots(results, v, z, tokens)

    # Run multi-sequence analysis
    all_corrs, all_mae = run_multiple_sequences()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nGenerated plots in: {PLOTS_DIR}")
    print("  - mlp_decoding_contributions.png")
    print("  - mlp_decoding_vs_position.png")
    print("  - mlp_orthogonality_verification.png")
    print("  - mlp_decoding_summary.png")
    print("\nKey findings:")
    print(f"  - Single sequence Pearson r: {results['pearson']:.4f}")
    print(f"  - Multi-sequence mean r: {np.mean(all_corrs):.4f}")
    print(f"  - Multi-sequence mean MAE: {np.mean(all_mae):.2f} positions")
