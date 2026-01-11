"""
Analyze trained NoPE models to test positional encoding emergence hypotheses.

This script tests three key hypotheses from the paper:
1. Uniform Attention Head: Does at least one attention head maintain near-uniform
   attention (variance decays as 1/(i+1))?
2. Decoding Vector: Can we decode position using w = W_V · Σ_j LN(E_j)?
3. Population Statistics: Does the network use population mean/std to infer position?

Usage:
    python analyze_trained_nope.py --checkpoint path/to/ckpt.pt --save_dir results/

For analyzing emergence over training:
    python analyze_trained_nope.py --checkpoint_dir out-nope-1layer-ln/ --save_dir results/ln/
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]

    # Create config - handle both old and new format
    config = GPTConfig(**model_args)
    config.log_attention_stats = False  # Disable during analysis

    model = GPT(config)

    # Load state dict
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def generate_random_sequences(model, n_samples, seq_len, device="cuda"):
    """Generate random token sequences for analysis."""
    vocab_size = model.config.vocab_size
    return torch.randint(0, vocab_size, (n_samples, seq_len), device=device)


def extract_attention_weights(model, input_ids):
    """
    Extract attention weights from all heads.

    Returns: attention weights [B, n_head, T, T]
    """
    model.eval()
    B, T = input_ids.shape

    with torch.no_grad():
        # Get embeddings
        tok_emb = model.transformer.wte(input_ids)
        x = model.transformer.drop(tok_emb)

        # Get attention weights from first (only) block
        block = model.transformer.h[0]
        x_ln = block.ln_1(x)

        # Manual attention computation to get weights
        n_head = model.config.n_head
        n_embd = model.config.n_embd
        head_dim = n_embd // n_head

        q, k, v = block.attn.c_attn(x_ln).split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        att = att.masked_fill(
            torch.triu(
                torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1
            ),
            float("-inf"),
        )
        att = F.softmax(att, dim=-1)

    return att


def analyze_attention_uniformity(model, n_samples=1000, seq_len=256, device="cuda"):
    """
    Hypothesis 1: Test if attention heads have near-uniform patterns.

    For uniform causal attention at position i, the variance of attention weights
    should be approximately 1/(i+1) (each of i+1 positions gets weight 1/(i+1)).

    We measure correlation between actual variance and expected 1/(i+1) pattern.
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 1: Attention Uniformity Analysis")
    print("=" * 60)

    # Generate random sequences
    input_ids = generate_random_sequences(model, n_samples, seq_len, device)

    # Extract attention weights
    with torch.no_grad():
        att = extract_attention_weights(model, input_ids)  # [B, H, T, T]

    # Compute variance of attention weights at each position
    att_var = att.var(dim=-1)  # [B, H, T]
    att_var_mean = att_var.mean(dim=0)  # [H, T] - average over samples

    # Expected uniform variance: for position i with uniform attention over 0..i,
    # variance = mean((w - 1/(i+1))^2) = 1/(i+1) - 1/(i+1)^2 = i/((i+1)^2*(i+1))
    # Actually, for uniform distribution over i+1 values, var = 0 ideally
    # But empirically, variance ~ 1/(i+1) captures the spread
    positions = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32)
    expected_uniform_var = 1.0 / positions

    # Compute correlation for each head
    n_heads = model.config.n_head
    correlations = []

    print(f"\nAnalyzing {n_heads} attention heads over {n_samples} samples:")
    print("-" * 50)

    for h in range(n_heads):
        actual = att_var_mean[h].cpu().numpy()
        expected = expected_uniform_var.cpu().numpy()
        corr, p_val = pearsonr(actual, expected)
        correlations.append(corr)

        if corr > 0.9:
            status = "HIGHLY UNIFORM"
        elif corr > 0.5:
            status = "PARTIALLY UNIFORM"
        else:
            status = "NOT UNIFORM"
        print(f"  Head {h:2d}: r = {corr:+.4f} (p={p_val:.2e}) [{status}]")

    # Summary
    best_head = int(np.argmax(correlations))
    n_uniform = sum(1 for c in correlations if c > 0.9)
    n_partial = sum(1 for c in correlations if 0.5 < c <= 0.9)

    print(f"\n  Summary:")
    print(f"    Most uniform head: {best_head} (r={correlations[best_head]:.4f})")
    print(f"    Highly uniform heads (r>0.9): {n_uniform}")
    print(f"    Partially uniform heads (0.5<r≤0.9): {n_partial}")

    return {
        "correlations": correlations,
        "best_head": best_head,
        "n_uniform_heads": n_uniform,
        "att_var_mean": att_var_mean.cpu().numpy(),
        "expected_var": expected_uniform_var.cpu().numpy(),
    }


def compute_decoding_vector(model, device="cuda"):
    """
    Compute the decoding vector: w = W_V · Σ_j LN(E_j)

    This vector can be used to decode position from post-attention representations.
    """
    with torch.no_grad():
        block = model.transformer.h[0]
        n_embd = model.config.n_embd

        # Get embeddings
        E = model.transformer.wte.weight  # [vocab_size, n_embd]

        # Apply normalization to each embedding
        ln = block.ln_1
        E_norm = ln(E)  # [vocab_size, n_embd]

        # Sum of normalized embeddings
        E_sum = E_norm.sum(dim=0)  # [n_embd]

        # Get W_V from attention (c_attn contains Q, K, V concatenated)
        W_qkv = block.attn.c_attn.weight  # [3*n_embd, n_embd]
        W_V = W_qkv[2 * n_embd :, :]  # [n_embd, n_embd]

        # Decoding vector: w = W_V @ E_sum
        w = W_V @ E_sum  # [n_embd]

    return w


def analyze_decoding_vector(model, n_samples=1000, seq_len=256, device="cuda"):
    """
    Hypothesis 2: Test if position can be decoded using the decoding vector.

    The decoding vector w = W_V · Σ_j LN(E_j) should allow position decoding
    when projected onto post-attention representations.
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 2: Decoding Vector Analysis")
    print("=" * 60)

    # Compute decoding vector
    w = compute_decoding_vector(model, device)
    print(f"\nDecoding vector: shape={w.shape}, norm={w.norm().item():.4f}")

    # Generate test data
    input_ids = generate_random_sequences(model, n_samples, seq_len, device)

    with torch.no_grad():
        # Get post-attention representations
        tok_emb = model.transformer.wte(input_ids)
        x = model.transformer.drop(tok_emb)

        block = model.transformer.h[0]
        x_ln = block.ln_1(x)
        x_attn = x + block.attn(x_ln)  # Post-attention with residual

        # Project onto decoding vector
        projections = x_attn @ w  # [B, T]

        # Average over samples for stable estimate
        proj_mean = projections.mean(dim=0).cpu().numpy()  # [T]

    # Correlate with true positions
    true_positions = np.arange(seq_len)
    corr, p_val = pearsonr(proj_mean, true_positions)
    spearman_corr, _ = spearmanr(proj_mean, true_positions)

    print(f"\nProjection-based decoding:")
    print(f"  Pearson correlation:  r = {corr:+.4f} (p={p_val:.2e})")
    print(f"  Spearman correlation: ρ = {spearman_corr:+.4f}")

    # Test counting method (how many value vectors have positive projection)
    print("\nCounting method analysis:")

    with torch.no_grad():
        E = model.transformer.wte.weight
        block = model.transformer.h[0]
        E_norm = block.ln_1(E)
        W_qkv = block.attn.c_attn.weight
        n_embd = model.config.n_embd
        W_V = W_qkv[2 * n_embd :, :]
        V = E_norm @ W_V.T  # [vocab_size, n_embd]

        # For subset of samples, count positive dot products
        n_test = min(100, n_samples)
        decoded_positions = []

        for b in range(n_test):
            tokens = input_ids[b]
            for i in range(seq_len):
                # Get value vectors for tokens 0..i
                v_prefix = V[tokens[: i + 1]]  # [i+1, n_embd]
                v_sum = v_prefix.sum(dim=0)  # [n_embd]
                z_i = v_sum / (i + 1)  # Average (simulating uniform attention)

                # Count how many tokens have positive projection
                count = (v_prefix @ z_i > 0).sum().item()
                decoded_positions.append((i, count))

    true_pos = np.array([p[0] for p in decoded_positions])
    decoded = np.array([p[1] for p in decoded_positions])
    count_corr, _ = pearsonr(true_pos, decoded)
    count_mae = np.abs(decoded - true_pos).mean()

    print(f"  Counting correlation: r = {count_corr:+.4f}")
    print(f"  Counting MAE: {count_mae:.2f}")

    return {
        "decoding_vector": w.cpu().numpy(),
        "projections": proj_mean,
        "pearson_corr": corr,
        "spearman_corr": spearman_corr,
        "counting_corr": count_corr,
        "counting_mae": count_mae,
    }


def analyze_population_statistics(model, n_samples=2000, seq_len=256, device="cuda"):
    """
    Hypothesis 3: Test if population statistics encode position.

    The LayerNorm paradox: while LN normalizes each sample to zero mean,
    population-level expectations differ by position due to position-correlated
    token distributions in causal attention.
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 3: Population Statistics Analysis")
    print("=" * 60)

    # Generate random sequences
    input_ids = generate_random_sequences(model, n_samples, seq_len, device)

    # Collect activations at different stages
    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        x = model.transformer.drop(tok_emb)

        block = model.transformer.h[0]

        # Post-LN1 (before attention)
        x_ln1 = block.ln_1(x)

        # Post-attention (with residual)
        x_attn = x + block.attn(x_ln1)

        # Post-LN2 (before MLP)
        x_ln2 = block.ln_2(x_attn)

        # Post-MLP (with residual)
        x_mlp = x_attn + block.mlp(x_ln2)

        # Final LN
        x_final = model.transformer.ln_f(x_mlp)

    # Analyze population statistics at each stage
    stages = {
        "post_embed": tok_emb,
        "post_ln1": x_ln1,
        "post_attn": x_attn,
        "post_ln2": x_ln2,
        "post_mlp": x_mlp,
        "final_ln": x_final,
    }

    print(f"\nPopulation statistics analysis ({n_samples} samples):")
    print("-" * 60)

    results = {}
    true_positions = np.arange(seq_len)

    for name, activations in stages.items():
        with torch.no_grad():
            # Population mean per position: E[h_i] over samples
            pop_mean = activations.mean(dim=0)  # [T, n_embd]
            pop_std = activations.std(dim=0)  # [T, n_embd]

            # Summarize across embedding dimensions
            mean_of_means = pop_mean.mean(dim=-1).cpu().numpy()  # [T]
            std_of_means = pop_mean.std(dim=-1).cpu().numpy()  # [T]
            mean_of_stds = pop_std.mean(dim=-1).cpu().numpy()  # [T]

            # Correlate with position
            mean_corr, _ = pearsonr(mean_of_means, true_positions)
            std_corr, _ = pearsonr(mean_of_stds, true_positions)

            # Variance across positions (how much position info in mean)
            position_info = np.var(mean_of_means)

        print(
            f"  {name:12s}: mean_corr={mean_corr:+.4f}, std_corr={std_corr:+.4f}, pos_info={position_info:.6f}"
        )

        results[name] = {
            "pop_mean": pop_mean.cpu().numpy(),
            "pop_std": pop_std.cpu().numpy(),
            "mean_of_means": mean_of_means,
            "mean_corr": mean_corr,
            "std_corr": std_corr,
            "position_info": position_info,
        }

    # Linear probe on population mean to predict position
    print("\nLinear probe on final_ln population mean:")

    with torch.no_grad():
        pop_mean_final = torch.tensor(
            results["final_ln"]["pop_mean"], device=device, dtype=torch.float32
        )
        y = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Least squares: w = (X^T X)^-1 X^T y
        X = pop_mean_final  # [T, n_embd]
        w_probe = torch.linalg.lstsq(X, y).solution
        y_pred = X @ w_probe

        probe_corr, _ = pearsonr(y_pred.cpu().numpy(), y.cpu().numpy())
        probe_mae = torch.abs(y_pred - y).mean().item()

    print(f"  Probe correlation: r = {probe_corr:+.4f}")
    print(f"  Probe MAE: {probe_mae:.2f}")

    results["probe"] = {
        "correlation": probe_corr,
        "mae": probe_mae,
        "weights": w_probe.cpu().numpy(),
    }

    return results


def plot_results(
    attn_results, decoding_results, pop_results, save_dir, model_name="NoPE"
):
    """Generate visualization plots for all analyses."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_name} Model Analysis", fontsize=14, fontweight="bold")

    # Plot 1: Attention uniformity by head
    ax = axes[0, 0]
    correlations = attn_results["correlations"]
    colors = [
        "green" if c > 0.9 else ("orange" if c > 0.5 else "red") for c in correlations
    ]
    bars = ax.bar(range(len(correlations)), correlations, color=colors)
    ax.axhline(
        y=0.9, color="green", linestyle="--", alpha=0.5, label="Highly uniform (0.9)"
    )
    ax.axhline(
        y=0.5,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="Partially uniform (0.5)",
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Correlation with 1/(i+1)")
    ax.set_title("H1: Attention Uniformity by Head")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.2, 1.1)

    # Plot 2: Decoding vector projection vs position
    ax = axes[0, 1]
    proj = decoding_results["projections"]
    positions = np.arange(len(proj))
    ax.scatter(positions, proj, alpha=0.5, s=10)
    ax.plot(
        positions,
        np.polyval(np.polyfit(positions, proj, 1), positions),
        "r-",
        linewidth=2,
        label=f"r={decoding_results['pearson_corr']:.3f}",
    )
    ax.set_xlabel("True Position")
    ax.set_ylabel("Projection onto Decoding Vector")
    ax.set_title("H2: Decoding Vector Analysis")
    ax.legend()

    # Plot 3: Population mean correlation by stage
    ax = axes[1, 0]
    stages = [s for s in pop_results.keys() if s != "probe"]
    mean_corrs = [pop_results[s]["mean_corr"] for s in stages]
    colors = ["green" if abs(c) > 0.5 else "gray" for c in mean_corrs]
    ax.bar(range(len(stages)), mean_corrs, color=colors)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha="right")
    ax.set_ylabel("Correlation with Position")
    ax.set_title("H3: Population Mean Correlation by Stage")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Plot 4: Attention variance decay for best head
    ax = axes[1, 1]
    best_head = attn_results["best_head"]
    actual_var = attn_results["att_var_mean"][best_head]
    expected_var = attn_results["expected_var"]
    positions = np.arange(len(actual_var))
    ax.plot(positions, actual_var, "b-", label="Actual variance", alpha=0.7)
    ax.plot(positions, expected_var, "r--", label="Expected 1/(i+1)", alpha=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("Attention Variance")
    ax.set_title(f"Attention Variance Decay (Head {best_head})")
    ax.legend()
    ax.set_xlim(0, len(actual_var))

    plt.tight_layout()
    save_path = os.path.join(save_dir, "analysis_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlots saved to {save_path}")


def print_summary(attn_results, decoding_results, pop_results, norm_type):
    """Print a summary of all hypothesis tests."""
    print("\n" + "=" * 60)
    print(f"SUMMARY ({norm_type.upper()})")
    print("=" * 60)

    # H1: Uniform attention
    best_corr = attn_results["correlations"][attn_results["best_head"]]
    n_uniform = attn_results["n_uniform_heads"]
    h1_status = "SUPPORTED" if n_uniform > 0 else "NOT SUPPORTED"
    print(f"\nH1 (Uniform Attention):")
    print(f"    Best head: {attn_results['best_head']} with r={best_corr:.4f}")
    print(f"    Uniform heads (r>0.9): {n_uniform}/12")
    print(f"    Status: {h1_status}")

    # H2: Decoding vector
    h2_corr = decoding_results["pearson_corr"]
    h2_status = (
        "SUPPORTED"
        if h2_corr > 0.9
        else ("PARTIAL" if h2_corr > 0.5 else "NOT SUPPORTED")
    )
    print(f"\nH2 (Decoding Vector):")
    print(f"    Pearson correlation: r={h2_corr:.4f}")
    print(f"    Counting correlation: r={decoding_results['counting_corr']:.4f}")
    print(f"    Status: {h2_status}")

    # H3: Population statistics
    h3_corr = pop_results["probe"]["correlation"]
    h3_status = (
        "SUPPORTED"
        if h3_corr > 0.9
        else ("PARTIAL" if h3_corr > 0.5 else "NOT SUPPORTED")
    )
    print(f"\nH3 (Population Statistics):")
    print(f"    Probe correlation: r={h3_corr:.4f}")
    print(f"    Probe MAE: {pop_results['probe']['mae']:.2f}")
    print(
        f"    Post-attn mean correlation: r={pop_results['post_attn']['mean_corr']:.4f}"
    )
    print(f"    Status: {h3_status}")

    print("\n" + "=" * 60)


def analyze_checkpoint(checkpoint_path, save_dir, n_samples=1000, device="cuda"):
    """Run full analysis on a single checkpoint."""
    # Load model
    model, checkpoint = load_model(checkpoint_path, device)

    # Print model info
    norm_type = model.config.norm_type
    iter_num = checkpoint.get("iter_num", "unknown")
    val_loss = checkpoint.get("best_val_loss", float("nan"))

    print(f"\nModel Configuration:")
    print(f"  Normalization: {norm_type}")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  Heads: {model.config.n_head}")
    print(f"  Embedding dim: {model.config.n_embd}")
    print(f"  Block size: {model.config.block_size}")
    print(f"  Positional embedding: {model.config.use_positional_embedding}")
    print(f"  Training iteration: {iter_num}")
    print(f"  Best val loss: {val_loss:.4f}")

    seq_len = min(256, model.config.block_size)

    # Run analyses
    attn_results = analyze_attention_uniformity(model, n_samples, seq_len, device)
    decoding_results = analyze_decoding_vector(model, n_samples, seq_len, device)
    pop_results = analyze_population_statistics(model, n_samples * 2, seq_len, device)

    # Print summary
    print_summary(attn_results, decoding_results, pop_results, norm_type)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "checkpoint_path": checkpoint_path,
        "iter_num": iter_num,
        "val_loss": val_loss,
        "norm_type": norm_type,
        "attention": attn_results,
        "decoding": decoding_results,
        "population": pop_results,
    }

    results_path = os.path.join(save_dir, "analysis_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    model_name = f"NoPE-{norm_type.upper()}"
    plot_results(attn_results, decoding_results, pop_results, save_dir, model_name)

    return results


def analyze_emergence(checkpoint_dir, save_dir, n_samples=500, device="cuda"):
    """
    Analyze multiple checkpoints to study emergence of positional encoding over training.
    """
    print(f"\nAnalyzing emergence from checkpoints in {checkpoint_dir}")

    # Find all checkpoints
    checkpoint_files = sorted(Path(checkpoint_dir).glob("ckpt_*.pt"))
    if not checkpoint_files:
        print("No checkpoint files found (ckpt_*.pt pattern)")
        return None

    print(f"Found {len(checkpoint_files)} checkpoints")

    emergence_data = {
        "iterations": [],
        "val_losses": [],
        "best_uniformity": [],
        "n_uniform_heads": [],
        "decoding_corr": [],
        "population_corr": [],
    }

    for ckpt_path in checkpoint_files:
        print(f"\n{'=' * 40}")
        print(f"Analyzing {ckpt_path.name}")
        print("=" * 40)

        model, checkpoint = load_model(str(ckpt_path), device)
        iter_num = checkpoint.get("iter_num", 0)
        val_loss = checkpoint.get("best_val_loss", float("nan"))

        seq_len = min(256, model.config.block_size)

        # Quick analysis with fewer samples
        attn_results = analyze_attention_uniformity(model, n_samples, seq_len, device)
        decoding_results = analyze_decoding_vector(
            model, n_samples // 2, seq_len, device
        )
        pop_results = analyze_population_statistics(model, n_samples, seq_len, device)

        emergence_data["iterations"].append(iter_num)
        emergence_data["val_losses"].append(val_loss)
        emergence_data["best_uniformity"].append(max(attn_results["correlations"]))
        emergence_data["n_uniform_heads"].append(attn_results["n_uniform_heads"])
        emergence_data["decoding_corr"].append(decoding_results["pearson_corr"])
        emergence_data["population_corr"].append(pop_results["probe"]["correlation"])

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Plot emergence over training
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Positional Encoding Emergence Over Training", fontsize=14, fontweight="bold"
    )

    iters = emergence_data["iterations"]

    ax = axes[0, 0]
    ax.plot(iters, emergence_data["val_losses"], "b-o")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Training Progress")

    ax = axes[0, 1]
    ax.plot(iters, emergence_data["best_uniformity"], "g-o", label="Best head")
    ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="Uniform threshold")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Uniformity Correlation")
    ax.set_title("H1: Attention Uniformity")
    ax.legend()
    ax.set_ylim(-0.2, 1.1)

    ax = axes[1, 0]
    ax.plot(iters, emergence_data["decoding_corr"], "m-o")
    ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="Strong decoding")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Decoding Correlation")
    ax.set_title("H2: Decoding Vector Effectiveness")
    ax.legend()
    ax.set_ylim(-0.2, 1.1)

    ax = axes[1, 1]
    ax.plot(iters, emergence_data["population_corr"], "c-o")
    ax.axhline(
        y=0.9, color="r", linestyle="--", alpha=0.5, label="Strong population signal"
    )
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Population Mean Correlation")
    ax.set_title("H3: Population Statistics")
    ax.legend()
    ax.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    emergence_plot_path = os.path.join(save_dir, "emergence_over_training.png")
    plt.savefig(emergence_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nEmergence plot saved to {emergence_plot_path}")

    # Save emergence data
    emergence_path = os.path.join(save_dir, "emergence_data.pkl")
    with open(emergence_path, "wb") as f:
        pickle.dump(emergence_data, f)
    print(f"Emergence data saved to {emergence_path}")

    return emergence_data


def main():
    parser = argparse.ArgumentParser(description="Analyze trained NoPE models")
    parser.add_argument("--checkpoint", type=str, help="Path to single checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory with multiple checkpoints for emergence analysis",
    )
    parser.add_argument(
        "--save_dir", type=str, default="analysis_results", help="Output directory"
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples for analysis"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.checkpoint:
        analyze_checkpoint(args.checkpoint, args.save_dir, args.n_samples, args.device)
    elif args.checkpoint_dir:
        analyze_emergence(
            args.checkpoint_dir, args.save_dir, args.n_samples, args.device
        )
    else:
        print("Please provide --checkpoint or --checkpoint_dir")
        parser.print_help()


if __name__ == "__main__":
    main()
