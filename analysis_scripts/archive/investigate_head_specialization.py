#!/usr/bin/env python3
"""
Investigate why Head 7 specialized to position 0 despite Head 4 having higher gradient.

Key questions:
1. What's different about Head 7 at initialization vs Head 4?
2. Does key variance track specialization?
3. Is high gradient a CAUSE or SYMPTOM of specialization attempts?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from pathlib import Path

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model import GPTConfig, GPT

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results" / "head_specialization_investigation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_fresh_model(seed=42):
    """Create a freshly initialized model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = GPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=12,
        n_embd=192,
        dropout=0.0,
        bias=True,
        use_positional_embedding=False,
    )
    model = GPT(config)
    return model.to(device)


def analyze_initialization(model, head_idx_list=[4, 7, 9, 10]):
    """Analyze W_Q @ W_K.T structure at initialization for specific heads."""
    results = {}

    for block_idx in [0, 1]:
        block = model.transformer.h[block_idx]

        # Extract Q, K, V weights
        # c_attn projects to Q, K, V concatenated
        W = block.attn.c_attn.weight  # [3*n_embd, n_embd]
        n_embd = model.config.n_embd
        n_head = model.config.n_head
        head_dim = n_embd // n_head

        W_Q = W[:n_embd, :]       # [n_embd, n_embd]
        W_K = W[n_embd:2*n_embd, :]  # [n_embd, n_embd]

        # Reshape to per-head
        W_Q_heads = W_Q.view(n_head, head_dim, n_embd)  # [n_head, head_dim, n_embd]
        W_K_heads = W_K.view(n_head, head_dim, n_embd)  # [n_head, head_dim, n_embd]

        results[f"block{block_idx}"] = {}

        for h in range(n_head):
            W_Q_h = W_Q_heads[h]  # [head_dim, n_embd]
            W_K_h = W_K_heads[h]  # [head_dim, n_embd]

            # W_Q @ W_K.T determines attention pattern structure
            # This is applied to (x_i, x_j) pairs: score = x_i @ W_Q_h.T @ W_K_h @ x_j
            QK = W_Q_h @ W_K_h.T  # [head_dim, head_dim]

            # Key metrics:
            # 1. Frobenius norm - overall magnitude
            # 2. Eigenvalue spectrum - structure
            # 3. Trace (sum of diagonal) - self-attention bias

            frob_norm = torch.norm(QK, 'fro').item()
            trace = torch.trace(QK).item()

            # Singular values
            svs = torch.linalg.svdvals(QK).cpu().numpy()
            sv_ratio = svs[0] / (svs[-1] + 1e-8)  # condition number-like

            # Key variance at init (W_K applied to random embeddings)
            # This tells us how spread out the keys will be
            test_emb = torch.randn(1000, n_embd, device=device)
            keys = test_emb @ W_K_h.T  # [1000, head_dim]
            key_variance = keys.var().item()

            results[f"block{block_idx}"][f"head{h}"] = {
                "QK_frob_norm": frob_norm,
                "QK_trace": trace,
                "sv_ratio": sv_ratio,
                "key_variance_init": key_variance,
                "top_singular_value": svs[0],
            }

    return results


def train_and_track_key_variance(n_iters=5000, log_interval=100):
    """
    Train model while tracking key variance for all heads.
    Key variance tells us how much the keys spread out - low variance = more uniform attention.
    """
    print("Training and tracking key variance for all heads...")

    # Same setup as the wandb script for consistency
    torch.manual_seed(42)
    np.random.seed(42)

    model = create_fresh_model(seed=42)

    # Create synthetic data with BOS at position 80
    block_size = 128
    bos_position = 80

    def get_batch(batch_size=32):
        data = torch.randint(1, 256, (batch_size, block_size + 1), device=device)
        data[:, bos_position] = 0  # BOS token
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Tracking
    history = {
        "iter": [],
        "loss": [],
    }
    for block_idx in [0, 1]:
        for h in range(12):
            history[f"block{block_idx}_head{h}_key_var"] = []
            history[f"block{block_idx}_head{h}_attn_to_pos0"] = []
            history[f"block{block_idx}_head{h}_attn_to_bos"] = []

    for it in range(n_iters):
        x, y = get_batch()

        with torch.cuda.amp.autocast():
            logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % log_interval == 0:
            history["iter"].append(it)
            history["loss"].append(loss.item())

            # Compute key variance and attention for all heads
            model.eval()
            with torch.no_grad():
                # Get embeddings
                tok_emb = model.transformer.wte(x)  # [B, T, C]

                # Track through both blocks
                x_curr = tok_emb

                for block_idx in [0, 1]:
                    block = model.transformer.h[block_idx]

                    # LayerNorm
                    x_ln = block.ln_1(x_curr)

                    # Get Q, K, V
                    B, T, C = x_ln.shape
                    n_head = model.config.n_head
                    head_dim = C // n_head

                    qkv = block.attn.c_attn(x_ln)
                    q, k, v = qkv.split(C, dim=2)

                    # Reshape to heads
                    q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, nh, T, hd]
                    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
                    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

                    # Attention
                    att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                    # Causal mask
                    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                    att = att.masked_fill(mask, float('-inf'))
                    att = F.softmax(att, dim=-1)

                    for h in range(n_head):
                        # Key variance for this head
                        k_h = k[:, h, :, :]  # [B, T, hd]
                        key_var = k_h.var().item()
                        history[f"block{block_idx}_head{h}_key_var"].append(key_var)

                        # Attention to position 0
                        att_h = att[:, h, :, :]  # [B, T, T]
                        # Mean attention weight to position 0 (from all queries that can see it)
                        attn_to_pos0 = att_h[:, :, 0].mean().item()
                        history[f"block{block_idx}_head{h}_attn_to_pos0"].append(attn_to_pos0)

                        # Attention to BOS (position 80) - only from positions > 80
                        attn_to_bos = att_h[:, 81:, bos_position].mean().item()
                        history[f"block{block_idx}_head{h}_attn_to_bos"].append(attn_to_bos)

                    # Forward through block for next iteration
                    attn_out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
                    attn_out = block.attn.c_proj(attn_out)
                    x_curr = x_curr + attn_out
                    x_curr = x_curr + block.mlp(block.ln_2(x_curr))

            model.train()

            if it % (log_interval * 10) == 0:
                print(f"Iter {it}: loss={loss.item():.4f}")

    return history


def plot_key_variance_vs_specialization(history):
    """Plot key variance evolution vs attention specialization."""

    iters = history["iter"]

    # Figure 1: Key variance over training for all Block 2 heads
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for h in range(12):
        ax = axes[h]

        key_var = history[f"block1_head{h}_key_var"]
        attn_pos0 = history[f"block1_head{h}_attn_to_pos0"]
        attn_bos = history[f"block1_head{h}_attn_to_bos"]

        ax2 = ax.twinx()

        l1, = ax.plot(iters, key_var, 'b-', label='Key Variance')
        l2, = ax2.plot(iters, attn_pos0, 'r-', label='Attn to Pos 0')
        l3, = ax2.plot(iters, attn_bos, 'g--', label='Attn to BOS')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Key Variance', color='b')
        ax2.set_ylabel('Attention Weight', color='r')
        ax.set_title(f'Block 2 Head {h}')

        if h == 0:
            ax.legend([l1, l2, l3], ['Key Var', 'Attn Pos0', 'Attn BOS'], loc='upper right')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "key_variance_vs_attention_all_heads.png", dpi=150)
    plt.savefig(RESULTS_DIR / "key_variance_vs_attention_all_heads.pdf")
    plt.close()

    # Figure 2: Compare Head 4 vs Head 7 specifically
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, h in enumerate([4, 7]):
        # Key variance
        ax = axes[0, idx]
        ax.plot(iters, history[f"block1_head{h}_key_var"], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Key Variance')
        ax.set_title(f'Head {h} - Key Variance')
        ax.grid(True, alpha=0.3)

        # Attention
        ax = axes[1, idx]
        ax.plot(iters, history[f"block1_head{h}_attn_to_pos0"], 'r-', linewidth=2, label='Attn to Pos 0')
        ax.plot(iters, history[f"block1_head{h}_attn_to_bos"], 'g--', linewidth=2, label='Attn to BOS')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Head {h} - Attention Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "head4_vs_head7_comparison.png", dpi=150)
    plt.savefig(RESULTS_DIR / "head4_vs_head7_comparison.pdf")
    plt.close()

    # Figure 3: Correlation between initial key variance and final specialization
    final_attn_pos0 = []
    init_key_var = []

    for h in range(12):
        init_key_var.append(history[f"block1_head{h}_key_var"][0])
        final_attn_pos0.append(history[f"block1_head{h}_attn_to_pos0"][-1])

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['red' if h in [4, 7] else 'blue' for h in range(12)]
    ax.scatter(init_key_var, final_attn_pos0, c=colors, s=100)

    for h in range(12):
        ax.annotate(f'H{h}', (init_key_var[h], final_attn_pos0[h]),
                   fontsize=10, ha='center', va='bottom')

    ax.set_xlabel('Initial Key Variance')
    ax.set_ylabel('Final Attention to Position 0')
    ax.set_title('Does Initial Key Variance Predict Specialization?\n(Red = Heads 4 and 7)')
    ax.grid(True, alpha=0.3)

    # Correlation
    corr = np.corrcoef(init_key_var, final_attn_pos0)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "init_key_var_vs_final_specialization.png", dpi=150)
    plt.savefig(RESULTS_DIR / "init_key_var_vs_final_specialization.pdf")
    plt.close()

    return {
        "init_key_var": init_key_var,
        "final_attn_pos0": final_attn_pos0,
        "correlation": corr,
    }


def analyze_gradient_vs_specialization(history):
    """
    Analyze if high gradient is cause or symptom.
    Key insight: If gradient drives specialization, high gradient should PRECEDE high attention.
    If gradient is a symptom, high gradient might appear in heads that DON'T specialize.
    """

    # Find which head actually specialized (highest final attention to pos 0)
    final_attn = [history[f"block1_head{h}_attn_to_pos0"][-1] for h in range(12)]
    specialized_head = np.argmax(final_attn)

    print(f"\nSpecialized head (highest final attn to pos 0): Head {specialized_head}")
    print(f"Final attention values:")
    for h in range(12):
        print(f"  Head {h}: {final_attn[h]:.4f}")

    # Find when each head crossed 50% attention threshold
    crossing_times = {}
    for h in range(12):
        attn_series = history[f"block1_head{h}_attn_to_pos0"]
        crossed = False
        for i, a in enumerate(attn_series):
            if a > 0.5:
                crossing_times[h] = history["iter"][i]
                crossed = True
                break
        if not crossed:
            crossing_times[h] = None

    print(f"\nCrossing times (iter when attn > 50%):")
    for h, t in crossing_times.items():
        print(f"  Head {h}: {t}")

    return {
        "specialized_head": int(specialized_head),
        "final_attn_to_pos0": final_attn,
        "crossing_times": crossing_times,
    }


def main():
    print("=" * 60)
    print("Investigating Head Specialization Mechanism")
    print("=" * 60)

    # 1. Analyze initialization
    print("\n1. Analyzing initialization structure...")
    model = create_fresh_model(seed=42)
    init_analysis = analyze_initialization(model)

    print("\nBlock 2 initialization analysis:")
    print(f"{'Head':<6} {'QK Frob':<12} {'Key Var Init':<14} {'SV Ratio':<12}")
    print("-" * 50)
    for h in range(12):
        data = init_analysis["block1"][f"head{h}"]
        marker = " <--" if h in [4, 7] else ""
        print(f"{h:<6} {data['QK_frob_norm']:<12.4f} {data['key_variance_init']:<14.4f} {data['sv_ratio']:<12.2f}{marker}")

    # Save init analysis
    with open(RESULTS_DIR / "init_analysis.json", "w") as f:
        json.dump(init_analysis, f, indent=2)

    # 2. Train and track key variance
    print("\n2. Training while tracking key variance...")
    history = train_and_track_key_variance(n_iters=5000, log_interval=50)

    # Save history
    np.savez(RESULTS_DIR / "training_history.npz", **{k: np.array(v) for k, v in history.items()})

    # 3. Plot and analyze
    print("\n3. Plotting key variance vs specialization...")
    var_corr = plot_key_variance_vs_specialization(history)

    print(f"\nCorrelation between initial key variance and final pos-0 attention: {var_corr['correlation']:.3f}")

    # 4. Analyze gradient vs specialization timing
    print("\n4. Analyzing specialization dynamics...")
    spec_analysis = analyze_gradient_vs_specialization(history)

    # Save all results
    results = {
        "init_analysis": init_analysis,
        "var_correlation": var_corr,
        "specialization_analysis": spec_analysis,
    }

    with open(RESULTS_DIR / "full_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x if isinstance(x, (int, float, str, list, dict, type(None))) else str(x))

    print(f"\nResults saved to {RESULTS_DIR}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Specialized head: {spec_analysis['specialized_head']}")
    print(f"Init key var correlation with specialization: {var_corr['correlation']:.3f}")

    # Check if Head 4 vs Head 7 init key variance predicts outcome
    h4_init = init_analysis["block1"]["head4"]["key_variance_init"]
    h7_init = init_analysis["block1"]["head7"]["key_variance_init"]
    print(f"\nHead 4 init key variance: {h4_init:.4f}")
    print(f"Head 7 init key variance: {h7_init:.4f}")

    if h7_init < h4_init:
        print("→ Head 7 started with LOWER key variance (more uniform keys)")
        print("  This may explain why Head 7 specialized, not Head 4!")
    else:
        print("→ Head 4 started with lower key variance")
        print("  Need to look at other factors...")


if __name__ == "__main__":
    main()
