"""
R2 Mechanism Analysis: How Does Attn2-Only Extract Position?

The R2 regime has:
- Block 1: FROZEN (random weights)
- Block 2 Attention: TRAINABLE
- Block 2 MLP: FROZEN
- Linear head: TRAINABLE

Input to Block 2 (after Block 1):
  r1_i = e_i + attn1_out_i

where attn1_out_i ≈ (1/(i+1)) * sum_{j=0}^{i} f(e_j)  (causal averaging with random W_V, W_O)

Question: What do the learned W_Q, W_K, W_V in Block 2 do to extract position?
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path

# Add parent to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig


def load_model(regime, device="cuda"):
    """Load a trained model for a specific regime."""
    checkpoint_path = f"out-2layer-mechanism/{regime}/best_ckpt.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]

    # Extract only the fields needed for the model config
    model_config_fields = [
        "block_size",
        "vocab_size",
        "n_embd",
        "n_head",
        "dropout",
        "norm_type",
    ]
    filtered_config = {
        k: config_dict[k] for k in model_config_fields if k in config_dict
    }

    # Add defaults
    filtered_config["bias"] = True
    filtered_config["use_regression"] = True

    config = TwoLayerMechanismConfig(**filtered_config)

    model = TwoLayerMechanismModel(config)

    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, config


def analyze_r2_mechanism(device="cuda"):
    """Deep analysis of how R2 extracts position."""

    print("=" * 80)
    print("R2 MECHANISM ANALYSIS: How Does Attn2-Only Extract Position?")
    print("=" * 80)

    # Load R2 model
    model, config = load_model("R2", device)

    # Load some data
    data_path = "data/openwebtext/train.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Get a batch
    batch_size = 32
    block_size = config.block_size
    D = config.n_embd
    n_head = config.n_head
    head_dim = D // n_head
    T = block_size

    ix = torch.randint(len(data) - block_size, (batch_size,))
    tokens = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    tokens = tokens.to(device)

    print(f"\nInput shape: {tokens.shape}")
    print(f"D={D}, n_head={n_head}, head_dim={head_dim}, T={T}")

    # =========================================================================
    # PART 1: Trace the forward pass step by step
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: FORWARD PASS TRACE")
    print("=" * 80)

    with torch.no_grad():
        # Step 1: Embedding
        e = model.wte(tokens)  # [B, T, D]
        print(f"\n1. Token embeddings e: {e.shape}")
        print(f"   ||e|| mean: {e.norm(dim=-1).mean():.4f}")

        # Step 2: Block 1 (FROZEN)
        # Block 1 pre-LN
        ln1_out = model.block1.ln_1(e)
        print(f"\n2. After Block1 LN1: {ln1_out.shape}")
        print(f"   ||ln1_out|| mean: {ln1_out.norm(dim=-1).mean():.4f}")

        # Block 1 Attention (frozen random)
        attn1 = model.block1.attn
        B = batch_size

        # Get Q, K, V for Block 1
        qkv1 = attn1.c_attn(ln1_out)  # [B, T, 3*D]
        q1, k1, v1 = qkv1.split(D, dim=2)

        # Reshape for attention
        q1 = q1.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, H, T, d]
        k1 = k1.view(B, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B, T, n_head, head_dim).transpose(1, 2)

        # Attention scores
        scores1 = (q1 @ k1.transpose(-2, -1)) / np.sqrt(head_dim)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores1 = scores1.masked_fill(causal_mask, float("-inf"))

        attn_weights1 = F.softmax(scores1, dim=-1)

        print(f"\n3. Block 1 Attention weights: {attn_weights1.shape}")

        # Check if Block 1 attention is uniform-ish
        print(f"\n   Uniformity check (deviation from 1/(i+1)):")
        for pos in [0, 16, 32, 64, 127]:
            if pos < T:
                weights_at_pos = attn_weights1[0, :, pos, : pos + 1]  # [H, pos+1]
                uniform_weight = 1.0 / (pos + 1)
                deviation = (weights_at_pos - uniform_weight).abs().mean().item()
                print(
                    f"   Position {pos:3d}: expected {uniform_weight:.4f}, deviation {deviation:.4f}"
                )

        # Block 1 attention output
        attn_out1 = (attn_weights1 @ v1).transpose(1, 2).reshape(B, T, D)
        attn_out1 = attn1.c_proj(attn_out1)

        print(f"\n4. Block 1 attention output: {attn_out1.shape}")
        print(f"   ||attn_out1|| mean: {attn_out1.norm(dim=-1).mean():.4f}")

        # Residual after Block 1 attention
        r1_attn = e + attn_out1

        # Block 1 LN2 and MLP
        ln2_out_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_out_b1)

        # Full Block 1 output (residual)
        r1 = r1_attn + mlp_out1  # This is the input to Block 2
        print(f"\n5. Block 1 full output r1 = e + attn_out1 + mlp_out1: {r1.shape}")
        print(f"   ||r1|| mean: {r1.norm(dim=-1).mean():.4f}")

        # Check position correlation in r1
        positions = torch.arange(T, device=device).float()
        r1_norms = r1[0].norm(dim=-1)  # [T]
        r1_pos_corr = torch.corrcoef(torch.stack([r1_norms, positions]))[0, 1].item()
        print(f"   Correlation(||r1||, position): {r1_pos_corr:.4f}")

        # =========================================================================
        # PART 2: Block 2 Attention Analysis (TRAINED)
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 2: BLOCK 2 ATTENTION ANALYSIS (TRAINED)")
        print("=" * 80)

        # Block 2 LN1
        ln1_out_b2 = model.block2.ln_1(r1)
        print(f"\n6. After Block2 LN1: {ln1_out_b2.shape}")
        print(f"   ||ln1_out_b2|| mean: {ln1_out_b2.norm(dim=-1).mean():.4f}")

        # Check position signal in ln1_out_b2
        ln2_norms = ln1_out_b2[0].norm(dim=-1)
        ln2_pos_corr = torch.corrcoef(torch.stack([ln2_norms, positions]))[0, 1].item()
        print(f"   Correlation(||ln1_out_b2||, position): {ln2_pos_corr:.4f}")

        # Block 2 Attention (TRAINED)
        attn2 = model.block2.attn

        # Get the learned W_Q, W_K, W_V, W_O
        W_qkv = attn2.c_attn.weight  # [3*D, D]
        b_qkv = attn2.c_attn.bias  # [3*D]
        W_o = attn2.c_proj.weight  # [D, D]
        b_o = attn2.c_proj.bias  # [D]

        W_q = W_qkv[:D, :]  # [D, D]
        W_k = W_qkv[D : 2 * D, :]  # [D, D]
        W_v = W_qkv[2 * D :, :]  # [D, D]

        b_q = b_qkv[:D]
        b_k = b_qkv[D : 2 * D]
        b_v = b_qkv[2 * D :]

        print(f"\n7. Learned Block 2 weights:")
        print(f"   W_Q: {W_q.shape}, ||W_Q|| = {W_q.norm():.4f}")
        print(f"   W_K: {W_k.shape}, ||W_K|| = {W_k.norm():.4f}")
        print(f"   W_V: {W_v.shape}, ||W_V|| = {W_v.norm():.4f}")
        print(f"   W_O: {W_o.shape}, ||W_O|| = {W_o.norm():.4f}")

        # Compute Q, K, V for Block 2
        qkv2 = attn2.c_attn(ln1_out_b2)
        q2, k2, v2 = qkv2.split(D, dim=2)

        # Reshape
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        print(f"\n8. Block 2 Q, K, V:")
        print(
            f"   ||Q2|| per position: min={q2[0].norm(dim=-1).min():.2f}, max={q2[0].norm(dim=-1).max():.2f}"
        )
        print(
            f"   ||K2|| per position: min={k2[0].norm(dim=-1).min():.2f}, max={k2[0].norm(dim=-1).max():.2f}"
        )
        print(
            f"   ||V2|| per position: min={v2[0].norm(dim=-1).min():.2f}, max={v2[0].norm(dim=-1).max():.2f}"
        )

        # Check position correlation in V2
        v2_norms = v2[0].norm(dim=-1).mean(dim=0)  # [T] average over heads
        v2_pos_corr = torch.corrcoef(torch.stack([v2_norms, positions]))[0, 1].item()
        print(f"   Correlation(||V2||, position): {v2_pos_corr:.4f}")

        # =========================================================================
        # PART 3: What Makes Position Decodable from V2?
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 3: HOW THE INPUT TO BLOCK 2 ENCODES POSITION")
        print("=" * 80)

        # Position correlation with each dimension of the input to Block 2
        dim_pos_corrs = []
        for d in range(D):
            corr = torch.corrcoef(torch.stack([ln1_out_b2[0, :, d], positions]))[
                0, 1
            ].item()
            if not np.isnan(corr):
                dim_pos_corrs.append((d, corr))

        dim_pos_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        print(
            f"\n9. Top dimensions in Block2 input (after LN) correlated with position:"
        )
        for d, corr in dim_pos_corrs[:10]:
            print(f"   Dim {d:3d}: correlation = {corr:+.4f}")

        # How many dimensions have |corr| > 0.5?
        n_strong = sum(1 for _, c in dim_pos_corrs if abs(c) > 0.5)
        n_moderate = sum(1 for _, c in dim_pos_corrs if abs(c) > 0.3)
        print(f"\n   Dimensions with |corr| > 0.5: {n_strong}")
        print(f"   Dimensions with |corr| > 0.3: {n_moderate}")

        # =========================================================================
        # PART 4: Attention Pattern Analysis
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 4: BLOCK 2 ATTENTION PATTERNS")
        print("=" * 80)

        # Attention scores
        scores2 = (q2 @ k2.transpose(-2, -1)) / np.sqrt(head_dim)
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))
        attn_weights2 = F.softmax(scores2, dim=-1)

        print(f"\n10. Block 2 attention patterns (at position 64):")
        pos = 64
        for h in range(n_head):
            weights = attn_weights2[0, h, pos, : pos + 1]
            max_weight = weights.max().item()
            argmax = weights.argmax().item()
            self_weight = weights[pos].item()
            first_weight = weights[0].item()

            # Classify pattern
            if self_weight > 0.3:
                pattern = "CURRENT-POS"
            elif first_weight > 0.5:
                pattern = "BOS"
            elif max_weight < 0.1:
                pattern = "UNIFORM-ISH"
            else:
                pattern = "OTHER"

            print(
                f"   Head {h:2d}: self={self_weight:.3f}, first={first_weight:.3f}, max={max_weight:.3f} at {argmax} [{pattern}]"
            )

        # =========================================================================
        # PART 5: The Key Insight - What Does Attn2 Output?
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 5: WHAT DOES ATTN2 OUTPUT?")
        print("=" * 80)

        # Attention output
        attn_out2 = (attn_weights2 @ v2).transpose(1, 2).reshape(B, T, D)
        attn_out2_proj = attn2.c_proj(attn_out2)

        print(f"\n11. Attn2 output (before projection): {attn_out2.shape}")
        print(f"    ||attn_out2|| mean: {attn_out2.norm(dim=-1).mean():.4f}")

        # Position correlation
        attn_out2_norms = attn_out2_proj[0].norm(dim=-1)
        attn_out2_pos_corr = torch.corrcoef(torch.stack([attn_out2_norms, positions]))[
            0, 1
        ].item()
        print(f"    Correlation(||attn_out2||, position): {attn_out2_pos_corr:.4f}")

        # Per-dimension correlation
        attn2_dim_corrs = []
        for d in range(D):
            corr = torch.corrcoef(torch.stack([attn_out2_proj[0, :, d], positions]))[
                0, 1
            ].item()
            if not np.isnan(corr):
                attn2_dim_corrs.append((d, corr))

        attn2_dim_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n12. Top dimensions in attn_out2 correlated with position:")
        for d, corr in attn2_dim_corrs[:10]:
            print(f"    Dim {d:3d}: correlation = {corr:+.4f}")

        # =========================================================================
        # PART 6: The Final Residual and Linear Head
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 6: FINAL RESIDUAL AND LINEAR HEAD")
        print("=" * 80)

        # After Block 2 attention
        r2_attn = r1 + attn_out2_proj  # Residual after attn

        # Block 2 LN2 and MLP (frozen in R2)
        ln2_out_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_out_b2)
        r2 = r2_attn + mlp_out2

        # Final LN
        final_ln = model.ln_f(r2)

        print(f"\n13. Final representations:")
        print(f"    r2 = r1 + attn_out2 + mlp_out2: {r2.shape}")
        print(f"    final_ln: {final_ln.shape}")

        # Position correlation in final
        final_norms = final_ln[0].norm(dim=-1)
        final_pos_corr = torch.corrcoef(torch.stack([final_norms, positions]))[
            0, 1
        ].item()
        print(f"    Correlation(||final||, position): {final_pos_corr:.4f}")

        # Linear head
        linear_weight = model.pos_head.weight  # [1, D]
        linear_bias = model.pos_head.bias  # [1]

        print(f"\n14. Linear head:")
        print(f"    Weight shape: {linear_weight.shape}")
        print(f"    ||weight||: {linear_weight.norm():.4f}")
        print(f"    Bias: {linear_bias.item():.4f}")

        # Prediction
        pred = (final_ln @ linear_weight.T).squeeze(-1) + linear_bias

        # Check prediction quality
        pred_pos_corr = torch.corrcoef(torch.stack([pred[0], positions]))[0, 1].item()
        print(f"    Correlation(prediction, position): {pred_pos_corr:.4f}")

        # Sample predictions
        print(f"\n15. Sample predictions vs actual positions:")
        for pos in [0, 16, 32, 64, 96, 127]:
            print(f"    Position {pos:3d}: predicted {pred[0, pos].item():.2f}")

        # =========================================================================
        # PART 7: THE KEY MECHANISM - What does W_linear select?
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 7: WHAT DOES THE LINEAR HEAD SELECT?")
        print("=" * 80)

        w = linear_weight.squeeze(0)  # [D]

        # Correlation between w and dimension-position correlations
        dim_corr_vector = torch.zeros(D, device=device)
        for d, corr in dim_pos_corrs:
            dim_corr_vector[d] = corr

        w_corr_alignment = F.cosine_similarity(
            w.unsqueeze(0), dim_corr_vector.unsqueeze(0)
        ).item()
        print(f"\n16. Cosine(w, dim_pos_corr_vector): {w_corr_alignment:.4f}")

        # Top dimensions by |w|
        w_abs = w.abs()
        top_w_dims = w_abs.argsort(descending=True)[:20]

        print(f"\n17. Top dimensions by |w| and their position correlation:")
        for d in top_w_dims[:15]:
            d = d.item()
            w_val = w[d].item()
            pos_corr = dim_corr_vector[d].item()
            print(
                f"    Dim {d:3d}: w = {w_val:+.4f}, pos_corr = {pos_corr:+.4f}, product = {w_val * pos_corr:+.4f}"
            )

    # =========================================================================
    # PART 8: VERIFICATION - Position Signal at Each Layer (OUTSIDE no_grad)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 8: POSITION SIGNAL AT EACH LAYER (Linear Probe R²)")
    print("=" * 80)

    from torch import nn, optim

    # Get more data for probe
    n_samples = 500
    ix = torch.randint(len(data) - block_size, (n_samples,))
    all_tokens = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    all_tokens = all_tokens.to(device)

    with torch.no_grad():
        # Forward pass to get all intermediate representations
        all_e = model.wte(all_tokens)

        # Block 1
        all_ln1_b1 = model.block1.ln_1(all_e)
        qkv = model.block1.attn.c_attn(all_ln1_b1)
        q, k, v = qkv.split(D, dim=2)
        q = q.view(n_samples, T, n_head, head_dim).transpose(1, 2)
        k = k.view(n_samples, T, n_head, head_dim).transpose(1, 2)
        v = v.view(n_samples, T, n_head, head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(head_dim)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn_out = (weights @ v).transpose(1, 2).reshape(n_samples, T, D)
        attn_out = model.block1.attn.c_proj(attn_out)
        all_r1_attn = all_e + attn_out

        all_ln2_b1 = model.block1.ln_2(all_r1_attn)
        mlp_out1 = model.block1.mlp(all_ln2_b1)
        all_r1 = all_r1_attn + mlp_out1

        # Block 2
        all_ln1_b2 = model.block2.ln_1(all_r1)
        qkv2 = model.block2.attn.c_attn(all_ln1_b2)
        q2, k2, v2 = qkv2.split(D, dim=2)
        q2 = q2.view(n_samples, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(n_samples, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(n_samples, T, n_head, head_dim).transpose(1, 2)
        scores2 = (q2 @ k2.transpose(-2, -1)) / np.sqrt(head_dim)
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))
        weights2 = F.softmax(scores2, dim=-1)
        attn_out2 = (weights2 @ v2).transpose(1, 2).reshape(n_samples, T, D)
        attn_out2 = model.block2.attn.c_proj(attn_out2)
        all_r2_attn = all_r1 + attn_out2

        all_ln2_b2 = model.block2.ln_2(all_r2_attn)
        mlp_out2 = model.block2.mlp(all_ln2_b2)
        all_r2 = all_r2_attn + mlp_out2

        all_final = model.ln_f(all_r2)

    # Targets
    y = torch.arange(T, device=device).float().repeat(n_samples)  # [n_samples * T]

    def train_probe(X, y, n_epochs=200, use_norm_only=False):
        """Train a linear probe and return R²."""
        # Clone to new tensors (avoid inference_mode issues)
        if not use_norm_only:
            X_flat = X.clone().detach().reshape(-1, X.shape[-1])
        else:
            X_flat = X.clone().detach().reshape(-1, 1)
        y_local = y.clone().detach()

        probe = nn.Linear(X_flat.shape[-1], 1).to(device)
        optimizer = optim.Adam(probe.parameters(), lr=0.01)

        for epoch in range(n_epochs):
            pred = probe(X_flat).squeeze(-1)
            loss = F.mse_loss(pred, y_local)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_pred = probe(X_flat).squeeze(-1)
            r2_score = 1 - F.mse_loss(final_pred, y_local) / y_local.var()
            mae = (final_pred - y_local).abs().mean()

        return r2_score.item(), mae.item()

    layers = [
        ("Embedding e", all_e),
        ("Block1 attn input (LN)", all_ln1_b1),
        ("Block1 attn residual", all_r1_attn),
        ("Block1 output r1", all_r1),
        ("Block2 attn input (LN)", all_ln1_b2),
        ("Block2 attn residual", all_r2_attn),
        ("Block2 output r2", all_r2),
        ("Final LN output", all_final),
    ]

    print(f"\n18. Linear probe R² at each layer:")
    print(f"    {'Layer':<30} {'Full R²':>10} {'Norm R²':>10} {'MAE':>10}")
    print(f"    {'-' * 60}")

    probe_results = {}
    for name, activations in layers:
        r2_full, mae_full = train_probe(activations, y)
        norms = activations.clone().detach().norm(dim=-1, keepdim=True)
        r2_norm, mae_norm = train_probe(norms, y, use_norm_only=True)

        print(f"    {name:<30} {r2_full:>10.4f} {r2_norm:>10.4f} {mae_full:>10.2f}")
        probe_results[name] = {
            "full_r2": r2_full,
            "norm_r2": r2_norm,
            "mae": mae_full,
        }

    # =========================================================================
    # PART 9: THE EXACT FORMULA
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 9: THE EXACT POSITION EXTRACTION FORMULA")
    print("=" * 80)

    print("""
THE R2 MECHANISM EXPLAINED:
===========================

1. INPUT TO BLOCK 1:
   e_i = Embedding(token_i)                    [D-dimensional]

2. BLOCK 1 (FROZEN, RANDOM WEIGHTS):
   
   LN1: x_i = LayerNorm(e_i)                   [Normalized embedding]
   
   Attn1: For each head h, with random W_Q^h, W_K^h, W_V^h:
          q_i^h = W_Q^h @ x_i                  [Query at position i]
          k_j^h = W_K^h @ x_j                  [Key at position j]
          v_j^h = W_V^h @ x_j                  [Value at position j]
          
          α_{ij}^h = softmax(q_i^h · k_j^h / √d)  [Attention weight]
          
          Due to causal mask: α_{ij} = 0 for j > i
          With random weights: α_{ij}^h ≈ 1/(i+1) for j ≤ i (roughly uniform)
          
          attn_out_i^h = Σ_{j=0}^i α_{ij}^h · v_j^h
                       ≈ (1/(i+1)) · Σ_{j=0}^i v_j^h  [Average of values]
   
   Residual: r1_attn_i = e_i + W_O @ concat(attn_out_i^1, ..., attn_out_i^H)
   
   MLP1: r1_i = r1_attn_i + MLP(LN(r1_attn_i))
   
   KEY INSIGHT: r1_i contains position signal because:
   - attn_out_i averages over i+1 positions
   - By Central Limit Theorem: Var(average of i+1 vectors) ~ 1/(i+1)
   - Position 0 has high variance (no averaging)
   - Position 127 has low variance (averaged over 128 vectors)

3. BLOCK 2 (ATTENTION TRAINED, MLP FROZEN):
   
   LN1: x2_i = LayerNorm(r1_i)                 [Normalized Block 1 output]
   
   The position signal (variance ~ 1/(i+1)) is in r1_i.
   After LayerNorm, this becomes a DIRECTION signal.
   
   Attn2: With LEARNED W_Q^h, W_K^h, W_V^h:
          The model learns to READ the position signal.
          
          Two strategies observed:
          (a) CURRENT-POSITION heads: α_{ii}^h is high (attend to self)
              → Reads the local representation which encodes position
          (b) BOS heads: α_{i0}^h is high (attend to position 0)
              → Reads the reference (un-averaged) representation
   
   The OUTPUT of Attn2:
          attn_out2_i = W_O @ concat(Σ_j α_{ij}^h · W_V^h @ x2_j)
          
          For current-position heads: attn_out2_i ≈ W_O @ W_V @ x2_i
          This COPIES the position-encoding representation.

4. LINEAR HEAD:
   
   final_i = LayerNorm(r1_i + attn_out2_i + mlp_out2_i)
   
   pred_i = w · final_i + b
   
   The linear head w is learned to weight dimensions that correlate with position.
   Since Attn2 copies/enhances the position signal, this is easily extracted.

SUMMARY:
========
- Block 1 (random) creates position signal through causal averaging
- Block 2 attention learns to READ this signal (current-position heads)
- The linear head extracts position from the enhanced representation

The key is: position is ALREADY in the Block 1 output!
Attn2 just learns to READ and AMPLIFY it.
    """)

    # Save results
    results = {
        "probe_results": probe_results,
        "position_correlations": {
            "r1_norm_pos_corr": r1_pos_corr,
            "block2_input_norm_pos_corr": ln2_pos_corr,
            "v2_norm_pos_corr": v2_pos_corr,
            "attn_out2_norm_pos_corr": attn_out2_pos_corr,
            "final_norm_pos_corr": final_pos_corr,
        },
        "linear_head": {
            "weight_norm": linear_weight.norm().item(),
            "bias": linear_bias.item(),
            "alignment_with_pos_corr_dims": w_corr_alignment,
        },
        "attention_patterns": {
            "n_strong_pos_corr_dims": n_strong,
            "n_moderate_pos_corr_dims": n_moderate,
        },
    }

    os.makedirs("out-2layer-mechanism/r2_analysis", exist_ok=True)
    with open("out-2layer-mechanism/r2_analysis/r2_mechanism_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\nResults saved to out-2layer-mechanism/r2_analysis/r2_mechanism_analysis.json"
    )


if __name__ == "__main__":
    analyze_r2_mechanism()
