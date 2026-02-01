"""
Ablation study: Test counting method at different activation points.

This script tests the decoding vector counting method at various stages:
1. Raw embeddings (before any processing)
2. Post-LN1 (after first LayerNorm, before attention)
3. Post-attention (after attention + residual)
4. Post-LN2 (after second LayerNorm, before MLP)
5. Post-MLP (after MLP + residual)
6. Post-final-LN (after final LayerNorm)

For each stage, we compute value vectors and test the counting method.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    config.log_attention_stats = False
    model = GPT(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def counting_method_on_activations(activations, input_ids, model, device="cuda"):
    """
    Apply counting method on given activations.

    Args:
        activations: [B, T, n_embd] - activations at some point in the model
        input_ids: [B, T] - token IDs
        model: the model (for getting W_V)

    Returns:
        correlation, mae between decoded counts and true positions
    """
    B, T, n_embd = activations.shape
    block = model.transformer.h[0]

    # Get W_V
    W_qkv = block.attn.c_attn.weight
    W_V = W_qkv[2 * n_embd :, :]  # [n_embd, n_embd]

    # Compute value vectors for the activations
    # V = activations @ W_V.T  but we need per-token values

    n_test = min(100, B)
    decoded_positions = []

    with torch.no_grad():
        for b in range(n_test):
            for i in range(T):
                # Get activations for tokens 0..i
                act_prefix = activations[b, : i + 1, :]  # [i+1, n_embd]

                # Compute value vectors
                v_prefix = act_prefix @ W_V.T  # [i+1, n_embd]

                # Average (simulating uniform attention)
                v_sum = v_prefix.sum(dim=0)  # [n_embd]
                z_i = v_sum / (i + 1)

                # Count positive dot products
                count = (v_prefix @ z_i > 0).sum().item()
                decoded_positions.append((i, count))

    true_pos = np.array([p[0] for p in decoded_positions])
    decoded = np.array([p[1] for p in decoded_positions])
    corr, _ = pearsonr(true_pos, decoded)
    mae = np.abs(decoded - true_pos).mean()

    return corr, mae


def run_ablation(checkpoint_path, n_samples=500, seq_len=256, device="cuda"):
    """Run ablation study at different activation points."""

    print(f"Loading model from {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)

    norm_type = model.config.norm_type
    val_loss = checkpoint.get("best_val_loss", float("nan"))
    if torch.is_tensor(val_loss):
        val_loss = val_loss.cpu().item()
    perplexity = np.exp(val_loss)

    print(f"\nModel: {norm_type.upper()}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"\nRunning ablation with {n_samples} samples, seq_len={seq_len}")
    print("=" * 70)

    # Generate random sequences
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)

    results = {}

    with torch.no_grad():
        # Get activations at each stage
        tok_emb = model.transformer.wte(input_ids)  # [B, T, n_embd]
        x = model.transformer.drop(tok_emb)

        block = model.transformer.h[0]

        # Stage 1: Raw embeddings
        print("\n1. Raw Embeddings (before any processing):")
        corr, mae = counting_method_on_activations(tok_emb, input_ids, model, device)
        print(f"   Correlation: r = {corr:.4f}, MAE = {mae:.2f}")
        results["raw_embed"] = {"corr": corr, "mae": mae}

        # Stage 2: Post-LN1
        x_ln1 = block.ln_1(x)
        print("\n2. Post-LN1 (after first LayerNorm):")
        corr, mae = counting_method_on_activations(x_ln1, input_ids, model, device)
        print(f"   Correlation: r = {corr:.4f}, MAE = {mae:.2f}")
        results["post_ln1"] = {"corr": corr, "mae": mae}

        # Stage 3: Post-attention
        x_attn = x + block.attn(x_ln1)
        print("\n3. Post-Attention (after attention + residual):")
        corr, mae = counting_method_on_activations(x_attn, input_ids, model, device)
        print(f"   Correlation: r = {corr:.4f}, MAE = {mae:.2f}")
        results["post_attn"] = {"corr": corr, "mae": mae}

        # Stage 4: Post-LN2
        x_ln2 = block.ln_2(x_attn)
        print("\n4. Post-LN2 (after second LayerNorm):")
        corr, mae = counting_method_on_activations(x_ln2, input_ids, model, device)
        print(f"   Correlation: r = {corr:.4f}, MAE = {mae:.2f}")
        results["post_ln2"] = {"corr": corr, "mae": mae}

        # Stage 5: Post-MLP
        x_mlp = x_attn + block.mlp(x_ln2)
        print("\n5. Post-MLP (after MLP + residual):")
        corr, mae = counting_method_on_activations(x_mlp, input_ids, model, device)
        print(f"   Correlation: r = {corr:.4f}, MAE = {mae:.2f}")
        results["post_mlp"] = {"corr": corr, "mae": mae}

        # Stage 6: Post-final-LN
        x_final = model.transformer.ln_f(x_mlp)
        print("\n6. Post-Final-LN (after final LayerNorm):")
        corr, mae = counting_method_on_activations(x_final, input_ids, model, device)
        print(f"   Correlation: r = {corr:.4f}, MAE = {mae:.2f}")
        results["post_final_ln"] = {"corr": corr, "mae": mae}

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Stage':<25} {'Correlation':>12} {'MAE':>10}")
    print("-" * 50)
    for stage, data in results.items():
        print(f"{stage:<25} {data['corr']:>12.4f} {data['mae']:>10.2f}")

    return results, perplexity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_ablation(args.checkpoint, args.n_samples, device=args.device)
