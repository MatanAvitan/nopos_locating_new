"""
Out-of-Distribution (OOD) Generalization Evaluation for NoPE Models

Evaluates trained models on context lengths longer than training (2048).
Tests whether NoPE mechanism generalizes to unseen positions.

Usage:
    python ood_generalization_eval.py --checkpoint path/to/ckpt.pt --max_ctx 8192
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


def load_model(checkpoint_path: str, device: str = "cuda") -> tuple[GPT, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]

    # Handle skip_ln2 for older checkpoints
    if "skip_ln2" not in model_args:
        model_args["skip_ln2"] = False

    # Handle use_batchnorm_ln2 for older checkpoints
    if "use_batchnorm_ln2" not in model_args:
        model_args["use_batchnorm_ln2"] = False

    config = GPTConfig(**model_args)
    model = GPT(config)

    # Handle _orig_mod prefix from torch.compile
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint.get("config", {})


def compute_perplexity(model, tokens, device="cuda"):
    """Compute perplexity on given tokens."""
    model.eval()
    with torch.no_grad():
        tokens = tokens.to(device)
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
        perplexity = torch.exp(loss).item()
    return perplexity


def compute_position_probe_r2(model, ctx_len, n_samples=100, device="cuda"):
    """
    Train a linear probe to predict position from post-LN2 activations.
    Returns R^2 score.
    """
    model.eval()

    # Generate random tokens
    vocab_size = model.config.vocab_size
    tokens = torch.randint(0, vocab_size, (n_samples, ctx_len), device=device)

    # Get activations (we need to hook into the model)
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    # Hook after final layer norm
    hook = model.transformer.ln_f.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, n_samples, 10):  # Process in batches
            batch = tokens[i:i+10]
            _ = model(batch)

    hook.remove()

    # Stack activations: [n_samples, ctx_len, n_embd]
    acts = torch.cat(activations, dim=0).numpy()

    # Flatten for regression: [n_samples * ctx_len, n_embd]
    n_total = acts.shape[0] * acts.shape[1]
    X = acts.reshape(n_total, -1)

    # Positions as targets
    positions = np.tile(np.arange(ctx_len), acts.shape[0])

    # Train probe
    probe = Ridge(alpha=1.0)
    probe.fit(X, positions)

    # Compute R^2
    predictions = probe.predict(X)
    r2 = 1 - np.sum((positions - predictions)**2) / np.sum((positions - positions.mean())**2)

    return r2


def evaluate_model(checkpoint_path: str, context_lengths: list, device: str = "cuda"):
    """Evaluate model on multiple context lengths."""
    print(f"\nLoading model from {checkpoint_path}")
    model, train_config = load_model(checkpoint_path, device)

    # Temporarily increase block_size for OOD evaluation
    original_block_size = model.config.block_size
    max_ctx = max(context_lengths)

    results = {
        "checkpoint": checkpoint_path,
        "train_block_size": original_block_size,
        "train_config": train_config,
        "model_config": {
            "use_positional_embedding": model.config.use_positional_embedding,
            "norm_type": model.config.norm_type,
            "skip_ln2": model.config.skip_ln2,
            "use_batchnorm_ln2": model.config.use_batchnorm_ln2,
        },
        "evaluations": {}
    }

    for ctx_len in context_lengths:
        print(f"\n  Evaluating context length: {ctx_len}")

        # Temporarily set block_size
        model.config.block_size = ctx_len

        # Skip if context too long (model can't handle it)
        # For models with PE, this is limited by wpe size
        if model.config.use_positional_embedding and ctx_len > original_block_size:
            print(f"    Skipping - model has PE and ctx_len > trained block_size")
            results["evaluations"][ctx_len] = {"skipped": "PE model cannot extrapolate"}
            continue

        try:
            # Compute metrics
            r2 = compute_position_probe_r2(model, ctx_len, n_samples=50, device=device)

            # Generate random tokens for perplexity
            vocab_size = model.config.vocab_size
            tokens = torch.randint(0, vocab_size, (10, ctx_len), device=device)
            ppl = compute_perplexity(model, tokens, device)

            results["evaluations"][ctx_len] = {
                "position_probe_r2": r2,
                "perplexity_random": ppl,
                "is_ood": ctx_len > original_block_size
            }

            print(f"    Position probe R^2: {r2:.4f}")
            print(f"    Perplexity (random): {ppl:.2f}")
            print(f"    OOD: {ctx_len > original_block_size}")

        except Exception as e:
            print(f"    Error: {e}")
            results["evaluations"][ctx_len] = {"error": str(e)}

    # Restore original block_size
    model.config.block_size = original_block_size

    return results


def main():
    parser = argparse.ArgumentParser(description="OOD Generalization Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--max_ctx", type=int, default=8192, help="Maximum context length to test")
    parser.add_argument("--save_dir", type=str, default="results/ood_generalization", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Context lengths to test
    context_lengths = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
    context_lengths = [c for c in context_lengths if c <= args.max_ctx]

    print(f"Testing context lengths: {context_lengths}")

    # Evaluate
    results = evaluate_model(args.checkpoint, context_lengths, args.device)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = Path(args.checkpoint).parent.name
    output_path = os.path.join(args.save_dir, f"{ckpt_name}_ood_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
