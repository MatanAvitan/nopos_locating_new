"""
Compare Forced-BOS LM vs Vanilla NoPE LM

Loads both models and compares:
1. Perplexity at training context length
2. Extrapolation performance at 2x, 4x, 8x context
3. Position encoding metrics (channel correlations)
4. Attention patterns

Usage:
    python analysis_scripts/compare_forced_bos_vs_vanilla.py \
        --forced_bos_checkpoint nanoGPT/out-lm-6layer-forced-bos/ckpt.pt \
        --vanilla_checkpoint nanoGPT/out-lm-6layer-fulltrain-ddp/ckpt.pt \
        --wandb
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint, detecting model type automatically."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]

    # Detect model type from checkpoint
    if "freeze_uniform_head_idx" in model_args:
        from model_nope_forced_bos import GPTConfigForcedBOS, GPTForcedBOS

        config = GPTConfigForcedBOS(**model_args)
        model = GPTForcedBOS(config)
        model_type = "forced_bos"
    else:
        from model_nope import GPTConfig, GPT

        config = GPTConfig(**model_args)
        model = GPT(config)
        model_type = "vanilla"

    # Handle state dict prefix from DDP/compile
    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    model.eval()

    train_iter = checkpoint.get("iter_num", 0)
    return model, model_args, model_type, train_iter


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Get a random batch of data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_perplexity(
    model,
    data: np.ndarray,
    context_length: int,
    batch_size: int,
    n_batches: int,
    device: str,
) -> dict:
    """Evaluate perplexity at a given context length."""
    model.eval()
    original_block_size = model.config.block_size
    model.config.block_size = max(context_length, original_block_size)

    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        losses.append(loss.item())

    model.config.block_size = original_block_size

    return {
        "loss": np.mean(losses),
        "loss_std": np.std(losses),
        "perplexity": np.exp(np.mean(losses)),
    }


@torch.no_grad()
def compute_channel_position_correlations(
    model,
    data: np.ndarray,
    context_length: int,
    batch_size: int,
    n_batches: int,
    device: str,
) -> dict:
    """Compute per-channel correlations with position."""
    model.eval()

    # Collect activations at each position
    all_acts = []
    positions = []

    for _ in range(n_batches):
        x, _ = get_batch(data, batch_size, context_length, device)

        # Forward pass to get post-LN2 activations from block 1 (the write bottleneck)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Embed
            tok_emb = model.transformer.wte(x)
            h = model.transformer.drop(tok_emb)

            # Block 0
            h = model.transformer.h[0](h)

            # Block 1 - get post-LN2 (before MLP)
            block1 = model.transformer.h[1]
            h_post_attn = h + block1.attn(block1.ln_1(h))
            h_post_ln2 = block1.ln_2(h_post_attn)

        # Collect [B, T, D]
        acts = h_post_ln2.float().cpu().numpy()
        for b in range(acts.shape[0]):
            for t in range(acts.shape[1]):
                all_acts.append(acts[b, t])
                positions.append(t)

    all_acts = np.array(all_acts)  # [N, D]
    positions = np.array(positions)  # [N]

    # Compute per-channel correlation with position
    n_channels = all_acts.shape[1]
    correlations = []
    for c in range(n_channels):
        r, p = stats.pearsonr(all_acts[:, c], positions)
        correlations.append({"channel": c, "correlation": r, "p_value": p})

    correlations_sorted = sorted(
        correlations, key=lambda x: abs(x["correlation"]), reverse=True
    )

    # Count channels with |r| > 0.95
    high_corr_channels = [c for c in correlations if abs(c["correlation"]) > 0.95]

    return {
        "n_high_corr_channels": len(high_corr_channels),
        "max_abs_corr": max(abs(c["correlation"]) for c in correlations),
        "mean_abs_corr": np.mean([abs(c["correlation"]) for c in correlations]),
        "top_10_channels": correlations_sorted[:10],
    }


def plot_comparison(forced_bos_results: dict, vanilla_results: dict, save_path: str):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Perplexity vs context length
    ax1 = axes[0]
    contexts = sorted(forced_bos_results["extrapolation"].keys())
    fb_ppls = [forced_bos_results["extrapolation"][c]["perplexity"] for c in contexts]
    v_ppls = [vanilla_results["extrapolation"][c]["perplexity"] for c in contexts]

    ax1.plot(contexts, fb_ppls, "o-", label="Forced BOS", color="blue", linewidth=2)
    ax1.plot(contexts, v_ppls, "s-", label="Vanilla NoPE", color="orange", linewidth=2)
    ax1.set_xlabel("Context Length", fontsize=12)
    ax1.set_ylabel("Perplexity", fontsize=12)
    ax1.set_title("Extrapolation Performance", fontsize=14)
    ax1.legend()
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: PPL degradation ratio
    ax2 = axes[1]
    train_ctx = min(contexts)
    fb_base = forced_bos_results["extrapolation"][train_ctx]["perplexity"]
    v_base = vanilla_results["extrapolation"][train_ctx]["perplexity"]
    fb_degrad = [
        forced_bos_results["extrapolation"][c]["perplexity"] / fb_base for c in contexts
    ]
    v_degrad = [
        vanilla_results["extrapolation"][c]["perplexity"] / v_base for c in contexts
    ]

    ax2.plot(contexts, fb_degrad, "o-", label="Forced BOS", color="blue", linewidth=2)
    ax2.plot(
        contexts, v_degrad, "s-", label="Vanilla NoPE", color="orange", linewidth=2
    )
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Context Length", fontsize=12)
    ax2.set_ylabel("PPL Degradation (relative)", fontsize=12)
    ax2.set_title("Degradation vs Training Context", fontsize=14)
    ax2.legend()
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Channel correlations comparison
    ax3 = axes[2]
    metrics = ["n_high_corr_channels", "max_abs_corr", "mean_abs_corr"]
    labels = ["# High |r|>0.95", "Max |r|", "Mean |r|"]
    fb_vals = [forced_bos_results["channel_correlations"][m] for m in metrics]
    v_vals = [vanilla_results["channel_correlations"][m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width / 2, fb_vals, width, label="Forced BOS", color="blue", alpha=0.7)
    ax3.bar(
        x + width / 2, v_vals, width, label="Vanilla NoPE", color="orange", alpha=0.7
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Value", fontsize=12)
    ax3.set_title("Position Encoding Metrics", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Forced-BOS vs Vanilla NoPE LM"
    )
    parser.add_argument(
        "--forced_bos_checkpoint",
        type=str,
        required=True,
        help="Path to forced-BOS model checkpoint",
    )
    parser.add_argument(
        "--vanilla_checkpoint",
        type=str,
        required=True,
        help="Path to vanilla NoPE model checkpoint",
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Context lengths to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=50,
        help="Number of batches per evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        help="Dataset for evaluation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/forced_bos_comparison",
        help="Directory to save results",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to W&B",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    print("\n" + "=" * 60)
    print("Loading Forced-BOS model...")
    fb_model, fb_args, fb_type, fb_iter = load_model(args.forced_bos_checkpoint, device)
    print(f"  Type: {fb_type}, Iterations: {fb_iter}")

    print("\nLoading Vanilla NoPE model...")
    v_model, v_args, v_type, v_iter = load_model(args.vanilla_checkpoint, device)
    print(f"  Type: {v_type}, Iterations: {v_iter}")
    print("=" * 60)

    # Load validation data
    data_path = (
        Path(__file__).parent.parent / "nanoGPT" / "data" / args.dataset / "val.bin"
    )
    val_data = np.memmap(data_path, dtype=np.uint16, mode="r")
    print(f"\nValidation data: {len(val_data):,} tokens")

    # Setup W&B
    if args.wandb:
        import wandb

        wandb.init(
            project="nope-lm",
            name="forced-bos-vs-vanilla-comparison",
            config=vars(args),
        )

    # Evaluate both models
    forced_bos_results = {"extrapolation": {}, "channel_correlations": {}}
    vanilla_results = {"extrapolation": {}, "channel_correlations": {}}

    print("\nEvaluating extrapolation...")
    print("-" * 60)

    for ctx in args.context_lengths:
        print(f"Context {ctx}:")

        # Forced BOS
        fb_ppl = evaluate_perplexity(
            fb_model, val_data, ctx, args.batch_size, args.n_batches, device
        )
        forced_bos_results["extrapolation"][ctx] = fb_ppl
        print(f"  Forced-BOS: PPL={fb_ppl['perplexity']:.2f}")

        # Vanilla
        v_ppl = evaluate_perplexity(
            v_model, val_data, ctx, args.batch_size, args.n_batches, device
        )
        vanilla_results["extrapolation"][ctx] = v_ppl
        print(f"  Vanilla:    PPL={v_ppl['perplexity']:.2f}")

    # Compute channel correlations
    print("\nComputing channel-position correlations...")
    train_ctx = min(args.context_lengths)

    forced_bos_results["channel_correlations"] = compute_channel_position_correlations(
        fb_model, val_data, train_ctx, args.batch_size, args.n_batches, device
    )
    print(
        f"  Forced-BOS: {forced_bos_results['channel_correlations']['n_high_corr_channels']} channels with |r|>0.95"
    )

    vanilla_results["channel_correlations"] = compute_channel_position_correlations(
        v_model, val_data, train_ctx, args.batch_size, args.n_batches, device
    )
    print(
        f"  Vanilla:    {vanilla_results['channel_correlations']['n_high_corr_channels']} channels with |r|>0.95"
    )

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)

    all_results = {
        "forced_bos": {
            "checkpoint": args.forced_bos_checkpoint,
            "train_iter": fb_iter,
            **forced_bos_results,
        },
        "vanilla": {
            "checkpoint": args.vanilla_checkpoint,
            "train_iter": v_iter,
            **vanilla_results,
        },
    }

    results_file = Path(args.save_dir) / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Create comparison plot
    plot_path = Path(args.save_dir) / "comparison_plot.png"
    plot_comparison(forced_bos_results, vanilla_results, str(plot_path))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for ctx in args.context_lengths:
        fb_ppl = forced_bos_results["extrapolation"][ctx]["perplexity"]
        v_ppl = vanilla_results["extrapolation"][ctx]["perplexity"]
        diff = ((fb_ppl - v_ppl) / v_ppl) * 100
        print(
            f"Context {ctx}: Forced-BOS={fb_ppl:.2f}, Vanilla={v_ppl:.2f} ({diff:+.1f}%)"
        )

    print("\nChannel correlations (|r|>0.95):")
    print(
        f"  Forced-BOS: {forced_bos_results['channel_correlations']['n_high_corr_channels']}"
    )
    print(
        f"  Vanilla:    {vanilla_results['channel_correlations']['n_high_corr_channels']}"
    )

    if args.wandb:
        import wandb

        wandb.log(
            {
                "forced_bos/ppl_128": forced_bos_results["extrapolation"][128][
                    "perplexity"
                ],
                "vanilla/ppl_128": vanilla_results["extrapolation"][128]["perplexity"],
                "forced_bos/n_high_corr_channels": forced_bos_results[
                    "channel_correlations"
                ]["n_high_corr_channels"],
                "vanilla/n_high_corr_channels": vanilla_results["channel_correlations"][
                    "n_high_corr_channels"
                ],
            }
        )
        wandb.log({"comparison_plot": wandb.Image(str(plot_path))})
        wandb.finish()

    return all_results


if __name__ == "__main__":
    main()
