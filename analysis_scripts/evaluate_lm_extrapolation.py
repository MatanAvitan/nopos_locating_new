"""
LM Extrapolation Evaluation Script

Evaluates perplexity of trained NoPE LMs at various sequence lengths to test extrapolation.
Compares forced-BOS mechanism LM against vanilla NoPE LM.

Usage:
    python analysis_scripts/evaluate_lm_extrapolation.py \
        --checkpoint nanoGPT/out-lm-6layer-forced-bos/ckpt.pt \
        --context_lengths 128 256 512 1024 2048 \
        --n_batches 50 \
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

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint, detecting model type automatically."""
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

    return model, model_args, model_type


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
    train_context: int,
) -> dict:
    """Evaluate perplexity at a given context length."""
    model.eval()

    # Temporarily adjust model's block_size for longer contexts
    original_block_size = model.config.block_size
    model.config.block_size = max(context_length, original_block_size)

    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, context_length, device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)

        losses.append(loss.item())

    # Restore original block_size
    model.config.block_size = original_block_size

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    perplexity = np.exp(mean_loss)

    return {
        "context_length": context_length,
        "loss": mean_loss,
        "loss_std": std_loss,
        "perplexity": perplexity,
        "extrapolation_ratio": context_length / train_context,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LM extrapolation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
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
        help="Number of batches per context length",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/lm_extrapolation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to W&B",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nope-lm",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not provided)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_args, model_type = load_model(args.checkpoint, device)
    train_context = model_args.get("block_size", 128)
    print(f"Model type: {model_type}")
    print(f"Training context length: {train_context}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Load validation data
    data_path = (
        Path(__file__).parent.parent / "nanoGPT" / "data" / args.dataset / "val.bin"
    )
    print(f"\nLoading validation data from: {data_path}")
    val_data = np.memmap(data_path, dtype=np.uint16, mode="r")
    print(f"Validation data size: {len(val_data):,} tokens")

    # Setup W&B
    if args.wandb:
        import wandb

        run_name = (
            args.wandb_run_name
            or f"extrapolation-{model_type}-{Path(args.checkpoint).parent.name}"
        )
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "checkpoint": args.checkpoint,
                "model_type": model_type,
                "train_context": train_context,
                "context_lengths": args.context_lengths,
                "batch_size": args.batch_size,
                "n_batches": args.n_batches,
                **model_args,
            },
        )

    # Evaluate at each context length
    results = []
    print("\nEvaluating perplexity at different context lengths:")
    print("-" * 60)

    for ctx_len in args.context_lengths:
        if ctx_len > len(val_data) - 1:
            print(f"Skipping context {ctx_len}: exceeds data length")
            continue

        print(f"Evaluating context length {ctx_len}...", end=" ", flush=True)

        result = evaluate_perplexity(
            model=model,
            data=val_data,
            context_length=ctx_len,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            device=device,
            train_context=train_context,
        )
        results.append(result)

        extrap_str = (
            f"({result['extrapolation_ratio']:.1f}x)"
            if result["extrapolation_ratio"] > 1
            else "(in-dist)"
        )
        print(f"PPL={result['perplexity']:.2f} {extrap_str}")

        if args.wandb:
            wandb.log(
                {
                    f"perplexity/ctx_{ctx_len}": result["perplexity"],
                    f"loss/ctx_{ctx_len}": result["loss"],
                    "context_length": ctx_len,
                }
            )

    # Print summary table
    print("\n" + "=" * 60)
    print("EXTRAPOLATION SUMMARY")
    print("=" * 60)
    print(f"{'Context':<10} {'Extrap Ratio':<15} {'Loss':<12} {'Perplexity':<12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['context_length']:<10} {r['extrapolation_ratio']:<15.1f} {r['loss']:<12.4f} {r['perplexity']:<12.2f}"
        )

    # Compute degradation metrics
    in_dist_result = next(
        (r for r in results if r["context_length"] == train_context), results[0]
    )
    for r in results:
        r["ppl_degradation"] = r["perplexity"] / in_dist_result["perplexity"]

    print("\n" + "-" * 60)
    print("Degradation relative to in-distribution:")
    for r in results:
        if r["extrapolation_ratio"] > 1:
            print(f"  {r['context_length']}: {r['ppl_degradation']:.2f}x worse")

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = Path(args.checkpoint).parent.name
    results_file = Path(args.save_dir) / f"extrapolation_{ckpt_name}.json"

    save_data = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "train_context": train_context,
        "model_args": model_args,
        "results": results,
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    if args.wandb:
        # Log summary table
        import wandb

        table = wandb.Table(
            columns=[
                "Context Length",
                "Extrapolation Ratio",
                "Loss",
                "Perplexity",
                "PPL Degradation",
            ],
            data=[
                [
                    r["context_length"],
                    r["extrapolation_ratio"],
                    r["loss"],
                    r["perplexity"],
                    r["ppl_degradation"],
                ]
                for r in results
            ],
        )
        wandb.log({"extrapolation_table": table})
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
