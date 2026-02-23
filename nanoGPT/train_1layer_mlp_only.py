"""
Training Script for 1-Layer MLP-Only Experiment

This experiment trains a 1-block NoPE transformer where:
- Embeddings (wte) are FROZEN
- Attention (block.attn) is FROZEN
- LayerNorms (ln_1, ln_2, ln_f) are FROZEN
- MLP (block.mlp) is TRAINABLE
- Position head (pos_head) is TRAINABLE

The goal is to test whether position can be decoded using ONLY the MLP,
given that attention creates position-dependent variance patterns.

This script mirrors the structure and logging of train_2layer_mechanism.py
"""

import os
import sys
import time
import math
import json
import argparse
import random
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_1layer_mechanism import (
    OneLayerMechanismModel,
    OneLayerMechanismConfig,
    compute_position_metrics,
    compute_per_position_mae,
)


# =============================================================================
# Experiment Configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for 1-layer MLP-only experiment."""

    # Experiment identification
    experiment_name: str = "1layer_mlp_only"

    # WandB
    wandb_project: str = "nope-1layer-mechanism"
    wandb_log: bool = True

    # Model architecture
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 128
    vocab_size: int = 50304
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    dropout: float = 0.0
    use_bos: bool = True

    # Training
    max_iters: int = 80000  # Train for 80K steps
    batch_size: int = 64
    gradient_accumulation_steps: int = 2  # Effective batch = 128
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 500
    lr_decay_iters: int = 80000  # Match max_iters
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50  # Match 2-layer script

    # Data
    dataset: str = "openwebtext"
    data_dir: str = "data/openwebtext"

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True
    seed: int = 42

    # Output
    out_dir: str = "out-1layer-mlp-only"

    # BOS token
    bos_token_id: int = 50256


def parse_args():
    parser = argparse.ArgumentParser(
        description="1-Layer MLP-Only Position Decoding Training"
    )
    parser.add_argument("--max_iters", type=int, default=80000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument("--out_dir", type=str, default="out-1layer-mlp-only")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no_bos",
        action="store_true",
        help="Do not inject BOS token at position 0",
    )
    return parser.parse_args()


# =============================================================================
# Data Loading
# =============================================================================

train_data = None
val_data = None


def load_data(config: ExperimentConfig):
    """Load memory-mapped data files."""
    global train_data, val_data

    data_dir = config.data_dir
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    print(
        f"Loaded data: train={len(train_data):,} tokens, val={len(val_data):,} tokens"
    )


def get_batch(split: str, config: ExperimentConfig, device: str):
    """Get a batch of sequences with BOS token inserted at position 0."""
    global train_data, val_data
    data = train_data if split == "train" else val_data

    if data is None:
        raise RuntimeError(f"Data not loaded. Call load_data() before get_batch()")

    if config.use_bos:
        tokens_needed = config.block_size - 1
        ix = torch.randint(len(data) - tokens_needed, (config.batch_size,))
        sequences = []
        for i in ix:
            i = i.item()
            after_bos = data[i : i + tokens_needed].astype(np.int64)
            seq = np.concatenate([[config.bos_token_id], after_bos])
            sequences.append(torch.from_numpy(seq))
        x = torch.stack(sequences)
    else:
        tokens_needed = config.block_size
        ix = torch.randint(len(data) - tokens_needed, (config.batch_size,))
        sequences = []
        for i in ix:
            i = i.item()
            seq = data[i : i + tokens_needed].astype(np.int64)
            sequences.append(torch.from_numpy(seq))
        x = torch.stack(sequences)

    # Position targets: each position predicts its own index (0 to block_size-1)
    pos_targets = (
        torch.arange(config.block_size)
        .unsqueeze(0)
        .expand(config.batch_size, -1)
        .clone()
    )

    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        pos_targets = pos_targets.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        pos_targets = pos_targets.to(device)

    return x, pos_targets


def set_seed(seed: int) -> None:
    """Set seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Training Functions
# =============================================================================


def get_lr(it: int, config: ExperimentConfig) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def evaluate(model, config: ExperimentConfig, device: str, ctx) -> Dict:
    """Comprehensive evaluation on train and val splits."""
    model.eval()
    results = {}

    for split in ["train", "val"]:
        losses = []
        all_preds = []
        all_targets = []

        for _ in range(config.eval_iters):
            x, targets = get_batch(split, config, device)

            with ctx:
                preds, loss = model(x, targets)

            losses.append(loss.item())
            all_preds.append(preds)
            all_targets.append(targets)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = compute_position_metrics(all_preds, all_targets)
        per_pos_mae = compute_per_position_mae(
            all_preds, all_targets, config.block_size
        )

        results[f"{split}_loss"] = np.mean(losses)
        results[f"{split}_mae"] = metrics["mae"]
        results[f"{split}_r2"] = metrics["r2"]
        results[f"{split}_rmse"] = metrics["rmse"]
        results[f"{split}_per_pos_mae"] = per_pos_mae.cpu().numpy()

    model.train()
    return results


def create_per_position_plot(
    per_pos_mae: np.ndarray, title: str, iter_num: int
) -> plt.Figure:
    """Create per-position MAE plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(per_pos_mae))
    ax.plot(positions, per_pos_mae, "b-", linewidth=1.5)
    ax.fill_between(positions, 0, per_pos_mae, alpha=0.3)
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title(f"{title} (iter {iter_num})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(per_pos_mae) - 1)
    plt.tight_layout()
    return fig


def create_attention_grid(attn: torch.Tensor, title: str) -> plt.Figure:
    """Create a grid of attention maps (one per head)."""
    attn_mean = attn.mean(dim=0)
    n_heads = attn_mean.shape[0]
    n_cols = 4
    n_rows = math.ceil(n_heads / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    vmax = float(attn_mean.max()) if attn_mean.numel() else 1.0

    for head_idx in range(n_rows * n_cols):
        row, col = divmod(head_idx, n_cols)
        ax = axes[row, col]
        if head_idx < n_heads:
            ax.imshow(
                attn_mean[head_idx].cpu().numpy(), cmap="Blues", vmin=0.0, vmax=vmax
            )
            ax.set_title(f"Head {head_idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def create_bos_head_bar(attn: torch.Tensor, title: str) -> plt.Figure:
    """Create BOS attention-by-head bar plot (attention to position 0)."""
    attn_mean = attn.mean(dim=0)
    bos_scores = attn_mean[:, 1:, 0].mean(dim=1).cpu().numpy()
    n_heads = len(bos_scores)

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#D55E00" if s > 0.5 else "#0072B2" for s in bos_scores]
    ax.bar(range(n_heads), bos_scores, color=colors)
    ax.set_xlabel("Head")
    ax.set_ylabel("Mean Attention to Pos0")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def sample_attention_maps(
    model: OneLayerMechanismModel, config: ExperimentConfig, device: str, ctx
) -> Optional[torch.Tensor]:
    """Run a single batch to capture attention weights."""
    was_training = model.training
    model.eval()

    x, targets = get_batch("val", config, device)
    with torch.no_grad():
        with ctx:
            model(x, targets)

    attn = model.get_attention_weights()
    if was_training:
        model.train()
    return attn.detach().cpu() if attn is not None else None


def train(config: ExperimentConfig, args) -> Dict:
    """Train the 1-layer MLP-only model."""

    # Setup
    set_seed(config.seed)

    device = config.device
    device_type = "cuda" if "cuda" in device else "cpu"

    # Mixed precision context
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Create model
    model_config = OneLayerMechanismConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        dropout=config.dropout,
        norm_type=config.norm_type,
    )
    model = OneLayerMechanismModel(model_config)

    # Apply freezing: only MLP and position head are trainable
    model.freeze_all_except_mlp()

    model.to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {total_params:,} total params, {trainable_params:,} trainable ({100 * trainable_params / total_params:.1f}%)"
    )
    print("Trainable components:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} params")

    # Compile model
    if config.compile_model and not args.no_compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Create optimizer
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = raw_model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        device_type,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

    # Initialize WandB
    if config.wandb_log:
        import wandb

        run_name = config.experiment_name

        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                **asdict(config),
                "trainable_params": trainable_params,
                "total_params": total_params,
            },
            tags=["1layer", "mlp_only", "position_regression"],
            notes="1-layer NoPE with frozen weights except MLP and position head",
        )

        wandb.run.summary["regime_description"] = (
            "1-Layer MLP-Only: Freeze Emb, Attn, LN; train MLP + Pos Head"
        )

        run_out_dir = os.path.join(config.out_dir, wandb.run.id)
    else:
        run_out_dir = config.out_dir

    # Setup output directory
    os.makedirs(run_out_dir, exist_ok=True)

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Training: 1-Layer MLP-Only Position Decoding")
    print(f"Max iterations: {config.max_iters}")
    print(f"{'=' * 60}\n")

    best_val_loss = float("inf")
    best_metrics = {}

    for iter_num in range(config.max_iters + 1):
        # Set learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation
        if iter_num % config.eval_interval == 0:
            metrics = evaluate(model, config, device, ctx)

            print(
                f"iter {iter_num:5d} | "
                f"train loss {metrics['train_loss']:.4f} | "
                f"val loss {metrics['val_loss']:.4f} | "
                f"val MAE {metrics['val_mae']:.3f} | "
                f"val R² {metrics['val_r2']:.4f} | "
                f"lr {lr:.2e}"
            )

            # Log to WandB
            if config.wandb_log:
                import wandb

                log_dict = {
                    "iter": iter_num,
                    "train/loss": metrics["train_loss"],
                    "train/mae": metrics["train_mae"],
                    "train/r2": metrics["train_r2"],
                    "val/loss": metrics["val_loss"],
                    "val/mae": metrics["val_mae"],
                    "val/r2": metrics["val_r2"],
                    "val/rmse": metrics["val_rmse"],
                    "lr": lr,
                }

                # Per-position MAE - only log for short contexts to avoid wandb overhead
                if config.block_size <= 256:
                    for pos in range(config.block_size):
                        log_dict[f"val/per_pos_mae/{pos}"] = metrics["val_per_pos_mae"][
                            pos
                        ]

                # Create and log per-position plot
                fig = create_per_position_plot(
                    metrics["val_per_pos_mae"],
                    "Per-Position MAE (1-Layer MLP-Only)",
                    iter_num,
                )
                log_dict["val/per_pos_mae_plot"] = wandb.Image(fig)
                plt.close(fig)

                # Sample and log attention maps
                attn = sample_attention_maps(raw_model, config, device, ctx)
                if attn is not None:
                    fig_attn = create_attention_grid(
                        attn, f"Block Attention (iter {iter_num})"
                    )
                    log_dict["val/attn_block"] = wandb.Image(fig_attn)
                    plt.close(fig_attn)

                    fig_bos = create_bos_head_bar(
                        attn, f"BOS Attention by Head (iter {iter_num})"
                    )
                    log_dict["val/bos_head_bar"] = wandb.Image(fig_bos)
                    plt.close(fig_bos)

                wandb.log(log_dict, step=iter_num)

            # Save checkpoint if best
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_metrics = metrics.copy()

                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "best_metrics": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in best_metrics.items()
                    },
                }
                torch.save(checkpoint, os.path.join(run_out_dir, "best_ckpt.pt"))

        # Training step
        if iter_num < config.max_iters:
            accum_loss = 0.0
            for micro_step in range(config.gradient_accumulation_steps):
                x, targets = get_batch("train", config, device)

                with ctx:
                    preds, loss = model(x, targets)
                    loss_micro = loss / config.gradient_accumulation_steps

                scaler.scale(loss_micro).backward()
                accum_loss += loss.item()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            if iter_num % config.log_interval == 0 and iter_num > 0:
                print(f"  iter {iter_num}: loss = {accum_loss:.4f}")

    # Final evaluation
    final_metrics = evaluate(model, config, device, ctx)

    # Save final checkpoint
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "iter_num": config.max_iters,
        "final_metrics": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in final_metrics.items()
        },
    }
    torch.save(checkpoint, os.path.join(run_out_dir, "final_ckpt.pt"))

    # Log final results to WandB
    if config.wandb_log:
        import wandb

        wandb.run.summary["final_val_mae"] = final_metrics["val_mae"]
        wandb.run.summary["final_val_r2"] = final_metrics["val_r2"]
        wandb.run.summary["final_val_loss"] = final_metrics["val_loss"]
        wandb.run.summary["best_val_mae"] = best_metrics.get(
            "val_mae", final_metrics["val_mae"]
        )
        wandb.run.summary["best_val_r2"] = best_metrics.get(
            "val_r2", final_metrics["val_r2"]
        )

        wandb.finish()

    return {
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "checkpoint_dir": run_out_dir,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # Create config
    config = ExperimentConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        seed=args.seed,
        wandb_log=args.wandb,
        compile_model=not args.no_compile,
        out_dir=args.out_dir,
        device=args.device,
        use_bos=not args.no_bos,
    )

    # Load data
    load_data(config)

    print(f"\n*** BOS token (ID {config.bos_token_id}) is inserted at position 0 ***\n")

    # Run training
    results = train(config, args)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final val MAE: {results['final_metrics']['val_mae']:.4f}")
    print(f"Final val R²: {results['final_metrics']['val_r2']:.4f}")
    print(f"Best val R²: {results['best_metrics'].get('val_r2', 'N/A')}")
    print("=" * 60)

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
