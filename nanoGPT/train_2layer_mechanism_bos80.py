"""
Training Script for 2-Layer Position Regressor with BOS at Position 80

Same as train_2layer_mechanism.py but with BOS token inserted at position 80
instead of position 0. This tests whether the model can learn to use a
non-standard BOS reference position.

Usage:
    # Train R0 regime (full training)
    CUDA_VISIBLE_DEVICES=0 python train_2layer_mechanism_bos80.py --regime R0 --wandb

    # With custom config
    python train_2layer_mechanism_bos80.py --regime R0 --max_iters 20000 --wandb
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_2layer_mechanism import (
    TwoLayerMechanismModel,
    TwoLayerMechanismConfig,
    compute_position_metrics,
    compute_per_position_mae,
)


# =============================================================================
# BOS Configuration
# =============================================================================

BOS_TOKEN_ID = 50256  # GPT-2 EOT token (used as BOS)
BOS_POSITION = 80     # Insert BOS at position 80


# =============================================================================
# Experiment Configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for mechanism dissection experiments with BOS at position 80."""

    # Experiment identification
    experiment_name: str = "bos80_posreg"
    regime: str = "R0"  # R0, R1, R2, R3, R4_linear, R4_mlp
    hypothesis: str = "bos_position_80"

    # WandB
    wandb_project: str = "nope-2layer-mechanism-bos80"
    wandb_log: bool = True

    # Model architecture (per spec: d=768, n_heads=12, d_head=64)
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 128  # Sequence length L
    vocab_size: int = 50304
    norm_type: str = "layernorm"
    dropout: float = 0.0

    # BOS configuration
    bos_position: int = 80  # Position where BOS token is inserted
    bos_token_id: int = 50256  # BOS token ID

    # Training
    max_iters: int = 20000
    batch_size: int = 64
    gradient_accumulation_steps: int = 2  # Effective batch = 128
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 500
    lr_decay_iters: int = 20000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50

    # Data
    dataset: str = "openwebtext"
    data_dir: str = "/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext"

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True
    seed: int = 42

    # Output
    out_dir: str = "out-2layer-mechanism-bos80"


def parse_args():
    parser = argparse.ArgumentParser(
        description="2-Layer Mechanism Training with BOS at Position 80"
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="R0",
        choices=["R0", "R1", "R2", "R3", "R4_linear", "R4_mlp", "all"],
        help="Training regime to run",
    )
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bos_position", type=int, default=80, help="Position for BOS token")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument("--out_dir", type=str, default="out-2layer-mechanism-bos80")
    parser.add_argument("--device", type=str, default="cuda")
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
    """Get a batch of sequences with BOS token inserted at position 80.

    The sequence structure is:
    - Positions 0 to bos_position-1: Text tokens from corpus
    - Position bos_position (80): BOS token (ID 50256)
    - Positions bos_position+1 to block_size-1: Text tokens from corpus
    """
    data = train_data if split == "train" else val_data
    bos_pos = config.bos_position
    bos_token = config.bos_token_id

    # We need (block_size - 1) tokens from corpus to fill all non-BOS positions
    tokens_needed = config.block_size - 1

    # Random starting indices
    ix = torch.randint(len(data) - tokens_needed, (config.batch_size,))

    # Build sequences with BOS at specified position
    sequences = []
    for i in ix:
        i = i.item()
        # Get tokens for positions before BOS (0 to bos_pos-1)
        before_bos = data[i : i + bos_pos].astype(np.int64)
        # Get tokens for positions after BOS (bos_pos+1 to block_size-1)
        after_bos = data[i + bos_pos : i + tokens_needed].astype(np.int64)
        # Construct sequence: [text_0:bos_pos-1, BOS, text_bos_pos:end]
        seq = np.concatenate([before_bos, [bos_token], after_bos])
        sequences.append(torch.from_numpy(seq))

    x = torch.stack(sequences)

    # Position targets remain unchanged: 0 to block_size-1
    pos_targets = (
        torch.arange(config.block_size).unsqueeze(0).expand(config.batch_size, -1)
    )

    x = x.to(device)
    pos_targets = pos_targets.to(device)

    return x, pos_targets


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
        maes = []
        r2s = []
        all_preds = []
        all_targets = []

        for _ in range(config.eval_iters):
            x, targets = get_batch(split, config, device)

            with ctx:
                output, loss = model(x, targets, capture_taps=False)

            losses.append(loss.item())

            # Get predictions
            if config.use_regression if hasattr(config, "use_regression") else True:
                preds = output.squeeze(-1)  # [B, T]
            else:
                preds = output.argmax(dim=-1).float()

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
    per_pos_mae: np.ndarray, title: str, iter_num: int, bos_position: int = 80
) -> plt.Figure:
    """Create per-position MAE plot with BOS position highlighted."""
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(per_pos_mae))
    ax.plot(positions, per_pos_mae, "b-", linewidth=1.5)
    ax.fill_between(positions, 0, per_pos_mae, alpha=0.3)

    # Highlight BOS position
    ax.axvline(x=bos_position, color='r', linestyle='--', linewidth=2,
               label=f'BOS position ({bos_position})')
    ax.scatter([bos_position], [per_pos_mae[bos_position]], color='r', s=100, zorder=5)

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title(f"{title} (iter {iter_num})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(per_pos_mae) - 1)
    ax.legend()
    plt.tight_layout()
    return fig


def train_regime(regime: str, config: ExperimentConfig, args) -> Dict:
    """Train a single regime and return results."""

    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

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
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Create model
    model_config = TwoLayerMechanismConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        dropout=config.dropout,
        norm_type=config.norm_type,
        use_regression=True,
    )
    model = TwoLayerMechanismModel(model_config)

    # Apply freezing regime
    if regime == "R0":
        model.apply_regime_R0()
    elif regime == "R1":
        model.apply_regime_R1()
    elif regime == "R2":
        model.apply_regime_R2()
    elif regime == "R3":
        model.apply_regime_R3()
    elif regime == "R4_linear":
        model.apply_regime_R4(use_mlp_probe=False)
    elif regime == "R4_mlp":
        model.apply_regime_R4(use_mlp_probe=True)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    model.to(device)

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

    # Setup output directory
    regime_out_dir = os.path.join(config.out_dir, regime)
    os.makedirs(regime_out_dir, exist_ok=True)

    # Initialize WandB
    if config.wandb_log:
        import wandb

        # Detailed run name
        run_name = f"{regime}_bos{config.bos_position}_{config.experiment_name}"

        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                **asdict(config),
                "regime": regime,
                "bos_position": config.bos_position,
                "bos_token_id": config.bos_token_id,
            },
            tags=[regime, "bos80", "position_regression"],
            notes=f"Regime {regime}: Position regression with BOS at position {config.bos_position}",
        )

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Training Regime: {regime}")
    print(f"BOS Position: {config.bos_position}")
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
                    # Log BOS position MAE separately
                    log_dict["val/bos_position_mae"] = metrics["val_per_pos_mae"][
                        config.bos_position
                    ]

                # Create and log per-position plot
                fig = create_per_position_plot(
                    metrics["val_per_pos_mae"],
                    f"Per-Position MAE ({regime}, BOS@{config.bos_position})",
                    iter_num,
                    config.bos_position
                )
                log_dict["val/per_pos_mae_plot"] = wandb.Image(fig)
                plt.close(fig)

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
                    "regime": regime,
                    "bos_position": config.bos_position,
                }
                torch.save(checkpoint, os.path.join(regime_out_dir, "best_ckpt.pt"))

        # Training step
        if iter_num < config.max_iters:
            for micro_step in range(config.gradient_accumulation_steps):
                x, targets = get_batch("train", config, device)

                with ctx:
                    output, loss = model(x, targets, capture_taps=False)
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            if iter_num % config.log_interval == 0 and iter_num > 0:
                loss_val = loss.item() * config.gradient_accumulation_steps
                print(f"  iter {iter_num}: loss = {loss_val:.4f}")

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
        "regime": regime,
        "bos_position": config.bos_position,
    }
    torch.save(checkpoint, os.path.join(regime_out_dir, "final_ckpt.pt"))

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
        wandb.run.summary["bos_position"] = config.bos_position

        wandb.finish()

    return {
        "regime": regime,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "checkpoint_dir": regime_out_dir,
    }


def run_all_regimes(config: ExperimentConfig, args) -> Dict:
    """Run all regimes and create comparison summary."""

    regimes = ["R0", "R1", "R2", "R3", "R4_linear", "R4_mlp"]
    all_results = {}

    print("\n" + "=" * 80)
    print(f"POSITION REGRESSION WITH BOS AT POSITION {config.bos_position}")
    print("=" * 80)
    print("\nRunning all regimes...")
    print("=" * 80 + "\n")

    for regime in regimes:
        print(f"\n{'=' * 60}")
        print(f"STARTING REGIME: {regime}")
        print(f"{'=' * 60}")

        results = train_regime(regime, config, args)
        all_results[regime] = results

    # Create summary table
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY (BOS @ position {config.bos_position})")
    print("=" * 80)
    print(f"\n{'Regime':<15} {'Val MAE':<12} {'Val R²':<12} {'Val Loss':<12}")
    print("-" * 51)

    for regime in regimes:
        r = all_results[regime]["final_metrics"]
        print(
            f"{regime:<15} {r['val_mae']:<12.4f} {r['val_r2']:<12.4f} {r['val_loss']:<12.4f}"
        )

    print("-" * 51)

    # Save summary
    summary_path = os.path.join(config.out_dir, "experiment_summary.json")
    summary = {
        "bos_position": config.bos_position,
        "regimes": {
            regime: {
                "val_mae": all_results[regime]["final_metrics"]["val_mae"],
                "val_r2": all_results[regime]["final_metrics"]["val_r2"],
                "val_loss": all_results[regime]["final_metrics"]["val_loss"],
            }
            for regime in regimes
        },
        "config": asdict(config),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return all_results


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # Create config
    config = ExperimentConfig(
        regime=args.regime,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        seed=args.seed,
        bos_position=args.bos_position,
        wandb_log=args.wandb,
        compile_model=not args.no_compile,
        out_dir=args.out_dir,
        device=args.device,
    )

    # Load data
    load_data(config)

    # Verify BOS position is valid
    if config.bos_position >= config.block_size:
        raise ValueError(
            f"BOS position ({config.bos_position}) must be less than block_size ({config.block_size})"
        )

    print(f"\n*** BOS token (ID {config.bos_token_id}) will be inserted at position {config.bos_position} ***\n")

    # Run experiments
    if args.regime == "all":
        results = run_all_regimes(config, args)
    else:
        results = train_regime(args.regime, config, args)

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
