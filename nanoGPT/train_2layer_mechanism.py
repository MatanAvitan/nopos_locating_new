"""
Training Script for 2-Layer Mechanism Dissection Experiments

This script implements Experiment 1 from the mechanism dissection spec:
- R0: Full training baseline
- R1: Block2-only (freeze Emb, Attn1, MLP1)
- R2: Attn2-only (freeze Emb, Attn1, MLP1, MLP2)
- R3: MLP2-only (freeze Emb, Attn1, MLP1, Attn2)
- R4: Head-only probe (linear and MLP variants)

All experiments use:
- WandB project: nope-2layer-mechanism
- Comprehensive logging of hypotheses, metrics, and diagnostic info
- Same random seed across regimes for fair comparison

Usage:
    # Train single regime
    python train_2layer_mechanism.py --regime R0 --wandb

    # Train all regimes
    python train_2layer_mechanism.py --regime all --wandb

    # With custom config
    python train_2layer_mechanism.py --regime R1 --max_iters 20000 --wandb
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
# Experiment Configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for mechanism dissection experiments."""

    # Experiment identification
    experiment_name: str = "exp1_freeze_train_matrix"
    regime: str = "R0"  # R0, R1, R2, R3, R4_linear, R4_mlp
    hypothesis: str = "mechanism_dissection"

    # WandB
    wandb_project: str = "nope-2layer-mechanism"
    wandb_log: bool = True

    # Model architecture (per spec: d=768, n_heads=12, d_head=64)
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 128  # Sequence length L
    vocab_size: int = 50304
    norm_type: str = "layernorm"
    dropout: float = 0.0
    head_on_post_attn: bool = False
    r2_attn_head_only: bool = False
    use_bos: bool = True

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
    data_dir: str = "data/openwebtext"

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True
    seed: int = 42

    # Output
    out_dir: str = "out-2layer-mechanism"

    # BOS token
    bos_token_id: int = 50256


def parse_args():
    parser = argparse.ArgumentParser(
        description="2-Layer Mechanism Dissection Training"
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
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument("--out_dir", type=str, default="out-2layer-mechanism")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--head_on_post_attn",
        action="store_true",
        help="Apply head on block2 post-attn residual (skip MLP2)",
    )
    parser.add_argument(
        "--r2_attn_head_only",
        action="store_true",
        help="R2 strict mode: train only Block2.Attn + Pos Head (freeze LN/MLP)",
    )
    parser.add_argument(
        "--no_bos",
        action="store_true",
        help="Do not inject BOS token at position 0",
    )
    return parser.parse_args()


# =============================================================================
# Hypothesis Definitions (for WandB logging)
# =============================================================================

HYPOTHESES = {
    "H1": {
        "name": "Iterated-Averaging / Harmonic-Profile",
        "description": "Block1 produces prefix-mean summary, Block2 applies another aggregation, "
        "creating harmonic-tail coefficients c_{i,k} ∝ (H_i - H_{k-1})/i",
        "predictions": {
            "R2_performance": "high",  # Attn2-only should work well
            "R3_performance": "low",  # MLP2-only should fail
            "R4_performance": "low",  # Head-only should fail
            "attention_pattern": "near-uniform",
            "coefficient_template": "harmonic",
        },
    },
    "H2": {
        "name": "Learned Prefix Kernel",
        "description": "Block2 attention learns non-uniform causal kernel (e.g., ramp/decay) "
        "that generates position-dependent statistic",
        "predictions": {
            "R2_performance": "high",
            "R3_performance": "low",
            "R4_performance": "low",
            "attention_pattern": "structured-kernel",
            "coefficient_template": "power-law or exponential",
        },
    },
    "H3": {
        "name": "Token-ID Leakage / Shortcut",
        "description": "Performance from e_i (token identity) rather than prefix geometry. "
        "Uses dataset artifacts like token frequency vs typical position",
        "predictions": {
            "R2_performance": "variable",
            "R3_performance": "moderate",
            "R4_performance": "moderate-high",  # Can exploit token identity
            "invariance_test": "fails",
        },
    },
    "H4": {
        "name": "Magnitude / Variance-Decay",
        "description": "Model uses variance-like magnitude statistics (e.g., ||prefix mean||) "
        "to infer position. LN-sensitive.",
        "predictions": {
            "norm_probe_r2": "high",
            "ln_ablation_effect": "catastrophic",
            "R2_vs_R3": "similar if magnitude-based",
        },
    },
}


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
    data = train_data if split == "train" else val_data

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
    model: TwoLayerMechanismModel, config: ExperimentConfig, device: str, ctx
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single batch to capture attention weights from both blocks."""
    was_training = model.training
    model.eval()

    x, targets = get_batch("val", config, device)
    with torch.no_grad():
        with ctx:
            model(x, targets, capture_taps=False)

    attn1, attn2 = model.get_attention_weights()
    if was_training:
        model.train()
    return attn1.detach().cpu(), attn2.detach().cpu()


def train_regime(regime: str, config: ExperimentConfig, args) -> Dict:
    """Train a single regime and return results."""

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
    if config.head_on_post_attn:
        model.set_post_attn_head(True)

    # Apply freezing regime
    if regime == "R0":
        model.apply_regime_R0()
    elif regime == "R1":
        model.apply_regime_R1()
    elif regime == "R2":
        if config.r2_attn_head_only:
            model.apply_regime_R2_attn_head_only()
        else:
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

    # Initialize WandB
    if config.wandb_log:
        import wandb

        # Detailed run name
        run_name = f"{regime}_{config.experiment_name}"

        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                **asdict(config),
                "regime": regime,
                "hypotheses_tested": ["H1", "H2", "H3", "H4"],
            },
            tags=[regime, "exp1", "mechanism_dissection"],
            notes=f"Regime {regime}: Testing hypotheses H1-H4 via freeze/train ablation",
        )

        # Log hypothesis descriptions
        wandb.run.summary["hypotheses"] = HYPOTHESES
        regime_description = {
            "R0": "Full training baseline - all parameters trainable",
            "R1": "Block2-only - freeze Emb, Attn1, MLP1, train Attn2, MLP2, Head",
            "R2": "Attn2-only - freeze Emb, Attn1, MLP1, MLP2, train Attn2, Head",
            "R3": "MLP2-only - freeze Emb, Attn1, MLP1, Attn2, train MLP2, Head",
            "R4_linear": "Head-only probe (linear) - freeze all, train linear head",
            "R4_mlp": "Head-only probe (MLP) - freeze all, train 2-layer MLP head",
        }[regime]
        if regime == "R2" and config.r2_attn_head_only:
            regime_description = "Attn2-only (strict) - train Block2.Attn + Pos Head; full block2 forward"
        wandb.run.summary["regime_description"] = regime_description

        regime_out_dir = os.path.join(config.out_dir, regime, wandb.run.id)
    else:
        regime_out_dir = os.path.join(config.out_dir, regime)

    # Setup output directory
    os.makedirs(regime_out_dir, exist_ok=True)

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Training Regime: {regime}")
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
                    metrics["val_per_pos_mae"], f"Per-Position MAE ({regime})", iter_num
                )
                log_dict["val/per_pos_mae_plot"] = wandb.Image(fig)
                plt.close(fig)

                attn1, attn2 = sample_attention_maps(raw_model, config, device, ctx)
                fig_attn1 = create_attention_grid(
                    attn1, f"Block1 Attention (iter {iter_num})"
                )
                log_dict["val/attn_block1"] = wandb.Image(fig_attn1)
                plt.close(fig_attn1)

                fig_attn2 = create_attention_grid(
                    attn2, f"Block2 Attention (iter {iter_num})"
                )
                log_dict["val/attn_block2"] = wandb.Image(fig_attn2)
                plt.close(fig_attn2)

                fig_bos = create_bos_head_bar(
                    attn2, f"BOS Attention by Head (iter {iter_num})"
                )
                log_dict["val/bos_head_bar"] = wandb.Image(fig_bos)
                plt.close(fig_bos)

                # Log hypothesis-relevant metrics
                log_dict["hypothesis/regime"] = regime
                log_dict["hypothesis/val_r2_for_comparison"] = metrics["val_r2"]

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
    print("EXPERIMENT 1: MECHANISM DISSECTION VIA FREEZE/TRAIN MATRIX")
    print("=" * 80)
    print("\nRunning all regimes to test hypotheses H1, H2, H3, H4")
    print("Hypotheses:")
    for h_name, h_info in HYPOTHESES.items():
        print(f"  {h_name}: {h_info['name']}")
    print("=" * 80 + "\n")

    for regime in regimes:
        print(f"\n{'=' * 60}")
        print(f"STARTING REGIME: {regime}")
        print(f"{'=' * 60}")

        results = train_regime(regime, config, args)
        all_results[regime] = results

    # Create summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Regime':<15} {'Val MAE':<12} {'Val R²':<12} {'Val Loss':<12}")
    print("-" * 51)

    for regime in regimes:
        r = all_results[regime]["final_metrics"]
        print(
            f"{regime:<15} {r['val_mae']:<12.4f} {r['val_r2']:<12.4f} {r['val_loss']:<12.4f}"
        )

    print("-" * 51)

    # Hypothesis interpretation
    print("\n" + "=" * 80)
    print("HYPOTHESIS INTERPRETATION")
    print("=" * 80)

    r0_r2 = all_results["R0"]["final_metrics"]["val_r2"]
    r1_r2 = all_results["R1"]["final_metrics"]["val_r2"]
    r2_r2 = all_results["R2"]["final_metrics"]["val_r2"]
    r3_r2 = all_results["R3"]["final_metrics"]["val_r2"]
    r4l_r2 = all_results["R4_linear"]["final_metrics"]["val_r2"]
    r4m_r2 = all_results["R4_mlp"]["final_metrics"]["val_r2"]

    print(f"\nKey comparisons:")
    print(f"  R0 (full) R² = {r0_r2:.4f}")
    print(f"  R1 (block2-only) R² = {r1_r2:.4f}")
    print(f"  R2 (attn2-only) R² = {r2_r2:.4f}")
    print(f"  R3 (mlp2-only) R² = {r3_r2:.4f}")
    print(f"  R4_linear (head-only) R² = {r4l_r2:.4f}")
    print(f"  R4_mlp (mlp-head-only) R² = {r4m_r2:.4f}")

    # Decision logic
    print("\n" + "-" * 40)
    print("Decision Logic:")

    # H1/H2: Attention creates structure
    if r2_r2 > 0.7 and r3_r2 < 0.3:
        print("✓ R2 >> R3: Attn2 is necessary → H1 or H2 likely")
    elif r2_r2 < 0.5 and r3_r2 > 0.5:
        print("✓ R3 >> R2: MLP2 sufficient → Neither H1 nor H2")

    # H3: Token shortcut
    if r4l_r2 > 0.5 or r4m_r2 > 0.5:
        print("⚠ R4 shows reasonable performance → H3 (shortcut) possible")
    else:
        print("✓ R4 fails → Signal not already present in block1 output")

    # Save summary
    summary_path = os.path.join(config.out_dir, "experiment1_summary.json")
    summary = {
        "regimes": {
            regime: {
                "val_mae": all_results[regime]["final_metrics"]["val_mae"],
                "val_r2": all_results[regime]["final_metrics"]["val_r2"],
                "val_loss": all_results[regime]["final_metrics"]["val_loss"],
            }
            for regime in regimes
        },
        "hypotheses": HYPOTHESES,
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
        wandb_log=args.wandb,
        compile_model=not args.no_compile,
        out_dir=args.out_dir,
        device=args.device,
        head_on_post_attn=args.head_on_post_attn,
        r2_attn_head_only=args.r2_attn_head_only,
        use_bos=not args.no_bos,
    )

    # Load data
    load_data(config)

    print(f"\n*** BOS token (ID {config.bos_token_id}) is inserted at position 0 ***\n")

    # Run experiments
    if args.regime == "all":
        results = run_all_regimes(config, args)
    else:
        results = train_regime(args.regime, config, args)

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
