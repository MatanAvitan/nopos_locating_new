"""
Training script for Position Classification model.

Trains a transformer to predict absolute position (0 to block_size-1)
at each position in the sequence.

Usage:
    python train_position_classifier.py config/train_position_classifier.py
    python train_position_classifier.py config/train_position_classifier_frozen.py
"""

import os
import sys
import time
import math
import pickle
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_position_classifier import GPTPositionClassifierConfig, GPTPositionClassifier

# -----------------------------------------------------------------------------
# Default config values - will be overridden by config file
# -----------------------------------------------------------------------------

# I/O
out_dir = "out-position-classifier"
eval_interval = 500
log_interval = 50
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# Eval-only LM checkpoint logging
lm_eval_ckpt_dir = ""
lm_eval_ckpt_steps = []
lm_eval_wandb_run_id = ""
lm_eval_wandb_project = "nope-lm"

# Wandb logging
wandb_log = True
wandb_project = "position-classifier"
wandb_run_name = "nope-1layer-posclf"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 128  # Sequence length = number of position classes

# Model
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE-specific
use_positional_embedding = False
norm_type = "layernorm"

# Training mode
freeze_transformer = False  # If True, only train position head (probing)
freeze_attention_only = False  # If True, freeze embeddings + attention, train MLPs
freeze_until_first_mlp = False  # If True, freeze embeddings + first block's attn/norms, train from first MLP onwards
use_regression = False  # If True, use MSE loss instead of classification
compute_lm_loss = False  # If True, also compute LM perplexity
train_lm_only = False  # If True, train on LM loss instead of position loss
use_ln2 = True  # If False, remove ln_2 from transformer blocks
mlp_ratio = 4  # Expansion ratio for MLP hidden dimension

# AdamW optimizer
learning_rate = 6e-4
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 6e-5

# DDP settings
backend = "nccl"

# System
device = "cuda"
dtype = "bfloat16"
compile = True

# Reproducibility
seed = 42

# -----------------------------------------------------------------------------
# Parse config file
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

# Set seed for reproducibility
torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
data_dir = os.path.join("data", dataset)

# Keep train data accessible for distinctive token evaluation
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    """
    Load a batch of data for position classification.

    Returns:
        x: Token IDs [batch_size, block_size]
        pos_targets: Position labels [batch_size, block_size], values 0 to block_size-1
        lm_targets: Next token targets [batch_size, block_size] (or None if compute_lm_loss=False)
    """
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    # Position labels: each position should predict its own index
    pos_targets = torch.arange(block_size).unsqueeze(0).expand(batch_size, -1)

    # LM targets: next token prediction (shifted by 1)
    if compute_lm_loss:
        lm_targets = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + block_size + 1]).astype(np.int64))
                for i in ix
            ]
        )
    else:
        lm_targets = None

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        pos_targets = pos_targets.to(device)
        if lm_targets is not None:
            lm_targets = lm_targets.pin_memory().to(device, non_blocking=True)
    else:
        x, pos_targets = x.to(device), pos_targets.to(device)
        if lm_targets is not None:
            lm_targets = lm_targets.to(device)

    return x, pos_targets, lm_targets


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

# Get vocab size from dataset metadata
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Build model args
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    use_positional_embedding=use_positional_embedding,
    norm_type=norm_type,
    use_regression=use_regression,
    compute_lm_loss=compute_lm_loss,
    use_ln2=use_ln2,
    mlp_expansion_ratio=mlp_ratio,
)

if init_from == "scratch":
    print("Initializing a new Position Classifier model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2: 50304")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTPositionClassifierConfig(**model_args)
    model = GPTPositionClassifier(gptconf)

elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    gptconf = GPTPositionClassifierConfig(**model_args)
    model = GPTPositionClassifier(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

model.to(device)

# Freeze transformer if in probing mode
if freeze_transformer:
    print("\n*** FROZEN MODE: Only training position head ***")
    model.freeze_transformer()
elif freeze_attention_only:
    model.freeze_attention_only()
elif freeze_until_first_mlp:
    model.freeze_until_first_mlp()

# -----------------------------------------------------------------------------
# Optimizer and training setup
# -----------------------------------------------------------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)

if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

# Compile model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP wrapper
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss_and_accuracy():
    """Estimate loss and accuracy on train and val splits."""
    out = {}
    model.eval()
    raw = model.module if ddp else model

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        lm_losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        per_position_correct = torch.zeros(block_size)
        per_position_total = torch.zeros(block_size)
        mae_sum = 0.0  # For regression

        for k in range(eval_iters):
            X, Y, LM_Y = get_batch(split)
            with ctx:
                output, pos_loss, lm_loss = model(X, Y, LM_Y)
            losses[k] = pos_loss.item() if pos_loss is not None else 0.0
            if lm_loss is not None:
                lm_losses[k] = lm_loss.item()

            if use_regression:
                # Regression: compute MAE and "accuracy" as within-threshold
                preds_normalized = torch.sigmoid(output)  # [batch, seq_len]
                preds = preds_normalized * (
                    block_size - 1
                )  # Scale back to [0, block_size-1]
                mae = (preds - Y.float()).abs().mean().item()
                mae_sum += mae
                # Count "correct" as prediction within 1 position
                correct = ((preds - Y.float()).abs() < 1.0).float()
            else:
                # Classification: argmax predictions
                preds = output.argmax(dim=-1)  # [batch, seq_len]
                correct = (preds == Y).float()

            accuracies[k] = correct.mean().item()

            # Per-position accuracy
            for pos in range(block_size):
                per_position_correct[pos] += correct[:, pos].sum().item()
                per_position_total[pos] += X.size(0)

        if train_lm_only and compute_lm_loss:
            out[f"{split}_loss"] = lm_losses.mean().item()
        else:
            out[f"{split}_loss"] = losses.mean().item()
        out[f"{split}_accuracy"] = accuracies.mean().item()
        out[f"{split}_per_pos_accuracy"] = (
            (per_position_correct / per_position_total).cpu().numpy()
        )
        if use_regression:
            out[f"{split}_mae"] = mae_sum / eval_iters
        if compute_lm_loss:
            out[f"{split}_lm_loss"] = lm_losses.mean().item()
            out[f"{split}_lm_perplexity"] = math.exp(out[f"{split}_lm_loss"])

    model.train()
    return out


def _load_checkpoint_weights(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    unwrapped = {}
    for key, value in state_dict.items():
        unwrapped[key[10:] if key.startswith("_orig_mod.") else key] = value
    raw = model.module if ddp else model
    raw.load_state_dict(unwrapped, strict=False)
    return checkpoint


def evaluate_lm_checkpoints():
    if not lm_eval_ckpt_dir:
        return

    if not compute_lm_loss:
        raise ValueError("compute_lm_loss must be True for LM evaluation")

    ckpt_dir = lm_eval_ckpt_dir
    steps = lm_eval_ckpt_steps
    if not steps:
        ckpt_files = sorted(Path(ckpt_dir).glob("ckpt_*.pt"))
        steps = [int(path.stem.split("_")[-1]) for path in ckpt_files]

    if master_process and wandb_log and lm_eval_wandb_run_id:
        import wandb

        wandb.init(
            project=lm_eval_wandb_project,
            id=lm_eval_wandb_run_id,
            resume="allow",
            name=wandb_run_name,
            config=config,
        )
        wandb.define_metric("eval/ckpt_step")
        wandb.define_metric("eval/*", step_metric="eval/ckpt_step")

    for step in sorted(steps):
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_{step:05d}.pt")
        if not os.path.exists(ckpt_path):
            continue

        _load_checkpoint_weights(ckpt_path)
        metrics = estimate_loss_and_accuracy()

        if master_process:
            print(
                f"eval step {step}: val lm loss {metrics['val_lm_loss']:.4f}, "
                f"ppl {metrics['val_lm_perplexity']:.1f}"
            )

        if master_process and wandb_log and lm_eval_wandb_run_id:
            import wandb

            wandb.log(
                {
                    "eval/ckpt_step": step,
                    "eval/val_lm_loss": metrics["val_lm_loss"],
                    "eval/val_lm_perplexity": metrics["val_lm_perplexity"],
                    "eval/train_lm_loss": metrics["train_lm_loss"],
                    "eval/train_lm_perplexity": metrics["train_lm_perplexity"],
                }
            )

    if master_process and wandb_log and lm_eval_wandb_run_id:
        import wandb

        wandb.finish()


def get_lr(it):
    """Learning rate schedule with warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Distinctive token evaluation for gradient graph
# -----------------------------------------------------------------------------

# Global tracking for gradient graph
distinctive_counts = [1, 2, 4, 8, 16, 32, 64, 128]
accuracy_history = {n: [] for n in distinctive_counts}
iteration_history = []


def get_distinctive_tokens_from_owt(n_distinctive, n_samples, seq_len):
    """
    Generate sequences using actual OWT tokens from the vocabulary.
    Uses the first n_distinctive unique tokens found in OWT data.
    """
    # Find n_distinctive unique tokens from OWT data
    unique_tokens = []
    for i in range(min(len(train_data), 100000)):  # Search first 100K tokens
        token = int(train_data[i])
        if token not in unique_tokens:
            unique_tokens.append(token)
        if len(unique_tokens) >= n_distinctive:
            break

    # Generate sequences cycling through these tokens
    tokens = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        for j in range(seq_len):
            tokens[i, j] = unique_tokens[j % n_distinctive]

    return tokens


@torch.no_grad()
def evaluate_distinctive_tokens(n_samples=500):
    """Evaluate position accuracy for each distinctive token count."""
    model.eval()
    results = {}

    for n_dist in distinctive_counts:
        if n_dist > block_size:
            continue

        tokens = get_distinctive_tokens_from_owt(n_dist, n_samples, block_size)
        tokens = tokens.to(device)
        targets = torch.arange(block_size).unsqueeze(0).expand(n_samples, -1).to(device)

        # Process in batches
        all_correct = []
        eval_batch_size = min(100, n_samples)
        for i in range(0, n_samples, eval_batch_size):
            batch_tokens = tokens[i : i + eval_batch_size]
            batch_targets = targets[i : i + eval_batch_size]

            with ctx:
                output, _, _ = model(batch_tokens, batch_targets)

            if use_regression:
                # Regression: compute accuracy as predictions within threshold
                preds_scaled = torch.sigmoid(output) * (block_size - 1)
                correct = ((preds_scaled - batch_targets.float()).abs() < 1.0).float()
            else:
                # Classification: argmax predictions
                preds = output.argmax(dim=-1)
                correct = (preds == batch_targets).float()

            all_correct.append(correct)

        all_correct = torch.cat(all_correct, dim=0)
        mean_accuracy = all_correct.mean().item()
        results[n_dist] = mean_accuracy

    model.train()
    return results


def log_distinctive_accuracy_to_wandb(iter_num, results):
    """Log distinctive token accuracy with aggregated summary plot."""
    import wandb

    iteration_history.append(iter_num)
    for n_dist, acc in results.items():
        accuracy_history[n_dist].append(acc)

    # Create aggregated summary plot (bar chart with mean accuracy per token count)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [str(n) for n in distinctive_counts if n in results]
    means = [results[n] for n in distinctive_counts if n in results]

    # Bar plot with mean accuracy
    bars = ax.bar(x_labels, means, color="steelblue", alpha=0.8)
    ax.set_xlabel("Number of Distinctive Tokens", fontsize=12)
    ax.set_ylabel("Mean Position Accuracy", fontsize=12)
    ax.set_title(f"Position Accuracy by Token Diversity (iter {iter_num})", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Log aggregated summary
    log_dict = {
        "distinctive_accuracy_summary": wandb.Image(fig),
        "distinctive/mean_accuracy": np.mean(means) if means else 0,
    }
    for n_dist, acc in results.items():
        log_dict[f"distinctive/{n_dist}_tokens_acc"] = acc

    wandb.log(log_dict, step=iter_num)
    plt.close(fig)

    # Also create gradient graph over time
    if len(iteration_history) > 1:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(distinctive_counts)))
        for i, n_dist in enumerate(distinctive_counts):
            if n_dist in accuracy_history and len(accuracy_history[n_dist]) > 0:
                ax2.plot(
                    iteration_history,
                    accuracy_history[n_dist],
                    color=colors[i],
                    linewidth=2,
                    label=f"{n_dist} tokens",
                )
        ax2.set_xlabel("Training Iteration", fontsize=12)
        ax2.set_ylabel("Mean Position Accuracy", fontsize=12)
        ax2.set_title("Position Accuracy by Token Diversity Over Training", fontsize=14)
        ax2.legend(title="Distinctive Tokens", loc="lower right")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        wandb.log({"distinctive_accuracy_gradient": wandb.Image(fig2)}, step=iter_num)
        plt.close(fig2)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

if wandb_log and master_process and not (eval_only and lm_eval_ckpt_dir):
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.run.summary["training_details"] = (
        f"dataset={dataset}, block_size={block_size}, n_layer={n_layer}, n_head={n_head}, "
        f"n_embd={n_embd}, freeze_until_first_mlp={freeze_until_first_mlp}, "
        f"train_lm_only={train_lm_only}, compute_lm_loss={compute_lm_loss}"
    )

X, Y, LM_Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model

print(f"\nStarting training for {max_iters} iterations...")
print(f"  Model: Position Classifier")
print(f"  Positional Embedding: {use_positional_embedding}")
print(f"  Frozen transformer: {freeze_transformer}")
print(f"  Freeze attention only: {freeze_attention_only}")
print(f"  Freeze until first MLP: {freeze_until_first_mlp}")
print(f"  Use regression: {use_regression}")
print(f"  Compute LM loss: {compute_lm_loss}")
print(f"  Train LM only: {train_lm_only}")
print(f"  Dataset: {dataset}")
print(
    f"  Batch size: {batch_size} x {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}"
)
print(f"  Block size: {block_size}")
print(f"  Position classes: {block_size}")
print(f"  Learning rate: {learning_rate} -> {min_lr}")
print()

if eval_only and lm_eval_ckpt_dir:
    evaluate_lm_checkpoints()
    if ddp:
        destroy_process_group()
    sys.exit(0)

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation and checkpointing
    if iter_num % eval_interval == 0 and master_process:
        metrics = estimate_loss_and_accuracy()
        ppl_str = (
            f", ppl {metrics.get('val_lm_perplexity', 0):.1f}"
            if compute_lm_loss
            else ""
        )
        if train_lm_only and compute_lm_loss:
            print(
                f"step {iter_num}: train lm loss {metrics['train_loss']:.4f}, "
                f"val lm loss {metrics['val_loss']:.4f}{ppl_str}, lr {lr:.2e}"
            )
        else:
            print(
                f"step {iter_num}: train loss {metrics['train_loss']:.4f}, "
                f"val loss {metrics['val_loss']:.4f}, "
                f"train acc {metrics['train_accuracy']:.4f}, "
                f"val acc {metrics['val_accuracy']:.4f}{ppl_str}, lr {lr:.2e}"
            )

        # Distinctive token evaluation
        distinctive_results = evaluate_distinctive_tokens()
        print(
            f"  Distinctive accuracy: "
            + ", ".join(
                [
                    f"{n}t:{distinctive_results.get(n, 0):.3f}"
                    for n in [1, 2, 4, 8, 16, 32, 64, 128]
                    if n in distinctive_results
                ]
            )
        )

        if wandb_log:
            import wandb

            log_dict = {
                "iter": iter_num,
                "train/loss": metrics["train_loss"],
                "val/loss": metrics["val_loss"],
                "lr": lr,
            }

            if not train_lm_only:
                log_dict["train/accuracy"] = metrics["train_accuracy"]
                log_dict["val/accuracy"] = metrics["val_accuracy"]

            # Log LM perplexity if enabled
            if compute_lm_loss:
                log_dict["val/lm_loss"] = metrics["val_lm_loss"]
                log_dict["val/lm_perplexity"] = metrics["val_lm_perplexity"]
                log_dict["train/lm_loss"] = metrics["train_lm_loss"]
                log_dict["train/lm_perplexity"] = metrics["train_lm_perplexity"]

            if not train_lm_only:
                # Log per-position accuracy as a line plot
                for pos in range(block_size):
                    log_dict[f"val/pos_accuracy/{pos}"] = metrics[
                        "val_per_pos_accuracy"
                    ][pos]

                # Log summary stats
                log_dict["val/pos_accuracy_mean"] = metrics[
                    "val_per_pos_accuracy"
                ].mean()
                log_dict["val/pos_accuracy_std"] = metrics["val_per_pos_accuracy"].std()
                log_dict["val/pos_accuracy_min"] = metrics["val_per_pos_accuracy"].min()
                log_dict["val/pos_accuracy_max"] = metrics["val_per_pos_accuracy"].max()

            wandb.log(log_dict, step=iter_num)

            # Log distinctive token evaluation with gradient graph
            if not train_lm_only:
                log_distinctive_accuracy_to_wandb(iter_num, distinctive_results)

        # Save checkpoint
        val_loss = metrics["val_loss"]
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, val_loss)
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "val_accuracy": metrics["val_accuracy"],
                    "distinctive_results": distinctive_results,
                }
                ckpt_path = os.path.join(out_dir, f"ckpt_{iter_num:05d}.pt")
                print(f"saving checkpoint to {ckpt_path}")
                torch.save(checkpoint, ckpt_path)
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # Forward/backward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, pos_loss, lm_loss = model(X, Y, LM_Y)
            # Choose loss based on training mode
            if train_lm_only:
                # Use LM loss for training (next token prediction)
                loss = lm_loss / gradient_accumulation_steps
            else:
                # Use position loss for training (primary objective)
                loss = pos_loss / gradient_accumulation_steps
        X, Y, LM_Y = get_batch("train")
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

# Cleanup
if ddp:
    destroy_process_group()

if master_process:
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}/")
