"""
Training script for NoPE GPT models.

This script is modified from nanoGPT's train.py to support:
- NoPE model (no positional embeddings)
- LayerNorm / RMSNorm variants
- Attention statistics logging during training
- Fixed seed for reproducibility

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_1layer_ln.py
    CUDA_VISIBLE_DEVICES=1 python train_nope.py config/train_nope_1layer_rms.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_nope import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default config values - will be overridden by config file
# -----------------------------------------------------------------------------

# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = False
wandb_project = "nope"
wandb_run_name = "nope-gpt"

# Data
dataset = "shakespeare"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 256

# Model
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE-specific
use_positional_embedding = False
norm_type = "layernorm"
log_attention_stats = True
skip_ln2 = False  # Skip second LayerNorm for ablation
use_batchnorm_ln2 = False  # Use BatchNorm instead of LayerNorm for ln_2

# AdamW optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 1e-4

# DDP settings
backend = "nccl"

# System
device = "cuda"
dtype = "bfloat16"
compile = True

# Reproducibility
seed = 42

# Stability tracking for no-LN2 experiments
track_stability = True  # Track gradient norms and loss variance
stability_log_interval = 100  # Log stability metrics every N iters

# -----------------------------------------------------------------------------
# Parse config file
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
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


def get_batch(split):
    """Load a batch of data from disk."""
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
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
    if device_type == "cuda":
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


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
    log_attention_stats=log_attention_stats,
    skip_ln2=skip_ln2,
    use_batchnorm_ln2=use_batchnorm_ln2,
)

if init_from == "scratch":
    print("Initializing a new NoPE model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2: 50304")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # Force critical config attributes to match checkpoint
    for k in [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
        "use_positional_embedding",
        "norm_type",
    ]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # Fix potential key prefix issues
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

# Crop block size if needed
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size

model.to(device)

# -----------------------------------------------------------------------------
# Optimizer and training setup
# -----------------------------------------------------------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)

if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # Free memory

# Compile model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP wrapper
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    """Learning rate schedule with warmup and cosine decay."""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def log_attention_statistics(raw_model, iter_num, wandb_log_enabled):
    """Log attention statistics (entropy, uniformity) for analysis."""
    attn_stats = raw_model.get_attention_stats()

    if not attn_stats:
        return

    for key, val in attn_stats.items():
        if "entropy" in key:
            # val is a tensor of shape [n_head]
            mean_entropy = val.mean().item()
            print(f"    {key}: mean={mean_entropy:.4f}")
            if wandb_log_enabled:
                import wandb

                wandb.log({f"{key}_mean": mean_entropy}, step=iter_num)
                for h, v in enumerate(val):
                    wandb.log({f"{key}_head_{h}": v.item()}, step=iter_num)

        elif "uniformity" in key:
            # val is a list of floats, one per head
            print(f"    {key}: {[f'{v:.3f}' for v in val]}")
            if wandb_log_enabled:
                import wandb

                for h, v in enumerate(val):
                    wandb.log({f"{key}_head_{h}": v}, step=iter_num)
                # Also log how many heads are "uniform" (corr > 0.9)
                n_uniform = sum(1 for v in val if v > 0.9)
                wandb.log({f"{key}_n_uniform_heads": n_uniform}, step=iter_num)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

print(f"\nStarting training for {max_iters} iterations...")
print(f"  Model: NoPE GPT with {norm_type}")
print(f"  Positional Embedding: {use_positional_embedding}")
print(f"  Dataset: {dataset}")
print(
    f"  Batch size: {batch_size} x {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}"
)
print(f"  Block size: {block_size}")
print(f"  Learning rate: {learning_rate} -> {min_lr}")
print()

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation and checkpointing
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}"
        )

        # Log attention statistics
        if log_attention_stats:
            log_attention_statistics(raw_model, iter_num, wandb_log)

        if wandb_log:
            import wandb

            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                },
                step=iter_num,
            )

        # Save checkpoint
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses["val"])
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                # Save with iteration number for emergence analysis
                ckpt_path = os.path.join(out_dir, f"ckpt_{iter_num:05d}.pt")
                print(f"saving checkpoint to {ckpt_path}")
                torch.save(checkpoint, ckpt_path)
                # Also save as latest
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
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Stability tracking (gradient norms, loss variance)
    if track_stability and iter_num % stability_log_interval == 0 and master_process:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"  [Stability] grad_norm: {total_norm:.4f}")
        if wandb_log:
            import wandb
            wandb.log({"stability/grad_norm": total_norm}, step=iter_num)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )

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
