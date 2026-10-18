"""
Training script for NoPE GPT with Forced BOS Mechanism.

This script trains a 6-layer NoPE transformer with hard-coded position encoding:
- Block 0, Head 0: Frozen to uniform causal attention (prefix averaging)
- Block 1, Head 0: Frozen to attend only to position 0 (BOS head)

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python train_forced_bos_lm.py config/train_lm_6layer_forced_bos.py

    # DDP (2 GPUs)
    torchrun --nproc_per_node=2 train_forced_bos_lm.py config/train_lm_6layer_forced_bos.py
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

from model_nope_forced_bos import GPTConfigForcedBOS, GPTForcedBOS

# -----------------------------------------------------------------------------
# Default config values
# -----------------------------------------------------------------------------

# I/O
out_dir = "out-lm-forced-bos"
eval_interval = 500
log_interval = 50
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-forced-bos-lm"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 2
batch_size = 512
block_size = 128

# Model
n_layer = 6
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE-specific
use_positional_embedding = False
norm_type = "layernorm"
log_attention_stats = True

# Forced BOS mechanism
use_forced_bos_model = True
freeze_uniform_head_idx = 0
freeze_bos_head_idx = 0
uniform_head_block = 0
bos_head_block = 1

# AdamW optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 3e-5

# DDP settings
backend = "nccl"

# System
device = "cuda"
dtype = "bfloat16"
compile = False
seed = 42

# -----------------------------------------------------------------------------
# Parse config file
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str, type(None)))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

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
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
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
# Init model
# -----------------------------------------------------------------------------

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
    freeze_uniform_head_idx=freeze_uniform_head_idx,
    freeze_bos_head_idx=freeze_bos_head_idx,
    uniform_head_block=uniform_head_block,
    bos_head_block=bos_head_block,
)

# Load meta vocab size if available
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304")
model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304

gptconf = GPTConfigForcedBOS(**model_args)
model = GPTForcedBOS(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# -----------------------------------------------------------------------------
# Training loop helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss():
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
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
iter_num = 0

if master_process:
    print(f"\nStarting training for {max_iters} iterations...")
    print(
        f"Forced BOS mechanism: Block {uniform_head_block} Head {freeze_uniform_head_idx} (uniform), "
        f"Block {bos_head_block} Head {freeze_bos_head_idx} (BOS)"
    )

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                }
            )
        if always_save_checkpoint:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

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

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss_batch": lossf,
                    "train/lr": lr,
                    "train/time_ms": dt * 1000,
                }
            )

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

if master_process:
    print("Training complete!")
    if wandb_log:
        wandb.finish()
