"""
Training script for NoPE GPT with BOS Q/K Frozen in Block 1.

This variant does NOT freeze any attention in block 0. Instead, it constrains a
single head in block 1 (second block) to be a BOS head by freezing the W_Q and
W_K weights for that head. W_V remains trainable.

The frozen W_Q/W_K slices are copied from a trained 2-layer position regression
model that converged to a BOS head in block 2.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_bos_qk_frozen_lm.py config/train_lm_6layer_bos_qk_frozen.py
    torchrun --nproc_per_node=2 train_bos_qk_frozen_lm.py config/train_lm_6layer_bos_qk_frozen.py
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
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# -----------------------------------------------------------------------------
# Default config values
# -----------------------------------------------------------------------------

# I/O
out_dir = "out-lm-6layer-bos-qk-frozen"
eval_interval = 500
log_interval = 50
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-bos-qk-frozen"

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
log_attention_stats = False

# BOS Q/K frozen configuration
bos_qk_source_checkpoint = "out-2layer-mechanism/R0/best_ckpt.pt"
bos_qk_source_block = 1
bos_qk_head_idx = None  # Auto-select BOS head from source model
bos_head_block = 1  # Block index to freeze Q/K in this model

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
# Helpers for BOS Q/K extraction
# -----------------------------------------------------------------------------


def get_batch_for_bos(data: np.ndarray, batch: int, length: int, device: str):
    ix = torch.randint(len(data) - length, (batch,))
    tokens = torch.stack(
        [torch.from_numpy((data[i : i + length]).astype(np.int64)) for i in ix]
    )
    return tokens.to(device)


def compute_bos_scores(
    model: TwoLayerMechanismModel,
    tokens: torch.Tensor,
    block_idx: int,
) -> list[dict]:
    model.eval()
    with torch.no_grad():
        tok_emb = model.wte(tokens)
        hidden = model.drop(tok_emb)
        if block_idx == 0:
            ln1_out = model.block1.ln_1(hidden)
            qkv = model.block1.attn.c_attn(ln1_out)
        else:
            hidden = model.block1(hidden, capture_taps=False)
            ln1_out = model.block2.ln_1(hidden)
            qkv = model.block2.attn.c_attn(ln1_out)

        n_embd = model.config.n_embd
        n_head = model.config.n_head
        head_dim = n_embd // n_head
        q, k, _ = qkv.split(n_embd, dim=2)
        k = k.view(tokens.size(0), tokens.size(1), n_head, head_dim).transpose(1, 2)
        q = q.view(tokens.size(0), tokens.size(1), n_head, head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        attn = torch.softmax(logits, dim=-1)

    scores = []
    for head_idx in range(attn.shape[1]):
        bos_mean = attn[:, head_idx, :, 0].mean().item()
        other_mean = attn[:, head_idx, :, 1:].mean().item()
        bos_logit = logits[:, head_idx, :, 0].mean().item()
        other_logit = logits[:, head_idx, :, 1:].mean().item()
        score = abs(bos_logit - 1.0) + abs(other_logit)
        scores.append(
            {
                "head_idx": head_idx,
                "bos_mean": bos_mean,
                "other_mean": other_mean,
                "bos_logit": bos_logit,
                "other_logit": other_logit,
                "score": score,
            }
        )
    scores.sort(key=lambda x: x["score"])
    return scores


def load_bos_qk_from_posreg(
    checkpoint_path: str,
    device: str,
    eval_length: int = 256,
    batch: int = 4,
    head_idx: int | None = None,
    block_idx: int = 1,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if head_idx is None:
        raise ValueError("bos_qk_head_idx must be set (use 6 or 9 as BOS heads)")

    ckpt_has_bias = any(key.endswith("bias") for key in ckpt["model"].keys())
    config = TwoLayerMechanismConfig(
        block_size=eval_length,
        n_embd=768,
        n_head=12,
        vocab_size=50304,
        dropout=0.0,
        bias=ckpt_has_bias,
        norm_type="layernorm",
        use_regression=True,
    )
    model = TwoLayerMechanismModel(config).to(device)
    model.load_state_dict(ckpt["model"])

    data_path = os.path.join("data", dataset, "train.bin")
    train_data = np.memmap(data_path, dtype=np.uint16, mode="r")
    tokens = get_batch_for_bos(train_data, batch, eval_length, device)

    scores = compute_bos_scores(model, tokens, block_idx=block_idx)
    chosen = next(s for s in scores if s["head_idx"] == head_idx)

    head_idx = chosen["head_idx"]
    head_dim = config.n_embd // config.n_head
    q_start = head_idx * head_dim
    q_end = q_start + head_dim
    k_start = config.n_embd + q_start
    k_end = config.n_embd + q_end

    if block_idx == 0:
        weight = model.block1.attn.c_attn.weight.detach().clone()
        bias = model.block1.attn.c_attn.bias
    else:
        weight = model.block2.attn.c_attn.weight.detach().clone()
        bias = model.block2.attn.c_attn.bias
    q_weight = weight[q_start:q_end, :].clone()
    k_weight = weight[k_start:k_end, :].clone()
    q_bias = None
    k_bias = None
    if bias is not None:
        q_bias = bias[q_start:q_end].detach().clone()
        k_bias = bias[k_start:k_end].detach().clone()

    return {
        "head_idx": head_idx,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "bos_mean": chosen["bos_mean"],
        "other_mean": chosen["other_mean"],
        "bos_logit": chosen["bos_logit"],
        "other_logit": chosen["other_logit"],
        "score": chosen["score"],
    }


def apply_fixed_qk(model, block_idx, head_idx, q_weight, k_weight, q_bias, k_bias):
    c_attn = model.transformer.h[block_idx].attn.c_attn
    head_dim = model.config.n_embd // model.config.n_head
    q_start = head_idx * head_dim
    q_end = q_start + head_dim
    k_start = model.config.n_embd + q_start
    k_end = model.config.n_embd + q_end

    with torch.no_grad():
        c_attn.weight[q_start:q_end, :] = q_weight
        c_attn.weight[k_start:k_end, :] = k_weight
        if c_attn.bias is not None and q_bias is not None and k_bias is not None:
            c_attn.bias[q_start:q_end] = q_bias
            c_attn.bias[k_start:k_end] = k_bias


def register_qk_freeze_hooks(model, block_idx, head_idx):
    c_attn = model.transformer.h[block_idx].attn.c_attn
    head_dim = model.config.n_embd // model.config.n_head
    q_start = head_idx * head_dim
    q_end = q_start + head_dim
    k_start = model.config.n_embd + q_start
    k_end = model.config.n_embd + q_end

    def weight_hook(grad):
        grad = grad.clone()
        grad[q_start:q_end, :] = 0
        grad[k_start:k_end, :] = 0
        return grad

    def bias_hook(grad):
        grad = grad.clone()
        grad[q_start:q_end] = 0
        grad[k_start:k_end] = 0
        return grad

    c_attn.weight.register_hook(weight_hook)
    if c_attn.bias is not None:
        c_attn.bias.register_hook(bias_hook)


def verify_bos_attention(model, block_idx, head_idx, tokens):
    model.eval()
    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)
        hidden = model.transformer.drop(tok_emb)
        for block in model.transformer.h[:block_idx]:
            hidden = block(hidden)
        ln1_out = model.transformer.h[block_idx].ln_1(hidden)
        qkv = model.transformer.h[block_idx].attn.c_attn(ln1_out)
        n_embd = model.config.n_embd
        n_head = model.config.n_head
        head_dim = n_embd // n_head
        q, k, _ = qkv.split(n_embd, dim=2)
        k = k.view(tokens.size(0), tokens.size(1), n_head, head_dim).transpose(1, 2)
        q = q.view(tokens.size(0), tokens.size(1), n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        causal_mask = torch.triu(
            torch.ones(
                tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        head_att = att[:, head_idx]
        bos_mean = head_att[:, :, 0].mean().item()
        other_mean = head_att[:, :, 1:].mean().item()
    return bos_mean, other_mean


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
)

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

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Load BOS Q/K from posreg and freeze
bos_source = load_bos_qk_from_posreg(
    bos_qk_source_checkpoint,
    device,
    eval_length=min(256, block_size),
    batch=4,
    head_idx=bos_qk_head_idx,
    block_idx=bos_qk_source_block,
)

if master_process:
    print(
        f"BOS head selected from posreg: head {bos_source['head_idx']} | "
        f"bos_logit={bos_source['bos_logit']:.4f} other_logit={bos_source['other_logit']:.4f} | "
        f"bos_mean={bos_source['bos_mean']:.4f} other_mean={bos_source['other_mean']:.4f}"
    )

apply_fixed_qk(
    model,
    bos_head_block,
    bos_source["head_idx"],
    bos_source["q_weight"],
    bos_source["k_weight"],
    bos_source["q_bias"],
    bos_source["k_bias"],
)
register_qk_freeze_hooks(model, bos_head_block, bos_source["head_idx"])

# Verify BOS attention in this model
if master_process:
    sample_tokens = get_batch("train")[0][:4]
    bos_mean, other_mean = verify_bos_attention(
        model, bos_head_block, bos_source["head_idx"], sample_tokens
    )
    print(
        f"BOS check (block {bos_head_block} head {bos_source['head_idx']}): "
        f"bos_mean={bos_mean:.4f}, other_mean={other_mean:.4f}"
    )

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
    wandb.config.update(
        {
            "bos_head_block": bos_head_block,
            "bos_head_idx": bos_source["head_idx"],
            "bos_source_checkpoint": bos_qk_source_checkpoint,
            "bos_source_bos_mean": bos_source["bos_mean"],
            "bos_source_other_mean": bos_source["other_mean"],
        }
    )


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
        f"BOS Q/K frozen: Block {bos_head_block} Head {bos_source['head_idx']} "
        f"(W_Q/W_K frozen, W_V trainable)"
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
                "bos_head_block": bos_head_block,
                "bos_head_idx": bos_source["head_idx"],
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

    # Re-apply fixed Q/K slices to avoid weight decay updates
    apply_fixed_qk(
        raw_model,
        bos_head_block,
        bos_source["head_idx"],
        bos_source["q_weight"],
        bos_source["k_weight"],
        bos_source["q_bias"],
        bos_source["k_bias"],
    )

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
