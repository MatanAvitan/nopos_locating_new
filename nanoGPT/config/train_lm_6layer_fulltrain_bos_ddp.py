# Configuration for LM Training - 6 Layer - Full Training, BOS-prepended (P4)
# Identical to train_lm_6layer_fulltrain_ddp.py except every training window
# has a fixed BOS token (50256) at position 0, matching the toys' protocol.
# Single-GPU B200: batch_size 512 x grad_accum 2 = 1024 seqs/iter, same
# effective batch as the original 2-GPU DDP run.

# I/O
out_dir = "out-lm-6layer-fulltrain-bos"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-fulltrain-lm-bos"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 2
batch_size = 512
block_size = 128

# BOS anchor (P4)
prepend_bos = True
bos_token_id = 50256

# Model - 6 layers
n_layer = 6
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"

# Training mode - all weights trainable
freeze_transformer = False
freeze_attention_only = False
freeze_until_first_mlp = False
use_regression = False
compute_lm_loss = True
train_lm_only = True

# AdamW optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 3e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = False
seed = 42
