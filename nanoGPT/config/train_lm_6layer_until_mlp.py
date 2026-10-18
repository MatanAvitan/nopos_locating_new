# Configuration for LM Training - 6 Layer - Freeze Until First MLP
# Same architecture as position regression, but trains on next-token prediction loss
# This allows comparing LM training vs position regression with same frozen layers

# I/O
out_dir = "out-lm-6layer-until-mlp"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-until-first-mlp-lm"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 1
batch_size = 512
block_size = 128

# Model - 6 layers (same as position regression)
n_layer = 6
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"

# Training mode - KEY DIFFERENCE from position regression
freeze_transformer = False
freeze_attention_only = False
freeze_until_first_mlp = True  # Freeze embeddings + block0 attn/norms
use_regression = False  # Not regression mode
compute_lm_loss = True  # Enable LM head
train_lm_only = True  # Train on LM loss instead of position loss

# AdamW optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 3e-5

# System
device = "cuda"
dtype = "bfloat16"  # H100 supports bfloat16
compile = False  # Disable - torch.compile hangs on this system
seed = 42
