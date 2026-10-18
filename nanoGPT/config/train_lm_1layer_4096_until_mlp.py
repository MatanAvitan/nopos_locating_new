# Configuration for LM Training - 1 Layer - 4096 Dim - Freeze Until First MLP
# Same training mode as 6-layer LM, but larger width and single block

# I/O
out_dir = "out-lm-1layer-4096-until-mlp"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-1layer-4096-until-first-mlp-lm"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 8
batch_size = 8
block_size = 128

# Model - 1 layer
n_layer = 1
n_head = 32
n_embd = 4096
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"

# Training mode
freeze_transformer = False
freeze_attention_only = False
freeze_until_first_mlp = True
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
