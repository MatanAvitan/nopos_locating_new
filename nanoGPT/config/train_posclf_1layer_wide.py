# Configuration for Position Classifier - 1 Layer WIDE (higher dim, more heads)
# Tests if wider model helps with limited depth

# I/O
out_dir = "out-posclf-1layer-wide"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "position-classifier"
wandb_run_name = "nope-1layer-wide-1536d-24h"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4
batch_size = 32  # Smaller batch due to larger model
block_size = 128

# Model - 1 layer but WIDE
n_layer = 1
n_head = 24       # More heads
n_embd = 1536     # 2x embedding dimension
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"

# Training
freeze_transformer = False
use_regression = False  # Classification

learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 3e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = True
seed = 42
