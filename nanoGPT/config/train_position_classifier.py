# Configuration for Position Classifier - Full Training
# Trains entire model end-to-end for position prediction

# I/O
out_dir = "out-position-classifier-full"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "position-classifier"
wandb_run_name = "nope-1layer-posclf-full"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 128  # Sequence length = position classes

# Model
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE - no positional embeddings
use_positional_embedding = False
norm_type = "layernorm"

# Training mode
freeze_transformer = False  # Full training

# Optimizer
learning_rate = 6e-4
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 6e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = True
seed = 42
