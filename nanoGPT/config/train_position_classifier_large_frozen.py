# Configuration for Position Classifier - LARGE Model FROZEN
# More layers, but only train the position head

# I/O
out_dir = "out-position-classifier-large-frozen"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "position-classifier"
wandb_run_name = "nope-6layer-posclf-frozen"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 128  # Sequence length = position classes

# Model - LARGER
n_layer = 6        # 6 layers instead of 1
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE - no positional embeddings
use_positional_embedding = False
norm_type = "layernorm"

# Training mode
freeze_transformer = True  # FROZEN - only train position head

# Optimizer - higher LR for probing
learning_rate = 1e-3
max_iters = 20000
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 1e-4

# System
device = "cuda"
dtype = "bfloat16"
compile = True
seed = 42
