# Configuration for Position Classifier - 12 Layer (GPT-2 scale)
# Full-scale model for position prediction

# I/O
out_dir = "out-posclf-12layer"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "position-classifier"
wandb_run_name = "nope-12layer-posclf"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 8  # More accumulation for larger model
batch_size = 32
block_size = 128

# Model - 12 layers (GPT-2 scale)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"

# Training
freeze_transformer = False
use_regression = False  # Classification

learning_rate = 2e-4  # Lower LR for larger model
max_iters = 30000     # More iterations
weight_decay = 0.1
warmup_iters = 2000
lr_decay_iters = 30000
min_lr = 2e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = True
seed = 42
