# Configuration for Position Regression - 4 Layer - MLP Only Training
# Freeze embeddings + attention, train MLPs + position head
# Uses regression loss (MSE) and computes LM perplexity

# I/O
out_dir = "out-posreg-4layer-mlp"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "position-regression"
wandb_run_name = "nope-4layer-mlp-only"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 128

# Model - 4 layers
n_layer = 4
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"

# Training mode
freeze_transformer = False
freeze_attention_only = True  # Freeze embeddings + attention, train MLPs
use_regression = True         # MSE loss
compute_lm_loss = True        # Also compute LM perplexity

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
compile = True
seed = 42
