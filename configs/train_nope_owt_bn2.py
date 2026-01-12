# Training config for NoPE transformer with BatchNorm2 on OpenWebText
# Tests hypothesis: Does BatchNorm preserve population positional statistics?
#
# Key features:
# - No positional embeddings (NoPE mode)
# - BatchNorm for second normalization (instead of LayerNorm)
# - Xavier initialization for attention Q, K, V
# - OpenWebText dataset (~9B tokens)
# - Context length 2048 for testing long-range position encoding
#
# Hypothesis: LayerNorm normalizes per-sample, potentially destroying population
# statistics that encode position. BatchNorm normalizes across the batch,
# potentially preserving population-level mean/variance positional signals.
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_owt_bn2.py

# I/O
out_dir = "out-nope-owt-bn2"
eval_interval = 500  # Checkpoint every 500 iters for resumability
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-emergence"
wandb_run_name = "nope-owt-bn2"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 1  # No accumulation - BatchNorm needs large batch per forward for population statistics
batch_size = (
    128  # Large batch for BatchNorm population statistics (128 * 1024 = 131K tokens)
)
block_size = 1024  # Context length

# Model architecture
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE-specific settings
use_positional_embedding = False
norm_type = "layernorm"  # LN for first norm
log_attention_stats = True
skip_ln2 = False
use_batchnorm_ln2 = True  # KEY: Use BatchNorm for ln_2

# Optimizer
learning_rate = 6e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 2000

# System
device = "cuda"
compile = True
dtype = "bfloat16"
seed = 42

# Stability tracking
track_stability = True
stability_log_interval = 100
