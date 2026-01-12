# Training config for BASELINE transformer WITH positional embeddings on OpenWebText
# Standard transformer baseline for comparison with NoPE variants
#
# Key features:
# - WITH positional embeddings (standard transformer)
# - WITH second LayerNorm (standard architecture)
# - OpenWebText dataset (~9B tokens)
# - Context length 2048 for fair comparison
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_baseline_owt_pe.py

# I/O
out_dir = "out-baseline-owt-pe"
eval_interval = 500  # Checkpoint every 500 iters for resumability
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-emergence"
wandb_run_name = "baseline-owt-pe"

# Data
dataset = "openwebtext"  # Full OpenWebText (~9B tokens)
gradient_accumulation_steps = (
    1  # No accumulation - need large batch for population statistics
)
batch_size = 128  # Large batch (128 * 1024 = 131K tokens)
block_size = 1024  # Context length for fair comparison

# Model architecture
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for clean analysis
bias = False  # Modern practice

# BASELINE settings - WITH positional embeddings
use_positional_embedding = True  # KEY: Standard positional embeddings
norm_type = "layernorm"  # Standard LayerNorm
log_attention_stats = True  # Log attention for comparison
skip_ln2 = False  # Standard architecture with LN2

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

# Reproducibility
seed = 42
