# Training config for 12-layer BASELINE transformer WITH positional embeddings on OpenWebText
# Standard transformer baseline for comparison with NoPE variants
#
# Key features:
# - WITH positional embeddings (standard transformer)
# - WITH second LayerNorm (standard architecture)
# - 12 layers (GPT-2 small scale)
# - OpenWebText dataset (~9B tokens)
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_baseline_owt_pe_12layer.py

# I/O
out_dir = "out-baseline-owt-pe-12layer"
eval_interval = 500  # Checkpoint every 500 iters for resumability
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-emergence"
wandb_run_name = "baseline-owt-pe-12layer"

# Data
dataset = "openwebtext"  # Full OpenWebText (~9B tokens)
gradient_accumulation_steps = (
    1  # No accumulation - need large batch for population statistics
)
batch_size = 128  # Large batch (128 * 512 = 65K tokens)
block_size = 512  # Context length for fair comparison

# Model architecture - 12 layers (GPT-2 small scale)
n_layer = 12
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
compile = False  # Disabled for GPU compatibility
dtype = "bfloat16"

# Reproducibility
seed = 42
