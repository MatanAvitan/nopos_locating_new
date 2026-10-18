# Training config for NoPE transformer WITHOUT LN2 on OpenWebText
# Tests hypothesis: Does LN2 hurt positional encoding by normalizing variance?
#
# Key features:
# - No positional embeddings (NoPE mode)
# - NO second LayerNorm (skip_ln2=True) - KEY ABLATION
# - Xavier initialization for attention Q, K, V
# - OpenWebText dataset (~9B tokens)
# - Context length 2048 for testing long-range position encoding
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_owt_no_ln2.py

# I/O
out_dir = "out-nope-owt-no-ln2"
eval_interval = 500  # Checkpoint every 500 iters for resumability
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-emergence"
wandb_run_name = "nope-owt-no-ln2"

# Data
dataset = "openwebtext"  # Full OpenWebText (~9B tokens)
gradient_accumulation_steps = 4  # Accumulate to effective batch 256
batch_size = 64  # Reduced batch for memory (64 * 4 * 512 = 131K tokens)
block_size = 512  # Context length

# Model architecture
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for clean analysis
bias = False  # Modern practice

# NoPE-specific settings
use_positional_embedding = False  # KEY: No positional embeddings
norm_type = "layernorm"  # Standard LayerNorm (for LN1)
log_attention_stats = True  # Log attention entropy/uniformity
skip_ln2 = True  # KEY ABLATION: Skip second LayerNorm

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
compile = False
dtype = "bfloat16"

# Reproducibility
seed = 42

# Stability tracking (important for no-LN2 variant)
track_stability = True
stability_log_interval = 100
