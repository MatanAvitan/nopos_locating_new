# Training config for 12-layer NoPE transformer WITHOUT LN2 on OpenWebText
# Tests hypothesis: Does LN2 hurt positional encoding by normalizing variance in deeper models?
#
# Key features:
# - No positional embeddings (NoPE mode)
# - NO second LayerNorm (skip_ln2=True) - KEY ABLATION
# - 12 layers (GPT-2 small scale)
# - OpenWebText dataset (~9B tokens)
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_owt_no_ln2_12layer.py

# I/O
out_dir = "out-nope-owt-no-ln2-12layer"
eval_interval = 500  # Checkpoint every 500 iters for resumability
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-emergence"
wandb_run_name = "nope-owt-no-ln2-12layer"

# Data
dataset = "openwebtext"  # Full OpenWebText (~9B tokens)
gradient_accumulation_steps = (
    1  # No accumulation - need large batch for population statistics
)
batch_size = 128  # Large batch (128 * 512 = 65K tokens)
block_size = 512  # Context length

# Model architecture - 12 layers (GPT-2 small scale)
n_layer = 12
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
compile = False  # Disabled for GPU compatibility
dtype = "bfloat16"

# Reproducibility
seed = 42

# Stability tracking (important for no-LN2 variant)
track_stability = True
stability_log_interval = 100
