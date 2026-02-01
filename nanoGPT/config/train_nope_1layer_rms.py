# Training config for 1-layer NoPE transformer with RMSNorm
# For comparing with LayerNorm variant - studying how normalization affects position encoding
#
# Key difference from LayerNorm:
# - RMSNorm does NOT center activations (no mean subtraction)
# - This tests whether the "LayerNorm paradox" mechanism is necessary
# - Position information may flow differently without centering
#
# Run with:
#   CUDA_VISIBLE_DEVICES=1 python train_nope.py config/train_nope_1layer_rms.py

# I/O
out_dir = "out-nope-1layer-rms"
eval_interval = 250  # Same as LN config
eval_iters = 100
log_interval = 10

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging (set to False if not configured)
wandb_log = False
wandb_project = "nope-emergence"
wandb_run_name = "nope-1layer-rms"

# Data (identical to LN config)
dataset = "shakespeare"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 256

# Model architecture (identical to LN config)
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE-specific settings - KEY DIFFERENCE: rmsnorm instead of layernorm
use_positional_embedding = False
norm_type = "rmsnorm"  # RMSNorm: only scaling, NO centering
log_attention_stats = True

# Optimizer (identical to LN config)
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

warmup_iters = 200

# System
device = "cuda"
compile = True
dtype = "bfloat16"

# Reproducibility - MUST match LayerNorm config for fair comparison
seed = 42
