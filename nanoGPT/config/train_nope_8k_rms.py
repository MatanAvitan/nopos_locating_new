# Training config for 1-layer NoPE transformer with RMSNorm - 8K Context
# For comparing RMSNorm (no mean centering) with LayerNorm at long context
#
# RMSNorm hypothesis: Without mean centering, positional information
# might be preserved better in the direction (relative neuron values)
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_8k_rms.py

# I/O
out_dir = "out-nope-8k-rms"
eval_interval = 500
eval_iters = 50
log_interval = 10

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = False
wandb_project = "nope-emergence"
wandb_run_name = "nope-8k-rms"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 64
batch_size = 2
block_size = 8192

# Model architecture
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE-specific settings
use_positional_embedding = False
norm_type = "rmsnorm"  # KEY: RMSNorm instead of LayerNorm
log_attention_stats = False

# Optimizer
learning_rate = 6e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

warmup_iters = 500

# System
device = "cuda"
compile = True
dtype = "bfloat16"

# Reproducibility - match LN config
seed = 42
