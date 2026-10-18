# Training config for NoPE transformer with LayerNorm on OpenWebText
# Larger scale training for studying positional encoding on real data
#
# Key features:
# - No positional embeddings (NoPE mode)
# - LayerNorm normalization (standard centering + scaling)
# - Xavier initialization for attention Q, K, V
# - OpenWebText dataset (~9B tokens)
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_owt_ln.py

# I/O
out_dir = "out-nope-owt-ln"
eval_interval = 500  # Checkpoint every 500 iters for resumability
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "nope-emergence"
wandb_run_name = "nope-owt-ln"

# Data
dataset = "openwebtext"  # Full OpenWebText (~9B tokens)
gradient_accumulation_steps = 4  # Accumulate to effective batch 256
batch_size = 64  # Reduced batch for memory (64 * 4 * 512 = 131K tokens)
block_size = 512  # Context length

# Model architecture - same as Shakespeare experiments
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for clean analysis
bias = False  # Modern practice

# NoPE-specific settings
use_positional_embedding = False  # KEY: No positional embeddings
norm_type = "layernorm"  # Standard LayerNorm
log_attention_stats = True  # Log attention entropy/uniformity during training
skip_ln2 = False  # Standard NoPE with LN2

# Optimizer
learning_rate = 6e-4  # Slightly lower LR for larger dataset
max_iters = 50000  # More iters for larger dataset
lr_decay_iters = 50000
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 2000

# System
device = "cuda"
compile = False  # Use torch.compile for speed
dtype = "bfloat16"

# Reproducibility
seed = 42
