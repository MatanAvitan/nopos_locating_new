# Training config for NoPE transformer with RMSNorm on OpenWebText
# Larger scale training for studying positional encoding on real data
#
# Key features:
# - No positional embeddings (NoPE mode)
# - RMSNorm normalization (scaling only, no centering)
# - Xavier initialization for attention Q, K, V
# - OpenWebText dataset (~9B tokens)
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_owt_rms.py

# I/O
out_dir = "out-nope-owt-rms"
eval_interval = 1000  # Checkpoint every 1000 iters
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging
wandb_log = False
wandb_project = "nope-emergence"
wandb_run_name = "nope-owt-rms"

# Data
dataset = "openwebtext"  # Full OpenWebText (~9B tokens)
gradient_accumulation_steps = 8
batch_size = 16  # Effective batch size = 16 * 8 = 128
block_size = 256  # Context length

# Model architecture - same as Shakespeare experiments
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for clean analysis
bias = False  # Modern practice

# NoPE-specific settings
use_positional_embedding = False  # KEY: No positional embeddings
norm_type = "rmsnorm"  # RMSNorm (no mean centering)
log_attention_stats = True  # Log attention entropy/uniformity during training

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
compile = True  # Use torch.compile for speed
dtype = "bfloat16"

# Reproducibility - MUST match LayerNorm config for fair comparison
seed = 42
