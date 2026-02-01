# Training config for 1-layer NoPE transformer with LayerNorm - 8K Context
# For validating positional encoding mechanism at long context lengths
#
# Key features:
# - No positional embeddings (NoPE mode)
# - 8K context length to test direction→norm transformation hypothesis
# - OpenWebText dataset (~9B tokens) for sufficient data
# - Reduced batch size to fit memory with long sequences
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_8k_ln.py

# I/O
out_dir = "out-nope-8k-ln"
eval_interval = 500  # Checkpoint every 500 iters
eval_iters = 50  # Fewer eval iters due to long sequences
log_interval = 10

always_save_checkpoint = True
init_from = "scratch"

# Wandb logging (set to False if not configured)
wandb_log = False
wandb_project = "nope-emergence"
wandb_run_name = "nope-8k-ln"

# Data
dataset = "openwebtext"  # ~9B tokens for sufficient 8K sequences
gradient_accumulation_steps = 64  # Maintain effective batch ~128 (2 * 64)
batch_size = 2  # Small batch due to 8K context memory requirements
block_size = 8192  # 8K context length

# Model architecture
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for clean analysis
bias = False

# NoPE-specific settings
use_positional_embedding = False  # KEY: No positional embeddings
norm_type = "layernorm"
log_attention_stats = False  # Disable for 8K - too slow/memory intensive

# Optimizer
learning_rate = 6e-4  # Slightly lower for longer sequences
max_iters = 10000  # More iters for larger dataset
lr_decay_iters = 10000
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

warmup_iters = 500

# System
device = "cuda"
compile = True  # Use torch.compile for speed
dtype = "bfloat16"  # Required for H200/A100

# Reproducibility
seed = 42
