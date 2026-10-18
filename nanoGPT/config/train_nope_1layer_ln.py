# Training config for 1-layer NoPE transformer with LayerNorm
# For studying positional encoding emergence without explicit PE
#
# Key features:
# - No positional embeddings (NoPE mode)
# - LayerNorm normalization (standard centering + scaling)
# - Xavier initialization for attention Q, K, V
# - Attention statistics logging (entropy, uniformity)
#
# Run with:
#   CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_1layer_ln.py

# I/O
out_dir = "out-nope-1layer-ln"
eval_interval = 250  # Checkpoint every 250 iters for emergence analysis
eval_iters = 100
log_interval = 10

always_save_checkpoint = True  # Save at every eval for emergence study
init_from = "scratch"

# Wandb logging (set to False if not configured)
wandb_log = False
wandb_project = "nope-emergence"
wandb_run_name = "nope-1layer-ln"

# Data
dataset = "shakespeare"  # BPE tokenized (~302K tokens, vocab=50257)
gradient_accumulation_steps = 4
batch_size = 32  # Effective batch size = 32 * 4 = 128
block_size = 256  # Context length

# Model architecture
n_layer = 1
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for clean analysis
bias = False  # Modern practice

# NoPE-specific settings
use_positional_embedding = False  # KEY: No positional embeddings
norm_type = "layernorm"  # Standard LayerNorm
log_attention_stats = True  # Log attention entropy/uniformity during training

# Optimizer
learning_rate = 1e-3  # Higher LR for small model
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
compile = True  # Use torch.compile for speed
dtype = "bfloat16"  # Use bfloat16 on A100

# Reproducibility - MUST match RMSNorm config for fair comparison
seed = 42
