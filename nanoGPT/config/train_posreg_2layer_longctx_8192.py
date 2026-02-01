# Configuration for Position Regression - 2 Layer - Long Context (8192)
# Train on 8192 context length to test 32x extrapolation
# Uses regression loss (MSE) to predict position

# I/O
out_dir = "out-posreg-2layer-longctx-8192"
eval_interval = 500
log_interval = 50
eval_iters = 100
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-position-regression"
wandb_run_name = "nope-2layer-longctx-8192"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 8  # Higher accumulation for long sequences
batch_size = 8  # Smaller batch due to memory
block_size = 8192  # Long context for extrapolation testing

# Model - 2 layers (same architecture as short context experiments)
n_layer = 2
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"
log_attention_stats = True

# Training mode - train the full 2-layer network with position regression
freeze_transformer = False
freeze_attention_only = False
freeze_until_first_mlp = (
    True  # Freeze embeddings + block0 attn/norms, train from first MLP
)
use_regression = True  # MSE loss for position prediction
compute_lm_loss = False  # Don't need LM loss for this experiment

# AdamW optimizer
learning_rate = 1e-4  # Lower LR for stability with long sequences
max_iters = 10000  # Fewer iterations since each is more expensive
weight_decay = 0.1
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = True
seed = 42
