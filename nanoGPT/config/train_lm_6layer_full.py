# Configuration for LM Training - 6 Layer - Full Training (All Weights Unfrozen)
# Train a standard 6-layer NoPE transformer as a language model
# All weights are trainable (unlike the frozen position regression experiments)

# I/O
out_dir = "out-lm-6layer-fulltrain"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-fulltrain-lm"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 128

# Model - 6 layers
n_layer = 6
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE - No positional embeddings
use_positional_embedding = False
norm_type = "layernorm"

# Training mode - ALL WEIGHTS TRAINABLE
freeze_transformer = False
freeze_attention_only = False
freeze_until_first_mlp = False  # KEY: Everything is trainable
use_regression = False  # Not regression mode
compute_lm_loss = True  # Enable LM head
train_lm_only = True  # Train on LM loss
use_ln2 = True  # Use LayerNorm2 in blocks
mlp_ratio = 4  # Standard MLP expansion

# AdamW optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 3e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = False  # Disable - torch.compile hangs on this system
seed = 42
