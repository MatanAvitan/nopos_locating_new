# Configuration for LM Training - 6 Layer - Full Training (DDP, 3 GPUs)
# Same architecture as 6-layer LM, but tuned for high-throughput training

# I/O
out_dir = "out-lm-6layer-fulltrain-ddp"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-fulltrain-lm-ddp"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 2  # divided by world_size (2) => 1
batch_size = 512
block_size = 128

# Model - 6 layers
n_layer = 6
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE
use_positional_embedding = False
norm_type = "layernorm"

# Training mode - all weights trainable
freeze_transformer = False
freeze_attention_only = False
freeze_until_first_mlp = False
use_regression = False
compute_lm_loss = True
train_lm_only = True

# AdamW optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 3e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = False
seed = 42
