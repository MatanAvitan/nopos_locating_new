# Configuration for LM Training - 6 Layer - BOS Q/K Frozen
# Freeze W_Q/W_K for one BOS head in block 1, allow W_V to train

# I/O
out_dir = "out-lm-6layer-bos-qk-frozen"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-bos-qk-frozen"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4  # divided by world_size
batch_size = 256
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
log_attention_stats = False

# BOS Q/K frozen source
bos_qk_source_checkpoint = "out-2layer-mechanism/R0/best_ckpt.pt"
bos_qk_source_block = 1
bos_qk_head_idx = 6  # use BOS head from source experiment
bos_head_block = 1  # second block (0-indexed)

# AdamW optimizer (same as vanilla LM)
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
