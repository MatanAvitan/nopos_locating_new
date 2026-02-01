# Configuration for LM Training - 6 Layer - Forced BOS Mechanism (DDP)
# Hard-codes position encoding via frozen attention heads:
# - Block 0, Head 0: Uniform causal attention (prefix averaging)
# - Block 1, Head 0: BOS attention (attend only to position 0)

# I/O
out_dir = "out-lm-6layer-forced-bos"
eval_interval = 500
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

# Wandb logging
wandb_log = True
wandb_project = "nope-lm"
wandb_run_name = "nope-6layer-forced-bos-lm"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 2  # divided by world_size
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
log_attention_stats = True

# Forced BOS mechanism
use_forced_bos_model = True
freeze_uniform_head_idx = 0  # Head 0 in block 0 -> uniform attention
freeze_bos_head_idx = 0  # Head 0 in block 1 -> BOS attention
uniform_head_block = 0
bos_head_block = 1

# Training mode - all weights trainable (except frozen attention patterns)
freeze_transformer = False
freeze_attention_only = False
freeze_until_first_mlp = False
use_regression = False
compute_lm_loss = True
train_lm_only = True

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
