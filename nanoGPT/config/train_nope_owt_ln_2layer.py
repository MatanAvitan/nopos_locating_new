# 2-LAYER NoPE transformer LM (next-token) on OpenWebText, LayerNorm.
# Mirrors the depth of the 2-layer mechanism model so the position-encoding
# mechanism (BOS-anchored attention decay -> two-direction -> MLP readout)
# can be tested on a REAL trained language model.
#
# Run: CUDA_VISIBLE_DEVICES=X python train_nope.py config/train_nope_owt_ln_2layer.py

# I/O
out_dir = "out-nope-owt-ln-2layer"
eval_interval = 500
eval_iters = 100
log_interval = 50
always_save_checkpoint = True
init_from = "scratch"

# Wandb
wandb_log = False
wandb_project = "nope-emergence"
wandb_run_name = "nope-owt-ln-2layer"

# Data
dataset = "openwebtext"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 512

# Model architecture - 2 layers to mirror the mechanism model
n_layer = 2
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# NoPE-specific
use_positional_embedding = False
norm_type = "layernorm"
log_attention_stats = False   # False -> flash attention -> fast training
skip_ln2 = False

# Optimizer
learning_rate = 6e-4
max_iters = 30000
lr_decay_iters = 30000
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 1000

# System
device = "cuda"
compile = False
dtype = "bfloat16"
seed = 42
