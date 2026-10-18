# Config: ATTN2-1H (R2) with 32K context length
# Tests whether the geometric clock mechanism scales to long sequences
# R2: single attention head, frozen block 1 + MLP2, trains only Block2.Attn + head
#
# Run with:
#   python train_2layer_mechanism.py --regime R2 --r2_attn_head_only \
#     --block_size 32768 --use_flash --batch_size 8 \
#     --gradient_accumulation_steps 4 --n_head 1 \
#     --max_iters 10000 --max_iters_override 10000 \
#     --wandb --out_dir out-mechanism-R2-32k
#
# Or source this file for reference values.

# Architecture
block_size = 32768  # 32K context
n_embd = 768
n_head = 1  # Single attention head for mechanistic clarity
regime = "R2"
r2_attn_head_only = True

# Training — effective batch = 8 * 4 = 32 sequences
# Single head uses less memory than 12 heads
batch_size = 8
gradient_accumulation_steps = 4
max_iters = 10000  # R2 needs more iters (fewer trainable params)
lr_decay_iters = 10000
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 400
eval_interval = 500
eval_iters = 20

# Flash attention required for 32K
use_flash = True

# Output
out_dir = "out-mechanism-R2-32k"
wandb_project = "nope-2layer-mechanism"
wandb_log = True
