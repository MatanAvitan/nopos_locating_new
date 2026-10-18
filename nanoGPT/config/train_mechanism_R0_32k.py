# Config: FULL-12H (R0) with 32K context length
# Tests whether the geometric clock mechanism scales to long sequences
#
# Run with:
#   python train_2layer_mechanism.py --regime R0 --block_size 32768 \
#     --use_flash --batch_size 4 --gradient_accumulation_steps 8 \
#     --max_iters 5000 --max_iters_override 5000 \
#     --n_head 12 --wandb --out_dir out-mechanism-R0-32k
#
# Or source this file for reference values.

# Architecture
block_size = 32768  # 32K context
n_embd = 768
n_head = 12
regime = "R0"

# Training — effective batch = 4 * 8 = 32 sequences
# Each iter: 32 * 32768 ≈ 1M tokens
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 5000  # ~5B tokens total
lr_decay_iters = 5000
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 200
eval_interval = 250
eval_iters = 20  # fewer eval iters for speed at 32K

# Flash attention required for 32K
use_flash = True

# Output
out_dir = "out-mechanism-R0-32k"
wandb_project = "nope-2layer-mechanism"
wandb_log = True
