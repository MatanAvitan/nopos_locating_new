#!/bin/bash
#SBATCH --job-name=LM6-BOS-L1024
#SBATCH --output=/home/nlp/matan_avitan/git/nopos_locating_new/logs/slurm_lm6_bos_L1024_%A_%a.out
#SBATCH --error=/home/nlp/matan_avitan/git/nopos_locating_new/logs/slurm_lm6_bos_L1024_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=1-2

# The BOS-trained 6-layer NoPE LM, trained at L = 1024 instead of 128.
# Same config as config/train_lm_6layer_fulltrain_bos_ddp.py, with the same tokens
# per optimizer step: 64 x 2 x 1024 = 131072 = 512 x 2 x 128. On smaller GPUs, set BATCH=32 ACCUM=4.
# The array index is the seed. A requeued or chained job resumes from ckpt.pt.
SEED=$SLURM_ARRAY_TASK_ID
eval "$(conda shell.bash hook)"
conda activate b200
cd /home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT
OUT=out-lm-6layer-fulltrain-bos-L1024-seed$SEED
RESUME=""
if [ -f "$OUT/ckpt.pt" ]; then
    ITER=$(python -c "import torch; print(torch.load('$OUT/ckpt.pt', map_location='cpu', weights_only=False, mmap=True)['iter_num'])")
    if [ "$ITER" -ge 20000 ]; then echo "seed $SEED already at iteration $ITER"; exit 0; fi
    RESUME="--init_from=resume"
fi
python -u train_nope.py config/train_lm_6layer_fulltrain_bos_ddp.py \
    --seed=$SEED --out_dir=$OUT --block_size=1024 --batch_size=${BATCH:-64} --gradient_accumulation_steps=${ACCUM:-2} \
    --wandb_log=False --dtype=bfloat16 $RESUME
