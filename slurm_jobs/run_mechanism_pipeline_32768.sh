#!/bin/bash
#SBATCH --job-name=A2F-32k-ana
#SBATCH --output=logs/slurm_pipeline_32768_%j.out
#SBATCH --error=logs/slurm_pipeline_32768_%j.err
#SBATCH --partition=B200-4h
#SBATCH --account=ug_dsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# P5: canonical mechanism pipeline on ATTN2-FULL L=32768.
# Stages are idempotent (each overwrites its own outputs); a requeue that
# reruns from the top costs extract time only.
eval "$(conda shell.bash hook)"
conda activate b200

cd /home/nlp/matan_avitan/git/nopos_locating_new
P=analysis_scripts/mechanism_pipeline
ARGS="--model attn2_full_32768 --context-length 32768 --batch-size 1"
RID="attn2_full_32768_L32768"
# val.bin holds only 135 non-overlapping 32767-token sequences; 32/16/80 fits.
python $P/extract.py $ARGS --n-ref 32 --n-cal 16 --n-eval 80
python $P/stats.py --run-id $RID
python $P/step2_geometry.py --run-id $RID --model attn2_full_32768
python $P/probes.py --run-id $RID $ARGS
python $P/interventions.py --run-id $RID $ARGS
python $P/eta_interventions.py --run-id $RID $ARGS --batch-size 2
python $P/step1_corrected.py --run-id $RID $ARGS
python $P/span2_decode.py --run-id $RID $ARGS
