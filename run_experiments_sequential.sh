#!/bin/bash
# Sequential runner script - runs experiments in batches of 2 using GPUs 0 and 1
# This is for when other GPUs are occupied

set -e

# Create results directories
mkdir -p results/token_position_correlation
mkdir -p results/comprehensive_probe_analysis
mkdir -p results/higher_order_statistics
mkdir -p results/decoding_vector_experiments
mkdir -p results/causal_interventions
mkdir -p results/training_dynamics
mkdir -p logs

echo "=========================================="
echo "NoPE Analysis - Sequential Batch Runner"
echo "=========================================="
echo "Start time: $(date)"
echo "Using GPUs 0 and 1 only"
echo ""

# Batch 1: Token-Position Correlation + Comprehensive Probe
echo "=== BATCH 1/3 ==="
echo "[GPU 0] Starting Token-Position Correlation..."
CUDA_VISIBLE_DEVICES=0 python analysis_scripts/token_position_correlation_natural_language.py \
    > logs/token_position_correlation.log 2>&1 &
PID1=$!

echo "[GPU 1] Starting Comprehensive Probe Analysis..."
CUDA_VISIBLE_DEVICES=1 python analysis_scripts/comprehensive_probe_analysis.py \
    > logs/comprehensive_probe_analysis.log 2>&1 &
PID2=$!

wait $PID1
if [ $? -ne 0 ]; then echo "[FAILED] Token-Position Correlation"; else echo "[DONE] Token-Position Correlation"; fi

wait $PID2
if [ $? -ne 0 ]; then echo "[FAILED] Comprehensive Probe Analysis"; else echo "[DONE] Comprehensive Probe Analysis"; fi

# Batch 2: Higher-Order Statistics + Decoding Vector
echo ""
echo "=== BATCH 2/3 ==="
echo "[GPU 0] Starting Higher-Order Statistics..."
CUDA_VISIBLE_DEVICES=0 python analysis_scripts/higher_order_statistics_analysis.py \
    > logs/higher_order_statistics.log 2>&1 &
PID3=$!

echo "[GPU 1] Starting Decoding Vector Experiments..."
CUDA_VISIBLE_DEVICES=1 python analysis_scripts/decoding_vector_experiments.py \
    > logs/decoding_vector_experiments.log 2>&1 &
PID4=$!

wait $PID3
if [ $? -ne 0 ]; then echo "[FAILED] Higher-Order Statistics"; else echo "[DONE] Higher-Order Statistics"; fi

wait $PID4
if [ $? -ne 0 ]; then echo "[FAILED] Decoding Vector Experiments"; else echo "[DONE] Decoding Vector Experiments"; fi

# Batch 3: Causal Interventions + Training Dynamics
echo ""
echo "=== BATCH 3/3 ==="
echo "[GPU 0] Starting Causal Intervention Experiments..."
CUDA_VISIBLE_DEVICES=0 python analysis_scripts/causal_intervention_experiments.py \
    > logs/causal_intervention_experiments.log 2>&1 &
PID5=$!

echo "[GPU 1] Starting Training Dynamics Analysis..."
CUDA_VISIBLE_DEVICES=1 python analysis_scripts/training_dynamics_analysis.py --gpu 1 \
    > logs/training_dynamics.log 2>&1 &
PID6=$!

wait $PID5
if [ $? -ne 0 ]; then echo "[FAILED] Causal Intervention Experiments"; else echo "[DONE] Causal Intervention Experiments"; fi

wait $PID6
if [ $? -ne 0 ]; then echo "[FAILED] Training Dynamics Analysis"; else echo "[DONE] Training Dynamics Analysis"; fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "Results saved to:"
echo "  - results/token_position_correlation/"
echo "  - results/comprehensive_probe_analysis/"
echo "  - results/higher_order_statistics/"
echo "  - results/decoding_vector_experiments/"
echo "  - results/causal_interventions/"
echo "  - results/training_dynamics/"
echo ""
echo "Figures saved to:"
echo "  - overleaf/nopos---claude-version/plots/"
