#!/bin/bash
# Master script to submit all NoPE analysis experiments to Slurm
# 
# Usage: ./submit_all_jobs.sh
#
# This submits all 6 experiments in parallel to the Slurm cluster.
# Each job uses 1 GPU and runs for up to 4 hours.

cd /home/nlp/matan_avitan/git/nopos_locating_new

echo "=============================================="
echo "NoPE Analysis - Slurm Job Submission"
echo "=============================================="
echo "Submitting 6 experiments to Slurm cluster..."
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit all jobs
echo "[1/6] Submitting Token-Position Correlation..."
JOB1=$(sbatch slurm_jobs/run_token_position_correlation.sh | awk '{print $4}')
echo "      Job ID: $JOB1"

echo "[2/6] Submitting Comprehensive Probe Analysis..."
JOB2=$(sbatch slurm_jobs/run_comprehensive_probe.sh | awk '{print $4}')
echo "      Job ID: $JOB2"

echo "[3/6] Submitting Higher-Order Statistics..."
JOB3=$(sbatch slurm_jobs/run_higher_order_stats.sh | awk '{print $4}')
echo "      Job ID: $JOB3"

echo "[4/6] Submitting Decoding Vector Experiments..."
JOB4=$(sbatch slurm_jobs/run_decoding_vector.sh | awk '{print $4}')
echo "      Job ID: $JOB4"

echo "[5/6] Submitting Causal Intervention Experiments..."
JOB5=$(sbatch slurm_jobs/run_causal_intervention.sh | awk '{print $4}')
echo "      Job ID: $JOB5"

echo "[6/6] Submitting Training Dynamics Analysis..."
JOB6=$(sbatch slurm_jobs/run_training_dynamics.sh | awk '{print $4}')
echo "      Job ID: $JOB6"

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: logs/slurm_*.out"
echo ""
echo "Job Summary:"
echo "  Token-Position Correlation: $JOB1"
echo "  Comprehensive Probe:        $JOB2"
echo "  Higher-Order Statistics:    $JOB3"
echo "  Decoding Vector:            $JOB4"
echo "  Causal Intervention:        $JOB5"
echo "  Training Dynamics:          $JOB6"
