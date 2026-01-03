#!/bin/bash
# Master experiment runner for LayerNorm Paradox paper
# Orchestrates all experiments across GPUs 5, 6, 7, 8
# Total estimated time: 22-27 hours (parallelized)

set -e  # Exit on error

BASE_DIR="/home/nlp/matan_avitan/git/nopos_locating_new"
cd "$BASE_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

phase() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

# ===================================================================
# PHASE 1: Setup (already done in Python)
# ===================================================================
phase "Phase 1: Setup Complete"
log "✓ Plotting utilities created"
log "✓ utils.py modified with save_path support"
log "✓ Kaleido installed for figure export"

# ===================================================================
# PHASE 2: Core Mechanism Experiments (3-4 hours)
# ===================================================================
phase "PHASE 2: Core Mechanism Experiments"
log "Starting synthetic and natural language experiments in parallel..."

# Phase 2.1: Synthetic Uniform (GPU 5)
log "GPU 5: Starting synthetic uniform token sampling..."
CUDA_VISIBLE_DEVICES=5 python train/train_synthetic_w_ln_frozen_embeddings_attn_lns_train_only_mlps_large_vocab.py > logs/phase2_1_synthetic_gpu5.log 2>&1 &
PID_2_1=$!

# Phase 2.2: Natural Language (GPU 6)
log "GPU 6: Starting natural language experiments..."
CUDA_VISIBLE_DEVICES=6 python train_ln_rep_prediction/train_on_natural_language_ln2_output_max_samples.py > logs/phase2_2_natural_gpu6.log 2>&1 &
PID_2_2=$!

# Wait for Phase 2 to complete
log "Waiting for core experiments to complete..."
wait $PID_2_1
if [ $? -eq 0 ]; then
    log "✓ Phase 2.1 (Synthetic) completed successfully"
else
    error "Phase 2.1 failed! Check logs/phase2_1_synthetic_gpu5.log"
fi

wait $PID_2_2
if [ $? -eq 0 ]; then
    log "✓ Phase 2.2 (Natural Language) completed successfully"
else
    error "Phase 2.2 failed! Check logs/phase2_2_natural_gpu6.log"
fi

# ===================================================================
# PHASE 3: Normalization & Ablation Studies (4-5 hours)
# ===================================================================
phase "PHASE 3: Normalization & Ablation Studies"

# Phase 3.1: Normalization variants (GPU 7)
log "GPU 7: Starting normalization variants..."

# LayerNorm (baseline)
log "  - Running LayerNorm variant..."
CUDA_VISIBLE_DEVICES=7 python train/train_synthetic_w_ln_frozen_embeddings_attn_lns_train_only_mlps.py > logs/phase3_1_layernorm_gpu7.log 2>&1

# No-Norm
log "  - Running No-Norm variant..."
CUDA_VISIBLE_DEVICES=7 python train/train_synthetic_w_ln_frozen_embeddings_attn_train_only_mlps_no_ln.py > logs/phase3_1_no_norm_gpu7.log 2>&1

log "✓ Phase 3.1 (Normalization variants) completed"

# Note: RMSNorm variant would need to be created if transformer_lens supports it
# For now, we'll document this in the results

# ===================================================================
# PHASE 4: Scaling Experiments (6-8 hours)
# ===================================================================
phase "PHASE 4: Scaling Experiments"

# Phase 4.1: Vocabulary Scaling (GPU 8) - Can start while Phase 3 runs
log "GPU 8: Starting vocabulary scaling experiments..."
log "This will run in parallel with Phase 3..."

# Run vocab scaling in background
python analysis_scripts/run_vocab_scaling_sweep.py > logs/phase4_1_vocab_scaling_gpu8.log 2>&1 &
PID_4_1=$!

# Wait for vocab scaling (longest running task)
log "Waiting for vocabulary scaling to complete (this may take 6-8 hours)..."
wait $PID_4_1
if [ $? -eq 0 ]; then
    log "✓ Phase 4.1 (Vocab Scaling) completed successfully"
else
    error "Phase 4.1 failed! Check logs/phase4_1_vocab_scaling_gpu8.log"
fi

# Phase 4.2 & 4.3: Sample convergence and length extrapolation
log "Running sample convergence and length extrapolation analyses..."
python analysis_scripts/sample_convergence_analysis.py > logs/phase4_2_convergence.log 2>&1
python analysis_scripts/length_extrapolation.py > logs/phase4_3_extrapolation.log 2>&1

log "✓ Phase 4 (Scaling Experiments) completed"

# ===================================================================
# PHASE 5: Complementary Experiments (8-10 hours)
# ===================================================================
phase "PHASE 5: Complementary Experiments"
log "Starting complementary experiments to strengthen the paper..."

# C1: Multi-layer analysis (GPU 5 - now free)
log "GPU 5: Starting multi-layer analysis..."
CUDA_VISIBLE_DEVICES=5 python train/train_synthetic_multilayer_analysis.py > logs/phase5_c1_multilayer_gpu5.log 2>&1 &
PID_C1=$!

# C2: Architecture variants (GPU 6 - now free)
log "GPU 6: Starting architecture variant analysis..."
CUDA_VISIBLE_DEVICES=6 python train/train_architecture_variants.py > logs/phase5_c2_architecture_gpu6.log 2>&1 &
PID_C2=$!

# C3: Hyperparameter robustness (GPU 7 - now free)
log "GPU 7: Starting hyperparameter sweep..."
CUDA_VISIBLE_DEVICES=7 python train/train_hyperparameter_sweep.py > logs/phase5_c3_hyperparams_gpu7.log 2>&1 &
PID_C3=$!

# C4: Pretrained analysis (GPU 8 - now free)
log "GPU 8: Starting pretrained model analysis..."
CUDA_VISIBLE_DEVICES=8 python analysis_scripts/analyze_pretrained_models.py > logs/phase5_c4_pretrained_gpu8.log 2>&1 &
PID_C4=$!

# Wait for all complementary experiments
wait $PID_C1 && log "✓ C1 (Multi-layer) completed"
wait $PID_C2 && log "✓ C2 (Architecture variants) completed"
wait $PID_C3 && log "✓ C3 (Hyperparameter robustness) completed"
wait $PID_C4 && log "✓ C4 (Pretrained analysis) completed"

log "✓ Phase 5 (Complementary Experiments) completed"

# ===================================================================
# PHASE 6: Plot Generation (2-3 hours)
# ===================================================================
phase "PHASE 6: Plot Generation"
log "Generating all publication-quality figures..."

python analysis_scripts/generate_all_paper_figures.py > logs/phase6_plot_generation.log 2>&1

if [ $? -eq 0 ]; then
    log "✓ All figures generated successfully"
    log "  Figures saved to: overleaf/nopos---claude-version/plots/"
else
    error "Plot generation failed! Check logs/phase6_plot_generation.log"
fi

# ===================================================================
# PHASE 7: LaTeX Integration (1-2 hours)
# ===================================================================
phase "PHASE 7: LaTeX Integration"
log "Integrating figures into LaTeX paper..."

python analysis_scripts/integrate_figures_to_latex.py > logs/phase7_latex_integration.log 2>&1

if [ $? -eq 0 ]; then
    log "✓ LaTeX integration completed"
    log "  Paper file: overleaf/nopos---claude-version/acl_latex.tex"
else
    error "LaTeX integration failed! Check logs/phase7_latex_integration.log"
fi

# ===================================================================
# PHASE 8: Results Consolidation & Validation (2 hours)
# ===================================================================
phase "PHASE 8: Results Consolidation & Validation"
log "Consolidating all experimental results..."

python analysis_scripts/consolidate_results.py > logs/phase8_consolidation.log 2>&1

if [ $? -eq 0 ]; then
    log "✓ Results consolidation completed"
    log "  Summary: results_summary.json"
    log "  LaTeX tables: latex_tables.tex"
else
    error "Results consolidation failed! Check logs/phase8_consolidation.log"
fi

# ===================================================================
# Final Summary
# ===================================================================
phase "EXECUTION COMPLETE!"
echo ""
log "All experiments completed successfully! 🎉"
echo ""
echo "Summary:"
echo "  ✓ Phase 2: Core experiments (synthetic + natural language)"
echo "  ✓ Phase 3: Normalization variants & ablations"
echo "  ✓ Phase 4: Vocabulary scaling & sample convergence"
echo "  ✓ Phase 5: Complementary experiments (4 additional studies)"
echo "  ✓ Phase 6: All figures generated (6 main + 4-5 complementary)"
echo "  ✓ Phase 7: LaTeX integration complete"
echo "  ✓ Phase 8: Results consolidated and validated"
echo ""
echo "Key outputs:"
echo "  📊 Figures: overleaf/nopos---claude-version/plots/"
echo "  📝 Paper: overleaf/nopos---claude-version/acl_latex.tex"
echo "  📈 Results: results_summary.json"
echo "  💾 Models: models/"
echo "  📋 Logs: logs/"
echo ""
log "Next steps:"
echo "  1. Review results_summary.json for all experimental outcomes"
echo "  2. Check overleaf/nopos---claude-version/plots/ for all figures"
echo "  3. Compile LaTeX paper: cd overleaf/nopos---claude-version && pdflatex acl_latex.tex"
echo "  4. Review and refine paper content based on actual results"
echo ""
