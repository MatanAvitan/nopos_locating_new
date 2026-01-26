# R0 Model Retraining Log

## Training Information

### Current Retraining Run
**Start Time**: 2026-01-25 17:24:30
**Status**: In Progress
**WandB Run ID**: `0fd63e8l`
**WandB Project**: `nope-2layer-mechanism`
**WandB URL**: https://wandb.ai/matan_avitan/nope-2layer-mechanism/runs/0fd63e8l

### Original Successful Run (Overwritten)
**Date**: 2026-01-20 21:46:01
**WandB Run ID**: `wd3y7olp`
**WandB Directory**: `nanoGPT/wandb/run-20260120_214601-wd3y7olp/`
**Runtime**: 363 seconds (~6 minutes)
**Hardware**: NVIDIA A100-SXM4-80GB

**Original Results (Confirmed from WandB logs):**
- ✅ val/r2: **0.9996** (exactly matches Table 1!)
- ✅ val/mae: **0.40** (exactly matches Table 1!)
- ✅ val/loss: **0.57** (exactly matches Table 1!)
- ✅ Completed all 20,000 iterations
- ✅ Configuration matched paper specifications perfectly

## Configuration

**Model Architecture:**
```python
n_embd: 768
n_head: 12
block_size: 128
vocab_size: 50304
dropout: 0.0
norm_type: layernorm
n_layers: 2  # (2-layer pre-norm causal transformer)
```

**Training Hyperparameters:**
```python
max_iters: 20000
batch_size: 64
gradient_accumulation_steps: 2  # Effective batch = 128
learning_rate: 6e-4
min_lr: 6e-5
warmup_iters: 500
lr_decay_iters: 20000
weight_decay: 0.1
beta1: 0.9
beta2: 0.99
grad_clip: 1.0
seed: 42
dataset: openwebtext
```

## Expected Results (From Paper)

**Overall Performance (Table 1):**
- R² = 0.9996 (or 0.966 from extrapolation table)
- MAE = 0.40
- Loss = 0.57

**BOS Heads (Figure 2):**
- Heads 6 and 9 should have >50% attention to position 0 in Block 2
- Visualized over 200 sequences from OpenWebText, L=128

**BOS Intervention (Figure 3):**
- Baseline R²: 0.999
- After masking BOS: R² = 0.914 (8.5% drop)

**Length Extrapolation (Figure 5a):**
- L=128: R²=0.966
- L=2048: R²=0.977
- L=8192: R²=0.923

**Write Bottleneck (Figure 6):**
- r₉₅ = 2

**Axis-Aligned Neurons (Table 3):**
- Block 2 post-attn: 1 neuron
- Block 2 LN2: 61 neurons
- Block 2 MLP out: 238 neurons

---

## Actual Results

### Training Convergence
**Status**: In Progress
**Completion**: TBD

- [ ] iter_num ≥ 19000
- [ ] best_val_loss ≈ 0.57 (±0.1)
- [ ] Final train loss: TBD
- [ ] Final val loss: TBD

### BOS Heads Verification
**Status**: Pending (awaiting training completion)

- [ ] Heads 6 and 9 have >50% attention to position 0
- [ ] Other heads have <50% attention to position 0
- [ ] Figure 2 generated successfully

**Actual BOS heads identified**: TBD

### Performance Metrics
**Status**: Pending

- [ ] R² achieved: TBD (expected: 0.9996 or 0.966)
- [ ] MAE achieved: TBD (expected: 0.40)
- [ ] Loss achieved: TBD (expected: 0.57)

---

## Experiments to Re-Run (Post-Training)

### Must Re-Run (Use retrained R0 checkpoint):
1. ✅ **Figure 2**: BOS heads attention patterns
   - Script: `analysis_scripts/generate_improved_attention_plots.py`
   - Status: Pending

2. ✅ **Figure 3**: BOS intervention experiment
   - Script: Check `analysis_scripts/generate_bos_intervention_plot.py`
   - Status: Pending

3. ✅ **Table 3**: Axis-aligned neuron counts
   - Script: Search for neuron/axis-aligned scripts
   - Status: Pending

4. ✅ **Figure 5a**: Length extrapolation (R0 curve)
   - Script: Search for extrapolation scripts
   - Status: Pending

5. ✅ **Figure 6**: Write bottleneck (R0 curve)
   - Script: Search for SVD/subspace scripts
   - Status: Pending

6. ✅ **Appendix Figure**: Full attention maps for R0
   - Script: `analysis_scripts/generate_improved_attention_plots.py`
   - Status: Pending

### Can Keep (Use existing R1/R2/R3 checkpoints):
- Table 1: R1, R2, R3, R4 results (only R0 row needs update)
- Figure 5a: R2 extrapolation curve (for comparison)
- Figure 6: R2 write bottleneck curve (for comparison)

---

## Discrepancies Found

### R² Value Discrepancy
- **Table 1**: Reports R²=0.9996
- **Table 2 (Extrapolation)**: Reports R²=0.966 at L=128
- **Hypothesis**: Different evaluation protocols (train vs val set, or different sampling)
- **Action**: TBD after retraining

### BOS Head Consistency
- **Paper Assumption**: Heads 6 and 9
- **Verification Needed**: Check if this is consistent with seed=42
- **Action**: TBD after retraining

---

## Backup Information

**Broken Checkpoint Backup Location**: `nanoGPT/out-2layer-mechanism/R0_backup_broken/`

**Broken Checkpoint Issues**:
- block_size: 2048 (should be 128)
- max_iters: 100 (should be 20000)
- iter_num: 0-100 (barely trained)
- No BOS heads emerged (all ~4-5% attention to position 0)

**R1/R2/R3 Status**: Intact with correct configuration (verified)

---

## Next Steps

1. ⏳ **Monitor Training** (~2-4 hours)
   - Check WandB for loss curves, R², MAE
   - Verify convergence at iter ~19000

2. ⏸️ **Verify BOS Heads**
   - Run `generate_improved_attention_plots.py`
   - Confirm heads 6 and 9 have >50% attention to position 0

3. ⏸️ **Run Dependent Experiments**
   - Generate all figures and tables that use R0
   - Compare results to paper claims

4. ⏸️ **Document Discrepancies**
   - Note any differences from paper
   - Update paper if necessary

---

**Last Updated**: 2026-01-25 17:25:00
**Updated By**: Claude (R0 retraining automation)
