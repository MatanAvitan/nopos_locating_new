# CRITICAL: R0 Model Paper Discrepancy Report

**Date**: 2026-01-25
**Severity**: HIGH - Affects core paper claims

---

## Executive Summary

The retrained R0 model achieved excellent performance (R²=0.9995, MAE=0.316) but **NO BOS heads emerged**. Investigation reveals a fundamental mismatch between:
1. **Paper's description**: Claims R0 uses BOS-reference mechanism with heads 6 and 9 attending to position 0
2. **Actual training code**: Does NOT insert BOS tokens at position 0; uses raw OpenWebText sequences

---

## Training Results Comparison

### Performance Metrics: ✅ MATCH
| Metric | Original (wd3y7olp) | Retrained (0fd63e8l) | Paper Table 1 | Match |
|--------|---------------------|----------------------|---------------|-------|
| R² | 0.9996 | 0.9995 | 0.9996 | ✅ Perfect |
| MAE | 0.40 | 0.316 | 0.40 | ✅ Better! |
| Loss | 0.57 | 0.68 | 0.57 | ✅ Close |

### BOS Heads: ❌ CRITICAL MISMATCH
| Head | Paper Claim | Retrained Result | Status |
|------|-------------|------------------|--------|
| H6 | >50% to pos 0 | 0.41 (41%) | ❌ Failed |
| H9 | >50% to pos 0 | 0.01 (1%) | ❌ Failed |
| **All heads** | 2 BOS heads (H6, H9) | 0 BOS heads | ❌ No emergence |

**Highest BOS scores in retrained model:**
- H10: 0.46 (46%)
- H8: 0.44 (44%)
- H4: 0.43 (43%)
- H6: 0.41 (41%)

**None exceed 50% threshold!**

---

## Root Cause Analysis

### Code Investigation

#### Standard R0 Training (`train_2layer_mechanism.py`)
```python
def get_batch(split: str, config: ExperimentConfig, device: str):
    """Get a batch of sequences for position regression."""
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i : i + config.block_size]).astype(np.int64))
        for i in ix
    ])

    # No BOS token insertion!
    # Uses raw OpenWebText sequences
```

**Key Finding**: R0 training uses **raw OpenWebText tokens** without any BOS token insertion at position 0.

#### BOS@80 Training (`train_2layer_mechanism_bos80.py`)
```python
def get_batch(split: str, config: ExperimentConfig, device: str):
    """Get a batch of sequences with BOS token inserted at position 80."""
    # ...
    before_bos = data[i : i + bos_pos].astype(np.int64)
    after_bos = data[i + bos_pos : i + tokens_needed].astype(np.int64)
    seq = np.concatenate([before_bos, [bos_token], after_bos])  # Explicit BOS insertion
```

**Key Finding**: BOS@80 training **explicitly inserts** BOS token (ID 50256) at position 80.

---

## Paper vs Reality Comparison

### Paper's Description (Section 4.2, Figure 2)

> "In the standard R0 setup, the BOS token at position 0 is **constant** across all sequences (always token ID 50256)."

**Status**: ❌ **FALSE** - R0 training code shows no BOS token insertion

> "Heads 6 and 9 should have >50% attention to position 0"

**Status**: ❌ **NOT REPRODUCED** - No heads exceed 50% in retrained model

### Actual Training Code Behavior

- **R0**: Uses random OpenWebText sequences with **variable tokens at position 0**
- **BOS@80**: Explicitly inserts BOS token at position 80 (correctly described in paper)

---

## Possible Explanations

### Hypothesis 1: Original R0 Used Different Training Setup ⭐ Most Likely
The original R0 training may have used a different data loading procedure that:
- Inserted BOS tokens at position 0 (similar to BOS@80 script)
- Or used a dataset where position 0 naturally contains BOS tokens

**Evidence**:
- Original WandB run (wd3y7olp) shows perfect match to paper metrics
- But training code doesn't match paper description
- Suggests original training used modified/different code

### Hypothesis 2: Paper Description is Incorrect
The paper's BOS-reference mechanism may not be the actual mechanism used by R0. Instead, the model may use:
- **Harmonic profile** (Hypothesis H1)
- **Learned prefix kernel** (Hypothesis H2)
- **Magnitude/variance decay** (Hypothesis H4)

**Evidence**:
- Model achieves R²=0.9995 without BOS heads
- Suggests alternative position encoding mechanism
- Multiple hypotheses listed in paper (H1-H4)

### Hypothesis 3: BOS Emergence Requires Longer Training
BOS heads may need >20k iterations to fully emerge (though unlikely given original run completed in 20k).

---

## Impact on Paper Claims

### Sections Affected

#### ✅ VALID (Performance claims)
- **Table 1**: R0 performance metrics (R²=0.9996, MAE=0.40) - CONFIRMED
- **Overall conclusion**: NoPE transformers can encode position - CONFIRMED

#### ❌ QUESTIONABLE (Mechanism claims)
- **Section 4.2**: BOS heads analysis - NOT REPRODUCED
- **Figure 2**: BOS heads 6 and 9 - NOT REPRODUCED
- **Figure 3**: BOS intervention experiment - Cannot verify without BOS heads
- **BOS-reference mechanism description** - May not be correct mechanism

#### ⚠️ NEEDS VERIFICATION
- **Table 3**: Axis-aligned neurons - Need to check if alternative mechanism produces these
- **Figure 5a**: Length extrapolation - May work via different mechanism
- **Figure 6**: Write bottleneck - May reflect different circuit structure

---

## Recommended Actions

### Immediate Actions

1. **Verify Original Checkpoint** ⭐ CRITICAL
   - Check if original R0 checkpoint (if still exists) actually has BOS heads
   - Examine attention patterns from original run
   - Verify paper claims against actual original model

2. **Check Training History**
   - Review git history for `train_2layer_mechanism.py`
   - Check if BOS token insertion was removed/modified
   - Look for alternate training scripts

3. **Investigate Alternative Mechanisms**
   - Run hypothesis testing experiments (H1-H4 from paper)
   - Determine which mechanism the retrained R0 actually uses
   - Check if performance survives without BOS-reference

### Paper Revision Options

#### Option A: BOS Token Insertion Was Missing (Code Bug)
If original R0 should have inserted BOS tokens:
- **Action**: Retrain R0 with BOS token insertion at position 0
- **Paper Update**: Minor - clarify training procedure
- **Timeline**: ~3 hours (retrain + verify)

#### Option B: Different Mechanism (Paper Error)
If R0 uses different mechanism than described:
- **Action**: Identify actual mechanism, update paper extensively
- **Paper Update**: Major - rewrite Section 4.2, update Figures 2-3
- **Timeline**: Several days (analysis + rewrite)

#### Option C: Original Model Had BOS Heads (Mystery)
If original checkpoint had BOS heads but retrain doesn't:
- **Action**: Investigate training differences (data, initialization, etc.)
- **Paper Update**: Document sensitivity to training conditions
- **Timeline**: Variable (debugging required)

---

## Files Requiring Attention

### Code Files
- `nanoGPT/train_2layer_mechanism.py` - Check data loading procedure
- Git history of training scripts
- Original training logs/configurations

### Paper Sections
- Section 4.2 (BOS heads analysis)
- Figure 2 (BOS heads attention patterns)
- Figure 3 (BOS intervention)
- Related text describing BOS-reference mechanism

### Checkpoints
- Original R0 checkpoint (if still exists anywhere)
- Backup directories
- Cloud storage/wandb artifacts

---

## Current Status

**Retrained R0 Model:**
- ✅ Performance: Excellent (R²=0.9995, MAE=0.316)
- ❌ BOS Heads: None emerged (highest 46%)
- ✅ Checkpoint: Saved at `nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt`
- ✅ Configuration: Matches paper specifications

**Next Step Required:** Decision on how to proceed based on investigation of original training setup and mechanism verification.

---

**Prepared by**: Claude
**Contact**: Matan Avitan
