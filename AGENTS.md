# AGENTS.md - Guidelines for AI Coding Agents

This document provides guidelines for AI agents working on the NoPE (No Positional Embedding) research codebase.

## Project Overview

Research project studying how causal transformers encode position without explicit positional embeddings. The codebase contains:
- Training scripts for NoPE transformers (nanoGPT-based)
- Analysis scripts for hypothesis testing
- LaTeX paper in `overleaf/nopos---claude-version/`
- Plotting utilities for publication-quality figures

## Build/Run Commands

### Environment Setup
```bash
# No requirements.txt - install dependencies manually:
pip install torch pytorch-lightning transformer_lens transformers datasets plotly matplotlib scipy numpy einops kaleido
```

### Training NoPE Models
```bash
# Prepare data first (from nanoGPT/ directory)
cd nanoGPT/data/shakespeare && python prepare.py

# Train with LayerNorm
CUDA_VISIBLE_DEVICES=0 python train_nope.py config/train_nope_1layer_ln.py

# Train with RMSNorm
CUDA_VISIBLE_DEVICES=1 python train_nope.py config/train_nope_1layer_rms.py
```

### Running Analysis Scripts
```bash
# Analyze trained checkpoints (with CLI arguments)
python analysis_scripts/analyze_trained_nope.py --checkpoint nanoGPT/out-nope-1layer-ln/ckpt.pt --save_dir results/ --n_samples 1000

# Most scripts are standalone - run directly
python analysis_scripts/<script_name>.py

# Run full experiment pipeline
./run_all_experiments.sh
```

### LaTeX Compilation
```bash
cd overleaf/nopos---claude-version && pdflatex acl_latex.tex
```

## Testing

**No formal test suite exists.** Scripts are validated by running them and checking outputs.

## Code Style Guidelines

### Import Order
1. Standard library (os, math, dataclasses)
2. Third-party (torch, numpy, scipy, plotly)
3. Local imports (model_nope, utils)

### Naming Conventions
- **Classes**: PascalCase (`CausalSelfAttention`, `LayerNorm`)
- **Functions/variables**: snake_case (`compute_attention_pattern`, `n_samples`)
- **Constants**: ALL_CAPS (`N_CTX`, `D_MODEL`, `BATCH_SIZE`)
- **Private methods**: single underscore prefix (`_init_weights`)

### Type Hints
Use type hints for function signatures and dataclasses:
```python
from typing import Literal
from dataclasses import dataclass

@dataclass
class GPTConfig:
    n_layer: int = 12
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"

def load_model(checkpoint_path: str, device: str = "cuda") -> tuple[GPT, dict]:
    ...
```

### Docstrings
Use triple-quoted docstrings for modules, classes, and functions:
```python
"""Module-level docstring explaining the script's purpose."""

def analyze_attention(model, n_samples=1000):
    """Analyze attention patterns for uniformity.
    
    Args:
        model: The GPT model to analyze
        n_samples: Number of random sequences to generate
    
    Returns:
        Dictionary with correlation scores per attention head
    """
```

### Device & Error Handling
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    _, cache = model.run_with_cache(tokens)
except RuntimeError as e:
    if 'CUDA out of memory' in str(e):
        torch.cuda.empty_cache()
    else:
        raise
```

### Formatting
- **Line length**: ~100-120 chars (no strict limit)
- **String quotes**: Prefer double quotes
- **Indentation**: 4 spaces
- **Blank lines**: 2 between top-level definitions, 1 within classes

### Plotting (Plotly)
```python
fig = px.line(data, template="plotly_white")
fig.update_layout(title=dict(text=title, font=dict(size=24, family="Serif")), width=800, height=500)
fig.write_image(f"{path}.png", width=800, height=500, scale=2)  # 300 DPI
fig.write_image(f"{path}.pdf")
```

## Project-Specific Conventions

### Layer Naming
Use component names, NOT layer indices:
- "post-LN", "post-attention", "post-MLP" (correct)
- "layer 0", "layer 1" (avoid)

### Model Configuration (NoPE)
```python
use_positional_embedding = False  # Critical for NoPE
norm_type = "layernorm"  # or "rmsnorm" for comparison
log_attention_stats = True  # For analysis
dropout = 0.0  # Clean analysis
```

### Reproducibility
```python
torch.manual_seed(42)
np.random.seed(42)
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `nanoGPT/model_nope.py` | NoPE GPT model (LN/RMSNorm variants) |
| `nanoGPT/train_nope.py` | Training script with attention logging |
| `utils.py` | Plotting utilities (line, imshow, model manipulation) |
| `analysis_scripts/analyze_trained_nope.py` | Hypothesis testing for trained models |
| `CLAUDE.md` | Detailed project context (paper overview, hypotheses) |

## Trained Model Checkpoints

Training is **COMPLETE**. Both models are saved:
- **LayerNorm**: `nanoGPT/out-nope-1layer-ln/ckpt.pt` (548MB, 5000 steps)
- **RMSNorm**: `nanoGPT/out-nope-1layer-rms/ckpt.pt` (548MB, 5000 steps)

Intermediate checkpoints available every 250 steps in the same directories.

## Common Gotchas

1. **Working directory**: nanoGPT scripts assume you're in the `nanoGPT/` directory
2. **Data path**: Training expects `data/<dataset>/train.bin` relative to nanoGPT/
3. **Checkpoint format**: Uses `_orig_mod.` prefix when torch.compile is used
4. **Flash attention**: Disabled when `log_attention_stats=True` to capture weights
5. **BFloat16**: Use `dtype="bfloat16"` for A100 GPUs, `float16` for older GPUs

## Directory Structure

```
nopos_locating_new/
├── analysis_scripts/       # Analysis and figure generation
├── nanoGPT/               # Training framework
│   ├── config/            # Training configurations
│   ├── data/              # Dataset preparation scripts
│   └── model_nope.py      # Main model file
├── overleaf/              # LaTeX paper source
├── train/                 # Additional training scripts
├── logs/                  # Training logs
├── models/                # Saved checkpoints
├── results/               # Experiment outputs
└── slurm_jobs/            # Slurm job submission scripts
```

## Slurm Cluster Usage

The BIU Slurm cluster is available for running GPU jobs. This is the preferred method for running long-running experiments.

### SSH Access

Connect to the Slurm login node using the `dsinlp01_id_rsa` SSH key:

```bash
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il
```

### Available Partitions

| Partition | Max Time | GPUs | Notes |
|-----------|----------|------|-------|
| `H200-4h` | 4h | 2 | **Recommended** - H200 GPUs on hpc8h200-01 |
| `H200-12h` | 12h | 2 | H200 GPUs for longer jobs |
| `generic` | 4h | 2 per job, 4 jobs max | General GPU jobs |
| `A100-4h` | 4h | 2 | A100 GPUs |
| `cpu1T-24h` | 24h | - | CPU jobs, 1TB RAM |
| `cpu192G-48h` | 48h | - | CPU jobs, 192GB RAM |

**Preferred partitions**: Use `H200-4h` or `H200-12h` for best GPU performance (node: hpc8h200-01).

### Submitting Jobs

Pre-made Slurm scripts are in `slurm_jobs/`:

```bash
# Submit all experiments at once
./slurm_jobs/submit_all_jobs.sh

# Or submit individual jobs
sbatch slurm_jobs/run_comprehensive_probe.sh
sbatch slurm_jobs/run_higher_order_stats.sh
sbatch slurm_jobs/run_decoding_vector.sh
sbatch slurm_jobs/run_causal_intervention.sh
sbatch slurm_jobs/run_training_dynamics.sh
sbatch slurm_jobs/run_token_position_correlation.sh
```

### Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

cd /home/nlp/matan_avitan/git/nopos_locating_new
python analysis_scripts/my_script.py
```

### Monitoring Jobs

```bash
# Check job queue
squeue -u $USER

# View job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# Use the monitor script (from login node)
./slurm_jobs/monitor.sh
```

### Running Slurm Commands Remotely

From the local machine, you can run Slurm commands via SSH:

```bash
# Check job status
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il "squeue -u \$USER"

# Submit a job
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il "cd /home/nlp/matan_avitan/git/nopos_locating_new && sbatch slurm_jobs/run_comprehensive_probe.sh"

# Check logs
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il "tail -20 /home/nlp/matan_avitan/git/nopos_locating_new/logs/slurm_*.out"
```

### Important Notes

1. **Default resources**: Jobs get 1 CPU and 16GB RAM by default; specify more with `--cpus-per-task` and `--mem`
2. **GPU allocation**: Must explicitly request GPUs with `--gres=gpu:N`
3. **Time limits**: Jobs exceeding time limits are suspended and requeued; implement checkpointing for long jobs
4. **Max jobs**: `generic` partition allows max 4 concurrent jobs per user
5. **Log files**: Output goes to `logs/slurm_<jobname>_<jobid>.out` and `.err`

## Experiment Results Summary

### Completed Experiments (as of Jan 11, 2026)

Results are stored in `results/<experiment_name>/`:

| Experiment | Status | Key Findings |
|------------|--------|--------------|
| `norm_intervention` | ✅ Complete | Full R²=0.04 pre-LN, R²=0.76 post-LN; direction R²=0.80 pre-LN |
| `layernorm_geometry` | ✅ Complete | LN linearizes position encoding, doesn't amplify |
| `single_sample_analysis` | ✅ Complete | Norm-position correlation: -0.76 post-attn, -0.97 post-LN |
| `decoding_vector_experiments` | ✅ Complete | Decoding vector correlates with position |
| `causal_interventions` | ✅ Complete | Attention intervention results |
| `token_position_correlation` | ✅ Complete | Natural language position correlations |
| `higher_order_statistics` | ✅ Complete | Eigenvalue analysis by position |
| `training_dynamics` | ✅ Complete | How position encoding emerges during training |
| `neuron_subgroup_analysis` | ✅ Complete | Per-neuron position correlation distribution |
| `attention_pattern_analysis` | ✅ Complete | Attention uniformity analysis |
| `comprehensive_probe_analysis` | 🔄 Running | Full probe analysis (job 991413) |
| `direction_norm_independence` | ✅ Complete | **Key source for Table 5**: dir/norm/full R² at each layer |
| `trained_model_analysis` | ✅ Complete | Random vs trained comparison on Shakespeare |

### Key Research Findings

#### The Core Mechanism
In NoPE transformers, causal attention naturally creates position-dependent activation patterns:
- Position i averages i+1 embeddings
- This creates a variance signal that decays as 1/(i+1)
- Correlation with theory: r = 0.999

#### LayerNorm's Role (CORRECTED)
**Previous (incorrect) claim**: "LayerNorm transforms variance into mean and amplifies position signal"

**Correct interpretation**: LayerNorm **linearizes** (not amplifies) the position encoding:
- **Pre-LN (post_attn)**: Full R² = 0.04, Direction R² = 0.39, Norm R² = 0.56
- **Post-LN (post_ln2)**: Full R² = 0.19, Direction R² = 0.19, Norm R² = 0.88
- **Post-MLP (post_mlp_residual)**: Full R² = 0.22, Direction R² = 0.35, Norm R² = 0.39

The position information exists pre-LN but in a complex, non-linear form (encoded in "directional structure"). LayerNorm makes it trivially decodable by a linear probe on the norm.

#### Layer-by-Layer Position Encoding (from direction_norm_independence_results.json)

| Layer | Full R² | Norm R² | Direction R² |
|-------|---------|---------|--------------|
| embed | ~0 | ~0 | ~0 |
| post_ln1 | ~0 | ~0 | ~0 |
| post_attn | 0.04 | 0.56 | 0.39 |
| post_ln2 | 0.19 | 0.88 | 0.19 |
| post_mlp_residual | 0.22 | 0.39 | 0.35 |

### Known Issues / Bugs to Fix

1. **`trained_model_direction_norm.py`** - Array dimension mismatch bug:
   - Line 140: `acts["post_attn"][0]` incorrectly indexes again after batch dim already removed
   - Fix: Change `acts["post_attn"][0]` to `acts["post_attn"]` (remove the `[0]`)
   - Same fix needed for `acts["post_ln2"][0]` on line 141

2. **`direction_norm_independence.py`** - Script runs but doesn't save JSON output
   - The script completes but results aren't persisted
   - Need to verify JSON saving at end of script

3. **`long_context_analysis.py`** - No output saved
   - Script runs quickly but results folder is empty
   - May need to check file writing logic

### Open Research Questions

1. **What is "directional structure"?**
   - Direction R² = 0.39 pre-LN but Full R² = 0.04 (updated values from JSON)
   - Why does unit vector (direction) encode position better than full activation?
   - Hypothesis: Position is encoded in angular relationships, not magnitudes

2. **Why does Full R² ≠ Direction R² + Norm R²?**
   - Pre-LN: Full=0.04, Norm=0.56, Direction=0.39
   - These don't add up - suggests complex interaction

3. **How does training affect position encoding?**
   - Random vs trained models - does the mechanism change?
   - Training dynamics analysis partially addresses this

## Paper Status

LaTeX source: `overleaf/nopos---claude-version/acl_latex.tex`

### Latest Commit (Jan 11, 2026)
`622cc64` - Update R² values to match direction_norm_independence JSON results

Changes made:
- Line 382: Changed post-LN R² from 0.76 to R²_norm=0.88
- Line 387 (figure caption): Updated all R² values to match JSON:
  - post_attn: dir=0.39, norm=0.56, full=0.04
  - post_ln2: norm=0.88, dir=0.19
- Line 601: Fixed MLP paragraph - changed "direction dominates (0.75 vs 0.37)" to "direction recovers (0.35 vs 0.39)"

### CRITICAL: Experiment Conditions Mismatch - RESOLVED

**✅ RESOLVED**: Added Appendix documenting experimental conditions for all figures/tables (commit `f888a73`).

The paper now clearly documents that different experiments use different conditions:

| Config Name | Model | Tokens | Context | n_embd | Used For |
|-------------|-------|--------|---------|--------|----------|
| Synthetic-Small | Random init | Uniform random | 64 | 256 | Table 5 (direction-norm) |
| Synthetic-Large | Random init | Uniform random | 64 | 768 | Most main figures |
| Trained | Trained 5K steps | Shakespeare | 256 | 768 | Random-vs-trained comparison |

**Key clarifications added to paper**:
1. New Appendix `\ref{app:experimental-conditions}` with full provenance table
2. Figure caption (line 578) now references appendix for correlation value explanation
3. Section explaining why R² values differ across experiments

**Remaining consideration**: The figure caption correlation values (r=-0.998, r=+0.86) come from `visualize_norm_over_positions.py` which uses uniform random tokens on context 64, while Table random-vs-trained uses Shakespeare on context 256. This is now documented but you may want to decide if experiments should be re-run with consistent settings.

### What Was Done in This Session

1. ✅ Updated R² values in main text (lines 382, 387, 601) to match `direction_norm_independence_results.json`
2. ✅ Fixed misleading "direction dominates" claim in line 601 (actual: dir=0.35 vs norm=0.39)
3. ✅ Committed and pushed changes (commit `622cc64`)
4. ✅ Added comprehensive Appendix documenting experimental conditions (commit `f888a73`)
5. ✅ Updated AGENTS.md with correct R² values and session summary

### What Still Could Be Done (Optional)

1. **Re-run experiments with consistent settings**: Either all synthetic or all Shakespeare
2. **Verify figure exactly matches caption**: Re-generate `norm_over_positions.png` and confirm r values match caption

### Key JSON Files for Reference

1. **Synthetic direction-norm experiment** (`results/direction_norm_independence/direction_norm_independence_results.json`):
   ```
   post_attn: direction_r2=0.393, norm_r2=0.556, full_r2=0.040
   post_ln2: direction_r2=0.193, norm_r2=0.880, full_r2=0.191
   post_mlp_residual: direction_r2=0.354, norm_r2=0.387, full_r2=0.216
   ```

2. **Random vs trained (Shakespeare)** (`results/trained_model_analysis/trained_model_results.json`):
   ```
   Random Init: post_ln2_norm_position_corr=-0.967, post_ln2_norm_r2=0.935
   LayerNorm trained: post_ln2_norm_position_corr=+0.154, post_ln2_norm_r2=0.024
   ```

**Repositories**:
- Main repo: `git@github.com:MatanAvitan/nopos_locating_new.git` (master branch)
- Overleaf repo: `git@github.com:MatanAvitan/nopos---claude-version.git` (main branch)
