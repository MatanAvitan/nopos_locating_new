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
