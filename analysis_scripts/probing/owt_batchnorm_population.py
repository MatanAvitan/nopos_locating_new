"""
OWT BatchNorm Population Statistics Analysis

Tests whether BatchNorm preserves population-level positional information
better than LayerNorm by leveraging batch/running statistics.

Key experiments:
1. Population mean probe: Train on E_n[h_i] for each position i
2. Mean-subtracted probe: Train on (h - mu_i) to test individual sample signal
3. Compare inference mode (running stats) vs training mode (batch stats)
4. Extract and analyze BatchNorm running_mean/running_var correlation with position
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import Ridge
from scipy import stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "owt_comprehensive"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"


@dataclass
class ExperimentConfig:
    name: str
    checkpoint_path: str
    use_positional_embedding: bool
    use_batchnorm_ln2: bool
    skip_ln2: bool


# Compare BN vs LN
EXPERIMENTS = [
    ExperimentConfig(
        name="NoPE + LayerNorm",
        checkpoint_path="out-nope-owt-ln/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=False,
        skip_ln2=False,
    ),
    ExperimentConfig(
        name="NoPE + BatchNorm2",
        checkpoint_path="out-nope-owt-bn2/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=True,
        skip_ln2=False,
    ),
]


def load_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Load trained model from checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / exp.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 512),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=exp.use_positional_embedding,
        norm_type=model_args.get("norm_type", "layernorm"),
        bias=model_args.get("bias", False),
        skip_ln2=exp.skip_ln2,
        use_batchnorm_ln2=exp.use_batchnorm_ln2,
    )

    model = GPT(config)

    state_dict = checkpoint["model"]
    unwrapped = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[10:]
        unwrapped[k] = v

    model.load_state_dict(unwrapped)
    model.to(DEVICE)

    return model, config


def get_post_ln2_activations(
    model: GPT,
    tokens: torch.Tensor,
    training_mode: bool = False,
) -> torch.Tensor:
    """Get post-LN2 activations in either training or inference mode."""
    if training_mode:
        model.train()
    else:
        model.eval()

    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)
        x = tok_emb

        block = model.transformer.h[0]

        x_ln1 = block.ln_1(x)
        attn_out = block.attn(x_ln1)
        x = x + attn_out

        if hasattr(block, 'ln_2'):
            x_ln2 = block.ln_2(x)
        else:
            x_ln2 = x

    return x_ln2.detach()


def collect_activations_by_position(
    model: GPT,
    config: GPTConfig,
    n_samples: int = 500,
    ctx: int = 512,
    training_mode: bool = False,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Collect activations organized by position.

    Returns:
        activations_by_pos: Dict mapping position -> (n_samples, d_model) array
        all_activations: (n_samples * ctx, d_model) array
    """
    vocab_size = config.vocab_size

    activations_by_pos = {i: [] for i in range(ctx)}
    all_activations = []
    all_positions = []

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, ctx), device=DEVICE)
        acts = get_post_ln2_activations(model, tokens, training_mode=training_mode)
        acts_np = acts[0].cpu().numpy()  # (ctx, d_model)

        for i in range(ctx):
            activations_by_pos[i].append(acts_np[i])

        all_activations.append(acts_np)
        all_positions.append(np.arange(ctx))

    # Stack
    for i in range(ctx):
        activations_by_pos[i] = np.vstack(activations_by_pos[i])

    all_activations = np.vstack(all_activations)
    all_positions = np.concatenate(all_positions)

    return activations_by_pos, all_activations, all_positions


def compute_population_means(activations_by_pos: Dict[int, np.ndarray]) -> np.ndarray:
    """Compute population mean for each position."""
    ctx = len(activations_by_pos)
    d_model = activations_by_pos[0].shape[1]

    pop_means = np.zeros((ctx, d_model))
    for i in range(ctx):
        pop_means[i] = activations_by_pos[i].mean(axis=0)

    return pop_means


def train_population_mean_probe(pop_means: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Train probe using only population means.
    This tests if population-level patterns are sufficient for position prediction.
    """
    ctx = pop_means.shape[0]
    positions = np.arange(ctx)

    # Use all positions for training (since we have one mean per position)
    probe = Ridge(alpha=1.0)
    probe.fit(pop_means, positions)
    y_pred = probe.predict(pop_means)

    # Compute R² on training data (all we have)
    ss_res = np.sum((positions - y_pred) ** 2)
    ss_tot = np.sum((positions - np.mean(positions)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Also compute correlation
    corr, _ = stats.pearsonr(y_pred, positions)

    return float(max(0, r2)), float(corr), y_pred


def train_mean_subtracted_probe(
    activations_by_pos: Dict[int, np.ndarray],
    pop_means: np.ndarray,
) -> Tuple[float, float]:
    """
    Train probe on mean-subtracted activations.
    Tests if individual sample signal (residual) is sufficient for position.
    """
    ctx = len(activations_by_pos)
    n_samples = activations_by_pos[0].shape[0]

    # Subtract population mean from each position's activations
    residuals = []
    positions = []
    for i in range(ctx):
        residual = activations_by_pos[i] - pop_means[i]
        residuals.append(residual)
        positions.append(np.full(n_samples, i))

    residuals = np.vstack(residuals)
    positions = np.concatenate(positions)

    # Train/test split
    n_total = len(positions)
    n_train = int(0.8 * n_total)
    idx = np.random.permutation(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    probe = Ridge(alpha=1.0)
    probe.fit(residuals[train_idx], positions[train_idx])
    y_pred = probe.predict(residuals[test_idx])

    ss_res = np.sum((positions[test_idx] - y_pred) ** 2)
    ss_tot = np.sum((positions[test_idx] - np.mean(positions[test_idx])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Correlation
    corr, _ = stats.pearsonr(y_pred, positions[test_idx])

    return float(max(0, r2)), float(corr)


def train_baseline_probe(
    all_activations: np.ndarray,
    all_positions: np.ndarray,
) -> Tuple[float, float]:
    """Train baseline probe on raw activations."""
    n_total = len(all_positions)
    n_train = int(0.8 * n_total)
    idx = np.random.permutation(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    probe = Ridge(alpha=1.0)
    probe.fit(all_activations[train_idx], all_positions[train_idx])
    y_pred = probe.predict(all_activations[test_idx])

    ss_res = np.sum((all_positions[test_idx] - y_pred) ** 2)
    ss_tot = np.sum((all_positions[test_idx] - np.mean(all_positions[test_idx])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    corr, _ = stats.pearsonr(y_pred, all_positions[test_idx])

    return float(max(0, r2)), float(corr)


def analyze_batchnorm_running_stats(model: GPT) -> Optional[Dict]:
    """Analyze BatchNorm running statistics correlation with position."""
    block = model.transformer.h[0]

    if not hasattr(block, 'ln_2'):
        return None

    ln_2 = block.ln_2

    # Check if it's BatchNorm
    if not hasattr(ln_2, 'bn'):
        return None

    bn = ln_2.bn

    running_mean = bn.running_mean.cpu().numpy()
    running_var = bn.running_var.cpu().numpy()

    # The running stats are per-channel, not per-position
    # But we can analyze their distribution
    return {
        "running_mean_mean": float(running_mean.mean()),
        "running_mean_std": float(running_mean.std()),
        "running_var_mean": float(running_var.mean()),
        "running_var_std": float(running_var.std()),
        "running_mean_range": [float(running_mean.min()), float(running_mean.max())],
        "running_var_range": [float(running_var.min()), float(running_var.max())],
    }


def analyze_population_statistics(
    model: GPT,
    config: GPTConfig,
    exp_name: str,
    n_samples: int = 500,
    ctx: int = 512,
) -> Dict:
    """Full population statistics analysis for a model."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing population statistics: {exp_name}")
    print(f"{'=' * 60}")

    results = {"experiment": exp_name, "n_samples": n_samples, "context": ctx}

    # Collect activations in inference mode
    print("Collecting activations (inference mode)...")
    acts_by_pos_infer, all_acts_infer, all_pos = collect_activations_by_position(
        model, config, n_samples=n_samples, ctx=ctx, training_mode=False
    )

    # Compute population means
    pop_means = compute_population_means(acts_by_pos_infer)

    # 1. Baseline probe (raw activations)
    print("Training baseline probe...")
    baseline_r2, baseline_corr = train_baseline_probe(all_acts_infer, all_pos)
    results["baseline_probe"] = {"r2": baseline_r2, "corr": baseline_corr}
    print(f"  Baseline: R²={baseline_r2:.4f}, r={baseline_corr:.4f}")

    # 2. Population mean probe
    print("Training population mean probe...")
    pop_mean_r2, pop_mean_corr, pop_mean_pred = train_population_mean_probe(pop_means)
    results["population_mean_probe"] = {"r2": pop_mean_r2, "corr": pop_mean_corr}
    print(f"  Population Mean: R²={pop_mean_r2:.4f}, r={pop_mean_corr:.4f}")

    # 3. Mean-subtracted probe (residual)
    print("Training mean-subtracted probe...")
    residual_r2, residual_corr = train_mean_subtracted_probe(acts_by_pos_infer, pop_means)
    results["residual_probe"] = {"r2": residual_r2, "corr": residual_corr}
    print(f"  Residual (h - μ): R²={residual_r2:.4f}, r={residual_corr:.4f}")

    # 4. Compare inference vs training mode (for BatchNorm)
    print("Collecting activations (training mode)...")
    acts_by_pos_train, all_acts_train, _ = collect_activations_by_position(
        model, config, n_samples=n_samples, ctx=ctx, training_mode=True
    )

    pop_means_train = compute_population_means(acts_by_pos_train)

    baseline_r2_train, baseline_corr_train = train_baseline_probe(all_acts_train, all_pos)
    pop_mean_r2_train, pop_mean_corr_train, _ = train_population_mean_probe(pop_means_train)
    residual_r2_train, residual_corr_train = train_mean_subtracted_probe(acts_by_pos_train, pop_means_train)

    results["training_mode"] = {
        "baseline_probe": {"r2": baseline_r2_train, "corr": baseline_corr_train},
        "population_mean_probe": {"r2": pop_mean_r2_train, "corr": pop_mean_corr_train},
        "residual_probe": {"r2": residual_r2_train, "corr": residual_corr_train},
    }

    print(f"\n  [Training Mode]")
    print(f"  Baseline: R²={baseline_r2_train:.4f}, r={baseline_corr_train:.4f}")
    print(f"  Population Mean: R²={pop_mean_r2_train:.4f}, r={pop_mean_corr_train:.4f}")
    print(f"  Residual (h - μ): R²={residual_r2_train:.4f}, r={residual_corr_train:.4f}")

    # 5. Mode difference
    results["mode_difference"] = {
        "baseline_r2_diff": baseline_r2_train - baseline_r2,
        "pop_mean_r2_diff": pop_mean_r2_train - pop_mean_r2,
        "residual_r2_diff": residual_r2_train - residual_r2,
    }

    # 6. BatchNorm running stats analysis
    bn_stats = analyze_batchnorm_running_stats(model)
    if bn_stats:
        results["batchnorm_running_stats"] = bn_stats
        print(f"\n  BatchNorm Running Stats:")
        print(f"    Mean: {bn_stats['running_mean_mean']:.4f} ± {bn_stats['running_mean_std']:.4f}")
        print(f"    Var:  {bn_stats['running_var_mean']:.4f} ± {bn_stats['running_var_std']:.4f}")

    # 7. Population mean norm by position
    pop_mean_norms = np.linalg.norm(pop_means, axis=1)
    pop_mean_norm_corr, _ = stats.pearsonr(pop_mean_norms, np.arange(ctx))
    results["population_mean_norm_position_corr"] = float(pop_mean_norm_corr)
    print(f"\n  Population Mean Norm vs Position: r={pop_mean_norm_corr:.4f}")

    return results


def main():
    print("=" * 70)
    print("OWT BATCHNORM POPULATION STATISTICS ANALYSIS")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'#' * 70}")
        print(f"# {exp.name}")
        print(f"{'#' * 70}")

        try:
            model, config = load_model(exp)
        except FileNotFoundError as e:
            print(f"Skipping {exp.name}: {e}")
            continue

        results = analyze_population_statistics(
            model, config, exp.name,
            n_samples=500, ctx=512,
        )

        all_results[exp.name] = results

        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = RESULTS_DIR / "owt_batchnorm_population_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary comparison
    print(f"\n{'=' * 70}")
    print("SUMMARY: BatchNorm vs LayerNorm Population Statistics")
    print("=" * 70)

    print(f"\n{'Experiment':<25} {'Baseline R²':>12} {'Pop Mean R²':>12} {'Residual R²':>12}")
    print("-" * 63)

    for name, data in all_results.items():
        baseline = data["baseline_probe"]["r2"]
        pop_mean = data["population_mean_probe"]["r2"]
        residual = data["residual_probe"]["r2"]
        print(f"{name:<25} {baseline:>12.4f} {pop_mean:>12.4f} {residual:>12.4f}")

    print(f"\n{'Experiment':<25} {'Mode Diff (Baseline)':>20} {'Mode Diff (Pop Mean)':>20}")
    print("-" * 67)

    for name, data in all_results.items():
        if "mode_difference" in data:
            baseline_diff = data["mode_difference"]["baseline_r2_diff"]
            pop_diff = data["mode_difference"]["pop_mean_r2_diff"]
            print(f"{name:<25} {baseline_diff:>+20.4f} {pop_diff:>+20.4f}")

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    results = main()
