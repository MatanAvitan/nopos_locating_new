"""
OWT LayerNorm Linearization Analysis

Tests the hypothesis that LayerNorm "linearizes" positional information
by transforming complex directional encoding into a simple norm-based signal.

Key hypothesis:
- Random init: Strong linearization (direction R² drops, norm R² jumps after LN)
- Trained: Weaker or inverted linearization effect

Compares across: LN, BN, No-LN2 variants
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


# Focus on the 3 NoPE variants for linearization analysis
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
    ExperimentConfig(
        name="NoPE + No LN2",
        checkpoint_path="out-nope-owt-no-ln2/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=False,
        skip_ln2=True,
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
    model.eval()
    model.to(DEVICE)

    return model, config


def create_random_model(config: GPTConfig) -> GPT:
    """Create randomly initialized model."""
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model


def get_pre_post_ln2_activations(
    model: GPT,
    tokens: torch.Tensor,
    skip_ln2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get activations before and after LN2."""
    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)

        if hasattr(model.transformer, 'wpe'):
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        block = model.transformer.h[0]

        x_ln1 = block.ln_1(x)
        attn_out = block.attn(x_ln1)
        x = x + attn_out

        # Pre-LN2 (post-attention residual)
        pre_ln2 = x.detach()

        # Post-LN2
        if not skip_ln2 and hasattr(block, 'ln_2'):
            post_ln2 = block.ln_2(x).detach()
        else:
            post_ln2 = x.detach()

    return pre_ln2, post_ln2


def compute_linearization_metrics(
    pre_ln2: np.ndarray,
    post_ln2: np.ndarray,
    positions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute linearization effect metrics.

    Linearization = how much positional info moves from direction to norm after LN2.
    """
    n_samples = len(positions)
    n_train = int(0.8 * n_samples)
    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def compute_r2(X, y, train_idx, test_idx, alpha=1.0):
        probe = Ridge(alpha=alpha)
        probe.fit(X[train_idx], y[train_idx])
        y_pred = probe.predict(X[test_idx])
        ss_res = np.sum((y[test_idx] - y_pred) ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
        return max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0

    def get_dir_norm(X):
        norms = np.linalg.norm(X, axis=1)
        dirs = X / (norms[:, np.newaxis] + 1e-8)
        return dirs, norms

    # Pre-LN2 metrics
    pre_dirs, pre_norms = get_dir_norm(pre_ln2)
    pre_full_r2 = compute_r2(pre_ln2, positions, train_idx, test_idx)
    pre_dir_r2 = compute_r2(pre_dirs, positions, train_idx, test_idx)
    pre_norm_r2 = compute_r2(pre_norms.reshape(-1, 1), positions, train_idx, test_idx)

    # Post-LN2 metrics
    post_dirs, post_norms = get_dir_norm(post_ln2)
    post_full_r2 = compute_r2(post_ln2, positions, train_idx, test_idx)
    post_dir_r2 = compute_r2(post_dirs, positions, train_idx, test_idx)
    post_norm_r2 = compute_r2(post_norms.reshape(-1, 1), positions, train_idx, test_idx)

    # Linearization effect
    # Positive = norm R² increased after LN (linearization occurred)
    # Negative = norm R² decreased after LN (inverse linearization)
    norm_r2_change = post_norm_r2 - pre_norm_r2
    dir_r2_change = post_dir_r2 - pre_dir_r2

    # Normalized linearization effect
    if pre_norm_r2 > 0.01:
        linearization_effect = norm_r2_change / pre_norm_r2
    else:
        linearization_effect = norm_r2_change * 10  # Scale when baseline is near zero

    # Position-norm correlation
    pre_norm_corr, _ = stats.pearsonr(pre_norms, positions)
    post_norm_corr, _ = stats.pearsonr(post_norms, positions)

    return {
        "pre_ln2": {
            "full_r2": float(pre_full_r2),
            "direction_r2": float(pre_dir_r2),
            "norm_r2": float(pre_norm_r2),
            "norm_position_corr": float(pre_norm_corr),
        },
        "post_ln2": {
            "full_r2": float(post_full_r2),
            "direction_r2": float(post_dir_r2),
            "norm_r2": float(post_norm_r2),
            "norm_position_corr": float(post_norm_corr),
        },
        "linearization": {
            "norm_r2_change": float(norm_r2_change),
            "direction_r2_change": float(dir_r2_change),
            "normalized_effect": float(linearization_effect),
            "norm_corr_change": float(post_norm_corr - pre_norm_corr),
        }
    }


def analyze_linearization(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    skip_ln2: bool = False,
    n_samples: int = 500,
    ctx: int = 512,
) -> Dict:
    """Analyze linearization effect for a model."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing linearization: {model_name}")
    print(f"skip_ln2={skip_ln2}")
    print(f"{'=' * 60}")

    vocab_size = config.vocab_size

    all_pre_ln2 = []
    all_post_ln2 = []
    all_positions = []

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, ctx), device=DEVICE)
        pre, post = get_pre_post_ln2_activations(model, tokens, skip_ln2=skip_ln2)

        all_pre_ln2.append(pre[0].cpu().numpy())
        all_post_ln2.append(post[0].cpu().numpy())
        all_positions.append(np.arange(ctx))

    pre_ln2 = np.vstack(all_pre_ln2)
    post_ln2 = np.vstack(all_post_ln2)
    positions = np.concatenate(all_positions)

    metrics = compute_linearization_metrics(pre_ln2, post_ln2, positions)
    metrics["model_name"] = model_name
    metrics["n_samples"] = n_samples
    metrics["context"] = ctx

    # Print results
    print(f"\nPre-LN2:  Full R²={metrics['pre_ln2']['full_r2']:.4f}, "
          f"Dir R²={metrics['pre_ln2']['direction_r2']:.4f}, "
          f"Norm R²={metrics['pre_ln2']['norm_r2']:.4f}, "
          f"Norm-Pos r={metrics['pre_ln2']['norm_position_corr']:.4f}")

    print(f"Post-LN2: Full R²={metrics['post_ln2']['full_r2']:.4f}, "
          f"Dir R²={metrics['post_ln2']['direction_r2']:.4f}, "
          f"Norm R²={metrics['post_ln2']['norm_r2']:.4f}, "
          f"Norm-Pos r={metrics['post_ln2']['norm_position_corr']:.4f}")

    lin = metrics['linearization']
    print(f"\nLinearization Effect:")
    print(f"  Norm R² change: {lin['norm_r2_change']:+.4f}")
    print(f"  Dir R² change:  {lin['direction_r2_change']:+.4f}")
    print(f"  Normalized effect: {lin['normalized_effect']:+.4f}")

    return metrics


def main():
    print("=" * 70)
    print("OWT LAYERNORM LINEARIZATION ANALYSIS")
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
            trained_model, config = load_model(exp)
        except FileNotFoundError as e:
            print(f"Skipping {exp.name}: {e}")
            continue

        # Trained model analysis
        trained_results = analyze_linearization(
            trained_model, config, f"{exp.name} (trained)",
            skip_ln2=exp.skip_ln2,
            n_samples=500, ctx=512,
        )

        # Random model analysis
        random_model = create_random_model(config)
        random_results = analyze_linearization(
            random_model, config, f"{exp.name} (random)",
            skip_ln2=exp.skip_ln2,
            n_samples=500, ctx=512,
        )

        all_results[exp.name] = {
            "trained": trained_results,
            "random": random_results,
        }

        # Print comparison
        print(f"\n--- Linearization Comparison: {exp.name} ---")
        t_lin = trained_results["linearization"]["normalized_effect"]
        r_lin = random_results["linearization"]["normalized_effect"]
        print(f"  Trained linearization effect: {t_lin:+.4f}")
        print(f"  Random linearization effect:  {r_lin:+.4f}")
        print(f"  Difference: {t_lin - r_lin:+.4f}")

        del trained_model, random_model
        torch.cuda.empty_cache()

    # Save results
    output_path = RESULTS_DIR / "owt_linearization_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY: Linearization Effects")
    print("=" * 70)

    print(f"\n{'Experiment':<25} {'Trained Lin':>12} {'Random Lin':>12} {'Trained Norm Δ':>14} {'Random Norm Δ':>14}")
    print("-" * 79)

    for name, data in all_results.items():
        t_lin = data["trained"]["linearization"]["normalized_effect"]
        r_lin = data["random"]["linearization"]["normalized_effect"]
        t_norm_change = data["trained"]["linearization"]["norm_r2_change"]
        r_norm_change = data["random"]["linearization"]["norm_r2_change"]
        print(f"{name:<25} {t_lin:>+12.4f} {r_lin:>+12.4f} {t_norm_change:>+14.4f} {r_norm_change:>+14.4f}")

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    results = main()
