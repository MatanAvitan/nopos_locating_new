"""
OWT Comprehensive Analysis: Direction vs Norm Position Encoding

Analyzes all 4 OWT-trained experiments to validate the positional encoding mechanism:
1. NoPE + LayerNorm
2. NoPE + BatchNorm2
3. NoPE + No LN2
4. Baseline + PE

For each experiment, compares trained model with random initialization to quantify
how training changes the position encoding mechanism.

Key metrics at each layer:
- Full activation R² (linear probe)
- Direction-only R² (unit vectors)
- Norm-only R² (scalar magnitudes)
- Norm-position Pearson correlation
- Direction/Norm R² ratio
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import Ridge
from scipy import stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "owt_comprehensive"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    checkpoint_path: str
    use_positional_embedding: bool
    use_batchnorm_ln2: bool
    skip_ln2: bool


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
    ExperimentConfig(
        name="Baseline + PE",
        checkpoint_path="out-baseline-owt-pe/ckpt.pt",
        use_positional_embedding=True,
        use_batchnorm_ln2=False,
        skip_ln2=False,
    ),
]


def load_trained_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Load a trained model from checkpoint."""
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

    # Unwrap torch.compile prefix if present
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
    """Create a randomly initialized model with the same config."""
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model


def get_activations(model: GPT, tokens: torch.Tensor, skip_ln2: bool = False) -> Dict[str, torch.Tensor]:
    """Get activations at key layers."""
    activations = {}

    with torch.no_grad():
        # Token embeddings
        tok_emb = model.transformer.wte(tokens)

        # Add positional embeddings if available
        if hasattr(model.transformer, 'wpe'):
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        activations["embed"] = x.detach()

        block = model.transformer.h[0]

        # Post LN1
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.detach()

        # Post attention (before residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.detach()

        # Post attention residual
        x = x + attn_out
        activations["post_attn_residual"] = x.detach()

        # Post LN2 (if not skipped)
        if not skip_ln2 and hasattr(block, 'ln_2'):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.detach()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.detach()  # Same as post_attn_residual
            mlp_input = x

        # Post MLP residual
        mlp_out = block.mlp(mlp_input)
        x = x + mlp_out
        activations["post_mlp_residual"] = x.detach()

    return activations


def compute_probing_metrics(
    activations: np.ndarray,
    positions: np.ndarray,
    train_ratio: float = 0.8,
) -> Dict[str, float]:
    """Compute all probing metrics for a set of activations."""
    n_samples = len(positions)
    n_train = int(train_ratio * n_samples)

    # Random split
    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Compute norms and directions
    norms = np.linalg.norm(activations, axis=1)
    directions = activations / (norms[:, np.newaxis] + 1e-8)

    # Train probes
    def fit_ridge(X_train, y_train, X_test, y_test, alpha=1.0):
        probe = Ridge(alpha=alpha)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return max(0, r2)

    # Full probe
    full_r2 = fit_ridge(
        activations[train_idx], positions[train_idx],
        activations[test_idx], positions[test_idx]
    )

    # Direction probe
    dir_r2 = fit_ridge(
        directions[train_idx], positions[train_idx],
        directions[test_idx], positions[test_idx]
    )

    # Norm probe
    norm_r2 = fit_ridge(
        norms[train_idx].reshape(-1, 1), positions[train_idx],
        norms[test_idx].reshape(-1, 1), positions[test_idx]
    )

    # Norm-position correlation
    norm_pos_corr, _ = stats.pearsonr(norms, positions)

    # Direction/Norm ratio
    dir_norm_ratio = dir_r2 / (norm_r2 + 1e-8)

    return {
        "full_r2": float(full_r2),
        "direction_r2": float(dir_r2),
        "norm_r2": float(norm_r2),
        "norm_position_corr": float(norm_pos_corr),
        "direction_norm_ratio": float(dir_norm_ratio),
    }


def analyze_model(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    skip_ln2: bool = False,
    n_samples: int = 500,
    context_length: Optional[int] = None,
) -> Dict:
    """Run full analysis on a model."""
    ctx = context_length or config.block_size
    vocab_size = config.vocab_size

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(f"Config: {config.n_layer}L, {config.n_head}H, {config.n_embd}D, ctx={ctx}")
    print(f"skip_ln2={skip_ln2}, use_batchnorm_ln2={getattr(config, 'use_batchnorm_ln2', False)}")
    print(f"{'=' * 60}")

    layers = ["embed", "post_ln1", "post_attn", "post_attn_residual", "post_ln2", "post_mlp_residual"]

    # Collect activations
    all_activations = {layer: [] for layer in layers}
    all_positions = []

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, ctx), device=DEVICE)
        acts = get_activations(model, tokens, skip_ln2=skip_ln2)

        for layer in layers:
            if layer in acts:
                all_activations[layer].append(acts[layer][0].cpu().numpy())

        all_positions.append(np.arange(ctx))

    # Stack activations
    for layer in layers:
        all_activations[layer] = np.vstack(all_activations[layer])

    positions = np.concatenate(all_positions)

    # Compute metrics for each layer
    results = {"model_name": model_name, "n_samples": n_samples, "context": ctx}

    print(f"\n{'Layer':<20} {'Full R²':>10} {'Dir R²':>10} {'Norm R²':>10} {'Norm-Pos r':>12} {'Dir/Norm':>10}")
    print("-" * 74)

    for layer in layers:
        metrics = compute_probing_metrics(all_activations[layer], positions)
        results[layer] = metrics

        print(f"{layer:<20} {metrics['full_r2']:>10.4f} {metrics['direction_r2']:>10.4f} "
              f"{metrics['norm_r2']:>10.4f} {metrics['norm_position_corr']:>12.4f} "
              f"{metrics['direction_norm_ratio']:>10.2f}")

    return results


def compute_attention_uniformity(model: GPT, n_samples: int = 100, ctx: int = 512) -> Dict:
    """Compute attention uniformity metrics."""
    print("\nComputing attention uniformity...")

    vocab_size = model.config.vocab_size
    all_uniformity = []

    # Temporarily enable attention logging
    block = model.transformer.h[0]
    original_log_setting = block.attn.log_attention_stats
    block.attn.log_attention_stats = True

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, ctx), device=DEVICE)
        with torch.no_grad():
            _ = model(tokens)

        if block.attn.last_attention_uniformity is not None:
            all_uniformity.append(block.attn.last_attention_uniformity)

    block.attn.log_attention_stats = original_log_setting

    if all_uniformity:
        uniformity = np.array(all_uniformity)
        mean_uniformity = uniformity.mean(axis=0)
        return {
            "per_head_uniformity": mean_uniformity.tolist(),
            "mean_uniformity": float(mean_uniformity.mean()),
            "std_uniformity": float(mean_uniformity.std()),
        }
    return {"per_head_uniformity": [], "mean_uniformity": 0.0, "std_uniformity": 0.0}


def main():
    print("=" * 70)
    print("OWT COMPREHENSIVE ANALYSIS: Direction vs Norm Position Encoding")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'#' * 70}")
        print(f"# Experiment: {exp.name}")
        print(f"{'#' * 70}")

        # Load trained model
        try:
            trained_model, config = load_trained_model(exp)
            print(f"\nLoaded checkpoint: {exp.checkpoint_path}")
        except FileNotFoundError as e:
            print(f"\nSkipping {exp.name}: {e}")
            continue

        # Analyze trained model
        trained_results = analyze_model(
            trained_model, config, f"{exp.name} (trained)",
            skip_ln2=exp.skip_ln2,
            n_samples=500,
            context_length=512,  # Use 512 for consistent comparison
        )

        # Compute attention uniformity for trained model
        attn_uniformity = compute_attention_uniformity(trained_model, n_samples=50, ctx=512)
        trained_results["attention_uniformity"] = attn_uniformity
        print(f"  Attention uniformity: mean={attn_uniformity['mean_uniformity']:.4f}")

        # Create and analyze random model
        random_model = create_random_model(config)
        random_results = analyze_model(
            random_model, config, f"{exp.name} (random)",
            skip_ln2=exp.skip_ln2,
            n_samples=500,
            context_length=512,
        )

        # Compute attention uniformity for random model
        random_attn_uniformity = compute_attention_uniformity(random_model, n_samples=50, ctx=512)
        random_results["attention_uniformity"] = random_attn_uniformity
        print(f"  Random attention uniformity: mean={random_attn_uniformity['mean_uniformity']:.4f}")

        # Store results
        all_results[exp.name] = {
            "trained": trained_results,
            "random": random_results,
            "config": {
                "use_positional_embedding": exp.use_positional_embedding,
                "use_batchnorm_ln2": exp.use_batchnorm_ln2,
                "skip_ln2": exp.skip_ln2,
            }
        }

        # Print comparison summary
        print(f"\n--- Trained vs Random Comparison for {exp.name} ---")
        print(f"{'Layer':<20} {'Trained Norm R²':>15} {'Random Norm R²':>15} {'Change':>10}")
        print("-" * 62)
        for layer in ["post_attn", "post_ln2", "post_mlp_residual"]:
            if layer in trained_results and layer in random_results:
                t_norm = trained_results[layer]["norm_r2"]
                r_norm = random_results[layer]["norm_r2"]
                change = t_norm - r_norm
                print(f"{layer:<20} {t_norm:>15.4f} {r_norm:>15.4f} {change:>+10.4f}")

        # Clean up
        del trained_model, random_model
        torch.cuda.empty_cache()

    # Save results
    output_path = RESULTS_DIR / "owt_comprehensive_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 70}")

    # Print final summary table
    print("\n" + "=" * 90)
    print("FINAL SUMMARY: Post-LN2 Layer (Trained Models)")
    print("=" * 90)
    print(f"{'Experiment':<25} {'Full R²':>10} {'Dir R²':>10} {'Norm R²':>10} {'Norm-Pos r':>12} {'Attn Unif':>10}")
    print("-" * 79)

    for name, data in all_results.items():
        if "trained" in data and "post_ln2" in data["trained"]:
            t = data["trained"]["post_ln2"]
            au = data["trained"].get("attention_uniformity", {}).get("mean_uniformity", 0)
            print(f"{name:<25} {t['full_r2']:>10.4f} {t['direction_r2']:>10.4f} "
                  f"{t['norm_r2']:>10.4f} {t['norm_position_corr']:>12.4f} {au:>10.4f}")

    return all_results


if __name__ == "__main__":
    results = main()
