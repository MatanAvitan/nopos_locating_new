"""
Trained OWT Experiments: Direction vs Norm Probing Analysis

Analyzes how positional information moves through the network by probing
at different layers (post_attn, post_ln2, post_mlp_residual) with:
- Direction-only probes (unit vectors)
- Norm-only probes (activation magnitudes)
- Full activation probes

Compares across the 4 trained experiments:
1. NoPE + LayerNorm
2. NoPE + BatchNorm2
3. NoPE + No LN2
4. Baseline with PE
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "owt_direction_norm_probing"


def load_trained_model(
    checkpoint_path: str,
    use_positional_embedding: bool = False,
    use_batchnorm_ln2: bool = None,
    skip_ln2: bool = None,
) -> Tuple[GPT, GPTConfig]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model_args = checkpoint.get("model_args", {})

    # Use provided values or fall back to model_args
    bn2_setting = (
        use_batchnorm_ln2
        if use_batchnorm_ln2 is not None
        else model_args.get("use_batchnorm_ln2", False)
    )
    skip_setting = (
        skip_ln2 if skip_ln2 is not None else model_args.get("skip_ln2", False)
    )

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 512),
        vocab_size=model_args.get("vocab_size", 50257),
        dropout=0.0,
        use_positional_embedding=use_positional_embedding,
        norm_type=model_args.get("norm_type", "layernorm"),
        bias=model_args.get("bias", False),
        skip_ln2=skip_setting,
        use_batchnorm_ln2=bn2_setting,
    )

    model = GPT(config)

    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[10:]
        unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.eval()
    model.to(DEVICE)

    return model, config


def get_activations(model: GPT, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Get activations at key layers."""
    activations = {}

    tok_emb = model.transformer.wte(tokens)
    activations["embed"] = tok_emb.detach()

    block = model.transformer.h[0]

    x = block.ln_1(tok_emb)
    activations["post_ln1"] = x.detach()

    attn_out = block.attn(x)
    activations["post_attn"] = attn_out.detach()

    x = tok_emb + attn_out
    activations["post_attn_residual"] = x.detach()

    if hasattr(block, "ln_2"):
        x_ln2 = block.ln_2(x)
        activations["post_ln2"] = x_ln2.detach()
    else:
        activations["post_ln2"] = x.detach()

    mlp_out = block.mlp(x_ln2 if hasattr(block, "ln_2") else x)
    x = x + mlp_out
    activations["post_mlp_residual"] = x.detach()

    return activations


def extract_direction_and_norm(
    activations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Separate activations into direction (unit vectors) and norm (scalars)."""
    norms = torch.norm(activations, dim=-1, keepdim=True)
    directions = activations / (norms + 1e-8)
    return directions, norms.squeeze(-1)


def train_ridge_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """Train ridge regression probe and return R² on test set."""
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return max(0, r2), y_pred


def probe_layer(
    activations: np.ndarray,
    positions: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, float]:
    """Run all three probe types on a layer."""
    results = {}

    norms = np.linalg.norm(activations, axis=1)
    directions = activations / (norms[:, np.newaxis] + 1e-8)
    full = activations

    # Full probe
    r2_full, _ = train_ridge_probe(
        full[train_idx], positions[train_idx], full[test_idx], positions[test_idx]
    )
    results["full_r2"] = r2_full

    # Direction probe
    r2_dir, _ = train_ridge_probe(
        directions[train_idx],
        positions[train_idx],
        directions[test_idx],
        positions[test_idx],
    )
    results["direction_r2"] = r2_dir

    # Norm probe
    r2_norm, _ = train_ridge_probe(
        norms[train_idx].reshape(-1, 1),
        positions[train_idx],
        norms[test_idx].reshape(-1, 1),
        positions[test_idx],
    )
    results["norm_r2"] = r2_norm

    return results


def analyze_model_across_layers(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    n_samples: int = 500,
    context: int = 512,
) -> Dict:
    """Run full probe analysis across all layers."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(
        f"Config: {config.n_layer}L, {config.n_head}H, {config.n_embd}D, ctx={context}"
    )
    print(f"{'=' * 60}")

    layers = [
        "embed",
        "post_ln1",
        "post_attn",
        "post_attn_residual",
        "post_ln2",
        "post_mlp_residual",
    ]

    all_activations = {layer: [] for layer in layers}
    all_positions = []

    vocab_size = config.vocab_size

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, context), device=DEVICE)
        activations = get_activations(model, tokens)

        for layer in layers:
            if layer in activations:
                all_activations[layer].append(activations[layer][0].cpu().numpy())

        all_positions.append(np.arange(context))

    for layer in layers:
        all_activations[layer] = np.vstack(all_activations[layer])

    positions = np.concatenate(all_positions)

    n_train = int(0.8 * len(positions))
    idx = np.random.permutation(len(positions))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    results = {"model_name": model_name}

    print(
        f"\n{'Layer':<20} {'Full R²':>10} {'Dir R²':>10} {'Norm R²':>10} {'Dir/Norm':>10}"
    )
    print("-" * 62)

    for layer in layers:
        layer_results = probe_layer(
            all_activations[layer], positions, train_idx, test_idx
        )
        results[layer] = layer_results

        dir_norm_ratio = layer_results["direction_r2"] / (
            layer_results["norm_r2"] + 1e-8
        )
        print(
            f"{layer:<20} {layer_results['full_r2']:>10.4f} {layer_results['direction_r2']:>10.4f} "
            f"{layer_results['norm_r2']:>10.4f} {dir_norm_ratio:>10.4f}"
        )

    return results


def compute_decoding_vector(model: GPT) -> np.ndarray:
    """Compute the theoretical decoding vector w = W_V · Σ LN(E_j)."""
    E = model.transformer.wte.weight.detach()
    block = model.transformer.h[0]
    ln1 = block.ln_1

    E_centered = E - E.mean(dim=-1, keepdim=True)
    E_std = E.std(dim=-1, keepdim=True)
    E_ln = E_centered / (E_std + 1e-5) * ln1.weight
    if ln1.bias is not None:
        E_ln = E_ln + ln1.bias

    sum_ln_E = E_ln.sum(dim=0)

    attn = block.attn
    n_embd = model.config.n_embd
    W_V = attn.c_attn.weight[2 * n_embd :, :].detach()

    w = W_V @ sum_ln_E
    w = w / (torch.norm(w) + 1e-8)

    return w.detach().cpu().numpy()


def analyze_decoding_vector(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    n_samples: int = 200,
    context: int = 512,
) -> Dict:
    """Analyze how well the theoretical decoding vector correlates with position."""
    print(f"\nDecoding Vector Analysis for {model_name}...")

    w = compute_decoding_vector(model)

    vocab_size = config.vocab_size
    all_projs = []
    all_positions = []

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, context), device=DEVICE)
        activations = get_activations(model, tokens)

        post_ln2 = activations["post_ln2"][0].cpu().numpy()
        projs = post_ln2 @ w
        all_projs.append(projs)
        all_positions.append(np.arange(context))

    projs = np.concatenate(all_projs)
    positions = np.concatenate(all_positions)

    corr = np.corrcoef(projs, positions)[0, 1]

    print(f"  Decoding vector correlation with position: r = {corr:.4f}")

    return {"decoding_vector_corr": corr}


def main():
    print("=" * 70)
    print("OWT EXPERIMENTS: DIRECTION vs NORM PROBING ANALYSIS")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    experiments = [
        {
            "name": "NoPE + LayerNorm",
            "checkpoint": "nanoGPT/out-nope-owt-ln/ckpt.pt",
            "use_pe": False,
        },
        {
            "name": "NoPE + BatchNorm2",
            "checkpoint": "nanoGPT/out-nope-owt-bn2/ckpt.pt",
            "use_pe": False,
        },
        {
            "name": "Baseline + PE",
            "checkpoint": "nanoGPT/out-baseline-owt-pe/ckpt_05000.pt",
            "use_pe": True,
        },
    ]

    all_results = {}

    for exp in experiments:
        path = Path(__file__).parent.parent / exp["checkpoint"]

        if not path.exists():
            print(f"\nCheckpoint not found: {path}")
            continue

        try:
            model, config = load_trained_model(
                str(path),
                use_positional_embedding=exp["use_pe"],
            )

            results = analyze_model_across_layers(
                model, config, exp["name"], n_samples=300, context=config.block_size
            )

            dv_results = analyze_decoding_vector(
                model, config, exp["name"], n_samples=100, context=config.block_size
            )
            results["decoding_vector"] = dv_results

            all_results[exp["name"]] = results

        except Exception as e:
            print(f"\nError analyzing {exp['name']}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON: Post-LN2 Layer")
    print("=" * 70)

    print(
        f"\n{'Experiment':<25} {'Full R²':>10} {'Dir R²':>10} {'Norm R²':>10} {'DV Corr':>10}"
    )
    print("-" * 67)

    for name, res in all_results.items():
        if "post_ln2" in res:
            post_ln2 = res["post_ln2"]
            dv_corr = res.get("decoding_vector", {}).get("decoding_vector_corr", 0)
            print(
                f"{name:<25} {post_ln2['full_r2']:>10.4f} {post_ln2['direction_r2']:>10.4f} "
                f"{post_ln2['norm_r2']:>10.4f} {dv_corr:>10.4f}"
            )

    print("\n" + "=" * 70)
    print("INFORMATION FLOW: Layer-by-Layer Comparison")
    print("=" * 70)

    layers = ["post_attn", "post_ln2", "post_mlp_residual"]

    print(f"\n{'Experiment':<25}", end="")
    for layer in layers:
        print(f" {layer[:8]:>8}", end="")
    print()
    print(" " * 25, end="")
    for _ in layers:
        print(f" {'Full':>8}", end="")
    print()

    for name, res in all_results.items():
        print(f"{name:<25}", end="")
        for layer in layers:
            if layer in res:
                print(f" {res[layer]['full_r2']:>8.4f}", end="")
        print()

    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output_path = RESULTS_DIR / "owt_direction_norm_results.json"
    with open(output_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
