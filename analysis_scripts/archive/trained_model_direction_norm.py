"""
Trained Model Direction vs Norm Analysis

Analyzes the existing trained NoPE models to validate that the direction→norm
transformation hypothesis holds for trained (not just random) models.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import Ridge
from scipy import stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "trained_model_analysis"


def load_trained_model(checkpoint_path):
    """Load a trained NoPE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Get config from checkpoint
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 256),
        vocab_size=model_args.get("vocab_size", 50257),
        dropout=0.0,
        use_positional_embedding=model_args.get("use_positional_embedding", False),
        norm_type=model_args.get("norm_type", "layernorm"),
        bias=model_args.get("bias", False),
    )

    model = GPT(config)

    # Handle torch.compile prefix
    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            unwrapped_state_dict[k[10:]] = v
        else:
            unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.eval()
    model.to(DEVICE)

    return model, config


def get_activations(model, tokens):
    """Get activations at key layers."""
    activations = {}

    tok_emb = model.transformer.wte(tokens)
    activations["embed"] = tok_emb.detach()

    block = model.transformer.h[0]

    x = block.ln_1(tok_emb)
    activations["post_ln1"] = x.detach()

    attn_out = block.attn(x)[0]
    activations["post_attn"] = attn_out.detach()

    x = tok_emb + attn_out
    activations["post_attn_residual"] = x.detach()

    x_ln2 = block.ln_2(x)
    activations["post_ln2"] = x_ln2.detach()

    mlp_out = block.mlp(x_ln2)
    x = x + mlp_out
    activations["post_mlp_residual"] = x.detach()

    return activations


def compute_decoding_vector(model):
    """Compute the theoretical decoding vector."""
    E = model.transformer.wte.weight.detach()
    ln1 = model.transformer.h[0].ln_1

    E_centered = E - E.mean(dim=-1, keepdim=True)
    E_std = E.std(dim=-1, keepdim=True)
    E_ln = E_centered / (E_std + 1e-5) * ln1.weight + ln1.bias

    sum_ln_E = E_ln.sum(dim=0)

    attn = model.transformer.h[0].attn
    n_embd = model.config.n_embd
    W_V = attn.c_attn.weight[:, 2 * n_embd :].detach()

    w = W_V @ sum_ln_E
    w = w / (torch.norm(w) + 1e-8)

    return w


def analyze_model(model, config, model_name, n_samples=500):
    """Full analysis of a model."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(
        f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim"
    )
    print(f"Context: {config.block_size}, Vocab: {config.vocab_size}")
    print(f"{'=' * 60}")

    n_ctx = config.block_size
    vocab_size = config.vocab_size

    results = {
        "model_name": model_name,
        "n_ctx": n_ctx,
        "n_embd": config.n_embd,
        "vocab_size": vocab_size,
    }

    # Collect activations
    all_post_attn = []
    all_post_ln2 = []
    all_positions = []

    for _ in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, n_ctx), device=DEVICE)
        acts = get_activations(model, tokens)

        all_post_attn.append(acts["post_attn"][0].cpu().numpy())
        all_post_ln2.append(acts["post_ln2"][0].cpu().numpy())
        all_positions.append(np.arange(n_ctx))

    post_attn = np.vstack(all_post_attn)
    post_ln2 = np.vstack(all_post_ln2)
    positions = np.concatenate(all_positions)

    # 1. Norm Analysis
    print("\n1. NORM-POSITION CORRELATION")

    post_attn_norms = np.linalg.norm(post_attn, axis=1)
    post_ln2_norms = np.linalg.norm(post_ln2, axis=1)

    corr_attn = np.corrcoef(post_attn_norms, positions)[0, 1]
    corr_ln2 = np.corrcoef(post_ln2_norms, positions)[0, 1]

    results["post_attn_norm_position_corr"] = corr_attn
    results["post_ln2_norm_position_corr"] = corr_ln2

    print(f"  Post-attention: r = {corr_attn:.4f}")
    print(f"  Post-LN2: r = {corr_ln2:.4f}")

    # 2. Direction vs Norm Probe
    print("\n2. DIRECTION VS NORM (Probe R²)")

    n_train = int(0.8 * len(positions))
    idx = np.random.permutation(len(positions))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def fit_probe(X_train, y_train, X_test, y_test):
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return max(0, 1 - ss_res / ss_tot)

    # Directions
    post_attn_dir = post_attn / (
        np.linalg.norm(post_attn, axis=1, keepdims=True) + 1e-8
    )
    post_ln2_dir = post_ln2 / (np.linalg.norm(post_ln2, axis=1, keepdims=True) + 1e-8)

    # Post-attention
    r2_attn_full = fit_probe(
        post_attn[train_idx],
        positions[train_idx],
        post_attn[test_idx],
        positions[test_idx],
    )
    r2_attn_dir = fit_probe(
        post_attn_dir[train_idx],
        positions[train_idx],
        post_attn_dir[test_idx],
        positions[test_idx],
    )
    r2_attn_norm = fit_probe(
        post_attn_norms[train_idx].reshape(-1, 1),
        positions[train_idx],
        post_attn_norms[test_idx].reshape(-1, 1),
        positions[test_idx],
    )

    # Post-LN2
    r2_ln2_full = fit_probe(
        post_ln2[train_idx],
        positions[train_idx],
        post_ln2[test_idx],
        positions[test_idx],
    )
    r2_ln2_dir = fit_probe(
        post_ln2_dir[train_idx],
        positions[train_idx],
        post_ln2_dir[test_idx],
        positions[test_idx],
    )
    r2_ln2_norm = fit_probe(
        post_ln2_norms[train_idx].reshape(-1, 1),
        positions[train_idx],
        post_ln2_norms[test_idx].reshape(-1, 1),
        positions[test_idx],
    )

    results["post_attn_full_r2"] = r2_attn_full
    results["post_attn_direction_r2"] = r2_attn_dir
    results["post_attn_norm_r2"] = r2_attn_norm
    results["post_ln2_full_r2"] = r2_ln2_full
    results["post_ln2_direction_r2"] = r2_ln2_dir
    results["post_ln2_norm_r2"] = r2_ln2_norm

    print(
        f"  Post-attn: full={r2_attn_full:.4f}, dir={r2_attn_dir:.4f}, norm={r2_attn_norm:.4f}"
    )
    print(
        f"  Post-LN2:  full={r2_ln2_full:.4f}, dir={r2_ln2_dir:.4f}, norm={r2_ln2_norm:.4f}"
    )

    # 3. Decoding Vector Analysis
    print("\n3. DECODING VECTOR HYPERPLANE")

    w = compute_decoding_vector(model).cpu().numpy()

    proj_attn = post_attn @ w
    proj_ln2 = post_ln2 @ w

    corr_proj_attn = np.corrcoef(proj_attn, positions)[0, 1]
    corr_proj_ln2 = np.corrcoef(proj_ln2, positions)[0, 1]

    results["decoding_vector_corr_post_attn"] = corr_proj_attn
    results["decoding_vector_corr_post_ln2"] = corr_proj_ln2

    print(f"  Decoding vector correlation (post-attn): r = {corr_proj_attn:.4f}")
    print(f"  Decoding vector correlation (post-LN2): r = {corr_proj_ln2:.4f}")

    # Is decoding vector aligned with norm direction?
    # Norm direction = direction that maximally correlates with norm
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(post_ln2, post_ln2_norms)
    norm_direction = reg.coef_ / (np.linalg.norm(reg.coef_) + 1e-8)

    alignment = np.abs(np.dot(w, norm_direction))
    results["decoding_norm_alignment"] = alignment
    print(f"  Decoding vector alignment with norm direction: {alignment:.4f}")

    # 4. Per-position mean norm
    mean_norm_by_pos = np.array(
        [post_ln2_norms[positions == i].mean() for i in range(n_ctx)]
    )

    # Theoretical 1/sqrt(i+1)
    theoretical = 1 / np.sqrt(np.arange(1, n_ctx + 1))
    # For post-LN, theoretical should be nearly constant - but there are small variations

    # Compute mean norm at different positions
    results["mean_norm_pos_0"] = mean_norm_by_pos[0]
    results["mean_norm_pos_mid"] = mean_norm_by_pos[n_ctx // 2]
    results["mean_norm_pos_end"] = mean_norm_by_pos[-1]
    results["norm_variation_percent"] = (
        (mean_norm_by_pos.max() - mean_norm_by_pos.min())
        / mean_norm_by_pos.mean()
        * 100
    )

    print(f"\n4. POST-LN2 NORM BY POSITION")
    print(f"  Position 0: {mean_norm_by_pos[0]:.4f}")
    print(f"  Position {n_ctx // 2}: {mean_norm_by_pos[n_ctx // 2]:.4f}")
    print(f"  Position {n_ctx - 1}: {mean_norm_by_pos[-1]:.4f}")
    print(f"  Variation: {results['norm_variation_percent']:.4f}%")

    return results


def main():
    print("=" * 70)
    print("TRAINED MODEL DIRECTION VS NORM ANALYSIS")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    # Check for trained models
    checkpoint_paths = [
        (
            "LayerNorm (trained)",
            Path(__file__).parent.parent / "nanoGPT" / "out-nope-1layer-ln" / "ckpt.pt",
        ),
        (
            "RMSNorm (trained)",
            Path(__file__).parent.parent
            / "nanoGPT"
            / "out-nope-1layer-rms"
            / "ckpt.pt",
        ),
    ]

    for name, path in checkpoint_paths:
        if path.exists():
            try:
                model, config = load_trained_model(path)
                results = analyze_model(model, config, name)
                all_results[name] = results
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"\nCheckpoint not found: {path}")

    # Also analyze a random model for comparison
    print("\n" + "=" * 60)
    print("RANDOM MODEL (for comparison)")
    print("=" * 60)

    from model_nope import GPT, GPTConfig

    config = GPTConfig(
        n_layer=1,
        n_head=12,
        n_embd=768,
        block_size=256,
        vocab_size=50257,
        dropout=0.0,
        use_positional_embedding=False,
        norm_type="layernorm",
    )

    random_model = GPT(config)
    random_model.eval()
    random_model.to(DEVICE)

    results = analyze_model(random_model, config, "Random Init")
    all_results["Random Init"] = results

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(
        "\n| Model | Dir R² (attn) | Norm R² (attn) | Dir R² (LN2) | Norm R² (LN2) | DV Corr |"
    )
    print(
        "|-------|---------------|----------------|--------------|---------------|---------|"
    )

    for name, res in all_results.items():
        print(
            f"| {name[:15]:15} | {res['post_attn_direction_r2']:13.4f} | "
            f"{res['post_attn_norm_r2']:14.4f} | {res['post_ln2_direction_r2']:12.4f} | "
            f"{res['post_ln2_norm_r2']:13.4f} | {res['decoding_vector_corr_post_ln2']:7.4f} |"
        )

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if "LayerNorm (trained)" in all_results and "Random Init" in all_results:
        trained = all_results["LayerNorm (trained)"]
        random = all_results["Random Init"]

        print("\nDirection encoding (post-attention):")
        print(f"  Trained: R² = {trained['post_attn_direction_r2']:.4f}")
        print(f"  Random:  R² = {random['post_attn_direction_r2']:.4f}")

        print("\nNorm encoding (post-LN2):")
        print(f"  Trained: R² = {trained['post_ln2_norm_r2']:.4f}")
        print(f"  Random:  R² = {random['post_ln2_norm_r2']:.4f}")

        print("\nDecoding vector effectiveness:")
        print(f"  Trained: r = {trained['decoding_vector_corr_post_ln2']:.4f}")
        print(f"  Random:  r = {random['decoding_vector_corr_post_ln2']:.4f}")

    # Save results
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

    with open(RESULTS_DIR / "trained_model_results.json", "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
