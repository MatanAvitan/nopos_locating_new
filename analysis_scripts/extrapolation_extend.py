"""
Extend extrapolation analysis to longer contexts.
Only tests 2048, 4096, 8192 to extend existing results.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    config = TwoLayerMechanismConfig(**model_args)
    model = TwoLayerMechanismModel(config)

    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            unwrapped_state_dict[k[len("_orig_mod.") :]] = v
        else:
            unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    model.eval()
    return model, config


def load_owt_data(data_dir: str = "nanoGPT/data/openwebtext"):
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    max_start = len(data) - block_size
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device)


def compute_metrics(model, data, context_length, device, batch_size=8, n_batches=50):
    """Compute R² for a given context length."""
    print(
        f"  L={context_length}: {n_batches} batches x {batch_size} = {batch_size * n_batches} sequences",
        flush=True,
    )

    model.eval()
    post_attn_acts = []
    positions_list = []

    with torch.no_grad():
        for batch_idx in range(n_batches):
            if batch_idx % 10 == 0:
                print(f"    batch {batch_idx}/{n_batches}", flush=True)

            tokens = get_batch(data, batch_size, context_length, device)
            B_size, T = tokens.shape
            _ = model(tokens, capture_taps=True)

            block2 = model.block2
            post_attn_acts.append(block2.last_post_attn.cpu())
            positions = torch.arange(T).unsqueeze(0).expand(B_size, -1)
            positions_list.append(positions)

            if context_length >= 4096 and batch_idx % 5 == 0:
                torch.cuda.empty_cache()

    post_attn_acts = torch.cat(post_attn_acts, dim=0).numpy()
    positions = torch.cat(positions_list, dim=0).numpy()

    N, T, d_model = post_attn_acts.shape
    positions_flat = positions.reshape(-1)
    acts_flat = post_attn_acts.reshape(-1, d_model)

    # Linear probe R²
    X_train, X_test, y_train, y_test = train_test_split(
        acts_flat, positions_flat, test_size=0.2, random_state=42
    )
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Per-channel correlations
    neuron_corrs = np.array(
        [stats.pearsonr(positions_flat, acts_flat[:, c])[0] for c in range(d_model)]
    )

    thresholds = [0.3, 0.5, 0.7, 0.9, 0.95]
    neuron_counts = {
        f"|r|>{t}": int(np.sum(np.abs(neuron_corrs) > t)) for t in thresholds
    }

    return {
        "context_length": context_length,
        "linear_probe_r2": float(r2),
        "neuron_counts": neuron_counts,
        "mean_abs_corr": float(np.mean(np.abs(neuron_corrs))),
        "max_abs_corr": float(np.max(np.abs(neuron_corrs))),
        "n_samples": batch_size * n_batches,
        "success": True,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "results/extrapolation_long_context"
    os.makedirs(save_dir, exist_ok=True)

    # Load existing results
    existing_path = "results/extrapolation_analysis/extrapolation_results.json"
    with open(existing_path, "r") as f:
        all_results = json.load(f)

    # Convert string keys to ints for consistent handling
    for model_name in ["R0", "R2"]:
        all_results[model_name] = {
            int(k): v for k, v in all_results[model_name].items()
        }

    print("Loaded existing results up to L=1024", flush=True)

    # Load data
    print("Loading data...", flush=True)
    val_data = load_owt_data()
    print(f"Validation data size: {len(val_data):,} tokens", flush=True)

    # New context lengths to test
    new_lengths = [2048, 4096, 8192]

    # Batch configs for different lengths
    batch_configs = {
        2048: (8, 50),  # 400 sequences
        4096: (4, 50),  # 200 sequences
        8192: (2, 50),  # 100 sequences
    }

    for model_name, ckpt_path in [
        ("R0", "nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt"),
        ("R2", "nanoGPT/out-2layer-mechanism/R2/best_ckpt.pt"),
    ]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Analyzing {model_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        model, config = load_model(ckpt_path, device)

        for L in new_lengths:
            if L in all_results[model_name]:
                print(f"\nSkipping L={L} (already computed)", flush=True)
                continue

            print(f"\nContext length: {L}", flush=True)
            batch_size, n_batches = batch_configs[L]

            try:
                result = compute_metrics(
                    model, val_data, L, device, batch_size, n_batches
                )
                all_results[model_name][L] = result
                print(f"  R²: {result['linear_probe_r2']:.4f}", flush=True)
                print(f"  Mean |r|: {result['mean_abs_corr']:.4f}", flush=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at L={L}, skipping...", flush=True)
                    torch.cuda.empty_cache()
                    all_results[model_name][L] = {"success": False, "error": "OOM"}
                else:
                    raise

            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    # Save extended results
    with open(os.path.join(save_dir, "extrapolation_extended_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for model_name, color, marker in [("R0", "#1f77b4", "o"), ("R2", "#ff7f0e", "s")]:
        lengths = []
        r2_values = []
        for L in sorted(all_results[model_name].keys()):
            data = all_results[model_name][L]
            if data.get("success", True):
                lengths.append(int(L))
                r2_values.append(data["linear_probe_r2"])

        ax.plot(
            lengths,
            r2_values,
            f"{marker}-",
            color=color,
            label=model_name,
            linewidth=2,
            markersize=8,
        )

    ax.axvline(x=128, color="gray", linestyle="--", alpha=0.5, label="Training length")
    ax.set_xlabel("Context length", fontsize=11)
    ax.set_ylabel("Linear probe R²", fontsize=11)
    ax.set_title("Position Decoding Extrapolation", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "extrapolation_long_context.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(save_dir, "extrapolation_long_context.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Print summary
    print(f"\n{'=' * 60}", flush=True)
    print("EXTRAPOLATION SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"{'Model':<8} {'L':<8} {'R²':<10}", flush=True)
    print("-" * 30, flush=True)

    for model_name in ["R0", "R2"]:
        for L in sorted(all_results[model_name].keys()):
            r = all_results[model_name][L]
            if r.get("success", True):
                print(
                    f"{model_name:<8} {L:<8} {r['linear_probe_r2']:<10.4f}", flush=True
                )

    # Degradation analysis
    print(f"\n{'=' * 60}", flush=True)
    print("DEGRADATION ANALYSIS (relative to L=128)", flush=True)
    print(f"{'=' * 60}", flush=True)

    for model_name in ["R0", "R2"]:
        results = all_results[model_name]
        base_r2 = results[128]["linear_probe_r2"]
        print(f"\n{model_name}:", flush=True)
        for L in sorted(results.keys()):
            if L > 128 and results[L].get("success", True):
                r2 = results[L]["linear_probe_r2"]
                pct_change = (r2 - base_r2) / base_r2 * 100
                print(f"  L={L}: R²={r2:.4f} ({pct_change:+.1f}%)", flush=True)

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
