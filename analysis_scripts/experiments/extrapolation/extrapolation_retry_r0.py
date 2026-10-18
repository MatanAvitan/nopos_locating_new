"""
Retry R0 at longer context lengths with batch_size=1.
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

BOS_TOKEN_ID = 50256


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
    max_start = len(data) - (block_size - 1)
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(
                np.concatenate(
                    [[BOS_TOKEN_ID], data[i : i + block_size - 1].astype(np.int64)]
                )
            )
            for i in ix
        ]
    )
    return x.to(device)


def compute_metrics(model, data, context_length, device, batch_size=1, n_batches=100):
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
            if batch_idx % 20 == 0:
                print(f"    batch {batch_idx}/{n_batches}", flush=True)

            tokens = get_batch(data, batch_size, context_length, device)
            B_size, T = tokens.shape
            _ = model(tokens, capture_taps=True)

            block2 = model.block2
            post_attn_acts.append(block2.last_post_attn.cpu())
            positions = torch.arange(T).unsqueeze(0).expand(B_size, -1)
            positions_list.append(positions)

            # Clear cache every batch for these long sequences
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r0_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/extrapolation_long_context"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics even if results exist",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nope-position-regression-metrics",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = args.save_dir

    # Load existing results
    existing_path = os.path.join(save_dir, "extrapolation_extended_results.json")
    with open(existing_path, "r") as f:
        all_results = json.load(f)

    # Convert string keys to ints
    for model_name in ["R0", "R2"]:
        all_results[model_name] = {
            int(k): v for k, v in all_results[model_name].items()
        }

    print("Loaded existing results", flush=True)

    # Load data
    print("Loading data...", flush=True)
    val_data = load_owt_data(args.data_dir)

    # Only retry R0 at 4096, 8192
    model_name = "R0"
    ckpt_path = args.r0_checkpoint

    print(f"\n{'=' * 60}", flush=True)
    print(f"Retrying {model_name} at longer contexts with batch_size=1", flush=True)
    print(f"{'=' * 60}", flush=True)

    model, config = load_model(ckpt_path, device)

    for L in [4096, 8192]:
        # Check if we already have a successful result
        if (
            not args.force
            and L in all_results[model_name]
            and all_results[model_name][L].get("success", False)
        ):
            print(f"\nSkipping L={L} (already have successful result)", flush=True)
            continue

        print(f"\nContext length: {L}", flush=True)
        n_batches = 100 if L == 4096 else 50  # Fewer batches for 8192

        try:
            result = compute_metrics(
                model, val_data, L, device, batch_size=1, n_batches=n_batches
            )
            all_results[model_name][L] = result
            print(f"  R²: {result['linear_probe_r2']:.4f}", flush=True)
            print(f"  Mean |r|: {result['mean_abs_corr']:.4f}", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Still OOM at L={L}", flush=True)
                torch.cuda.empty_cache()
                all_results[model_name][L] = {"success": False, "error": "OOM"}
            else:
                raise

        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # Save updated results
    results_path = os.path.join(save_dir, "extrapolation_extended_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Replot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for model_name, color, marker in [("R0", "#1f77b4", "o"), ("R2", "#ff7f0e", "s")]:
        lengths = []
        r2_values = []
        for L in sorted(all_results[model_name].keys()):
            data = all_results[model_name][L]
            if data.get("success", True) and "linear_probe_r2" in data:
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
    ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_pdf = os.path.join(save_dir, "extrapolation_long_context.pdf")
    plot_png = os.path.join(save_dir, "extrapolation_long_context.png")
    plt.savefig(plot_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(plot_png, bbox_inches="tight", dpi=300)
    plt.close()

    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or "extrapolation-retry-r0"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "r0_checkpoint": ckpt_path,
                "data_dir": args.data_dir,
                "save_dir": save_dir,
                "device": device,
                "retry_lengths": [4096, 8192],
                "n_batches_4096": 100,
                "n_batches_8192": 50,
            },
        )

        r0_results = all_results.get("R0", {})
        for length in [4096, 8192]:
            if length in r0_results and r0_results[length].get("success", True):
                wandb.log(
                    {
                        f"r0/r2_L{length}": r0_results[length]["linear_probe_r2"],
                        f"r0/mean_abs_corr_L{length}": r0_results[length][
                            "mean_abs_corr"
                        ],
                    }
                )

        wandb.log({"plots/extrapolation_long_context": wandb.Image(plot_png)})
        artifact = wandb.Artifact("extrapolation_long_context", type="analysis")
        artifact.add_file(results_path)
        artifact.add_file(plot_pdf)
        artifact.add_file(plot_png)
        wandb.log_artifact(artifact)
        wandb.finish()

    # Print summary
    print(f"\n{'=' * 60}", flush=True)
    print("UPDATED SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"{'Model':<8} {'L':<8} {'R²':<10}", flush=True)
    print("-" * 30, flush=True)

    for model_name in ["R0", "R2"]:
        for L in sorted(all_results[model_name].keys()):
            r = all_results[model_name][L]
            if r.get("success", True) and "linear_probe_r2" in r:
                print(
                    f"{model_name:<8} {L:<8} {r['linear_probe_r2']:<10.4f}", flush=True
                )

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
