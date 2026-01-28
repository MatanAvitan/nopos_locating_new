"""Plot extrapolation curves for Attn2-1H and Full-12H models."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: Path) -> TwoLayerMechanismModel:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TwoLayerMechanismConfig(**filtered)

    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def load_data(data_dir: Path) -> np.memmap:
    return np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")


def evaluate_extrapolation(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    context_lengths: list[int],
    n_batches: int,
    batch_size: int,
) -> dict:
    results = {}

    for L in context_lengths:
        preds_all = []
        targets_all = []

        for _ in range(n_batches):
            ix = torch.randint(len(data) - (L - 1), (batch_size,))
            tokens = torch.stack(
                [
                    torch.from_numpy(
                        np.concatenate(
                            [[BOS_TOKEN_ID], data[i : i + L - 1].astype(np.int64)]
                        )
                    )
                    for i in ix
                ]
            ).to(DEVICE)

            targets = (
                torch.arange(L, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            with torch.no_grad():
                output, _ = model(tokens, capture_taps=False)
                preds = output.squeeze(-1)
            preds_all.append(preds)
            targets_all.append(targets)

        preds_all = torch.cat(preds_all, dim=0).flatten().cpu().numpy()
        targets_all = torch.cat(targets_all, dim=0).flatten().cpu().numpy()
        r = np.corrcoef(targets_all, preds_all)[0, 1]
        r2 = float(r * r)
        results[str(L)] = {"r2": r2, "n": n_batches * batch_size}

    return results


def evaluate_extrapolation_scheduled(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    context_lengths: list[int],
    batch_schedule: dict[int, int],
    target_n: int,
) -> dict:
    results = {}

    for L in context_lengths:
        batch_size = batch_schedule[L]
        n_batches = int(np.ceil(target_n / batch_size))
        preds_all = []
        targets_all = []

        for _ in range(n_batches):
            ix = torch.randint(len(data) - (L - 1), (batch_size,))
            tokens = torch.stack(
                [
                    torch.from_numpy(
                        np.concatenate(
                            [[BOS_TOKEN_ID], data[i : i + L - 1].astype(np.int64)]
                        )
                    )
                    for i in ix
                ]
            ).to(DEVICE)

            targets = (
                torch.arange(L, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            with torch.no_grad():
                output, _ = model(tokens, capture_taps=False)
                preds = output.squeeze(-1)
            preds_all.append(preds)
            targets_all.append(targets)

        preds_all = torch.cat(preds_all, dim=0).flatten().cpu().numpy()
        targets_all = torch.cat(targets_all, dim=0).flatten().cpu().numpy()
        r = np.corrcoef(targets_all, preds_all)[0, 1]
        r2 = float(r * r)
        results[str(L)] = {"r2": r2, "n": n_batches * batch_size}

    return results


def plot_extrapolation(
    attn2_results: dict,
    full_results: dict,
    context_lengths: list[int],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    lengths = np.array(context_lengths)
    attn2_vals = np.array([attn2_results[str(L)]["r2"] for L in context_lengths])
    full_vals = np.array([full_results[str(L)]["r2"] for L in context_lengths])

    ax.plot(
        lengths,
        attn2_vals,
        color="#0072B2",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Attn2-1H",
    )
    ax.plot(
        lengths,
        full_vals,
        color="#D55E00",
        marker="s",
        markersize=4,
        linewidth=1.5,
        label="Full-12H",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Context length")
    ax.set_ylabel("R$^2$")
    ax.set_title("Length Extrapolation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn2_ckpt",
        type=str,
        default=(
            "nanoGPT/out-2layer-mechanism-r2-1head-attnonly-fullblock-40k/"
            "R2/o4w7v8dv/best_ckpt.pt"
        ),
    )
    parser.add_argument(
        "--full12h_ckpt",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/extrapolation_attn2_full12h"
    )
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--full_target_n", type=int, default=64)
    args = parser.parse_args()

    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(ROOT_DIR / args.data_dir)
    attn2_model = load_model(ROOT_DIR / args.attn2_ckpt)
    full_model = load_model(ROOT_DIR / args.full12h_ckpt)

    context_lengths = [128, 256, 512, 1024, 2048, 4096]

    attn2_results = evaluate_extrapolation(
        attn2_model, data, context_lengths, args.n_batches, args.batch_size
    )

    batch_schedule = {
        128: 32,
        256: 16,
        512: 8,
        1024: 4,
        2048: 2,
        4096: 1,
    }
    full_results = evaluate_extrapolation_scheduled(
        full_model, data, context_lengths, batch_schedule, args.full_target_n
    )

    save_path = save_dir / "extrapolation_attn2_full12h.pdf"
    plot_extrapolation(attn2_results, full_results, context_lengths, save_path)

    with open(save_dir / "extrapolation_results.json", "w") as f:
        json.dump(
            {
                "attn2": attn2_results,
                "full12h": full_results,
                "attn2_n_sequences": args.n_batches * args.batch_size,
                "full12h_target_n": args.full_target_n,
            },
            f,
            indent=2,
        )

    paper_path = paper_dir / save_path.name
    if save_path.exists():
        paper_path.write_bytes(save_path.read_bytes())

    print("Saved:", save_path)
    print("Copied to:", paper_path)


if __name__ == "__main__":
    main()
