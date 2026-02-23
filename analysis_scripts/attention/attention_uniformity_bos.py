"""Measure attention uniformity and BOS bias, and regenerate attention maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ICML-style plotting defaults
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


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


def get_batch(data: np.ndarray, batch_size: int, block_size: int) -> torch.Tensor:
    ix = torch.randint(len(data) - (block_size - 1), (batch_size,))
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
    return x.to(DEVICE)


def compute_block1_entropy(attn1: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy per position and head, averaged over batch."""
    _, _, T, _ = attn1.shape
    entropies = []
    for t in range(T):
        probs = attn1[:, :, t, : t + 1]
        probs = torch.clamp(probs, min=eps)
        ent = -(probs * probs.log()).sum(dim=-1)
        entropies.append(ent.mean(dim=0))
    return torch.stack(entropies, dim=1)


def compute_bos_ratio(
    attn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mean BOS weight, mean others weight, and ratio per head."""
    _, _, T, _ = attn.shape
    mask = torch.tril(torch.ones(T, T, device=attn.device))
    mask[:, 0] = 0
    denom = torch.arange(T, device=attn.device).float().view(1, 1, T)
    denom[..., 0] = 1.0

    others_sum = (attn * mask).sum(dim=-1)
    others_mean = others_sum / denom

    w0 = attn[..., 0]
    w0 = w0[:, :, 1:]
    others_mean = others_mean[:, :, 1:]

    w0_sum = w0.sum(dim=(0, 2))
    others_sum = others_mean.sum(dim=(0, 2))
    count = w0.shape[0] * w0.shape[2]

    w0_mean = w0_sum / count
    others_mean = others_sum / count
    ratio = w0_mean / others_mean

    return w0_mean, others_mean, ratio


def plot_attention_grid(
    attn_weights: np.ndarray,
    title: str,
    save_path: Path,
    highlight_col0: bool,
    head_labels: list[str] | None = None,
    cmap: str = "magma",
    show_head_titles: bool = True,
    save_dpi: int = 600,
) -> None:
    n_head, T, _ = attn_weights.shape
    ncols = 4 if n_head > 1 else 1
    nrows = int(math.ceil(n_head / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False
    )

    # Attention weights are probabilities in [0, 1].
    # Keep a fixed color scale so the colorbar remains interpretable
    # and row-0 self-attention (exactly 1.0) maps to the top of the bar.
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)

    im = None
    for idx in range(n_head):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        im = ax.imshow(
            attn_weights[idx],
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        if highlight_col0:
            ax.axvline(0.5, color="#C44E52", linewidth=1.2)
            ax.add_patch(
                Rectangle(
                    (-0.5, -0.5),
                    1.0,
                    T,
                    fill=True,
                    color="#C44E52",
                    alpha=0.12,
                    linewidth=0,
                )
            )
        if show_head_titles:
            label = f"Head {idx}"
            if head_labels is not None:
                label = f"{label} · {head_labels[idx]}"
            ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for idx in range(n_head, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.subplots_adjust(right=0.86, wspace=0.08, hspace=0.12)
    cax = fig.add_axes((0.88, 0.15, 0.02, 0.7))
    if im is None:
        return
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention weight", fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    fig.suptitle(title, y=0.98, fontsize=11)
    fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    batch_size: int,
    metric_batches: int,
) -> dict:
    entropy_sum: Optional[torch.Tensor] = None
    w0_sum_block2: Optional[torch.Tensor] = None
    others_sum_block2: Optional[torch.Tensor] = None
    w0_sum_block1: Optional[torch.Tensor] = None
    others_sum_block1: Optional[torch.Tensor] = None
    count = 0

    for _ in range(metric_batches):
        tokens = get_batch(data, batch_size, model.config.block_size)
        with torch.no_grad():
            model(tokens, capture_taps=True)
            attn1, attn2 = model.get_attention_weights()

        entropy = compute_block1_entropy(attn1)
        w0_mean_block2, others_mean_block2, _ = compute_bos_ratio(attn2)
        w0_mean_block1, others_mean_block1, _ = compute_bos_ratio(attn1)

        entropy_cpu = entropy.detach().cpu()
        w0_block2_cpu = w0_mean_block2.detach().cpu()
        others_block2_cpu = others_mean_block2.detach().cpu()
        w0_block1_cpu = w0_mean_block1.detach().cpu()
        others_block1_cpu = others_mean_block1.detach().cpu()

        if entropy_sum is None:
            entropy_sum = entropy_cpu
            w0_sum_block2 = w0_block2_cpu
            others_sum_block2 = others_block2_cpu
            w0_sum_block1 = w0_block1_cpu
            others_sum_block1 = others_block1_cpu
        else:
            if (
                entropy_sum is None
                or w0_sum_block2 is None
                or others_sum_block2 is None
                or w0_sum_block1 is None
                or others_sum_block1 is None
            ):
                raise RuntimeError("Metric accumulation missing prior state")
            entropy_sum += entropy_cpu
            w0_sum_block2 += w0_block2_cpu
            others_sum_block2 += others_block2_cpu
            w0_sum_block1 += w0_block1_cpu
            others_sum_block1 += others_block1_cpu

        count += 1

    if (
        entropy_sum is None
        or w0_sum_block2 is None
        or others_sum_block2 is None
        or w0_sum_block1 is None
        or others_sum_block1 is None
    ):
        raise RuntimeError("Metric accumulation failed to initialize")

    entropy_avg = entropy_sum / count
    w0_avg_block2 = w0_sum_block2 / count
    others_avg_block2 = others_sum_block2 / count
    ratio_block2 = w0_avg_block2 / others_avg_block2

    w0_avg_block1 = w0_sum_block1 / count
    others_avg_block1 = others_sum_block1 / count
    ratio_block1 = w0_avg_block1 / others_avg_block1

    return {
        "block1_entropy": entropy_avg.tolist(),
        "block1_bos_mean": w0_avg_block1.tolist(),
        "block1_others_mean": others_avg_block1.tolist(),
        "block1_bos_ratio": ratio_block1.tolist(),
        "block2_bos_mean": w0_avg_block2.tolist(),
        "block2_others_mean": others_avg_block2.tolist(),
        "block2_bos_ratio": ratio_block2.tolist(),
    }


def compute_attention_maps(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    batch_size: int,
    map_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    attn1_sum: Optional[torch.Tensor] = None
    attn2_sum: Optional[torch.Tensor] = None

    for _ in range(map_batches):
        tokens = get_batch(data, batch_size, model.config.block_size)
        with torch.no_grad():
            model(tokens, capture_taps=True)
            attn1, attn2 = model.get_attention_weights()
        attn1_mean = attn1.mean(dim=0)
        attn2_mean = attn2.mean(dim=0)

        if attn1_sum is None:
            attn1_sum = attn1_mean.detach().cpu()
            attn2_sum = attn2_mean.detach().cpu()
        else:
            attn1_sum += attn1_mean.detach().cpu()
            attn2_sum += attn2_mean.detach().cpu()

    if attn1_sum is None or attn2_sum is None:
        raise RuntimeError("Attention map accumulation failed")

    attn1_avg = (attn1_sum / map_batches).numpy()
    attn2_avg = (attn2_sum / map_batches).numpy()
    return attn1_avg, attn2_avg


def main() -> None:
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
    parser.add_argument("--save_dir", type=str, default="results/attention_uniformity")
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--metric_batches", type=int, default=20)
    parser.add_argument("--map_batches", type=int, default=8)
    args = parser.parse_args()

    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    paper_attn_dir = paper_dir / "attention_uniformity"
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_attn_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(ROOT_DIR / args.data_dir)
    attn2_model = load_model(ROOT_DIR / args.attn2_ckpt)
    full_model = load_model(ROOT_DIR / args.full12h_ckpt)

    results = {
        "attn2": compute_metrics(
            attn2_model, data, args.batch_size, args.metric_batches
        ),
        "full12h": compute_metrics(
            full_model, data, args.batch_size, args.metric_batches
        ),
        "metric_batches": args.metric_batches,
        "batch_size": args.batch_size,
        "block_size": attn2_model.config.block_size,
    }

    attn2_attn1, attn2_attn2 = compute_attention_maps(
        attn2_model, data, args.batch_size, args.map_batches
    )
    full_attn1, full_attn2 = compute_attention_maps(
        full_model, data, args.batch_size, args.map_batches
    )

    attn2_ratio = results["attn2"]["block2_bos_ratio"]
    full_ratio = results["full12h"]["block2_bos_ratio"]

    plot_attention_grid(
        attn2_attn1,
        "ATTN2-1H Block 1 Attention",
        save_dir / "attn2_1h_block1_attention.pdf",
        highlight_col0=False,
        show_head_titles=False,
    )
    plot_attention_grid(
        attn2_attn2,
        "ATTN2-1H Block 2 Attention",
        save_dir / "attn2_1h_block2_attention.pdf",
        highlight_col0=True,
        show_head_titles=False,
    )

    plot_attention_grid(
        full_attn1,
        "FULL-12H Block 1 Attention",
        save_dir / "full12h_block1_attention.pdf",
        highlight_col0=False,
    )
    plot_attention_grid(
        full_attn2,
        "FULL-12H Block 2 Attention",
        save_dir / "full12h_block2_attention.pdf",
        highlight_col0=True,
        head_labels=[f"ratio {r:.2f}" for r in full_ratio],
    )

    plot_bases = [
        "attn2_1h_block1_attention.pdf",
        "attn2_1h_block2_attention.pdf",
        "full12h_block1_attention.pdf",
        "full12h_block2_attention.pdf",
    ]
    for path in plot_bases:
        src_pdf = save_dir / path
        src_png = src_pdf.with_suffix(".png")

        if src_pdf.exists():
            (paper_dir / src_pdf.name).write_bytes(src_pdf.read_bytes())
            (paper_attn_dir / src_pdf.name).write_bytes(src_pdf.read_bytes())
        if src_png.exists():
            (paper_dir / src_png.name).write_bytes(src_png.read_bytes())
            (paper_attn_dir / src_png.name).write_bytes(src_png.read_bytes())

    # Sanity: row-0 can only attend to itself under causal masking.
    row0_self = {
        "attn2_block1_min": float(attn2_attn1[:, 0, 0].min()),
        "attn2_block1_max": float(attn2_attn1[:, 0, 0].max()),
        "attn2_block2_min": float(attn2_attn2[:, 0, 0].min()),
        "attn2_block2_max": float(attn2_attn2[:, 0, 0].max()),
        "full12h_block1_min": float(full_attn1[:, 0, 0].min()),
        "full12h_block1_max": float(full_attn1[:, 0, 0].max()),
        "full12h_block2_min": float(full_attn2[:, 0, 0].min()),
        "full12h_block2_max": float(full_attn2[:, 0, 0].max()),
    }
    results["row0_self_attention"] = row0_self

    with open(save_dir / "attention_uniformity_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved metrics to:", save_dir / "attention_uniformity_metrics.json")
    print("Saved attention maps to:", save_dir)
    print("Copied attention maps to:", paper_attn_dir)
    print("Row-0 self-attention sanity:", row0_self)


if __name__ == "__main__":
    main()
