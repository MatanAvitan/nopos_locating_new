"""Combine Step-4 attention-output projections for R2 Attn2-1H and Attn2-12H.

Reads the per-tag projection_results JSONs produced by
r0_attention_output_projection.py and renders two comparison figures:
  (1) a single-axes overlay (1H solid, 12H dashed), and
  (2) a side-by-side two-panel figure.

Both heads are pure R2 (only Layer-2 attention + linear head trained) at L=1024.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES_DIR = ROOT / "results" / "r0_attention_output_projection"
PAPER_DIRS = [
    ROOT / "overleaf" / "nopos_nips" / "plots" / "r0_attention_output_projection",
    ROOT / "overleaf" / "nopos_nips" / "plots",
]

COLOR_BOS = "#009E73"
COLOR_OTHERS = "#D55E00"


def load(tag: str) -> dict:
    with open(RES_DIR / f"projection_results_{tag}.json") as f:
        data = json.load(f)
    (key,) = list(data.keys())  # e.g. "L1024"
    res = data[key]
    res["_L"] = int(key[1:])
    return res


def save(fig, name: str) -> None:
    pdf = RES_DIR / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    for d in PAPER_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.pdf").write_bytes(pdf.read_bytes())
        (d / f"{name}.png").write_bytes(pdf.with_suffix(".png").read_bytes())
    print(f"saved {name} (.pdf/.png) -> results + {len(PAPER_DIRS)} paper dirs")


def overlay(one: dict, twelve: dict) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.2))
    pos1 = np.arange(one["_L"])
    pos12 = np.arange(twelve["_L"])

    ax.plot(pos1, one["proj_bos"], color=COLOR_BOS, lw=1.3, ls="-",
            label=f"1H BOS (ρ={one['spearman_bos']:.2f})")
    ax.plot(pos1, one["proj_non_bos"], color=COLOR_OTHERS, lw=1.3, ls="-",
            label=f"1H non-BOS (ρ={one['spearman_non_bos']:.2f})")
    ax.plot(pos12, twelve["proj_bos"], color=COLOR_BOS, lw=1.3, ls="--",
            label=f"12H BOS (ρ={twelve['spearman_bos']:.2f})")
    ax.plot(pos12, twelve["proj_non_bos"], color=COLOR_OTHERS, lw=1.3, ls="--",
            label=f"12H non-BOS (ρ={twelve['spearman_non_bos']:.2f})")

    ax.set_title("Attention Output Projection (R2: 1H vs 12H)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Projection")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=6.5, loc="best")
    plt.tight_layout()
    save(fig, "attention_output_projection_attn2_1h_vs_12h_overlay")
    plt.close(fig)


def panels(one: dict, twelve: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=False)
    for ax, res, title in (
        (axes[0], one, "Attn2-1H"),
        (axes[1], twelve, "Attn2-12H"),
    ):
        pos = np.arange(res["_L"])
        ax.plot(pos, res["proj_bos"], color=COLOR_BOS, lw=1.3,
                label=f"BOS dir (ρ={res['spearman_bos']:.2f})")
        ax.plot(pos, res["proj_non_bos"], color=COLOR_OTHERS, lw=1.3,
                label=f"non-BOS dir (ρ={res['spearman_non_bos']:.2f})")
        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Projection")
    plt.tight_layout()
    save(fig, "attention_output_projection_attn2_1h_12h_panels")
    plt.close(fig)


def main() -> None:
    one = load("attn2_1h_matched")
    twelve = load("attn2_12h")
    print(f"1H  L={one['_L']}  mean-curve rho BOS/nonBOS = "
          f"{one['spearman_bos']:.3f}/{one['spearman_non_bos']:.3f}  "
          f"sample-median = {one['spearman_stats']['bos']['sample_sequence_median_rho']:.3f}/"
          f"{one['spearman_stats']['non_bos']['sample_sequence_median_rho']:.3f}")
    print(f"12H L={twelve['_L']}  mean-curve rho BOS/nonBOS = "
          f"{twelve['spearman_bos']:.3f}/{twelve['spearman_non_bos']:.3f}  "
          f"sample-median = {twelve['spearman_stats']['bos']['sample_sequence_median_rho']:.3f}/"
          f"{twelve['spearman_stats']['non_bos']['sample_sequence_median_rho']:.3f}")
    overlay(one, twelve)
    panels(one, twelve)


if __name__ == "__main__":
    main()
