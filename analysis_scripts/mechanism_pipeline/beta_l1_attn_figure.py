"""Figure: the Layer-1 attention pattern of 1H models as the training length grows.

Row 1: mean Layer-1 attention (queries and keys 0..15) for one seed per
       training length, next to exactly uniform attention; one shared log scale.
Row 2: (left) mean BOS weight against position for every seed, with the
       uniform value 1/(j+1); (right) distance of the attention rows from
       uniform (total variation, mean over queries 1..127) against length.

Usage: python beta_l1_attn_figure.py --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import CHECKPOINTS, RESULTS_ROOT, make_adapter, spec_splits
from common import batch_from_offsets

LS = [128, 256, 512, 1024, 2048, 8192]
SEEDS = [42, 1, 2]
N_SEQ = 64
K = 16
J_MAX = 1023


def key_for(L, s):
    if s == 42 and L == 1024:
        return "1h"
    if s == 42 and L == 8192:
        return "1h_8192"
    return f"sw_L{L}_s{s}"


def finished(key):
    if not key.startswith("sw_"):
        return True
    return os.path.exists(str(CHECKPOINTS[key]).replace("best_ckpt", "final_ckpt"))


@torch.no_grad()
def measure(key, dev):
    ad, L, _ = make_adapter(key, dev)
    offs = spec_splits(L)["evaluation"][:N_SEQ]
    bs = 1 if L >= 8192 else (2 if L >= 2048 else 8)
    T = min(L, J_MAX + 1)
    amap = torch.zeros(K, K, dtype=torch.float64); bos = torch.zeros(T, dtype=torch.float64)
    tv = 0.0; n = 0
    Tt = min(L, 128)
    i = torch.arange(Tt, device=dev, dtype=torch.float64)
    uni = torch.tril(torch.ones(Tt, Tt, device=dev, dtype=torch.float64)) / (i[:, None] + 1)
    for s in range(0, N_SEQ, bs):
        idx = batch_from_offsets(offs[s:s + bs], L, dev)
        w = ad.pre(idx)["attn1_w"][:, 0].double()                   # one head: [B,T,T]
        amap += w[:, :K, :K].sum(0).cpu()
        bos += w[:, :T, 0].sum(0).cpu()
        tv += float((0.5 * (w[:, :Tt, :Tt] - uni).abs().sum(-1))[:, 1:].mean(1).sum())
        n += idx.shape[0]
    del ad; torch.cuda.empty_cache()
    return {"map": (amap / n).numpy(), "bos": (bos / n).numpy(), "tv": tv / n, "L": L}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    res = {}
    for L in LS:
        for s in SEEDS:
            k = key_for(L, s)
            if finished(k):
                res[(L, s)] = measure(k, args.device)
                print(L, s, "TV %.2f" % res[(L, s)]["tv"], flush=True)

    blues = plt.get_cmap("Blues")
    col = {L: blues(0.35 + 0.65 * n / (len(LS) - 1)) for n, L in enumerate(LS)}
    fig = plt.figure(figsize=(15, 8.6))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.3], hspace=0.04)
    gs = top.add_gridspec(1, len(LS) + 1, wspace=0.12)
    gb = bottom.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.22)
    norm = LogNorm(vmin=1e-3, vmax=1.0)
    uni = np.tril(np.ones((K, K))) / (np.arange(K)[:, None] + 1)
    panels = [("uniform\n(reference)", uni)] + [(f"trained, L = {L}", res[(L, 42)]["map"]) for L in LS if (L, 42) in res]
    for c, (title, M) in enumerate(panels):
        ax = top.add_subplot(gs[0, c])
        Mm = np.where(np.tril(np.ones((K, K))) > 0, np.maximum(M, 1e-3), np.nan)
        im = ax.imshow(Mm, cmap="magma_r", norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-0.5, -0.5), 1, K, fill=False, ec="#1f77b4", lw=1.2))      # BOS column
        ax.add_patch(Rectangle((-0.5, 0.5), 2, 1, fill=False, ec="#2ca02c", lw=1.2))       # position-1 row
        ax.set_xticks([0, 1, 5, 15]); ax.set_yticks([0, 1, 5, 15])
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("query position i", fontsize=8)
        ax.set_xlabel("key position j", fontsize=8)
    cb = top.colorbar(im, ax=top.axes, fraction=0.012, pad=0.01); cb.set_label("attention weight", fontsize=8); cb.ax.tick_params(labelsize=7)

    top.suptitle("Mean Layer-1 attention, positions 0-15 (blue box: attention on BOS; green box: position 1)", fontsize=10)
    ax = bottom.add_subplot(gb[0, 0])
    j = np.arange(1, J_MAX + 1)
    ax.plot(j, 1 / (j + 1), color="0.35", ls="--", lw=1.5, label="uniform, 1/(j+1)")
    for L in LS:
        first = True
        for s in SEEDS:
            if (L, s) not in res:
                continue
            b = res[(L, s)]["bos"]
            jj = np.arange(1, len(b))
            ax.plot(jj, np.maximum(b[1:], 1e-6), color=col[L], lw=1.4 if s == 42 else 0.8,
                    alpha=1.0 if s == 42 else 0.7, label=f"L = {L}" if first else None)
            first = False
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 1.3)
    ax.set_xlabel("query position i", fontsize=9); ax.set_ylabel("Layer-1 attention on BOS", fontsize=9)
    ax.set_title("Attention on BOS: above uniform for short training contexts;\n"
                 "position 1 copies BOS and later positions fall below uniform for L ≥ 512", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="lower left")
    ax.grid(True, which="major", color="0.9", lw=0.6)

    ax = bottom.add_subplot(gb[0, 1])
    for L in LS:
        tvs = [res[(L, s)]["tv"] for s in SEEDS if (L, s) in res]
        ax.scatter([L] * len(tvs), tvs, color=col[L], s=40, edgecolor="white", linewidth=1.5, zorder=3)
    med = [(L, np.median([res[(L, s)]["tv"] for s in SEEDS if (L, s) in res])) for L in LS if any((L, s) in res for s in SEEDS)]
    ax.plot([m[0] for m in med], [m[1] for m in med], color="0.6", lw=1, zorder=2)
    ax.axhline(0, color="0.35", ls="--", lw=1)
    ax.text(140, 0.02, "uniform", fontsize=7.5, color="0.35")
    ax.set_xscale("log", base=2); ax.set_xticks(LS); ax.set_xticklabels(LS, fontsize=8)
    ax.set_xlabel("training length L", fontsize=9); ax.set_ylabel("distance from uniform (TV)", fontsize=9)
    ax.set_title("Every seed at every length is non-uniform;\nthe distance grows with L", fontsize=9)
    ax.grid(True, axis="y", color="0.9", lw=0.6)

    od = RESULTS_ROOT / "sweep"
    for ext in ("png", "pdf"):
        fig.savefig(od / f"layer1_attention_by_L.{ext}", dpi=160, bbox_inches="tight")
    (od / "layer1_attention_by_L.json").write_text(json.dumps(
        {f"L{L}_s{s}": {"tv": r["tv"], "bos_1_5": r["bos"][1:6].tolist(), "bos_64": float(r["bos"][64])} for (L, s), r in res.items()}, indent=2))
    print("saved", od / "layer1_attention_by_L.png")


if __name__ == "__main__":
    main()
