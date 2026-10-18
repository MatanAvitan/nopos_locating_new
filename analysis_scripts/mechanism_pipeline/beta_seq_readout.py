"""Per-sequence Layer-2 readout: o_i - b_attn projected on w_BOS-hat and on
the Delta-w axis (w_nonBOS - w_BOS, unit), for individual evaluation
sequences, with no averaging. Also the 2D trajectory in the plane of
w_BOS-hat and the part of Delta-w orthogonal to it.

For multi-head models the output is the sum over the BOS heads, projected
on the aggregate axes (as in beta_analyze.py). Reads beta/evaluation.npz
and beta/ref.pt. Writes beta/figures/seq_readout.png.

Usage: python beta_seq_readout.py --model 1h [--n-seq 12]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import MODELS, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n-seq", type=int, default=12)
    args = ap.parse_args()
    _, _, L, run_id = MODELS[args.model]
    od = out_dir(run_id)
    meta = json.loads((od / "meta.json").read_text())
    R = torch.load(od / "ref.pt", weights_only=False)
    E = np.load(od / "evaluation.npz")
    H = meta["n_head"]
    heads = meta["bos_heads"] or [int(np.argmax(meta["r_h"]))]
    wb = R["w_bos"][heads].sum(0).double().numpy()
    dw = (R["w_nonbos"] - R["w_bos"])[heads].sum(0).double().numpy()
    uw, ud = wb / np.linalg.norm(wb), dw / np.linalg.norm(dw)
    cos = float(uw @ ud)
    zw = E["z_w"][:, H].astype(np.float64)          # along w_BOS-hat
    zd = E["z_dw"][:, H].astype(np.float64)         # along Delta-w-hat
    # coordinate along the part of Delta-w orthogonal to w_BOS
    zperp = (zd - cos * zw) / np.sqrt(max(1 - cos ** 2, 1e-12))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(zw), args.n_seq, replace=False)
    i = np.arange(1, L)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    cm = plt.get_cmap("tab20")
    for k, s in enumerate(sel):
        c = cm(k % 20)
        ax[0].plot(i, zw[s, 1:], color=c, lw=.8)
        ax[1].plot(i, zd[s, 1:], color=c, lw=.8)
        ax[2].plot(zw[s, 1:], zperp[s, 1:], color=c, lw=.6, alpha=.8)
        ax[2].scatter(zw[s, 1:], zperp[s, 1:], c=np.log(i), cmap="viridis", s=2, zorder=3)
    ax[0].set_title(r"$\langle o_i - b_{attn},\ \hat w_{BOS}\rangle$, one line per sequence")
    ax[1].set_title(r"$\langle o_i - b_{attn},\ \widehat{\Delta w}\rangle$, $\Delta w = w_{nonBOS} - w_{BOS}$")
    ax[2].set_title(f"2D: $\\hat w_{{BOS}}$ vs. $\\Delta w$ part orthogonal to it (cos = {cos:.4f}); color = log i")
    for a in ax[:2]:
        a.set_xscale("log"); a.set_xlabel("position i")
    ax[2].set_xlabel(r"along $\hat w_{BOS}$"); ax[2].set_ylabel(r"along $\Delta w_\perp$")
    fig.suptitle(f"{run_id}: {args.n_seq} individual evaluation sequences (no averaging), BOS heads {heads}")
    fig.tight_layout()
    p = od / "figures" / "seq_readout.png"
    fig.savefig(p, dpi=130)
    print(p, "cos(w_BOS, dw) =", round(cos, 5))


if __name__ == "__main__":
    main()
