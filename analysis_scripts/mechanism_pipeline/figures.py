"""Regenerate mechanism figures from canonical artifacts (exps.md).

Every figure is derived only from results/mechanism/<run_id>/ files.
Usage: python figures.py --run-id attn2_1h_L1024 [--head 0]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT

QCOLORS = {"q05": 0.15, "q25": 0.3, "q75": 0.3, "q95": 0.15}


def load(out, key):
    a = np.load(out / (key.replace("/", "__") + ".npy"))
    return a if a.ndim == 3 else a[:, None, :]


def band_plot(ax, pos, arr, color="C0", label=None, log_x=True, log_y=False):
    """arr: [n_seq, T'] aligned with pos."""
    med = np.nanmedian(arr, 0)
    q05, q25, q75, q95 = (np.nanquantile(arr, q, axis=0)
                          for q in (0.05, 0.25, 0.75, 0.95))
    ax.plot(pos, med, color=color, lw=1.5, label=label)
    ax.fill_between(pos, q25, q75, color=color, alpha=0.25, lw=0)
    ax.fill_between(pos, q05, q95, color=color, alpha=0.12, lw=0)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--head", type=int, default=None,
                    help="head to plot (default: all heads overlaid where sane)")
    args = ap.parse_args()
    out = RESULTS_ROOT / args.run_id
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    cfg = json.loads((out / "config.json").read_text())
    L = cfg["context_length"]
    pos = np.arange(1, L)

    def sel_heads(arr):
        H = arr.shape[1]
        if args.head is not None:
            return [args.head]
        return range(H)

    # ---- Step 1: uniformity ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key, ax, ylab in [("step1/norm_entropy", axes[0], "normalized entropy"),
                          ("step1/tv_uniform", axes[1], "TV to uniform"),
                          ("step1/alpha_bos_ratio", axes[2],
                           r"$\alpha_{BOS}\cdot(i{+}1)$")]:
        a = load(out, key)
        for h in sel_heads(a):
            band_plot(ax, pos, a[:, h, 1:], color=f"C{h % 10}")
        ax.set_xlabel("position $i$"); ax.set_ylabel(ylab)
    axes[2].axhline(1.0, ls="--", c="k", lw=0.8)
    fig.suptitle(f"Step 1: Layer-1 attention uniformity ({args.run_id})")
    fig.tight_layout(); fig.savefig(fig_dir / "step1_uniformity.png", dpi=200)
    plt.close(fig)

    # ---- Step 1: signal / error / SNR -------------------------------------
    S = load(out, "step1/S")[:, 0]
    E = load(out, "step1/E")[:, 0]
    A_i = np.load(out / "step1_A_i.npy")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    band_plot(axes[0], pos, S[:, 1:], "C0", "actual $S_{b,i}$")
    axes[0].plot(pos, A_i[1:], "k--", lw=1, label="ideal $A_i$")
    axes[0].legend(); axes[0].set_ylabel("BOS projection")
    band_plot(axes[1], pos, np.abs(E[:, 1:]), "C1", "$|E_{b,i}|$", log_y=True)
    axes[1].set_ylabel("content error")
    snr = np.abs(A_i[None, 1:]) / np.abs(E[:, 1:])
    band_plot(axes[2], pos, snr, "C2", "SNR", log_y=True)
    ref = A_i[1:] / np.nanmedian(np.abs(E[:, 1:]))
    axes[2].plot(pos, ref, "k--", lw=1, label=r"$\propto i^{-1/2}$")
    axes[2].legend(); axes[2].set_ylabel("signal-to-error ratio")
    for ax in axes:
        ax.set_xlabel("position $i$")
    fig.suptitle(f"Step 1: BOS signal / content error / SNR ({args.run_id})")
    fig.tight_layout(); fig.savefig(fig_dir / "step1_signal_error.png", dpi=200)
    plt.close(fig)

    # ---- Step 3: cbar stability + alpha scaling ----------------------------
    logc = load(out, "step3/log_cbar")
    alpha = load(out, "step3/alpha_bos")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for h in sel_heads(logc):
        band_plot(axes[0], pos, logc[:, h, 1:], f"C{h % 10}")
    axes[0].set_ylabel(r"$\log \bar c_{b,i}$")
    for h in sel_heads(alpha):
        band_plot(axes[1], pos, alpha[:, h, 1:], f"C{h % 10}", log_y=True)
    axes[1].set_ylabel(r"$\alpha_{i,BOS}$")
    for h in sel_heads(alpha):
        band_plot(axes[2], pos, pos[None, :] * alpha[:, h, 1:], f"C{h % 10}")
    axes[2].set_ylabel(r"$i\,\alpha_{i,BOS}$")
    for ax in axes:
        ax.set_xlabel("position $i$")
    fig.suptitle(f"Step 3: softmax-ratio stability and BOS-weight scaling ({args.run_id})")
    fig.tight_layout(); fig.savefig(fig_dir / "step3_cbar_alpha.png", dpi=200)
    plt.close(fig)

    # ---- Step 4: residual + factorization ----------------------------------
    e = load(out, "step4/e")
    A_conc = load(out, "step4/A_conc")
    R_scale = load(out, "step4/R_scale")
    D = load(out, "step4/D"); C = load(out, "step4/C")
    coher = np.sqrt(np.clip(1 + C / np.where(D > 0, D, np.nan), 0, None))
    cosgw = load(out, "step4/cos_g_w")
    nr = load(out, "step4/normratio_g_w")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for h in sel_heads(e):
        c = f"C{h % 10}"
        band_plot(axes[0, 0], pos, e[:, h, 1:], c, log_y=True)
        band_plot(axes[0, 1], pos, cosgw[:, h, 1:], c)
        band_plot(axes[0, 2], pos, nr[:, h, 1:], c)
        band_plot(axes[1, 0], pos, np.sqrt(pos)[None, :] * e[:, h, 1:], c)
        band_plot(axes[1, 1], pos, A_conc[:, h, 1:], c, log_y=True)
        band_plot(axes[1, 2], pos, coher[:, h, 1:], c)
    axes[0, 0].set_ylabel(r"$e_{b,i}$ (relative residual)")
    ref = np.nanmedian(e[:, :, 16]) * np.sqrt(16) / np.sqrt(pos)
    axes[0, 0].plot(pos, ref, "k--", lw=1, label=r"$\propto i^{-1/2}$")
    axes[0, 0].legend()
    axes[0, 1].set_ylabel(r"$\cos(g, w_{nonBOS})$")
    axes[0, 2].set_ylabel(r"$\|g\|/\|w_{nonBOS}\|$")
    axes[0, 2].axhline(1.0, ls="--", c="k", lw=0.8)
    axes[1, 0].set_ylabel(r"$\sqrt{i}\,e_{b,i}$")
    axes[1, 1].set_ylabel(r"$A_{b,i}=i\sum_j q^2$")
    axes[1, 2].set_ylabel(r"coherence $\sqrt{1+C/D}$")
    axes[1, 2].axhline(1.0, ls="--", c="k", lw=0.8)
    for ax in axes.flat:
        ax.set_xlabel("position $i$")
    fig.suptitle(f"Step 4: fixed-vector residual and exact factorization ({args.run_id})")
    fig.tight_layout(); fig.savefig(fig_dir / "step4_residual.png", dpi=200)
    plt.close(fig)

    # ---- Step 4.8 / reconstruction -----------------------------------------
    rec = np.load(out / "step5__recon_rel_err.npy")
    fig, ax = plt.subplots(figsize=(6, 4))
    band_plot(ax, pos, rec[:, 1:], "C0")
    ax.set_xlabel("position $i$")
    ax.set_ylabel(r"$\|o - o^\ast\| / \|o - b_{attn}\|$")
    ax.set_title(f"Ideal-mixture reconstruction error ({args.run_id})")
    fig.tight_layout(); fig.savefig(fig_dir / "step48_reconstruction.png", dpi=200)
    plt.close(fig)

    # ---- Step 5.1: affine coordinate ---------------------------------------
    Y = load(out, "step5/Y_proj")
    import torch
    refs = torch.load(out / "ref_stats.pt", map_location="cpu", weights_only=False)
    w_nb = refs["w_nonbos_2"].numpy(); w_b = refs["w_bos_2"].numpy()
    dw = w_nb - w_b
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h0 = args.head if args.head is not None else 0
    a_flat = alpha[:, h0, 1:].reshape(-1)
    y_flat = Y[:, h0, 1:].reshape(-1)
    sub = np.random.default_rng(0).choice(len(a_flat), min(20000, len(a_flat)),
                                          replace=False)
    axes[0].scatter(a_flat[sub], y_flat[sub], s=2, alpha=0.2)
    c0 = w_nb[h0] @ dw[h0]; d2 = dw[h0] @ dw[h0]
    aa = np.linspace(np.nanmin(a_flat), np.nanmax(a_flat), 50)
    axes[0].plot(aa, c0 - aa * d2, "r--", lw=1.5, label="theoretical affine")
    axes[0].set_xlabel(r"$\alpha_{i,BOS}$"); axes[0].set_ylabel(r"$Y_{b,i}$")
    axes[0].legend(); axes[0].set_title(f"head {h0}")
    band_plot(axes[1], pos, Y[:, h0, 1:], "C0")
    axes[1].set_xlabel("position $i$"); axes[1].set_ylabel(r"$Y_{b,i}$")
    fig.suptitle(f"Step 5.1: affine BOS-weight coordinate ({args.run_id})")
    fig.tight_layout(); fig.savefig(fig_dir / "step51_affine.png", dpi=200)
    plt.close(fig)

    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
