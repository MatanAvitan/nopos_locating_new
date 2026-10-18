"""How Layer 1 shapes the BOS share: the transfer curve f and noise transfer.

Model of the share at the Layer-2 input:
    beta_j = f(s_j) + eta_j,   s_j = alpha1_{j,BOS} + eps_j
where s_j is the centered BOS share after Layer-1 attention and the residual
add (the input to the Layer-1 MLP), and f is what the MLP and the Layer-2
LayerNorm compute. Reads beta_v2/evaluation.npz (b_post1, b_x2, a1_bos).

Measurements:
  1. the curve f: beta_j at the Layer-2 input against the Layer-1 BOS weight
     and against s_j, one point per token and sequence;
  2. noise transfer per position j: sd over sequences of s_j (noise before),
     sd of beta_j (noise after), and the local slope f'_j = cov/var of beta_j
     on s_j across sequences at fixed j (at fixed j, all variation across
     sequences is noise, so f'_j is exactly how much noise passes);
  3. distinguishability before f: gap_j = E[s_j] - E[s_{j+1}] over sd(s_j),
     and the first j where it drops below 1.

Usage: python beta_f_curve.py --models 1h pv_unifL1_s42 12h 1h_8192 1h_seed1 lm
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
from beta_common import MODELS, RESULTS_ROOT

N_SCATTER_SEQ = 200


def analyse(key):
    _, _, L, run_id = MODELS[key]
    d = RESULTS_ROOT / run_id / "beta_v2"
    E = np.load(d / "evaluation.npz")
    s = E["b_post1"].astype(np.float64)          # input to f
    b = E["b_x2"].astype(np.float64)             # output of f
    a1 = E["a1_bos"].astype(np.float64)          # Layer-1 BOS weight (head mean)
    J = min(L - 1, 1023)
    j = np.arange(1, J + 1)
    sd_in, sd_out = s[:, j].std(0), b[:, j].std(0)
    sc = s[:, j] - s[:, j].mean(0); bc = b[:, j] - b[:, j].mean(0)
    slope = (sc * bc).mean(0) / (sc * sc).mean(0)
    r2 = (sc * bc).mean(0) ** 2 / ((sc * sc).mean(0) * (bc * bc).mean(0))
    m = s.mean(0)
    gap = np.full(len(j), np.nan)
    ok = j + 1 <= L - 1
    gap[ok] = m[j[ok]] - m[j[ok] + 1]
    dist = gap / sd_in
    below = np.nonzero(dist < 1)[0]
    j_lost = int(j[below[0]]) if len(below) else None
    pick = lambda arr, ks: {str(k): float(arr[k - 1]) for k in ks if k <= J}
    ks = [1, 2, 3, 4, 6, 8, 16, 64, 256, 1023]
    out = {"key": key, "L": L,
           "mean_s": pick(s[:, j].mean(0), ks), "mean_beta": pick(b[:, j].mean(0), ks),
           "mean_alpha1": pick(a1[:, j].mean(0), ks),
           "sd_before": pick(sd_in, ks), "sd_after": pick(sd_out, ks),
           "local_slope": pick(slope, ks), "local_r2": pick(r2, ks),
           "gap_over_sd_before": pick(dist, ks), "first_j_gap_below_sd": j_lost,
           "noise_ratio_after_over_before_median_j16on": float(np.median(sd_out[15:] / sd_in[15:])) if J > 16 else None}
    return out, (s, b, a1, j, sd_in, sd_out, slope, dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    od = RESULTS_ROOT / "sweep"
    od.mkdir(parents=True, exist_ok=True)
    res, figs = [], {}
    for k in args.models:
        o, arrs = analyse(k)
        res.append(o); figs[k] = arrs
    (od / "f_curve.json").write_text(json.dumps(res, indent=2))

    n = len(args.models)
    fig, ax = plt.subplots(n, 3, figsize=(17, 4.2 * n), squeeze=False)
    rng = np.random.default_rng(0)
    for r, k in enumerate(args.models):
        s, b, a1, j, sd_in, sd_out, slope, dist = figs[k]
        sel = rng.choice(len(s), min(N_SCATTER_SEQ, len(s)), replace=False)
        jj = j[j <= 64]
        cols = np.log(np.broadcast_to(jj, (len(sel), len(jj))))
        ax[r, 0].scatter(a1[np.ix_(sel, jj)], b[np.ix_(sel, jj)], c=cols, s=2, cmap="viridis", alpha=.5)
        ax[r, 0].set_xlabel(r"Layer-1 BOS weight $\alpha^{(1)}_{j,BOS}$"); ax[r, 0].set_ylabel(r"share $\beta_j$ at Layer-2 input")
        ax[r, 0].set_xscale("log"); ax[r, 0].set_title(f"{k}: the curve f (j ≤ 64, color = log j)")
        ax[r, 1].scatter(s[np.ix_(sel, jj)], b[np.ix_(sel, jj)], c=cols, s=2, cmap="viridis", alpha=.5)
        ax[r, 1].plot(s[:, jj].mean(0), b[:, jj].mean(0), "r.-", ms=3, lw=.8, label="mean per j")
        ax[r, 1].set_xlabel(r"share before the MLP, $s_j$"); ax[r, 1].set_ylabel(r"$\beta_j$"); ax[r, 1].legend(fontsize=7)
        ax[r, 1].set_title("f against its actual input")
        ax[r, 2].plot(j, sd_in, label="sd before f (s_j)")
        ax[r, 2].plot(j, sd_out, label=r"sd after f ($\beta_j$)")
        ax[r, 2].plot(j, np.abs(slope), label=r"local slope $|f'_j|$")
        ax[r, 2].plot(j, np.clip(dist, 1e-4, None), "k--", lw=.8, label="gap(j, j+1) / sd before")
        ax[r, 2].axhline(1, color="k", lw=.4)
        ax[r, 2].set_xscale("log"); ax[r, 2].set_yscale("log"); ax[r, 2].legend(fontsize=7)
        ax[r, 2].set_xlabel("position j"); ax[r, 2].set_title("noise transfer")
    fig.tight_layout(); fig.savefig(od / "f_curve.png", dpi=110)
    for o in res:
        print(o["key"], "first j with gap < sd:", o["first_j_gap_below_sd"],
              "| after/before noise (j>=16):", o["noise_ratio_after_over_before_median_j16on"])
        for f in ["mean_alpha1", "mean_s", "mean_beta", "sd_before", "sd_after", "local_slope", "gap_over_sd_before"]:
            print(f"   {f:20s}", {k: round(v, 4) for k, v in o[f].items()})


if __name__ == "__main__":
    main()
