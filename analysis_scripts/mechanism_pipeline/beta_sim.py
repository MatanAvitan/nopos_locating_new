"""G1 of the BOS-share spec: idealized simulation, no trained model.

Tokens x_t ~ N(0, I_d) i.i.d., a fixed x_BOS ~ N(0, I_d), Layer-2 inputs
x_j = (x_BOS + sum_{t<=j} x_t)/sqrt(j+1), a random OV matrix with N(0, 1/d)
entries, and w = OV x_BOS. Every readout needs only two scalars per token,
<x_t, OV^T w> and <x_t, x_BOS>. These are drawn from their exact joint
Gaussian law, so the simulation is exact in distribution and cheap at any d.

Settings (readout z_i):
  1 layer1   <x_i, x_BOS>
  2 random   uniform attention over 0..i, values OV x_j, projected on w
  3 qk       softmax_j(Bs beta_j) attention, values OV x_j
  4 ov       uniform attention, values beta_j w
  5 qk_ov    softmax_j(Bs beta_j) attention, values beta_j w
Metric: sd_seq(z_i) / (i |f'(i)|), f the smoothed mean curve; median in bins.

Usage: python beta_sim.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import RESULTS_ROOT, smooth_deriv

DS = [256, 768, 2048]
LS = [1024, 8192]
BS = [2, 3, 5, 7, 9]
N_SEQ = 150


def simulate(d, L, n, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    xb = torch.randn(d, generator=g, dtype=torch.float64)
    OV = torch.randn(d, d, generator=g, dtype=torch.float64) / np.sqrt(d)
    w = OV @ xb
    u = OV.T @ w
    cov = torch.tensor([[u @ u, u @ xb], [u @ xb, xb @ xb]], dtype=torch.float64)
    Lc = torch.linalg.cholesky(cov)
    tok = (torch.randn(n, L - 1, 2, generator=g, dtype=torch.float64) @ Lc.T).to(device)
    j1 = torch.sqrt(torch.arange(1, L + 1, device=device, dtype=torch.float64))
    cs = torch.cumsum(tok, 1)
    a = torch.cat([torch.full((n, 1), float(xb @ u), device=device, dtype=torch.float64),
                   float(xb @ u) + cs[..., 0]], 1) / j1           # <OV x_j, w>
    xbn = float(xb @ xb)
    beta = torch.cat([torch.full((n, 1), xbn, device=device, dtype=torch.float64),
                      xbn + cs[..., 1]], 1) / j1 / xbn
    return a, beta, float(w @ w), xbn


def readouts(a, beta, ww, xbn, Bs):
    n, L = a.shape
    cnt = torch.arange(1, L + 1, device=a.device, dtype=torch.float64)
    e = torch.exp(Bs * (beta - 1.0))
    ce = torch.cumsum(e, 1)
    return {"1_layer1": xbn * beta,
            "2_random": torch.cumsum(a, 1) / cnt,
            "3_qk": torch.cumsum(e * a, 1) / ce,
            "4_ov": ww * torch.cumsum(beta, 1) / cnt,
            "5_qk_ov": ww * torch.cumsum(e * beta, 1) / ce}


def ratio(z, L):
    z = z.cpu().numpy()
    _, der = smooth_deriv(z.mean(0), L)
    i = np.arange(L, dtype=np.float64)
    return z.std(0) / (i * np.abs(der))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = RESULTS_ROOT / "beta_sim"
    od.mkdir(parents=True, exist_ok=True)
    res, curves = {}, {}
    for d in DS:
        for L in LS:
            a, beta, ww, xbn = simulate(d, L, N_SEQ, seed=1000 + d + L, device=dev)
            for Bs in BS:
                for k, z in readouts(a, beta, ww, xbn, Bs).items():
                    if Bs != BS[0] and k in ("1_layer1", "2_random", "4_ov"):
                        continue
                    r = ratio(z, L)
                    i = np.arange(L)
                    m = (i >= 16) & np.isfinite(r) & (r > 0)
                    key = f"d{d}_L{L}_{k}" + (f"_B{Bs}" if k in ("3_qk", "5_qk_ov") else "")
                    res[key] = {"median_16_127": float(np.nanmedian(r[16:128])),
                                "median_128_end": float(np.nanmedian(r[128:])),
                                "loglog_slope_16_end": float(np.polyfit(np.log(i[m]), np.log(r[m]), 1)[0])}
                    if d == 768 and L == 8192:
                        curves[key] = r
            print(d, L, "done")
    (od / "summary.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(figsize=(7, 5))
    i = np.arange(8192)
    for k, r in curves.items():
        if "_B" in k and not k.endswith("_B5"):
            continue
        ax.plot(i[16:], r[16:], label=k.replace("d768_L8192_", ""))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_xlabel("i"); ax.set_ylabel(r"sd$(z_i)/(i|f'(i)|)$"); ax.set_title("G1, d=768, L=8192")
    fig.tight_layout(); fig.savefig(od / "G1_ratio_curves.png", dpi=120)
    for k, v in res.items():
        print(f"{k:32s} {v['median_16_127']:.4f} {v['median_128_end']:.4f} slope={v['loglog_slope_16_end']:+.3f}")


if __name__ == "__main__":
    main()
