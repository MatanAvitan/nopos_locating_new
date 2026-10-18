"""Additions of the revised BOS-share spec (beta_spec.md): sanity check 2,
t-hat, C2b, C3b, C6, C9, C10, B4, D4-D7. Needs beta_extract.py outputs.

Usage: python beta_extras.py --model 1h --device cuda:1
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
from beta_common import (MODELS, attn_scores_weights, bins_for, gamma_fit,
                         gamma_windows, make_adapter, out_dir, spec_splits)
from beta_extract import batches, per_query_ols, share
from common import attn_bias_vector, head_weights, ov_images

PAIRS = [(1, 0), (2, 0), (5, 1), (10, 3), (50, 20), (200, 100)]
LAGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def lag_corr(X, lags):
    """Mean over j of corr(X[:, j], X[:, j+k]) after removing the column mean."""
    Xc = X - X.mean(0)
    sd = Xc.std(0)
    out = {}
    for k in lags:
        if k >= X.shape[1] - 2:
            continue
        c = (Xc[:, :-k] * Xc[:, k:]).mean(0) / (sd[:-k] * sd[k:])
        out[str(k)] = float(np.nanmean(c))
    return out


@torch.no_grad()
def gpu_pass(ad, R, offsets, L, bs, dev):
    H = ad.H
    heads = R["bos_heads"] or [int(R["r_h"].argmax())]
    hw1, hw2 = head_weights(ad.attn1), head_weights(ad.attn2)
    xb2, xb1 = R["xb2"], R["xb1"]
    Bh = R["B_h"]
    wb = R["w_bos"]                                             # [H,d]
    uW = wb / wb.norm(dim=-1, keepdim=True)
    pm = torch.einsum("htd,hd->ht", R["m_j"], uW)               # <m_j, w_hat>
    battn = attn_bias_vector(hw2).double()
    kx = torch.einsum("hed,d->he", hw2.W_k.double(), xb2)       # W_K x_BOS
    o = {k: [] for k in ["t_hat", "c6_cos", "neff", "frac_1_10", "zeta",
                         "d7_early", "d7_late", "a1_bos", "ev1", "r2_1", "slope1"]}
    pair_sd = {p: [] for p in PAIRS if p[0] < L}
    rebuild = 0.0
    for _, idx in batches(offsets, L, bs, dev):
        p = ad.pre(idx)
        x2 = p["x2"].double()
        beta = share(x2, xb2)
        A = p["attn2_w"].double()                                # [B,H,T,T]
        # t-hat from this sequence's own shares
        lg = Bh[None, :, None] * (beta[:, None, :] - 1)
        e = torch.exp(lg - lg.max(-1, keepdim=True).values)
        o["t_hat"].append(((torch.cumsum(e * beta[:, None], -1)) / torch.cumsum(e, -1)).float().cpu())
        # sanity 2: o2 - b_attn = sum_h (t_h w_BOS^h + R_h)
        y = ov_images(p["x2"], hw2).double()                     # [B,H,T,d]
        xi_img = y - beta[:, None, :, None] * wb[None, :, None, :]
        t = torch.einsum("bhij,bj->bhi", A, beta)
        Rh = torch.einsum("bhij,bhjd->bhid", A[..., 1:], xi_img[:, :, 1:])
        rec = (t[..., None] * wb[None, :, None, :] + Rh).sum(1)
        rebuild = max(rebuild, float((rec - (p["o2"].double() - battn)).abs().max()))
        # C6: cos(W_K^T q_i, x_BOS)
        q = torch.einsum("btd,hed->bhte", x2, hw2.W_q.double()) + hw2.b_q.double()[None, :, None]
        kq = torch.einsum("bhte,hed->bhtd", q, hw2.W_k.double())
        o["c6_cos"].append((kq @ xb2 / (kq.norm(dim=-1) * xb2.norm())).float().cpu())
        # D6
        nb = A[..., 1:]
        o["neff"].append((nb.sum(-1) ** 2 / (nb ** 2).sum(-1)).float().cpu())
        o["frac_1_10"].append((nb[..., :10].sum(-1) / nb.sum(-1)).float().cpu())
        # D4/D5/D7: zeta_j = <B_OV xi_j - m_j, w_hat>
        zeta = torch.einsum("bhtd,hd->bht", xi_img, uW) - pm[None]
        zeta[..., 0] = 0
        o["zeta"].append(zeta.float().cpu())
        o["d7_early"].append(torch.einsum("bhij,bhj->bhi", A[..., 1:11], zeta[..., 1:11]).float().cpu())
        o["d7_late"].append(torch.einsum("bhij,bhj->bhi", A[..., 11:], zeta[..., 11:]).float().cpu())
        # C2b: log(alpha_ij / alpha_ik) = s_ij - s_ik across queries i >= j
        s = p["scores2"].double()
        for (j, k) in pair_sd:
            dlt = s[..., j:, j] - s[..., j:, k]                  # [B,H,T-j]
            pair_sd[(j, k)].append(torch.stack([dlt.std(-1), dlt.mean(-1),
                                                (Bh[None] * (beta[:, j] - beta[:, k])[:, None])], -1).cpu())
        # C9: Layer-1 heads
        s1, a1 = attn_scores_weights(p["x1"], hw1)
        b1 = share(p["x1"].double(), xb1)
        q1 = torch.einsum("btd,hed->bhte", p["x1"].double(), hw1.W_q.double()) + hw1.b_q.double()[None, :, None]
        B1i = torch.einsum("bhte,he->bht", q1, torch.einsum("hed,d->he", hw1.W_k.double(), xb1)) * hw1.scale
        sl1, r21, cov1, varb1, vars1 = per_query_ols(s1.double(), b1)
        ev1 = 1 - (vars1 - 2 * B1i * cov1 + B1i ** 2 * varb1) / vars1
        o["a1_bos"].append(a1[..., 0].float().cpu())
        o["ev1"].append(ev1.float().cpu()); o["r2_1"].append(r21.float().cpu())
        o["slope1"].append(sl1.float().cpu())
        del p, A, y, xi_img, Rh, rec, kq, s, s1, a1
    res = {k: torch.cat(v).numpy() for k, v in o.items()}
    res["pairs"] = {f"{j}_{k}": torch.cat(v).numpy() for (j, k), v in pair_sd.items()}
    return res, rebuild


def ml_B(t, beta, grid):
    """Maximum-likelihood B: minimise the cross-entropy of the measured rows
    under softmax(B beta). CE_i(B) = -B t_i + logsumexp_{j<=i}(B beta_j)."""
    ce = []
    for B in grid:
        lcs = np.logaddexp.accumulate(B * beta, axis=1)          # [n,T]
        ce.append(np.nanmean((-B * t + lcs)[:, 1:]))
    return float(grid[int(np.argmin(ce))]), np.array(ce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ad, L, run_id = make_adapter(args.model, args.device)
    od = out_dir(run_id)
    fd = od / "figures"
    R = {k: (v.to(args.device) if torch.is_tensor(v) else v)
         for k, v in torch.load(od / "ref.pt", weights_only=False).items()}
    splits = spec_splits(L)
    if ad.kind == "lm":
        ad.fit_frozen(splits["calibration"], L, args.batch_size)
    X, rebuild = gpu_pass(ad, R, splits["evaluation"], L, args.batch_size, args.device)
    np.savez(od / "extras.npz", **{k: v for k, v in X.items() if k != "pairs"})
    E = dict(np.load(od / "evaluation.npz"))
    meta = json.loads((od / "meta.json").read_text())
    H = meta["n_head"]
    heads = meta["bos_heads"] or [int(np.argmax(meta["r_h"]))]
    Bh = np.array(meta["B_h"])
    bins = bins_for(L)
    i = np.arange(L)
    S = {"sanity": {"score_identity": meta.get("c1_identity"),
                    "o2_rebuild_max_abs_err": rebuild,
                    "o2_scale_max_abs": None}}
    S["C2b"], S["C3b"], S["C6"], S["C10"], S["D6"], S["D7"], S["D4"], S["D5"] = ({} for _ in range(8))
    wn = E["wnorm"]
    for h in heads:
        tag = f"h{h}"
        S["C2b"][tag] = {pr: {"sd_over_i_of_log_ratio_median": float(np.median(v[:, h, 0])),
                              "measured_log_ratio_median": float(np.median(v[:, h, 1])),
                              "predicted_B_dbeta_median": float(np.median(v[:, h, 2]))}
                         for pr, v in X["pairs"].items()}
        t, th = E["t"][:, h].astype(np.float64), X["t_hat"][:, h].astype(np.float64)
        zt = E["z_w"][:, h] / wn[h]
        S["C3b"][tag] = {
            "t_hat_relerr_median_by_bin": {n: float(np.median(np.abs(th - t)[:, a:b + 1] / np.abs(t[:, a:b + 1])))
                                           for n, (a, b) in bins.items()},
            "t_hat_vs_t_mean_at": {str(k): [float(th[:, k].mean()), float(t[:, k].mean())] for k in [1, 16, 128, 1023, 8191] if k < L},
            "z_over_wnorm_mean_at": {str(k): float(zt[:, k].mean()) for k in [1, 16, 128, 1023, 8191] if k < L},
            "t_loglog_slope_16_end": gamma_fit(t.mean(0), 16, L - 1),
            "t_over_2_sqrt_i_at": {str(k): float(t[:, k].mean() * np.sqrt(k) / 2) for k in [16, 128, 1023] if k < L}}
        S["C6"][tag] = {"median": float(np.nanmedian(X["c6_cos"][:, h, 1:]))}
        grid = np.linspace(0.01, max(3 * abs(Bh[h]), abs(Bh[h]) + 5), 1201)
        Bml, _ = ml_B(t, E["beta2"].astype(np.float64), grid)
        S["C10"][tag] = {"B_ml": Bml, "B_median_slope": float(Bh[h])}
        S["D6"][tag] = {"neff_mean_at": {str(k): float(X["neff"][:, h, k].mean()) for k in [2, 16, 128, 1023, 8191] if k < L},
                        "frac_keys_1_10_at": {str(k): float(X["frac_1_10"][:, h, k].mean()) for k in [16, 128, 1023, 8191] if k < L}}
        S["D7"][tag] = {"sd_keys_1_10_at": {str(k): float(X["d7_early"][:, h, k].std()) for k in [16, 128, 1023] if k < L},
                        "sd_keys_11_on_at": {str(k): float(X["d7_late"][:, h, k].std()) for k in [16, 128, 1023] if k < L}}
        S["D4"][tag] = lag_corr(X["zeta"][:, h, 1:].astype(np.float64), LAGS)
        tot = X["d7_early"][:, h] + X["d7_late"][:, h]
        sd = tot.std(0)
        m = (i >= 16) & (sd > 0)
        S["D5"][tag] = {"sd_at": {str(k): float(sd[k]) for k in [16, 128, 1023] if k < L},
                        "loglog_slope_16_end": float(np.polyfit(np.log(i[m]), np.log(sd[m]), 1)[0])}
    S["B4"] = lag_corr(E["beta2"][:, 1:].astype(np.float64), LAGS)
    # C9: Layer-1 control, one-parameter fit alpha = e^B/(e^B + i)
    S["C9"] = {}
    a1 = X["a1_bos"].mean(0)
    for h in range(a1.shape[0]):
        grid = np.linspace(-5, 15, 2001)
        err = [np.mean((np.log(np.exp(B) / (np.exp(B) + i[1:])) - np.log(a1[h, 1:])) ** 2) for B in grid]
        Bf = float(grid[int(np.argmin(err))])
        pred = np.log(np.exp(Bf) / (np.exp(Bf) + i[1:]))
        ss = np.sum((np.log(a1[h, 1:]) - np.log(a1[h, 1:]).mean()) ** 2)
        S["C9"][f"L1_h{h}"] = {
            "B_fit": Bf, "log_r2": float(1 - np.sum((pred - np.log(a1[h, 1:])) ** 2) / ss),
            "gamma": {w: gamma_fit(a1[h], a, b) for w, (a, b) in gamma_windows(L).items()},
            "ev_median_by_bin": {n: float(np.nanmedian(X["ev1"][:, h, a:b + 1])) for n, (a, b) in bins.items()},
            "r2_median_by_bin": {n: float(np.nanmedian(X["r2_1"][:, h, a:b + 1])) for n, (a, b) in bins.items()},
            "slope_median": float(np.nanmedian(X["slope1"][:, h, 16:]))}
    prev = json.loads((od / "summary.json").read_text())
    prev.update(S)
    (od / "summary.json").write_text(json.dumps(prev, indent=2))

    h0 = heads[0]
    logi = i[1:]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(logi, E["t"][:, h0].mean(0)[1:], "k", label="t measured")
    ax[0].plot(logi, X["t_hat"][:, h0].mean(0)[1:], "r--", label="t-hat")
    ax[0].plot(logi, (E["z_w"][:, h0] / wn[h0]).mean(0)[1:], "b:", label="z/|w|")
    ax[0].plot(logi, 2 / np.sqrt(logi), "g-.", lw=.8, label="2/sqrt(i)")
    ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=1e-2); ax[0].legend(fontsize=7)
    ax[0].set_title(f"C3b h{h0}")
    for h in heads:
        ax[1].plot(logi, X["neff"][:, h].mean(0)[1:], lw=.8, label=f"h{h}")
    ax[1].plot(logi, logi, "k:", lw=.8, label="i")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_title("D6: N_eff"); ax[1].legend(fontsize=6)
    ax[2].plot(logi, X["d7_early"][:, h0].std(0)[1:], label="keys 1-10")
    ax[2].plot(logi, X["d7_late"][:, h0].std(0)[1:], label="keys 11+")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_title(f"D7 h{h0}: sd of content noise"); ax[2].legend()
    fig.tight_layout(); fig.savefig(fd / "C3b_D6_D7.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for h in range(a1.shape[0]):
        Bf = S["C9"][f"L1_h{h}"]["B_fit"]
        l, = ax.plot(logi, a1[h, 1:], lw=1, label=f"L1 h{h}")
        ax.plot(logi, np.exp(Bf) / (np.exp(Bf) + logi), "--", color=l.get_color(), lw=.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(r"C9: Layer-1 $\alpha_{i,0}$ vs $e^B/(e^B+i)$")
    if a1.shape[0] <= 3:
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fd / "C9_layer1.png", dpi=120); plt.close(fig)
    print(json.dumps({k: S[k] for k in ["sanity", "C2b", "C3b", "C6", "C10", "D6", "D7", "B4", "D4", "D5"]}, indent=1)[:6000])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("B_fit", "log_r2", "gamma", "r2_median_by_bin")}
                      for k, v in list(S["C9"].items())[:3]}, indent=1))


if __name__ == "__main__":
    main()
