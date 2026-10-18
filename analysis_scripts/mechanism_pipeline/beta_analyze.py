"""Analysis and figures for the BOS-share spec (beta_spec.md), CPU only.

Reads results/mechanism/<run_id>/beta/{ref.pt,calibration.npz,evaluation.npz}
written by beta_extract.py and writes summary.json plus figures for
A0-A4, B1, C1-C5, C7, C8, D1-D3.

Usage: python beta_analyze.py --model 1h
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
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, bins_for, err_by_bin, gamma_fit,
                         gamma_windows, out_dir, smooth_deriv)

N_PANEL_SEQS = 8
N_BUNDLE = 40


def rowwise_spearman(Z):
    """Spearman of each row of Z [n,T'] with its column index."""
    r = np.argsort(np.argsort(Z, 1), 1).astype(np.float64)
    k = np.arange(Z.shape[1], dtype=np.float64)
    r -= r.mean(1, keepdims=True); k -= k.mean()
    return (r @ k) / np.sqrt((r * r).sum(1) * (k * k).sum())


def strictly_monotone(Z):
    d = np.diff(Z, axis=1)
    return (d > 0).all(1) | (d < 0).all(1)


def bin_median(v, L, bins):
    return {n: float(np.nanmedian(v[..., a:b + 1])) for n, (a, b) in bins.items()}


def curve_stats(X):
    return {"mean": np.nanmean(X, 0), "sd": np.nanstd(X, 0),
            "q10": np.nanquantile(X, 0.1, 0), "q90": np.nanquantile(X, 0.9, 0)}


def alpha_hat_from_curve(B, beta_curve):
    lg = B * (beta_curve - 1.0)
    return np.exp(-np.logaddexp.accumulate(lg))


def isotonic_decode(z_cal, z_eval, L):
    i = np.arange(1, L)
    zc = z_cal[:, 1:].reshape(-1)
    tc = np.tile(i, len(z_cal)).astype(np.float64)
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(zc, tc)
    pred = np.zeros_like(z_eval, dtype=np.float64)
    pred[:, 1:] = iso.predict(z_eval[:, 1:].reshape(-1)).reshape(len(z_eval), -1)
    return pred


def a2_metrics(z_cal, z_eval, L, bins):
    pred = isotonic_decode(z_cal, z_eval, L)
    rho = rowwise_spearman(z_eval[:, 1:])
    i = np.arange(L)
    out = {"spearman_per_seq_median": float(np.median(np.abs(rho))),
           "spearman_per_seq_q10": float(np.quantile(np.abs(rho), 0.1)),
           "frac_strictly_monotone": float(strictly_monotone(z_eval[:, 1:]).mean()),
           "exact_token_acc": float((np.rint(pred[:, 1:]) == i[None, 1:]).mean())}
    out.update(err_by_bin(pred, L, bins))
    return out, pred


def a4_ratio(z, L):
    """sd over sequences of z_i divided by i |f'(i)|."""
    f = z.mean(0)
    _, der = smooth_deriv(f, L)
    i = np.arange(L, dtype=np.float64)
    return z.std(0) / (i * np.abs(der))


def loglog_slope(r, lo, hi):
    i = np.arange(len(r))
    m = (i >= lo) & (i <= hi) & np.isfinite(r) & (r > 0)
    return float(np.polyfit(np.log(i[m]), np.log(r[m]), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--init", action="store_true")
    args = ap.parse_args()
    _, _, L, run_id = MODELS[args.model]
    if args.init:
        run_id += "_init"
    od = out_dir(run_id)
    fd = od / "figures"
    meta = json.loads((od / "meta.json").read_text())
    R = torch.load(od / "ref.pt", weights_only=False)
    E = dict(np.load(od / "evaluation.npz"))
    C = dict(np.load(od / "calibration.npz"))
    H = meta["n_head"]
    heads = meta["bos_heads"] or [int(np.argmax(meta["r_h"]))]
    AGG = H                           # row index of the aggregate readout
    bins = bins_for(L)
    i = np.arange(L)
    n = len(E["beta2"])
    rng = np.random.default_rng(0)
    S = {"meta": meta}
    Bh = np.array(meta["B_h"])

    # ---------------- A3: readout direction ----------------------------------
    wb, wnb = R["w_bos"].numpy(), R["w_nonbos"].numpy()
    dw = wnb - wb
    cosA3 = [float(dw[h] @ -wb[h] / np.linalg.norm(dw[h]) / np.linalg.norm(wb[h]))
             for h in range(H)]
    S["A3_cos_dw_neg_wbos"] = {f"h{h}": cosA3[h] for h in range(H)}

    # ---------------- A1/A2/A4: sequence-level readout -----------------------
    readouts = {"L2_agg_w": (C["z_w"][:, AGG], E["z_w"][:, AGG]),
                "L2_agg_dw": (C["z_dw"][:, AGG], E["z_dw"][:, AGG]),
                "L1_w": (C["z1_w"], E["z1_w"]), "L1_dw": (C["z1_dw"], E["z1_dw"])}
    if len(heads) > 1:
        for h in heads:
            readouts[f"L2_h{h}_w"] = (C["z_w"][:, h], E["z_w"][:, h])
    S["A2"], preds = {}, {}
    for k, (zc, ze) in readouts.items():
        S["A2"][k], preds[k] = a2_metrics(zc.astype(np.float64), ze.astype(np.float64), L, bins)
    S["A2"]["frozen_head"] = err_by_bin(E["frozen"].astype(np.float64), L, bins)
    S["A4"] = {}
    a4 = {}
    for k in ["L1_w", "L2_agg_w", "L2_agg_dw"]:
        a4[k] = a4_ratio(readouts[k][1].astype(np.float64), L)
        hi = L - 1
        S["A4"][k] = {"loglog_slope_16_end": loglog_slope(a4[k], 16, hi),
                      "median_16_127": float(np.nanmedian(a4[k][16:128])),
                      "median_128_end": float(np.nanmedian(a4[k][128:])) if L > 128 else None}

    # ---------------- B1: shares ---------------------------------------------
    S["B1"] = {}
    for k in ["beta2", "beta2c", "beta1", "beta1c"]:
        cs = curve_stats(E[k])
        jj = np.arange(16, L)
        y = cs["mean"][16:]
        ok = y > 0
        p = float(-np.polyfit(np.log(jj[ok] + 1), np.log(y[ok]), 1)[0]) if ok.sum() > 3 else None
        S["B1"][k] = {"p_fit_16_end": p,
                      "mean_at": {str(j): float(cs["mean"][j]) for j in [1, 4, 16, 64, 256, 1023, 8191] if j < L},
                      "sd_at": {str(j): float(cs["sd"][j]) for j in [1, 16, 256, 1023, 8191] if j < L}}

    # ---------------- C1/C2/C8: attention follows the shares ------------------
    S["C1"] = {"identity": meta.get("c1_identity")}
    S["C2"], S["C8"], S["C5"] = {}, {}, {}
    for h in range(H):
        Bi = E["Bi"][:, h].astype(np.float64)
        cv_i = np.nanmedian(Bi[:, 1:].std(1) / np.abs(Bi[:, 1:].mean(1)))
        cv_b = np.nanmedian(Bi[:, 1:].std(0) / np.abs(Bi[:, 1:].mean(0)))
        S["C1"][f"h{h}"] = {"ev_median_by_bin": bin_median(np.nanmedian(E["ev"][:, h], 0), L, bins),
                            "ev_median_all_queries_by_bin": {
                                nm: float(np.nanmedian(E["ev"][:, h, a:b + 1])) for nm, (a, b) in bins.items()},
                            "Bi_cv_across_i": float(cv_i), "Bi_cv_across_seq": float(cv_b),
                            "Bi_mean": float(np.nanmean(Bi[:, 1:]))}
        sl = E["slope"][:, h, 16:]
        S["C2"][f"h{h}"] = {"slope_q": {q: float(np.nanquantile(sl, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                            "r2_median_by_bin": {nm: float(np.nanmedian(E["r2"][:, h, a:b + 1])) for nm, (a, b) in bins.items()},
                            "B_h_ref": float(Bh[h])}
        x, y = E["beta2"][:, 1:].reshape(-1).astype(np.float64), Bi[:, 1:].reshape(-1)
        c = np.polyfit(x, y, 1)
        S["C8"][f"h{h}"] = {"slope": float(c[0]), "r2": float(np.corrcoef(x, y)[0, 1] ** 2)}
        S["C5"][f"h{h}"] = {"B_h": float(Bh[h]), "ln_r_h": float(np.log(meta["r_h"][h])),
                            "bos_head": h in heads}

    # ---------------- C3/C4/C7: predicted vs measured attention ---------------
    S["C3"], S["C4"], S["C7"] = {}, {}, {}
    beta_ref = R["beta2_ref"].numpy()
    lc = E["logcorr_sums"]
    for h in heads:
        a_meas = E["alpha_bos"][:, h].astype(np.float64)
        a_own = E["alpha_hat_bos"][:, h].astype(np.float64)
        a_refc = alpha_hat_from_curve(Bh[h], beta_ref)
        mm, mo = a_meas.mean(0), a_own.mean(0)
        N, sx, sy, sxy, sxx, syy = lc[h]
        cov = sxy / N - sx / N * sy / N
        r = cov / np.sqrt((sxx / N - (sx / N) ** 2) * (syy / N - (sy / N) ** 2))
        sub = E["sub_relerr"][:, h]; subi = E["sub_i"]
        S["C3"][f"h{h}"] = {
            "relerr_bos_own_by_bin": bin_median(np.abs(a_own - a_meas) / a_meas, L, bins),
            "relerr_bos_refmean_by_bin": bin_median(np.abs(a_refc[None] - a_meas) / a_meas, L, bins),
            "relerr_all_entries_own_by_bin": {nm: float(np.median(sub[(subi >= a) & (subi <= b)]))
                                              for nm, (a, b) in bins.items()},
            "corr_log_alpha_all_entries_own": float(r),
            "gamma": {w: {"measured": gamma_fit(mm, a, b), "pred_own": gamma_fit(mo, a, b),
                          "pred_refmean": gamma_fit(a_refc, a, b)}
                      for w, (a, b) in gamma_windows(L).items()}}
        S["C4"][f"h{h}"] = {w: gamma_fit(mm, a, b) for w, (a, b) in gamma_windows(L).items()}
        f = np.log(mm)
        _, der = smooth_deriv(f, L)
        loc = -(i * E["alpha_ii"][:, h].mean(0))
        dl = i * der
        sel = np.arange(L) >= 16
        S["C7"][f"h{h}"] = {"median_abs_diff_16_end": float(np.nanmedian(np.abs(loc - dl)[sel])),
                            "at": {str(k): [float(loc[k]), float(dl[k])] for k in [16, 64, 127, 256, 1023, 8191] if k < L}}

    # ---------------- D1/D2/D3: output composition ---------------------------
    S["D1"], S["D2"], S["D3"] = {}, {}, {}
    wn = E["wnorm"]
    mj_dw = np.einsum("htd,hd->ht", R["m_j"].numpy(), dw / np.linalg.norm(dw, axis=1, keepdims=True))
    w_dot_dwhat = np.einsum("hd,hd->h", wb, dw / np.linalg.norm(dw, axis=1, keepdims=True))
    for h in heads:
        t = E["t"][:, h].astype(np.float64)
        dif = np.diff(t[:, 1:], axis=1)
        rho = rowwise_spearman(t[:, 1:])
        S["D1"][f"h{h}"] = {"frac_t_monotone_decreasing": float((dif <= 0).all(1).mean()),
                            "frac_t_spearman_lt_-0.9": float((rho < -0.9).mean()),
                            "t_spearman_median": float(np.median(rho)),
                            "carried_share_of_t_at": {str(k): float(np.mean((t[:, k] - E["alpha_bos"][:, h, k]) / t[:, k]))
                                                      for k in [16, 128, 512, 1023, 8191] if k < L}}
        Rw = E["z_w"][:, h] - t * wn[h]
        Rdw = E["z_dw"][:, h] - t * w_dot_dwhat[h]
        S["D2"][f"h{h}"] = {"R_w_mean_at": {str(k): float(Rw[:, k].mean()) for k in [1, 16, 128, 1023] if k < L},
                            "R_w_sd_at": {str(k): float(Rw[:, k].std()) for k in [1, 16, 128, 1023] if k < L},
                            "R_dw_mean_at": {str(k): float(Rdw[:, k].mean()) for k in [1, 16, 128, 1023] if k < L},
                            "m_j_dw_range_j_ge_1": [float(mj_dw[h, 1:].min()), float(mj_dw[h, 1:].max())],
                            "m_j_dw_sd_j_ge_1": float(mj_dw[h, 1:].std())}
        z = E["z_w"][:, h].astype(np.float64)
        p1 = E["alpha_bos"][:, h] * wn[h]
        p2 = (t - E["alpha_bos"][:, h]) * wn[h]
        p3 = E["d3_mean"][:, h].astype(np.float64)
        p4 = z - p1 - p2 - p3
        parts = {"1_boskey": p1, "2_carried": p2, "3_mean_content": p3, "4_remainder": p4}
        sl = slice(1, L)
        zm = z.mean(0)[sl]
        zc = z - z.mean(0)
        S["D3"][f"h{h}"] = {
            "trend_cov_share": {k: float(np.cov(v.mean(0)[sl], zm)[0, 1] / zm.var(ddof=1)) for k, v in parts.items()},
            "trend_var_ratio": {k: float(v.mean(0)[sl].var() / zm.var()) for k, v in parts.items()},
            "seq_cov_share": {k: float(((v - v.mean(0)) * zc)[:, sl].sum() / (zc[:, sl] ** 2).sum()) for k, v in parts.items()}}

    # ---------------- A0: per-sequence fractions -----------------------------
    S["A0"] = {}
    bdec = rowwise_spearman(E["beta2"][:, 1:]) < -0.9
    per_head = {}
    for h in heads:
        r2ok = np.nanmedian(E["r2"][:, h, 16:], 1) > 0.8
        aerr = np.nanmedian(np.abs(E["alpha_hat_bos"][:, h] - E["alpha_bos"][:, h])[:, 1:]
                            / E["alpha_bos"][:, h, 1:], 1) < 0.2
        tdec = (np.diff(E["t"][:, h, 1:], axis=1) <= 0).all(1)
        tdec_rho = rowwise_spearman(E["t"][:, h, 1:]) < -0.9
        per_head[h] = (r2ok, aerr, tdec, tdec_rho)
    zmono = strictly_monotone(E["z_w"][:, AGG, 1:])
    zrho = np.abs(rowwise_spearman(E["z_w"][:, AGG, 1:])) > 0.9
    allh = lambda k: np.all([per_head[h][k] for h in heads], 0)
    S["A0"]["fractions"] = {
        "beta_decreasing_spearman_lt_-0.9": float(bdec.mean()),
        "r2_median_gt_0.8(all BOS heads)": float(allh(0).mean()),
        "alpha_relerr_median_lt_0.2(all BOS heads)": float(allh(1).mean()),
        "t_monotone_decreasing(all BOS heads)": float(allh(2).mean()),
        "t_spearman_lt_-0.9(all BOS heads)": float(allh(3).mean()),
        "z_strictly_monotone": float(zmono.mean()),
        "z_abs_spearman_gt_0.9": float(zrho.mean())}
    if len(heads) > 1:
        S["A0"]["per_head"] = {f"h{h}": {k: float(v.mean()) for k, v in zip(
            ["r2", "alpha_err", "t_mono", "t_rho"], per_head[h])} for h in heads}

    (od / "summary.json").write_text(json.dumps(S, indent=2))

    # ======================= figures =========================================
    logi = i[1:]
    sel_seq = rng.choice(n, N_PANEL_SEQS, replace=False)
    bundle = rng.choice(n, min(N_BUNDLE, n), replace=False)
    iso_pred = preds["L2_agg_w"]

    # A0 panels
    rows_h = heads if len(heads) <= 3 else heads[:3]
    for s_ in sel_seq:
        fig, ax = plt.subplots(5, 1, figsize=(7, 15))
        ax[0].plot(logi, E["beta2"][s_, 1:], lw=.7, label=r"$\beta_j$ (this seq)")
        ax[0].plot(logi, beta_ref[1:], "k--", lw=1, label=r"$\bar\beta_j$ (ref)")
        ax[0].set_ylabel("BOS share"); ax[0].legend(fontsize=7)
        for h in rows_h:
            ax[1].plot(logi, E["alpha_bos"][s_, h, 1:], lw=.8, label=f"h{h} measured")
            ax[1].plot(logi, E["alpha_hat_bos"][s_, h, 1:], "--", lw=.8, label=f"h{h} predicted")
            tt = E["t"][s_, h]
            ax[2].plot(logi, E["alpha_bos"][s_, h, 1:], lw=.8, label=f"h{h} BOS-key")
            ax[2].plot(logi, (tt - E["alpha_bos"][s_, h])[1:], lw=.8, label=f"h{h} carried")
            ax[2].plot(logi, tt[1:], "k", lw=.8, label=f"h{h} t_i")
        ax[1].set_yscale("log"); ax[1].set_ylabel(r"$\alpha_{i,BOS}$"); ax[1].legend(fontsize=6)
        ax[2].set_ylabel("BOS in output"); ax[2].legend(fontsize=6)
        ax[3].plot(logi, E["z_w"][s_, AGG, 1:], lw=.8, label=r"along $\hat w_{BOS}$")
        ax[3].plot(logi, E["z_m"][s_, AGG, 1:], lw=.8, label=r"along $\bar m_\perp$")
        ax[3].set_ylabel("output proj."); ax[3].legend(fontsize=7)
        ins = ax[3].inset_axes([0.62, 0.45, 0.35, 0.5])
        ins.scatter(E["z_w"][s_, AGG, 1:], E["z_m"][s_, AGG, 1:], c=np.log(logi), s=2, cmap="viridis")
        ins.set_xticks([]); ins.set_yticks([])
        ax[4].plot(logi, iso_pred[s_, 1:], lw=.8, label="isotonic decode")
        ax[4].plot(logi, logi, "k--", lw=.8, label="true")
        ax[4].set_yscale("log"); ax[4].set_ylabel("decoded position"); ax[4].legend(fontsize=7)
        for a in ax:
            a.set_xscale("log")
        ax[4].set_xlabel("position i")
        fig.suptitle(f"{run_id}: eval seq {s_}")
        fig.tight_layout(); fig.savefig(fd / f"A0_seq{s_}.png", dpi=110); plt.close(fig)

    # A1 bundles
    keys = [("L2_agg_w", E["z_w"][:, AGG]), ("L2_agg_dw", E["z_dw"][:, AGG]),
            ("L1_w", E["z1_w"]), ("L1_dw", E["z1_dw"])]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, (k, Z) in zip(ax, keys):
        for s_ in bundle:
            a.plot(logi, Z[s_, 1:], lw=.3, alpha=.5)
        mu, sd = Z.mean(0)[1:], Z.std(0)[1:]
        a.plot(logi, mu, "k", lw=1.2); a.fill_between(logi, mu - sd, mu + sd, color="k", alpha=.15)
        a.set_xscale("log"); a.set_title(k); a.set_xlabel("position i")
    fig.tight_layout(); fig.savefig(fd / "A1_bundles.png", dpi=120); plt.close(fig)
    if len(heads) > 1:
        nc = min(len(heads), 5); nr = int(np.ceil(len(heads) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(3.6 * nc, 3 * nr), squeeze=False)
        for a, h in zip(ax.flat, heads):
            Z = E["z_w"][:, h]
            for s_ in bundle:
                a.plot(logi, Z[s_, 1:], lw=.3, alpha=.5)
            a.plot(logi, Z.mean(0)[1:], "k"); a.set_xscale("log"); a.set_title(f"head {h}")
        fig.tight_layout(); fig.savefig(fd / "A1_per_head.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for s_ in bundle[:15]:
        sc = ax.scatter(E["z_w"][s_, AGG, 1:], E["z_m"][s_, AGG, 1:], c=np.log(logi), s=1.5, cmap="viridis")
    plt.colorbar(sc, label="log i"); ax.set_xlabel(r"along $\hat w_{BOS}$"); ax.set_ylabel(r"along $\bar m_\perp$")
    fig.tight_layout(); fig.savefig(fd / "A1_traj2d.png", dpi=120); plt.close(fig)

    # A2 scatter
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, k in zip(ax, ["L2_agg_w", "L1_w"]):
        P = preds[k][:200, 1:]
        a.scatter(np.tile(logi, len(P)), P.reshape(-1), s=.2, alpha=.2)
        a.plot(logi, logi, "k--", lw=.8); a.set_xscale("log"); a.set_yscale("log")
        a.set_title(f"{k} isotonic decode"); a.set_xlabel("true i")
    fig.tight_layout(); fig.savefig(fd / "A2_scatter.png", dpi=120); plt.close(fig)

    # A4
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, r in a4.items():
        ax.plot(i[4:], r[4:], label=k)
    ax.plot(i[4:], np.sqrt(i[4:]) * np.nanmedian(a4["L1_w"][4:20] / np.sqrt(i[4:20])), "k:", label=r"$\sqrt{i}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(); ax.set_xlabel("i")
    ax.set_ylabel(r"sd$_{seq}(z_i)/(i|f'(i)|)$")
    fig.tight_layout(); fig.savefig(fd / "A4_noise_ratio.png", dpi=120); plt.close(fig)

    # B1
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, k in zip(ax, ["beta2", "beta2c", "beta1", "beta1c"]):
        cs = curve_stats(E[k])
        a.plot(logi, cs["mean"][1:], "k"); a.fill_between(logi, cs["q10"][1:], cs["q90"][1:], alpha=.2)
        a.fill_between(logi, (cs["mean"] - cs["sd"])[1:], (cs["mean"] + cs["sd"])[1:], alpha=.2)
        a.set_xscale("log"); a.set_title(k); a.set_xlabel("j")
    fig.tight_layout(); fig.savefig(fd / "B1_shares.png", dpi=120); plt.close(fig)

    # C3 overlay + C7
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for h in heads:
        l, = ax[0].plot(logi, E["alpha_bos"][:, h].mean(0)[1:], lw=1, label=f"h{h} measured")
        ax[0].plot(logi, E["alpha_hat_bos"][:, h].mean(0)[1:], "--", color=l.get_color(), lw=1)
        ax[0].plot(logi, alpha_hat_from_curve(Bh[h], beta_ref)[1:], ":", color=l.get_color(), lw=1)
        _, der = smooth_deriv(np.log(E["alpha_bos"][:, h].mean(0)), L)
        ax[1].plot(logi, -(i * E["alpha_ii"][:, h].mean(0))[1:], color=l.get_color(), lw=1)
        ax[1].plot(logi, (i * der)[1:], "--", color=l.get_color(), lw=1)
    ax[0].plot(logi, 1 / (logi + 1), "k:", lw=.8, label="1/(i+1)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_title("C3: measured (—), own-β (--), ref-β (··)")
    ax[0].legend(fontsize=6)
    ax[1].set_xscale("log"); ax[1].set_title(r"C7: $-i\alpha_{ii}$ (—) vs $d\log\bar\alpha/d\log i$ (--)")
    fig.tight_layout(); fig.savefig(fd / "C3_C7.png", dpi=120); plt.close(fig)

    # C5
    fig, ax = plt.subplots(figsize=(5, 4.5))
    lr = np.log(meta["r_h"])
    ax.scatter(lr, Bh, c=["C3" if h in heads else "C0" for h in range(H)])
    for h in range(H):
        ax.annotate(str(h), (lr[h], Bh[h]), fontsize=7)
    lim = [min(lr.min(), 0), max(lr.max(), 1)]
    ax.plot(lim, lim, "k:"); ax.set_xlabel(r"$\ln r_h$"); ax.set_ylabel(r"$B^h$")
    if np.abs(Bh).max() > 50:
        ax.set_yscale("symlog")
    fig.tight_layout(); fig.savefig(fd / "C5_B_vs_lnr.png", dpi=120); plt.close(fig)

    # D1 routes (mean and 6 sequences) + D2
    h0 = heads[0]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for s_ in sel_seq[:6]:
        tt = E["t"][s_, h0]
        l, = ax[0].plot(logi, tt[1:], lw=.8)
        ax[0].plot(logi, E["alpha_bos"][s_, h0, 1:], "--", color=l.get_color(), lw=.6)
        ax[0].plot(logi, (tt - E["alpha_bos"][s_, h0])[1:], ":", color=l.get_color(), lw=.6)
    ax[0].set_xscale("log"); ax[0].set_title(f"D1 h{h0}: t (—), BOS-key (--), carried (··)")
    Rw = E["z_w"][:, h0] - E["t"][:, h0] * wn[h0]
    ax[1].plot(logi, Rw.mean(0)[1:], "k"); ax[1].fill_between(logi, (Rw.mean(0) - Rw.std(0))[1:], (Rw.mean(0) + Rw.std(0))[1:], alpha=.2)
    ax[1].set_xscale("log"); ax[1].set_title(r"D2: $\langle R_i,\hat w_{BOS}\rangle$")
    for h in heads:
        ax[2].plot(logi, mj_dw[h, 1:], lw=.7, label=f"h{h}")
    ax[2].set_xscale("log"); ax[2].set_title(r"D2: $\langle m_j,\widehat{\Delta w}\rangle$"); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(fd / "D1_D2.png", dpi=120); plt.close(fig)
    print(json.dumps({k: S[k] for k in ["A0", "A2", "A4"]}, indent=1)[:4000])


if __name__ == "__main__":
    main()
