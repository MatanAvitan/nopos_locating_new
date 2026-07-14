"""Aggregate cached per-sequence arrays into the statistics exps.md requires.

Consumes results/mechanism/<run_id>/*.npy from extract.py, writes:
  - per_position.parquet (position-wise quantiles for every diagnostic)
  - summary.json sections step1/step3/step4 (bootstrap CIs, exponent fits,
    ordering fractions, decision-gate quantities)
  - figures/*.png

CPU only. Usage: python stats.py --run-id attn2_1h_L1024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, bootstrap_ci, position_bins, update_summary

QUANTS = [0.05, 0.25, 0.5, 0.75, 0.95]
QUANTS_HEAVY = QUANTS + [0.9, 0.99]

# exponent-fit ranges chosen BEFORE examining slopes (declared in exps.md
# protocol): fit on positions 16..min(512, L//2) to avoid the earliest
# prefixes and the boundary.
def fit_range(L):
    return 16, min(512, L // 2)


def loglog_slope(x, y):
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    c = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return float(c[0]), float(c[1])


def slope_ci(x, per_seq, n_boot=1000, seed=0, stat=np.nanmean):
    """Bootstrap sequences, refit slope each time. per_seq: [n, len(x)]."""
    rng = np.random.default_rng(seed)
    n = len(per_seq)
    slopes = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        curve = stat(per_seq[idx], axis=0)
        s, _ = loglog_slope(x, curve)
        slopes.append(s)
    slopes = np.array(slopes)
    base, _ = loglog_slope(x, stat(per_seq, axis=0))
    return {"slope": base, "ci95": [float(np.nanquantile(slopes, 0.025)),
                                    float(np.nanquantile(slopes, 0.975))]}


def qtable(arr, name, rows):
    """arr: [n_seq, (H,) T]. Append per-position quantile rows to `rows`."""
    if arr.ndim == 2:
        arr = arr[:, None, :]
    n, H, T = arr.shape
    for h in range(H):
        a = arr[:, h, :]
        med = np.nanmedian(a, 0)
        mean = np.nanmean(a, 0)
        qs = np.nanquantile(a, QUANTS, axis=0)
        for i in range(T):
            rows.append({"metric": name, "head": h, "pos": i,
                         "mean": mean[i], "median": med[i],
                         **{f"q{int(q*100):02d}": qs[k, i]
                            for k, q in enumerate(QUANTS)}})


def load(out, key):
    return np.load(out / (key.replace("/", "__") + ".npy"))


def per_seq_spearman(arr, x, sign=1):
    """arr: [n, len(x)] values aligned with x -> per-sequence spearman."""
    vals = []
    for b in range(arr.shape[0]):
        v = arr[b]
        m = np.isfinite(v)
        if m.sum() > 3:
            vals.append(spearmanr(v[m], x[m]).statistic * sign)
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    out = RESULTS_ROOT / args.run_id
    cfg = json.loads((out / "config.json").read_text())
    L = cfg["context_length"]
    lo, hi = fit_range(L)
    pos_fit = np.arange(lo, hi + 1)
    bins = position_bins(L)
    rows = []

    # ================= Step 1 =================
    s1 = {}
    for key in ["entropy", "norm_entropy", "kl_uniform", "tv_uniform",
                "alpha_bos_ratio", "max_attn"]:
        a = load(out, f"step1/{key}")
        qtable(a, f"step1/{key}", rows)
        s1[f"{key}_pooled_median"] = float(np.nanmedian(a))
        for bname, (a_, b_) in bins.items():
            s1[f"{key}_{bname}_median"] = float(np.nanmedian(a[..., a_:b_ + 1]))
    for key in ["approx_actual_vs_unif", "approx_unif_vs_proxy",
                "approx_actual_vs_proxy"]:
        for suf in ["cos", "scalefit_resid", "relerr"]:
            a = load(out, f"step1/{key}_{suf}")
            qtable(a, f"step1/{key}_{suf}", rows)
            s1[f"{key}_{suf}_median"] = float(np.nanmedian(a[:, 1:]))

    # 1.3 signal/error algebra
    S = load(out, "step1/S")          # [n,T]
    E = load(out, "step1/E")
    Tc = load(out, "step1/T")
    A_i = np.load(out / "step1_A_i.npy")
    pred = A_i[None, :] + E
    m = np.isfinite(S[:, 1:])
    x_, y_ = pred[:, 1:][m], S[:, 1:][m]
    c = np.polyfit(x_, y_, 1)
    ss = 1 - ((y_ - (c[0] * x_ + c[1])) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()
    s1["algebra_slope"] = float(c[0])
    s1["algebra_intercept"] = float(c[1])
    s1["algebra_r2"] = float(ss)
    s1["algebra_relerr_median"] = float(np.nanmedian(
        np.abs(S[:, 1:] - pred[:, 1:]) / np.abs(S[:, 1:])))
    qtable(S, "step1/S", rows)
    qtable(E, "step1/E", rows)
    # exponent fits (declared range)
    sig_curve = A_i[pos_fit]
    s1["signal_slope_theoretical"] = loglog_slope(pos_fit, sig_curve)[0]
    s1["S_mean_slope"] = slope_ci(pos_fit, S[:, pos_fit])
    rmsE = np.sqrt(np.nanmean(E ** 2, 0))
    s1["E_rms_slope"] = loglog_slope(pos_fit, rmsE[pos_fit])[0]
    snr = np.abs(A_i[None, :]) / np.abs(E)
    s1["snr_median_slope"] = slope_ci(pos_fit, snr[:, pos_fit],
                                      stat=lambda a, axis: np.nanmedian(a, axis=axis))
    s1["T_rms_slope"] = loglog_slope(
        pos_fit, np.sqrt(np.nanmean(Tc ** 2, 0))[pos_fit])[0]
    qtable(snr, "step1/snr", rows)

    # 1.4 orthogonality
    x1n = load(out, "step1/x1_norm")
    cwx0 = load(out, "step1/cos_with_x0")
    s1["x1_norm_median"] = float(np.nanmedian(x1n[:, 1:]))
    s1["cos_with_x0_absmedian"] = float(np.nanmedian(np.abs(cwx0[:, 1:])))
    lag_summary = {}
    for f in sorted(out.glob("step1__paircos_lag*.npy")):
        lag = f.stem.split("lag")[1]
        v = np.load(f)
        lag_summary[lag] = {"mean": float(np.nanmean(v)),
                            "q05": float(np.nanquantile(v, 0.05)),
                            "q95": float(np.nanquantile(v, 0.95))}
    s1["paircos_by_lag"] = lag_summary
    update_summary(out, "step1", s1)

    # ================= Step 3 =================
    s3 = {}
    logc = load(out, "step3/log_cbar")      # [n,H,T]
    alpha = load(out, "step3/alpha_bos")
    iden = load(out, "step3/identity_abs_err")
    s3["identity_max_abs_err"] = float(np.nanmax(iden))
    qtable(logc, "step3/log_cbar", rows)
    qtable(alpha, "step3/alpha_bos", rows)
    if logc.ndim == 2:
        logc, alpha = logc[:, None, :], alpha[:, None, :]
    H = logc.shape[1]
    cbar = np.exp(logc)
    per_head = {}
    for h in range(H):
        ch = cbar[:, h, 1:]
        lh = logc[:, h, 1:]
        ah = alpha[:, h, :]
        pos = np.arange(1, L)
        cv = np.nanstd(ch, 0) / np.abs(np.nanmean(ch, 0))
        i_alpha = np.arange(L) * ah
        # scaling exponent of alpha in declared range
        gamma = slope_ci(pos_fit, ah[:, pos_fit],
                         stat=lambda a, axis: np.nanmean(a, axis=axis))
        # sample-level ordering
        rho_seq = per_seq_spearman(ah[:, 1:], pos, sign=-1)
        inv = (ah[:, 1:-1] < ah[:, 2:])  # adjacent inversion (alpha should fall)
        exact_cond = (pos[None, :-1] * ch[:, :-1]) < (pos[None, 1:] * ch[:, 1:])
        # 3.4 position-only prediction error
        err_ref = load(out, "step3/alpha_err_refpred")
        err_fo = load(out, "step3/alpha_err_firstorder")
        if err_ref.ndim == 2:
            err_ref, err_fo = err_ref[:, None, :], err_fo[:, None, :]
        rel_err = np.abs(err_ref[:, h, 1:]) / np.abs(ah[:, 1:])
        fo_match = np.abs(err_ref[:, h, 1:] - (-err_fo[:, h, 1:]))
        per_head[f"head{h}"] = {
            "cbar_cv_median_over_pos": float(np.nanmedian(cv)),
            "cbar_cv_early": float(np.nanmedian(cv[:15])),
            "cbar_cv_late": float(np.nanmedian(cv[min(127, L - 2):])),
            "logc_std_median_over_pos": float(np.nanmedian(np.nanstd(lh, 0))),
            "logc_drift_medianpos": {str(k): float(np.nanmedian(lh[:, k - 1]))
                                     for k in [1, 4, 16, 64, 256, L - 1] if k < L},
            "i_alpha_median_at": {str(k): float(np.nanmedian(i_alpha[:, k]))
                                  for k in [1, 4, 16, 64, 256, L - 1] if k < L},
            "alpha_scaling_gamma": gamma,
            "mean_curve_spearman": float(spearmanr(
                np.nanmean(ah[:, 1:], 0), pos).statistic),
            "per_seq_spearman_median": float(np.nanmedian(rho_seq)),
            "adjacent_inversion_rate": float(np.nanmean(inv)),
            "exact_adjacent_condition_frac": float(np.nanmean(exact_cond)),
            "refpred_alpha_abs_err_median": float(np.nanmedian(
                np.abs(err_ref[:, h, 1:]))),
            "refpred_alpha_rel_err_median": float(np.nanmedian(rel_err)),
            "refpred_alpha_rel_err_q95": float(np.nanquantile(rel_err, 0.95)),
            "firstorder_match_median_abs": float(np.nanmedian(fo_match)),
        }
        # single constant c* decision-gate check
        c_star = float(np.nanmedian(ch))
        alpha_cstar = 1.0 / (1.0 + pos[None, :] * c_star)
        rel_cstar = np.abs(alpha_cstar - ah[:, 1:]) / ah[:, 1:]
        per_head[f"head{h}"]["cstar_global_rel_err_median"] = float(
            np.nanmedian(rel_cstar))
        per_head[f"head{h}"]["cstar_global_rel_err_q95"] = float(
            np.nanquantile(rel_cstar, 0.95))
    s3["per_head"] = per_head
    update_summary(out, "step3", s3)

    # ================= Step 4 =================
    s4 = {}
    e = load(out, "step4/e")
    if e.ndim == 2:
        expand = lambda a: a[:, None, :]
    else:
        expand = lambda a: a
    e = expand(e)
    H = e.shape[1]
    keys = ["cos_g_w", "normratio_g_w", "bound_B", "D", "C", "sumq2",
            "A_conc", "R_scale", "maxq", "entropy_q", "mu_term_norm",
            "eps_term_norm", "mu_eps_inner", "rho", "rho_over_alpha",
            "rho_over_mass", "e_uniform_control", "e_shuffled_control",
            "r_norm_by_j"]
    data = {k: expand(load(out, f"step4/{k}")) for k in keys}
    data["e"] = e
    for k in ["e", "cos_g_w", "normratio_g_w", "bound_B", "A_conc",
              "R_scale", "maxq", "entropy_q", "rho"]:
        qtable(data[k], f"step4/{k}", rows)
    alpha3 = expand(load(out, "step3/alpha_bos"))
    per_head = {}
    pos = np.arange(1, L)
    for h in range(H):
        eh = e[:, h, 1:]
        D, C = data["D"][:, h, 1:], data["C"][:, h, 1:]
        CD = C / np.where(D > 0, D, np.nan)
        coher = np.sqrt(np.clip(1 + CD, 0, None))
        d4 = {
            "e_median_by_bin": {n: float(np.nanmedian(e[:, h, a:b + 1]))
                                for n, (a, b) in bins.items()},
            "e_q90_by_bin": {n: float(np.nanquantile(e[:, h, a:b + 1], 0.9))
                             for n, (a, b) in bins.items()},
            "cos_g_w_median_by_bin": {
                n: float(np.nanmedian(data["cos_g_w"][:, h, a:b + 1]))
                for n, (a, b) in bins.items()},
            "normratio_median_by_bin": {
                n: float(np.nanmedian(data["normratio_g_w"][:, h, a:b + 1]))
                for n, (a, b) in bins.items()},
            "bound_holds_frac": float(np.nanmean(
                eh <= data["bound_B"][:, h, 1:] * (1 + 1e-9))),
            "CD_median": float(np.nanmedian(CD)),
            "CD_pospart_median": float(np.nanmedian(np.clip(CD, 0, None))),
            "coherence_mult_median_by_bin": {
                n: float(np.nanmedian(coher[:, max(a - 1, 0):b]))
                for n, (a, b) in bins.items()},
            "frac_C_le_KD": {str(K): float(np.nanmean(CD <= K))
                             for K in [1, 2, 4, 8]},
            "e_median_slope": slope_ci(
                pos_fit, e[:, h, pos_fit],
                stat=lambda a, axis: np.nanmedian(a, axis=axis)),
            "e_rms_slope": loglog_slope(
                pos_fit, np.sqrt(np.nanmean(e[:, h, pos_fit] ** 2, 0)))[0],
            "e_q90_slope": loglog_slope(
                pos_fit, np.nanquantile(e[:, h, pos_fit], 0.9, axis=0))[0],
            "A_conc_median_by_bin": {
                n: float(np.nanmedian(data["A_conc"][:, h, a:b + 1]))
                for n, (a, b) in bins.items()},
            "A_conc_slope": loglog_slope(
                pos_fit, np.nanmedian(data["A_conc"][:, h, pos_fit], 0))[0],
            "R_scale_median_by_bin": {
                n: float(np.nanmedian(data["R_scale"][:, h, a:b + 1]))
                for n, (a, b) in bins.items()},
            "sqrt_i_e_floor_check": {
                str(k): float(np.nanmedian(np.sqrt(k) * e[:, h, k]))
                for k in [4, 16, 64, 256, L - 1] if k < L},
            "mu_term_median_by_bin": {
                n: float(np.nanmedian(data["mu_term_norm"][:, h, a:b + 1]))
                for n, (a, b) in bins.items()},
            "eps_term_median_by_bin": {
                n: float(np.nanmedian(data["eps_term_norm"][:, h, a:b + 1]))
                for n, (a, b) in bins.items()},
            "e_uniform_ctrl_median": float(np.nanmedian(
                data["e_uniform_control"][:, h, 1:])),
            "e_shuffled_ctrl_median": float(np.nanmedian(
                data["e_shuffled_control"][:, h, 1:])),
            "e_actual_median": float(np.nanmedian(eh)),
            "r_norm_by_j_median": float(np.nanmedian(data["r_norm_by_j"][:, h])),
            "r_norm_j_first16_median": float(np.nanmedian(
                data["r_norm_by_j"][:, h, :16])),
            "r_norm_j_late_median": float(np.nanmedian(
                data["r_norm_by_j"][:, h, min(127, L - 2):])),
        }
        # 4.7 adjacent ordering
        rho = data["rho"][:, h, 1:]
        ah = alpha3[:, h, 1:]
        d_alpha = ah[:, :-1] - ah[:, 1:]
        d_rho = rho[:, :-1] - rho[:, 1:]
        ratio = np.abs(d_rho) / np.abs(d_alpha)
        proj_actual = -ah + rho  # affine coordinate up to const/scale
        order_pred = (proj_actual[:, 1:] > proj_actual[:, :-1])
        d4["adj_rho_lt_alpha_frac"] = float(np.nanmean(np.abs(d_rho) < np.abs(d_alpha)))
        d4["adj_ratio_median"] = float(np.nanmedian(ratio))
        d4["adj_order_preserved_frac"] = float(np.nanmean(order_pred))
        d4["rho_abs_median"] = float(np.nanmedian(np.abs(rho)))
        d4["rho_over_alpha_median"] = float(np.nanmedian(
            data["rho_over_alpha"][:, h, 1:]))
        per_head[f"head{h}"] = d4
    s4["per_head"] = per_head

    # 4.8 multi-head reconstruction (also valid for H=1)
    s5r = {}
    rec = load(out, "step5/recon_rel_err")
    prec = load(out, "step5/recon_proj_err")
    s5r["recon_rel_err_median_by_bin"] = {
        n: float(np.nanmedian(rec[:, a:b + 1])) for n, (a, b) in bins.items()}
    s5r["recon_rel_err_median"] = float(np.nanmedian(rec[:, 1:]))
    s5r["recon_proj_err_median_abs"] = float(np.nanmedian(np.abs(prec[:, 1:])))
    qtable(rec, "step5/recon_rel_err", rows)
    s4["multi_head_reconstruction"] = s5r
    update_summary(out, "step4", s4)

    # ================= Step 5.1 affine coordinate =================
    s5 = {}
    Y = load(out, "step5/Y_proj")
    if Y.ndim == 2:
        Y = Y[:, None, :]
    ref = __import__("torch").load(out / "ref_stats.pt", map_location="cpu",
                                   weights_only=False)
    w_nb = ref["w_nonbos_2"].numpy()
    w_b = ref["w_bos_2"].numpy()
    dw = w_nb - w_b
    per_head5 = {}
    for h in range(Y.shape[1]):
        c0 = float(w_nb[h] @ dw[h])
        dwn2 = float(dw[h] @ dw[h])
        yh = Y[:, h, 1:].reshape(-1)
        ah = alpha3[:, h, 1:].reshape(-1)
        pred_aff = c0 - ah * dwn2
        A = np.stack([ah, np.ones_like(ah)], 1)
        coef, *_ = np.linalg.lstsq(A, yh, rcond=None)
        resid = yh - pred_aff
        per_head5[f"head{h}"] = {
            "theoretical_slope": -dwn2, "theoretical_intercept": c0,
            "fitted_slope": float(coef[0]), "fitted_intercept": float(coef[1]),
            "r2_theoretical": float(1 - (resid ** 2).sum()
                                    / ((yh - yh.mean()) ** 2).sum()),
            "resid_over_dwnorm2_median": float(np.nanmedian(np.abs(resid)) / dwn2),
        }
    s5["affine_coordinate_per_head"] = per_head5
    update_summary(out, "step5_1", s5)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_position.parquet", index=False)
    print(f"wrote {len(df)} rows -> per_position.parquet; summary sections "
          f"step1/step3/step4/step5_1 updated")


if __name__ == "__main__":
    main()
