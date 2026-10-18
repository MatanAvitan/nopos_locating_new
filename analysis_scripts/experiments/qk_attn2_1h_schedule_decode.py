"""
Does the ATTN2-1H BOS coefficient follow a constant-gap reciprocal schedule,
and is position recoverable from it despite the 1/i compression?

Executes the diagnostic battery:
  (1) Recover the learned gap   g_i = log i + logit(alpha_i)   and  K_i = e^{g_i}
      = i * alpha_i / (1 - alpha_i).  Constant g_i  <=>  reciprocal model.
  (2) Decode position i from the scalar alpha_i three ways and compare
      test R^2 and late-window MAE:
        (a) linear      i ~ a*alpha + b
        (b) reciprocal  i ~ K*(1/alpha - 1)            (analytic inverse)
        (c) nonlinear   ridge on [alpha, alpha^2, 1/alpha, log alpha, logit]
  (3) Staged linear probes (closed-form ridge, train/test split by sample):
        - o_i^(2)  : Layer-2 attention output vector  (block2_attn)
        - h_i^(2)  : post-MLP, pre-head representation  ln_f(block2_out)
        - model's own pos_head (reference)
  (4) Adjacent-position SNR of the model's prediction:
        SNR_i = |E[p_{i+1}] - E[p_i]| / sqrt(0.5 Var(p_i) + 0.5 Var(p_{i+1}))
  (5) Projected noise along the decoding axis delta in attention-output space:
        per-position mean and std of <o_i, delta_hat>, and its adjacent SNR.

Output: results/qk_attn2_1h/schedule_decode.{png,pdf} + metrics json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "analysis_scripts" / "experiments"))
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

import qk_attn2_1h_analysis as Q  # reuse load_model / load_data / sample_tokens

LATE = (768, 1024)  # late window (i >= d = 768)


@torch.no_grad()
def collect(model, data, T, batch_size, n_batches):
    """Return per-(sample,pos) arrays: alpha, prediction (positions), o vec, h vec."""
    alpha, pred, O, H = [], [], [], []
    torch.manual_seed(0)
    Tblk = model.config.block_size
    for _ in range(n_batches):
        tok = Q.sample_tokens(data, T, batch_size)
        out, _ = model(tok, capture_taps=True)
        taps = model.get_all_taps()
        _, a2 = model.get_attention_weights()
        o = taps["block2_attn"]  # [B,T,d] Layer-2 attention output
        h = model.ln_f(taps["block2_out"])  # [B,T,d] head input
        alpha.append(a2[:, 0, :, 0].float().cpu().numpy())
        # head outputs normalised position in [0,1]; scale back to positions
        pred.append((out.squeeze(-1).float().cpu().numpy()) * Tblk)
        O.append(o.float().cpu().numpy())
        H.append(h.float().cpu().numpy())
    return (
        np.concatenate(alpha, 0),
        np.concatenate(pred, 0),
        np.concatenate(O, 0),
        np.concatenate(H, 0),
    )


def ridge_probe(Xtr, ytr, Xte, yte, lam=1.0):
    """Closed-form ridge; return test R^2, test MAE, and predictions on test."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    yp = Xte @ w
    ss_res = ((yte - yp) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return float(r2), float(np.abs(yte - yp).mean()), yp


def late_mae(pos_te, y_te, y_pred):
    m = (pos_te >= LATE[0]) & (pos_te < LATE[1])
    if m.sum() == 0:
        return float("nan")
    return float(np.abs(y_te[m] - y_pred[m]).mean())


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=Q.DEFAULT_CKPT)
    p.add_argument("--data_dir", default=Q.DEFAULT_DATA)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_batches", type=int, default=16)
    p.add_argument("--save_dir", default="results/qk_attn2_1h")
    args = p.parse_args()

    model = Q.load_model(ROOT_DIR / args.ckpt)
    data = Q.load_data(ROOT_DIR / args.data_dir)
    T = min(args.T, model.config.block_size)

    alpha, pred, O, H = collect(model, data, T, args.batch_size, args.n_batches)
    N = alpha.shape[0]
    pos_row = np.arange(T)

    # ---------- (1) schedule g_i and K_i ----------
    a_mean = alpha.mean(0)  # population schedule
    eps = 1e-6
    ac = np.clip(a_mean, eps, 1 - eps)
    valid = pos_row >= 1
    g_i = np.log(pos_row[valid]) + np.log(ac[valid] / (1 - ac[valid]))  # = logit + log i
    K_i = pos_row[valid] * ac[valid] / (1 - ac[valid])
    pv = pos_row[valid]

    # ---------- build flat dataset, split by sample ----------
    ntr = N // 2
    sl_tr, sl_te = slice(0, ntr), slice(ntr, N)
    def flat(x, s):  # [N,T,...] -> [(n*T'),...] excluding pos 0
        return x[s][:, 1:].reshape(-1, *x.shape[2:])
    pos_tr = np.tile(pos_row[1:], ntr).astype(float)
    pos_te = np.tile(pos_row[1:], N - ntr).astype(float)
    a_tr, a_te = flat(alpha, sl_tr), flat(alpha, sl_te)
    o_tr, o_te = flat(O, sl_tr), flat(O, sl_te)
    h_tr, h_te = flat(H, sl_tr), flat(H, sl_te)
    ac_tr = np.clip(a_tr, eps, 1 - eps); ac_te = np.clip(a_te, eps, 1 - eps)

    decode = {}
    # (a) linear in alpha
    r2, mae, yp = ridge_probe(a_tr[:, None], pos_tr, a_te[:, None], pos_te)
    decode["linear_alpha"] = {"r2": r2, "mae": mae, "late_mae": late_mae(pos_te, pos_te, yp)}
    # (b) reciprocal feature (1/alpha - 1)
    recip = lambda ac: (1.0 / ac - 1.0)[:, None]
    r2, mae, yp = ridge_probe(recip(ac_tr), pos_tr, recip(ac_te), pos_te)
    decode["reciprocal_alpha"] = {"r2": r2, "mae": mae, "late_mae": late_mae(pos_te, pos_te, yp)}
    # (c) nonlinear poly/feature probe on alpha
    feat = lambda ac: np.stack([ac, ac**2, 1.0 / ac, np.log(ac), np.log(ac / (1 - ac))], 1)
    r2, mae, yp = ridge_probe(feat(ac_tr), pos_tr, feat(ac_te), pos_te)
    decode["nonlinear_alpha"] = {"r2": r2, "mae": mae, "late_mae": late_mae(pos_te, pos_te, yp)}
    # (4-stage) linear probe on attention output o
    r2, mae, yp_o = ridge_probe(o_tr, pos_tr, o_te, pos_te, lam=10.0)
    decode["linear_o_attn"] = {"r2": r2, "mae": mae, "late_mae": late_mae(pos_te, pos_te, yp_o)}
    # (5-stage) linear probe on post-MLP head input h
    r2, mae, yp_h = ridge_probe(h_tr, pos_tr, h_te, pos_te, lam=10.0)
    decode["linear_h_postmlp"] = {"r2": r2, "mae": mae, "late_mae": late_mae(pos_te, pos_te, yp_h)}

    # model's own head, evaluated on test rows
    pred_te = flat(pred, sl_te)
    ss_res = ((pos_te - pred_te) ** 2).sum(); ss_tot = ((pos_te - pos_te.mean()) ** 2).sum()
    decode["model_head"] = {
        "r2": float(1 - ss_res / ss_tot),
        "mae": float(np.abs(pos_te - pred_te).mean()),
        "late_mae": late_mae(pos_te, pos_te, pred_te),
    }

    # ---------- (4) adjacent SNR of model prediction ----------
    pmean = pred.mean(0); pvar = pred.var(0)
    sep = np.abs(np.diff(pmean))
    pooled = np.sqrt(0.5 * (pvar[:-1] + pvar[1:]) + 1e-12)
    snr = sep / pooled  # length T-1, index i -> between i and i+1

    # ---------- (5) decoding-axis projection in o-space ----------
    # delta axis = least-squares direction mapping o -> position (use probe weights)
    # reuse: fit direction via normal equations on standardized o (train)
    muo = o_tr.mean(0); sdo = o_tr.std(0) + 1e-8
    Xo = (o_tr - muo) / sdo
    wdir = np.linalg.solve(Xo.T @ Xo + 10.0 * np.eye(Xo.shape[1]), Xo.T @ pos_tr)
    proj_all = ((O - muo) / sdo) @ wdir  # [N,T]
    proj_mean = proj_all.mean(0); proj_std = proj_all.std(0)
    proj_sep = np.abs(np.diff(proj_mean))
    proj_pooled = np.sqrt(0.5 * (proj_std[:-1] ** 2 + proj_std[1:] ** 2) + 1e-12)
    proj_snr = proj_sep / proj_pooled

    # ---------- figure ----------
    save_dir = ROOT_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    ax[0, 0].plot(pv, g_i, color="C0")
    ax[0, 0].axhline(np.median(g_i[pv >= 50]), color="k", ls="--", lw=0.8)
    ax[0, 0].set_title(r"Recovered schedule $g_i=\log i+\mathrm{logit}(\alpha_i)$")
    ax[0, 0].set_xlabel("position $i$"); ax[0, 0].set_ylabel(r"$g_i$ (= $\log K_i$)")
    ax[0, 0].text(0.4, 0.1, "flat ⇒ constant-gap reciprocal", transform=ax[0, 0].transAxes, fontsize=9)

    ax[0, 1].plot(pv, K_i, color="C1")
    ax[0, 1].set_title(r"$K_i=i\,\alpha_i/(1-\alpha_i)$ (should be $\approx e^\Delta$)")
    ax[0, 1].set_xlabel("position $i$"); ax[0, 1].set_ylabel(r"$K_i$")

    names = ["linear_alpha", "reciprocal_alpha", "nonlinear_alpha",
             "linear_o_attn", "linear_h_postmlp", "model_head"]
    late = [decode[n]["late_mae"] for n in names]
    allm = [decode[n]["mae"] for n in names]
    xb = np.arange(len(names))
    ax[0, 2].bar(xb - 0.2, allm, 0.4, label="all-pos MAE")
    ax[0, 2].bar(xb + 0.2, late, 0.4, label=f"late MAE [{LATE[0]},{LATE[1]})")
    ax[0, 2].set_xticks(xb); ax[0, 2].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax[0, 2].set_ylabel("MAE (positions)"); ax[0, 2].legend(fontsize=8)
    ax[0, 2].set_title("Decoding stages: position MAE")

    ax[1, 0].scatter(pos_te[::57], yp[::57], s=3, alpha=0.3, color="C2")
    ax[1, 0].plot([0, T], [0, T], "k--", lw=0.7)
    ax[1, 0].set_title(fr"nonlinear($\alpha$) decode  ($R^2$={decode['nonlinear_alpha']['r2']:.3f})")
    ax[1, 0].set_xlabel("true $i$"); ax[1, 0].set_ylabel(r"$\hat i$")

    ax[1, 1].plot(pos_row[:-1], snr, color="C3", lw=0.8)
    ax[1, 1].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[1, 1].set_title("Adjacent-position SNR of model prediction")
    ax[1, 1].set_xlabel("position $i$"); ax[1, 1].set_ylabel(r"SNR$_i$ ($i\to i{+}1$)")
    ax[1, 1].set_yscale("log")

    ax[1, 2].plot(pos_row[:-1], proj_snr, color="C4", lw=0.8)
    ax[1, 2].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[1, 2].set_title(r"Adjacent SNR along $o$-decoding axis")
    ax[1, 2].set_xlabel("position $i$"); ax[1, 2].set_ylabel("SNR$_i$")
    ax[1, 2].set_yscale("log")

    fig.suptitle("ATTN2-1H: schedule recovery, reciprocal decoding, and adjacent-position SNR", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = save_dir / "schedule_decode.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "schedule_decode.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- numeric summary ----------
    def at(arr, idxs):
        return {int(i): float(arr[i]) for i in idxs if i < len(arr)}
    summary = {
        "schedule_g_i": {
            "g_mean_pos50plus": float(np.mean(g_i[pv >= 50])),
            "g_std_pos50plus": float(np.std(g_i[pv >= 50])),
            "g_mean_early_2_16": float(np.mean(g_i[(pv >= 2) & (pv < 16)])),
            "g_mean_late_768plus": float(np.mean(g_i[pv >= 768])),
            "K_i_at": at(np.concatenate([[np.nan], K_i]) if pv[0] == 1 else K_i, [16, 128, 512, 1000]),
        },
        "decode": decode,
        "adjacent_snr_model": at(snr, [16, 64, 128, 256, 512, 768, 1000]),
        "adjacent_snr_o_axis": at(proj_snr, [16, 64, 128, 256, 512, 768, 1000]),
        "n_samples": int(N),
        "T": int(T),
    }
    with open(save_dir / "schedule_decode_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", png)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
