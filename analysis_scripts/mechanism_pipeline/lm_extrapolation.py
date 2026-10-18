"""P6: length extrapolation of the BOS-trained NoPE LM (lm6bos).

The L=128 checkpoint is evaluated at longer contexts. Because the model is
causal and forward_blocks builds its mask from the actual sequence length,
positions < 128 of a long sequence are distributed exactly as in training and
every position >= 128 is genuinely unseen.

Per eval length L in --lengths, fit on the calibration split:
  frozen probe : per-block ridge on residuals at positions 1..127 only
  oracle probe : per-block ridge on all positions 1..L-1
  analytic     : log alpha = c + gamma*log i fit on 16..64 for block-1 head 0,
                 inverted as i_hat = exp((log alpha - c)/gamma)
then evaluate on the eval split with per-bin MAE/relMAE and per-sequence
Spearman restricted to unseen positions, plus the anchor-head alpha schedule
(gamma trained vs unseen, per-seq Spearman, per-position CV) and LM
cross-entropy per bin.

Usage:
  python lm_extrapolation.py --lengths 128,256,512,1024,2048
Outputs results/mechanism/lm6bos_extrap_L{L}/ (arrays + summary.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, VAL_BIN, build_splits, git_commit, run_dir,
                    sha256_file, update_summary)
from lm_mechanism import Ridge, batch_from_offsets, forward_blocks, load_lm
from stats import loglog_slope

TRAIN_L = 128
FIT_LO, FIT_HI = 16, 64          # fit_range(128), the trained-range fit
ANCHOR_BLOCK, ANCHOR_HEAD = 1, 0  # per lm6bos summary.json (P4)


def targets_from_offsets(offsets, L, device):
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    y = np.stack([data[off: off + L].astype(np.int64) for off in offsets])
    return torch.from_numpy(y).to(device)


def bins_for(L):
    out = {"trained_1_127": (1, min(127, L - 1))}
    lo = 128
    while lo < L:
        hi = min(2 * lo - 1, L - 1)
        out[f"unseen_{lo}_{hi}"] = (lo, hi)
        lo = 2 * lo
    return out


def binned(preds, L):
    """preds [n, L] -> dict of R2/MAE/relMAE per bin + unseen Spearman."""
    t = np.arange(L)[None, :].astype(np.float64)
    err = np.abs(preds - t)
    res = {}
    for name, (a, b) in bins_for(L).items():
        e = err[:, a:b + 1]
        r = e / t[:, a:b + 1]
        res[name] = {"MAE": float(np.nanmean(e)),
                     "relMAE_mean": float(np.nanmean(r)),
                     "relMAE_med": float(np.nanmedian(r))}
    tt = t.repeat(len(preds), 0)
    fin = np.isfinite(preds)
    ssr = ((preds - tt)[fin] ** 2).sum()
    sst = ((tt - tt[fin].mean())[fin] ** 2).sum()
    res["r2_all"] = float(1 - ssr / sst)
    if L > TRAIN_L:
        rho = [spearmanr(preds[b, TRAIN_L:], np.arange(TRAIN_L, L)).statistic
               for b in range(min(len(preds), 200))]
        res["unseen_spearman_median"] = float(np.nanmedian(rho))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nanoGPT/out-lm-6layer-fulltrain-bos/ckpt.pt")
    ap.add_argument("--lengths", default="128,256,512,1024,2048")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-cal", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=256)
    args = ap.parse_args()
    device = args.device
    ckpt_path = str(REPO_ROOT / args.ckpt)
    model, cfg, ck = load_lm(ckpt_path, device)
    n_layer, d = cfg.n_layer, cfg.n_embd

    for L in [int(x) for x in args.lengths.split(",")]:
        print(f"===== L={L} =====")
        out = run_dir(f"lm6bos_extrap_L{L}")
        bs = min(32, max(2, 8192 // L))
        splits = build_splits(L, seed=1234, sizes={
            "reference": 8, "calibration": args.n_cal, "evaluation": args.n_eval})
        np.savez(out / "sequence_ids.npz", **splits)
        (out / "config.json").write_text(json.dumps({
            "model": "lm6bos", "checkpoint_path": ckpt_path,
            "checkpoint_hash": sha256_file(ckpt_path),
            "iter_num": ck.get("iter_num"), "context_length": L,
            "trained_context_length": TRAIN_L, "use_bos": True,
            "splits": {k: len(v) for k, v in splits.items()},
            "batch_size": bs, "git_commit": git_commit(),
            "run_command": " ".join(sys.argv)}, indent=2))

        def iter_split(name):
            offs = splits[name]
            for s in range(0, len(offs), bs):
                o = offs[s:s + bs]
                yield o, batch_from_offsets(o, L, device, use_bos=True)

        # ---- calibration: probes + anchor-head alpha schedule ----------
        print("calibration pass...")
        frozen = [Ridge(d, device) for _ in range(n_layer)]
        oracle = [Ridge(d, device) for _ in range(n_layer)]
        tgt = torch.arange(L, dtype=torch.float32, device=device)
        alpha_sum = torch.zeros(L, device=device, dtype=torch.float64)
        n_cal_seq = 0
        for _, idx in iter_split("calibration"):
            caps = forward_blocks(model, idx)
            B = idx.shape[0]
            for l, c in enumerate(caps):
                frozen[l].add(c["resid_out"][:, 1:TRAIN_L],
                              tgt[1:TRAIN_L].expand(B, TRAIN_L - 1))
                oracle[l].add(c["resid_out"][:, 1:], tgt[1:].expand(B, L - 1))
            a = caps[ANCHOR_BLOCK]["weights"][:, ANCHOR_HEAD, :, 0].double()
            alpha_sum += a.sum(0)
            n_cal_seq += B
            del caps
        for l in range(n_layer):
            frozen[l].solve()
            oracle[l].solve()
        alpha_cal = (alpha_sum / n_cal_seq).cpu().numpy()
        i_fit = np.arange(FIT_LO, FIT_HI + 1)
        gamma_tr, c_tr = loglog_slope(i_fit, alpha_cal[FIT_LO:FIT_HI + 1])
        print(f"anchor fit on trained range: gamma={gamma_tr:.4f} c={c_tr:.4f}")

        # ---- evaluation ------------------------------------------------
        print("evaluation pass...")
        acc = {f"L{l}__{k}": [] for l in range(n_layer)
               for k in ("frozen_pred", "oracle_pred")}
        acc["alpha_anchor"] = []
        acc["analytic_pred"] = []
        ce_sum = torch.zeros(L, device=device, dtype=torch.float64)
        n_ev_seq = 0
        for offs, idx in iter_split("evaluation"):
            caps = forward_blocks(model, idx)
            for l, c in enumerate(caps):
                acc[f"L{l}__frozen_pred"].append(
                    frozen[l].predict(c["resid_out"]).float().cpu().numpy())
                acc[f"L{l}__oracle_pred"].append(
                    oracle[l].predict(c["resid_out"]).float().cpu().numpy())
            a = caps[ANCHOR_BLOCK]["weights"][:, ANCHOR_HEAD, :, 0].double()
            acc["alpha_anchor"].append(a.float().cpu().numpy())
            ihat = torch.exp((torch.log(a.clamp_min(1e-30)) - c_tr) / gamma_tr)
            acc["analytic_pred"].append(ihat.float().cpu().numpy())
            # LM cross-entropy per position (chunked over positions)
            x = model.transformer.ln_f(caps[-1]["resid_out"])
            y = targets_from_offsets(offs, L, device)
            for t0 in range(0, L, 256):
                logits = model.lm_head(x[:, t0:t0 + 256])
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y[:, t0:t0 + 256].reshape(-1), reduction="none")
                ce_sum[t0:t0 + 256] += ce.reshape(len(idx), -1).double().sum(0)
            n_ev_seq += len(idx)
            del caps
        arrs = {k: np.concatenate(v, 0) for k, v in acc.items()}
        for k, v in arrs.items():
            np.save(out / f"{k}.npy", v)
        np.save(out / "alpha_cal_meancurve.npy", alpha_cal)
        ce_pos = (ce_sum / n_ev_seq).cpu().numpy()
        np.save(out / "ce_per_position.npy", ce_pos)

        # ---- summaries -------------------------------------------------
        summary = {"gamma_trained_fit": float(gamma_tr),
                   "intercept_trained_fit": float(c_tr),
                   "bins": {k: list(v) for k, v in bins_for(L).items()},
                   "probes": {}, "anchor": {}, "lm_ce_per_bin": {}}
        al = arrs["alpha_anchor"]
        mean_curve = np.nanmean(al, 0)
        if L > TRAIN_L:
            iu = np.arange(160, L) if L > 160 else np.arange(TRAIN_L, L)
            gamma_un, _ = loglog_slope(iu, mean_curve[iu])
            rho_un = [spearmanr(al[b, TRAIN_L:], np.arange(TRAIN_L, L)).statistic
                      for b in range(min(len(al), 200))]
            cv_un = np.nanstd(al[:, TRAIN_L:], 0) / \
                np.clip(np.nanmean(al[:, TRAIN_L:], 0), 1e-30, None)
            summary["anchor"] = {
                "gamma_unseen": float(gamma_un),
                "gamma_gap": float(abs(gamma_un - gamma_tr)),
                "unseen_per_seq_spearman_median": float(np.nanmedian(rho_un)),
                "unseen_alpha_cv_median": float(np.nanmedian(cv_un)),
                "alpha_at": {str(i): float(mean_curve[i])
                             for i in (64, 127, 255, 511, 1023, 2047)
                             if i < L}}
        summary["probes"]["analytic_alpha"] = binned(arrs["analytic_pred"], L)
        for l in range(n_layer):
            for kind in ("frozen", "oracle"):
                summary["probes"][f"L{l}_{kind}"] = binned(
                    arrs[f"L{l}__{kind}_pred"], L)
        for name, (a, b) in bins_for(L).items():
            summary["lm_ce_per_bin"][name] = float(np.nanmean(ce_pos[a:b + 1]))
        update_summary(out, "lm_extrapolation", summary)
        print(json.dumps({"anchor": summary["anchor"],
                          "analytic": summary["probes"]["analytic_alpha"],
                          "L1_frozen": summary["probes"]["L1_frozen"],
                          "L1_oracle": summary["probes"]["L1_oracle"],
                          "lm_ce": summary["lm_ce_per_bin"]}, indent=2))


if __name__ == "__main__":
    main()
