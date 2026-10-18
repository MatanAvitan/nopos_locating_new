"""Check the general form of the mechanism: Layer-2 attention = softmax_j(B beta_j)
with beta_j each key's own BOS projection (graded or two-level), and the
output = attention-weighted value, with values linear in beta.

Per BOS head, with beta from two definitions
  centered  beta_j = <x_j - mu, x_BOS - mu> / |x_BOS - mu|^2
  raw       beta_j = <x_j, x_BOS> / |x_BOS|^2
(x = Layer-2 input, mu = mean non-BOS Layer-2 input, reference split):
  B_ml      maximum-likelihood B of the measured rows (calibration split)
  B_med     median per-query OLS slope of the scores on beta
  attention prediction alpha_hat_ij = softmax_j(B_ml beta_j), each sequence's own beta:
            median relative error of the BOS weight by bin, mean TV distance
            between predicted and measured rows (0 = identical, 1 = disjoint),
            correlation of log alpha over all entries
  values    v_j = projection of the head's value on its BOS-value direction;
            R^2 of v on beta (linear fit), over all keys
  readout   z_hat_i = sum_j alpha_hat_ij (c + lambda beta_j), built from beta
            only; R^2 against the measured output projection z_i, and the
            isotonic decoding error of z_hat and of z (relMAE by bin)

Usage: python softmax_beta_check.py --model lmbos_s42 --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_analyze import isotonic_decode
from beta_common import MODELS, RESULTS_ROOT, bins_for, err_by_bin, make_adapter, spec_splits
from beta_extract import per_query_ols
from common import batch_from_offsets, head_weights, ov_images

N_EVAL, N_CAL = 400, 200


@torch.no_grad()
def collect(ad, L, heads, offs, dev, bs, ref):
    hw2 = head_weights(ad.attn2)
    out = {k: [] for k in ["b_cen", "b_raw", "A", "v", "z", "slope_cen", "slope_raw"]}
    for s in range(0, len(offs), bs):
        idx = batch_from_offsets(offs[s:s + bs], L, dev)
        p = ad.pre(idx)
        x2 = p["x2"].double()
        bc = ((x2 - ref["mu"]) @ ref["dc"]) / (ref["dc"] @ ref["dc"])
        br = (x2 @ ref["xb"]) / (ref["xb"] @ ref["xb"])
        A = p["attn2_w"].double()[:, heads]
        y = ov_images(p["x2"], hw2).double()[:, heads]
        v = torch.einsum("bhtd,hd->bht", y, ref["uv"])
        out["b_cen"].append(bc.cpu()); out["b_raw"].append(br.cpu())
        out["A"].append(A.float().cpu()); out["v"].append(v.cpu())
        out["z"].append(torch.einsum("bhij,bhj->bhi", A, v).cpu())
        S = p["scores2"].double()[:, heads]
        out["slope_cen"].append(per_query_ols(S, bc)[0].cpu()); out["slope_raw"].append(per_query_ols(S, br)[0].cpu())
        del p, A, y, S
    return {k: torch.cat(v_).numpy() for k, v_ in out.items()}


def ml_B(A, beta, grid):
    """B maximising sum_{i>=1} sum_j alpha_ij log softmax_j(B beta_j)."""
    t = np.einsum("bij,bj->bi", A, beta)                                     # sum_j alpha_ij beta_j
    best, bestB = -np.inf, None
    for B in grid:
        lcs = np.logaddexp.accumulate(B * beta, axis=1)
        ll = np.mean((B * t - lcs)[:, 1:])
        if ll > best:
            best, bestB = ll, B
    return float(bestB)


def predict_rows(B, beta):
    lg = B * beta
    lcs = np.logaddexp.accumulate(lg, axis=1)                                 # [n,T]
    T = beta.shape[1]
    logA = lg[:, None, :] - lcs[:, :, None]                                   # [n,i,j]
    mask = np.tril(np.ones((T, T), bool))
    return np.where(mask[None], np.exp(logA), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-T", type=int, default=256, help="positions used (attention rows are T x T per sequence)")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    meta = json.loads((RESULTS_ROOT / run_id / "beta" / "meta.json").read_text())
    heads = meta["bos_heads"] or [int(np.argmax(meta["r_h"]))]
    sp = spec_splits(L); bins = bins_for(L)
    hw2 = head_weights(ad.attn2)
    with torch.no_grad():
        xs = None; n = 0; xb = None
        for s in range(0, 160, args.batch_size):
            x2 = ad.pre(batch_from_offsets(sp["reference"][s:s + args.batch_size], L, dev))["x2"].double()
            xb = x2[0, 0].clone() if xb is None else xb
            xs = x2[:, 1:].sum((0, 1)) if xs is None else xs + x2[:, 1:].sum((0, 1)); n += x2.shape[0] * (L - 1)
        mu = xs / n
        M = torch.einsum("hde,hef->hdf", hw2.W_o.double(), hw2.W_v.double())[heads]
        wb = torch.einsum("hdf,f->hd", M, xb)
        ref = {"mu": mu, "dc": xb - mu, "xb": xb, "uv": wb / wb.norm(dim=-1, keepdim=True)}
    E = collect(ad, L, heads, sp["evaluation"][:N_EVAL], dev, args.batch_size, ref)
    C = collect(ad, L, heads, sp["calibration"][:N_CAL], dev, args.batch_size, ref)
    T = min(L, args.max_T)
    bsub = {b: (lo, min(hi, T - 1)) for b, (lo, hi) in bins.items() if lo < T}
    S = {"model": args.model, "bos_heads": heads, "T_used": T, "heads": {}}
    for k, h in enumerate(heads):
        res = {}
        for name in ["b_cen", "b_raw"]:
            bc, be = C[name][:, :T], E[name][:, :T]
            Ac, Ae = C["A"][:, k, :T, :T].astype(np.float64), E["A"][:, k, :T, :T].astype(np.float64)
            grid = np.concatenate([np.linspace(-20, 20, 801)])
            Bml = ml_B(Ac, bc, grid)
            Bmed = float(np.nanmedian(C["slope_" + ("cen" if name == "b_cen" else "raw")][:, k, 16:T]))
            Ah = predict_rows(Bml, be)
            ab, abh = Ae[:, :, 0], Ah[:, :, 0]
            tv = 0.5 * np.abs(Ah - Ae).sum(-1)[:, 1:].mean()
            msk = np.tril(np.ones((T, T), bool)); msk[0] = False
            la, lah = np.log(np.clip(Ae[:, msk], 1e-12, None)), np.log(np.clip(Ah[:, msk], 1e-12, None))
            sel = np.random.default_rng(0).choice(la.size, min(la.size, 2_000_000), replace=False)
            corr = float(np.corrcoef(la.ravel()[sel], lah.ravel()[sel])[0, 1])
            # value side and readout from beta alone
            v = E["v"][:, k, :T]; vc_ = C["v"][:, k, :T]
            lam, c0 = np.polyfit(bc[:, 1:].ravel(), vc_[:, 1:].ravel(), 1)
            vfit_r2 = float(1 - np.var(v - (c0 + lam * be)) / np.var(v))
            zh = np.einsum("bij,bj->bi", Ah, c0 + lam * be)
            zhc = np.einsum("bij,bj->bi", predict_rows(Bml, bc), c0 + lam * bc)
            z, zc = E["z"][:, k, :T], C["z"][:, k, :T]
            Lz = T
            ez = err_by_bin(np.pad(isotonic_decode(zc, z, Lz), ((0, 0), (0, 0))), Lz, bsub)
            ezh = err_by_bin(isotonic_decode(zhc, zh, Lz), Lz, bsub)
            res[name] = {"B_ml": Bml, "B_median_slope": Bmed,
                         "bos_weight_relerr_by_bin": {b: float(np.median(np.abs(abh - ab)[:, lo:hi + 1] / ab[:, lo:hi + 1])) for b, (lo, hi) in bsub.items()},
                         "mean_tv_rows": float(tv), "corr_log_alpha_all_entries": corr,
                         "value_linear_in_beta_r2": vfit_r2,
                         "readout_from_beta_r2": float(1 - np.var(z[:, 1:] - zh[:, 1:]) / np.var(z[:, 1:])),
                         "decode_relmae_measured_z": {b: ez[f"relmae_{b}"] for b in bsub},
                         "decode_relmae_z_from_beta_only": {b: ezh[f"relmae_{b}"] for b in bsub}}
        S["heads"][f"h{h}"] = res
    od = RESULTS_ROOT / "story_evidence" / args.model; od.mkdir(parents=True, exist_ok=True)
    (od / "softmax_beta_check.json").write_text(json.dumps(S, indent=2))
    print(f"== {args.model} heads {heads} (positions 0..{T - 1})")
    for hk, res in S["heads"].items():
        for name, r in res.items():
            print(f"  {hk} {name}: B_ml {r['B_ml']:.2f} (median slope {r['B_median_slope']:.2f}) | BOS-weight relerr {({b: round(x, 3) for b, x in r['bos_weight_relerr_by_bin'].items()})} "
                  f"| row TV {r['mean_tv_rows']:.3f} | corr log alpha {r['corr_log_alpha_all_entries']:.3f} | value~beta R2 {r['value_linear_in_beta_r2']:.2f} "
                  f"| readout-from-beta R2 {r['readout_from_beta_r2']:.2f} | decode z {({b: round(x, 3) for b, x in r['decode_relmae_measured_z'].items()})} "
                  f"vs from beta {({b: round(x, 3) for b, x in r['decode_relmae_z_from_beta_only'].items()})}")


if __name__ == "__main__":
    main()
