"""Extraction for the BOS-share spec (beta_spec.md): A0-A2, A4, B1, C1-C3, D1-D3.

Pass 1 (reference split): x_BOS at both layers, mean shares, the head's
single BOS-share weight B^h (median per-query OLS slope, i >= 16), w_BOS,
w_nonBOS, position-wise mean content values m_j, BOS-bias ratios r_h.
Pass 2 (calibration, evaluation): per-sequence, per-position, per-head
arrays. Everything reported comes from the evaluation split; calibration
arrays feed the isotonic decoder (A2).

Usage:
  python beta_extract.py --model 1h  --device cuda:1
  python beta_extract.py --model 12h --batch-size 4
  python beta_extract.py --model lm  --batch-size 32
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import MODELS, make_adapter, out_dir, spec_splits
from common import batch_from_offsets, head_weights, ov_images

BOS_RATIO_MIN = 10.0
SLOPE_I_MIN = 16


def batches(offsets, L, bs, device):
    for s in range(0, len(offsets), bs):
        yield s, batch_from_offsets(offsets[s:s + bs], L, device)


def share(x, xb, bias=None):
    """beta_j = <x_j, x_BOS>/||x_BOS||^2, optionally after removing the LN
    bias vector from both. x [B,T,d] f64, xb [d] f64."""
    if bias is not None:
        x = x - bias
        xb = xb - bias
    return (x @ xb) / (xb @ xb)


def per_query_ols(s, beta):
    """Per-query OLS of s_{i,j} on beta_j over j = 0..i, with intercept.

    s [B,H,T,T] f64 with -inf above the diagonal; beta [B,T] f64.
    Returns slope, R^2, cov, var_beta, var_s, each [B,H,T] (nan at i = 0).
    """
    T = s.shape[-1]
    M = torch.tril(torch.ones(T, T, device=s.device, dtype=torch.bool))
    s0 = torch.where(M, s, torch.zeros((), device=s.device, dtype=s.dtype))
    n = torch.arange(1, T + 1, device=s.device, dtype=s.dtype)
    mb = torch.cumsum(beta, -1) / n                       # [B,T]
    mbb = torch.cumsum(beta * beta, -1) / n
    ms = s0.sum(-1) / n                                   # [B,H,T]
    mss = (s0 * s0).sum(-1) / n
    msb = torch.einsum("bhij,bj->bhi", s0, beta) / n
    cov = msb - ms * mb[:, None]
    varb = (mbb - mb * mb)[:, None].expand_as(cov)
    vars_ = mss - ms * ms
    slope = cov / varb
    r2 = cov * cov / (vars_ * varb)
    for a in (slope, r2):
        a[..., 0] = float("nan")
    return slope, r2, cov, varb, vars_


@torch.no_grad()
def reference_pass(ad, offsets, L, bs, device):
    H, d = ad.H, ad.d
    hw1, hw2 = head_weights(ad.attn1), head_weights(ad.attn2)
    xb1 = xb2 = None
    max_bos_dev = 0.0
    sum_b2 = torch.zeros(L, device=device, dtype=torch.float64)
    sum_b1 = torch.zeros_like(sum_b2)
    sum_y = torch.zeros(H, L, d, device=device, dtype=torch.float64)
    sum_y1 = torch.zeros(L, d, device=device, dtype=torch.float64)
    sum_x2 = torch.zeros(d, device=device, dtype=torch.float64)
    sum_ab = torch.zeros(H, L, device=device, dtype=torch.float64)
    sum_nbk = torch.zeros(H, L, device=device, dtype=torch.float64)
    slopes = [[] for _ in range(H)]
    n = 0
    i_idx = torch.arange(L, device=device, dtype=torch.float64)
    for _, idx in batches(offsets, L, bs, device):
        p = ad.pre(idx)
        x1, x2 = p["x1"].double(), p["x2"].double()
        if xb2 is None:
            xb1, xb2 = x1[0, 0].clone(), x2[0, 0].clone()
        max_bos_dev = max(max_bos_dev, float((x2[:, 0] - xb2).abs().max()),
                          float((x1[:, 0] - xb1).abs().max()))
        b2, b1 = share(x2, xb2), share(x1, xb1)
        sum_b2 += b2.sum(0); sum_b1 += b1.sum(0)
        sum_y += ov_images(p["x2"], hw2).double().sum(0)
        sum_y1 += ov_images(p["x1"], hw1).double().sum((0, 1))
        sum_x2 += x2[:, 1:].sum((0, 1))
        a = p["attn2_w"].double()
        sum_ab += a[..., 0].sum(0)
        sum_nbk += ((1 - a[..., 0]) / i_idx.clamp(min=1)).sum(0)
        sl, *_ = per_query_ols(p["scores2"].double(), b2)
        for h in range(H):
            slopes[h].append(sl[:, h, SLOPE_I_MIN:].reshape(-1).cpu())
        n += idx.shape[0]
        del p, a, sl
    R = {"xb1": xb1, "xb2": xb2, "max_bos_dev": max_bos_dev,
         "beta2_ref": sum_b2 / n, "beta1_ref": sum_b1 / n,
         "xbar2_nonbos": sum_x2 / (n * (L - 1))}
    R["beta_bar"] = float(R["beta2_ref"][1:].mean())
    ybar = sum_y / n                                              # [H,T,d]
    v0 = torch.einsum("hed,d->he", hw2.W_v.double(), xb2)
    R["w_bos"] = torch.einsum("hde,he->hd", hw2.W_o.double(), v0)  # [H,d]
    R["w_nonbos"] = ybar[:, 1:].mean(1)
    R["m_j"] = ybar - R["beta2_ref"][None, :, None] * R["w_bos"][:, None, :]
    R["m_j"][:, 0] = 0.0
    R["mbar"] = R["m_j"][:, 1:].mean(1)
    v1 = torch.einsum("hed,d->he", hw1.W_v.double(), xb1)
    R["w1_bos"] = torch.einsum("hde,he->d", hw1.W_o.double(), v1)  # summed heads
    R["w1_nonbos"] = (sum_y1 / n)[1:].mean(0)
    ab = (sum_ab / n)[:, 1:].mean(1)
    nbk = (sum_nbk / n)[:, 1:].mean(1)
    R["r_h"] = ab / nbk
    R["B_h"] = torch.tensor([float(torch.cat(s).nanmedian()) for s in slopes],
                            dtype=torch.float64, device=device)
    R["bos_heads"] = [h for h in range(H) if float(R["r_h"][h]) > BOS_RATIO_MIN]
    return R


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def readout_axes(R, heads):
    """Per-head unit axes plus the aggregate over the BOS heads (last row)."""
    wb, wnb, mb = R["w_bos"], R["w_nonbos"], R["mbar"]
    agg = lambda v: v[heads].sum(0, keepdim=True)
    W = torch.cat([wb, agg(wb)])
    D = torch.cat([wnb - wb, agg(wnb - wb)])
    Mb = torch.cat([mb, agg(mb)])
    uW = unit(W)
    Mperp = Mb - (Mb * uW).sum(-1, keepdim=True) * uW
    return uW, unit(D), unit(Mperp), W.norm(dim=-1)


@torch.no_grad()
def eval_pass(ad, R, offsets, L, bs, device, check_identity=True):
    H = ad.H
    heads = R["bos_heads"] or [int(R["r_h"].argmax())]
    hw1, hw2 = head_weights(ad.attn1), head_weights(ad.attn2)
    uW, uD, uM, wnorm = readout_axes(R, heads)
    uW32, uD32, uM32 = uW.float(), uD.float(), uM.float()
    pm = torch.einsum("htd,hd->ht", R["m_j"], uW[:H])            # <m_j, w_hat>
    u1W = unit(R["w1_bos"]).float()
    u1D = unit(R["w1_nonbos"] - R["w1_bos"]).float()
    Bh = R["B_h"]
    xb2, xb1 = R["xb2"], R["xb1"]
    b2c = ad.ln2_bias.double() if ad.ln2_bias is not None else None
    b1c = ad.ln1_bias.double() if ad.ln1_bias is not None else None
    kb = torch.einsum("hed,d->he", hw2.W_k.double(), xb2)          # [H,dh]
    out = {k: [] for k in [
        "beta2", "beta2c", "beta1", "beta1c", "Bi", "slope", "r2", "ev",
        "alpha_bos", "alpha_hat_bos", "t", "z_w", "z_dw", "z_m",
        "d3_mean", "z1_w", "z1_dw", "alpha_ii", "frozen"]}
    corr = torch.zeros(H, 6, device=device, dtype=torch.float64)
    sub_err, sub_i = [], []
    ident = {}
    g = torch.Generator(device=device).manual_seed(0)
    T = L
    tri = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    tri_nb = tri.clone(); tri_nb[0] = False                       # queries i >= 1
    for bi, (_, idx) in enumerate(batches(offsets, L, bs, device)):
        p = ad.pre(idx)
        x2, x1 = p["x2"].double(), p["x1"].double()
        beta = share(x2, xb2)
        out["beta2"].append(beta)
        out["beta2c"].append(share(x2, xb2, b2c) if b2c is not None else beta)
        out["beta1"].append(share(x1, xb1))
        out["beta1c"].append(share(x1, xb1, b1c) if b1c is not None else share(x1, xb1))
        q = torch.einsum("btd,hed->bhte", x2, hw2.W_q.double()) + hw2.b_q.double()[None, :, None]
        Bi = torch.einsum("bhte,he->bht", q, kb) * hw2.scale
        out["Bi"].append(Bi)
        s = p["scores2"].double()
        if check_identity and bi == 0:
            xi = x2 - beta[..., None] * xb2
            k64 = torch.einsum("btd,hed->bhte", x2, hw2.W_k.double()) + hw2.b_k.double()[None, :, None]
            s64 = torch.einsum("bhie,bhje->bhij", q, k64) * hw2.scale
            E = torch.einsum("bhie,bhje->bhij", q,
                             torch.einsum("btd,hed->bhte", xi, hw2.W_k.double())) * hw2.scale
            bq = torch.einsum("bhte,he->bht", q, hw2.b_k.double()) * hw2.scale
            recon = beta[:, None, None, :] * Bi[..., None] + E + bq[..., None]
            ident = {"max_abs_err_fp64": float((s64 - recon)[..., tri].abs().max()),
                     "max_abs_s": float(s64[..., tri].abs().max()),
                     "max_abs_err_fp32_scores_vs_fp64": float((s - s64)[..., tri].abs().max())}
            del xi, k64, s64, E, recon
        slope, r2, cov, varb, vars_ = per_query_ols(s, beta)
        ev = 1 - (vars_ - 2 * Bi * cov + Bi * Bi * varb) / vars_
        ev[..., 0] = float("nan")
        out["slope"].append(slope); out["r2"].append(r2); out["ev"].append(ev)
        logA = torch.log_softmax(s, -1)                              # f64, -inf masked
        A = logA.exp()
        out["alpha_bos"].append(A[..., 0].clone())
        out["alpha_ii"].append(torch.diagonal(A, dim1=-2, dim2=-1).clone())
        t = torch.einsum("bhij,bj->bhi", A, beta)
        out["t"].append(t)
        # predicted attention from this sequence's own shares
        lg = Bh[None, :, None] * (beta[:, None, :] - 1)             # [B,H,T]
        logcs = torch.logcumsumexp(lg, -1)
        out["alpha_hat_bos"].append(torch.exp(-logcs))
        logAh = lg[:, :, None, :] - logcs[..., None]                 # [B,H,i,j]
        X, Y = logAh[..., tri_nb], logA[..., tri_nb]                 # [B,H,N]
        cnt = torch.full((H,), float(X.shape[0] * X.shape[-1]), device=device,
                         dtype=torch.float64)
        corr += torch.stack([cnt, X.sum(-1).sum(0),
                             Y.sum(-1).sum(0), (X * Y).sum(-1).sum(0),
                             (X * X).sum(-1).sum(0), (Y * Y).sum(-1).sum(0)], -1)
        pick = torch.randint(0, X.shape[-1], (4096,), generator=g, device=device)
        ii = tri_nb.nonzero()[:, 0][pick]
        sub_err.append(((X[..., pick].exp() - Y[..., pick].exp()).abs()
                        / Y[..., pick].exp()).float().cpu())
        sub_i.append(ii.cpu().expand(idx.shape[0], -1))
        del logAh, X, Y, logA
        # readouts (fp32 for the [B,H,T,d] tensors)
        y = ov_images(p["x2"], hw2)
        ho = torch.einsum("bhij,bhjd->bhid", p["attn2_w"].float(), y)
        agg = ho[:, heads].sum(1, keepdim=True)
        hoa = torch.cat([ho, agg], 1)                                # [B,H+1,T,d]
        out["z_w"].append(torch.einsum("bhtd,hd->bht", hoa, uW32).double())
        out["z_dw"].append(torch.einsum("bhtd,hd->bht", hoa, uD32).double())
        out["z_m"].append(torch.einsum("bhtd,hd->bht", hoa, uM32).double())
        out["d3_mean"].append(torch.einsum("bhij,hj->bhi", A, pm))
        y1 = ov_images(p["x1"], hw1)
        o1 = torch.einsum("bhij,bhjd->bid", p["attn1_w"].float(), y1)
        out["z1_w"].append((o1 @ u1W).double()); out["z1_dw"].append((o1 @ u1D).double())
        fr = p.get("pred")
        if fr is None and ad.probe is not None:
            feats, fr, _ = ad.finish(p["h1"], p["o2"])
        out["frozen"].append(fr.double() if fr is not None else torch.zeros_like(beta))
        del p, y, ho, hoa, y1, o1, A, s
    res = {k: torch.cat(v, 0).float().cpu().numpy() for k, v in out.items()}
    res["logcorr_sums"] = corr.cpu().numpy()
    res["sub_relerr"] = torch.cat(sub_err, 0).numpy()               # [n,H,4096]
    res["sub_i"] = torch.cat(sub_i, 0).numpy()
    res["wnorm"] = wnorm.cpu().numpy()
    return res, ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", action="store_true", help="untrained control (G2)")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    ad, L, run_id = make_adapter(args.model, dev, init_only=args.init)
    if args.init:
        run_id += "_init"
    od = out_dir(run_id)
    splits = spec_splits(L)
    if ad.kind == "lm":
        ad.fit_frozen(splits["calibration"], L, args.batch_size)
    R = reference_pass(ad, splits["reference"], L, args.batch_size, dev)
    print(f"[ref] {time.time() - t0:.0f}s  B_h={R['B_h'].cpu().numpy().round(2)} "
          f"r_h={R['r_h'].cpu().numpy().round(1)} bos_heads={R['bos_heads']} "
          f"beta_bar={R['beta_bar']:.4f} max_bos_dev={R['max_bos_dev']:.2e}")
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in R.items()},
               od / "ref.pt")
    meta = {"model": args.model, "run_id": run_id, "L": L,
            "splits": {k: len(v) for k, v in splits.items()},
            "adapter_meta": {k: v for k, v in ad.meta.items()
                             if isinstance(v, (str, int, float, bool, type(None)))},
            "bos_heads": R["bos_heads"], "B_h": R["B_h"].tolist(),
            "r_h": R["r_h"].tolist(), "beta_bar": R["beta_bar"],
            "max_bos_dev": R["max_bos_dev"], "n_head": ad.H,
            "cmd": " ".join(sys.argv)}
    for split in ["calibration", "evaluation"]:
        res, ident = eval_pass(ad, R, splits[split], L, args.batch_size, dev,
                               check_identity=(split == "evaluation"))
        np.savez(od / f"{split}.npz", **res)
        if ident:
            meta["c1_identity"] = ident
        print(f"[{split}] {time.time() - t0:.0f}s  identity={ident}")
    (od / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
