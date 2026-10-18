"""Condition K (keys read the BOS share) and condition V (values read it too).

For a Layer-2 head h with OV map M (= W_O W_V) and readout unit vector u:
  value read direction   r_V = M^T u        (v_j = <x_j, r_V> is what lands on u)
  key read direction     r_K = W_K^T q_bar  (q_bar = mean query over tokens)
  BOS direction          d   = x_BOS - mu   (centered Layer-2 input of BOS)
and the BOS share of key j is beta_j = <x_j - mu, d>/|d|^2.

1. cos(r_V, d), cos(r_K, d), cos(r_V, r_K).
2. v_j = c + lambda beta_j + eps_j: fit on calibration; report R^2 over all
   keys and over the tail (j >= 16), and sd(eps) in the tail / |lambda|.
3. Output noise from the values: sd over sequences of sum_j alpha_ij eps_j,
   against the neighbor gap |lambda (t_{i+1} - t_i)| of the mean curve
   (t_i = sum_j alpha_ij beta_j).
4. Same regression in Layer 1 (x1, summed OV, u = unit Layer-1 BOS value).
5. Causal, BOS heads only: "values": v_j -> c + lambda beta_j exactly (the
   value vector loses eps_j along u, nothing else changes); "keys": attention
   -> softmax_j(B beta_j) exactly; "both". Late relMAE of a refit probe on the
   final features and of the isotonic decoder on z_i = <head outputs, u>.

Usage: python beta_kv_conditions.py --model 1h --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_analyze import isotonic_decode
from beta_common import (MODELS, Ridge, bins_for, err_by_bin, make_adapter,
                         out_dir, smooth_deriv, spec_splits)
from beta_extract import per_query_ols
from common import attn_bias_vector, batch_from_offsets, head_weights, ov_images

N_CAL, N_EVAL = 320, 800


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, bs = args.device, args.batch_size
    ad, L, run_id = make_adapter(args.model, dev)
    od = out_dir(run_id, "beta_v2")
    sp = spec_splits(L); bins = bins_for(L)
    hw2, hw1 = head_weights(ad.attn2), head_weights(ad.attn1)
    H = ad.H

    def batches(name, n):
        offs = sp[name][:n]
        for s in range(0, len(offs), bs):
            yield batch_from_offsets(offs[s:s + bs], L, dev)

    # ---- reference: centered BOS directions, BOS heads, B, q_bar ------------
    s2 = torch.zeros(ad.d, device=dev, dtype=torch.float64); s1 = torch.zeros_like(s2)
    qs = torch.zeros(H, hw2.W_q.shape[1], device=dev, dtype=torch.float64)
    ab = torch.zeros(H, device=dev, dtype=torch.float64); nb = torch.zeros_like(ab)
    n = 0; xb2 = xb1 = None
    for idx in batches("reference", 320):
        p = ad.pre(idx)
        x2, x1 = p["x2"].double(), p["x1"].double()
        if xb2 is None:
            xb2, xb1 = x2[0, 0].clone(), x1[0, 0].clone()
        s2 += x2[:, 1:].sum((0, 1)); s1 += x1[:, 1:].sum((0, 1)); n += idx.shape[0] * (L - 1)
        q = torch.einsum("btd,hed->bhte", x2, hw2.W_q.double()) + hw2.b_q.double()[None, :, None]
        qs += q[:, :, 1:].sum((0, 2))
        A = p["attn2_w"].double()
        ab += A[..., 1:, 0].sum((0, 2)); nb += ((1 - A[..., 1:, 0]) / torch.arange(1, L, device=dev)).sum((0, 2))
    mu2, mu1 = s2 / n, s1 / n
    d2, d1 = xb2 - mu2, xb1 - mu1
    qbar = qs / n
    r_h = (ab / nb).cpu().numpy()
    heads = [h for h in range(H) if r_h[h] > 10] or [int(np.argmax(r_h))]
    M = torch.einsum("hde,hef->hdf", hw2.W_o.double(), hw2.W_v.double())       # [H,d,d] OV per head
    wbos = torch.einsum("hdf,f->hd", M, xb2)
    wnb = None
    share = lambda x, mu, d: ((x - mu) @ d) / (d @ d)

    # B per head: median per-query slope of scores on the centered share
    Bh = {}
    sl_all = {h: [] for h in heads}
    for idx in batches("reference", 64):
        p = ad.pre(idx)
        beta = share(p["x2"].double(), mu2, d2)
        sl, *_ = per_query_ols(p["scores2"].double(), beta)
        for h in heads:
            sl_all[h].append(sl[:, h, 16:].reshape(-1).cpu())
    for h in heads:
        Bh[h] = float(torch.cat(sl_all[h]).nanmedian())

    S = {"model": args.model, "bos_heads": heads, "B": Bh, "heads": {}}
    for uname in ["w_bos", "dw"]:
        pass
    # ---- 1. read directions + 2. value split (fit on calibration) ------------
    fits = {}
    for h in heads:
        u = wbos[h] / wbos[h].norm()
        rV = M[h].T @ u
        rK = hw2.W_k[h].double().T @ qbar[h]
        S["heads"][f"h{h}"] = {"cos_value_read_vs_bos": cos(rV, d2), "cos_key_read_vs_bos": cos(rK, d2),
                               "cos_value_read_vs_key_read": cos(rV, rK)}
        fits[h] = {"u": u, "rV": rV}
    Xs = {h: [] for h in heads}; Ys = {h: [] for h in heads}; J = []
    for idx in batches("calibration", N_CAL):
        p = ad.pre(idx)
        x2 = p["x2"].double(); beta = share(x2, mu2, d2)
        for h in heads:
            Xs[h].append(beta[:, 1:].reshape(-1).cpu()); Ys[h].append((x2[:, 1:] @ fits[h]["rV"]).reshape(-1).cpu())
        J.append(torch.arange(1, L).repeat(idx.shape[0]))
    J = torch.cat(J).numpy()
    for h in heads:
        x, y = torch.cat(Xs[h]).numpy(), torch.cat(Ys[h]).numpy()
        lam, c = np.polyfit(x, y, 1)
        res = y - (c + lam * x)
        tail = J >= 16
        xt, yt = x[tail], y[tail]
        lt, ct = np.polyfit(xt, yt, 1)
        fits[h].update({"c": float(c), "lam": float(lam)})
        S["heads"][f"h{h}"].update({
            "value_fit_all": {"c": float(c), "lambda": float(lam), "r2": float(1 - res.var() / y.var())},
            "value_fit_tail": {"r2": float(1 - (yt - (ct + lt * xt)).var() / yt.var()),
                               "sd_eps_tail_over_lambda": float(res[tail].std() / abs(lam))}})

    # ---- 4. same split in Layer 1 (summed heads) -----------------------------
    M1 = torch.einsum("hde,hef->df", hw1.W_o.double(), hw1.W_v.double())
    w1 = M1 @ xb1; u1 = w1 / w1.norm(); rV1 = M1.T @ u1
    xs1, ys1 = [], []
    for idx in batches("calibration", 64):
        p = ad.pre(idx); x1 = p["x1"].double()
        xs1.append(share(x1, mu1, d1)[:, 1:].reshape(-1).cpu()); ys1.append((x1[:, 1:] @ rV1).reshape(-1).cpu())
    x, y = torch.cat(xs1).numpy(), torch.cat(ys1).numpy()
    lam1, c1 = np.polyfit(x, y, 1); res1 = y - (c1 + lam1 * x)
    S["layer1_value_fit"] = {"cos_value_read_vs_bos": cos(rV1, d1), "lambda": float(lam1),
                             "r2": float(1 - res1.var() / y.var()), "sd_eps_over_lambda": float(res1.std() / abs(lam1)),
                             "sd_share_nonbos": float(x.std())}

    # ---- 3 + 5. output noise and causal arms ---------------------------------
    battn = attn_bias_vector(hw2).float()
    arms = ["clean", "values", "keys", "both"]
    tgt = torch.arange(L, device=dev, dtype=torch.float64)

    def arm_out(p, arm):
        x2 = p["x2"].double(); beta = share(x2, mu2, d2)
        A = p["attn2_w"].double().clone()
        y = ov_images(p["x2"], hw2).double()                             # [B,H,T,d]
        if arm in ("keys", "both"):
            T = beta.shape[1]
            mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
            for h in heads:
                sc = (Bh[h] * beta)[:, None, :].expand(-1, T, -1).masked_fill(mask, float("-inf"))
                A[:, h] = torch.softmax(sc, -1)
        eps = {}
        for h in heads:
            v = x2 @ fits[h]["rV"]
            eps[h] = v - (fits[h]["c"] + fits[h]["lam"] * beta)
            if arm in ("values", "both"):
                y[:, h] = y[:, h] - eps[h][..., None] * fits[h]["u"]
        ho = torch.einsum("bhij,bhjd->bhid", A, y)
        o2 = ho.sum(1).float() + battn
        z = {h: (ho[:, h] @ fits[h]["u"]) for h in heads}
        noise = {h: torch.einsum("bij,bj->bi", A[:, h], eps[h]) for h in heads}
        t = {h: torch.einsum("bij,bj->bi", A[:, h], beta) for h in heads}
        return o2, z, noise, t

    ridges = {a: Ridge(ad.d, dev) for a in arms}
    zcal = {a: {h: [] for h in heads} for a in arms}
    for idx in batches("calibration", N_CAL):
        p = ad.pre(idx)
        for a in arms:
            o2, z, _, _ = arm_out(p, a)
            feats, _, _ = ad.finish(p["h1"], o2)
            ridges[a].add(feats, tgt.expand(idx.shape[0], L))
            for h in heads:
                zcal[a][h].append(z[h].cpu())
    for r in ridges.values():
        r.solve()
    ev = {a: {"refit": [], "z": {h: [] for h in heads}, "noise": {h: [] for h in heads}, "t": {h: [] for h in heads}} for a in arms}
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        for a in arms:
            o2, z, noise, t = arm_out(p, a)
            feats, _, _ = ad.finish(p["h1"], o2)
            ev[a]["refit"].append(ridges[a].predict(feats).cpu())
            for h in heads:
                ev[a]["z"][h].append(z[h].cpu()); ev[a]["noise"][h].append(noise[h].cpu()); ev[a]["t"][h].append(t[h].cpu())
    i = np.arange(L)
    last = list(bins)[-1]
    S["output_noise_from_values"] = {}
    for h in heads:
        nz = torch.cat(ev["clean"]["noise"][h]).numpy(); tt = torch.cat(ev["clean"]["t"][h]).numpy()
        _, der = smooth_deriv(tt.mean(0), L)
        gap = np.abs(fits[h]["lam"] * der)
        sdn = nz.std(0)
        S["output_noise_from_values"][f"h{h}"] = {
            b: {"sd_value_noise": float(np.median(sdn[max(lo, 4):hi + 1])), "neighbor_gap": float(np.median(gap[max(lo, 4):hi + 1])),
                "ratio": float(np.median((sdn / gap)[max(lo, 4):hi + 1])),
                "relative_ratio_sd_over_i_gap": float(np.median((sdn / (i * gap))[max(lo, 4):hi + 1]))}
            for b, (lo, hi) in bins.items()}
    S["causal"] = {}
    for a in arms:
        e = {"refit": err_by_bin(torch.cat(ev[a]["refit"]).numpy(), L, bins)}
        # isotonic on the aggregate z over BOS heads
        zc = sum(torch.cat(zcal[a][h]).numpy() for h in heads)
        ze = sum(torch.cat(ev[a]["z"][h]).numpy() for h in heads)
        e["isotonic_z"] = err_by_bin(isotonic_decode(zc, ze, L), L, bins)
        S["causal"][a] = e
    (od / "kv_conditions.json").write_text(json.dumps(S, indent=2))

    print(f"== {args.model}: BOS heads {heads}, B {({h: round(b, 2) for h, b in Bh.items()})}")
    for h in heads:
        d_ = S["heads"][f"h{h}"]
        print(f"  h{h}: cos(value read, BOS)={d_['cos_value_read_vs_bos']:.3f} cos(key read, BOS)={d_['cos_key_read_vs_bos']:.3f} "
              f"cos(value read, key read)={d_['cos_value_read_vs_key_read']:.3f} | value fit R2 all={d_['value_fit_all']['r2']:.3f} "
              f"tail={d_['value_fit_tail']['r2']:.3f} sd(eps)/lambda tail={d_['value_fit_tail']['sd_eps_tail_over_lambda']:.4f}")
        o = S["output_noise_from_values"][f"h{h}"]
        print("       value noise in output / neighbor gap:", {b: round(v["ratio"], 3) for b, v in o.items()},
              "| / (i x gap):", {b: round(v["relative_ratio_sd_over_i_gap"], 4) for b, v in o.items()})
    l1 = S["layer1_value_fit"]
    print(f"  Layer 1: cos(value read, BOS)={l1['cos_value_read_vs_bos']:.3f} R2={l1['r2']:.3f} sd(eps)/lambda={l1['sd_eps_over_lambda']:.4f} (share sd {l1['sd_share_nonbos']:.4f})")
    for a, e in S["causal"].items():
        print(f"  causal {a:6s}: refit {last} relMAE {e['refit'][f'relmae_{last}']:.4f} middle {e['refit']['relmae_middle']:.4f} | "
              f"isotonic z {last} {e['isotonic_z'][f'relmae_{last}']:.4f} middle {e['isotonic_z']['relmae_middle']:.4f}")


if __name__ == "__main__":
    main()
