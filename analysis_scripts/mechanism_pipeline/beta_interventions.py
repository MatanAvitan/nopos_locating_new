"""Causal interventions of the BOS-share spec (beta_spec.md F1-F5).

Each arm edits the Layer-2 attention input before one projection (query,
key, or value), recomputes the Layer-2 attention and everything downstream,
and reports:
  frozen : the frozen decoder (trained pos_head, or the LM's clean ridge probe)
  refit  : a ridge probe on the same features, refit on intervened calibration
  gamma  : log-log slope of the mean BOS weight of every BOS head
  loss   : (LM only) change in mean next-token loss over positions 0..L-2

Arms (edits apply to key/value/query position j >= 1; BOS is unchanged):
  F1_keys      k_j = beta_bar x_BOS + xi_j
  F2_values    v_j = beta_bar x_BOS + xi_j
  F3_queries   q_i = xi_i
  F4_all       k_j = v_j = xi_j
  F5a_perm     k_j = beta_{pi(j)} x_BOS + xi_j, pi a random permutation of 1..T-1
  F5b_rand<s>  F1 with x_BOS replaced by a random vector of the same norm
  *_bosheads   the same edit on the BOS heads only (multi-head models)

Usage: python beta_interventions.py --model 1h --device cuda:1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, Ridge, attn_from_inputs, bins_for, err_by_bin,
                         gamma_fit, gamma_windows, make_adapter, out_dir,
                         spec_splits)
from common import batch_from_offsets

N_RAND = 5


def build_arms(R, H, heads, d, device, multihead):
    xb = R["xb2"].float().to(device)
    bbar = float(R["beta_bar"])
    xbar = R["xbar2_nonbos"].float().to(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    rands = []
    for _ in range(N_RAND):
        r = torch.randn(d, generator=g).to(device)
        r = r / r.norm() * xb.norm()
        rands.append((r, float(xbar @ r / (r @ r))))

    def split(x, ref):
        b = (x @ ref) / (ref @ ref)
        return b, x - b[..., None] * ref

    def keep_bos(x_new, x):
        x_new = x_new.clone(); x_new[:, 0] = x[:, 0]
        return x_new

    def f1(x, beta, xi, gen):
        return keep_bos(bbar * xb + xi, x)

    def f4(x, beta, xi, gen):
        return keep_bos(xi, x)

    def f5a(x, beta, xi, gen):
        Bn, T = beta.shape
        perm = torch.stack([torch.randperm(T - 1, generator=gen) + 1 for _ in range(Bn)]).to(x.device)
        bp = torch.gather(beta[:, 1:], 1, perm - 1)
        return keep_bos(torch.cat([beta[:, :1], bp], 1)[..., None] * xb + xi, x)

    def f5b(r, rb):
        def fn(x, beta, xi, gen):
            _, xir = split(x, r)
            return keep_bos(rb * r + xir, x)
        return fn

    arms = {"clean": {}, "F1_keys": {"k": f1}, "F2_values": {"v": f1},
            "F3_queries": {"q": f4}, "F4_all": {"k": f4, "v": f4},
            "F5a_perm": {"k": f5a}}
    for s, (r, rb) in enumerate(rands):
        arms[f"F5b_rand{s}"] = {"k": f5b(r, rb)}
    if multihead:
        for base in ["F1_keys", "F2_values", "F4_all"]:
            arms[f"{base}_bosheads"] = dict(arms[base], heads=heads)
    return arms, split, xb


@torch.no_grad()
def run_arm(ad, arm, p, beta, xi, gen, H):
    x2 = p["x2"].float()
    ins = {}
    for role in "qkv":
        if role in arm:
            e = arm[role](x2, beta, xi, gen)
            if "heads" in arm:
                full = x2[:, None].expand(-1, H, -1, -1).clone()
                full[:, arm["heads"]] = e[:, None]
                e = full
            ins[role] = e
        else:
            ins[role] = x2
    w, o2 = attn_from_inputs(ad.attn2, ins["q"], ins["k"], ins["v"])
    return w, o2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    ad, L, run_id = make_adapter(args.model, dev)
    od = out_dir(run_id)
    R = {k: (v.to(dev) if torch.is_tensor(v) else v)
         for k, v in torch.load(od / "ref.pt", weights_only=False).items()}
    H, d = ad.H, ad.d
    heads = R["bos_heads"] or [int(R["r_h"].argmax())]
    arms, split, xb = build_arms(R, H, heads, d, dev, multihead=H > 1)
    splits = spec_splits(L)
    if ad.kind == "lm":
        ad.fit_frozen(splits["calibration"], L, args.batch_size)
    bins = bins_for(L)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)

    def batches(name):
        offs = splits[name]
        for s in range(0, len(offs), args.batch_size):
            yield batch_from_offsets(offs[s:s + args.batch_size], L, dev)

    # phase 1: refit probes on intervened calibration activations
    ridges = {a: Ridge(d, dev) for a in arms}
    gen = torch.Generator().manual_seed(1)
    for idx in batches("calibration"):
        p = ad.pre(idx)
        beta, xi = split(p["x2"].float(), xb)
        for a, arm in arms.items():
            _, o2 = run_arm(ad, arm, p, beta, xi, gen, H)
            feats, _, _ = ad.finish(p["h1"], o2)
            ridges[a].add(feats, tgt.expand(idx.shape[0], L))
    for r in ridges.values():
        r.solve()
    print(f"[cal] {time.time() - t0:.0f}s")

    # phase 2: evaluation
    frozen = {a: [] for a in arms}
    refit = {a: [] for a in arms}
    abos = {a: torch.zeros(H, L, device=dev, dtype=torch.float64) for a in arms}
    loss = {a: [] for a in arms}
    rebuild_err = 0.0
    n = 0
    gen = torch.Generator().manual_seed(2)
    for idx in batches("evaluation"):
        p = ad.pre(idx)
        beta, xi = split(p["x2"].float(), xb)
        for a, arm in arms.items():
            w, o2 = run_arm(ad, arm, p, beta, xi, gen, H)
            if a == "clean":
                rebuild_err = max(rebuild_err, float((o2 - p["o2"]).abs().max()))
            feats, fr, ls = ad.finish(p["h1"], o2, idx if ad.kind == "lm" else None)
            frozen[a].append(fr.double().cpu())
            refit[a].append(ridges[a].predict(feats).cpu())
            abos[a] += w[..., 0].double().sum(0)
            if ls is not None:
                loss[a].append(ls.double().cpu())
        n += idx.shape[0]
    print(f"[eval] {time.time() - t0:.0f}s  clean rebuild max|do2|={rebuild_err:.2e}")

    S = {"rebuild_max_abs_err": rebuild_err, "bos_heads": heads,
         "beta_bar": float(R["beta_bar"]), "arms": {}}
    for a in arms:
        fz = torch.cat(frozen[a]).numpy(); rf = torch.cat(refit[a]).numpy()
        ab = (abos[a] / n).cpu().numpy()
        e = {"frozen": err_by_bin(fz, L, bins), "refit": err_by_bin(rf, L, bins),
             "gamma": {f"h{h}": {w: gamma_fit(ab[h], lo, hi)
                                 for w, (lo, hi) in gamma_windows(L).items()}
                       for h in heads},
             "alpha_bos_mean_at": {f"h{h}": {str(k): float(ab[h, k]) for k in [1, 16, 128, 1023, 8191] if k < L}
                                   for h in heads}}
        if loss[a]:
            lm = torch.cat(loss[a]).mean().item()
            e["lm_loss"] = lm
        S["arms"][a] = e
    if loss["clean"]:
        base = S["arms"]["clean"]["lm_loss"]
        for a in arms:
            S["arms"][a]["lm_loss_delta"] = S["arms"][a]["lm_loss"] - base
    (od / "interventions.json").write_text(json.dumps(S, indent=2))
    for a, e in S["arms"].items():
        g = np.nanmedian([v.get("16_127", np.nan) for v in e["gamma"].values()])
        late = "late" if "late" in bins else "middle"
        print(f"{a:22s} frozen relMAE {late}={e['frozen'][f'relmae_{late}']:.3f} "
              f"refit={e['refit'][f'relmae_{late}']:.3f}  gamma16_127={g:.3f}"
              + (f"  dloss={e['lm_loss_delta']:+.4f}" if "lm_loss_delta" in e else ""))


if __name__ == "__main__":
    main()
