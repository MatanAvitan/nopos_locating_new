"""Causal test of "score = B x beta" in every BOS head, valid for any beta profile.

Edit: add delta (x_BOS - mu) to the Layer-2 KEY input of every key j >= 1
(queries and values unchanged), so every non-BOS key's centered BOS share
beta_j rises by exactly delta. If the score of key j is a_i + B beta_j, every
non-BOS score rises by B delta, so the log-odds of attention on BOS,
  logit_i = log(alpha_i0 / (1 - alpha_i0)),
must fall by exactly B delta, for every query and whatever the beta profile.
Measured: median over sequences and queries (i >= 4) of the change in logit,
divided by -delta (the "causal B"), against B fitted from the attention rows
(maximum likelihood, calibration split).
Control: the same edit along a random direction with the same norm (5 draws).
Downstream: refit-probe position error (relMAE, last bin) and, for the LM, the
change in next-token loss.

Usage: python causal_story.py --model lmbos_s42 --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, RESULTS_ROOT, Ridge, attn_from_inputs, bins_for,
                         err_by_bin, make_adapter, spec_splits)
from common import batch_from_offsets
from softmax_beta_check import ml_B

DELTAS = [-0.2, -0.1, 0.05, 0.1, 0.2]
N_RAND = 5
N_EVAL, N_CAL = 200, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    dev, bs = args.device, args.batch_size
    ad, L, run_id = make_adapter(args.model, dev)
    meta = json.loads((RESULTS_ROOT / run_id / "beta" / "meta.json").read_text())
    heads = meta["bos_heads"] or [int(np.argmax(meta["r_h"]))]
    sp = spec_splits(L); bins = bins_for(L); last = list(bins)[-1]
    if ad.kind == "lm":
        ad.fit_frozen(sp["calibration"][:N_CAL], L, bs)

    def batches(name, n):
        offs = sp[name][:n]
        for s in range(0, len(offs), bs):
            yield batch_from_offsets(offs[s:s + bs], L, dev)

    with torch.no_grad():
        xs = None; n = 0; xb = None
        for idx in batches("reference", 160):
            x2 = ad.pre(idx)["x2"].double()
            xb = x2[0, 0].clone() if xb is None else xb
            xs = x2[:, 1:].sum((0, 1)) if xs is None else xs + x2[:, 1:].sum((0, 1)); n += x2.shape[0] * (L - 1)
        mu = xs / n; d = (xb - mu)
        g = torch.Generator(device="cpu").manual_seed(0)
        rands = [(lambda r: r / r.norm() * d.norm())(torch.randn(ad.d, generator=g, dtype=torch.float64).to(dev)) for _ in range(N_RAND)]

    # B from the attention rows (ML on calibration), per BOS head
    Tm = min(L, 256)
    with torch.no_grad():
        bet, Ar = [], []
        for idx in batches("calibration", 64):
            p = ad.pre(idx); x2 = p["x2"].double()
            bet.append((((x2 - mu) @ d) / (d @ d))[:, :Tm].cpu().numpy())
            Ar.append(p["attn2_w"][:, heads, :Tm, :Tm].double().cpu().numpy())
    bet = np.concatenate(bet); Ar = np.concatenate(Ar)
    B_fit = {h: ml_B(Ar[:, k], bet, np.linspace(-20, 20, 801)) for k, h in enumerate(heads)}

    arms = {"clean": None}
    for dl in DELTAS:
        arms[f"bos_dir_{dl:+g}"] = (dl, d)
    for r_i, r in enumerate(rands):
        arms[f"random{r_i}_+0.2"] = (0.2, r)

    @torch.no_grad()
    def run(p, arm):
        x2 = p["x2"]
        k_in = x2
        if arm is not None:
            dl, direction = arm
            k_in = x2.double().clone()
            k_in[:, 1:] += dl * direction
            k_in = k_in.float()
        return attn_from_inputs(ad.attn2, x2, k_in, x2)

    tgt = torch.arange(L, device=dev, dtype=torch.float64)
    ridges = {a: Ridge(ad.d, dev) for a in arms}
    with torch.no_grad():
        for idx in batches("calibration", N_CAL):
            p = ad.pre(idx)
            for a, arm in arms.items():
                _, o2 = run(p, arm)
                feats, _, _ = ad.finish(p["h1"], o2)
                ridges[a].add(feats, tgt.expand(idx.shape[0], L))
    for r in ridges.values():
        r.solve()
    logit = {a: [] for a in arms}; refit = {a: [] for a in arms}; loss = {a: [] for a in arms}
    with torch.no_grad():
        for idx in batches("evaluation", N_EVAL):
            p = ad.pre(idx)
            for a, arm in arms.items():
                w, o2 = run(p, arm)
                ab = w[:, heads, :, 0].double().clamp(1e-12, 1 - 1e-12)
                logit[a].append(torch.log(ab / (1 - ab)).cpu())
                feats, _, ls = ad.finish(p["h1"], o2, idx if ad.kind == "lm" else None)
                refit[a].append(ridges[a].predict(feats).cpu())
                if ls is not None:
                    loss[a].append(ls.double().mean().item())
    lg = {a: torch.cat(v).numpy() for a, v in logit.items()}
    S = {"model": args.model, "bos_heads": heads, "B_fit_attention": B_fit, "arms": {}}
    for a in arms:
        e = {"refit_relmae_last": err_by_bin(torch.cat(refit[a]).numpy(), L, bins)[f"relmae_{last}"]}
        if loss[a]:
            e["lm_loss"] = float(np.mean(loss[a]))
        if a != "clean":
            dl = arms[a][0]
            dlog = lg[a] - lg["clean"]                                      # [n,H,T]
            e["causal_B"] = {f"h{h}": float(np.median(dlog[:, k, 4:]) / -dl) for k, h in enumerate(heads)}
        S["arms"][a] = e
    if loss["clean"]:
        for a in arms:
            S["arms"][a]["lm_loss_delta"] = S["arms"][a]["lm_loss"] - S["arms"]["clean"]["lm_loss"]
    od = RESULTS_ROOT / "story_evidence" / args.model; od.mkdir(parents=True, exist_ok=True)
    (od / "causal_story.json").write_text(json.dumps(S, indent=2))
    print(f"== {args.model}: B fitted from attention", {f"h{h}": round(b, 2) for h, b in B_fit.items()})
    for a, e in S["arms"].items():
        cb = {h: round(v, 2) for h, v in e.get("causal_B", {}).items()}
        print(f"  {a:14s} refit {last} relMAE {e['refit_relmae_last']:.4f}" + (f"  dloss {e['lm_loss_delta']:+.4f}" if "lm_loss_delta" in e else "")
              + (f"  causal B {cb}" if cb else ""))


if __name__ == "__main__":
    main()
