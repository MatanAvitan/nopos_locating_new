"""Causal interventions for the denoising view, one property per row.

Each edit removes (or preserves) exactly one property that a step of the
mechanism claims, and each has a control: the same edit along a random
direction with the same norm. Position error is the MAE in tokens of the
frozen position head (1H, 12H) or the frozen Layer-2 probe (LM), by bin; for
the LM we also report the increase in next-token loss.

Directions: u1 = unit direction of BOS's state after Layer-1 attention (the
Layer-1 BOS direction); d = x_BOS - mu at the Layer-2 input (the Layer-2 BOS
direction), with share beta_j = <x_j - mu, d>/|d|^2.

  preserve      at the Layer-2 key and value inputs, set the share of every
                ordinary key (j >= a) to the mean ordinary share
  mlp_mean      keep the Layer-1 MLP's mean write along u1 but remove its
                token-by-token part (the per-token cancellation), j >= a
  keys_noshare  remove the share from every Layer-2 key input (all j)
  values_noshare remove the share from every Layer-2 value input (all j)
  *_rand        the same edit along a random direction of the same norm

Usage: python denoise_interventions_v2.py --model 1h --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, RESULTS_ROOT, attn_from_inputs, bins_for, err_by_bin,
                         ln, make_adapter, spec_splits)
from common import batch_from_offsets
from denoise_interventions import blocks_of

N_EVAL, N_REF = 1600, 160


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    dev, bs = args.device, args.batch_size
    ad, L, run_id = make_adapter(args.model, dev)
    sp = spec_splits(L); bins = bins_for(L)
    mlp, ln_pre, ln2 = blocks_of(ad)
    if ad.kind == "lm":
        ad.fit_frozen(sp["calibration"][:320], L, 32)

    def batches(name, n):
        offs = sp[name][:n]
        for s in range(0, len(offs), bs):
            yield batch_from_offsets(offs[s:s + bs], L, dev)

    # reference: directions, means, BOS-salient positions
    s2 = None; n = 0; bos_post = xb2 = None; shares = []
    for idx in batches("reference", N_REF):
        p = ad.pre(idx)
        if bos_post is None:
            bos_post, xb2 = p["post1"][0, 0].double(), p["x2"][0, 0].double()
        x2 = p["x2"].double()
        s2 = x2[:, 1:].sum((0, 1)) if s2 is None else s2 + x2[:, 1:].sum((0, 1)); n += idx.shape[0] * (L - 1)
    mu2 = s2 / n
    d = xb2 - mu2; dd = d @ d
    u1 = bos_post / bos_post.norm()
    g = torch.Generator(device="cpu").manual_seed(0)
    def rand_like(norm):
        r = torch.randn(ad.d, generator=g, dtype=torch.float64).to(dev); return r / r.norm() * norm
    r2 = rand_like(d.norm())                      # random Layer-2 direction, norm of d
    r1 = rand_like(1.0)                           # random unit Layer-1 direction
    # BOS-salient keys and mean shares / mean MLP writes (reference split)
    sh, shr, mw, mwr = [], [], [], []
    for idx in batches("reference", N_REF):
        p = ad.pre(idx); x2 = p["x2"].double()
        sh.append((((x2 - mu2) @ d) / dd).cpu()); shr.append((((x2 - mu2) @ r2) / (r2 @ r2)).cpu())
        m = (p["h1"] - p["post1"]).double()
        mw.append((m @ u1).cpu()); mwr.append((m @ r1).cpu())
    sh, shr, mw, mwr = (torch.cat(x).numpy() for x in (sh, shr, mw, mwr))
    sinks = [int(j) + 1 for j in np.nonzero((sh[:, 1:] > 0.5).mean(0) > 0.9)[0]]
    a = 1 + len(sinks)
    ordm = np.ones(L, bool); ordm[:a] = False
    beta_ord, beta_ord_r = float(sh[:, ordm].mean()), float(shr[:, ordm].mean())
    mw_ord, mwr_ord = float(mw[:, ordm].mean()), float(mwr[:, ordm].mean())

    def set_share(x, direc, target, from_j):
        """Set the share of positions >= from_j along direc to target."""
        x = x.double().clone()
        b = ((x - mu2) @ direc) / (direc @ direc)
        x[:, from_j:] += (target - b[:, from_j:])[..., None] * direc
        return x.float()

    def remove_share(x, direc):
        x = x.double()
        b = ((x - mu2) @ direc) / (direc @ direc)
        return (x - b[..., None] * direc).float()

    def mlp_mean(p, direc, mean_val):
        m = (p["h1"] - p["post1"]).double()
        w = m @ direc
        m = m.clone(); m[:, a:] += ((mean_val - w[:, a:])[..., None] * direc)
        return (p["post1"].double() + m).float()

    arms = ["clean", "preserve", "preserve_rand", "mlp_mean", "mlp_mean_rand",
            "keys_noshare", "keys_noshare_rand", "values_noshare", "values_noshare_rand"]

    def run(p, arm):
        h1 = p["h1"]
        if arm.startswith("mlp_mean"):
            h1 = mlp_mean(p, u1 if arm == "mlp_mean" else r1, mw_ord if arm == "mlp_mean" else mwr_ord)
            x2 = ln(h1, ln2)
            return h1, attn_from_inputs(ad.attn2, x2, x2, x2)[1]
        x2 = p["x2"]
        if arm == "clean":
            return h1, p["o2"]
        if arm.startswith("preserve"):
            direc, tgt = (d, beta_ord) if arm == "preserve" else (r2, beta_ord_r)
            e = set_share(x2, direc, tgt, a)
            return h1, attn_from_inputs(ad.attn2, x2, e, e)[1]
        direc = d if not arm.endswith("_rand") else r2
        e = remove_share(x2, direc)
        if arm.startswith("keys"):
            return h1, attn_from_inputs(ad.attn2, x2, e, x2)[1]
        return h1, attn_from_inputs(ad.attn2, x2, x2, e)[1]

    preds = {a_: [] for a_ in arms}; loss = {a_: [] for a_ in arms}
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        for arm in arms:
            h1, o2 = run(p, arm)
            _, fr, ls = ad.finish(h1, o2, idx if ad.kind == "lm" else None)
            preds[arm].append(fr.double().cpu())
            if ls is not None:
                loss[arm].append(float(ls.double().mean()))
    S = {"model": args.model, "a": a, "sinks": sinks, "arms": {}}
    for arm in arms:
        e = err_by_bin(torch.cat(preds[arm]).numpy(), L, bins)
        if loss[arm]:
            e["loss_increase"] = float(np.mean(loss[arm]) - np.mean(loss["clean"]))
        S["arms"][arm] = e
    od = RESULTS_ROOT / "story_evidence" / args.model; od.mkdir(parents=True, exist_ok=True)
    (od / "denoise_interventions_v2.json").write_text(json.dumps(S, indent=2))
    np.savez(od / "denoise_interventions_v2_preds.npz", **{x: torch.cat(preds[x]).numpy().astype(np.float32) for x in preds})
    print(f"== {args.model}: a={a}")
    for arm, e in S["arms"].items():
        print(f"  {arm:20s} MAE " + " ".join(f"{b}={e[f'mae_{b}']:.1f}" for b in bins)
              + (f"  dloss {e['loss_increase']:+.4f}" if "loss_increase" in e else ""))


if __name__ == "__main__":
    main()
