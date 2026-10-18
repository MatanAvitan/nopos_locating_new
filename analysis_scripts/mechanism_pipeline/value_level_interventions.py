"""Value-level interventions for Steps 4 and 5, one property per row.

Each edit acts on one selected second-layer head h along its BOS value
direction w_h = w_BOS,h / |w_BOS,h|, and leaves every other output direction
unchanged. With v_j = <B_OV,h x_j, w_h> the mapped value projection and c_h its
mean over reference keys j >= a, the head's projected update is
z_i = sum_j alpha_ij v_j, and its projected error is
e_i = sum_{j>=a} alpha_ij (v_j - c_h).

  levels_exact    set every later value projection to c_h, so e_i = 0
  error_refmean   replace e_i by its position-wise reference mean
  gap_collapse    set the salient value projections (j < a) to c_h
  readout_remove  remove z_i, the head's whole projected update
  *_rand          the same edit along a random unit output direction

Selected heads have reference BOS ratio r_h > 10. Readouts stay frozen.
Usage: python value_level_interventions.py --model 1h --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, RESULTS_ROOT, bins_for, err_by_bin, make_adapter,
                         spec_splits)
from common import batch_from_offsets, head_weights

# The selected LM is registered the same way as in the shared-input runs.
MODELS.setdefault("lmbos_L1024_s17", ("lm", "out-lm-6layer-fulltrain-bos-L1024-seed17/ckpt_20000.pt",
                                      1024, "seeds/lmbos_L1024_s17"))
N_EVAL, N_REF = 1600, 160
ARMS = ["clean", "levels_exact", "levels_exact_rand", "error_refmean", "error_refmean_rand",
        "gap_collapse", "gap_collapse_rand", "readout_remove", "readout_remove_rand"]


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
    if ad.kind == "lm":
        ad.fit_frozen(sp["calibration"][:320], L, 32)
    meta = json.loads((RESULTS_ROOT / run_id / "beta" / "meta.json").read_text())
    heads = [h for h, r in enumerate(meta["r_h"]) if r > 10]
    hw = head_weights(ad.attn2)

    def batches(name, n):
        offs = sp[name][:n]
        for s in range(0, len(offs), bs):
            yield batch_from_offsets(offs[s:s + bs], L, dev)

    # Output directions and their value-space axes, from the fixed BOS input.
    xb2 = next(iter(batches("reference", 1)))
    xb2 = ad.pre(xb2)["x2"][0, 0].double()
    g = torch.Generator(device="cpu").manual_seed(0)
    dirs, axes = {}, {}
    for kind in ["bos", "rand"]:
        D, A = [], []
        for h in heads:
            Wv, Wo = hw.W_v[h].double(), hw.W_o[h].double()          # [dh,d], [d,dh]
            if kind == "bos":
                u = Wo @ (Wv @ xb2)
            else:
                u = torch.randn(ad.d, generator=g, dtype=torch.float64).to(dev)
            u = u / u.norm()
            D.append(u); A.append(Wv.T @ (Wo.T @ u))
        dirs[kind], axes[kind] = torch.stack(D), torch.stack(A)        # [K,d]

    def parts(p, kind):
        """Per-head projections v [B,K,T], attention [B,K,T,T], and z, e terms."""
        v = torch.einsum("btd,kd->bkt", p["x2"].double(), axes[kind])
        w = p["attn2_w"][:, heads].double()
        return v, w

    # Reference: salient count a, levels c_h, and the mean projected error curve.
    sums = {k: [0.0, 0] for k in dirs}
    # BOS-salient keys follow the same rule as the shared-input interventions.
    share_json = RESULTS_ROOT / "story_evidence" / args.model / "denoise_interventions_v2.json"
    a = json.loads(share_json.read_text())["a"]
    for idx in batches("reference", N_REF):
        p = ad.pre(idx)
        for kind in dirs:
            v, _ = parts(p, kind)
            sums[kind][0] = sums[kind][0] + v[:, :, a:].sum((0, 2)); sums[kind][1] += v.shape[0] * (L - a)
    level = {k: (s / n) for k, (s, n) in sums.items()}                 # [K]
    err_sum = {k: torch.zeros(len(heads), L, dtype=torch.float64, device=dev) for k in dirs}; n_ref = 0
    for idx in batches("reference", N_REF):
        p = ad.pre(idx)
        for kind in dirs:
            v, w = parts(p, kind)
            dev_v = (v - level[kind][None, :, None])
            dev_v[:, :, :a] = 0
            err_sum[kind] += torch.einsum("bkts,bks->bkt", w, dev_v).sum(0)
        n_ref += idx.shape[0]
    err_ref = {k: s / n_ref for k, s in err_sum.items()}

    def run(p, arm):
        o2 = p["o2"].double()
        if arm == "clean":
            return o2.float()
        kind = "rand" if arm.endswith("_rand") else "bos"
        name = arm.removesuffix("_rand")
        v, w = parts(p, kind)
        c = level[kind][None, :, None]
        if name == "levels_exact":
            new_v = v.clone(); new_v[:, :, a:] = c.expand_as(v[:, :, a:])
        elif name == "gap_collapse":
            new_v = v.clone(); new_v[:, :, :a] = c.expand_as(v[:, :, :a])
        elif name == "readout_remove":
            new_v = torch.zeros_like(v)
        if name == "error_refmean":
            dev_v = v - c; dev_v[:, :, :a] = 0
            e = torch.einsum("bkts,bks->bkt", w, dev_v)
            delta = err_ref[kind][None] - e
        else:
            delta = torch.einsum("bkts,bks->bkt", w, new_v - v)
        return (o2 + torch.einsum("bkt,kd->btd", delta, dirs[kind])).float()

    preds = {x: [] for x in ARMS}; loss = {x: [] for x in ARMS}
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        for arm in ARMS:
            _, fr, ls = ad.finish(p["h1"], run(p, arm), idx if ad.kind == "lm" else None)
            preds[arm].append(fr.double().cpu())
            if ls is not None:
                loss[arm].append(float(ls.double().mean()))
    S = {"model": args.model, "a": a, "heads": heads,
         "levels": {k: v.tolist() for k, v in level.items()}, "arms": {}}
    for arm in ARMS:
        e = err_by_bin(torch.cat(preds[arm]).numpy(), L, bins)
        if loss[arm]:
            e["loss_increase"] = float(np.mean(loss[arm]) - np.mean(loss["clean"]))
        S["arms"][arm] = e
    od = RESULTS_ROOT / "story_evidence" / args.model
    (od / "value_level_interventions.json").write_text(json.dumps(S, indent=2))
    print(f"== {args.model}: a={a} heads={heads}")
    for arm, e in S["arms"].items():
        print(f"  {arm:22s} relMAE " + " ".join(f"{b}={100 * e[f'relmae_{b}']:.1f}" for b in bins)
              + (f"  dloss {e['loss_increase']:+.4f}" if "loss_increase" in e else ""))


if __name__ == "__main__":
    main()
