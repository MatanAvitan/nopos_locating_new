"""G3 of the BOS-share spec: a hand-set Layer 2 inside the real 1H model.

The construction sets W_Q = 0, b_Q = sqrt(d_h) B e_1, and row 1 of W_K to
x_BOS^T/||x_BOS||^2. The score of key j is then B beta_j plus a constant
(the key bias), which cancels in the softmax. So the attention row is
softmax_j(B beta_j) exactly, and this script computes it in that form.
Optional OV: row 1 of W_V is x_BOS^T/||x_BOS||^2, b_V = 0, and column 1 of
W_O is the trained w_BOS, so the head writes t_i w_BOS (+ b_O).

The rest of the model is (a) at initialization or (b) trained. Refit ridge
probes (calibration split) read ln_f(h2) and the attention output o2.

Usage: python beta_construct.py --device cuda:3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (Ridge, ToyAdapter, attn_from_inputs, bins_for,
                         err_by_bin, out_dir, spec_splits)
from common import batch_from_offsets

B_SWEEP = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0]


@torch.no_grad()
def constructed_o2(ad, p, xb, Bs, ov, w_bos_trained):
    x2 = p["x2"].double()
    beta = (x2 @ xb) / (xb @ xb)                                   # [B,T]
    T = beta.shape[1]
    s = (Bs * beta)[:, None, :].expand(-1, T, -1)
    mask = torch.triu(torch.ones(T, T, device=s.device, dtype=torch.bool), 1)
    a = torch.softmax(s.masked_fill(mask, float("-inf")), -1)       # [B,T,T]
    bo = ad.attn2.c_proj.bias
    bo = bo.double() if bo is not None else 0.0
    if ov == "constructed":
        t = a @ beta[..., None]                                     # [B,T,1]
        return (t * w_bos_trained + bo).float()
    # trained OV: the head's own value path on the constructed attention
    from common import head_weights
    hw = head_weights(ad.attn2)
    v = torch.einsum("btd,ed->bte", x2, hw.W_v[0].double()) + hw.b_v[0].double()
    o = torch.einsum("bij,bje->bie", a, v)
    return (torch.einsum("bte,de->btd", o, hw.W_o[0].double()) + bo).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    L = 1024
    splits = spec_splits(L)
    bins = bins_for(L)
    od = out_dir("attn2_full_L1024")
    Rt = torch.load(od / "ref.pt", weights_only=False)
    w_bos_trained = Rt["w_bos"][0].double().to(dev)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)
    res = {}
    for rest in ["trained", "init"]:
        ad = ToyAdapter("attn2_full", dev, init_only=(rest == "init"))
        H = ad.H
        assert H == 1
        idx0 = batch_from_offsets(splits["reference"][:1], L, dev)
        xb = ad.pre(idx0)["x2"][0, 0].double()
        arms = {"trained_layer2": None}
        for ov in ["trained", "constructed"]:
            for Bs in B_SWEEP:
                arms[f"B{Bs:g}_ov-{ov}"] = (Bs, ov)
        ridges = {(a, r): Ridge(ad.d, dev) for a in arms for r in ["final", "o2"]}

        def outputs(p, arm):
            if arm is None:
                o2 = p["o2"]
            else:
                o2 = constructed_o2(ad, p, xb, arm[0], arm[1], w_bos_trained)
            feats, _, _ = ad.finish(p["h1"], o2)
            return {"final": feats, "o2": o2}

        for s in range(0, len(splits["calibration"]), args.batch_size):
            idx = batch_from_offsets(splits["calibration"][s:s + args.batch_size], L, dev)
            p = ad.pre(idx)
            for a, arm in arms.items():
                for r, X in outputs(p, arm).items():
                    ridges[(a, r)].add(X, tgt.expand(idx.shape[0], L))
        for rg in ridges.values():
            rg.solve()
        preds = {k: [] for k in ridges}
        for s in range(0, len(splits["evaluation"]), args.batch_size):
            idx = batch_from_offsets(splits["evaluation"][s:s + args.batch_size], L, dev)
            p = ad.pre(idx)
            for a, arm in arms.items():
                for r, X in outputs(p, arm).items():
                    preds[(a, r)].append(ridges[(a, r)].predict(X).cpu())
        for (a, r), v in preds.items():
            res[f"rest-{rest}/{a}/probe-{r}"] = err_by_bin(torch.cat(v).numpy(), L, bins)
        print(f"[{rest}] {time.time() - t0:.0f}s")
        del ad
        torch.cuda.empty_cache()
    (od / "G3_construct.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:55s} relMAE e/m/l = {v['relmae_early']:.3f} {v['relmae_middle']:.3f} "
              f"{v['relmae_late']:.4f}  MAE all = {v['mae_all']:.2f}")


if __name__ == "__main__":
    main()
