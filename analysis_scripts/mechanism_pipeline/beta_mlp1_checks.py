"""Two checks on the Layer-1 BOS-removal construction.

1. Activation swap: evaluate with ReLU in place of GELU in the Layer-1 MLP
   (same weights). Report the beta profile, tail noise and refit late error.
2. Where the cap comes from: the pre-MLP LayerNorm divides token j by its
   own scale sigma_j. For the top removal neurons (from mlp1_construct.json),
   compare the pre-activation z_k against the BOS share s_j
     (a) as is:              z_k = W_k . LN(post1_j) + b_k
     (b) scale put back:     z~_k = (z_k - b_k - W_k . ln_bias) * sigma_j
                                  = W_k . (gamma * (post1_j - mean(post1_j)))
   (b) is exactly linear in post1_j. Report the R^2 of a linear fit on s for
   (a) and (b), and the curvature (quadratic term), on tokens j <= 64.

Usage: python beta_mlp1_checks.py --model pv_unifL1_s42 --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, Ridge, attn_from_inputs, bins_for, err_by_bin,
                         ln, make_adapter, out_dir, spec_splits)
from common import batch_from_offsets

N_EVAL = 400
N_SCATTER = 64
TOP = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    od = out_dir(run_id, "beta_v2"); fd = od / "figures"
    C = json.loads((od / "mlp1_construct.json").read_text())
    neurons = [t["neuron"] for t in C["bins"]["j5-16"]["top10"]][:TOP]
    b1 = ad.model.block1
    mlp, ln2, blk2_ln = b1.mlp, b1.ln_2, ad.model.block2.ln_1
    splits = spec_splits(L); bins = bins_for(L)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)

    def batches(name, n=None):
        offs = splits[name][:n] if n else splits[name]
        for s in range(0, len(offs), args.batch_size):
            yield batch_from_offsets(offs[s:s + args.batch_size], L, dev)

    # reference: BOS direction for s (post1 space)
    sp = torch.zeros(ad.d, device=dev, dtype=torch.float64); n = 0; bos = None
    for idx in batches("reference"):
        p = ad.pre(idx)
        bos = p["post1"][0, 0].double() if bos is None else bos
        sp += p["post1"][:, 1:].double().sum((0, 1)); n += idx.shape[0] * (L - 1)
    mu = sp / n; v = bos - mu; vv = v @ v

    # ---- check 1: GELU -> ReLU ------------------------------------------------
    acts = {"gelu": F.gelu, "relu": F.relu}

    def h1_of(p, act):
        return p["post1"] + mlp.c_proj(acts[act](mlp.c_fc(p["h1bar"])))

    ridges = {a: Ridge(ad.d, dev) for a in acts}
    x2sum = {a: torch.zeros(ad.d, device=dev, dtype=torch.float64) for a in acts}
    x2bos, ncal = {}, 0
    for idx in batches("calibration"):
        p = ad.pre(idx)
        for a in acts:
            h1 = h1_of(p, a); x2 = ln(h1, blk2_ln)
            x2sum[a] += x2[:, 1:].double().sum((0, 1)); x2bos.setdefault(a, x2[0, 0].double().clone())
            _, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, _, _ = ad.finish(h1, o2)
            ridges[a].add(feats, tgt.expand(idx.shape[0], L))
        ncal += idx.shape[0] * (L - 1)
    for r in ridges.values():
        r.solve()
    mu2 = {a: x2sum[a] / ncal for a in acts}; v2 = {a: x2bos[a] - mu2[a] for a in acts}
    res = {a: {"beta": [], "refit": [], "frozen": []} for a in acts}
    zs, zt, ss, js = [], [], [], []
    for bi, idx in enumerate(batches("evaluation", N_EVAL)):
        p = ad.pre(idx)
        for a in acts:
            h1 = h1_of(p, a); x2 = ln(h1, blk2_ln)
            _, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, fr, _ = ad.finish(h1, o2)
            res[a]["beta"].append((((x2.double() - mu2[a]) @ v2[a]) / (v2[a] @ v2[a])).cpu())
            res[a]["refit"].append(ridges[a].predict(feats).cpu()); res[a]["frozen"].append(fr.double().cpu())
        # ---- check 2 data -------------------------------------------------
        post = p["post1"][:, 1:N_SCATTER + 1].double()
        s = ((post - mu) @ v) / vv
        cen = post - post.mean(-1, keepdim=True)
        sigma = torch.sqrt(cen.pow(2).mean(-1) + 1e-5)
        W = mlp.c_fc.weight[neurons].double(); bk = mlp.c_fc.bias[neurons].double() if mlp.c_fc.bias is not None else 0
        g = ln2.weight.double(); lb = ln2.bias.double() if ln2.bias is not None else torch.zeros_like(g)
        z = (cen / sigma[..., None] * g + lb) @ W.T + bk                     # (a) actual pre-activation
        ztil = (cen * g) @ W.T                                               # (b) scale put back
        zs.append(z.reshape(-1, len(neurons)).cpu()); zt.append(ztil.reshape(-1, len(neurons)).cpu())
        ss.append(s.reshape(-1).cpu()); js.append(torch.arange(1, N_SCATTER + 1).repeat(idx.shape[0]))
    S = {"model": args.model, "neurons": neurons, "activation_swap": {}}
    for a in acts:
        B_ = torch.cat(res[a]["beta"]).numpy()
        S["activation_swap"][a] = {"beta_profile": {str(j): float(B_[:, j].mean()) for j in [1, 2, 3, 4, 6, 8, 16, 64, 256] if j < L},
                                   "tail_sd_j16on": float(B_[:, 16:].std()),
                                   "refit": err_by_bin(torch.cat(res[a]["refit"]).numpy(), L, bins),
                                   "frozen": err_by_bin(torch.cat(res[a]["frozen"]).numpy(), L, bins)}
    z = torch.cat(zs).numpy(); zt_ = torch.cat(zt).numpy(); s = torch.cat(ss).numpy(); j = torch.cat(js).numpy()
    S["cap"] = {}
    for k, nk in enumerate(neurons):
        out = {}
        for tag, y in [("a_actual", z[:, k]), ("b_scale_put_back", zt_[:, k])]:
            c1 = np.polyfit(s, y, 1); c2 = np.polyfit(s, y, 2)
            r2_1 = 1 - np.var(y - np.polyval(c1, s)) / np.var(y)
            r2_2 = 1 - np.var(y - np.polyval(c2, s)) / np.var(y)
            # slope on the low half and the high half of s: a cap shows as a drop
            lo = s < np.quantile(s, 0.8); hi = s >= np.quantile(s, 0.95)
            out[tag] = {"r2_linear": float(r2_1), "r2_quadratic": float(r2_2),
                        "slope_low_s": float(np.polyfit(s[lo], y[lo], 1)[0]),
                        "slope_top5pct_s": float(np.polyfit(s[hi], y[hi], 1)[0])}
        S["cap"][f"n{nk}"] = out
    (od / "mlp1_checks.json").write_text(json.dumps(S, indent=2))

    fig, ax = plt.subplots(2, TOP, figsize=(3.3 * TOP, 6.4))
    for k, nk in enumerate(neurons):
        for r, (tag, y) in enumerate([("(a) actual z", z[:, k]), ("(b) z × σ (LN scale put back)", zt_[:, k])]):
            ax[r, k].scatter(s, y, c=np.log(j), s=1, alpha=.4, cmap="viridis")
            ax[r, k].set_title(f"n{nk} {tag}", fontsize=8); ax[r, k].set_xlabel("s", fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "mlp1_cap_check.png", dpi=110); plt.close(fig)
    for a, d_ in S["activation_swap"].items():
        print(f"{a:5s}", {jj: round(x, 3) for jj, x in d_["beta_profile"].items()}, "tail sd %.4f" % d_["tail_sd_j16on"],
              "refit late %.4f frozen late %.4f" % (d_["refit"].get("relmae_late", np.nan), d_["frozen"].get("relmae_late", np.nan)))
    for nk, d_ in S["cap"].items():
        print(nk, {t: {k2: round(v2_, 3) for k2, v2_ in x.items()} for t, x in d_.items()})


if __name__ == "__main__":
    main()
