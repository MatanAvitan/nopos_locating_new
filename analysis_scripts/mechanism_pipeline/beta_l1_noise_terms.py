"""Does content noise from Layer 1 reach the Layer-2 keys?

After Layer-1 attention the residual of token i is
  h~_i = e_i + sum_h alpha^h_{i,0} w^h_BOS + sum_h sum_{j>=1} alpha^h_{ij} OV^h x_j + b_attn.
Its BOS amount along u = w_BOS (Layer-1 value of BOS, summed over heads),
  amount_i = <h~_i, u>/|u|^2, splits exactly into
    signal   sum_h alpha^h_{i,0} <w^h_BOS, u>/|u|^2
    own      <e_i, u>/|u|^2                        (the token's own embedding)
    prefix   sum_h sum_{j>=1} alpha^h_{ij} <OV^h x_j, u>/|u|^2   (earlier tokens' values)
    const    <b_attn, u>/|u|^2.
Per position we report the mean signal and the sd over sequences of every term.
Then, at fixed position j (across sequences), how much of the variation of
  (a) the BOS amount at the MLP input (LN of h~, before the Layer-1 MLP), and
  (b) the BOS amount of the Layer-2 key (the centered share at the Layer-2 input)
the two noise terms explain (R^2 of a linear fit on own + prefix).

Usage: python beta_l1_noise_terms.py --models 1h sw_L1024_unifL1_s42 12h lmbos_s42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import MODELS, RESULTS_ROOT, ln, make_adapter, spec_splits
from common import attn_bias_vector, batch_from_offsets, head_weights, ov_images

N_SEQ = 256
POS = [1, 2, 4, 8, 16, 64, 127, 256, 1000]


@torch.no_grad()
def run(key, dev):
    ad, L, _ = make_adapter(key, dev)
    offs = spec_splits(L)["evaluation"][:N_SEQ]
    hw1 = head_weights(ad.attn1)
    ln_pre = ad.model.block1.ln_2 if ad.kind == "toy" else ad.blocks[0].ln_2
    rec = {k: [] for k in ["signal", "own", "prefix", "total", "check", "mlp_in", "key2"]}
    u = None
    for s in range(0, N_SEQ, 8):
        idx = batch_from_offsets(offs[s:s + 8], L, dev)
        p = ad.pre(idx)
        x1 = p["x1"]
        y = ov_images(x1, hw1).double()                                 # [B,H,T,d] OV^h x_j
        if u is None:
            u = y[0, :, 0].sum(0)                                        # w_BOS (summed heads)
            uu = u @ u
            wh = (y[0, :, 0] @ u) / uu                                   # per-head <w^h_BOS,u>/|u|^2
            const = float((attn_bias_vector(hw1).double() @ u) / uu)
        A = p["attn1_w"].double()                                        # [B,H,T,T]
        proj = (y @ u) / uu                                              # [B,H,T]
        signal = (A[..., 0] * wh[None, :, None]).sum(1)
        prefix = torch.einsum("bhij,bhj->bi", A[..., 1:], proj[..., 1:])
        emb = (p["post1"] - p["o1"]).double()
        own = (emb @ u) / uu
        total = (p["post1"].double() @ u) / uu
        rec["signal"].append(signal.cpu()); rec["own"].append(own.cpu()); rec["prefix"].append(prefix.cpu())
        rec["total"].append(total.cpu()); rec["check"].append((total - signal - own - prefix - const).abs().max().cpu()[None])
        mi = ln(p["post1"], ln_pre).double()                              # MLP input
        rec["mlp_in"].append(mi.cpu()); rec["key2"].append(p["x2"].double().cpu())
    R = {k: torch.cat(v) for k, v in rec.items() if k not in ("mlp_in", "key2", "check")}
    check = float(torch.cat(rec["check"]).max())
    # BOS amount at the MLP input and at the Layer-2 input (centered, own BOS direction)
    out = {}
    for tag in ["mlp_in", "key2"]:
        X = torch.cat(rec[tag])                                           # [n,T,d]
        mu = X[:, 1:].mean((0, 1)); v = X[0, 0] - mu
        out[tag] = (((X - mu) @ v) / (v @ v)).numpy()
        del X
    return L, R, check, const, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    res = {}
    for key in args.models:
        L, R, check, const, amt = run(key, args.device)
        pos = [i for i in POS if i < L]
        sig, own, pre, tot = (R[k].numpy() for k in ["signal", "own", "prefix", "total"])
        rows = {}
        for i in pos:
            Xn = np.stack([own[:, i], pre[:, i], np.ones(len(own))], 1)
            r2 = {}
            for tag in ["mlp_in", "key2"]:
                yv = amt[tag][:, i]
                c, *_ = np.linalg.lstsq(Xn, yv, rcond=None)
                r2[tag] = float(1 - np.var(yv - Xn @ c) / np.var(yv))
            rows[str(i)] = {"signal_mean": float(sig[:, i].mean()), "signal_sd": float(sig[:, i].std()),
                            "own_sd": float(own[:, i].std()), "prefix_sd": float(pre[:, i].std()),
                            "total_sd": float(tot[:, i].std()),
                            "mlp_in_amount_mean": float(amt["mlp_in"][:, i].mean()), "mlp_in_amount_sd": float(amt["mlp_in"][:, i].std()),
                            "key2_amount_mean": float(amt["key2"][:, i].mean()), "key2_amount_sd": float(amt["key2"][:, i].std()),
                            "r2_noise_terms_mlp_in": r2["mlp_in"], "r2_noise_terms_key2": r2["key2"]}
        res[key] = {"exact_split_max_err": check, "const": const, "by_position": rows}
        print(f"== {key}  (split error {check:.1e})")
        print("   i  | signal mean (sd) | own sd | prefix sd | total sd || MLP-in amount mean, sd, R2(noise) || L2-key amount mean, sd, R2(noise)")
        for i, r in rows.items():
            print(f"  {i:>4} | {r['signal_mean']:.4f} ({r['signal_sd']:.4f}) | {r['own_sd']:.4f} | {r['prefix_sd']:.4f} | {r['total_sd']:.4f} || "
                  f"{r['mlp_in_amount_mean']:.4f}, {r['mlp_in_amount_sd']:.4f}, {r['r2_noise_terms_mlp_in']:.2f} || "
                  f"{r['key2_amount_mean']:.4f}, {r['key2_amount_sd']:.4f}, {r['r2_noise_terms_key2']:.2f}", flush=True)
    od = RESULTS_ROOT / "sweep"; od.mkdir(parents=True, exist_ok=True)
    (od / "l1_noise_terms.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
