"""Signal and noise of the decoded scalar in the hand-set constructions.

For every variant, on the same evaluation sequences, take the exact scalar z_i
that goes into the isotonic decoder and measure, per position i:
  noise    N_i = sd over sequences of z_i
  signal   S_i = i |d E[z_i] / d i|   (smoothed mean curve, degree-6 poly in log i)
  ratio    N_i / S_i   (local relative position error)

Variants:
  layer1_readout   the Layer-1 attention output along its write direction:
                   z_i = mean of v_k over k = 0..i (uniform attention, weight
                   1/(i+1) on every key); gain only scales z, so it cancels
  one_layer        the Layer-2 head applied to the LayerNorm'd embeddings
                   (scores B x key amount; z = attention-weighted key amount)
  no_mlp1, full    the two-layer constructions (without / with the Layer-1 MLP)

For the attention variants the scores are exactly B a_j (constant query,
key = BOS amount a_j), so z_i = sum_j alpha_ij a_j with alpha = softmax(B a).
Split z_i - z_i^clean exactly into
  score part  sum_j (alpha_ij - alpha~_ij) abar_j
  value part  sum_j alpha_ij (a_j - abar_j)
where abar_j is the mean key amount at position j over sequences and
alpha~ = softmax(B abar) (the noise-free attention). Report the sd of each part,
the sd of their sum, and their correlation, per position bin.

Usage: python beta_build_noise.py --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_build import L, N_EVAL, build, calibrate
from beta_common import RESULTS_ROOT, bins_for, ln, smooth_deriv, spec_splits
from common import batch_from_offsets

CONFIGS = [("full", 2.0, 5.0), ("full", 2.0, 7.0), ("full", 4.0, 6.0),
           ("no_mlp1", 2.0, 5.0), ("no_mlp1", 2.0, 7.0),
           ("one_layer", 2.0, 5.0), ("one_layer", 2.0, 7.0)]


@torch.no_grad()
def collect(variant, gain, B, dev):
    splits = spec_splits(L)
    m, e, w, _ = build(gain, B, variant, dev)
    cal = calibrate(m, e, w, B, variant, splits["reference"], dev)
    a2 = m.block2.attn
    k = a2.c_attn.weight[768].double(); kb = a2.c_attn.bias[768].double()
    b_hat = m.block1.attn.c_attn.weight[2 * 768].double()            # Layer-1 value row
    amts, l1 = [], []
    for s in range(0, N_EVAL, 8):
        idx = batch_from_offsets(splits["evaluation"][s:s + 8], L, dev)
        x2 = cal["x2_of"](idx).double()
        amts.append((x2 @ k + kb).cpu())
        v = ln(m.wte(idx), m.block1.ln_1).double() @ b_hat           # Layer-1 value numbers
        l1.append((torch.cumsum(v, 1) / torch.arange(1, L + 1, device=dev)).cpu())
    return torch.cat(amts).numpy(), torch.cat(l1).numpy()


def attn_readout(a, B):
    """z_i = sum_{j<=i} e^{B a_j} a_j / sum_{j<=i} e^{B a_j}, per sequence."""
    lg = B * a
    mx = lg.max(1, keepdims=True)
    e = np.exp(lg - mx)
    return np.cumsum(e * a, 1) / np.cumsum(e, 1), e


def ratio_stats(z, bins):
    i = np.arange(L, dtype=np.float64)
    f = z.mean(0)
    _, der = smooth_deriv(f, L)
    noise = z.std(0)
    signal = i * np.abs(der)
    r = noise / signal
    out = {}
    for name, (lo, hi) in bins.items():
        lo = max(lo, 4)
        out[name] = {"noise": float(np.median(noise[lo:hi + 1])), "signal": float(np.median(signal[lo:hi + 1])),
                     "ratio": float(np.median(r[lo:hi + 1]))}
    m = i >= 16
    for key, arr in [("noise", noise), ("signal", signal), ("ratio", r)]:
        ok = m & np.isfinite(arr) & (arr > 0)
        out[f"loglog_slope_{key}_16_end"] = float(np.polyfit(np.log(i[ok]), np.log(arr[ok]), 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    bins = bins_for(L)
    res = {}
    for variant, gain, B in CONFIGS:
        a, l1 = collect(variant, gain, B, args.device)
        key = f"{variant}_g{gain:g}_B{B:g}"
        z, e = attn_readout(a, B)
        abar = a.mean(0, keepdims=True)
        zc, ec = attn_readout(abar, B)                               # noise-free output
        val = np.cumsum(e * (a - abar), 1) / np.cumsum(e, 1)         # value part
        score = z - zc - val                                         # score part (exact remainder)
        split = {}
        for name, (lo, hi) in bins.items():
            sl = slice(max(lo, 4), hi + 1)
            sd_s, sd_v, sd_t = score[:, sl].std(0), val[:, sl].std(0), (z - zc)[:, sl].std(0)
            cs = np.mean([np.corrcoef(score[:, j], val[:, j])[0, 1] for j in range(sl.start, sl.stop, 7)])
            split[name] = {"sd_score_part": float(np.median(sd_s)), "sd_value_part": float(np.median(sd_v)),
                           "sd_total": float(np.median(sd_t)), "corr_score_value": float(cs)}
        res[key] = {"decoded_scalar": ratio_stats(z, bins), "split": split}
        if variant == "full" and gain == 2.0 and B == 5.0:
            res["layer1_readout"] = {"decoded_scalar": ratio_stats(l1, bins)}
        print(key, json.dumps(res[key]["decoded_scalar"]), flush=True)
        print("   split", json.dumps(split), flush=True)
    print("layer1_readout", json.dumps(res["layer1_readout"]["decoded_scalar"]))
    od = RESULTS_ROOT / "construction"
    (od / "noise_signal.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
