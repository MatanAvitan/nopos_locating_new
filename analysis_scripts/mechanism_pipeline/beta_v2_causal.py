"""BOS-share spec phase 2, causal tests: N7 (a)-(d) and threshold-neuron
zeroing. Edits touch the Layer-2 key and/or value input only (all heads),
except the neuron arms, which zero Layer-1 MLP neurons at one position and
recompute everything after them. Needs beta_v2.py outputs (ref.pt).

Reported per arm: frozen and refit errors by bin, the free fit
1/alpha_BOS = a' + c n_ord on each BOS head, i* = a'/c, the position of the
minimum of the refit relative-error curve, and the LM loss change.

Usage: python beta_v2_causal.py --model 1h --device cuda:1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, Ridge, attn_from_inputs, bins_for, err_by_bin,
                         ln, make_adapter, out_dir, spec_splits)
from beta_v2 import free_fit, mlp1_of, n_ordinary
from common import batch_from_offsets

DELTAS = [-0.1, 0.05, 0.1, 0.2]
SIGMAS = [0.001, 0.003, 0.01, 0.03, 0.08]
N_RAND = 5
N_ZERO = 16
NEURON_COUNTS = [16, 64, 256]


def build(R, L, d, dev):
    mu, xb = R["mu"]["x2"].to(dev).float(), R["bos"]["x2"].to(dev).float()
    v = xb - mu
    sinks = R["sinks2"]
    ordinary = torch.ones(L, device=dev); ordinary[0] = 0
    for s in sinks:
        ordinary[s] = 0
    bbar = float(R["beta_bar2"])
    g = torch.Generator(device="cpu").manual_seed(0)
    rand = []
    for _ in range(N_RAND):
        r = torch.randn(d, generator=g).to(dev)
        rand.append(r / r.norm() * v.norm())

    def cs(x, u):
        return ((x - mu) @ u) / (u @ u)

    def shift(delta, u):
        return lambda x, gen: x + delta * ordinary[None, :, None] * u

    def sink_off(u):
        def fn(x, gen):
            if not sinks:
                return x
            x = x.clone()
            for s in sinks:
                x[:, s] = x[:, s] + (bbar - cs(x[:, s], u))[:, None] * u
            return x
        return fn

    def noise(sigma, u):
        def fn(x, gen):
            eps = torch.randn(x.shape[0], x.shape[1], generator=gen).to(x.device) * sigma
            return x + (eps * ordinary[None])[..., None] * u
        return fn

    arms = {"clean": {}}
    for dl in DELTAS:
        arms[f"a_shift_{dl:+g}"] = {"k": shift(dl, v)}
    arms["b_sink_off_keys"] = {"k": sink_off(v)}
    arms["b_sink_off_values"] = {"v": sink_off(v)}
    arms["b_sink_off_both"] = {"k": sink_off(v), "v": sink_off(v)}
    for sg in SIGMAS:
        arms[f"c_noise_{sg:g}"] = {"k": noise(sg, v)}
    for r_i, r in enumerate(rand):
        arms[f"d_rand{r_i}_shift_+0.2"] = {"k": shift(0.2, r)}
        arms[f"d_rand{r_i}_sink_off_keys"] = {"k": sink_off(r)}
        arms[f"d_rand{r_i}_noise_0.08"] = {"k": noise(0.08, r)}
    arms["n_zero_neurons_pos1"] = {"zero": 1}
    arms["n_zero_neurons_pos0"] = {"zero": 0}
    return arms


@torch.no_grad()
def layer2_from_h1(ad, h1, arm, gen):
    blk2_ln = ad.model.block2.ln_1 if ad.kind == "toy" else ad.blocks[1].ln_1
    x2 = ln(h1, blk2_ln)
    ins = {r: (arm[r](x2, gen) if r in arm else x2) for r in "qkv"}
    return attn_from_inputs(ad.attn2, ins["q"], ins["k"], ins["v"])


@torch.no_grad()
def zeroed_h1(ad, p, neurons, pos):
    mlp = mlp1_of(ad)
    hid = mlp.gelu(mlp.c_fc(p["h1bar"]))              # the model's own activation
    hid[:, pos, neurons] = 0
    return p["post1"] + mlp.c_proj(hid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neuron-sweep", action="store_true",
                    help="only zero the top-k threshold neurons at positions 0 or 1, "
                         "for k in NEURON_COUNTS (ranked on the full MLP width)")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    od = out_dir(run_id, "beta_v2")
    R = torch.load(od / "ref.pt", weights_only=False)
    S0 = json.loads((od / "summary.json").read_text())
    heads = S0["bos_heads"]
    neurons = torch.as_tensor(R["neurons"][:N_ZERO], device=dev)
    arms = build(R, L, ad.d, dev)
    if args.neuron_sweep:
        A = R["hid_mean"].double()
        score = torch.minimum(A[0], A[1]) - A[2:].abs().max(0).values
        order = torch.argsort(score, descending=True)
        n_pos = int((score > 0).sum())
        arms = {"clean": {}}
        for k in NEURON_COUNTS + [n_pos]:
            for pos in (0, 1):
                arms[f"n_zero_top{k}_pos{pos}"] = {"zero": pos, "neurons": order[:k].to(dev)}
        print("neurons with positive score:", n_pos)
    splits = spec_splits(L)
    if ad.kind == "lm":
        ad.fit_frozen(splits["calibration"], L, args.batch_size)
    bins = bins_for(L)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)

    def run(split, seed):
        gen = torch.Generator().manual_seed(seed)
        offs = splits[split]
        for s in range(0, len(offs), args.batch_size):
            idx = batch_from_offsets(offs[s:s + args.batch_size], L, dev)
            p = ad.pre(idx)
            for a, arm in arms.items():
                h1 = zeroed_h1(ad, p, arm.get("neurons", neurons), arm["zero"]) if "zero" in arm else p["h1"]
                w, o2 = layer2_from_h1(ad, h1, arm, gen)
                feats, fr, ls = ad.finish(h1, o2, idx if (ad.kind == "lm" and split == "evaluation") else None)
                yield a, idx, w, feats, fr, ls

    ridges = {a: Ridge(ad.d, dev) for a in arms}
    for a, idx, w, feats, fr, ls in run("calibration", 1):
        ridges[a].add(feats, tgt.expand(idx.shape[0], L))
    for r in ridges.values():
        r.solve()
    fz = {a: [] for a in arms}; rf = {a: [] for a in arms}; lo = {a: [] for a in arms}
    ab = {a: torch.zeros(ad.H, L, dtype=torch.float64) for a in arms}
    n = {a: 0 for a in arms}
    for a, idx, w, feats, fr, ls in run("evaluation", 2):
        fz[a].append(fr.double().cpu()); rf[a].append(ridges[a].predict(feats).cpu())
        ab[a] += w[..., 0].double().sum(0).cpu(); n[a] += idx.shape[0]
        if ls is not None:
            lo[a].append(ls.double().cpu())
    nord = n_ordinary(L, R["sinks2"])
    i = np.arange(L)
    S = {"heads": heads, "sinks": R["sinks2"], "neurons_zeroed": R["neurons"][:N_ZERO].tolist(), "arms": {}}
    for a in arms:
        pf, pr = torch.cat(fz[a]).numpy(), torch.cat(rf[a]).numpy()
        am = (ab[a] / n[a]).numpy()
        rel = np.median(np.abs(pr - i[None]) / np.maximum(i[None], 1), 0)
        m = i >= 4
        fits = {}
        for h in heads:
            a_f, c_f = free_fit(am[h], nord)
            fits[f"h{h}"] = {"a_prime": a_f, "c": c_f, "i_star": a_f / c_f if c_f > 0 else None}
        e = {"frozen": err_by_bin(pf, L, bins), "refit": err_by_bin(pr, L, bins), "fit": fits,
             "refit_relerr_argmin": int(i[m][np.argmin(rel[m])]),
             "refit_relerr_at": {str(k): float(rel[k]) for k in [4, 16, 64, 256, 1023, 4096, 8191] if k < L}}
        if lo[a]:
            e["lm_loss"] = float(torch.cat(lo[a]).mean())
        S["arms"][a] = e
    if "lm_loss" in S["arms"]["clean"]:
        for a in arms:
            S["arms"][a]["lm_loss_delta"] = S["arms"][a]["lm_loss"] - S["arms"]["clean"]["lm_loss"]
    (od / ("causal_neuron_sweep.json" if args.neuron_sweep else "causal.json")).write_text(json.dumps(S, indent=2))
    late = "late" if "late" in bins else "middle"
    for a, e in S["arms"].items():
        f0 = e["fit"][f"h{heads[0]}"]
        print(f"{a:28s} frozen {late}={e['frozen'][f'relmae_{late}']:.4f} refit {late}={e['refit'][f'relmae_{late}']:.4f} "
              f"a'={f0['a_prime']:.3f} c={f0['c']:.5f} i*={f0['i_star'] if f0['i_star'] is None else round(f0['i_star'], 1)} "
              f"argmin={e['refit_relerr_argmin']}" + (f" dloss={e['lm_loss_delta']:+.4f}" if "lm_loss_delta" in e else ""))


if __name__ == "__main__":
    main()
