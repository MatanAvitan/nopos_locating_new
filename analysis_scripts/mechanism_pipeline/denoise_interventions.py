"""Causal interventions for the denoising view (paper section sec:denoise_evidence).

Each arm edits one component and recomputes everything downstream. Position
error is the MAE in tokens of the frozen position head (1H, 12H) or of the
frozen Layer-2 probe (LM, a ridge probe on the clean Layer-2 residual fitted
on calibration sequences), by bin. For the LM we also report the change in
next-token loss.

Notation: s_j = BOS amount entering the Layer-1 MLP along v = post1_BOS - mu,
beta_j = centered BOS share at the Layer-2 input (direction d = x_BOS - mu2).

Mechanism-preserving arms (idealized replacements):
  mlp_rank1      the Layer-1 MLP's BOS component replaced by the double
                 projection on v through LayerNorm: c_j = -G sigma_tail (r_j - r_tail)
  keys_ideal     each BOS head's attention replaced by softmax_j(B beta_j)
  values_ideal   each BOS head's value projection on its BOS-value direction set
                 to c + lambda beta_j (the rest of the value is kept)
  keys_values    both of the above
  random_key     the key input of every key j >= 1 shifted by 0.2 along a random
                 direction with the norm of d (control)
Ablations:
  mlp_off_pos2on   the Layer-1 MLP output set to zero from position 2 on
  removal_neurons  the 128 Layer-1 MLP neurons that subtract the most BOS set
                   to zero at positions >= 1
  key_shift        the key input of every key j >= 1 shifted by 0.2 d
                   (every share rises by 0.2)
  sink_off         the key input of every BOS-salient key (share > 0.5 in 90% of
                   sequences, BOS excluded) set to the ordinary share

Usage: python denoise_interventions.py --model 1h --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, RESULTS_ROOT, attn_from_inputs, bins_for, err_by_bin,
                         ln, make_adapter, spec_splits)
from common import batch_from_offsets, head_weights, ov_images
from softmax_beta_check import ml_B

N_EVAL, N_CAL, N_REF = 1600, 160, 160
TOPK = 128


def blocks_of(ad):
    if ad.kind == "toy":
        m = ad.model
        return m.block1.mlp, m.block1.ln_2, m.block2.ln_1
    return ad.blocks[0].mlp, ad.blocks[0].ln_2, ad.blocks[1].ln_1


@torch.no_grad()
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
    sp = spec_splits(L); bins = bins_for(L)
    mlp, ln_pre, ln2 = blocks_of(ad)
    hw2 = head_weights(ad.attn2)
    if ad.kind == "lm":
        ad.fit_frozen(sp["calibration"][:320], L, 32)

    def batches(name, n, start=0):
        offs = sp[name][start:start + n]
        for s in range(0, len(offs), bs):
            yield batch_from_offsets(offs[s:s + bs], L, dev)

    # ---- reference statistics -------------------------------------------------
    sp1 = torch.zeros(ad.d, device=dev, dtype=torch.float64); sm = torch.zeros_like(sp1); s2 = torch.zeros_like(sp1)
    n = 0; bos_post = xb2 = None
    for idx in batches("reference", N_REF):
        p = ad.pre(idx)
        m = mlp(p["h1bar"])
        if bos_post is None:
            bos_post, xb2 = p["post1"][0, 0].double(), p["x2"][0, 0].double()
        sp1 += p["post1"][:, 1:].double().sum((0, 1)); sm += m[:, 1:].double().sum((0, 1))
        s2 += p["x2"][:, 1:].double().sum((0, 1)); n += idx.shape[0] * (L - 1)
    mu1, mum, mu2 = sp1 / n, sm / n, s2 / n
    v = bos_post - mu1; vv = v @ v
    vc = v - v.mean(); vcc = vc @ vc
    d = xb2 - mu2; dd = d @ d
    g = torch.Generator(device="cpu").manual_seed(0)
    rnd = torch.randn(ad.d, generator=g, dtype=torch.float64).to(dev); rnd = rnd / rnd.norm() * d.norm()
    M2 = torch.einsum("hde,hef->hdf", hw2.W_o.double(), hw2.W_v.double())
    wb2 = torch.einsum("hdf,f->hd", M2, xb2)
    uv = wb2 / wb2.norm(dim=-1, keepdim=True)                                   # per-head BOS-value direction
    rV = torch.einsum("hdf,hd->hf", M2, uv)                                     # value read direction

    # ---- calibration: rank-1 MLP gain, removal neurons, B, value fits, sinks ----
    W_out = mlp.c_proj.weight.double()
    wk = (W_out.T @ v) / vv                                                     # BOS written per unit activation
    r_l, c_l, sig_l, s_l, a_l = [], [], [], [], []
    bet, Ar, vals = [], [], []
    for idx in batches("calibration", 64):
        p = ad.pre(idx)
        post = p["post1"].double()
        cen = post - post.mean(-1, keepdim=True)
        sig = cen.norm(dim=-1) / np.sqrt(ad.d)
        r_l.append(((cen / sig[..., None]) @ vc / vcc)[:, 1:].cpu())
        m = mlp(p["h1bar"]).double()
        c_l.append((((m - mum) @ v) / vv)[:, 1:].cpu()); sig_l.append(sig[:, min(128, L // 2):].reshape(-1).cpu())
        s_l.append((((post - mu1) @ v) / vv)[:, 5:17].reshape(-1).cpu())
        a_l.append(mlp.gelu(mlp.c_fc(p["h1bar"]))[:, 5:17].double().reshape(-1, W_out.shape[1]).cpu())
        x2 = p["x2"].double(); b = ((x2 - mu2) @ d) / dd
        bet.append(b.cpu()); Ar.append(p["attn2_w"][:, heads].double().cpu())
        vals.append(torch.einsum("btd,hd->bht", x2, rV[heads]).cpu())
    r_all, c_all = torch.cat(r_l).numpy(), torch.cat(c_l).numpy()
    late0 = min(128, L // 2)                                                    # typical ordinary tokens
    sigma_tail = float(torch.cat(sig_l).mean()); r_tail = float(r_all[:, late0 - 1:].mean())
    xf = -sigma_tail * (r_all - r_tail)
    G = float((xf * c_all).sum() / (xf * xf).sum())
    s_all = torch.cat(s_l).double(); a_all = torch.cat(a_l)
    sc = s_all - s_all.mean(); slope_a = ((a_all - a_all.mean(0)) * sc[:, None]).mean(0) / (sc * sc).mean()
    q = (wk.cpu() * slope_a).numpy()
    removal = torch.as_tensor(np.argsort(q)[:TOPK], device=dev)                 # most negative: subtract BOS
    bet = torch.cat(bet).numpy(); Ar = torch.cat(Ar).numpy(); vals = torch.cat(vals).numpy()
    Tm = min(L, 256)
    Bh = {h: ml_B(Ar[:, k, :Tm, :Tm], bet[:, :Tm], np.linspace(-20, 20, 801)) for k, h in enumerate(heads)}
    vfit = {}
    for k, h in enumerate(heads):
        lam, c0 = np.polyfit(bet[:, 1:].ravel(), vals[:, k, 1:].ravel(), 1)
        vfit[h] = (float(c0), float(lam))
    frac = (bet[:, 1:] > 0.5).mean(0)
    sinks = [int(j) + 1 for j in np.nonzero(frac > 0.9)[0]]
    ns = np.ones(L, bool); ns[:16] = False; ns[sinks] = False
    beta_ord = float(bet[:, ns].mean())

    # ---- arms -------------------------------------------------------------------
    def h1_of(p, arm):
        if arm in ("mlp_rank1", "mlp_off_pos2on", "removal_neurons"):
            post = p["post1"]
            if arm == "removal_neurons":
                hid = mlp.gelu(mlp.c_fc(p["h1bar"])).clone(); hid[:, 1:, removal] = 0
                return post + mlp.c_proj(hid)
            m = mlp(p["h1bar"])
            if arm == "mlp_off_pos2on":
                m = m.clone(); m[:, 2:] = 0
                return post + m
            postd, md = post.double(), m.double()
            cen = postd - postd.mean(-1, keepdim=True)
            sig = cen.norm(dim=-1) / np.sqrt(ad.d)
            r = (cen / sig[..., None]) @ vc / vcc
            c = ((md - mum) @ v) / vv
            corr = -G * sigma_tail * (r - r_tail) - c
            corr[:, 0] = 0
            return (postd + md + corr[..., None] * v).float()
        return p["h1"]

    def attn_of(p, h1, arm):
        x2 = ln(h1, ln2) if arm in ("mlp_rank1", "mlp_off_pos2on", "removal_neurons") else p["x2"]
        if arm in ("key_shift", "random_key", "sink_off"):
            k_in = x2.double().clone()
            if arm == "key_shift":
                k_in[:, 1:] += 0.2 * d
            elif arm == "random_key":
                k_in[:, 1:] += 0.2 * rnd
            else:
                if not sinks:
                    return None
                b = ((k_in - mu2) @ d) / dd
                for s_ in sinks:
                    k_in[:, s_] += (beta_ord - b[:, s_])[:, None] * d
            _, o2 = attn_from_inputs(ad.attn2, x2, k_in.float(), x2)
            return o2
        if arm in ("keys_ideal", "values_ideal", "keys_values"):
            x2d = x2.double(); beta = ((x2d - mu2) @ d) / dd
            A = p["attn2_w"].double().clone()
            y = ov_images(x2, hw2).double()
            T = beta.shape[1]
            if arm in ("keys_ideal", "keys_values"):
                mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
                for h in heads:
                    sc_ = (Bh[h] * beta)[:, None, :].expand(-1, T, -1).masked_fill(mask, float("-inf"))
                    A[:, h] = torch.softmax(sc_, -1)
            if arm in ("values_ideal", "keys_values"):
                for h in heads:
                    eps = x2d @ rV[h] - (vfit[h][0] + vfit[h][1] * beta)
                    y[:, h] = y[:, h] - eps[..., None] * uv[h]
            o2 = torch.einsum("bhij,bhjd->bid", A, y)
            o2 = o2 + torch.einsum("he,hde->d", hw2.b_v.double(), hw2.W_o.double())
            if ad.attn2.c_proj.bias is not None:
                o2 = o2 + ad.attn2.c_proj.bias.double()
            return o2.float()
        if arm in ("mlp_rank1", "mlp_off_pos2on", "removal_neurons"):
            _, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            return o2
        return p["o2"]

    arms = ["clean", "mlp_rank1", "keys_ideal", "values_ideal", "keys_values", "random_key",
            "mlp_off_pos2on", "removal_neurons", "key_shift", "sink_off"]
    preds = {a: [] for a in arms}; loss = {a: [] for a in arms}
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        for a in arms:
            h1 = h1_of(p, a)
            o2 = attn_of(p, h1, a)
            if o2 is None:
                continue
            _, fr, ls = ad.finish(h1, o2, idx if ad.kind == "lm" else None)
            preds[a].append(fr.double().cpu())
            if ls is not None:
                loss[a].append(float(ls.double().mean()))
    S = {"model": args.model, "bos_heads": heads, "sinks": sinks, "G_rank1": G, "B": Bh, "arms": {}}
    for a in arms:
        if not preds[a]:
            S["arms"][a] = None; continue
        e = err_by_bin(torch.cat(preds[a]).numpy(), L, bins)
        if loss[a]:
            e["loss"] = float(np.mean(loss[a]))
        S["arms"][a] = e
    if loss["clean"]:
        for a in arms:
            if S["arms"][a]:
                S["arms"][a]["loss_increase"] = S["arms"][a]["loss"] - S["arms"]["clean"]["loss"]
    od = RESULTS_ROOT / "story_evidence" / args.model; od.mkdir(parents=True, exist_ok=True)
    (od / "denoise_interventions.json").write_text(json.dumps(S, indent=2))
    print(f"== {args.model}: heads {heads} sinks {sinks} G {G:.3f}")
    for a, e in S["arms"].items():
        if e is None:
            print(f"  {a:16s} n/a"); continue
        print(f"  {a:16s} MAE " + " ".join(f"{b}={e[f'mae_{b}']:.2f}" for b in bins) + f" all={e['mae_all']:.2f}"
              + (f"  dloss {e['loss_increase']:+.4f}" if "loss_increase" in e else ""))


if __name__ == "__main__":
    main()
