"""LayerNorm as the source of the erasure cap in the Layer-1 MLP.

Notation (residual space before the Layer-1 MLP, v = post1_BOS - mu_post1):
  s_j = <post1_j - mu, v>/|v|^2   BOS amount entering the MLP
  c_j = <m_j - mu_m, v>/|v|^2     BOS the MLP writes (negative = erased)
Model: the MLP reads BOS after the pre-MLP LayerNorm, which divides by the
token's own norm, so
  c(s) = -G rho s / sqrt(s^2 + rho^2) + c0,   rho = |content| / |v|.

1. Per-token fit of (G, rho, c0); compare rho with the measured ratio
   |content_j| / |v| on late tokens (both after per-token mean removal).
2. Causal: scale BOS's Layer-1 value by lambda before the MLP. Token j gets
   post1_j + (lambda - 1) alpha1_{j,BOS} w1_BOS (w1_BOS = Layer-1 OV image of
   x_BOS, summed over heads; BOS itself has alpha = 1). Refit the cap G rho
   (in units of the arm's own BOS vector) and report beta_1, beta_2, sinks,
   and the refit late error. Prediction: cap ~ 1/lambda.
3. Sufficiency: replace the MLP's BOS component by the rank-1 map
   c_hat_j = -G sigma_tail (r_j - r_tail),  r_j = <(post1_j - mean_j)/sigma_j, v_c>/|v_c|^2
   (v_c = v minus its mean over dimensions, sigma_j the per-token scale),
   G fitted by regression; compare with the trained MLP and ideal cancel.

Usage: python beta_mlp1_ln_cap.py --model pv_unifL1_s42 --device cuda:0
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
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, Ridge, attn_from_inputs, bins_for, err_by_bin,
                         ln, make_adapter, out_dir, spec_splits)
from common import batch_from_offsets, head_weights

LAMBDAS = [0.5, 0.75, 1.0, 1.5, 2.0]
N_EVAL = 400
J_FIT = 64


def cap_curve(s, G, rho, c0):
    return -G * rho * s / np.sqrt(s ** 2 + rho ** 2) + c0


def fit_cap(s, c):
    p, _ = curve_fit(cap_curve, s, c, p0=[1.0, 0.25, 0.0], maxfev=20000)
    pred = cap_curve(s, *p)
    return {"G": float(p[0]), "rho": float(p[1]), "c0": float(p[2]), "cap_G_rho": float(p[0] * p[1]),
            "r2": float(1 - np.var(c - pred) / np.var(c))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    od = out_dir(run_id, "beta_v2"); fd = od / "figures"
    b1 = ad.model.block1
    mlp, ln_pre, blk2_ln = b1.mlp, b1.ln_2, ad.model.block2.ln_1
    splits = spec_splits(L); bins = bins_for(L)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)
    hw1 = head_weights(ad.attn1)

    def batches(name, n=None):
        offs = splits[name][:n] if n else splits[name]
        for s in range(0, len(offs), args.batch_size):
            yield batch_from_offsets(offs[s:s + args.batch_size], L, dev)

    def post1_arm(p, lam):
        if lam == 1.0:
            return p["post1"]
        x0 = p["x1"][0, 0].double()
        w1 = torch.einsum("hde,he->hd", hw1.W_o.double(), torch.einsum("hed,d->he", hw1.W_v.double(), x0))  # [H,d]
        add = torch.einsum("bht,hd->btd", p["attn1_w"][..., 0].double(), w1)                                # alpha * w
        return (p["post1"].double() + (lam - 1) * add).float()

    def mlp_out(post):
        return mlp(ln(post, ln_pre))

    # ---- reference per lambda: mu of post1, mu of MLP output, BOS vector --------
    ref = {}
    for lam in LAMBDAS:
        sp = torch.zeros(ad.d, device=dev, dtype=torch.float64); sm = torch.zeros_like(sp); n = 0; bos = None
        for idx in batches("reference"):
            p = ad.pre(idx)
            post = post1_arm(p, lam); m = mlp_out(post)
            bos = post[0, 0].double() if bos is None else bos
            sp += post[:, 1:].double().sum((0, 1)); sm += m[:, 1:].double().sum((0, 1)); n += idx.shape[0] * (L - 1)
        mu, mum = sp / n, sm / n
        ref[lam] = {"mu": mu, "mum": mum, "v": bos - mu}
    R0 = ref[1.0]
    v = R0["v"]; vv = v @ v
    vc = v - v.mean(); vcc = vc @ vc

    # ---- per-token data for the fits (clean and each lambda) --------------------
    data = {lam: ([], []) for lam in LAMBDAS}
    rho_meas, rr, cc, sig_tail = [], [], [], []
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        for lam in LAMBDAS:
            post = post1_arm(p, lam).double(); m = mlp_out(post.float()).double()
            vl = ref[lam]["v"]
            s = ((post - ref[lam]["mu"]) @ vl) / (vl @ vl)
            c = ((m - ref[lam]["mum"]) @ vl) / (vl @ vl)
            data[lam][0].append(s[:, 1:J_FIT + 1].reshape(-1).cpu()); data[lam][1].append(c[:, 1:J_FIT + 1].reshape(-1).cpu())
            if lam == 1.0:
                cen = post - post.mean(-1, keepdim=True)
                sig = cen.norm(dim=-1) / np.sqrt(ad.d)
                content = cen - s[..., None] * vc                     # residual minus its BOS component
                rho_meas.append((content[:, 128:].norm(dim=-1) / vc.norm()).reshape(-1).cpu())
                r = (cen / sig[..., None]) @ vc / vcc
                rr.append(r[:, 1:].cpu()); cc.append(c[:, 1:].cpu()); sig_tail.append(sig[:, 128:].reshape(-1).cpu())
    S = {"model": args.model}
    s1, c1 = torch.cat(data[1.0][0]).numpy(), torch.cat(data[1.0][1]).numpy()
    S["fit_clean"] = fit_cap(s1, c1)
    S["rho_measured_late_tokens"] = float(torch.cat(rho_meas).mean())
    S["lambda_sweep_fit"] = {str(l): fit_cap(torch.cat(data[l][0]).numpy(), torch.cat(data[l][1]).numpy()) for l in LAMBDAS}

    # rank-1 map: c_hat = -G sigma_tail (r - r_tail); fit G by regression on all tokens
    r_all, c_all = torch.cat(rr).numpy(), torch.cat(cc).numpy()
    sigma_tail = float(torch.cat(sig_tail).mean())
    r_tail = float(r_all[:, 127:].mean())
    xfeat = -sigma_tail * (r_all - r_tail)
    G_fit = float((xfeat * c_all).sum() / (xfeat * xfeat).sum())
    S["rank1"] = {"G": G_fit, "sigma_tail": sigma_tail, "r_tail": r_tail,
                  "r2_per_token": float(1 - np.var(c_all - G_fit * xfeat) / np.var(c_all))}

    # ---- causal arms --------------------------------------------------------
    @torch.no_grad()
    def h1_arm(p, arm):
        kind, lam = arm
        post = post1_arm(p, lam)
        m = mlp_out(post)
        if kind in ("rank1", "ideal"):
            postd, md = post.double(), m.double()
            s = ((postd - R0["mu"]) @ v) / vv
            c = ((md - R0["mum"]) @ v) / vv
            if kind == "ideal":
                target = -s
            else:
                cen = postd - postd.mean(-1, keepdim=True)
                sig = cen.norm(dim=-1) / np.sqrt(ad.d)
                r = (cen / sig[..., None]) @ vc / vcc
                target = -G_fit * sigma_tail * (r - r_tail)
            corr = target - c
            corr[:, 0] = 0
            m = (md + corr[..., None] * v).float()
        return post + m

    arms = {f"lambda_{l:g}": ("mlp", l) for l in LAMBDAS}
    arms["rank1_map"] = ("rank1", 1.0); arms["ideal_cancel"] = ("ideal", 1.0)
    ridges = {k: Ridge(ad.d, dev) for k in arms}
    x2s = {k: torch.zeros(ad.d, device=dev, dtype=torch.float64) for k in arms}; x2b = {}; ncal = 0
    for idx in batches("calibration"):
        p = ad.pre(idx)
        for k, arm in arms.items():
            h1 = h1_arm(p, arm); x2 = ln(h1, blk2_ln)
            x2s[k] += x2[:, 1:].double().sum((0, 1)); x2b.setdefault(k, x2[0, 0].double().clone())
            _, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, _, _ = ad.finish(h1, o2)
            ridges[k].add(feats, tgt.expand(idx.shape[0], L))
        ncal += idx.shape[0] * (L - 1)
    for r_ in ridges.values():
        r_.solve()
    mu2 = {k: x2s[k] / ncal for k in arms}; v2 = {k: x2b[k] - mu2[k] for k in arms}
    out = {k: {"beta": [], "refit": [], "ab": torch.zeros(L, dtype=torch.float64)} for k in arms}
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        for k, arm in arms.items():
            h1 = h1_arm(p, arm); x2 = ln(h1, blk2_ln)
            w, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, _, _ = ad.finish(h1, o2)
            out[k]["beta"].append((((x2.double() - mu2[k]) @ v2[k]) / (v2[k] @ v2[k])).cpu())
            out[k]["refit"].append(ridges[k].predict(feats).cpu())
            out[k]["ab"] += w[:, 0, :, 0].double().sum(0).cpu()
    S["causal"] = {}
    for k in arms:
        B_ = torch.cat(out[k]["beta"]).numpy()
        frac = (B_[:, 1:] > 0.5).mean(0)
        sinks = [int(j) + 1 for j in np.nonzero(frac > 0.9)[0]]
        am = (out[k]["ab"] / len(B_)).numpy()
        S["causal"][k] = {"beta_profile": {str(j): float(B_[:, j].mean()) for j in [1, 2, 3, 4, 6, 8, 16, 64] if j < L},
                          "sinks": sinks, "tail_sd_j16on": float(B_[:, 16:].std()),
                          "alpha_bos_at": {str(j): float(am[j]) for j in [1, 16, 128, 1023] if j < L},
                          "refit": err_by_bin(torch.cat(out[k]["refit"]).numpy(), L, bins)}
    (od / "mlp1_ln_cap.json").write_text(json.dumps(S, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ss = np.linspace(0, max(1.0, s1.max()), 200)
    for l in LAMBDAS:
        sl, cl = torch.cat(data[l][0]).numpy(), torch.cat(data[l][1]).numpy()
        f = S["lambda_sweep_fit"][str(l)]
        pts = ax[0].scatter(sl[::7], cl[::7], s=1, alpha=.25)
        ax[0].plot(ss, cap_curve(ss, f["G"], f["rho"], f["c0"]), color=pts.get_facecolor()[0][:3], lw=1.2,
                   label=f"λ={l:g}: cap Gρ={f['cap_G_rho']:.3f}, ρ={f['rho']:.3f}")
    ax[0].set_xlabel("s (BOS entering the MLP, own BOS units)"); ax[0].set_ylabel("c (BOS written by the MLP)")
    ax[0].legend(fontsize=7); ax[0].set_title(r"per-token fit $c=-G\rho\,s/\sqrt{s^2+\rho^2}+c_0$")
    for k in ["lambda_1", "rank1_map", "ideal_cancel"] + [f"lambda_{l:g}" for l in LAMBDAS if l != 1.0]:
        prof = S["causal"][k]["beta_profile"]
        ax[1].plot(list(map(int, prof)), list(prof.values()), marker="o", ms=3, label=k)
    ax[1].set_xscale("symlog", linthresh=4); ax[1].axhline(0, color="k", lw=.4); ax[1].legend(fontsize=7)
    ax[1].set_title(r"$\beta_j$ at the Layer-2 input")
    fig.tight_layout(); fig.savefig(fd / "mlp1_ln_cap.png", dpi=120); plt.close(fig)

    print("fit clean:", {k: round(x, 4) for k, x in S["fit_clean"].items()}, "| rho measured:", round(S["rho_measured_late_tokens"], 4))
    for l, f in S["lambda_sweep_fit"].items():
        print(f"  lambda {l}: cap={f['cap_G_rho']:.4f} rho={f['rho']:.4f} G={f['G']:.3f} r2={f['r2']:.3f}")
    print("rank1:", {k: round(x, 4) for k, x in S["rank1"].items()})
    for k, d_ in S["causal"].items():
        print(f"{k:14s}", {j: round(x, 3) for j, x in d_["beta_profile"].items()}, "sinks", d_["sinks"][:4],
              "tail sd %.4f" % d_["tail_sd_j16on"], "refit late %.4f" % d_["refit"].get("relmae_late", np.nan))


if __name__ == "__main__":
    main()
