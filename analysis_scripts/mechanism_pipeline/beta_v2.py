"""BOS-share spec, phase 2 (sinks, pull, transition point): N1-N4, N6, and
the Layer-1 threshold-neuron checks. Outputs go to
results/mechanism/<run_id>/beta_v2/.

Definitions (see beta_spec.md, "Phase 2"):
  centered share  beta_j = <x_j - mu, x_BOS - mu> / ||x_BOS - mu||^2,
                  mu = mean non-BOS input of that layer (reference split)
  sinks           keys j >= 1 with beta_j > 0.5 in more than 90% of reference
                  sequences; a = 1 + number of sinks (BOS counts)
  beta_bar        mean centered share over non-sink keys with j >= 16
  B               median per-query OLS slope of the scores on the centered
                  share, queries i >= 16 (reference split)
  schedule        alpha_BOS(i) = e^B / (a e^B + n_i e^{B beta_bar}),
                  n_i = number of ordinary keys in 0..i
  transition      i* = a e^{B (1 - beta_bar)}

Usage: python beta_v2.py --model 1h --device cuda:1 [--init]
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
from beta_common import (MODELS, attn_scores_weights, bins_for, make_adapter,
                         out_dir, smooth_deriv, spec_splits)
from beta_extract import batches, per_query_ols
from common import head_weights, ov_images

SINK_SHARE, SINK_FRAC = 0.5, 0.9
N_NEURONS = 16
STAGES = ["x1", "o1", "post1", "h1", "x2"]


def mlp1_of(ad):
    return ad.model.block1.mlp if ad.kind == "toy" else ad.blocks[0].mlp


def cshare(x, xb, mu):
    """Centered share. x [B,T,d], xb [d], mu [d] (all fp64)."""
    v = xb - mu
    return ((x - mu) @ v) / (v @ v)


def stage_tensors(p):
    return {"x1": p["x1"], "o1": p["o1"], "post1": p["post1"], "h1": p["h1"], "x2": p["x2"]}


@torch.no_grad()
def reference(ad, offsets, L, bs, dev):
    H = ad.H
    hw1, hw2 = head_weights(ad.attn1), head_weights(ad.attn2)
    mlp1 = mlp1_of(ad)
    sums = {s: torch.zeros(ad.d, device=dev, dtype=torch.float64) for s in STAGES}
    bos = {}
    n = 0
    hid_sum = None
    for _, idx in batches(offsets, L, bs, dev):
        p = ad.pre(idx)
        st = stage_tensors(p)
        for s, v in st.items():
            v = v.double()
            if s not in bos:
                bos[s] = v[0, 0].clone()
            sums[s] += v[:, 1:].sum((0, 1))
        hid = mlp1.gelu(mlp1.c_fc(p["h1bar"]).float()).double()      # [B,T,4d]
        hid_sum = hid.sum(0) if hid_sum is None else hid_sum + hid.sum(0)
        n += idx.shape[0]
        del p, hid
    mu = {s: sums[s] / (n * (L - 1)) for s in STAGES}
    R = {"mu": mu, "bos": bos, "hid_mean": hid_sum / n}
    # second pass: sinks, B (layer 1 and 2), beta_bar
    b2_all, b1_all, sl2, sl1 = [], [], [[] for _ in range(H)], [[] for _ in range(ad.H)]
    for _, idx in batches(offsets, L, bs, dev):
        p = ad.pre(idx)
        b2 = cshare(p["x2"].double(), bos["x2"], mu["x2"])
        b1 = cshare(p["x1"].double(), bos["x1"], mu["x1"])
        b2_all.append(b2.cpu()); b1_all.append(b1.cpu())
        s2, _ = per_query_ols(p["scores2"].double(), b2)[:2]
        s1_scores, _ = attn_scores_weights(p["x1"], hw1)
        s1, _ = per_query_ols(s1_scores.double(), b1)[:2]
        for h in range(H):
            sl2[h].append(s2[:, h, 16:].reshape(-1).cpu())
            sl1[h].append(s1[:, h, 16:].reshape(-1).cpu())
        del p
    b2_all, b1_all = torch.cat(b2_all).numpy(), torch.cat(b1_all).numpy()
    for tag, ball in [("2", b2_all), ("1", b1_all)]:
        frac = (ball[:, 1:] > SINK_SHARE).mean(0)
        sinks = [int(j) + 1 for j in np.nonzero(frac > SINK_FRAC)[0]]
        ns = np.ones(L, bool); ns[0] = False; ns[sinks] = False; ns[:16] = False
        R[f"sinks{tag}"] = sinks
        R[f"a{tag}"] = 1 + len(sinks)
        R[f"beta_bar{tag}"] = float(ball[:, ns].mean())
        R[f"beta_ref{tag}"] = ball.mean(0)
    R["B2"] = np.array([float(torch.cat(s).nanmedian()) for s in sl2])
    R["B1"] = np.array([float(torch.cat(s).nanmedian()) for s in sl1])
    # threshold-neuron candidates: active at j = 0, 1 and silent from j = 2 on
    A = R["hid_mean"]                                             # [T,4d]
    late = A[2:].abs().max(0).values
    score = torch.minimum(A[0], A[1]) - late
    R["neurons"] = torch.argsort(score, descending=True)[:N_NEURONS].cpu().numpy()
    R["neuron_score"] = score[torch.as_tensor(R["neurons"], device=score.device)].cpu().numpy()
    return R


@torch.no_grad()
def evaluate(ad, R, old, offsets, L, bs, dev):
    H = ad.H
    hw1, hw2 = head_weights(ad.attn1), head_weights(ad.attn2)
    mlp1 = mlp1_of(ad)
    mu, bos = R["mu"], R["bos"]
    heads = old["bos_heads"]
    B2 = torch.tensor(R["B2"], device=dev, dtype=torch.float64)
    v2 = bos["x2"] - mu["x2"]
    kv2 = torch.einsum("hed,d->he", hw2.W_k.double(), v2)
    # layer-1 BOS value direction (summed heads) and layer-2 readout axes
    w1 = torch.einsum("hde,he->d", hw1.W_o.double(), torch.einsum("hed,d->he", hw1.W_v.double(), bos["x1"]))
    u1 = w1 / w1.norm()
    wb2 = old["w_bos"].to(dev).double()
    uw2 = wb2 / wb2.norm(dim=-1, keepdim=True)
    agg = wb2[heads].sum(0); uagg = agg / agg.norm()
    neurons = torch.as_tensor(R["neurons"], device=dev)
    keys = ["b_x1", "b_o1", "b_post1", "b_h1", "b_x2", "b_x2_uncentered",
            "norm_post1", "norm_h1", "norm_mlp1", "a1_bos", "x2_along_u1",
            "alpha_bos", "t", "slope", "r2", "ev", "zeta2", "zeta1",
            "z_agg", "neuron_pre", "pred"]
    out = {k: [] for k in keys}
    for _, idx in batches(offsets, L, bs, dev):
        p = ad.pre(idx)
        st = {k: v.double() for k, v in stage_tensors(p).items()}
        for s in STAGES:
            out[f"b_{s}"].append(cshare(st[s], bos[s], mu[s]).float().cpu())
        xb2 = bos["x2"]
        out["b_x2_uncentered"].append(((st["x2"] @ xb2) / (xb2 @ xb2)).float().cpu())
        out["norm_post1"].append(st["post1"].norm(dim=-1).float().cpu())
        out["norm_h1"].append(st["h1"].norm(dim=-1).float().cpu())
        out["norm_mlp1"].append((st["h1"] - st["post1"]).norm(dim=-1).float().cpu())
        out["a1_bos"].append(p["attn1_w"][..., 0].mean(1).float().cpu())
        out["x2_along_u1"].append(((st["x2"] - mu["x2"]) @ u1).float().cpu())
        pre = mlp1.c_fc(p["h1bar"]).float()[..., neurons]
        out["neuron_pre"].append(pre.cpu())
        b2 = cshare(st["x2"], bos["x2"], mu["x2"])
        A = p["attn2_w"].double()
        out["alpha_bos"].append(A[..., 0].float().cpu())
        out["t"].append(torch.einsum("bhij,bj->bhi", A, b2).float().cpu())
        s = p["scores2"].double()
        sl, r2, cov, varb, vars_ = per_query_ols(s, b2)
        q = torch.einsum("btd,hed->bhte", st["x2"], hw2.W_q.double()) + hw2.b_q.double()[None, :, None]
        Bi = torch.einsum("bhte,he->bht", q, kv2) * hw2.scale
        ev = 1 - (vars_ - 2 * Bi * cov + Bi ** 2 * varb) / vars_
        out["slope"].append(sl.float().cpu()); out["r2"].append(r2.float().cpu())
        out["ev"].append(ev.float().cpu())
        # content fluctuation along each layer's BOS value direction
        y2 = ov_images(p["x2"], hw2).double()
        xi2 = y2 - b2[:, None, :, None] * torch.einsum("hde,he->hd", hw2.W_o.double(),
                                                        torch.einsum("hed,d->he", hw2.W_v.double(), v2))[None, :, None, :]
        out["zeta2"].append(torch.einsum("bhtd,hd->bht", xi2, uw2).float().cpu())
        b1 = cshare(st["x1"], bos["x1"], mu["x1"])
        y1 = ov_images(p["x1"], hw1).double().sum(1)
        w1c = torch.einsum("hde,he->d", hw1.W_o.double(),
                           torch.einsum("hed,d->he", hw1.W_v.double(), bos["x1"] - mu["x1"]))
        out["zeta1"].append(((y1 - b1[..., None] * w1c) @ u1).float().cpu())
        ho = torch.einsum("bhij,bhjd->bhid", p["attn2_w"][:, heads].float(), ov_images(p["x2"], hw2)[:, heads])
        out["z_agg"].append((ho.sum(1).double() @ uagg).float().cpu())
        _, fr, _ = ad.finish(p["h1"], p["o2"])
        out["pred"].append((fr if fr is not None else torch.zeros_like(b2)).float().cpu())
        del p, st, A, s, y2, xi2, ho
    return {k: torch.cat(v).numpy() for k, v in out.items()}


# --------------------------------------------------------------------------
# analysis helpers
# --------------------------------------------------------------------------

def n_ordinary(L, sinks):
    """Number of ordinary (non-sink, non-BOS) keys among 0..i, for each i."""
    is_ord = np.ones(L); is_ord[0] = 0; is_ord[sinks] = 0
    return np.cumsum(is_ord)


def schedule(B, a, bbar, nord, sinks_in_prefix):
    """alpha_BOS and t under the two-level pull model."""
    eb, eo = np.exp(B), np.exp(B * bbar)
    den = sinks_in_prefix * eb + nord * eo
    return eb / den, (sinks_in_prefix * eb + nord * bbar * eo) / den


def free_fit(alpha, nord, lo=1):
    """Least squares on 1/alpha = a' + c * n_ord (i >= lo)."""
    y = 1.0 / alpha[lo:]
    X = np.stack([np.ones_like(y), nord[lo:]], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def binned_relerr(pred, meas, bins):
    return {n: float(np.median(np.abs(pred[a:b + 1] - meas[a:b + 1]) / np.abs(meas[a:b + 1])))
            for n, (a, b) in bins.items()}


def cleanliness(beta, sinks, L):
    ns = np.ones(L, bool); ns[:16] = False; ns[sinks] = False
    X = beta[:, ns].astype(np.float64)
    total = X.std()
    across_seq = X.mean(1).std()
    across_tok = X.std(1).mean()
    return {"sd_total": float(total), "sd_across_seq": float(across_seq),
            "sd_across_token": float(across_tok),
            "separation_ratio": float((1 - X.mean()) / total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev, init_only=args.init)
    if args.init:
        run_id += "_init"
    od = out_dir(run_id, "beta_v2"); fd = od / "figures"
    oldd = out_dir(run_id, "beta")
    old = torch.load(oldd / "ref.pt", weights_only=False)
    old_meta = json.loads((oldd / "meta.json").read_text())
    heads = old_meta["bos_heads"] or [int(np.argmax(old_meta["r_h"]))]
    old["bos_heads"] = heads
    splits = spec_splits(L)
    if ad.kind == "lm":
        ad.fit_frozen(splits["calibration"], L, args.batch_size)
    R = reference(ad, splits["reference"], L, args.batch_size, dev)
    print(f"[ref] sinks2={R['sinks2']} a2={R['a2']} bbar2={R['beta_bar2']:.4f} "
          f"sinks1={R['sinks1']} bbar1={R['beta_bar1']:.4f} B2[heads]={R['B2'][heads].round(2)}")
    E = evaluate(ad, R, old, splits["evaluation"], L, args.batch_size, dev)
    np.savez(od / "evaluation.npz", **E)
    torch.save({k: v for k, v in R.items() if k not in ("mu", "bos", "hid_mean")}
               | {"mu": {k: v.cpu() for k, v in R["mu"].items()},
                  "bos": {k: v.cpu() for k, v in R["bos"].items()},
                  "hid_mean": R["hid_mean"].cpu()}, od / "ref.pt")

    bins = bins_for(L)
    i = np.arange(L)
    S = {"model": args.model, "run_id": run_id, "L": L, "bos_heads": heads,
         "sinks_L2": R["sinks2"], "a_L2": R["a2"], "beta_bar_L2": R["beta_bar2"],
         "sinks_L1": R["sinks1"], "a_L1": R["a1"], "beta_bar_L1": R["beta_bar1"],
         "B_L2": R["B2"].tolist(), "B_L1": R["B1"].tolist()}

    # N1: share profiles
    jj = [j for j in range(17) if j < L]
    S["N1_profile"] = {}
    for k in ["b_x2", "b_x2_uncentered", "b_x1"]:
        X = E[k]
        S["N1_profile"][k] = {"per_j_mean": {str(j): float(X[:, j].mean()) for j in jj},
                              "per_j_sd": {str(j): float(X[:, j].std()) for j in jj},
                              "binned_mean": {n: float(X[:, a:b + 1].mean()) for n, (a, b) in bins.items()}}
    S["N1_C1_C2"] = {f"h{h}": {"ev_median_by_bin": {n: float(np.nanmedian(E["ev"][:, h, a:b + 1])) for n, (a, b) in bins.items()},
                               "r2_median_by_bin": {n: float(np.nanmedian(E["r2"][:, h, a:b + 1])) for n, (a, b) in bins.items()},
                               "B_centered": float(R["B2"][h])} for h in heads}

    # N2/N3: schedule fits for every BOS head, transition point, resolution
    nord2 = n_ordinary(L, R["sinks2"])
    sinks_pref = 1 + np.array([sum(1 for s in R["sinks2"] if s <= k) for k in range(L)])
    S["N2"], S["N3"] = {}, {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for h in heads:
        B = R["B2"][h]; a = R["a2"]; bb = R["beta_bar2"]
        am = E["alpha_bos"][:, h].astype(np.float64).mean(0)
        pa, pt = schedule(B, a, bb, nord2, sinks_pref)
        a_f, c_f = free_fit(am, nord2)
        istar = a * np.exp(B * (1 - bb))
        tm = E["t"][:, h].astype(np.float64).mean(0)
        S["N2"][f"h{h}"] = {
            "B": float(B), "a": a, "beta_bar": bb, "i_star": float(istar),
            "pred_alpha_relerr_by_bin": binned_relerr(pa, am, bins),
            "pred_t_relerr_by_bin": binned_relerr(pt, tm, bins),
            "free_fit": {"a_prime": a_f, "c": c_f, "c_pred": float(np.exp(B * (bb - 1))),
                         "i_star_free": float(a_f / c_f) if c_f > 0 else None},
            "free_fit_relerr_by_bin": binned_relerr(1 / (a_f + c_f * nord2), am, bins)}
        l, = ax[0].plot(i[1:], am[1:], lw=1, label=f"h{h} measured")
        ax[0].plot(i[1:], pa[1:], "--", color=l.get_color(), lw=1)
        ax[0].axvline(istar, color=l.get_color(), ls=":", lw=.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(fontsize=6)
    ax[0].set_title(r"N2: $\alpha_{i,BOS}$ measured (—) vs $e^B/(a e^B + n e^{B\bar\beta})$ (--), $i^*$ (··)")
    # N3 on the aggregate readout, prediction from the first BOS head's schedule
    z = E["z_agg"].astype(np.float64)
    f = z.mean(0)
    _, der = smooth_deriv(f, L)
    ratio = z.std(0) / (i * np.abs(der))
    h0 = heads[0]
    _, pt0 = schedule(R["B2"][h0], R["a2"], R["beta_bar2"], nord2, sinks_pref)
    # z moves by t times the centered BOS value, projected on the readout axis
    hw2 = head_weights(ad.attn2)
    vc = (R["bos"]["x2"] - R["mu"]["x2"]).double()
    wc = torch.einsum("hde,he->hd", hw2.W_o.double(), torch.einsum("hed,d->he", hw2.W_v.double(), vc))[heads].sum(0)
    wagg = old["w_bos"][heads].sum(0).double().to(wc.device)
    wn = float(wc @ (wagg / wagg.norm()))
    S["N3_centered_value_along_readout"] = wn
    dpt = np.gradient(pt0)
    pred_ratio = z.std(0) / (i * np.abs(dpt) * wn)
    m = i >= 4
    imin = int(i[m][np.nanargmin(ratio[m])])
    istar0 = R["a2"] * np.exp(R["B2"][h0] * (1 - R["beta_bar2"]))
    S["N3"] = {"argmin_measured": imin, "argmin_predicted": int(i[m][np.nanargmin(pred_ratio[m])]),
               "i_star_h0": float(istar0),
               "ratio_at": {str(k): [float(ratio[k]), float(pred_ratio[k])] for k in [4, 16, 64, 256, 1023, 4096, 8191] if k < L}}
    ax[1].plot(i[4:], ratio[4:], label="measured")
    ax[1].plot(i[4:], pred_ratio[4:], "--", label=r"$\sigma(z)/(i|\partial_i\hat t|\,\|w\|)$")
    ax[1].axvline(istar0, color="k", ls=":", label=r"$i^*$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7); ax[1].set_title("N3: relative resolution")
    fig.tight_layout(); fig.savefig(fd / "N2_N3.png", dpi=120); plt.close(fig)

    # N4: cleanliness, layer 1 vs layer 2
    S["N4"] = {"L2": cleanliness(E["b_x2"], R["sinks2"], L),
               "L1": cleanliness(E["b_x1"], R["sinks1"], L)}
    for tag, zk in [("L2", "zeta2"), ("L1", "zeta1")]:
        Z = E[zk] if E[zk].ndim == 2 else E[zk][:, h0]
        S["N4"][tag]["content_sd_across_seq_along_wbos_j16on"] = float(Z[:, 16:].std(0).mean())
        S["N4"][tag]["content_sd_rel_to_bos_norm"] = None
    S["N4"]["L1_schedule"] = {}
    nord1 = n_ordinary(L, R["sinks1"])
    sp1 = 1 + np.array([sum(1 for s in R["sinks1"] if s <= k) for k in range(L)])
    a1 = E["a1_bos"].astype(np.float64)
    a1m = a1.mean(0)
    for h in range(min(ad.H, 12)):
        B = R["B1"][h]
        pa, _ = schedule(B, R["a1"], R["beta_bar1"], nord1, sp1)
        a_f, c_f = free_fit(a1m, nord1)
        S["N4"]["L1_schedule"][f"h{h}"] = {"B": float(B), "pred_alpha_relerr_by_bin_headmean": binned_relerr(pa, a1m, bins),
                                           "free_fit_headmean": [a_f, c_f]}
    S["N4"]["L1_alpha_bos_cv_across_seq_at"] = {str(k): float(a1[:, k].std() / a1[:, k].mean()) for k in [16, 128, 1023] if k < L}
    S["N4"]["L2_alpha_bos_cv_across_seq_at"] = {str(k): float(E["alpha_bos"][:, h0, k].std() / E["alpha_bos"][:, h0, k].mean())
                                               for k in [16, 128, 1023] if k < L}

    # N6: where the step comes from
    J = [j for j in list(range(9)) + [16, 64, 256, 1000] if j < L]
    S["N6"] = {"j": J,
               "share_by_stage": {s: [float(E[f"b_{s}"][:, j].mean()) for j in J] for s in STAGES},
               "layer1_bos_weight": [float(E["a1_bos"][:, j].mean()) for j in J],
               "norm_post1": [float(E["norm_post1"][:, j].mean()) for j in J],
               "norm_mlp1_out": [float(E["norm_mlp1"][:, j].mean()) for j in J],
               "norm_h1": [float(E["norm_h1"][:, j].mean()) for j in J],
               "x2_along_layer1_bos_value": [float(E["x2_along_u1"][:, j].mean()) for j in J]}
    # threshold neurons: profile, and cutoff in terms of the layer-1 BOS weight
    NP = E["neuron_pre"].astype(np.float64)                        # [n,T,k]
    ab = E["a1_bos"].astype(np.float64)
    neu = []
    for k in range(NP.shape[-1]):
        x = ab[:, 1:].reshape(-1); y = NP[:, 1:, k].reshape(-1)
        c = np.polyfit(x, y, 1)
        neu.append({"neuron": int(R["neurons"][k]), "score": float(R["neuron_score"][k]),
                    "mean_pre_by_j": [float(NP[:, j, k].mean()) for j in J],
                    "corr_pre_vs_layer1_bos_weight": float(np.corrcoef(x, y)[0, 1]),
                    "cutoff_bos_weight": float(-c[1] / c[0]) if c[0] != 0 else None})
    S["N6"]["threshold_neurons"] = neu
    (od / "summary.json").write_text(json.dumps(S, indent=2, default=float))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for s in STAGES:
        ax[0].plot(J, S["N6"]["share_by_stage"][s], "o-", ms=3, label=s)
    ax[0].plot(J, S["N6"]["layer1_bos_weight"], "k--", label="L1 BOS weight")
    ax[0].set_xscale("symlog", linthresh=8); ax[0].legend(fontsize=7); ax[0].set_title("N6: centered share by stage")
    ax[1].plot(J, S["N6"]["norm_post1"], "o-", label="post-attn"); ax[1].plot(J, S["N6"]["norm_mlp1_out"], "o-", label="MLP out")
    ax[1].plot(J, S["N6"]["norm_h1"], "o-", label="h1"); ax[1].set_xscale("symlog", linthresh=8); ax[1].set_yscale("log")
    ax[1].legend(fontsize=7); ax[1].set_title("N6: norms after Layer 1")
    for nrec in neu[:6]:
        ax[2].plot(J, nrec["mean_pre_by_j"], "o-", ms=3, label=f"n{nrec['neuron']}")
    ax[2].axhline(0, color="k", lw=.5); ax[2].set_xscale("symlog", linthresh=8); ax[2].legend(fontsize=6)
    ax[2].set_title("N6: top threshold neurons, pre-activation")
    fig.tight_layout(); fig.savefig(fd / "N6.png", dpi=120); plt.close(fig)
    print(json.dumps({k: S[k] for k in ["sinks_L2", "a_L2", "beta_bar_L2", "N2", "N3", "N4"]}, indent=1, default=float)[:5000])
    print(json.dumps({k: v for k, v in S["N6"].items() if k != "threshold_neurons"}, default=float))
    for nrec in neu[:5]:
        print(nrec)


if __name__ == "__main__":
    main()
