"""How the Layer-1 MLP removes the BOS share: the neuron-level construction.

Notation (residual space, one fixed direction v = post1_BOS - mu_post1):
  s_j  = <post1_j - mu_post1, v>/|v|^2         BOS amount entering the MLP
  c_j  = <m_j - mu_m, v>/|v|^2                 BOS amount the MLP writes
  a_kj = GELU(z_kj), z_kj = pre-activation of neuron k at token j
  w_k  = <W_out[:, k], v>/|v|^2                BOS written per unit activation
Because c_j = sum_k w_k a_kj - const, the regression slope of c on s splits
exactly into per-neuron parts:
  slope(c|s) = sum_k q_k,   q_k = w_k * slope(a_k|s)
and slope(a_k|s) = E[GELU'(z_k)] * slope(z_k|s) to first order, so each part is
(read slope r_k = slope(z_k|s)) x (gate) x (write weight w_k).

Reported per position bin (j in 1-4, 5-16, 17+): the total slope, the per-token
slope at fixed j (across sequences), the neuron parts q_k, how many neurons
reach 50% / 90% of the total, and the read/write signs. Causal arms (all
positions j >= 1): ablate the top-K removal neurons, and replace the MLP's
BOS component by exactly -s_j ("ideal cancel").

Usage: python beta_mlp1_construct.py --model pv_unifL1_s42 --device cuda:0
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

BINS = {"j1-4": (1, 4), "j5-16": (5, 16), "j17+": (17, 10 ** 9)}
FIXED_J = [1, 2, 3, 4, 6, 8, 16, 64, 256]
TOPK = [8, 32, 128, 512]
N_EVAL = 400


class Moments:
    """Streaming sums for slope(y|s) of many outputs y against one s."""

    def __init__(self, dim, dev):
        z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float64)
        self.n, self.s, self.ss = z(()), z(()), z(())
        self.y, self.ys = z(dim), z(dim)

    def add(self, s, Y):            # s [N], Y [N, dim]
        self.n += len(s); self.s += s.sum(); self.ss += (s * s).sum()
        self.y += Y.sum(0); self.ys += (Y * s[:, None]).sum(0)

    def slope(self):
        ms, my = self.s / self.n, self.y / self.n
        return (self.ys / self.n - ms * my) / (self.ss / self.n - ms * ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    od = out_dir(run_id, "beta_v2"); fd = od / "figures"
    mlp = ad.model.block1.mlp
    blk2_ln = ad.model.block2.ln_1
    splits = spec_splits(L)
    D4 = mlp.c_fc.weight.shape[0]
    act = mlp.gelu                                                   # the model's own activation
    if isinstance(act, torch.nn.ReLU):
        act_prime = lambda z: (z > 0).double()
    else:                                                        act_prime = lambda z: 0.5 * (1 + torch.erf(z / 2 ** .5)) + z * torch.exp(-z ** 2 / 2) / (2 * np.pi) ** .5

    def batches(name, n=None):
        offs = splits[name][:n] if n else splits[name]
        for s in range(0, len(offs), args.batch_size):
            yield batch_from_offsets(offs[s:s + args.batch_size], L, dev)

    # reference means and BOS direction
    sp = torch.zeros(ad.d, device=dev, dtype=torch.float64); sm = torch.zeros_like(sp)
    n = 0; bos = None
    for idx in batches("reference"):
        p = ad.pre(idx)
        m = mlp(p["h1bar"])
        if bos is None:
            bos = p["post1"][0, 0].double()
        sp += p["post1"][:, 1:].double().sum((0, 1)); sm += m[:, 1:].double().sum((0, 1))
        n += idx.shape[0] * (L - 1)
    mu_p, mu_m = sp / n, sm / n
    v = bos - mu_p
    vv = v @ v
    W_out = mlp.c_proj.weight.double()
    w = (W_out.T @ v) / vv                                           # [4d]

    # pass over evaluation tokens
    mom = {b: {"a": Moments(D4, dev), "z": Moments(D4, dev), "c": Moments(1, dev)} for b in BINS}
    gate = {b: torch.zeros(D4, device=dev, dtype=torch.float64) for b in BINS}
    cnt = {b: 0 for b in BINS}
    per_j = {j: ([], []) for j in FIXED_J if j < L}
    scat = []
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        z = mlp.c_fc(p["h1bar"]).double()                            # [B,T,4d]
        a = act(z)
        c = (((mlp.c_proj(act(z.float())).double() - mu_m) @ v) / vv)
        s = (((p["post1"].double() - mu_p) @ v) / vv)
        for j in per_j:
            per_j[j][0].append(s[:, j].cpu()); per_j[j][1].append(c[:, j].cpu())
        scat.append(torch.stack([s[:, 1:65], c[:, 1:65]], -1).cpu())
        for b, (lo, hi) in BINS.items():
            sl = slice(lo, min(hi, L - 1) + 1)
            sb = s[:, sl].reshape(-1)
            mom[b]["a"].add(sb, a[:, sl].reshape(-1, D4))
            mom[b]["z"].add(sb, z[:, sl].reshape(-1, D4))
            mom[b]["c"].add(sb, c[:, sl].reshape(-1, 1))
            zb = z[:, sl].reshape(-1, D4)
            gate[b] += act_prime(zb).sum(0)
            cnt[b] += zb.shape[0]
        del z, a
    S = {"model": args.model, "bins": {}}
    wn = w.cpu().numpy()
    figdata = {}
    for b in BINS:
        sa = mom[b]["a"].slope().cpu().numpy()
        r = mom[b]["z"].slope().cpu().numpy()
        tot = float(mom[b]["c"].slope()[0])
        q = wn * sa
        g = (gate[b] / cnt[b]).cpu().numpy()
        order = np.argsort(-np.abs(q))
        cum = np.cumsum(q[order]) / q.sum()
        n50 = int(np.argmax(cum >= 0.5) + 1); n90 = int(np.argmax(cum >= 0.9) + 1)
        neg = q < 0
        S["bins"][b] = {
            "slope_c_on_s": tot, "sum_q": float(q.sum()),
            "n_neurons_50pct": n50, "n_neurons_90pct": n90,
            "share_of_sum_from_read_pos_write_neg": float(q[(r > 0) & (wn < 0)].sum() / q.sum()),
            "share_of_sum_from_read_neg_write_pos": float(q[(r < 0) & (wn > 0)].sum() / q.sum()),
            "top10": [{"neuron": int(k), "q": float(q[k]), "read_slope": float(r[k]),
                       "gate_mean_gelu_prime": float(g[k]), "write_w": float(wn[k])} for k in order[:10]]}
        figdata[b] = (q, r, g, order, cum)
    S["per_token_slope_at_fixed_j"] = {}
    for j, (ss, cc) in per_j.items():
        ss, cc = torch.cat(ss).numpy(), torch.cat(cc).numpy()
        sc = ss - ss.mean(); ccc = cc - cc.mean()
        S["per_token_slope_at_fixed_j"][str(j)] = {"slope": float((sc * ccc).mean() / (sc * sc).mean()),
                                                   "r2": float((sc * ccc).mean() ** 2 / ((sc * sc).mean() * (ccc * ccc).mean())),
                                                   "mean_s": float(ss.mean()), "mean_c": float(cc.mean())}

    # ---- causal arms ---------------------------------------------------------
    order_all = figdata["j5-16"][3]
    rem = [k for k in order_all if figdata["j5-16"][0][k] < 0]       # removal neurons, strongest first
    arms = {"clean": None, "ideal_cancel": "ideal"}
    for K in TOPK:
        arms[f"ablate_top{K}_removal"] = torch.as_tensor(rem[:K], device=dev)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)
    bins = bins_for(L)

    @torch.no_grad()
    def h1_of(p, arm):
        z = mlp.c_fc(p["h1bar"])
        a = act(z)
        if arm is not None and not isinstance(arm, str):
            a = a.clone(); a[:, 1:, arm] = 0
        m = mlp.c_proj(a)
        h1 = p["post1"] + m
        if arm == "ideal":
            s = ((p["post1"].double() - mu_p) @ v) / vv
            c = ((m.double() - mu_m) @ v) / vv
            corr = -(s + c)                                          # make s + c = 0 exactly
            corr[:, 0] = 0
            h1 = (h1.double() + corr[..., None] * v).float()
        return h1

    ridges = {k: Ridge(ad.d, dev) for k in arms}
    x2sum = {k: torch.zeros(ad.d, device=dev, dtype=torch.float64) for k in arms}
    x2bos, ncal = {}, 0
    for idx in batches("calibration"):
        p = ad.pre(idx)
        for k, arm in arms.items():
            h1 = h1_of(p, arm); x2 = ln(h1, blk2_ln)
            x2sum[k] += x2[:, 1:].double().sum((0, 1)); x2bos.setdefault(k, x2[0, 0].double().clone())
            _, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, _, _ = ad.finish(h1, o2)
            ridges[k].add(feats, tgt.expand(idx.shape[0], L))
        ncal += idx.shape[0] * (L - 1)
    mu2 = {k: x2sum[k] / ncal for k in arms}
    v2 = {k: x2bos[k] - mu2[k] for k in arms}
    for rr in ridges.values():
        rr.solve()
    res = {k: {"beta": [], "refit": []} for k in arms}
    for idx in batches("evaluation", N_EVAL):
        p = ad.pre(idx)
        for k, arm in arms.items():
            h1 = h1_of(p, arm); x2 = ln(h1, blk2_ln)
            _, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, _, _ = ad.finish(h1, o2)
            res[k]["beta"].append((((x2.double() - mu2[k]) @ v2[k]) / (v2[k] @ v2[k])).cpu())
            res[k]["refit"].append(ridges[k].predict(feats).cpu())
    S["causal"] = {}
    for k in arms:
        B_ = torch.cat(res[k]["beta"]).numpy()                         # centered share, own arm
        S["causal"][k] = {"beta_profile": {str(j): float(B_[:, j].mean()) for j in [1, 2, 3, 4, 6, 8, 16, 64, 256] if j < L},
                          "tail_sd_j16on": float(B_[:, 16:].std()),
                          "refit": err_by_bin(torch.cat(res[k]["refit"]).numpy(), L, bins)}
    (od / "mlp1_construct.json").write_text(json.dumps(S, indent=2))

    # ---- figures -------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    sc = torch.cat(scat).numpy()
    jj = np.broadcast_to(np.arange(1, 65), sc.shape[:2])
    ax[0].scatter(sc[..., 0].ravel(), sc[..., 1].ravel(), c=np.log(jj.ravel()), s=1, alpha=.3, cmap="viridis")
    lim = np.array([sc[..., 0].min(), sc[..., 0].max()])
    ax[0].plot(lim, -lim, "r--", lw=1, label="c = -s (full cancel)")
    ax[0].set_xlabel("BOS entering the MLP, s"); ax[0].set_ylabel("BOS written by the MLP, c"); ax[0].legend(fontsize=7)
    ax[0].set_title("per token, j ≤ 64 (color = log j)")
    q, r, g, order, cum = figdata["j5-16"]
    ax[1].plot(np.arange(1, 301), cum[:300]); ax[1].axhline(1, color="k", lw=.4)
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of neurons (sorted by |q_k|)")
    ax[1].set_ylabel("cumulative share of slope(c|s)"); ax[1].set_title("how many neurons do the removal (j 5-16)")
    big = np.abs(q) > np.abs(q[order[50]])
    ax[2].scatter(r[~big], w.cpu().numpy()[~big], s=2, color="0.7")
    ax[2].scatter(r[big], w.cpu().numpy()[big], s=10, c=g[big], cmap="coolwarm", vmin=0, vmax=1.1)
    ax[2].axhline(0, color="k", lw=.4); ax[2].axvline(0, color="k", lw=.4)
    ax[2].set_xlabel("read slope r_k = d z_k / d s"); ax[2].set_ylabel("write weight w_k (BOS per unit activation)")
    ax[2].set_title("top-50 neurons (color = mean GELU'), gray = rest")
    # activation vs s for the top 6 neurons, from one extra batch
    idx = next(batches("evaluation", args.batch_size))
    p = ad.pre(idx)
    s = (((p["post1"].double() - mu_p) @ v) / vv)[:, 1:65].cpu().numpy().ravel()
    z = mlp.c_fc(p["h1bar"])[:, 1:65].double().cpu().numpy().reshape(-1, D4)
    for k in order[:6]:
        ax[3].scatter(s, act(torch.tensor(z[:, k])).numpy(), s=2, alpha=.5, label=f"n{k}: w={w[k].item():+.4f}")
    ax[3].set_xlabel("s"); ax[3].set_ylabel("activation a_k"); ax[3].legend(fontsize=6); ax[3].set_title("top-6 removal neurons")
    fig.tight_layout(); fig.savefig(fd / "mlp1_construct.png", dpi=115); plt.close(fig)

    for b, d_ in S["bins"].items():
        print(b, {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in d_.items() if k != "top10"})
        for t in d_["top10"][:4]:
            print("    ", {k: round(v_, 4) if isinstance(v_, float) else v_ for k, v_ in t.items()})
    print("per-token slope at fixed j:", {j: (round(v_["slope"], 3), round(v_["r2"], 3)) for j, v_ in S["per_token_slope_at_fixed_j"].items()})
    for k, d_ in S["causal"].items():
        print(f"{k:24s}", {j: round(x, 3) for j, x in d_["beta_profile"].items()}, "tail sd %.4f" % d_["tail_sd_j16on"],
              "refit late %.4f" % d_["refit"].get("relmae_late", float("nan")))


if __name__ == "__main__":
    main()
