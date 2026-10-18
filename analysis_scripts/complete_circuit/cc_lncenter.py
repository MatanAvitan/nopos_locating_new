"""Does the first MLP's LayerNorm centering create the BOS sign split, and does
the MLP use it to remove BOS-related variation?

Site: block 0, MLP-input LayerNorm (ln_2). h_i = residual after block-0
attention (resid_mid, block 0). Stages, per token over the d features:
  h  ->  c = h - mu 1  ->  n = c / sigma  ->  y = gamma * n + b_LN
with sigma = sqrt(mean(c^2) + eps), eps = 1e-5 (the model's F.layer_norm).

Direction: w = sum_h W_O,h W_V,h xbar_0 (block-0 BOS value write, no biases;
xbar_0 = calibration mean of the block-0 attention input at position 0).

Neurons (rules of the earlier scripts, recomputed on calibration):
  T   16 threshold neurons: score = min(A_0, A_1) - max_{j>=2} |A_j|, A = mean
      post-activation by position (beta_v2.py N6)
  E   50 neurons with the largest |mean contribution| to the MLP output along w
      at any position (beta_mlp1.py rule, measured along w)
For neuron m: u_m = gamma * W_fc[m], a_m = W_fc[m] . b_LN + b_fc[m],
  z = u.h/sigma  -  mu (u.1)/sigma  +  a     (input, centering, bias terms).

Intervention on a neuron set S, lambda in {0, 0.5, 1}:
  z^(lambda) = (u.h - lambda mu u.1)/sigma + a, sigma cached from the original h,
other neurons and all parameters unchanged. Sets: T, E, ALL (every neuron),
RAND (16 random neurons, control). lambda = 1 must reproduce the forward.

Usage: python cc_lncenter.py --model 1h_s42 --device cuda:0
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from cc_common import EPS_LN, OUT_ROOT, SPECS, Capture, bins_for, jdump, load_net, splits_for
from common import RESULTS_ROOT

N_CAL, N_VAL = 160, 160
N_T, N_E = 16, 50
LAMBDAS = (0.0, 0.5, 1.0)
PRIOR = {"1h_s42": "attn2_full_L1024", "12h_s42": "full12h_L1024", "1h_s3": "seeds/1h_s3",
         "12h_s1": "seeds/12h_s1", "12h_s2": "seeds/12h_s2", "12h_s3": "seeds/12h_s3", "lm_bosin": "lm6_L128"}


def pos_groups(L):
    g = [(f"{j}", j, j) for j in range(8)] + [("8-15", 8, 15), ("16-127", 16, min(127, L - 1))]
    if L > 128:
        g.append(("128-%d" % (L - 1), 128, L - 1))
    return g


class Ridge:
    def __init__(self, dim, dev, lam=1e-4):
        self.A = torch.zeros(dim + 1, dim + 1, dtype=torch.float64, device=dev)
        self.b = torch.zeros(dim + 1, dtype=torch.float64, device=dev); self.lam = lam

    def add(self, X, y):
        X = X.double().reshape(-1, X.shape[-1]); X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1)
        self.A += X1.T @ X1; self.b += X1.T @ y.double().reshape(-1)

    def solve(self):
        n = self.A.shape[0]; reg = self.lam * torch.diag(self.A).mean() * torch.eye(n, dtype=self.A.dtype, device=self.A.device)
        reg[-1, -1] = 0; self.w = torch.linalg.solve(self.A + reg, self.b); return self

    def predict(self, X):
        X = X.double(); return torch.cat([X, torch.ones(*X.shape[:-1], 1, dtype=X.dtype, device=X.device)], -1) @ self.w


@torch.no_grad()
def run(args):
    dev, bs = args.device, args.batch_size
    net, _ = load_net(args.model, dev)
    spec = net.spec
    L, d = spec.L, net.d
    blk = net.blocks[0]
    out = {"model": args.model, "site": "block 0 MLP-input norm (ln_2)", "norm": type(blk.ln_2).__name__}
    if blk.skip_mlp or blk.ln_2 is None:
        out["verdict"] = "no MLP-input LayerNorm at block 0"; jdump(out, OUT_ROOT / "lncenter" / f"{args.model}.json"); return
    if type(blk.ln_2).__name__ == "RMSNorm":
        out["verdict"] = "RMSNorm at this site: no mean subtraction, the centering explanation cannot apply"
        jdump(out, OUT_ROOT / "lncenter" / f"{args.model}.json"); return
    sp, prov = splits_for(args.model)
    gamma = blk.ln_2.weight.double(); bLN = blk.ln_2.bias.double() if blk.ln_2.bias is not None else torch.zeros(d, dtype=torch.float64, device=dev)
    Wfc = blk.mlp.c_fc.weight.double(); bfc = blk.mlp.c_fc.bias.double() if blk.mlp.c_fc.bias is not None else torch.zeros(Wfc.shape[0], dtype=torch.float64, device=dev)
    Wout = blk.mlp.c_proj.weight.double()                                        # [d, 4d]
    act = (lambda t: F.relu(t)) if isinstance(blk.mlp.gelu, torch.nn.ReLU) else (lambda t: F.gelu(t))
    U = gamma[None] * Wfc                                                         # [4d, d] u_m
    A_bias = Wfc @ bLN + bfc                                                      # [4d]
    U1 = U.sum(1)                                                                 # u_m . 1
    want = {("x_attn", 0), ("resid_mid", 0), ("x_mlp", 0), ("mlp_pre", 0), ("mlp_out", 0)}

    def fwd(offs, edits=None, extra=()):
        idx = net.batch(offs)
        cap = Capture(want=want | set(extra), edits=edits)
        o = net.forward(idx, cap)
        return idx, cap.store, o

    # ---- calibration: direction and neuron sets ------------------------------
    t0 = time.time()
    x0s, n = 0, 0
    hid_sum, con_sum = 0, 0
    cal = sp["calibration"][:N_CAL]
    for s in range(0, len(cal), bs):
        idx, st, _ = fwd(cal[s:s + bs])
        x0s = x0s + st[("x_attn", 0)][:, 0].double().sum(0); n += idx.shape[0]
    xbar0 = x0s / n
    w = torch.einsum("hdf,f->d", net.ov(0), xbar0); wh = w / w.norm()
    wout_w = Wout.T @ wh                                                          # [4d] BOS-direction output per unit activation
    for s in range(0, len(cal), bs):
        idx, st, _ = fwd(cal[s:s + bs])
        hid = act(st[("mlp_pre", 0)].double())
        hid_sum = hid_sum + hid.sum(0); con_sum = con_sum + (hid * wout_w).sum(0)
    A = hid_sum / n; Cm = con_sum / n                                             # [T,4d]
    score = torch.minimum(A[0], A[1]) - A[2:].abs().max(0).values
    setT = torch.argsort(score, descending=True)[:N_T]
    setE = torch.argsort(Cm.abs().max(0).values, descending=True)[:N_E]
    g = torch.Generator(device="cpu").manual_seed(0)
    setR = torch.randperm(Wfc.shape[0], generator=g)[:N_T].to(dev)
    sets = {"T": setT, "E": setE, "ALL": torch.arange(Wfc.shape[0], device=dev), "RAND": setR}
    prior = None
    pr = RESULTS_ROOT / PRIOR.get(args.model, "_none_") / "beta_v2" / "summary.json"
    if pr.exists():
        try:
            pl = [r["neuron"] for r in json.loads(pr.read_text())["N6"]["threshold_neurons"]]
            prior = {"prior_threshold_neurons": pl, "overlap_with_T": len(set(pl) & set(setT.tolist()))}
        except Exception as e:  # noqa: BLE001
            prior = {"error": repr(e)}
    out.update({"direction": "unit sum_h W_O W_V xbar_0 (block 0)", "overlap_w_with_ones": float(wh.sum() / d ** 0.5),
                "sum_w": float(wh.sum()), "sets": {k: v.tolist() for k, v in sets.items() if k != "ALL"},
                "prior": prior, "splits": prov})
    print(f"calibration {time.time() - t0:.0f}s overlap(w, 1/sqrt d) = {out['overlap_w_with_ones']:.4f}", flush=True)

    # ---- validation, original forward: stages and neuron terms ---------------
    t0 = time.time()
    val = sp["validation"][:N_VAL]
    G = pos_groups(L)
    keep = {k: [] for k in ("ph", "pc", "pn", "py", "cent", "g", "post", "sigma", "mu")}
    nset = torch.unique(torch.cat([setT, setE]))
    NI, NC, NZ, NA, NO = [], [], [], [], []
    rec_err = 0.0; pre_err = 0.0
    for s in range(0, len(val), bs):
        idx, st, _ = fwd(val[s:s + bs])
        h = st[("resid_mid", 0)].double()
        mu = h.mean(-1, keepdim=True); c = h - mu
        sig = (c.pow(2).mean(-1, keepdim=True) + EPS_LN).sqrt()
        nn_ = c / sig; y = gamma * nn_ + bLN
        rec_err = max(rec_err, float((y - st[("x_mlp", 0)].double()).abs().max()))
        z_full = st[("mlp_pre", 0)].double()
        zt = (h @ U.T) / sig - mu * U1 / sig + A_bias
        pre_err = max(pre_err, float((zt - z_full).abs().max()))
        keep["ph"].append((h @ wh).cpu()); keep["pc"].append((c @ wh).cpu()); keep["pn"].append((nn_ @ wh).cpu())
        keep["py"].append((y @ wh).cpu()); keep["cent"].append((mu[..., 0] * wh.sum()).cpu())
        mo = st[("mlp_out", 0)].double()
        keep["g"].append((mo @ wh).cpu()); keep["post"].append(((h + mo) @ wh).cpu())
        keep["sigma"].append(sig[..., 0].cpu()); keep["mu"].append(mu[..., 0].cpu())
        Us = U[nset]
        NI.append(((h @ Us.T) / sig).float().cpu()); NC.append((mu * U1[nset] / sig).float().cpu())
        NZ.append(z_full[..., nset].float().cpu())
        hid = act(z_full[..., nset]); NA.append(hid.float().cpu()); NO.append((hid * wout_w[nset]).float().cpu())
    K = {k: torch.cat(v).numpy() for k, v in keep.items()}
    NI, NC, NZ, NA, NO = (torch.cat(v).numpy() for v in (NI, NC, NZ, NA, NO))
    out["reconstruction"] = {"ln_output_max_abs_err": rec_err, "preactivation_max_abs_err": pre_err}
    print(f"validation {time.time() - t0:.0f}s, LN recon err {rec_err:.2e}, preact err {pre_err:.2e}", flush=True)

    def stats(v, lo, hi):
        x = v[:, lo:hi + 1]
        return {"mean": float(x.mean()), "sd_seq": float(x.std(0).mean()), "frac_pos": float((x > 0).mean())}

    stage = {}
    for name, lo, hi in G:
        r = {k: stats(K[k], lo, hi) for k in ("ph", "pc", "pn", "py", "cent", "g", "post")}
        flip = np.sign(K["ph"][:, lo:hi + 1]) != np.sign(K["pc"][:, lo:hi + 1])
        r["frac_sign_change_centering"] = float(flip.mean())
        r["frac_sign_change_affine"] = float((np.sign(K["pn"][:, lo:hi + 1]) != np.sign(K["py"][:, lo:hi + 1])).mean())
        stage[name] = r
    out["stages"] = stage

    # MLP cancellation of the residual BOS component, original forward
    def cancel(Kd, lo, hi):
        ph, gg, post = Kd["ph"][:, lo:hi + 1], Kd["g"][:, lo:hi + 1], Kd["post"][:, lo:hi + 1]
        vpre, vpost = ph.var(0), post.var(0)
        cg = ((ph - ph.mean(0)) * (gg - gg.mean(0))).mean(0)
        return {"var_pre": float(vpre.mean()), "var_post": float(vpost.mean()),
                "var_reduction": float(1 - vpost.mean() / max(vpre.mean(), 1e-300)),
                "slope_g_on_pre_within_position": float((cg / np.maximum(vpre, 1e-300)).mean()),
                "mean_pre": float(ph.mean()), "mean_g": float(gg.mean()), "mean_post": float(post.mean())}
    out["mlp_cancellation"] = {name: cancel(K, lo, hi) for name, lo, hi in G}

    # neuron decomposition
    early_m = slice(0, 2); late_lo = 16
    neu = []
    total_g = K["g"]
    for k, m in enumerate(nset.tolist()):
        I, C, Z = NI[..., k], NC[..., k], NZ[..., k]
        IC = I - C
        e = lambda v: float(v[:, early_m].mean()); lt = lambda v: float(v[:, late_lo:].mean())
        terms = {"input": (e(I), lt(I)), "input_minus_centering": (e(IC), lt(IC)), "preactivation": (e(Z), lt(Z))}
        split = {t: bool(np.sign(a) != np.sign(b)) for t, (a, b) in terms.items()}
        if split["input"]:
            origin = "before LN (input term)"
        elif split["input_minus_centering"]:
            origin = "centering"
        elif split["preactivation"]:
            origin = "bias (LN bias / MLP bias)"
        else:
            origin = "no sign split"
        con = NO[..., k]
        neu.append({"neuron": m, "in_T": m in setT.tolist(), "in_E": m in setE.tolist(),
                    "terms_early_late": terms, "sign_split": split, "split_origin": origin,
                    "centering_term_early_late": (e(C), lt(C)), "bias_term": float(A_bias[m]),
                    "act_early_late": (e(NA[..., k]), lt(NA[..., k])),
                    "contrib_w_early_late": (e(con), lt(con)), "wout_w": float(wout_w[m]),
                    "by_group": {name: {"input": float(I[:, lo:hi + 1].mean()), "centering": float(C[:, lo:hi + 1].mean()),
                                        "pre": float(Z[:, lo:hi + 1].mean()), "act": float(NA[..., k][:, lo:hi + 1].mean()),
                                        "contrib": float(con[:, lo:hi + 1].mean()),
                                        "frac_flip_centering": float((np.sign(I[:, lo:hi + 1]) != np.sign(IC[:, lo:hi + 1])).mean())}
                                 for name, lo, hi in G}})
    out["neurons"] = neu
    for sname in ("T", "E"):
        idxs = [i for i, m in enumerate(nset.tolist()) if m in sets[sname].tolist()]
        sub = NO[..., idxs].sum(-1)
        out.setdefault("set_contribution", {})[sname] = {
            name: {"set_mean": float(sub[:, lo:hi + 1].mean()), "mlp_total_mean": float(total_g[:, lo:hi + 1].mean()),
                   "corr_set_vs_total_within_position": float(np.nanmean([np.corrcoef(sub[:, j], total_g[:, j])[0, 1]
                                                                          for j in range(lo, hi + 1) if sub[:, j].std() > 0 and total_g[:, j].std() > 0]))
                   if hi >= lo else None,
                   "origin_counts": {o: sum(1 for r in neu if r["in_" + sname] and r["split_origin"] == o)
                                     for o in ("before LN (input term)", "centering", "bias (LN bias / MLP bias)", "no sign split")}}
            for name, lo, hi in G}

    # ---- interventions -------------------------------------------------------
    # layer-2 heads to watch: block-1 heads with r_h > 10 in the screen (else top 3)
    meta = OUT_ROOT / "screen" / args.model / "meta.json"
    rh = np.array(json.loads(meta.read_text())["r_h"]["L1"]) if meta.exists() else np.zeros(net.H)
    watch = [int(h) for h in np.nonzero(rh > 10)[0]] or [int(h) for h in np.argsort(-rh)[:3]]
    hw1 = net.hw[1]
    b_ln1 = net.ln_params(net.blocks[1].ln_1)[1]
    is_lm = spec.kind == "lm"
    feat = ("final_ln", net.n_layer) if not is_lm else ("resid_out", 1)
    M1 = net.ov(1)

    def edits_for(S, lam):
        if lam == 1.0 and S is None:
            return None
        st_ = {}

        def keep_h(t):
            st_["h"] = t.double(); return t

        def pre(t):
            h = st_["h"]; mu = h.mean(-1, keepdim=True)
            sig = ((h - mu).pow(2).mean(-1, keepdim=True) + EPS_LN).sqrt()
            t = t.clone()
            Us = U[S]
            t[..., S] = (((h @ Us.T) - lam * mu * U1[S]) / sig + A_bias[S]).float()
            return t
        return {("resid_mid", 0): keep_h, ("mlp_pre", 0): pre}

    conds = [("orig", None, 1.0)] + [(f"{s}_lam{l_}", s, l_) for s in ("T", "E", "ALL", "RAND") for l_ in LAMBDAS]
    # layer-2 anchor for beta (calibration, original)
    x0s2 = 0
    for s in range(0, len(cal), bs):
        _, st, _ = fwd(cal[s:s + bs], extra=[("x_attn", 1)])
        x0s2 = x0s2 + st[("x_attn", 1)][:, 0].double().sum(0)
    d2 = x0s2 / n - b_ln1
    wb2 = torch.einsum("hdf,f->hd", M1, x0s2 / n)

    def measure(offs, ed):
        R = {k: [] for k in ("ph", "post", "actT", "pred", "feat", "beta2", "abos", "zhead", "Bi", "loss")}
        for s in range(0, len(offs), bs):
            idx, st, o = fwd(offs[s:s + bs], edits=ed, extra=[("x_attn", 1), ("weights", 1), feat])
            h = st[("resid_mid", 0)].double(); mo = st[("mlp_out", 0)].double()
            R["ph"].append((h @ wh).cpu()); R["post"].append(((h + mo) @ wh).cpu())
            R["actT"].append(act(st[("mlp_pre", 0)].double()[..., setT]).mean(-1).cpu())
            x2 = st[("x_attn", 1)].double()
            R["beta2"].append(((x2 - b_ln1) @ d2 / (d2 @ d2)).cpu())
            Aw = st[("weights", 1)].double()
            R["abos"].append(Aw[:, watch, :, 0].cpu())
            Y = torch.einsum("hdf,btf->bhtd", M1[watch], x2)
            uu = (wb2[watch] / wb2[watch].norm(dim=-1, keepdim=True))
            R["zhead"].append(torch.einsum("bhts,bhs->bht", Aw[:, watch], torch.einsum("bhsd,hd->bhs", Y, uu)).cpu())
            q = torch.einsum("btd,hed->bhte", x2, hw1.W_q[watch].double()) + hw1.b_q[watch].double()[None, :, None]
            R["Bi"].append((torch.einsum("bhte,he->bht", q, torch.einsum("hed,d->he", hw1.W_k[watch].double(), d2)) * hw1.scale).cpu())
            R["feat"].append(st[feat].float())
            if is_lm:
                R["loss"].append(net.lm_loss(idx, o).float().cpu())
            else:
                R["pred"].append(o.float().cpu())
        out_ = {k: (torch.cat(v).numpy() if k != "feat" else torch.cat(v)) for k, v in R.items() if v}
        return out_

    tgt = torch.arange(L, device=dev, dtype=torch.float64)
    res = {}
    base = None
    bins = bins_for(L)
    ii = np.arange(L)

    def relmae(pred):
        o = {}
        for bn, (lo, hi) in bins.items():
            e = np.abs(pred[:, lo:hi + 1] - ii[lo:hi + 1]) / ii[lo:hi + 1]
            o[bn] = float(np.median(e))
        return o

    for cname, S, lam in conds:
        t0 = time.time()
        ed = edits_for(sets[S] if S else None, lam)
        Mc = measure(cal, ed); Mv = measure(val, ed)
        ridge = Ridge(d, dev)
        for s in range(0, Mc["feat"].shape[0], 32):
            ridge.add(Mc["feat"][s:s + 32], tgt.expand(min(32, Mc["feat"].shape[0] - s), L))
        ridge.solve()
        probe = torch.cat([ridge.predict(Mv["feat"][s:s + 32]) for s in range(0, Mv["feat"].shape[0], 32)]).cpu().numpy()
        r = {"lambda": lam, "set": S}
        if base is None:
            base = Mv; base_ridge = ridge
        r["ridge_refit"] = relmae(probe)
        r["ridge_frozen"] = relmae(torch.cat([base_ridge.predict(Mv["feat"][s:s + 32]) for s in range(0, Mv["feat"].shape[0], 32)]).cpu().numpy())
        if not is_lm:
            r["frozen_pos_head"] = relmae(Mv["pred"])
        else:
            r["lm_loss_mean"] = float(Mv["loss"].mean()); r["lm_loss_delta"] = float(Mv["loss"].mean() - base["loss"].mean())
        # isotonic decoder on the summed watched-head scalar, refit and frozen
        zc, zv = Mc["zhead"].sum(1), Mv["zhead"].sum(1)
        if cname == "orig":
            res["zcal0"] = zc
        def iso(zcal, zev):
            m_ = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(zcal[:, 1:].ravel(), np.tile(ii[1:], len(zcal)).astype(float))
            p = np.zeros_like(zev); p[:, 1:] = m_.predict(zev[:, 1:].ravel()).reshape(len(zev), -1); return p
        r["head_scalar_iso_refit"] = relmae(iso(zc, zv)); r["head_scalar_iso_frozen"] = relmae(iso(res["zcal0"], zv))
        grp = {}
        for name, lo, hi in G:
            sl = slice(lo, hi + 1)
            gg = {"actT_mean": float(Mv["actT"][:, sl].mean()),
                  "post_mean": float(Mv["post"][:, sl].mean()), "post_sd_seq": float(Mv["post"][:, sl].std(0).mean()),
                  "pre_sd_seq": float(Mv["ph"][:, sl].std(0).mean()),
                  "var_reduction_by_mlp": float(1 - Mv["post"][:, sl].var(0).mean() / max(Mv["ph"][:, sl].var(0).mean(), 1e-300)),
                  "beta2_mean": float(Mv["beta2"][:, sl].mean()), "beta2_sd_seq": float(Mv["beta2"][:, sl].std(0).mean()),
                  "beta2_mean_shift_vs_orig": float(np.abs(Mv["beta2"][:, sl].mean(0) - base["beta2"][:, sl].mean(0)).mean()),
                  "abos_sd_seq": [float(Mv["abos"][:, k, sl].std(0).mean()) for k in range(len(watch))],
                  "abos_mean_shift_vs_orig": [float(np.abs(Mv["abos"][:, k, sl].mean(0) - base["abos"][:, k, sl].mean(0)).mean()) for k in range(len(watch))],
                  "zhead_sd_seq": [float(Mv["zhead"][:, k, sl].std(0).mean()) for k in range(len(watch))],
                  "Bi_sd_seq": [float(Mv["Bi"][:, k, sl].std(0).mean()) for k in range(len(watch))]}
            grp[name] = gg
        r["by_group"] = grp
        if cname == "orig":
            r["identity_check"] = None
        if cname.endswith("lam1.0"):
            r["identity_check_max_abs_vs_orig"] = {"post": float(np.abs(Mv["post"] - base["post"]).max()),
                                                   **({"pred": float(np.abs(Mv["pred"] - base["pred"]).max())} if not is_lm else
                                                      {"loss": float(np.abs(Mv["loss"] - base["loss"]).max())})}
        res[cname] = r
        print(f"{cname:12s} {time.time() - t0:.0f}s ridge refit {r['ridge_refit']} "
              f"{'frozen head ' + str(r.get('frozen_pos_head')) if not is_lm else 'dloss %.4f' % r['lm_loss_delta']}", flush=True)
    res.pop("zcal0", None)
    out["watched_layer2_heads"] = watch
    out["interventions"] = res

    # ---- figures ---------------------------------------------------------------
    od = OUT_ROOT / "lncenter"
    od.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    jj = np.arange(L)
    for k, (key, title) in enumerate((("ph", "h (pre-LN residual)"), ("pc", "h - mu 1 (centered)"),
                                       ("pn", "(h - mu 1)/sigma"), ("py", "gamma * n + b (LN output)"))):
        m_, s_ = K[key].mean(0), K[key].std(0)
        ax[0, k].plot(jj, m_, lw=1); ax[0, k].fill_between(jj, m_ - s_, m_ + s_, alpha=.3)
        ax[0, k].axhline(0, color="k", lw=.5); ax[0, k].set_xscale("symlog", linthresh=8); ax[0, k].set_title(title + " . w")
        ax[1, k].plot(jj, (K[key] > 0).mean(0), lw=1); ax[1, k].set_xscale("symlog", linthresh=8); ax[1, k].set_ylim(-.05, 1.05)
        ax[1, k].set_title("fraction positive"); ax[1, k].set_xlabel("position")
    fig.suptitle(f"{args.model}: block-0 MLP-input LN stages projected on the block-0 BOS write (|<w,1>|/sqrt d = {abs(out['overlap_w_with_ones']):.3f})")
    fig.tight_layout(); fig.savefig(od / f"{args.model}_stages.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    top = sorted(neu, key=lambda r: -abs(r["contrib_w_early_late"][0] - r["contrib_w_early_late"][1]))[:24]
    names = [str(r["neuron"]) for r in top]
    for k, (key, title) in enumerate((("input", "input term u.h/sigma"), ("centering", "centering term mu u.1/sigma"), ("pre", "pre-activation"))):
        M_ = np.array([[r["by_group"][g_[0]][key] for g_ in G] for r in top])
        im = ax[k].imshow(M_, aspect="auto", cmap="RdBu_r", vmin=-np.abs(M_).max(), vmax=np.abs(M_).max())
        ax[k].set_xticks(range(len(G))); ax[k].set_xticklabels([g_[0] for g_ in G], rotation=45, fontsize=7)
        ax[k].set_yticks(range(len(top))); ax[k].set_yticklabels(names, fontsize=6); ax[k].set_title(title); fig.colorbar(im, ax=ax[k])
    fig.suptitle(f"{args.model}: 24 neurons with the largest early-vs-late change in BOS-direction output")
    fig.tight_layout(); fig.savefig(od / f"{args.model}_neurons.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    for S in ("T", "E", "ALL", "RAND"):
        xs = list(LAMBDAS)
        lastg = G[-1][0]
        ax[0].plot(xs, [res[f"{S}_lam{l_}"]["by_group"][lastg]["post_sd_seq"] for l_ in xs], "o-", label=S)
        ax[1].plot(xs, [res[f"{S}_lam{l_}"]["by_group"][lastg]["beta2_sd_seq"] for l_ in xs], "o-", label=S)
        lb = list(bins)[-1]
        ax[2].plot(xs, [res[f"{S}_lam{l_}"]["ridge_refit"][lb] for l_ in xs], "o-", label=S + " refit")
        if not is_lm:
            ax[2].plot(xs, [res[f"{S}_lam{l_}"]["frozen_pos_head"][lb] for l_ in xs], "x--", label=S + " frozen")
    ax[0].set_title(f"post-MLP BOS-direction SD across sequences ({G[-1][0]})"); ax[1].set_title(f"block-1 beta SD ({G[-1][0]})")
    ax[2].set_title(f"position relMAE ({list(bins)[-1]})"); ax[2].set_yscale("log")
    for a_ in ax:
        a_.set_xlabel("lambda (1 = original centering)"); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(od / f"{args.model}_interventions.png", dpi=120); plt.close(fig)
    jdump(out, od / f"{args.model}.json")
    print("wrote", od / f"{args.model}.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    run(ap.parse_args())
