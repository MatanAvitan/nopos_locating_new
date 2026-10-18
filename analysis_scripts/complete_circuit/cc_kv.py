"""Task D: joint key/value analysis of selected heads, with realizable edits.

Convention: center m = LN bias of the head's attention input, global anchor
d = calibration mean x_0 - m, readout u = unit(w_nonBOS - w_BOS) (Section 1.6).

Score operations at head (l, h); other heads and the query are unchanged:
  S1 restricted   keys rebuilt as W_K (m + beta_j d) + b_K (E removed, exact)
  S2 resid-mean   logits s_ij - E_ij + Ebar(i, j); Ebar = calibration mean at
                  (query position, key position); logit patch, oracle diagnostic
  S3 stabilized   keys from x_j - (beta_j - betabar_j) d; betabar = calibration
                  mean profile; oracle diagnostic
  S4 indicator    keys from x_j - (beta_j - 1[j in S]) d, S = {0} + sinks
Value operations:
  V1 restricted   projected value -> c + lambda beta_j (structural)
  V1f             projected value -> c_f + lambda_f beta_j (OLS on calibration)
  V2 resid-mean   projected value -> c + lambda beta_j + rbar_j (oracle)
  V3 stabilized   full value vector from x_j - (beta_j - betabar_j) d (oracle)
  V4 indicator    full value vector from x_j - (beta_j - 1[j in S]) d
V1, V1f and V2 edit the head value along g = W_O,h^T u only:
  V'_j = V_j + (target_j - v_j) g / |g|^2,  v_j = <V_j - b_V, g>.
Pairs (S, V): P1 = (S1, V1), P1f = (S1, V1f), P2 = (S2, V2), P3 = (S3, V3),
P4 = (S4, V4). For each pair: z11 orig/orig, z10 orig scores + new values,
z01 new scores + orig values, z00 both; K = z10 - z00, V = z01 - z00,
I = z11 - z10 - z01 + z00 (exact).

Each condition runs the full model; the endpoint is the frozen position head
(position models) or next-token loss and refit probes (LM).

Usage: python cc_kv.py --model 1h_s42 --heads 1:0 [--split test] --device cuda:0
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from cc_common import OUT_ROOT, SPECS, Capture, bins_for, jdump, load_net, splits_for

PAIRS = {"P1": ("S1", "V1"), "P1f": ("S1", "V1f"), "P2": ("S2", "V2"), "P3": ("S3", "V3"),
         "P2t": ("S2t", "V2t"), "P2e": ("S2e", "V2e"), "P2s": ("S2s", "V2s"),
         "P3t": ("S3t", "V3t"), "P3e": ("S3e", "V3e"), "P3s": ("S3s", "V3s"), "P4": ("S4", "V4")}
# key groups (calibration): sinks = BOS + positions with beta > 0.5 in > 90% of
# sequences; early = other keys 1..15 (includes transitional keys that are
# sinks in some sequences only); tail = other keys >= 16


def conditions():
    c = {"orig": (None, None)}
    for p, (s, v) in PAIRS.items():
        c[f"{p}_z10"] = (None, v)
        c[f"{p}_z01"] = (s, None)
        c[f"{p}_z00"] = (s, v)
    return c


def positions(L):
    p = list(range(min(L, 128)))
    if L > 128:
        p += list(range(128, L, 4))
    return np.array(p)


class Ridge:
    def __init__(self, dim, dev, lam=1e-4):
        self.A = torch.zeros(dim + 1, dim + 1, dtype=torch.float64, device=dev)
        self.b = torch.zeros(dim + 1, dtype=torch.float64, device=dev)
        self.lam = lam

    def add(self, X, y):
        X = X.double().reshape(-1, X.shape[-1]); X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1)
        self.A += X1.T @ X1; self.b += X1.T @ y.double().reshape(-1)

    def solve(self):
        n = self.A.shape[0]
        reg = self.lam * torch.diag(self.A).mean() * torch.eye(n, dtype=self.A.dtype, device=self.A.device); reg[-1, -1] = 0
        self.w = torch.linalg.solve(self.A + reg, self.b)
        return self

    def predict(self, X):
        X = X.double(); return torch.cat([X, torch.ones(*X.shape[:-1], 1, dtype=X.dtype, device=X.device)], -1) @ self.w


class HeadEdit(Capture):
    """Applies one (S, V) condition to every selected head of layer l and
    records the head scalar z for each selected head."""

    def __init__(self, net, l, heads, P, cond, want=None):
        super().__init__(want=want or set())
        self.net, self.l, self.heads, self.P, self.cond = net, l, heads, P, cond
        self.hw = net.hw[l]
        self.z = {}

    def __call__(self, stage, l, t):
        if l == self.l:
            S, V = self.cond
            if stage == "x_attn":
                self.x = t.double()
                self.beta = {}
                for h, P in self.P.items():
                    self.beta[h] = ((self.x - P["m"]) @ P["d"]) / (P["d"] @ P["d"])
            elif stage == "k" and S in ("S1", "S3", "S3t", "S3e", "S3s", "S4"):
                t = t.clone()
                for h, P in self.P.items():
                    b = self.beta[h]
                    if S == "S1":
                        xk = P["m"] + b[..., None] * P["d"]
                    else:
                        tgt = self.target(P, b, S[1:])
                        xk = self.x - (b - tgt)[..., None] * P["d"]
                    t[:, h] = (torch.einsum("btd,ed->bte", xk, self.hw.W_k[h].double()) + self.hw.b_k[h].double()).float()
            elif stage == "v" and V is not None:
                t = t.clone()
                for h, P in self.P.items():
                    b = self.beta[h]
                    if V in ("V3", "V3t", "V3e", "V3s", "V4"):
                        tgt = self.target(P, b, V[1:])
                        xv = self.x - (b - tgt)[..., None] * P["d"]
                        t[:, h] = (torch.einsum("btd,ed->bte", xv, self.hw.W_v[h].double()) + self.hw.b_v[h].double()).float()
                    else:
                        g = P["g"]; vh = t[:, h].double()
                        vproj = (vh - self.hw.b_v[h].double()) @ g
                        if V == "V1":
                            target = P["c"] + P["lam"] * b
                        elif V == "V1f":
                            target = P["cf"] + P["lamf"] * b
                        else:                                  # V2, V2t, V2e, V2s
                            target = P["c"] + P["lam"] * b + P["rbar"]
                            target = vproj + (target - vproj) * self.group_mask(P, V[2:])
                        t[:, h] = (vh + ((target - vproj) / (g @ g))[..., None] * g).float()
            elif stage == "scores" and S in ("S2", "S2t", "S2e", "S2s"):
                t = t.clone()
                for h, P in self.P.items():
                    xi = self.x - P["m"] - self.beta[h][..., None] * P["d"]
                    q = torch.einsum("btd,ed->bte", self.x, self.hw.W_q[h].double()) + self.hw.b_q[h].double()
                    E = torch.einsum("bie,bje->bij", q, torch.einsum("btd,ed->bte", xi, self.hw.W_k[h].double())) * self.hw.scale
                    t[:, h] = (t[:, h].double() - (E - P["Ebar"]) * self.group_mask(P, S[2:])).float()
            elif stage == "z":
                for h, P in self.P.items():
                    self.z[h] = (t[:, h].double() @ P["g"]) - self.hw.b_v[h].double() @ P["g"]
        return super().__call__(stage, l, t)

    @staticmethod
    def group_mask(P, g):
        """Key mask [T] for '' (all keys), 't' tail, 'e' early non-sink, 's' sinks."""
        return {"": torch.ones_like(P["ind"]), "t": P["tail"], "e": P["early"], "s": P["ind"]}[g]

    @staticmethod
    def target(P, b, kind):
        """Feature target: '3' betabar on all keys, '3t' tail keys only, '3e'
        early non-sink keys only, '3s' sinks only (BOS is fixed), '4' the sink
        indicator."""
        if kind == "4":
            return P["ind"].expand_as(b)
        mask = {"3": torch.ones_like(P["ind"]), "3t": P["tail"], "3e": P["early"], "3s": P["ind"]}[kind]
        return b + (P["betabar"] - b) * mask


@torch.no_grad()
def run(args):
    dev, bs = args.device, args.batch_size
    net, _ = load_net(args.model, dev)
    spec = net.spec
    L = spec.L
    sp, prov = splits_for(args.model)
    heads_by_layer = {}
    for s in args.heads:
        l, h = map(int, s.split(":"))
        heads_by_layer.setdefault(l, []).append(h)
    assert len(heads_by_layer) == 1, "one layer per run (heads of that layer are edited jointly)"
    l, heads = next(iter(heads_by_layer.items()))
    P_ = positions(L); Pt = torch.as_tensor(P_, device=dev)
    hw = net.hw[l]
    bins = bins_for(L)
    is_lm = spec.kind == "lm"
    # refit probe features: what the frozen head reads (ln_f output) for the
    # position models; the edited block's output for the LM
    feat_stage = ("final_ln", net.n_layer) if not is_lm else ("resid_out", l)

    def run_batch(offs, cond, P, want=()):
        idx = net.batch(offs)
        he = HeadEdit(net, l, heads, P, cond, want=set(want))
        out = net.forward(idx, he)
        return idx, he, out

    # ---- calibration statistics (pass 1) -----------------------------------
    t0 = time.time()
    cal = sp["calibration"][:args.n_cal]
    x0s = 0; xnb = 0; n = 0
    for s in range(0, len(cal), bs):
        idx = net.batch(cal[s:s + bs]); cap = Capture(want={("x_attn", l)}); net.forward(idx, cap, stop_after=l)
        x = cap.store[("x_attn", l)].double(); x0s = x0s + x[:, 0].sum(0); xnb = xnb + x[:, 1:].sum((0, 1)); n += idx.shape[0]
    x0 = x0s / n; xnb = xnb / (n * (L - 1))
    m = net.ln_params(net.blocks[l].ln_1)[1]
    d = x0 - m
    M = net.ov(l)
    P = {}
    for h in heads:
        wb, wn = M[h] @ x0, M[h] @ xnb
        u = (wn - wb) / (wn - wb).norm()
        g = hw.W_o[h].double().T @ u                                        # [dh]
        P[h] = {"m": m, "d": d, "u": u, "g": g, "c": float((M[h] @ m) @ u), "lam": float((M[h] @ d) @ u)}
    acc = {h: {"bs": 0, "bss": 0, "bgt": 0, "rs": 0, "Es": 0, "ols": np.zeros(5)} for h in heads}
    for s in range(0, len(cal), bs):
        idx = net.batch(cal[s:s + bs]); cap = Capture(want={("x_attn", l)}); net.forward(idx, cap, stop_after=l)
        x = cap.store[("x_attn", l)].double()
        beta = ((x - m) @ d) / (d @ d)
        xi = x - m - beta[..., None] * d
        for h in heads:
            a = acc[h]
            a["bs"] = a["bs"] + beta.sum(0); a["bss"] = a["bss"] + (beta ** 2).sum(0)
            a["bgt"] = a["bgt"] + (beta > 0.5).double().sum(0)
            r = torch.einsum("btd,d->bt", torch.einsum("de,bte->btd", M[h], xi), P[h]["u"])
            a["rs"] = a["rs"] + r.sum(0)
            q = torch.einsum("btd,ed->bte", x, hw.W_q[h].double()) + hw.b_q[h].double()
            E = torch.einsum("bie,bje->bij", q, torch.einsum("btd,ed->bte", xi, hw.W_k[h].double())) * hw.scale
            a["Es"] = a["Es"] + E.sum(0)
            v = (torch.einsum("btd,d->bt", torch.einsum("de,bte->btd", M[h], x), P[h]["u"]))[:, 1:].reshape(-1)
            bb = beta[:, 1:].reshape(-1)
            a["ols"] += np.array([len(v), float(bb.sum()), float((bb * bb).sum()), float(v.sum()), float((bb * v).sum())])
    for h in heads:
        a = acc[h]
        bbar = a["bs"] / n
        sinks = [int(j) for j in torch.nonzero(a["bgt"] / n > 0.9).flatten().tolist() if j >= 1]
        ind = torch.zeros(L, dtype=torch.float64, device=dev); ind[0] = 1; ind[sinks] = 1
        nn_, sb, sbb, sv, sbv = a["ols"]
        lamf = (sbv / nn_ - sb / nn_ * sv / nn_) / (sbb / nn_ - (sb / nn_) ** 2)
        jj = torch.arange(L, device=dev)
        tail = ((jj >= 16) & (ind == 0)).double(); early = ((jj >= 1) & (jj < 16) & (ind == 0)).double()
        bsd = (a["bss"] / n - bbar ** 2).clamp_min(0).sqrt()
        P[h].update({"betabar": bbar, "ind": ind, "tail": tail, "early": early, "sinks": sinks,
                     "beta_sd_early": bsd[:16].tolist(), "rbar": a["rs"] / n, "Ebar": a["Es"] / n,
                     "cf": sv / nn_ - lamf * sb / nn_, "lamf": lamf})
    print(f"calibration stats {time.time() - t0:.0f}s; sinks {[P[h]['sinks'] for h in heads]}", flush=True)
    # freeze: the first run writes the constants; later runs (e.g. --split test)
    # load them and record how far a fresh recomputation is from them
    fz = OUT_ROOT / "configs" / f"frozen_kv_{args.model}_L{l}_h{'-'.join(map(str, heads))}.pt"
    tens = lambda v: torch.is_tensor(v)
    if fz.exists():
        old = torch.load(fz, map_location=dev)
        drift = max(float((old[h][k].double() - P[h][k].double()).abs().max()) if tens(P[h][k]) else abs(float(old[h][k]) - float(P[h][k]))
                    for h in heads for k in P[h] if k not in ("sinks", "beta_sd_early"))
        P = {h: {k: (v.to(dev) if tens(v) else v) for k, v in old[h].items()} for h in heads}
        frozen_info = {"path": str(fz), "loaded": True, "max_drift_vs_recomputed": drift}
    else:
        fz.parent.mkdir(parents=True, exist_ok=True)
        torch.save({h: {k: (v.cpu() if tens(v) else v) for k, v in P[h].items()} for h in heads}, fz)
        frozen_info = {"path": str(fz), "loaded": False}
    print("frozen constants:", frozen_info, flush=True)

    conds = conditions()
    tgt_pos = torch.arange(L, device=dev, dtype=torch.float64)

    def sweep(offs, ridge=None):
        """Run every condition on offs. Returns per-condition arrays."""
        R = {c: {"z": {h: [] for h in heads}, "pred": [], "loss": [], "probe": []} for c in conds}
        for s in range(0, len(offs), bs):
            for c, cond in conds.items():
                idx, he, out = run_batch(offs[s:s + bs], cond, P, want=[feat_stage])
                for h in heads:
                    R[c]["z"][h].append(he.z[h][:, Pt].float().cpu())
                f = he.store[feat_stage]
                if is_lm:
                    R[c]["loss"].append(net.lm_loss(idx, out).float().cpu())
                else:
                    R[c]["pred"].append(out[:, Pt].float().cpu())
                if ridge is None:
                    R[c].setdefault("_ridge", Ridge(net.d, dev)).add(f, tgt_pos.expand(idx.shape[0], L))
                else:
                    R[c]["probe"].append(ridge[c].predict(f)[:, Pt].float().cpu())
        for c in conds:
            for h in heads:
                R[c]["z"][h] = torch.cat(R[c]["z"][h]).numpy()
            for k in ("pred", "loss", "probe"):
                R[c][k] = torch.cat(R[c][k]).numpy() if R[c][k] else None
        return R

    t0 = time.time()
    RC = sweep(sp["calibration"][:args.n_cal])
    ridges = {c: RC[c].pop("_ridge").solve() for c in conds}
    print(f"calibration sweep {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    split = sp[args.split][:args.n_eval]
    RE = sweep(split, ridges)
    print(f"{args.split} sweep {time.time() - t0:.0f}s ({len(split)} sequences)", flush=True)

    # ---- metrics ---------------------------------------------------------------
    rng = np.random.default_rng(0)
    nb = args.n_boot
    boot_idx = [rng.integers(0, len(split), len(split)) for _ in range(nb)]

    def err_stats(pred):
        """relMAE (median |e|/i) and MAE per bin, with sequence-bootstrap 95% CI."""
        sel = P_ >= 1
        e = np.abs(pred[:, sel] - P_[sel][None]); ii = P_[sel][None].repeat(len(pred), 0)
        out = {}
        for bn, (lo, hi) in bins.items():
            mk = (P_[sel] >= lo) & (P_[sel] <= hi)
            re = e[:, mk] / ii[:, mk]
            est = float(np.median(re)); mae = float(e[:, mk].mean())
            bs_ = [float(np.median(re[b])) for b in boot_idx]
            out[bn] = {"relmae": est, "relmae_ci": [float(np.quantile(bs_, .025)), float(np.quantile(bs_, .975))], "mae": mae}
        return out

    def paired_diff(pa, pb):
        """Bootstrap CI of relMAE(b) - relMAE(a) per bin, paired over sequences."""
        sel = P_ >= 1
        ea = np.abs(pa[:, sel] - P_[sel]) / np.maximum(P_[sel], 1); eb = np.abs(pb[:, sel] - P_[sel]) / np.maximum(P_[sel], 1)
        out = {}
        for bn, (lo, hi) in bins.items():
            mk = (P_[sel] >= lo) & (P_[sel] <= hi)
            ds = [float(np.median(eb[b][:, mk]) - np.median(ea[b][:, mk])) for b in boot_idx]
            out[bn] = {"diff": float(np.median(eb[:, mk]) - np.median(ea[:, mk])),
                       "ci": [float(np.quantile(ds, .025)), float(np.quantile(ds, .975))]}
        return out

    def iso(zc, ze):
        sel = P_ >= 1
        m_ = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(zc[:, sel].ravel(), np.tile(P_[sel], len(zc)).astype(float))
        out = np.zeros_like(ze, dtype=np.float64); out[:, sel] = m_.predict(ze[:, sel].ravel()).reshape(len(ze), -1)
        return out

    def curve(z):
        """Mean, SD, local ratio SD/(i|f'|) by bin; f' from a degree-6 fit in log i."""
        f = z.mean(0); sd = z.std(0); sel = P_ >= 4
        c = np.polyfit(np.log(P_[sel]), f[sel], 6)
        der = np.polyval(np.polyder(c), np.log(P_[sel])) / P_[sel]
        ratio = sd[sel] / (P_[sel] * np.abs(der))
        o = {}
        for bn, (lo, hi) in bins.items():
            mk = (P_[sel] >= max(lo, 4)) & (P_[sel] <= hi)
            o[bn] = {"local_ratio_median": float(np.median(ratio[mk])) if mk.any() else None,
                     "sd_median": float(np.median(sd[sel][mk])) if mk.any() else None,
                     "slope_i_fprime_median": float(np.median(P_[sel][mk] * np.abs(der[mk]))) if mk.any() else None,
                     "mean_shift_vs_orig": None}
        return o, f

    res = {"model": args.model, "layer": l, "heads": heads, "split": args.split, "n_eval": len(split), "n_cal": args.n_cal,
           "convention": "center = LN bias, global anchor, readout = unit(w_nonBOS - w_BOS)", "splits": prov,
           "frozen": frozen_info,
           "head_params": {h: {"c": P[h]["c"], "lambda": P[h]["lam"], "c_fit": P[h]["cf"], "lambda_fit": P[h]["lamf"],
                               "sinks": P[h]["sinks"], "beta_sd_early": P[h]["beta_sd_early"]} for h in heads}}
    # four-condition decomposition per head and pair
    dec = {}
    for h in heads:
        dec[h] = {}
        z11 = RE["orig"]["z"][h]
        for p in PAIRS:
            z10, z01, z00 = (RE[f"{p}_{k}"]["z"][h] for k in ("z10", "z01", "z00"))
            K, V, I = z10 - z00, z01 - z00, z11 - z10 - z01 + z00
            ident = float(np.abs(z00 + K + V + I - z11).max())
            per = {}
            for bn, (lo, hi) in bins.items():
                mk = (P_ >= max(lo, 1)) & (P_ <= hi)
                T = np.stack([z00[:, mk], K[:, mk], V[:, mk], I[:, mk]])            # [4, n, p]

                def stats(sel):
                    """Per-position covariance of the centered terms (centered within
                    each position across sequences), then the median over positions."""
                    Ts = T[:, sel]; Tc = Ts - Ts.mean(1, keepdims=True)
                    C = np.einsum("anp,bnp->abp", Tc, Tc) / Ts.shape[1]
                    o = {"var_z11": float(np.median(z11[sel][:, mk].var(0))), "var_z00": float(np.median(C[0, 0])),
                         "var_K": float(np.median(C[1, 1])), "var_V": float(np.median(C[2, 2])), "var_I": float(np.median(C[3, 3])),
                         "two_cov_K_V": float(np.median(2 * C[1, 2])), "two_cov_K_I": float(np.median(2 * C[1, 3])),
                         "two_cov_V_I": float(np.median(2 * C[2, 3])), "two_cov_z00_K": float(np.median(2 * C[0, 1])),
                         "two_cov_z00_V": float(np.median(2 * C[0, 2])), "two_cov_z00_I": float(np.median(2 * C[0, 3])),
                         "corr_K_V": float(np.median(C[1, 2] / np.sqrt(C[1, 1] * C[2, 2] + 1e-300)))}
                    return o, C
                est, C = stats(np.arange(T.shape[1]))
                bt = [stats(b)[0] for b in boot_idx]
                ci = {k: [float(np.quantile([x[k] for x in bt], .025)), float(np.quantile([x[k] for x in bt], .975))] for k in est}
                per[bn] = {**est, "ci": ci, "cov_z00_K_V_I": np.median(C, -1).tolist(),
                           "mean_terms": {nm: float(np.mean(T[i].mean(0))) for i, nm in enumerate(["z00", "K", "V", "I"])}}
            dec[h][p] = {"identity_max_abs": ident, "bins": per}
    res["decomposition"] = dec
    # scalar curves, decoders and endpoints per condition
    ev = {}
    orig_pred = RE["orig"]["pred"]
    for c in conds:
        e = {"z": {}}
        for h in heads:
            cv, f = curve(RE[c]["z"][h])
            off = float((RC[c]["z"][h][:, P_ >= 1] - RC["orig"]["z"][h][:, P_ >= 1]).mean())     # one calibration constant
            e["z"][h] = {"curve": cv, "iso_refit": err_stats(iso(RC[c]["z"][h], RE[c]["z"][h])),
                         "iso_frozen": err_stats(iso(RC["orig"]["z"][h], RE[c]["z"][h])),
                         "iso_frozen_offset_removed": err_stats(iso(RC["orig"]["z"][h], RE[c]["z"][h] - off)), "offset": off,
                         "mean_abs_shift_vs_orig": float(np.abs(f - RE["orig"]["z"][h].mean(0)).mean())}
        if is_lm:
            lo = RE[c]["loss"]; lo0 = RE["orig"]["loss"]
            dl = lo.mean(1) - lo0.mean(1)
            bs_ = [float(dl[b].mean()) for b in boot_idx]
            e["lm_loss"] = {"mean": float(lo.mean()), "delta_vs_orig": float(dl.mean()),
                            "delta_ci": [float(np.quantile(bs_, .025)), float(np.quantile(bs_, .975))]}
        else:
            e["frozen_head"] = err_stats(RE[c]["pred"])
            offp = float((RC[c]["pred"][:, P_ >= 1] - RC["orig"]["pred"][:, P_ >= 1]).mean())
            e["frozen_head_offset_removed"] = err_stats(RE[c]["pred"] - offp); e["pred_offset"] = offp
            if c != "orig":
                e["frozen_head_diff_vs_orig"] = paired_diff(orig_pred, RE[c]["pred"])
                e["frozen_head_offset_removed_diff_vs_orig"] = paired_diff(orig_pred, RE[c]["pred"] - offp)
        e["ridge_refit"] = err_stats(RE[c]["probe"])
        if c != "orig":
            e["ridge_refit_diff_vs_orig"] = paired_diff(RE["orig"]["probe"], RE[c]["probe"])
            for h in heads:
                e["z"][h]["iso_refit_diff_vs_orig"] = paired_diff(iso(RC["orig"]["z"][h], RE["orig"]["z"][h]), iso(RC[c]["z"][h], RE[c]["z"][h]))
        ev[c] = e
    res["conditions"] = ev

    # ---- D3: first-order sensitivity around the stabilized-feature state ----
    sens = {}
    offs = split[:min(len(split), 64)]
    for h in heads:
        Ph = P[h]
        rows = {}
        for pset, kmask in (("all", torch.ones_like(Ph["ind"])), ("tail", Ph["tail"]), ("early", Ph["early"]), ("sinks", Ph["ind"])):
            rows[pset] = {}
            for eps in (1.0, 0.5, 0.1, 0.01):
                zs, lin_v, lin_s = [], [], []
                for s in range(0, len(offs), bs):
                    idx = net.batch(offs[s:s + bs]); cap = Capture(want={("x_attn", l), ("q", l)}); net.forward(idx, cap, stop_after=l)
                    x = cap.store[("x_attn", l)].double()
                    beta = ((x - Ph["m"]) @ Ph["d"]) / (Ph["d"] @ Ph["d"])
                    dx = ((beta - Ph["betabar"]) * kmask)[..., None] * Ph["d"]
                    xref = x - dx
                    q = cap.store[("q", l)][:, h].double()
                    def zfun(xx):
                        k = torch.einsum("btd,ed->bte", xx, hw.W_k[h].double()) + hw.b_k[h].double()
                        sc = torch.einsum("bie,bje->bij", q, k) * hw.scale
                        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1)
                        a = torch.softmax(sc.masked_fill(mask, float("-inf")), -1)
                        v = torch.einsum("btd,d->bt", torch.einsum("de,bte->btd", M[h], xx), Ph["u"])
                        return a, sc, v, torch.einsum("bij,bj->bi", a, v)
                    a0, s0, v0, z0 = zfun(xref)
                    _, _, _, z1 = zfun(xref + eps * dx)
                    dv = eps * torch.einsum("btd,d->bt", torch.einsum("de,bte->btd", M[h], dx), Ph["u"])
                    ds = eps * torch.einsum("bie,bje->bij", q, torch.einsum("btd,ed->bte", dx, hw.W_k[h].double())) * hw.scale
                    lv = torch.einsum("bij,bj->bi", a0, dv)
                    ls = torch.einsum("bij,bij->bi", a0 * (v0[:, None, :] - z0[..., None]), torch.where(torch.isfinite(s0), ds, torch.zeros_like(ds)))
                    zs.append((z1 - z0)[:, Pt].cpu().numpy()); lin_v.append(lv[:, Pt].cpu().numpy()); lin_s.append(ls[:, Pt].cpu().numpy())
                dz, lv, ls = np.concatenate(zs), np.concatenate(lin_v), np.concatenate(lin_s)
                per = {}
                for bn, (lo, hi) in bins.items():
                    mk = (P_ >= max(lo, 1)) & (P_ <= hi)
                    a_, b_, t_ = lv[:, mk], ls[:, mk], dz[:, mk]
                    cov = ((a_ - a_.mean(0)) * (b_ - b_.mean(0))).mean(0)
                    per[bn] = {"lin_rel_err": float(np.median(np.abs(t_ - a_ - b_).mean(0) / (np.abs(t_).mean(0) + 1e-300))),
                               "var_value_term": float(np.median(a_.var(0))), "var_score_term": float(np.median(b_.var(0))),
                               "two_cov": float(np.median(2 * cov)), "var_exact": float(np.median(t_.var(0))),
                               "corr_value_score": float(np.median(cov / np.sqrt(a_.var(0) * b_.var(0) + 1e-300)))}
                rows[pset][str(eps)] = per
        sens[h] = rows
    res["sensitivity_D3"] = sens

    od = OUT_ROOT / "kv" / args.model
    tag = f"L{l}_h{'-'.join(map(str, heads))}_{args.split}"
    jdump(res, od / f"kv_{tag}.json")
    np.savez_compressed(od / f"kv_{tag}_scalars.npz", positions=P_, offsets=split,
                        **{f"{c}|z|h{h}": RE[c]["z"][h] for c in conds for h in heads},
                        **({f"{c}|pred": RE[c]["pred"] for c in conds} if not is_lm else {f"{c}|loss": RE[c]["loss"] for c in conds}))
    print("wrote", od / f"kv_{tag}.json", flush=True)
    for h in heads:
        print(f"head {h}:", {c: (round(ev[c]['z'][h]['iso_refit'][list(bins)[-1]]['relmae'], 4),
                                round(ev[c]['frozen_head'][list(bins)[-1]]['relmae'], 4) if not is_lm else round(ev[c]['lm_loss']['delta_vs_orig'], 4))
                             for c in conds}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--heads", nargs="+", required=True, help="layer:head, all in one layer")
    ap.add_argument("--split", default="validation", choices=["validation", "test", "test_strict"])
    ap.add_argument("--n-cal", type=int, default=320)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    run(ap.parse_args())
