"""Task C screen: feature, score and value fits for every layer and head.

Conventions (frozen on calibration before validation is read):
  center m      ln_bias  : attention-input LN bias (zero without a bias)
                cal_mean : calibration mean of nonBOS attention inputs
  anchor d      global   : calibration mean of x_0, minus m
                seq      : each sequence's own x_0 minus m (native LM only)
  readout u     dw       : unit(w_nonBOS - w_BOS), w = B_OV applied to the
                           calibration mean nonBOS / first-token input
                bos      : unit(w_BOS)
beta_j = <x_j - m, d>/|d|^2, xi_j = x_j - m - beta_j d,
B_i = q_i.W_K d/sqrt(dh), E_ij = q_i.W_K xi_j/sqrt(dh),
v_j = <B_OV x_j, u> = c + lambda beta_j + r_j (structural, exact),
z_i = sum_j a_ij v_j = c + lambda t_i + R_i.

Attention ladder (each a softmax over visible keys):
  K2  observed beta, observed B_i, E omitted
  K3  observed beta, one calibration constant B (mean B_i over i >= 16)
  K4  calibration mean beta profile, constant B
  K5  two classes: BOS and calibration sinks at beta = 1, ordinary keys at
      the calibration mean ordinary beta, constant B
Value models: Vs structural c + lambda beta; Vf OLS fit on calibration keys.

Passes: A (calibration means), B (calibration fits), C (calibration
evaluation, for decoder fitting), D (validation evaluation). The test split
is not read.

Usage: python cc_screen.py --model 1h_s42 --device cuda:0 [--batch-size 4]
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from cc_common import OUT_ROOT, SPECS, Capture, bins_for, jdump, load_net, splits_for

N_CAL, N_VAL = 320, 160
RUNGS = ["K2", "K3", "K4", "K5"]
ZVARS = ["z", "zVs", "zVf", "zK2", "zK3", "zK4", "zK5", "zK3Vf", "zK2Vs", "t", "R"]


def positions(L):
    p = list(range(min(L, 128)))
    if L > 128:
        p += list(range(128, L, 4))
    return np.array(p)


def query_sample(L):
    q = np.unique(np.round(np.geomspace(1, L - 1, 64)).astype(int))
    return q


def qbin(i, L):
    return "early" if i < 16 else ("middle" if i < 128 else "late")


class Acc:
    """Sums keyed by name, fp64."""

    def __init__(self):
        self.s = {}

    def add(self, k, v):
        v = v.double() if torch.is_tensor(v) else torch.tensor(float(v), dtype=torch.float64)
        self.s[k] = self.s[k] + v if k in self.s else v.clone()

    def get(self, k):
        return self.s[k]


def masked_softmax(sc, T, dev):
    mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
    return torch.softmax(sc.masked_fill(mask, float("-inf")), -1)


@torch.no_grad()
def run(args):
    dev, bs = args.device, args.batch_size
    net, _ = load_net(args.model, dev)
    spec = net.spec
    L, H, nl = spec.L, net.H, net.n_layer
    sp, prov = splits_for(args.model)
    P = positions(L)
    Pt = torch.as_tensor(P, device=dev)
    Qs = query_sample(L)
    Qt = torch.as_tensor(Qs, device=dev)
    qnames = np.array([qbin(int(i), L) for i in Qs])
    qgroups = {g: torch.as_tensor(qnames == g, device=dev) for g in np.unique(qnames)}
    convs = [("ln_bias", "global"), ("cal_mean", "global")] + ([] if spec.bos else [("ln_bias", "seq")])
    want = {(s, l) for l in range(nl) for s in ("x_attn", "scores", "weights")}

    def batches(name, n):
        offs = sp[name][:n]
        for s in range(0, len(offs), bs):
            idx = net.batch(offs[s:s + bs])
            cap = Capture(want=want)
            net.forward(idx, cap)
            yield idx, cap.store

    # ---- pass A: calibration means ------------------------------------------
    t0 = time.time()
    A = Acc(); n_seq = 0
    for idx, st in batches("calibration", N_CAL):
        n_seq += idx.shape[0]
        for l in range(nl):
            x = st[("x_attn", l)].double()
            A.add(f"x0_{l}", x[:, 0].sum(0)); A.add(f"xnb_{l}", x[:, 1:].sum((0, 1)))
            A.add(f"x0sq_{l}", (x[:, 0] ** 2).sum(0))
    lay = []
    for l in range(nl):
        x0 = A.get(f"x0_{l}") / n_seq
        xnb = A.get(f"xnb_{l}") / (n_seq * (L - 1))
        g, b = net.ln_params(net.blocks[l].ln_1)
        M = net.ov(l)
        wb, wn = torch.einsum("hdf,f->hd", M, x0), torch.einsum("hdf,f->hd", M, xnb)
        U = {"dw": (wn - wb) / (wn - wb).norm(dim=-1, keepdim=True), "bos": wb / wb.norm(dim=-1, keepdim=True)}
        lay.append({"x0": x0, "xnb": xnb, "m": {"ln_bias": b, "cal_mean": xnb}, "M": M, "U": U,
                    "x0_sd_rel": float((A.get(f"x0sq_{l}") / n_seq - x0 ** 2).clamp_min(0).sum().sqrt() / x0.norm())})
    print(f"pass A {time.time() - t0:.0f}s", flush=True)

    def feat(x, l, conv):
        cname, aname = conv
        m = lay[l]["m"][cname]
        if aname == "global":
            d = (lay[l]["x0"] - m)[None].expand(x.shape[0], -1)
        else:
            d = x[:, 0] - m
        beta = torch.einsum("btd,bd->bt", x - m, d) / (d * d).sum(-1, keepdim=True)
        xi = x - m - beta[..., None] * d[:, None]
        return m, d, beta, xi

    def head_terms(x, l, h, conv, m, d, beta, xi):
        hw = net.hw[l]
        q = torch.einsum("btd,ed->bte", x, hw.W_q[h].double()) + hw.b_q[h].double()
        Kd = torch.einsum("ed,bd->be", hw.W_k[h].double(), d)
        Bi = torch.einsum("bte,be->bt", q, Kd) * hw.scale
        return q, Bi

    # ---- pass B: calibration fits -------------------------------------------
    t0 = time.time()
    Bacc = Acc(); n_seq = 0
    for idx, st in batches("calibration", N_CAL):
        n_seq += idx.shape[0]
        for l in range(nl):
            x = st[("x_attn", l)].double()
            Aw = st[("weights", l)].double()
            for ci, conv in enumerate(convs):
                m, d, beta, xi = feat(x, l, conv)
                Bacc.add(f"bsum_{l}_{ci}", beta.sum(0)); Bacc.add(f"bsq_{l}_{ci}", (beta ** 2).sum(0))
                Bacc.add(f"bgt_{l}_{ci}", (beta > 0.5).double().sum(0))
                for h in range(H):
                    _, Bi = head_terms(x, l, h, conv, m, d, beta, xi)
                    Bacc.add(f"Bi_{l}_{ci}_{h}", Bi[:, 16:].sum()); Bacc.add(f"Bin_{l}_{ci}_{h}", Bi[:, 16:].numel())
                    Y = torch.einsum("df,btf->btd", lay[l]["M"][h], x)
                    for un, U in lay[l]["U"].items():
                        v = Y[:, 1:] @ U[h]; bb = beta[:, 1:]
                        for grp, sel in (("all", slice(None)), ("tail", slice(15, None))):
                            vv, be = v[:, sel], bb[:, sel]
                            k = f"ols_{l}_{ci}_{h}_{un}_{grp}"
                            for nm, val in (("n", vv.numel()), ("b", be.sum()), ("bb", (be * be).sum()),
                                            ("v", vv.sum()), ("bv", (be * vv).sum()), ("vv", (vv * vv).sum())):
                                Bacc.add(k + nm, val)
            Bacc.add(f"abos_{l}", Aw[:, :, 1:, 0].sum((0, 2)))
            inv = 1.0 / torch.arange(1, L, device=dev, dtype=torch.float64)
            Bacc.add(f"anb_{l}", ((1 - Aw[:, :, 1:, 0]) * inv).sum((0, 2)))
    fit = {}
    for l in range(nl):
        for ci, conv in enumerate(convs):
            bmean = Bacc.get(f"bsum_{l}_{ci}") / n_seq
            frac = Bacc.get(f"bgt_{l}_{ci}") / n_seq
            sinks = [int(j) for j in torch.nonzero(frac > 0.9).flatten().tolist() if j >= 1]
            ordm = torch.ones(L, dtype=torch.bool, device=dev); ordm[0] = False; ordm[sinks] = False
            tail_ord = ordm.clone(); tail_ord[:16] = False
            b_ord = float(bmean[tail_ord].mean()) if tail_ord.any() else float(bmean[ordm].mean())
            cls = torch.full((L,), b_ord, device=dev, dtype=torch.float64); cls[0] = 1.0; cls[sinks] = 1.0
            F_ = {"beta_mean": bmean, "sinks": sinks, "ordinary_mask": ordm, "beta_ord": b_ord, "beta_cls": cls,
                  "beta_sd": (Bacc.get(f"bsq_{l}_{ci}") / n_seq - bmean ** 2).clamp_min(0).sqrt(), "heads": {}}
            for h in range(H):
                Bc = float(Bacc.get(f"Bi_{l}_{ci}_{h}") / Bacc.get(f"Bin_{l}_{ci}_{h}"))
                hf = {"B_const": Bc, "ols": {}}
                for un in lay[l]["U"]:
                    for grp in ("all", "tail"):
                        k = f"ols_{l}_{ci}_{h}_{un}_{grp}"
                        n_, b_, bb_, v_, bv_ = (float(Bacc.get(k + s)) for s in ("n", "b", "bb", "v", "bv"))
                        lam = (bv_ / n_ - b_ / n_ * v_ / n_) / (bb_ / n_ - (b_ / n_) ** 2)
                        hf["ols"][(un, grp)] = (v_ / n_ - lam * b_ / n_, lam)
                F_["heads"][h] = hf
            fit[(l, ci)] = F_
    r_h = {l: (Bacc.get(f"abos_{l}") / Bacc.get(f"anb_{l}")).cpu().numpy() for l in range(nl)}
    print(f"pass B {time.time() - t0:.0f}s", flush=True)

    # ---- passes C/D: evaluation ---------------------------------------------
    def evaluate(split, n, collect_ss):
        store = {}
        S = Acc()
        for idx, st in batches(split, n):
            b = idx.shape[0]
            for l in range(nl):
                x = st[("x_attn", l)].double()
                Aw = st[("weights", l)].double()
                sc = st[("scores", l)].double()
                hw = net.hw[l]
                for ci, conv in enumerate(convs):
                    F_ = fit[(l, ci)]
                    m, d, beta, xi = feat(x, l, conv)
                    key = (l, ci)
                    store.setdefault(("beta",) + key, []).append(beta[:, Pt].float().cpu())
                    for h in range(H):
                        hf = F_["heads"][h]
                        q, Bi = head_terms(x, l, h, conv, m, d, beta, xi)
                        store.setdefault(("Bi",) + key + (h,), []).append(Bi[:, Pt].float().cpu())
                        a = Aw[:, h]
                        Bc = hf["B_const"]
                        lad = {"K2": masked_softmax(Bi[:, :, None] * beta[:, None, :], L, dev),
                               "K3": masked_softmax(Bc * beta[:, None, :].expand(-1, L, -1), L, dev),
                               "K4": masked_softmax((Bc * F_["beta_mean"])[None, None].expand(b, L, -1), L, dev),
                               "K5": masked_softmax((Bc * F_["beta_cls"])[None, None].expand(b, L, -1), L, dev)}
                        for rn, ah in lad.items():
                            tv = 0.5 * (ah - a).abs().sum(-1)                      # [b,T]
                            store.setdefault(("tv", rn) + key + (h,), []).append(tv[:, Pt].float().cpu())
                            store.setdefault(("abos", rn) + key + (h,), []).append(ah[:, Pt, 0].float().cpu())
                        if ci == 0:
                            store.setdefault(("abos", "obs", l, h), []).append(a[:, Pt, 0].float().cpu())
                        # score reconstruction (E omitted) on sampled queries
                        if collect_ss:
                            qq = q[:, Qt]
                            Kxi = torch.einsum("ed,btd->bte", hw.W_k[h].double(), xi)
                            E = torch.einsum("bqe,bte->bqt", qq, Kxi) * hw.scale
                            ds = sc[:, h, Qt, :1] - sc[:, h, Qt, :]
                            pred = (1 - beta)[:, None, :] * Bi[:, Qt, None]
                            vis = (torch.arange(L, device=dev)[None, :] <= Qt[:, None])[None].expand(b, -1, -1)
                            zero = torch.zeros((), device=dev, dtype=torch.float64)
                            for kg, kmask in (("all", torch.ones(L, dtype=torch.bool, device=dev)), ("ord", F_["ordinary_mask"])):
                                sel = vis & kmask[None, None]
                                cnt = sel.sum(-1, keepdim=True).clamp_min(1)

                                def rowc(t):
                                    t = torch.where(sel, t, zero)
                                    return torch.where(sel, t - t.sum(-1, keepdim=True) / cnt, zero)
                                parts = {"res": rowc(ds - pred), "tot": rowc(ds), "E": rowc(E), "pred": rowc(pred)}
                                for pn, t in parts.items():
                                    per_q = (t ** 2).sum((0, 2))                       # [nQ]
                                    for g, qm in qgroups.items():
                                        S.add(f"ss_{l}_{ci}_{h}_{kg}_{g}{pn}", per_q[qm].sum())
                        # key read direction per query against d
                        if collect_ss and ci == 0:
                            rK = torch.einsum("bte,ed->btd", q, hw.W_k[h].double()) * hw.scale
                            cK = (rK @ d[0]) / (rK.norm(dim=-1) * d[0].norm())
                            S.add(f"cosK_{l}_{h}", cK[:, 1:].sum()); S.add(f"cosKn_{l}_{h}", cK[:, 1:].numel())
                            S.add(f"rKsum_{l}_{h}", rK[:, 1:].sum((0, 1)))
                        Y = torch.einsum("df,btf->btd", lay[l]["M"][h], x)
                        for un, U in lay[l]["U"].items():
                            u = U[h]
                            v = Y @ u
                            Mm = lay[l]["M"][h]
                            c = (Mm @ m) @ u
                            lam = torch.einsum("de,be->bd", Mm, d) @ u                 # [b]
                            r = torch.einsum("btd,d->bt", torch.einsum("de,bte->btd", Mm, xi), u)
                            vs = c + lam[:, None] * beta
                            cf, lf = hf["ols"][(un, "all")]
                            vf = cf + lf * beta
                            Z = {"z": a @ v[..., None], "zVs": a @ vs[..., None], "zVf": a @ vf[..., None],
                                 "zK2": lad["K2"] @ v[..., None], "zK3": lad["K3"] @ v[..., None],
                                 "zK4": lad["K4"] @ v[..., None], "zK5": lad["K5"] @ v[..., None],
                                 "zK3Vf": lad["K3"] @ vf[..., None], "zK2Vs": lad["K2"] @ vs[..., None],
                                 "t": a @ beta[..., None], "R": a @ r[..., None]}
                            for zn, zz in Z.items():
                                store.setdefault((zn, un) + key + (h,), []).append(zz[..., 0][:, Pt].float().cpu())
                            store.setdefault(("lam", un) + key + (h,), []).append(lam.float().cpu())
                            store.setdefault(("c", un) + key + (h,), [c.float().cpu()])
                            if collect_ss:
                                for kg, ks in (("all", slice(1, None)), ("tail", slice(16, None))):
                                    vv = v[:, ks]
                                    pre = f"vv_{l}_{ci}_{h}_{un}_{kg}"
                                    S.add(pre + "n", vv.numel()); S.add(pre + "s", vv.sum()); S.add(pre + "ss", (vv * vv).sum())
                                    S.add(pre + "rs", ((vv - vs[:, ks]) ** 2).sum()); S.add(pre + "rf", ((vv - vf[:, ks]) ** 2).sum())
                                    S.add(pre + "bs", (vv - vs[:, ks]).sum()); S.add(pre + "bf", (vv - vf[:, ks]).sum())
                                    ordk = F_["ordinary_mask"][ks]
                                    rr = r[:, ks][:, ordk]
                                    lb = (lam[:, None] * beta)[:, ks][:, ordk]
                                    S.add(pre + "rn", rr.numel()); S.add(pre + "r1", rr.sum()); S.add(pre + "r2", (rr * rr).sum())
                                    S.add(pre + "lb1", lb.sum()); S.add(pre + "lb2", (lb * lb).sum()); S.add(pre + "lbr", (lb * rr).sum())
                                    S.add(pre + "lam", lam.sum()); S.add(pre + "lamn", lam.numel())
        out = {k: (torch.cat(v).numpy() if len(v) > 1 or v[0].dim() > 0 else v[0].numpy()) for k, v in store.items()}
        return out, S

    t0 = time.time()
    cal, _ = evaluate("calibration", N_CAL, False)
    print(f"pass C {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    val, SS = evaluate("validation", N_VAL, True)
    nval = len(sp["validation"][:N_VAL])
    print(f"pass D {time.time() - t0:.0f}s", flush=True)

    # ---- summaries -----------------------------------------------------------
    bins = bins_for(L)

    def decode(zc, ze):
        """Isotonic z -> position on calibration; errors on validation by bin."""
        sel = P >= 1
        iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        iso.fit(zc[:, sel].reshape(-1), np.tile(P[sel], len(zc)).astype(np.float64))
        pred = iso.predict(ze[:, sel].reshape(-1)).reshape(len(ze), -1)
        e = np.abs(pred - P[sel][None]); ii = P[sel][None].repeat(len(ze), 0)
        o = {"increasing": bool(iso.increasing_)}
        for bn, (lo, hi) in bins.items():
            mk = (ii >= lo) & (ii <= hi)
            o[bn] = {"relmae": float(np.median(e[mk] / ii[mk])), "mae": float(e[mk].mean())}
        return o

    def local_ratio(zc, ze):
        """SD over sequences of z_i / (i |f'(i)|), f = calibration mean, fitted
        as a degree-6 polynomial in log i over i >= 4 on the position grid."""
        f = zc.mean(0); sel = P >= 4
        cpoly = np.polyfit(np.log(P[sel]), f[sel], 6)
        der = np.polyval(np.polyder(cpoly), np.log(P[sel])) / P[sel]
        ratio = ze[:, sel].std(0) / (P[sel] * np.abs(der))
        flat = np.abs(der) * P[sel] < 1e-3 * max(f[sel].max() - f[sel].min(), 1e-30)
        o = {}
        for bn, (lo, hi) in bins.items():
            mk = (P[sel] >= max(lo, 4)) & (P[sel] <= hi)
            o[bn] = {"median": float(np.median(ratio[mk])) if mk.any() else None,
                     "n_flat": int(flat[mk].sum()),
                     "n_sign_changes_fprime": int((np.diff(np.sign(der[mk])) != 0).sum()) if mk.sum() > 1 else 0}
        return o

    def var_split(zv, conv_seq):
        """At each position: var(lambda t), var(R), 2 cov, var(z); medians by bin."""
        lam = zv["lam"][:, None] if conv_seq else zv["lam"].mean()
        lt = lam * zv["t"]; R = zv["R"]
        o = {}
        for bn, (lo, hi) in bins.items():
            mk = (P >= max(lo, 1)) & (P <= hi)
            a, bb = lt[:, mk], R[:, mk]
            va, vb = a.var(0), bb.var(0)
            cv = ((a - a.mean(0)) * (bb - bb.mean(0))).mean(0)
            vz = (a + bb).var(0)
            o[bn] = {"var_lam_t": float(np.median(va)), "var_R": float(np.median(vb)), "two_cov": float(np.median(2 * cv)),
                     "var_z": float(np.median(vz)), "corr_lam_t_R": float(np.median(cv / np.sqrt(va * vb + 1e-300)))}
        return o

    rows = []
    for l in range(nl):
        for ci, conv in enumerate(convs):
            F_ = fit[(l, ci)]
            key = (l, ci)
            for h in range(H):
                row = {"model": args.model, "family": spec.family, "layer": l, "head": h,
                       "center": conv[0], "anchor": conv[1], "r_h": float(r_h[l][h]),
                       "B_const": F_["heads"][h]["B_const"], "sinks": F_["sinks"], "beta_ord": F_["beta_ord"]}
                Bi = val[("Bi",) + key + (h,)]
                tailP = P >= 16
                row["B_i"] = {"mean_tail": float(Bi[:, tailP].mean()), "sd_seq_tail_median": float(np.median(Bi[:, tailP].std(0))),
                              "mean_early": float(Bi[:, (P >= 1) & (P < 16)].mean()),
                              "drift_tail": float(Bi[:, tailP].mean(0)[-1] - Bi[:, tailP].mean(0)[0])}
                mt = row["B_i"]["mean_tail"]
                row["B_i"]["cv_tail"] = row["B_i"]["sd_seq_tail_median"] / abs(mt) if abs(mt) > 0.25 else None
                sc = {}
                for kg in ("all", "ord"):
                    for g in ("early", "middle", "late"):
                        pre = f"ss_{l}_{ci}_{h}_{kg}_{g}"
                        if pre + "res" in SS.s and float(SS.get(pre + "tot")) > 0:
                            sp_, se_, st_ = (float(SS.get(pre + s)) for s in ("pred", "E", "tot"))
                            sc[f"{kg}_{g}"] = {"r2_E_omitted": 1 - float(SS.get(pre + "res")) / st_,
                                               "sd_E_over_sd_pred": float(np.sqrt(se_ / max(sp_, 1e-300))),
                                               # delta s = pred - E, so cov(pred, E) = (var pred + var E - var ds)/2
                                               "corr_pred_E": (sp_ + se_ - st_) / 2 / np.sqrt(max(sp_ * se_, 1e-300))}
                row["score_fit"] = sc
                lad = {}
                obs = val[("abos", "obs", l, h)]
                for rn in RUNGS:
                    tv = val[("tv", rn) + key + (h,)]; ab = val[("abos", rn) + key + (h,)]
                    lad[rn] = {}
                    for bn, (lo, hi) in bins.items():
                        mk = (P >= lo) & (P <= hi)
                        mo, mp = obs[:, mk].mean(0), ab[:, mk].mean(0)
                        so, spp = obs[:, mk].std(0), ab[:, mk].std(0)
                        oc, pc = obs[:, mk] - mo, ab[:, mk] - mp
                        ok = (so > 0) & (spp > 0)
                        cc = ((oc * pc).mean(0)[ok] / (so[ok] * spp[ok])).tolist()
                        lad[rn][bn] = {"tv_mean": float(tv[:, mk].mean()),
                                       "abos_mean_curve_relerr": float(np.median(np.abs(mp - mo) / np.maximum(mo, 1e-12))),
                                       "abos_sd_ratio": float(np.median(spp / np.maximum(so, 1e-15))),
                                       "abos_seq_corr_median": float(np.median(cc)) if cc else None}
                row["attention_ladder"] = lad
                cosk = SS.s.get(f"cosK_{l}_{h}")
                if cosk is not None and ci == 0:
                    row["cos_key_read_vs_d_mean"] = float(cosk / SS.get(f"cosKn_{l}_{h}"))
                for un in lay[l]["U"]:
                    r2 = dict(row); r2["readout"] = un
                    vf = {}
                    for kg in ("all", "tail"):
                        pre = f"vv_{l}_{ci}_{h}_{un}_{kg}"
                        n_ = float(SS.get(pre + "n")); s_ = float(SS.get(pre + "s")); ss_ = float(SS.get(pre + "ss"))
                        tot = ss_ - s_ * s_ / n_
                        rn_ = float(SS.get(pre + "rn"))
                        lam_m = float(SS.get(pre + "lam") / SS.get(pre + "lamn"))
                        rsd = float(np.sqrt(max(float(SS.get(pre + "r2")) / rn_ - (float(SS.get(pre + "r1")) / rn_) ** 2, 0))) if rn_ > 0 else None
                        if rn_ > 0:
                            m_r, m_lb = float(SS.get(pre + "r1")) / rn_, float(SS.get(pre + "lb1")) / rn_
                            v_r = float(SS.get(pre + "r2")) / rn_ - m_r ** 2
                            v_lb = float(SS.get(pre + "lb2")) / rn_ - m_lb ** 2
                            c_lr = float(SS.get(pre + "lbr")) / rn_ - m_lb * m_r
                            ordcov = {"var_lam_beta": v_lb, "var_r": v_r, "two_cov": 2 * c_lr,
                                      "var_v": v_lb + v_r + 2 * c_lr,
                                      "corr": c_lr / np.sqrt(max(v_lb * v_r, 1e-300))}
                        else:
                            ordcov = None
                        vf[kg] = {"ordinary_key_split": ordcov,
                                  "r2_structural": 1 - float(SS.get(pre + "rs")) / tot, "r2_fitted": 1 - float(SS.get(pre + "rf")) / tot,
                                  "bias_structural": float(SS.get(pre + "bs")) / n_, "bias_fitted": float(SS.get(pre + "bf")) / n_,
                                  "rms_structural": float(np.sqrt(float(SS.get(pre + "rs")) / n_)),
                                  "rms_fitted": float(np.sqrt(float(SS.get(pre + "rf")) / n_)),
                                  "lambda_structural": lam_m, "sd_r_ordinary": rsd,
                                  "sd_r_over_abs_lambda": rsd / abs(lam_m) if (rsd is not None and abs(lam_m) > 1e-9) else None}
                    cf, lf = F_["heads"][h]["ols"][(un, "all")]
                    vf["fitted_coeffs"] = {"c": cf, "lambda": lf}
                    r2["value_fit"] = vf
                    if ci == 0:
                        rV = lay[l]["M"][h].T @ lay[l]["U"][un][h]
                        d0 = lay[l]["x0"] - lay[l]["m"]["ln_bias"]
                        rK = SS.get(f"rKsum_{l}_{h}")
                        r2["cos_value_read_vs_d"] = float(rV @ d0 / (rV.norm() * d0.norm()))
                        r2["cos_value_read_vs_mean_key_read"] = float(rV @ rK / (rV.norm() * rK.norm()))
                        r2["cos_mean_key_read_vs_d"] = float(rK @ d0 / (rK.norm() * d0.norm()))
                    zc = {zn: cal[(zn, un) + key + (h,)] for zn in ZVARS}
                    zv = {zn: val[(zn, un) + key + (h,)] for zn in ZVARS}
                    zv["lam"] = val[("lam", un) + key + (h,)]
                    r2["decode_refit"] = {zn: decode(zc[zn], zv[zn]) for zn in ZVARS if zn not in ("R",)}
                    r2["decode_frozen"] = {zn: decode(zc["z"], zv[zn]) for zn in ZVARS if zn.startswith("zK") or zn.startswith("zV")}
                    r2["local_ratio"] = {zn: local_ratio(zc[zn], zv[zn]) for zn in ("z", "zVs", "zK2", "zK3", "zK3Vf")}
                    r2["var_split"] = var_split(zv, conv[1] == "seq")
                    rows.append(r2)
    od = OUT_ROOT / "screen" / args.model
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "head_metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    # per-sequence scalars, primary convention (ln_bias, global): enough to redo
    # decoding and covariance without a new activation dump
    prim = {}
    for split, D in (("cal", cal), ("val", val)):
        for k, v in D.items():
            if k[0] in ZVARS + ["lam"] and len(k) == 5 and k[3] == 0:          # (name, readout, l, ci, h)
                prim[f"{split}|{k[0]}|{k[1]}|L{k[2]}|h{k[4]}"] = v
            elif k[0] == "beta" and k[2] == 0:                                    # (beta, l, ci)
                prim[f"{split}|beta|L{k[1]}"] = v
            elif k[0] == "Bi" and k[2] == 0:                                      # (Bi, l, ci, h)
                prim[f"{split}|Bi|L{k[1]}|h{k[3]}"] = v
            elif k[0] == "abos" and k[1] == "obs":                                # (abos, obs, l, h)
                prim[f"{split}|abos_obs|L{k[2]}|h{k[3]}"] = v
    np.savez_compressed(od / "scalars_primary.npz", positions=P, **prim)
    curves = {}
    for (l, ci), F_ in fit.items():
        curves[f"beta_mean|L{l}|c{ci}"] = F_["beta_mean"].cpu().numpy()
        curves[f"beta_sd|L{l}|c{ci}"] = F_["beta_sd"].cpu().numpy()
    np.savez_compressed(od / "calibration_curves.npz", **curves)
    meta = {"model": args.model, "conventions": convs, "positions": "0..127 then every 4th to L-1" if L > 128 else "all",
            "query_sample": Qs.tolist(), "n_cal": N_CAL, "n_val": N_VAL, "splits": prov,
            "x0_sd_rel": [lay[l]["x0_sd_rel"] for l in range(nl)],
            "sinks": {f"L{l}|c{ci}": fit[(l, ci)]["sinks"] for (l, ci) in fit},
            "r_h": {f"L{l}": r_h[l].tolist() for l in range(nl)}}
    jdump(meta, od / "meta.json")
    print(f"wrote {len(rows)} rows to {od}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    run(ap.parse_args())
