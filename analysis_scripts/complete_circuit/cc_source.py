"""Task B1/B2: where the block-l BOS feature comes from, and what the MLPs do to it.

With the primary center m = LN bias, the block-l feature of key j is exactly

    beta_j = <LN(h_j) - b, d>/|d|^2 = <h_j, rho>/sigma_j,
    rho    = center(gamma * d)/|d|^2      (center: subtract the feature mean),

where h_j is the block-l input residual and sigma_j its LayerNorm scale. The
numerator n_j = <h_j, rho> is linear in the residual, so it splits exactly
into the embedding, each earlier head's BOS-key part and nonBOS-key part,
each attention bias, and each MLP update. sigma_j is a separate factor.

B2, for every earlier block b and two directions w (unit rho of the target
layer; unit total block-b BOS write sum_h B_OV,h x_0):
  a_j = <h~_j, w>  (MLP-input residual coefficient), g_j = -<MLP_j, w>,
  fits on calibration of g = c0 + k a (linear) and of the capped map
  g = c0 + G r (a - a0)/sqrt((a - a0)^2 + r^2), scored on validation;
  local sensitivity s_j = <J w, w> of h~ -> h~ + MLP(LN(h~)) by forward-mode AD
  in fp64, and the same with the LN scale fixed at its calibration mean for
  that position (a labeled intervention).

Usage: python cc_source.py --model 1h_s42 --layer 1 --device cuda:0
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit

from cc_common import EPS_LN, OUT_ROOT, SPECS, Capture, bins_for, jdump, load_net, splits_for

N_CAL, N_VAL = 160, 160
GROUPS = {"bos": (0, 0), "pos1": (1, 1), "early": (2, 15), "middle": (16, 127), "late": (128, 10 ** 9)}


def capped(a, c0, G, r, a0):
    x = a - a0
    return c0 + G * r * x / np.sqrt(x * x + r * r)


@torch.no_grad()
def run(args):
    dev, bs = args.device, args.batch_size
    net, _ = load_net(args.model, dev)
    spec = net.spec
    L, H, tl = spec.L, net.H, args.layer
    sp, prov = splits_for(args.model)
    want = {(s, l) for l in range(tl + 1) for s in ("resid_in", "x_attn", "weights", "resid_mid", "mlp_out", "x_mlp")}

    def batches(name, n):
        offs = sp[name][:n]
        for s in range(0, len(offs), bs):
            idx = net.batch(offs[s:s + bs])
            cap = Capture(want=want)
            net.forward(idx, cap, stop_after=tl)
            yield idx, cap.store

    # ---- calibration: anchor of the target layer -----------------------------
    x0s, nseq = 0, 0
    sig_sum = {b: 0 for b in range(tl)}
    for idx, st in batches("calibration", N_CAL):
        x0s = x0s + st[("x_attn", tl)][:, 0].double().sum(0); nseq += idx.shape[0]
        for b in range(tl):
            hm = st[("resid_mid", b)].double()
            sig_sum[b] = sig_sum[b] + (hm.var(-1, unbiased=False) + EPS_LN).sqrt().sum(0)
    g_t, b_t = net.ln_params(net.blocks[tl].ln_1)
    d = x0s / nseq - b_t
    rho = g_t * d / (d @ d)
    rho = rho - rho.mean()
    sig_mean = {b: sig_sum[b] / nseq for b in range(tl)}       # [L] calibration mean LN-2 scale per position
    # block-b BOS write direction: sum_h B_OV,h x_0 at block b (x_0 fixed for BOS
    # inputs; its calibration mean otherwise)
    x0b_sum, nb = {b: 0 for b in range(tl)}, 0
    for idx, st in batches("calibration", 32):
        nb += idx.shape[0]
        for b in range(tl):
            x0b_sum[b] = x0b_sum[b] + st[("x_attn", b)][:, 0].double().sum(0)
    dirs = {}
    for b in range(tl):
        wb = torch.einsum("hdf,f->d", net.ov(b), x0b_sum[b] / nb)
        dirs[b] = {"rho": rho / rho.norm(), "wbos": wb / wb.norm()}

    # ---- term names ------------------------------------------------------------
    names = ["emb"]
    for b in range(tl):
        names += [f"b{b}h{h}_bos" for h in range(H)] + [f"b{b}h{h}_rest" for h in range(H)] + [f"b{b}_bias", f"b{b}_mlp"]
    nT = len(names)

    def terms(st):
        """[B,T,nT] exact numerator contributions, plus sigma [B,T] and beta [B,T]."""
        h0 = st[("resid_in", 0)].double()
        out = [h0 @ rho]
        for b in range(tl):
            x = st[("x_attn", b)].double(); A = st[("weights", b)].double()
            Y = torch.einsum("hdf,btf->bhtd", net.ov(b), x)
            proj = Y @ rho                                                   # [B,H,T]
            bos = A[..., 0] * proj[:, :, :1]                                  # [B,H,T]
            rest = torch.einsum("bhts,bhs->bht", A[..., 1:], proj[:, :, 1:])
            out += [bos[:, h] for h in range(H)] + [rest[:, h] for h in range(H)]
            out.append(torch.full_like(out[0], float(net.attn_bias(b) @ rho)))
            out.append(st[("mlp_out", b)].double() @ rho if ("mlp_out", b) in st else torch.zeros_like(out[0]))
        Tm = torch.stack(out, -1)
        hin = st[("resid_in", tl)].double()
        sigma = (hin.var(-1, unbiased=False) + EPS_LN).sqrt()
        beta = (st[("x_attn", tl)].double() - b_t) @ d / (d @ d)
        return Tm, sigma, beta

    # ---- validation: decomposition statistics ---------------------------------
    t0 = time.time()
    S1 = torch.zeros(L, nT, dtype=torch.float64, device=dev); S2 = torch.zeros(L, nT, nT, dtype=torch.float64, device=dev)
    sg1 = torch.zeros(L, dtype=torch.float64, device=dev); sg2 = torch.zeros_like(sg1)
    rec_err = 0.0; n = 0
    mlp_rows = {b: {k: [] for k in ("a_rho", "g_rho", "a_wbos", "g_wbos", "pos")} for b in range(tl)}
    mlp_cal = {b: {k: [] for k in ("a_rho", "g_rho", "a_wbos", "g_wbos", "pos")} for b in range(tl)}

    def mlp_collect(st, rows):
        for b in range(tl):
            if ("mlp_out", b) not in st:
                continue
            hm = st[("resid_mid", b)].double(); mo = st[("mlp_out", b)].double()
            for dn, w in dirs[b].items():
                rows[b][f"a_{dn}"].append((hm @ w).float().cpu()); rows[b][f"g_{dn}"].append((-(mo @ w)).float().cpu())
            rows[b]["pos"].append(torch.arange(L).expand(hm.shape[0], -1))

    for idx, st in batches("calibration", N_CAL):
        mlp_collect(st, mlp_cal)
    for idx, st in batches("validation", N_VAL):
        Tm, sigma, beta = terms(st)
        rec_err = max(rec_err, float((Tm.sum(-1) / sigma - beta).abs().max()))
        S1 += Tm.sum(0); S2 += torch.einsum("bti,btj->tij", Tm, Tm)
        sg1 += sigma.sum(0); sg2 += (sigma ** 2).sum(0); n += idx.shape[0]
        mlp_collect(st, mlp_rows)
    mean = S1 / n
    cov = S2 / n - torch.einsum("ti,tj->tij", mean, mean)                     # per position, across sequences
    print(f"decomposition {time.time() - t0:.0f}s, max |sum/sigma - beta| = {rec_err:.2e}", flush=True)

    bins = {k: v for k, v in GROUPS.items() if v[0] < L}
    pos = np.arange(L)
    summ = {}
    for g, (lo, hi) in bins.items():
        mk = torch.as_tensor((pos >= lo) & (pos <= min(hi, L - 1)), device=dev)
        C = cov[mk].mean(0)                                                    # mean within-position covariance
        mu = mean[mk].mean(0)
        tot_var = float(C.sum())
        summ[g] = {"numerator_mean": float(mu.sum()), "numerator_sd_seq": float(np.sqrt(max(tot_var, 0))),
                   "sigma_mean": float((sg1 / n)[mk].mean()),
                   "terms": {nm: {"mean": float(mu[i]), "sd_seq": float(C[i, i].clamp_min(0).sqrt()),
                                  "cov_with_total": float(C[i].sum())} for i, nm in enumerate(names)}}
    # top-level groups: emb / attention BOS / attention rest / bias / MLP, per block
    def grp(nm):
        if nm.endswith("_bos") or nm.endswith("_rest"):           # b<k>h<h>_bos -> b<k>_attn_bos
            return nm.split("h")[0] + "_attn_" + nm.rsplit("_", 1)[1]
        return nm
    gnames = sorted({grp(nm) for nm in names}, key=lambda s: names.index(next(x for x in names if grp(x) == s)))
    Gm = torch.zeros(nT, len(gnames), dtype=torch.float64, device=dev)
    for i, nm in enumerate(names):
        Gm[i, gnames.index(grp(nm))] = 1
    gsum = {}
    for g, (lo, hi) in bins.items():
        mk = torch.as_tensor((pos >= lo) & (pos <= min(hi, L - 1)), device=dev)
        C = Gm.T @ cov[mk].mean(0) @ Gm; mu = mean[mk].mean(0) @ Gm
        gsum[g] = {"cov": C.cpu().numpy().tolist(), "mean": mu.cpu().numpy().tolist()}

    # ---- B2: MLP read-and-subtract --------------------------------------------
    mlp_res = {}
    for b in range(tl):
        if not mlp_cal[b]["pos"]:
            mlp_res[f"b{b}"] = "no MLP"
            continue
        R = {}
        for dn in dirs[b]:
            ac = torch.cat(mlp_cal[b][f"a_{dn}"]).numpy().ravel(); gc = torch.cat(mlp_cal[b][f"g_{dn}"]).numpy().ravel()
            av = torch.cat(mlp_rows[b][f"a_{dn}"]).numpy().ravel(); gv = torch.cat(mlp_rows[b][f"g_{dn}"]).numpy().ravel()
            pv = torch.cat(mlp_rows[b]["pos"]).numpy().ravel()
            k, c0 = np.polyfit(ac, gc, 1)
            sub = np.random.default_rng(0).choice(len(ac), min(len(ac), 200000), replace=False)
            # several starting points; keep the best calibration fit
            pc, best = None, np.inf
            amed, asd = np.median(ac), np.std(ac) + 1e-6
            for a0 in (amed, np.quantile(ac, 0.1), np.quantile(ac, 0.9), 0.0):
                for r0 in (asd / 4, asd, 4 * asd, np.ptp(ac) / 2 + 1e-6):
                    p0 = [np.median(gc) - k * amed, max(abs(k), 0.1), r0, a0]
                    try:
                        p, _ = curve_fit(capped, ac[sub], gc[sub], p0=p0, maxfev=5000)
                    except Exception:  # noqa: BLE001
                        continue
                    sse = float(((gc[sub] - capped(ac[sub], *p)) ** 2).sum())
                    if sse < best:
                        pc, best = p, sse
            out = {"linear": {"k": float(k), "c0": float(c0)}, "capped": None if pc is None else dict(zip(["c0", "G", "r", "a0"], map(float, pc)))}
            for gname, (lo, hi) in bins.items():
                mk = (pv >= lo) & (pv <= hi)
                if not mk.any():
                    continue
                a, g = av[mk], gv[mk]
                tot = ((g - g.mean()) ** 2).sum()
                e_lin = g - (c0 + k * a)
                row = {"a_mean": float(a.mean()), "a_sd": float(a.std()), "g_mean": float(g.mean()), "g_sd": float(g.std()),
                       "post_mean": float((a - g).mean()), "post_sd": float((a - g).std()),
                       "r2_linear": float(1 - (e_lin ** 2).sum() / tot) if tot > 0 else None,
                       "corr_a_g": float(np.corrcoef(a, g)[0, 1]) if a.std() > 0 and g.std() > 0 else None}
                if pc is not None:
                    e_c = g - capped(a, *pc)
                    row["r2_capped"] = float(1 - (e_c ** 2).sum() / tot) if tot > 0 else None
                out[gname] = row
            R[dn] = out
        mlp_res[f"b{b}"] = R

    # ---- B2: local sensitivity of the block map h~ -> h~ + MLP(LN(h~)) -------
    sens = {}
    blk_ids = [b for b in range(tl) if not net.blocks[b].skip_mlp]
    idx = net.batch(sp["validation"][:8])
    cap = Capture(want={("resid_mid", b) for b in blk_ids})
    net.forward(idx, cap, stop_after=tl)
    probe_pos = [p for p in [0, 1, 2, 4, 8, 16, 64, 256, 1000] if p < L]
    for b in blk_ids:
        blk = net.blocks[b]
        g2 = blk.ln_2.weight.double(); b2 = blk.ln_2.bias.double() if blk.ln_2.bias is not None else None
        W1, c1 = blk.mlp.c_fc.weight.double(), blk.mlp.c_fc.bias
        W2, c2 = blk.mlp.c_proj.weight.double(), blk.mlp.c_proj.bias
        act = (lambda t: F.relu(t)) if isinstance(blk.mlp.gelu, torch.nn.ReLU) else (lambda t: F.gelu(t))

        def block_map(h, fixed_sigma=None):
            mu = h.mean(-1, keepdim=True)
            if fixed_sigma is None:
                xs = (h - mu) / ((h - mu).pow(2).mean(-1, keepdim=True) + EPS_LN).sqrt()
            else:
                xs = (h - mu) / fixed_sigma
            xm = xs * g2 + (b2 if b2 is not None else 0)
            u = xm @ W1.T + (c1.double() if c1 is not None else 0)
            return h + act(u) @ W2.T + (c2.double() if c2 is not None else 0)

        hm = cap.store[("resid_mid", b)].double()
        res = {}
        for dn, w in dirs[b].items():
            row = {}
            for p in probe_pos:
                hp = hm[:, p]
                _, jv = torch.func.jvp(block_map, (hp,), (w.expand_as(hp),))
                _, jvf = torch.func.jvp(lambda t: block_map(t, sig_mean[b][p]), (hp,), (w.expand_as(hp),))
                fd = {}
                for eps in (1e-1, 1e-2, 1e-3):
                    step = eps * hp.norm(dim=-1, keepdim=True)
                    fd[str(eps)] = float((((block_map(hp + step * w) - block_map(hp)) @ w) / step[:, 0]).mean())
                row[str(p)] = {"s_mean": float((jv @ w).mean()), "s_sd": float((jv @ w).std()),
                               "s_fixed_ln_scale_mean": float((jvf @ w).mean()), "finite_diff": fd}
            res[dn] = row
        sens[f"b{b}"] = res

    od = OUT_ROOT / "source" / args.model
    out = {"model": args.model, "target_layer": tl, "convention": "center = LN bias of the target attention input; "
           "global anchor from calibration", "names": names, "groups": gnames, "max_recon_err_beta": rec_err,
           "summary_by_position_group": summ, "group_cov": gsum, "mlp": mlp_res, "sensitivity": sens,
           "n_cal": N_CAL, "n_val": N_VAL, "splits": prov, "probe_positions": probe_pos}
    jdump(out, od / f"source_L{tl}.json")
    np.savez_compressed(od / f"source_L{tl}_curves.npz", mean=mean.cpu().numpy(),
                        sd=torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0).sqrt().cpu().numpy(),
                        sigma_mean=(sg1 / n).cpu().numpy(), names=np.array(names))
    print("wrote", od, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    run(ap.parse_args())
