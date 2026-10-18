"""Block-0 MLP denoising along the block-0 BOS write (two-layer position models).

Site and direction (as in cc_lncenter.py): h_i = block-0 residual after
attention; w = unit sum_h W_O,h W_V,h xbar_0.

Task 1, exact source split of the pre-MLP projection r_i = <h_i, w>:
  r_i = <emb_i, w> + sum_h a_hi0 <Y_h0, w> + sum_h a_hii <Y_hi, w>
        + sum_h sum_{1<=j<i} a_hij <Y_hj, w> + <b_attn, w>
  (emb, bos, self, prefix, bias; Y_hj = B_OV,h x_j). At i = 0 the BOS term
  is the token itself.

Task 2, exact variance identity at each position, with the MLP correction
m_i = <MLP_i, w> split by neuron group (T threshold, E\\T other effect
neurons, REST all other neurons, bias):
  Var(r + m) = Var r + sum_g Var m_g + 2 sum_g Cov(r, m_g) + 2 sum_{g<g'} Cov(m_g, m_g')
Causal: remove a group's output along w only (other output components kept),
at all positions or only at declared positions; controls remove the same
group's output along a random unit direction orthogonal to w, and the
output along w of 50 random neurons outside E. Downstream: block-1 feature,
BOS weights and head scalars of the block-1 BOS heads, frozen position head
(raw and with one calibration-estimated constant offset removed), refit
ridge on ln_f features, refit isotonic decoder on the summed head scalar.

Frozen quantities (w, neuron sets, block-1 anchor, head readouts, random
directions) are saved on the first run and loaded on later runs.

Usage: python cc_denoise.py --model 1h_s42 --split validation --device cuda:0
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

N_CAL = 160
N_T, N_E = 16, 50
TERMS = ["emb", "bos", "self", "prefix", "bias"]
GROUPS = ["T", "EnotT", "REST", "bias"]          # disjoint: T, E without T, neither, MLP output bias


def pos_groups(L):
    g = [(f"{j}", j, j) for j in range(8)] + [("8-15", 8, 15), ("16-127", 16, min(127, L - 1))]
    if L > 128:
        g.append((f"128-{L - 1}", 128, L - 1))
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
    assert spec.kind == "toy" and net.n_layer == 2, "two-layer position models only"
    L, d, H = spec.L, net.d, net.H
    blk = net.blocks[0]
    sp, prov = splits_for(args.model)
    Wout = blk.mlp.c_proj.weight.double()
    bout = blk.mlp.c_proj.bias.double() if blk.mlp.c_proj.bias is not None else torch.zeros(d, dtype=torch.float64, device=dev)
    act = (lambda t: F.relu(t)) if isinstance(blk.mlp.gelu, torch.nn.ReLU) else (lambda t: F.gelu(t))
    cal = sp["calibration"][:N_CAL]
    G = pos_groups(L)
    bins = bins_for(L)
    ii = np.arange(L)

    def fwd(offs, edits=None, want=()):
        idx = net.batch(offs)
        cap = Capture(want=set(want), edits=edits)
        o = net.forward(idx, cap)
        return idx, cap.store, o

    # ---- frozen quantities ------------------------------------------------------
    fz_path = OUT_ROOT / "configs" / f"frozen_denoise_{args.model}.pt"
    if fz_path.exists():
        FZ = torch.load(fz_path, map_location=dev)
        print("loaded frozen", fz_path, flush=True)
    else:
        x0s = 0; x0s1 = 0; xnb1 = 0; n = 0; hid_sum = 0; con_sum = 0
        for s in range(0, len(cal), bs):
            idx, st, _ = fwd(cal[s:s + bs], want=[("x_attn", 0), ("x_attn", 1)])
            x0s = x0s + st[("x_attn", 0)][:, 0].double().sum(0)
            x1 = st[("x_attn", 1)].double(); x0s1 = x0s1 + x1[:, 0].sum(0); xnb1 = xnb1 + x1[:, 1:].sum((0, 1))
            n += idx.shape[0]
        wv = torch.einsum("hdf,f->d", net.ov(0), x0s / n); wh = wv / wv.norm()
        wout_w = Wout.T @ wh
        for s in range(0, len(cal), bs):
            idx, st, _ = fwd(cal[s:s + bs], want=[("mlp_pre", 0)])
            hid = act(st[("mlp_pre", 0)].double()); hid_sum = hid_sum + hid.sum(0); con_sum = con_sum + (hid * wout_w).sum(0)
        A = hid_sum / n; Cm = con_sum / n
        score = torch.minimum(A[0], A[1]) - A[2:].abs().max(0).values
        setT = torch.argsort(score, descending=True)[:N_T]
        setE = torch.argsort(Cm.abs().max(0).values, descending=True)[:N_E]
        gen = torch.Generator(device="cpu").manual_seed(0)
        outside = torch.tensor([k for k in range(Wout.shape[1]) if k not in set(setE.tolist())])
        setR = outside[torch.randperm(len(outside), generator=gen)[:N_E]].to(dev)
        rd = torch.randn(d, generator=gen, dtype=torch.float64).to(dev); rd = rd - (rd @ wh) * wh; rd = rd / rd.norm()
        x0b1 = x0s1 / n; xnb1 = xnb1 / (n * (L - 1))
        b_ln1 = net.ln_params(net.blocks[1].ln_1)[1]
        M1 = net.ov(1)
        meta = OUT_ROOT / "screen" / args.model / "meta.json"
        rh = np.array(json.loads(meta.read_text())["r_h"]["L1"])
        watch = [int(h) for h in np.nonzero(rh > 10)[0]] or [int(np.argmax(rh))]
        wb2, wn2 = torch.einsum("hdf,f->hd", M1, x0b1), torch.einsum("hdf,f->hd", M1, xnb1)
        U2 = (wn2 - wb2) / (wn2 - wb2).norm(dim=-1, keepdim=True)
        FZ = {"w": wh, "T": setT, "E": setE, "RAND": setR, "rand_dir": rd, "d1": x0b1 - b_ln1, "b_ln1": b_ln1,
              "watch": torch.tensor(watch), "U2": U2[watch]}
        fz_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in FZ.items()}, fz_path)
        FZ = {k: v.to(dev) for k, v in FZ.items()}
        print("froze", fz_path, flush=True)
    wh, setT, setE, setR = FZ["w"], FZ["T"], FZ["E"], FZ["RAND"]
    wout_w = Wout.T @ wh
    nT = setT.tolist(); EnotT = [k for k in setE.tolist() if k not in set(nT)]
    gidx = {"T": torch.tensor(nT, device=dev), "EnotT": torch.tensor(EnotT, device=dev),
            "E": setE, "RAND": setR, "ALL": torch.arange(Wout.shape[1], device=dev)}
    rest_mask = torch.ones(Wout.shape[1], dtype=torch.bool, device=dev); rest_mask[setE] = False; rest_mask[setT] = False   # REST = not in T or E
    gidx["REST"] = torch.nonzero(rest_mask).flatten()
    watch = FZ["watch"].tolist()
    evals = sp[args.split][:args.n_eval]

    # ---- tasks 1 and 2: decomposition and variance identity ---------------------
    t0 = time.time()
    want = [("resid_in", 0), ("x_attn", 0), ("weights", 0), ("resid_mid", 0), ("mlp_pre", 0), ("mlp_out", 0)]
    terms = {k: [] for k in TERMS}; rr, mm = [], []
    mg = {g: [] for g in GROUPS}
    perhead = {k: [] for k in ("bos", "self", "prefix")}
    err_r = err_m = 0.0
    tri_strict = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev), -1)
    for s in range(0, len(evals), bs):
        idx, st, _ = fwd(evals[s:s + bs], want=want)
        x = st[("x_attn", 0)].double(); A = st[("weights", 0)].double()
        proj = torch.einsum("hf,btf->bht", torch.einsum("d,hdf->hf", wh, net.ov(0)), x)              # <Y_hj, w>  [B,H,T]
        bos = A[..., 0] * proj[:, :, :1]                                                         # [B,H,T]
        diag = torch.diagonal(A, dim1=-2, dim2=-1)                                               # a_hii
        self_ = diag * proj; self_[..., 0] = 0                                                   # at i = 0 the BOS term holds it
        prefix = torch.einsum("bhij,bhj->bhi", A * tri_strict, proj) - A[..., 0] * proj[:, :, :1]
        prefix[..., 0] = 0
        emb = st[("resid_in", 0)].double() @ wh
        bias = torch.full_like(emb, float(net.attn_bias(0) @ wh))
        r = st[("resid_mid", 0)].double() @ wh
        err_r = max(err_r, float((emb + bos.sum(1) + self_.sum(1) + prefix.sum(1) + bias - r).abs().max()))
        for k, v in zip(TERMS, (emb, bos.sum(1), self_.sum(1), prefix.sum(1), bias)):
            terms[k].append(v.cpu())
        for k, v in zip(("bos", "self", "prefix"), (bos, self_, prefix)):
            perhead[k].append(v.float().cpu())
        hid = act(st[("mlp_pre", 0)].double())
        con = hid * wout_w
        m = st[("mlp_out", 0)].double() @ wh
        parts = {"T": con[..., gidx["T"]].sum(-1), "EnotT": con[..., gidx["EnotT"]].sum(-1) if len(EnotT) else torch.zeros_like(m),
                 "REST": con[..., gidx["REST"]].sum(-1), "bias": torch.full_like(m, float(bout @ wh))}
        err_m = max(err_m, float((sum(parts.values()) - m).abs().max()))
        for g in GROUPS:
            mg[g].append(parts[g].cpu())
        rr.append(r.cpu()); mm.append(m.cpu())
    Tm = {k: torch.cat(v).numpy() for k, v in terms.items()}
    PH = {k: torch.cat(v).numpy() for k, v in perhead.items()}
    R_, M_ = torch.cat(rr).numpy(), torch.cat(mm).numpy()
    MG = {g: torch.cat(v).numpy() for g, v in mg.items()}
    print(f"decomposition {time.time() - t0:.0f}s; recon err r {err_r:.2e}, m {err_m:.2e}", flush=True)

    def cov(a, b):
        return ((a - a.mean(0)) * (b - b.mean(0))).mean(0)                   # per position, across sequences

    dec, var_id = {}, {}
    for name, lo, hi in G:
        sl = slice(lo, hi + 1)
        row = {k: {"mean": float(Tm[k][:, sl].mean()), "var": float(Tm[k][:, sl].var(0).mean())} for k in TERMS}
        C = np.array([[float(cov(Tm[a][:, sl], Tm[b][:, sl]).mean()) for b in TERMS] for a in TERMS])
        row["cov_matrix"] = C.tolist(); row["var_total"] = float(R_[:, sl].var(0).mean()); row["mean_total"] = float(R_[:, sl].mean())
        row["per_head_mean"] = {k: PH[k][:, :, sl].mean((0, 2)).tolist() for k in PH}
        dec[name] = row
        r, m = R_[:, sl], M_[:, sl]
        vi = {"var_r": float(r.var(0).mean()), "var_m": float(m.var(0).mean()), "two_cov_r_m": float(2 * cov(r, m).mean()),
              "var_r_plus_m": float((r + m).var(0).mean())}
        vi["identity_residual"] = vi["var_r_plus_m"] - (vi["var_r"] + vi["var_m"] + vi["two_cov_r_m"])
        vi["variance_reduction"] = 1 - vi["var_r_plus_m"] / max(vi["var_r"], 1e-300)
        per = {}
        for g in GROUPS:
            x_ = MG[g][:, sl]
            per[g] = {"mean": float(x_.mean()), "var": float(x_.var(0).mean()), "two_cov_with_r": float(2 * cov(r, x_).mean()),
                      "var_r_plus_mg_alone": float((r + x_).var(0).mean()),
                      "share_of_total_two_cov": float(cov(r, x_).mean() / cov(r, m).mean()) if abs(cov(r, m).mean()) > 0 else None}
        per["cross_two_cov"] = {f"{a}|{b}": float(2 * cov(MG[a][:, sl], MG[b][:, sl]).mean())
                                for i_, a in enumerate(GROUPS) for b in GROUPS[i_ + 1:]}
        vi["groups"] = per
        var_id[name] = vi

    # ---- causal: remove group outputs along w ------------------------------------
    early = torch.arange(L, device=dev) < 8
    pos01 = torch.arange(L, device=dev) < 2
    conds = {"orig": None}
    for g in ("T", "EnotT", "E", "REST", "ALL", "RAND"):
        conds[f"rm_{g}"] = (g, None, "w")
    conds.update({"rm_T_pos0-1": ("T", pos01, "w"), "rm_T_pos2+": ("T", ~pos01, "w"),
                  "rm_E_pos0-7": ("E", early, "w"), "rm_E_pos8+": ("E", ~early, "w"),
                  "rm_ALL_pos0-7": ("ALL", early, "w"), "rm_ALL_pos8+": ("ALL", ~early, "w"),
                  "ctrl_E_randdir": ("E", None, "rand"), "ctrl_ALL_randdir": ("ALL", None, "rand")})

    def edits(c):
        if c is None:
            return None
        g, pm, dn = c
        direc = wh if dn == "w" else FZ["rand_dir"]
        st_ = {}

        def keep(t):
            st_["pre"] = t.double(); return t

        def mlp(t):
            hid = act(st_["pre"][..., gidx[g]])
            comp = hid @ (Wout[:, gidx[g]].T @ direc)                     # group output along direc  [B,T]
            if pm is not None:
                comp = comp * pm.double()
            return (t.double() - comp[..., None] * direc).float()
        return {("mlp_pre", 0): keep, ("mlp_out", 0): mlp}

    d1, b_ln1, U2 = FZ["d1"], FZ["b_ln1"], FZ["U2"]
    M1 = net.ov(1)[watch]
    tgt = torch.arange(L, device=dev, dtype=torch.float64)

    def measure(offs, ed):
        Rr = {k: [] for k in ("post", "beta1", "abos", "z", "pred")}; feats = []
        for s in range(0, len(offs), bs):
            idx, st, o = fwd(offs[s:s + bs], ed, want=[("resid_out", 0), ("x_attn", 1), ("weights", 1), ("final_ln", 2)])
            Rr["post"].append((st[("resid_out", 0)].double() @ wh).cpu())
            x1 = st[("x_attn", 1)].double()
            Rr["beta1"].append(((x1 - b_ln1) @ d1 / (d1 @ d1)).cpu())
            Aw = st[("weights", 1)].double()[:, watch]
            Rr["abos"].append(Aw[..., 0].cpu())
            v = torch.einsum("hd,bhtd->bht", U2, torch.einsum("hdf,btf->bhtd", M1, x1))
            Rr["z"].append(torch.einsum("bhts,bhs->bht", Aw, v).cpu())
            Rr["pred"].append(o.double().cpu()); feats.append(st[("final_ln", 2)].float())
        return {k: torch.cat(v).numpy() for k, v in Rr.items()}, torch.cat(feats)

    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(evals), len(evals)) for _ in range(args.n_boot)]

    def relmae_arr(pred):
        return {bn: np.abs(pred[:, lo:hi + 1] - ii[lo:hi + 1]) / ii[lo:hi + 1] for bn, (lo, hi) in bins.items()}

    def summary(pred, ref=None):
        E = relmae_arr(pred); o = {}
        Er = relmae_arr(ref) if ref is not None else None
        for bn, e in E.items():
            est = float(np.median(e)); bs_ = [float(np.median(e[b])) for b in boot]
            o[bn] = {"relmae": est, "ci": [float(np.quantile(bs_, .025)), float(np.quantile(bs_, .975))]}
            if Er is not None:
                dd = [float(np.median(e[b]) - np.median(Er[bn][b])) for b in boot]
                o[bn]["diff_vs_orig"] = est - float(np.median(Er[bn]))
                o[bn]["diff_ci"] = [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))]
        return o

    def iso(zc, ze):
        m_ = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(zc[:, 1:].ravel(), np.tile(ii[1:], len(zc)).astype(float))
        p = np.zeros_like(ze); p[:, 1:] = m_.predict(ze[:, 1:].ravel()).reshape(len(ze), -1); return p

    res = {}
    base = basec = None
    for cname, c in conds.items():
        t0 = time.time()
        ed = edits(c)
        Mc, Fc = measure(cal, ed); Mv, Fv = measure(evals, ed)
        if base is None:
            base, basec = Mv, Mc
        ridge = Ridge(d, dev)
        for s in range(0, len(Fc), 32):
            ridge.add(Fc[s:s + 32], tgt.expand(len(Fc[s:s + 32]), L))
        ridge.solve()
        probe = torch.cat([ridge.predict(Fv[s:s + 32]) for s in range(0, len(Fv), 32)]).cpu().numpy()
        off = float((Mc["pred"][:, 1:] - basec["pred"][:, 1:]).mean())               # one constant, calibration
        zc, zv = Mc["z"].sum(1), Mv["z"].sum(1)
        r = {"frozen_head": summary(Mv["pred"], base["pred"] if cname != "orig" else None),
             "frozen_head_offset_removed": summary(Mv["pred"] - off, base["pred"] if cname != "orig" else None),
             "offset": off, "ridge_refit": summary(probe), "head_scalar_iso_refit": summary(iso(zc, zv)),
             "head_scalar_iso_frozen": summary(iso(basec["z"].sum(1), zv)),
             "head_scalar_iso_frozen_offset_removed": summary(iso(basec["z"].sum(1), zv - float((zc[:, 1:] - basec["z"].sum(1)[:, 1:]).mean())))}
        by = {}
        for name, lo, hi in G:
            sl = slice(lo, hi + 1)
            by[name] = {"post_var": float(Mv["post"][:, sl].var(0).mean()), "post_mean_shift": float((Mv["post"][:, sl].mean(0) - base["post"][:, sl].mean(0)).mean()),
                        "beta1_sd": float(Mv["beta1"][:, sl].std(0).mean()), "beta1_mean_shift": float((Mv["beta1"][:, sl].mean(0) - base["beta1"][:, sl].mean(0)).mean()),
                        "abos_sd": Mv["abos"][:, :, sl].std(0).mean(-1).tolist(),
                        "abos_mean_relshift": (np.abs(Mv["abos"][:, :, sl].mean(0) - base["abos"][:, :, sl].mean(0)) / np.maximum(base["abos"][:, :, sl].mean(0), 1e-12)).mean(-1).tolist(),
                        "z_sd": Mv["z"][:, :, sl].std(0).mean(-1).tolist(),
                        "z_mean_shift": (Mv["z"][:, :, sl].mean(0) - base["z"][:, :, sl].mean(0)).mean(-1).tolist()}
        r["by_group"] = by
        res[cname] = r
        lb = list(bins)[-1]
        print(f"{cname:16s} {time.time() - t0:.0f}s frozen {r['frozen_head'][lb]['relmae']:.4f} off-removed {r['frozen_head_offset_removed'][lb]['relmae']:.4f} "
              f"ridge {r['ridge_refit'][lb]['relmae']:.4f} iso {r['head_scalar_iso_refit'][lb]['relmae']:.4f} "
              f"| post var {by[G[-1][0]]['post_var']:.3g} beta1 sd {by[G[-1][0]]['beta1_sd']:.2e}", flush=True)

    out = {"model": args.model, "split": args.split, "n_eval": len(evals), "n_cal": N_CAL, "splits": prov,
           "site": "block 0, residual after attention (pre-MLP) and MLP output, along unit sum_h W_O W_V xbar_0",
           "sets": {"T": nT, "EnotT": EnotT, "RAND": setR.tolist()}, "watched_block1_heads": watch,
           "recon_err": {"r": err_r, "m": err_m}, "decomposition": dec, "variance_identity": var_id, "interventions": res}
    od = OUT_ROOT / "denoise"
    jdump(out, od / f"{args.model}_{args.split}.json")
    np.savez_compressed(od / f"{args.model}_{args.split}_scalars.npz", offsets=evals, r=R_, m=M_,
                        **{f"term_{k}": v for k, v in Tm.items()}, **{f"mg_{g}": v for g, v in MG.items()})

    # ---- figures -------------------------------------------------------------------
    names = [g[0] for g in G]
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    xs = np.arange(len(names)); wdt = 0.16
    for k, t in enumerate(TERMS):
        ax[0].bar(xs + (k - 2) * wdt, [dec[n_][t]["mean"] for n_ in names], wdt, label=t)
        ax[1].bar(xs + (k - 2) * wdt, [dec[n_][t]["var"] for n_ in names], wdt, label=t)
    ax[0].plot(xs, [dec[n_]["mean_total"] for n_ in names], "k.-", label="total r")
    ax[1].plot(xs, [dec[n_]["var_total"] for n_ in names], "k.-", label="Var r")
    ax[0].set_title("pre-MLP projection on w: term means"); ax[1].set_title("term variance across sequences"); ax[1].set_yscale("symlog", linthresh=1e-2)
    for g in GROUPS:
        ax[2].plot(xs, [var_id[n_]["groups"][g]["two_cov_with_r"] for n_ in names], "o-", label=f"2 Cov(r, m_{g})")
    ax[2].plot(xs, [var_id[n_]["var_r"] for n_ in names], "k--", label="Var r")
    ax[2].plot(xs, [var_id[n_]["var_r_plus_m"] for n_ in names], "k-", label="Var(r + m)")
    ax[2].set_yscale("symlog", linthresh=1e-2); ax[2].set_title("variance identity by neuron group")
    for a_ in ax:
        a_.set_xticks(xs); a_.set_xticklabels(names, rotation=45, fontsize=7); a_.legend(fontsize=7)
    fig.suptitle(f"{args.model} ({args.split}): source of the pre-MLP BOS projection and MLP denoising"); fig.tight_layout()
    fig.savefig(od / f"{args.model}_{args.split}_decomposition.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    cn = list(res); lb = list(bins)[-1]
    for k, key in enumerate(("frozen_head", "frozen_head_offset_removed", "ridge_refit", "head_scalar_iso_refit")):
        v = [res[c][key][lb]["relmae"] for c in cn]
        ax[0].plot(range(len(cn)), v, "o-", label=key)
    ax[0].set_yscale("log"); ax[0].set_xticks(range(len(cn))); ax[0].set_xticklabels(cn, rotation=60, fontsize=7)
    ax[0].set_title(f"position relMAE, {lb} bin"); ax[0].legend(fontsize=7)
    ax[1].plot(range(len(cn)), [res[c]["by_group"][G[-1][0]]["beta1_sd"] for c in cn], "o-", label=f"block-1 beta SD ({G[-1][0]})")
    ax[1].plot(range(len(cn)), [res[c]["by_group"]["16-127"]["beta1_sd"] for c in cn], "o-", label="block-1 beta SD (16-127)")
    ax[1].set_yscale("log"); ax[1].set_xticks(range(len(cn))); ax[1].set_xticklabels(cn, rotation=60, fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(od / f"{args.model}_{args.split}_interventions.png", dpi=120); plt.close(fig)
    print("wrote", od, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--split", default="validation", choices=["validation", "test", "test_strict"])
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    run(ap.parse_args())
