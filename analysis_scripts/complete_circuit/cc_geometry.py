"""Is the block-0 MLP's BOS-direction denoising a local property of a tight
ordinary-token cluster that sits opposite BOS along the BOS write?

Site: block 0, MLP input. h = residual after block-0 attention, n = (h - mu)/sigma
(the LN-normalized input without gamma, b), w = unit block-0 BOS write
(frozen in configs/frozen_denoise_<model>.pt), u_ord = unit calibration mean of
n over positions >= 16, n_BOS = n at position 0.

1. Geometry by position group: cos(n_i, w), cos(n_i, n_BOS), cos(n_i, u_ord);
   cos(u_ord, w) and cos(u_ord, n_BOS) (antipodal would be -1).
2. Concentration of late tokens: mean cos(n_i, u_ord); fraction of the across-
   token variance of n (late bin) that lies along w and along u_ord.
3. Local sensitivity s_v = <J v, w> of the block map F(h) = h + MLP(LN(h)),
   fp64 forward-mode AD, for v = w, u_ord, a random unit vector, at
   (a) actual tokens by position and (b) points on the straight path from a late
   ordinary token to BOS in h space, h(t) = (1 - t) h_i + t h_BOS.
   s_w = 1 means a push along w survives; s_w = 0 means the MLP cancels it.

Usage: python cc_geometry.py --model 1h_s42 --device cuda:0
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from cc_common import EPS_LN, OUT_ROOT, SPECS, Capture, jdump, load_net, splits_for


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    net, _ = load_net(a.model, dev)
    L, d = net.spec.L, net.d
    sp, _ = splits_for(a.model)
    FZ = torch.load(OUT_ROOT / "configs" / f"frozen_denoise_{a.model}.pt", map_location=dev)
    w = FZ["w"].double()
    blk = net.blocks[0]
    g2, b2 = blk.ln_2.weight.double(), blk.ln_2.bias.double()
    W1, c1 = blk.mlp.c_fc.weight.double(), blk.mlp.c_fc.bias.double()
    W2, c2 = blk.mlp.c_proj.weight.double(), blk.mlp.c_proj.bias.double()
    act = (lambda t: F.relu(t)) if isinstance(blk.mlp.gelu, torch.nn.ReLU) else (lambda t: F.gelu(t))

    def norm(h):
        c = h - h.mean(-1, keepdim=True)
        return c / (c.pow(2).mean(-1, keepdim=True) + EPS_LN).sqrt()

    def block_map(h):
        return h + act((norm(h) * g2 + b2) @ W1.T + c1) @ W2.T + c2

    def hs(name, n_):
        out = []
        offs = sp[name][:n_]
        for s in range(0, len(offs), 4):
            idx = net.batch(offs[s:s + 4]); cap = Capture(want={("resid_mid", 0)}); net.forward(idx, cap, stop_after=0)
            out.append(cap.store[("resid_mid", 0)].double())
        return torch.cat(out)

    Hc = hs("calibration", 64)
    u_ord = norm(Hc[:, 16:]).mean((0, 1)); u_ord = u_ord / u_ord.norm()
    Hv = hs("validation", 64)
    N = norm(Hv)
    nb = N[0, 0]
    cosf = lambda X, v: (X @ v) / (X.norm(dim=-1) * v.norm())
    groups = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16-127", 16, min(127, L - 1))]
    if L > 128:
        groups.append(("128+", 128, L - 1))
    geo = {}
    for gname, lo, hi in groups:
        X = N[:, lo:hi + 1].reshape(-1, d)
        geo[gname] = {k: {"mean": float(v.mean()), "sd": float(v.std()) if v.numel() > 1 else 0.0}
                      for k, v in (("cos_w", cosf(X, w)), ("cos_bos", cosf(X, nb)), ("cos_ord_mean", cosf(X, u_ord)))}
    late = N[:, 128:] if L > 128 else N[:, 16:]
    Xl = late.reshape(-1, d); Xc = Xl - Xl.mean(0)
    tot = float((Xc ** 2).sum(-1).mean())
    conc = {"mean_cos_to_ord_mean": float(cosf(Xl, u_ord).mean()),
            "frac_var_along_w": float(((Xc @ w) ** 2).mean() / tot),
            "frac_var_along_u_ord": float(((Xc @ u_ord) ** 2).mean() / tot),
            "frac_var_along_random_dir_expected": 1.0 / d,
            "cos_u_ord_w": float(u_ord @ w), "cos_u_ord_bos": float(cosf(u_ord[None], nb)[0])}
    gen = torch.Generator(device="cpu").manual_seed(1)
    rnd = torch.randn(d, generator=gen, dtype=torch.float64).to(dev); rnd = rnd / rnd.norm()
    dirs = {"w": w, "u_ord": u_ord, "random": rnd}

    def sens(Hp):
        out = {}
        for dn, v in dirs.items():
            _, jv = torch.func.jvp(block_map, (Hp,), (v.expand_as(Hp),))
            s = jv @ w
            out[dn] = {"mean": float(s.mean()), "sd": float(s.std())}
        return out

    probe = [p for p in (0, 1, 2, 3, 4, 8, 16, 64, 256, 1000) if p < L]
    sens_tok = {str(p): sens(Hv[:16, p]) for p in probe}
    li = min(500, L - 1)
    hb = Hv[:16, 0]
    path = {}
    for t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        Hp = (1 - t) * Hv[:16, li] + t * hb
        r = sens(Hp)
        r["cos_w_of_point"] = float(cosf(norm(Hp), w).mean())
        r["mlp_write_along_w"] = float(((block_map(Hp) - Hp) @ w).mean())
        path[str(t)] = r
    out = {"model": a.model, "geometry": geo, "late_concentration": conc, "sensitivity_by_position": sens_tok,
           "path_late_token_to_bos": {"token_position": li, "points": path}}
    jdump(out, OUT_ROOT / "geometry" / f"{a.model}.json")
    print(a.model)
    for g_, x in geo.items():
        print(f"  {g_:7s} cos w {x['cos_w']['mean']:+.3f}±{x['cos_w']['sd']:.3f}  cos BOS {x['cos_bos']['mean']:+.3f}  cos ord-mean {x['cos_ord_mean']['mean']:+.3f}±{x['cos_ord_mean']['sd']:.3f}")
    print("  late:", {k: round(v, 4) for k, v in conc.items()})
    print("  s_w at tokens:", {p: (round(v["w"]["mean"], 3), round(v["u_ord"]["mean"], 3), round(v["random"]["mean"], 3)) for p, v in sens_tok.items()})
    print("  path ord->BOS (t: cos_w, s_w, MLP write along w):", {t: (round(v["cos_w_of_point"], 2), round(v["w"]["mean"], 3), round(v["mlp_write_along_w"], 1)) for t, v in path.items()})


if __name__ == "__main__":
    main()
