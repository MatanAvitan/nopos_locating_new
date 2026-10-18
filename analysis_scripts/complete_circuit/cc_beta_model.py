"""Test the beta-centered account of a layer-l head, term by term.

Feature (center m = LN bias of the head's attention input, anchor d = xbar_0 - m):
  beta_j = <x_j - m, d>/|d|^2;  beta_ord = calibration mean of beta over ordinary
  keys j >= 16; beta'_j = (beta_j - beta_ord)/(1 - beta_ord), so BOS = 1 and the
  ordinary baseline = 0.
Score scale in beta' units, d' = (1 - beta_ord) d:
  B'_i = b_Q.W_K d'/sqrt(dh)  (query-bias part, constant)  +  (W_Q x_i).W_K d'/sqrt(dh)  (input part)
  s_ij = row constant + B'_i beta'_j + E_ij (exact).
Value: v_j = <B_OV x_j, u>, u = unit(w_nonBOS - w_BOS); OLS v = c + lambda beta' on calibration.
Predictions of z_i = sum_j a_ij v_j:
  zA  c + lambda sum_j a_ij beta'_j            observed attention, value model
  zB  c + lambda sum_j ahat_ij beta'_j         ahat = softmax_j(Bbar beta'_j), Bbar = calibration mean B'_i (i >= 16)
  zC  c + lambda 2 e^Bbar / (2 e^Bbar + i - 1) two identical sinks, no fit
  zC* c + lambda a e^B / (a e^B + i + 1 - a)   (a, B) fitted to the calibration mean sink mass
Sink mass p_i = a_i0 + a_i1 (observed) against the same models.

Usage: python cc_beta_model.py --model 1h_s42 --heads 1:0 --split test --device cuda:0
"""

import argparse
import time

import numpy as np
import torch
from scipy.optimize import curve_fit

from cc_common import OUT_ROOT, SPECS, Capture, bins_for, jdump, load_net, splits_for


def masked_softmax(sc, T, dev):
    mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
    return torch.softmax(sc.masked_fill(mask, float("-inf")), -1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--heads", nargs="+", required=True, help="layer:head, one layer per run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-cal", type=int, default=320)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    dev, bs = a.device, a.batch_size
    net, _ = load_net(a.model, dev)
    L = net.spec.L
    sp, prov = splits_for(a.model)
    lay = sorted({int(s.split(":")[0]) for s in a.heads}); assert len(lay) == 1
    l = lay[0]; heads = [int(s.split(":")[1]) for s in a.heads]
    hw = net.hw[l]; M = net.ov(l)
    m = net.ln_params(net.blocks[l].ln_1)[1]
    bins = bins_for(L)
    ii = np.arange(L)

    def batches(name, n):
        offs = sp[name][:n]
        for s in range(0, len(offs), bs):
            idx = net.batch(offs[s:s + bs]); cap = Capture(want={("x_attn", l), ("weights", l), ("scores", l)})
            net.forward(idx, cap, stop_after=l)
            yield cap.store

    # ---- calibration -------------------------------------------------------------
    t0 = time.time()
    x0s = xnb = 0; n = 0
    for st in batches("calibration", a.n_cal):
        x = st[("x_attn", l)].double(); x0s = x0s + x[:, 0].sum(0); xnb = xnb + x[:, 1:].sum((0, 1)); n += x.shape[0]
    x0 = x0s / n; xnb = xnb / (n * (L - 1)); d = x0 - m
    bsum = 0; P = {}
    for h in heads:
        wb, wn = M[h] @ x0, M[h] @ xnb
        P[h] = {"u": (wn - wb) / (wn - wb).norm(), "Bsum": 0.0, "Bn": 0, "ols": np.zeros(5), "p_cal": 0}
    for st in batches("calibration", a.n_cal):
        x = st[("x_attn", l)].double(); beta = ((x - m) @ d) / (d @ d); bsum = bsum + beta.sum(0)
    bmean = bsum / n
    beta_ord = float(bmean[16:].mean())
    dprime = (1 - beta_ord) * d
    for st in batches("calibration", a.n_cal):
        x = st[("x_attn", l)].double(); A = st[("weights", l)].double()
        bp = (((x - m) @ d) / (d @ d) - beta_ord) / (1 - beta_ord)
        for h in heads:
            q = torch.einsum("btd,ed->bte", x, hw.W_q[h].double()) + hw.b_q[h].double()
            Bp = (q @ (hw.W_k[h].double() @ dprime)) * hw.scale
            P[h]["Bsum"] += float(Bp[:, 16:].sum()); P[h]["Bn"] += Bp[:, 16:].numel()
            v = (torch.einsum("de,bte->btd", M[h], x) @ P[h]["u"]).reshape(-1); bb = bp.reshape(-1)
            P[h]["ols"] += np.array([len(v), float(bb.sum()), float((bb * bb).sum()), float(v.sum()), float((bb * v).sum())])
            P[h]["p_cal"] = P[h]["p_cal"] + (A[:, h, :, 0] + A[:, h, :, 1]).sum(0)
    for h in heads:
        nn_, sb, sbb, sv, sbv = P[h]["ols"]
        lam = (sbv / nn_ - sb / nn_ * sv / nn_) / (sbb / nn_ - (sb / nn_) ** 2)
        P[h].update({"Bbar": P[h]["Bsum"] / P[h]["Bn"], "lam": lam, "c": sv / nn_ - lam * sb / nn_})
        pc = (P[h]["p_cal"] / n).cpu().numpy()
        f = lambda i_, la, B: np.exp(la + B) / (np.exp(la + B) + i_ + 1 - np.exp(la))      # a = e^la
        try:
            (la, Bf), _ = curve_fit(f, ii[16:], pc[16:], p0=[np.log(2), P[h]["Bbar"]], maxfev=20000)
            P[h]["fit_a"], P[h]["fit_B"] = float(np.exp(la)), float(Bf)
        except Exception:  # noqa: BLE001
            P[h]["fit_a"], P[h]["fit_B"] = None, None
    print(f"calibration {time.time() - t0:.0f}s beta_ord {beta_ord:.4f}", flush=True)

    # ---- evaluation --------------------------------------------------------------
    t0 = time.time()
    keep = {h: {k: [] for k in ("bp", "Bp", "Bbias", "Bin", "z", "zA", "zB", "p", "pB", "a0", "a0B", "r2n", "r2d")} for h in heads}
    for st in batches(a.split, a.n_eval):
        x = st[("x_attn", l)].double(); A = st[("weights", l)].double(); S = st[("scores", l)].double()
        bp = (((x - m) @ d) / (d @ d) - beta_ord) / (1 - beta_ord)
        xi = x - m - (((x - m) @ d) / (d @ d))[..., None] * d
        for h in heads:
            Ph = P[h]; K = hw.W_k[h].double() @ dprime
            Bbias = float(hw.b_q[h].double() @ K) * hw.scale
            Bin = (torch.einsum("btd,ed->bte", x, hw.W_q[h].double()) @ K) * hw.scale
            Bp = Bbias + Bin
            a_ = A[:, h]
            v = torch.einsum("de,bte->btd", M[h], x) @ Ph["u"]
            z = torch.einsum("bij,bj->bi", a_, v)
            ahat = masked_softmax(Ph["Bbar"] * bp[:, None, :].expand(-1, L, -1), L, dev)
            k = keep[h]
            k["bp"].append(bp.cpu()); k["Bp"].append(Bp.cpu()); k["Bin"].append(Bin.cpu())
            k["Bbias"].append(Bbias)
            k["z"].append(z.cpu()); k["zA"].append((Ph["c"] + Ph["lam"] * torch.einsum("bij,bj->bi", a_, bp)).cpu())
            k["zB"].append((Ph["c"] + Ph["lam"] * torch.einsum("bij,bj->bi", ahat, bp)).cpu())
            k["p"].append((a_[..., 0] + a_[..., 1]).cpu()); k["pB"].append((ahat[..., 0] + ahat[..., 1]).cpu())
            k["a0"].append(a_[..., 0].cpu()); k["a0B"].append(ahat[..., 0].cpu())
            # within-query score R^2 over ordinary keys j >= 16 (E omitted), queries i >= 16
            ds = S[:, h, :, :1] - S[:, h]; pred = (1 - bp)[:, None, :] * Bp[..., None]
            tri = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev)); tri[:, :16] = False; tri[:16] = False
            cnt = tri.sum(-1, keepdim=True).clamp_min(1)
            def rc(t):
                t = torch.where(tri, t, torch.zeros((), dtype=t.dtype, device=dev)); return torch.where(tri, t - t.sum(-1, keepdim=True) / cnt, torch.zeros((), dtype=t.dtype, device=dev))
            k["r2n"].append(float((rc(ds - pred) ** 2).sum())); k["r2d"].append(float((rc(ds) ** 2).sum()))
    print(f"evaluation {time.time() - t0:.0f}s", flush=True)

    def by_bin(fn):
        return {bn: fn(lo, hi) for bn, (lo, hi) in bins.items()}

    res = {"model": a.model, "layer": l, "split": a.split, "n_eval": int(len(sp[a.split][:a.n_eval])), "beta_ord": beta_ord,
           "beta_center": "LN bias of the attention input; anchor = calibration mean x_0", "heads": {}}
    for h in heads:
        k = {kk: (torch.cat(v).numpy() if kk not in ("Bbias", "r2n", "r2d") else np.array(v)) for kk, v in keep[h].items()}
        Ph = P[h]
        bp = k["bp"]
        prof = {str(j): {"mean": float(bp[:, j].mean()), "sd": float(bp[:, j].std()), "q05": float(np.quantile(bp[:, j], .05)),
                         "q95": float(np.quantile(bp[:, j], .95))} for j in range(8)}
        tail = bp[:, 16:]
        prof["tail"] = {"mean": float(tail.mean()), "sd_seq_mean_over_positions": float(tail.std(0).mean()),
                        "frac_abs_lt_0.1": float((np.abs(tail) < 0.1).mean())}
        Bp, Bin = k["Bp"], k["Bin"]; Bb = float(k["Bbias"][0])
        Bdec = by_bin(lambda lo, hi: {"B_mean": float(Bp[:, lo:hi + 1].mean()), "bias_part": Bb,
                                      "input_part_mean": float(Bin[:, lo:hi + 1].mean()),
                                      "input_part_sd_seq": float(Bin[:, lo:hi + 1].std(0).mean()),
                                      "bias_share_of_mean": Bb / float(Bp[:, lo:hi + 1].mean()) if abs(float(Bp[:, lo:hi + 1].mean())) > 1e-9 else None})
        z = k["z"]
        def curve_err(pred):
            return by_bin(lambda lo, hi: {
                "mean_curve_rel_err": float(np.median(np.abs(pred[:, lo:hi + 1].mean(0) - z[:, lo:hi + 1].mean(0))
                                                      / np.maximum(np.abs(z[:, lo:hi + 1].mean(0) - Ph["c"]), 1e-9))),
                "seq_var_explained_median": float(np.median([1 - np.var(z[:, j] - pred[:, j]) / max(np.var(z[:, j]), 1e-300)
                                                             for j in range(max(lo, 1), hi + 1)])) if pred.ndim == 2 and pred.shape[0] > 1 else None})
        B0 = Ph["Bbar"]
        i1 = np.maximum(ii, 1)
        zC = Ph["c"] + Ph["lam"] * 2 * np.exp(B0) / (2 * np.exp(B0) + i1 - 1); zC[0] = Ph["c"] + Ph["lam"]
        zC2 = None
        if Ph["fit_a"] is not None:
            aa, BB = Ph["fit_a"], Ph["fit_B"]
            zC2 = Ph["c"] + Ph["lam"] * aa * np.exp(BB) / (aa * np.exp(BB) + i1 + 1 - aa)
        pC = 2 * np.exp(B0) / (2 * np.exp(B0) + i1 - 1)
        res["heads"][h] = {
            "beta_prime_profile": prof, "B_decomposition": Bdec,
            "score_r2_ordinary_E_omitted": float(1 - k["r2n"].sum() / k["r2d"].sum()),
            "value_model": {"c": Ph["c"], "lambda": Ph["lam"]},
            "Bbar": B0, "fit_two_param": {"a": Ph["fit_a"], "B": Ph["fit_B"]},
            "z_predictions": {"zA_observed_attention": curve_err(k["zA"]), "zB_constant_B": curve_err(k["zB"]),
                              "zC_two_sinks_no_fit": curve_err(np.broadcast_to(zC, z.shape)),
                              "zC_fit_a_B": None if zC2 is None else curve_err(np.broadcast_to(zC2, z.shape))},
            "sink_mass": by_bin(lambda lo, hi: {"observed": float(k["p"][:, lo:hi + 1].mean()), "constant_B_model": float(k["pB"][:, lo:hi + 1].mean()),
                                                "two_sink_formula": float(pC[lo:hi + 1].mean()),
                                                "bos_weight_observed": float(k["a0"][:, lo:hi + 1].mean()),
                                                "bos_weight_formula": float((np.exp(B0) / (2 * np.exp(B0) + i1 - 1))[lo:hi + 1].mean())}),
            "z_first_positions": {str(j): {"observed_mean": float(z[:, j].mean()), "observed_sd": float(z[:, j].std()),
                                           "zA_mean": float(k["zA"][:, j].mean()), "two_sink_formula": float(zC[j])} for j in range(5)}}
        r = res["heads"][h]
        print(f"h{h}: beta' pos1 {prof['1']['mean']:.3f}±{prof['1']['sd']:.3f} pos2 {prof['2']['mean']:.3f}±{prof['2']['sd']:.3f} tail {prof['tail']['mean']:.4f} "
              f"| Bbar {B0:.2f} bias part {Bb:.2f} | score R2 {r['score_r2_ordinary_E_omitted']:.3f} | fit a {Ph['fit_a']} B {Ph['fit_B']}", flush=True)
    od = OUT_ROOT / "beta_model" / a.model
    jdump(res, od / f"L{l}_h{'-'.join(map(str, heads))}_{a.split}.json")
    print("wrote", od, flush=True)


if __name__ == "__main__":
    main()
