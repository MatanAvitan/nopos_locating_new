"""Mechanism-specific causal interventions (exps.md C1-C7, Step 5.5, OV-SVD).

Every intervention modifies the Layer-2 attention update (or Layer-1 BOS value
path for C7) and recomputes the remaining forward exactly. Reported on the
evaluation split with zero-shot (frozen head) metrics and an affine
recalibration fitted on the calibration split.

Usage: python interventions.py --run-id attn2_1h_L1024 --model attn2_1h --context-length 1024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS_ROOT, attn_bias_vector, attn_scores_weights,
                    forward_capture, head_weights, iter_batches, load_model,
                    ov_images, position_bins, update_summary)


def orth_basis(vectors: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis [d, k] for the span of rows of `vectors` [k, d]."""
    Q, _ = torch.linalg.qr(vectors.T.double())
    return Q


def project(x, Q):
    return (x @ Q) @ Q.T


@torch.no_grad()
def finish_from_o2(model, h1, o2_mod):
    post2 = h1 + o2_mod.float()
    ln4 = F.layer_norm(post2, model.block2.ln_2.weight.shape,
                       model.block2.ln_2.weight, model.block2.ln_2.bias, 1e-5)
    m2 = model.block2.mlp(ln4)
    h2 = post2 + m2
    x = model.ln_f(h2)
    return model.pos_head(x).squeeze(-1)


@torch.no_grad()
def finish_from_o2_custom(model, h1, o2_mod, mlp_fn=None, skip_mlp=False,
                          m2_project=None):
    post2 = h1 + o2_mod.float()
    ln4 = F.layer_norm(post2, model.block2.ln_2.weight.shape,
                       model.block2.ln_2.weight, model.block2.ln_2.bias, 1e-5)
    if skip_mlp:
        h2 = post2
    else:
        m2 = mlp_fn(ln4) if mlp_fn is not None else model.block2.mlp(ln4)
        if m2_project is not None:
            m2 = m2_project(m2)
        h2 = post2 + m2
    x = model.ln_f(h2)
    return model.pos_head(x).squeeze(-1)


class MetricAcc:
    def __init__(self, L):
        self.L = L
        self.preds = []

    def add(self, pred_norm, train_L):
        self.preds.append((pred_norm.detach().cpu().double().numpy()
                           * (train_L - 1)))

    def result(self, bins):
        p = np.concatenate(self.preds, 0)  # [n, T]
        t = np.arange(self.L)[None, :].repeat(len(p), 0).astype(np.float64)
        pf, tf = p.reshape(-1), t.reshape(-1)
        ss_tot = ((tf - tf.mean()) ** 2).sum()
        r2 = float(1 - ((pf - tf) ** 2).sum() / ss_tot)
        rr = float(np.corrcoef(pf, tf)[0, 1] ** 2) if np.std(pf) > 0 else 0.0
        out = {"r2_zeroshot": r2, "pearson_r2": rr,
               "mae": float(np.abs(pf - tf).mean())}
        for name, (a, b) in bins.items():
            sel = (t >= a) & (t <= b)
            out[f"mae_{name}"] = float(np.abs(p[sel] - t[sel]).mean())
        return out, p

    @staticmethod
    def recalibrate(p_cal, p_eval, L):
        """Affine recalibration fitted on calibration predictions."""
        t = np.arange(L)[None, :].repeat(len(p_cal), 0).reshape(-1)
        x = p_cal.reshape(-1)
        A = np.stack([x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        pe = p_eval * coef[0] + coef[1]
        te = np.arange(L)[None, :].repeat(len(p_eval), 0)
        ss_tot = ((te - te.mean()) ** 2).sum()
        return {"r2_recal": float(1 - ((pe - te) ** 2).sum() / ss_tot),
                "mae_recal": float(np.abs(pe - te).mean())}


def build_interventions(model, ref, device, H, d, seed=0):
    """Return dict name -> fn(parts) -> o2_mod (fp32 [B,T,d]).

    parts: dict with alpha [B,H,T] f64, g [B,H,T,d] f64 (normalized nonBOS
    aggregate), q [B,H,T,T-1] f32, y2 [B,H,T,d] f32, o2 [B,T,d] (actual).
    """
    w_b = ref["w_bos_2"].to(device)      # [H,d] f64
    w_nb = ref["w_nonbos_2"].to(device)
    b_attn = ref["b_attn_2"].to(device)  # [d]
    alpha_ref = ref["alpha_bos_ref_mean"].to(device)  # [H,T]
    dw = w_nb - w_b
    g_rng = torch.Generator(device="cpu").manual_seed(seed)

    hw2 = head_weights(model.block2.attn)
    B_ov_full = (hw2.W_o.double().reshape(-1, d).T.reshape(d, H, -1)
                 .permute(1, 0, 2))  # [H, d, dh]
    # full multi-head OV map as a single d x d matrix: sum_h W_O^h W_V^h
    B_full = sum((hw2.W_o[h].double() @ hw2.W_v[h].double()) for h in range(H))
    U, S, _ = torch.linalg.svd(B_full)

    mech_vecs = torch.cat([w_b, w_nb], 0)             # [2H, d]
    Q_mech = orth_basis(mech_vecs)
    Q_svd2 = U[:, :2]
    Q_dw = orth_basis(dw)                             # [d, H'] span of dws
    dw_sum = dw.sum(0)
    u_dwsum = (dw_sum / dw_sum.norm()).unsqueeze(1)
    u_svd1 = U[:, :1]

    rand_Qs = []
    for k in range(5):
        M = torch.randn(d, 2, generator=g_rng).double().to(device)
        rand_Qs.append(torch.linalg.qr(M)[0])
    rand_us = []
    for k in range(5):
        v = torch.randn(d, generator=g_rng).double().to(device)
        rand_us.append((v / v.norm()).unsqueeze(1))
    # energy-matched: random 2-dim inside top-10 SVD subspace
    energy_Qs = []
    for k in range(3):
        M = torch.randn(10, 2, generator=g_rng).double().to(device)
        energy_Qs.append(U[:, :10] @ torch.linalg.qr(M)[0])

    def mix(alpha, bos_vec, nonbos_term):
        return (alpha[..., None] * bos_vec[None, :, None, :]
                + nonbos_term).sum(1) + b_attn[None, None, :]

    iv = {}
    iv["baseline"] = lambda p: p["o2"].double()
    # C1 / C5(eta=0): ideal mixture with realized BOS weight
    iv["C1_ideal_mixture"] = lambda p: mix(
        p["alpha"], w_b, (1 - p["alpha"][..., None]) * w_nb[None, :, None, :])
    # C5 residual-only: keep BOS part + realized residual, drop fixed nonBOS
    iv["C5_residual_only"] = lambda p: mix(
        p["alpha"], w_b,
        (1 - p["alpha"][..., None]) * (p["g"] - w_nb[None, :, None, :]))
    # C4 collapse endpoints
    iv["C4_bos_to_nonbos"] = lambda p: mix(
        p["alpha"], w_nb, (1 - p["alpha"][..., None]) * p["g"])
    iv["C4_nonbos_to_bos"] = lambda p: mix(
        p["alpha"], w_b, (1 - p["alpha"][..., None]) * w_b[None, :, None, :])

    def centered(fn):
        def wrapper(p):
            o2c = p["o2"].double() - b_attn[None, None, :]
            return fn(o2c) + b_attn[None, None, :]
        return wrapper
    # C2 span retention / ablation
    iv["C2_retain_mech_span"] = centered(lambda o: project(o, Q_mech))
    iv["C2_ablate_mech_span"] = centered(lambda o: o - project(o, Q_mech))
    iv["C2_retain_top2_svd"] = centered(lambda o: project(o, Q_svd2))
    iv["C2_ablate_top2_svd"] = centered(lambda o: o - project(o, Q_svd2))
    for k, Q in enumerate(rand_Qs):
        iv[f"C2_retain_random2d_{k}"] = centered(lambda o, Q=Q: project(o, Q))
    for k, Q in enumerate(energy_Qs):
        iv[f"C2_retain_energymatched2d_{k}"] = centered(lambda o, Q=Q: project(o, Q))
    # C3 difference-axis ablation
    iv["C3_ablate_dw_sum"] = centered(lambda o: o - project(o, u_dwsum))
    iv["C3_ablate_dw_span"] = centered(lambda o: o - project(o, Q_dw))
    iv["C3_ablate_top1_svd"] = centered(lambda o: o - project(o, u_svd1))
    for k, u in enumerate(rand_us):
        iv[f"C3_ablate_random1d_{k}"] = centered(lambda o, u=u: o - project(o, u))
    # C6 BOS-weight interventions (renormalize nonBOS mass explicitly)
    def c6(alpha_new_fn):
        def fn(p):
            a_new = alpha_new_fn(p).clamp(1e-6, 1 - 1e-6)
            return mix(a_new, w_b, (1 - a_new[..., None]) * p["g"])
        return fn
    iv["C6_alpha_refmean"] = c6(lambda p: alpha_ref[None].expand_as(p["alpha"]))
    def perm_alpha(p):
        T = p["alpha"].shape[-1]
        perm = torch.randperm(T - 1, generator=g_rng).to(p["alpha"].device)
        a = p["alpha"].clone()
        a[..., 1:] = a[..., 1:][..., perm]
        return a
    iv["C6_alpha_permuted"] = c6(perm_alpha)
    iv["C6_alpha_clamped"] = c6(
        lambda p: p["alpha"][..., 1:].mean(-1, keepdim=True).expand_as(p["alpha"]))
    def uniform_q(p):
        # preserve alpha, replace realized aggregate g by uniform nonBOS mean
        return mix(p["alpha"], w_b,
                   (1 - p["alpha"][..., None]) * p["g_unif"])
    iv["C6_uniform_nonbos_attn"] = uniform_q
    return iv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = RESULTS_ROOT / args.run_id
    L = args.context_length
    device = args.device
    model, meta = load_model(args.model, device)
    ref = torch.load(out / "ref_stats.pt", map_location="cpu", weights_only=False)
    splits = np.load(out / "sequence_ids.npz")
    H, d = model.config.n_head, model.config.n_embd
    hw2 = head_weights(model.block2.attn)
    hw1 = head_weights(model.block1.attn)
    train_L = meta["block_size_train"]
    bins = position_bins(L)
    w_nb = ref["w_nonbos_2"].to(device)
    b_attn2 = ref["b_attn_2"].to(device)

    iv = build_interventions(model, ref, device, H, d)

    # 5.5 MLP interventions need a fitted linearization of MLP2 (on calibration)
    A_acc = torch.zeros(d + 1, d + 1, device=device, dtype=torch.float64)
    B_acc = torch.zeros(d + 1, d, device=device, dtype=torch.float64)

    @torch.no_grad()
    def parts_of(cap):
        alpha = cap["attn2_w"][:, :, :, 0].double()
        y2 = ov_images(cap["x2"], hw2)
        m_mass = (1.0 - alpha).clamp_min(1e-30)
        q = cap["attn2_w"][..., 1:].float() / m_mass.float()[..., None]
        g = torch.einsum("bhts,bhsd->bhtd", q, y2[:, :, 1:]).double()
        i_idx = torch.arange(L, device=device, dtype=torch.float32)
        y2_cummean = y2[:, :, 1:].cumsum(2) / i_idx[None, None, 1:, None].clamp(min=1)
        g_unif = torch.cat([torch.zeros_like(y2[:, :, :1]), y2_cummean], 2).double()
        return {"alpha": alpha, "g": g, "g_unif": g_unif, "q": q, "y2": y2,
                "o2": cap["o2"]}

    # ---- calibration pass: MLP linearization + per-intervention preds ----
    print("calibration pass...")
    cal_preds = {}
    for s, idx in iter_batches(splits["calibration"], L, args.batch_size, device):
        cap = forward_capture(model, idx)
        X = cap["ln4"].double().reshape(-1, d)
        X1 = torch.cat([X, torch.ones(len(X), 1, device=device,
                                      dtype=torch.float64)], 1)
        Ym = cap["m2"].double().reshape(-1, d)
        A_acc += X1.T @ X1
        B_acc += X1.T @ Ym
        p = parts_of(cap)
        for name, fn in iv.items():
            pred = finish_from_o2(model, cap["h1"], fn(p))
            cal_preds.setdefault(name, []).append(
                pred.cpu().double().numpy() * (train_L - 1))
        del cap, p
    reg = 1e-4 * torch.diag(A_acc).mean() * torch.eye(d + 1, device=device,
                                                      dtype=torch.float64)
    reg[-1, -1] = 0
    W_lin = torch.linalg.solve(A_acc + reg, B_acc)  # [d+1, d]

    def mlp_linearized(ln4):
        X = ln4.double().reshape(-1, d)
        out_ = X @ W_lin[:-1] + W_lin[-1]
        return out_.reshape(ln4.shape).float()

    wg = (model.pos_head.weight.float().squeeze(0) * model.ln_f.weight.float())
    u_read = (wg / wg.norm()).double()

    def m2_keep_read(m2):
        return ((m2.double() @ u_read)[..., None] * u_read[None, None, :]).float()

    # extra 5.5 variants (need custom finish)
    extra = {
        "S5_remove_mlp2": dict(skip_mlp=True),
        "S5_linearize_mlp2": dict(mlp_fn=mlp_linearized),
        "S5_mlp2_readdir_only": dict(m2_project=m2_keep_read),
    }
    # patch BOS-weight coordinate to its reference position curve
    alpha_ref = ref["alpha_bos_ref_mean"].to(device)
    w_b_ = ref["w_bos_2"].to(device)
    dw_ = (w_nb - w_b_)
    dw_sum = dw_.sum(0)
    u_dw = dw_sum / dw_sum.norm()

    def patch_bos_coord(p):
        o2c = p["o2"].double() - b_attn2[None, None, :]
        coord = o2c @ u_dw
        # reference coordinate: from ideal mixture with reference alpha
        ref_mix = ((alpha_ref[..., None] * w_b_[:, None, :]
                    + (1 - alpha_ref[..., None]) * w_nb[:, None, :]).sum(0))
        ref_coord = ref_mix @ u_dw                     # [T]
        o2c = o2c + (ref_coord[None, :] - coord)[..., None] * u_dw[None, None, :]
        return o2c + b_attn2[None, None, :]

    iv2 = {"S5_patch_bos_coordinate": patch_bos_coord}

    # C7: layer-1 BOS value path
    y1_bos_fixed = None
    torch.manual_seed(7)

    @torch.no_grad()
    def c7_forward(idx, mode):
        emb = model.wte(idx)
        x1 = F.layer_norm(emb, model.block1.ln_1.weight.shape,
                          model.block1.ln_1.weight, model.block1.ln_1.bias, 1e-5)
        _, a1 = attn_scores_weights(x1, hw1)
        y1 = ov_images(x1, hw1)                       # [B,H,T,d]
        o1 = torch.einsum("bhts,bhsd->bhtd", a1, y1).sum(1) \
            + attn_bias_vector(hw1)[None, None, :]
        y1_bos = y1[:, :, 0]                          # [B,H,d]
        a_bos1 = a1[:, :, :, 0]                       # [B,H,T]
        if mode == "remove":
            repl = torch.zeros_like(y1_bos)
        elif mode == "fixed_random":
            nonlocal y1_bos_fixed
            if y1_bos_fixed is None:
                r = torch.randn_like(y1_bos[0])
                y1_bos_fixed = r / r.norm(dim=-1, keepdim=True) \
                    * y1_bos[0].norm(dim=-1, keepdim=True)
            repl = y1_bos_fixed[None].expand_as(y1_bos)
        elif mode == "seq_varying":
            j = torch.randint(1, idx.shape[1], (1,)).item()
            repl = y1[:, :, j]
        delta = torch.einsum("bht,bhd->btd", a_bos1, repl - y1_bos)
        o1_mod = o1 + delta
        post1 = emb + o1_mod
        ln2 = F.layer_norm(post1, model.block1.ln_2.weight.shape,
                           model.block1.ln_2.weight, model.block1.ln_2.bias, 1e-5)
        m1 = model.block1.mlp(ln2)
        h1 = post1 + m1
        x2 = F.layer_norm(h1, model.block2.ln_1.weight.shape,
                          model.block2.ln_1.weight, model.block2.ln_1.bias, 1e-5)
        s2, a2 = attn_scores_weights(x2, hw2)
        y2 = ov_images(x2, hw2)
        o2 = torch.einsum("bhts,bhsd->bhtd", a2, y2).sum(1) \
            + attn_bias_vector(hw2)[None, None, :]
        pred = finish_from_o2(model, h1, o2.double())
        alpha2 = a2[:, :, :, 0]
        return pred, alpha2, ln2

    # second calibration pass for the S5 family (needs W_lin fitted above)
    print("calibration pass (S5 family)...")
    for s, idx in iter_batches(splits["calibration"], L, args.batch_size, device):
        cap = forward_capture(model, idx)
        p = parts_of(cap)
        for name, kw in extra.items():
            pred = finish_from_o2_custom(model, cap["h1"],
                                         cap["o2"].double(), **kw)
            cal_preds.setdefault(name, []).append(
                pred.cpu().double().numpy() * (train_L - 1))
        for name, fn in iv2.items():
            pred = finish_from_o2(model, cap["h1"], fn(p))
            cal_preds.setdefault(name, []).append(
                pred.cpu().double().numpy() * (train_L - 1))
        del cap, p

    # ---- evaluation pass -------------------------------------------------
    print("evaluation pass...")
    accs = {name: MetricAcc(L) for name in
            list(iv) + list(extra) + list(iv2)
            + ["C7_remove", "C7_fixed_random", "C7_seq_varying"]}
    c7_alpha2 = {m: [] for m in ["actual", "C7_remove", "C7_fixed_random",
                                 "C7_seq_varying"]}
    c7_S = {m: [] for m in c7_alpha2}
    x0_1 = ref["x0_1"].to(device).float()

    for s, idx in iter_batches(splits["evaluation"], L, args.batch_size, device):
        cap = forward_capture(model, idx)
        p = parts_of(cap)
        for name, fn in iv.items():
            accs[name].add(finish_from_o2(model, cap["h1"], fn(p)), train_L)
        for name, kw in extra.items():
            accs[name].add(finish_from_o2_custom(
                model, cap["h1"], cap["o2"].double(), **kw), train_L)
        for name, fn in iv2.items():
            accs[name].add(finish_from_o2(model, cap["h1"], fn(p)), train_L)
        c7_alpha2["actual"].append(
            cap["attn2_w"][:, :, :, 0].mean(1).cpu().numpy())
        c7_S["actual"].append((cap["h1bar"] @ x0_1).cpu().numpy())
        for mode in ["remove", "fixed_random", "seq_varying"]:
            pred, alpha2, ln2 = c7_forward(idx, mode)
            accs[f"C7_{mode}"].add(pred, train_L)
            c7_alpha2[f"C7_{mode}"].append(alpha2.mean(1).cpu().numpy())
            c7_S[f"C7_{mode}"].append((ln2 @ x0_1).cpu().numpy())
        del cap, p

    results = {}
    eval_preds = {}
    for name, acc in accs.items():
        res, p_eval = acc.result(bins)
        eval_preds[name] = p_eval
        if name in cal_preds:
            p_cal = np.concatenate(cal_preds[name], 0)
            res.update(MetricAcc.recalibrate(p_cal, p_eval, L))
        results[name] = res
    # aggregate random controls
    for fam in ["C2_retain_random2d", "C3_ablate_random1d",
                "C2_retain_energymatched2d"]:
        vals = [results[k]["r2_zeroshot"] for k in results if k.startswith(fam)]
        if vals:
            results[fam + "_mean_r2"] = float(np.mean(vals))
    # C7 side effects: alpha2/S curves
    c7_summary = {}
    for m in c7_alpha2:
        a = np.concatenate(c7_alpha2[m], 0).mean(0)
        S = np.concatenate(c7_S[m], 0).mean(0)
        c7_summary[m] = {
            "alpha2_mean_at": {str(k): float(a[k]) for k in
                               [1, 4, 16, 64, 256, min(1023, L - 1)] if k < L},
            "S_mean_at": {str(k): float(S[k]) for k in
                          [1, 4, 16, 64, 256, min(1023, L - 1)] if k < L},
        }
    results["C7_side_effects"] = c7_summary
    np.savez(out / "intervention_preds.npz",
             **{k: v for k, v in eval_preds.items()})
    update_summary(out, "interventions", results)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items()
                           if kk in ("r2_zeroshot", "pearson_r2", "r2_recal")})
                      for k, v in results.items() if k != "C7_side_effects"},
                     indent=2))


if __name__ == "__main__":
    main()
