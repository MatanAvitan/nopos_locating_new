"""Canonical extraction pipeline (exps.md P0).

Phase A (reference split): estimate the frozen global vectors and
position-wise reference statistics.
Phase B (evaluation split): compute per-sequence, per-position (per-head)
diagnostics for Steps 1, 3, 4, 5, streaming to chunked arrays.

Usage:
  python extract.py --model attn2_1h --context-length 1024
  python extract.py --model full12h  --context-length 1024 --batch-size 4
  python extract.py --model attn2_1h --context-length 1024 --init   # at-init control
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BOS_TOKEN_ID, CHECKPOINTS, RESULTS_ROOT, SPLIT_SIZES,
                    attn_bias_vector, attn_scores_weights, batch_from_offsets,
                    build_splits, forward_capture, head_weights, iter_batches,
                    load_model, ov_images, run_dir, save_config, update_summary)


# ---------------------------------------------------------------------------
# Phase A: reference statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def reference_pass(model, offsets, L, batch_size, device):
    """Estimate global vectors on the reference split (frozen before eval).

    Returns dict with:
      x0_1: x^{(1)}_0 (fixed layer-1 BOS input)          [d]
      w_bos_1, w_nonbos_1: layer-1 LN2 mean directions    [d]
      x0_2: x^{(2)}_0                                     [d]
      w_bos_2: per-head B_OV x0                           [H, d]
      w_nonbos_2: per-head E_{b,j>0}[y_j]                 [H, d]
      w_nonbos_2_half1/half2: split-half stability        [H, d]
      mu_j: per-head, per-position E_b[y_j] (j>=1)        [H, T, d]
      cbar_ref_mean/median: position-wise log-cbar stats  [H, T]
      alpha_bos_ref_mean: position-wise mean BOS weight   [H, T]
      b_attn_2: layer-2 attention affine bias             [d]
    """
    H = model.config.n_head
    d = model.config.n_embd
    n = len(offsets)
    half = n // 2

    sum_h1bar_bos = torch.zeros(d, device=device, dtype=torch.float64)
    sum_h1bar_nonbos = torch.zeros(d, device=device, dtype=torch.float64)
    cnt_nonbos = 0
    sum_y = torch.zeros(H, d, device=device, dtype=torch.float64)
    sum_y_h1 = torch.zeros(H, d, device=device, dtype=torch.float64)
    sum_y_h2 = torch.zeros(H, d, device=device, dtype=torch.float64)
    cnt_y = cnt_y_h1 = cnt_y_h2 = 0
    sum_mu = torch.zeros(H, L, d, device=device, dtype=torch.float64)
    cnt_seq = 0
    logc_all = []
    alpha_bos_sum = torch.zeros(H, L, device=device, dtype=torch.float64)

    x0_1 = x0_2 = None
    hw2 = head_weights(model.block2.attn)

    for s, idx in iter_batches(offsets, L, batch_size, device):
        cap = forward_capture(model, idx)
        B = idx.shape[0]
        if x0_1 is None:
            x0_1 = cap["x1"][0, 0].double().clone()
            x0_2 = cap["x2"][0, 0].double().clone()
        sum_h1bar_bos += cap["h1bar"][:, 0].double().sum(0)
        sum_h1bar_nonbos += cap["h1bar"][:, 1:].double().sum((0, 1))
        cnt_nonbos += B * (L - 1)

        y = ov_images(cap["x2"], hw2)  # [B,H,T,d]
        sum_y += y[:, :, 1:].double().sum((0, 2))
        cnt_y += B * (L - 1)
        in_h1 = s < half
        if in_h1:
            sum_y_h1 += y[:, :, 1:].double().sum((0, 2)); cnt_y_h1 += B * (L - 1)
        else:
            sum_y_h2 += y[:, :, 1:].double().sum((0, 2)); cnt_y_h2 += B * (L - 1)
        sum_mu += y.double().sum(0)
        cnt_seq += B

        # Step-3 reference: log cbar per head (float64 logsumexp)
        logc = compute_log_cbar(cap["scores2"])  # [B,H,T]
        logc_all.append(logc.cpu())
        alpha_bos_sum += cap["attn2_w"][:, :, :, 0].double().sum(0)
        del cap, y
    logc_all = torch.cat(logc_all, 0)  # [n,H,T]

    ref = {
        "x0_1": x0_1.cpu(), "x0_2": x0_2.cpu(),
        "w_bos_1": (sum_h1bar_bos / n).cpu(),
        "w_nonbos_1": (sum_h1bar_nonbos / cnt_nonbos).cpu(),
        "w_nonbos_2": (sum_y / cnt_y).cpu(),
        "w_nonbos_2_half1": (sum_y_h1 / max(cnt_y_h1, 1)).cpu(),
        "w_nonbos_2_half2": (sum_y_h2 / max(cnt_y_h2, 1)).cpu(),
        "mu_j": (sum_mu / cnt_seq).cpu(),  # [H,L,d]; row 0 is BOS (unused)
        "logcbar_ref_mean": logc_all.mean(0),
        "logcbar_ref_median": logc_all.median(0).values,
        "cbar_ref_mean": logc_all.exp().mean(0),
        "cbar_ref_median": logc_all.exp().median(0).values,
        "alpha_bos_ref_mean": (alpha_bos_sum / n).cpu(),
        "b_attn_2": attn_bias_vector(hw2).double().cpu(),
    }
    # w_bos_2 = B_OV^h x0_2 (manuscript convention: no value bias, see ov_images)
    v0 = torch.einsum("hed,d->he", hw2.W_v.double(), x0_2)
    ref["w_bos_2"] = torch.einsum("hde,he->hd", hw2.W_o.double(), v0).cpu()
    return ref


def compute_log_cbar(scores2: torch.Tensor) -> torch.Tensor:
    """log cbar_{b,i} = logsumexp_{j=1..i}(s_ij - s_i0) - log i, float64.

    scores2: [B,H,T,T] with -inf above diagonal. Position i=0 -> nan.
    """
    s = scores2.double()
    rel = s - s[..., 0:1]
    rel = rel[..., 1:]  # drop BOS key; masked entries stay -inf
    lse = torch.logsumexp(rel, dim=-1)  # [B,H,T]
    T = s.shape[-2]
    i_idx = torch.arange(T, device=s.device, dtype=torch.float64)
    logc = lse - torch.log(i_idx.clamp(min=1))
    logc[..., 0] = float("nan")
    return logc


# ---------------------------------------------------------------------------
# Phase B helpers
# ---------------------------------------------------------------------------

def uniformity_metrics(attn_w: torch.Tensor):
    """Step 1.1 metrics per (b,h,i) for one block's attention weights.

    Returns dict of [B,H,T] tensors: entropy, norm_entropy, kl_uniform,
    tv_uniform, alpha_bos_ratio (alpha_bos * (i+1)), max_attn.
    """
    B, H, T, _ = attn_w.shape
    a = attn_w.double().clamp_min(1e-30)
    mask = torch.tril(torch.ones(T, T, device=a.device, dtype=torch.bool))
    a = a * mask  # zero out (numerically negligible) upper entries
    ent = -(a * a.log()).where(mask, torch.zeros_like(a)).sum(-1)
    i_idx = torch.arange(T, device=a.device, dtype=torch.float64)
    log_n = torch.log(i_idx + 1)
    norm_ent = ent / log_n.clamp(min=1e-12)
    norm_ent[..., 0] = 1.0
    # KL(attn || uniform) = log(i+1) - H(attn)
    kl = log_n - ent
    # uniform prob on the causal prefix of query i is 1/(i+1) for keys j<=i
    unif = (1.0 / (i_idx + 1))[None, None, :, None]
    tv = 0.5 * (((a - unif) * mask).abs()).sum(-1)
    alpha_bos_ratio = a[..., 0] * (i_idx + 1)
    max_attn = a.max(-1).values
    return {"entropy": ent, "norm_entropy": norm_ent, "kl_uniform": kl,
            "tv_uniform": tv, "alpha_bos_ratio": alpha_bos_ratio,
            "max_attn": max_attn}


def ln_no_affine(x, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / (var + eps).sqrt()


def apply_ln(module, x):
    return F.layer_norm(x, module.weight.shape, module.weight, module.bias, 1e-5)


def cos_and_relerr(a, b, eps=1e-12):
    """Cosine, best scalar fit residual rel-norm, and rel norm error along dim=-1."""
    dot = (a * b).sum(-1)
    na = a.norm(dim=-1).clamp_min(eps)
    nb = b.norm(dim=-1).clamp_min(eps)
    cos = dot / (na * nb)
    # best scalar c minimizing ||a - c b||: c = dot/nb^2; residual rel to ||a||
    resid = (a - (dot / nb.pow(2)).unsqueeze(-1) * b).norm(dim=-1) / na
    relerr = (a - b).norm(dim=-1) / nb
    return cos, resid, relerr


# ---------------------------------------------------------------------------
# Phase B: evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluation_pass(model, ref, offsets, L, batch_size, device, out: Path,
                    lag_set=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    H = model.config.n_head
    d = model.config.n_embd
    hw1 = head_weights(model.block1.attn)
    hw2 = head_weights(model.block2.attn)

    w_nb = ref["w_nonbos_2"].to(device)            # [H,d] float64
    w_b = ref["w_bos_2"].to(device)                # [H,d]
    mu_j = ref["mu_j"].to(device)                  # [H,L,d]
    dw = w_nb - w_b                                # Delta w per head
    dw_norm2 = dw.pow(2).sum(-1)                   # [H]
    w_nb_norm = w_nb.norm(dim=-1)                  # [H]
    x0_1 = ref["x0_1"].to(device)                  # [d]
    b_attn2 = ref["b_attn_2"].to(device)           # [d]
    cbar_ref_med = ref["cbar_ref_median"].to(device)  # [H,T]
    alpha_ref_mean = ref["alpha_bos_ref_mean"].to(device)  # [H,T]

    # accumulators (streamed to lists of numpy chunks)
    acc = {}

    def push(key, tensor):
        acc.setdefault(key, []).append(tensor.detach().cpu().float().numpy())

    b_attn1 = attn_bias_vector(hw1).to(device)
    # position-conditioned mean trajectories (Step 2.3 / 4.8 dimensionality)
    sum_o2c = torch.zeros(L, d, device=device, dtype=torch.float64)
    sum_ohead = torch.zeros(H, L, d, device=device, dtype=torch.float64)
    n_traj = 0

    t0 = time.time()
    for s, idx in iter_batches(offsets, L, batch_size, device):
        cap = forward_capture(model, idx)
        B = idx.shape[0]
        i_idx = torch.arange(L, device=device, dtype=torch.float64)

        # ---------------- Step 1.1: layer-1 attention uniformity ----------
        for k, v in uniformity_metrics(cap["attn1_w"]).items():
            push(f"step1/{k}", v)

        # ---------------- Step 1.2: three approximations ------------------
        x1 = cap["x1"].double()                       # [B,T,d]
        # counterfactual: uniform attention, actual OV/residual/LN
        y1 = ov_images(cap["x1"], hw1)                # [B,H,T,d] fp32
        y1_cum = y1.cumsum(2) / (i_idx.float()[None, None, :, None] + 1)
        o1_unif = y1_cum.sum(1) + b_attn1[None, None, :]
        post1_unif = cap["emb"] + o1_unif
        h1bar_unif = apply_ln(model.block1.ln_2, post1_unif).double()
        # stylized proxy
        p = x1.cumsum(1) / (i_idx[None, :, None] + 1).sqrt()
        h1bar = cap["h1bar"].double()
        for name, (a, b) in {
            "actual_vs_unif": (h1bar, h1bar_unif),
            "unif_vs_proxy": (h1bar_unif, p),
            "actual_vs_proxy": (h1bar, p),
        }.items():
            cos, scalefit, relerr = cos_and_relerr(a, b)
            push(f"step1/approx_{name}_cos", cos)
            push(f"step1/approx_{name}_scalefit_resid", scalefit)
            push(f"step1/approx_{name}_relerr", relerr)

        # ---------------- Step 1.3: signal/error algebra ------------------
        S = (h1bar * x0_1[None, None, :]).sum(-1)                    # [B,T]
        x0n2 = x0_1.pow(2).sum()
        A_i = x0n2 / (i_idx + 1).sqrt()                              # [T]
        Tcum = ((x1 * x0_1[None, None, :]).sum(-1)).cumsum(1) \
            - (x1[:, 0] * x0_1[None, :]).sum(-1, keepdim=True)       # sum_{j=1..i}
        E = Tcum / (i_idx[None, :] + 1).sqrt()
        push("step1/S", S)
        push("step1/E", E)
        push("step1/T", Tcum)
        if s == 0:
            np.save(out / "step1_A_i.npy", A_i.cpu().numpy())

        # ---------------- Step 1.4: orthogonality -------------------------
        push("step1/x1_norm", x1.norm(dim=-1))
        push("step1/cos_with_x0", (x1 @ x0_1) / (x1.norm(dim=-1) * x0_1.norm()).clamp_min(1e-12))
        x1n = x1 / x1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        for lag in lag_set:
            if lag < L - 1:
                c = (x1n[:, 1:L - lag] * x1n[:, 1 + lag:]).sum(-1)   # nonBOS pairs
                push(f"step1/paircos_lag{lag}", c.mean(1))           # per-seq mean

        # ---------------- Step 3: cbar --------------------------------------
        logc = compute_log_cbar(cap["scores2"])                      # [B,H,T]
        alpha_bos = cap["attn2_w"][:, :, :, 0].double()              # [B,H,T]
        push("step3/log_cbar", logc)
        push("step3/alpha_bos", alpha_bos)
        # exact identity check: alpha == 1/(1+i cbar)
        alpha_pred = 1.0 / (1.0 + i_idx[None, None, :] * logc.exp())
        iden_err = (alpha_pred - alpha_bos).abs()
        iden_err[..., 0] = float("nan")
        push("step3/identity_abs_err", iden_err)
        # 3.4 position-only prediction from reference median cbar
        alpha_ref_pred = 1.0 / (1.0 + i_idx[None, None, :] * cbar_ref_med[None])
        push("step3/alpha_err_refpred", (alpha_ref_pred - alpha_bos))
        # first-order sensitivity prediction
        sens = -(i_idx[None, None, :] / (1 + i_idx[None, None, :] * cbar_ref_med[None]).pow(2)) \
            * (logc.exp() - cbar_ref_med[None])
        push("step3/alpha_err_firstorder", sens)

        # ---------------- Step 4: residual suite (per head) ---------------
        # Vector-valued matmuls in fp32 (TF32-eligible); scalar reductions fp64.
        w_nb32 = w_nb.float()
        a2 = cap["attn2_w"].float()                                  # [B,H,T,T]
        y2 = ov_images(cap["x2"], hw2)                               # [B,H,T,d] fp32
        m_mass = (1.0 - alpha_bos).clamp_min(1e-30)                  # [B,H,T] f64
        q = a2[..., 1:] / m_mass.float()[..., None]                  # [B,H,T,T-1]
        q64 = q.double()
        r_norm = (y2[:, :, 1:].double() - w_nb[None, :, None, :]).norm(dim=-1)
        # realized aggregate and residual
        g = torch.einsum("bhts,bhsd->bhtd", q, y2[:, :, 1:]).double()  # [B,H,T,d]
        rbar = g - w_nb[None, :, None, :]
        rbar_norm = rbar.norm(dim=-1)
        e = rbar_norm / w_nb_norm[None, :, None]
        push("step4/e", e)
        push("step4/cos_g_w", (g * w_nb[None, :, None, :]).sum(-1)
             / (g.norm(dim=-1) * w_nb_norm[None, :, None]).clamp_min(1e-30))
        push("step4/normratio_g_w", g.norm(dim=-1) / w_nb_norm[None, :, None])
        # convex bound
        Bnd = torch.einsum("bhts,bhs->bht", q64, r_norm) / w_nb_norm[None, :, None]
        push("step4/bound_B", Bnd)
        # diagonal/cross decomposition
        D = torch.einsum("bhts,bhs->bht", q64.pow(2), r_norm.pow(2))
        C = rbar_norm.pow(2) - D
        push("step4/D", D)
        push("step4/C", C)
        sumq2 = q64.pow(2).sum(-1)
        push("step4/sumq2", sumq2)
        push("step4/A_conc", i_idx[None, None, :] * sumq2)
        push("step4/R_scale", (D / sumq2.clamp_min(1e-30)).sqrt() / w_nb_norm[None, :, None])
        push("step4/maxq", q64.max(-1).values)
        ent_q = -(q64 * q64.clamp_min(1e-30).log()).sum(-1)
        push("step4/entropy_q", ent_q / torch.log(i_idx.clamp(min=2))[None, None, :])
        # 4.5 position-specific means
        mu_term = torch.einsum("bhts,hsd->bhtd", q,
                               (mu_j[:, 1:] - w_nb[:, None, :]).float()).double()
        eps_term = rbar - mu_term
        push("step4/mu_term_norm", mu_term.norm(dim=-1) / w_nb_norm[None, :, None])
        push("step4/eps_term_norm", eps_term.norm(dim=-1) / w_nb_norm[None, :, None])
        push("step4/mu_eps_inner", (mu_term * eps_term).sum(-1) / w_nb_norm[None, :, None].pow(2))
        # 4.6 controls: uniform weights; shuffled attention-residual pairing
        y2_cummean = y2[:, :, 1:].cumsum(2) / i_idx.float()[None, None, 1:, None].clamp(min=1)
        # g_unif at query i averages keys 1..i -> cummean index i-1; pad front
        g_unif = torch.cat([torch.full_like(y2[:, :, :1], float("nan")),
                            y2_cummean], dim=2).double()
        e_unif = (g_unif - w_nb[None, :, None, :]).norm(dim=-1) / w_nb_norm[None, :, None]
        push("step4/e_uniform_control", e_unif)
        if B > 1:
            y2_shuf = y2.roll(1, dims=0)
            g_shuf = torch.einsum("bhts,bhsd->bhtd", q, y2_shuf[:, :, 1:]).double()
            e_shuf = (g_shuf - w_nb[None, :, None, :]).norm(dim=-1) / w_nb_norm[None, :, None]
            push("step4/e_shuffled_control", e_shuf)
        # 4.7 projected residual
        eta = m_mass[..., None] * rbar
        rho = (eta * dw[None, :, None, :]).sum(-1) / dw_norm2[None, :, None]
        push("step4/rho", rho)
        push("step4/rho_over_alpha", rho.abs() / alpha_bos.clamp_min(1e-30))
        push("step4/rho_over_mass", rho / m_mass)
        # per-key-position residual norms (j-indexed; store once per seq)
        push("step4/r_norm_by_j", r_norm / w_nb_norm[None, :, None])

        # ---------------- Step 4.8 / Step 5 -------------------------------
        o2c = cap["o2"].double() - b_attn2[None, None, :]            # bias-centered
        # per-head realized output (bias-free): alpha*w_bos + m*g
        o_head = (alpha_bos[..., None] * w_b[None, :, None, :]
                  + m_mass[..., None] * g)                           # [B,H,T,d]
        sum_o2c += o2c.sum(0)
        sum_ohead += o_head.sum(0)
        n_traj += B
        # ideal mixture reconstruction (sum over heads)
        o_star = (alpha_bos[..., None] * w_b[None, :, None, :]
                  + (1 - alpha_bos[..., None]) * w_nb[None, :, None, :]).sum(1)
        recon_err = (o2c - o_star).norm(dim=-1) / o2c.norm(dim=-1).clamp_min(1e-30)
        push("step5/recon_rel_err", recon_err)
        dw_sum = dw.sum(0)
        proj_recon_err = ((o2c - o_star) @ dw_sum) / dw_sum.pow(2).sum().clamp_min(1e-30)
        push("step5/recon_proj_err", proj_recon_err)
        # affine coordinate (per head projection on its own dw)
        Y = torch.einsum("btd,hd->bht", o2c, dw)
        push("step5/Y_proj", Y)
        # readout decomposition through ln_f (exact per-token affine)
        h2 = cap["h2"].double()
        lnf = model.ln_f
        gamma = lnf.weight.double()
        beta = lnf.bias.double() if lnf.bias is not None else torch.zeros_like(gamma)
        w_read = model.pos_head.weight.double().squeeze(0)
        b_read = model.pos_head.bias.double().squeeze() if model.pos_head.bias is not None else 0.0
        mu_h = h2.mean(-1, keepdim=True)
        sd_h = (h2.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        wg = w_read * gamma                                          # [d]
        scale = 1.0 / sd_h.squeeze(-1)                               # [B,T]
        h1_ = cap["h1"].double()
        o2_ = cap["o2"].double()
        m2_ = cap["m2"].double()
        term_h1 = (h1_ @ wg) * scale
        term_o2 = (o2_ @ wg) * scale
        term_m2 = (m2_ @ wg) * scale
        const = -(mu_h.squeeze(-1) * wg.sum()) * scale + (w_read * beta).sum() + b_read
        push("step5/term_h1", term_h1)
        push("step5/term_o2", term_o2)
        push("step5/term_m2", term_m2)
        push("step5/term_const", const)
        push("step5/pred", cap["pred"].double())
        del cap, y1, y2, q, q64, g, rbar, a2, mu_term, eps_term
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        done = s + B
        print(f"  eval {done}/{len(offsets)} seqs  ({time.time()-t0:.0f}s)", flush=True)

    np.save(out / "o2c_mean_traj.npy", (sum_o2c / n_traj).cpu().numpy())
    np.save(out / "ohead_mean_traj.npy", (sum_ohead / n_traj).cpu().numpy())
    # save chunked accumulators
    for key, chunks in acc.items():
        arr = np.concatenate(chunks, 0)
        fname = key.replace("/", "__") + ".npy"
        np.save(out / fname, arr)
    return sorted(acc.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CHECKPOINTS), required=True)
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", action="store_true", help="analyze at initialization")
    ap.add_argument("--n-ref", type=int, default=SPLIT_SIZES["reference"])
    ap.add_argument("--n-cal", type=int, default=SPLIT_SIZES["calibration"])
    ap.add_argument("--n-eval", type=int, default=SPLIT_SIZES["evaluation"])
    ap.add_argument("--split-seed", type=int, default=1234)
    args = ap.parse_args()

    L = args.context_length
    run_id = f"{args.model}_L{L}" + ("_init" if args.init else "")
    out = run_dir(run_id)
    model, meta = load_model(args.model, args.device, init_only=args.init)
    model.config.block_size = max(model.config.block_size, L)

    sizes = {"reference": args.n_ref, "calibration": args.n_cal,
             "evaluation": args.n_eval}
    splits = build_splits(L, seed=args.split_seed, sizes=sizes)
    np.savez(out / "sequence_ids.npz", **splits)
    save_config(out, meta, {
        "context_length": L, "splits": {k: len(v) for k, v in splits.items()},
        "split_seed": args.split_seed, "batch_size": args.batch_size,
        "dataset": str(Path("nanoGPT/data/openwebtext/val.bin")),
        "tokenizer": "gpt2-bpe", "dtype_forward": "float32",
        "dtype_accumulation": "float64",
        "run_command": " ".join(sys.argv),
    })
    (out / "run_command.txt").write_text(" ".join(sys.argv) + "\n")

    print(f"[{run_id}] Phase A: reference pass ({len(splits['reference'])} seqs)")
    ref = reference_pass(model, splits["reference"], L, args.batch_size, args.device)
    torch.save(ref, out / "ref_stats.pt")

    print(f"[{run_id}] Phase B: evaluation pass ({len(splits['evaluation'])} seqs)")
    keys = evaluation_pass(model, ref, splits["evaluation"], L,
                           args.batch_size, args.device, out)
    update_summary(out, "extract", {"saved_arrays": keys, "run_id": run_id})
    print(f"[{run_id}] done -> {out}")


if __name__ == "__main__":
    main()
