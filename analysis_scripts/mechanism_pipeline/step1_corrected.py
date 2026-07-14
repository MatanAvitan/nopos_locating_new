"""Corrected Step-1 signal/error algebra (follow-up to the failed stylized test).

The manuscript's stylized Step 1 projects the post-LN state onto the raw BOS
input x0^(1). Empirically that projection is content-dominated: the actual
Layer-1 update is o1 = sum_j alpha_ij B_OV^(1) x_j (+ b_attn) added to the
token-embedding residual, so the BOS reference that actually enters the
residual stream is the OV image y0 = B_OV^(1) x0, not x0 itself.

This script tests the mechanism-aligned decomposition exactly:
  <post1_i, u> = <emb_i, u> + <b_attn, u> + alpha_i0 ||y0||
                 + sum_{j>=1} alpha_ij <y_j, u>,   u := y0/||y0||
and after LayerNorm:
  S2_i := <h1bar_i, u> = (<post1_i, u> - mu_i * sum(u)) / sigma_i.

Reported: position-wise curves and log-log slopes of
  - the BOS signal term alpha_i0 ||y0|| / sigma_i  (predicted ~ 1/i under
    uniform attention with roughly constant sigma),
  - the content term (everything else),
  - their ratio (empirical SNR), plus R^2 of the exact identity (sanity).

Usage: python step1_corrected.py --run-id attn2_1h_L1024 --model attn2_1h --context-length 1024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS_ROOT, attn_bias_vector, forward_capture,
                    head_weights, iter_batches, load_model, ov_images,
                    update_summary)
from stats import fit_range, loglog_slope, slope_ci


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = RESULTS_ROOT / args.run_id
    L = args.context_length
    device = args.device
    model, meta = load_model(args.model, device)
    splits = np.load(out / "sequence_ids.npz")
    hw1 = head_weights(model.block1.attn)
    b_attn1 = attn_bias_vector(hw1).double()

    S2_all, sig_all, content_all, emb_all = [], [], [], []
    for s, idx in iter_batches(splits["evaluation"], L, args.batch_size, device):
        cap = forward_capture(model, idx)
        x1 = cap["x1"]
        y1 = ov_images(x1, hw1).double().sum(1)      # [B,T,d] summed heads
        # per-head BOS image summed over heads with per-head alpha:
        a1 = cap["attn1_w"].double()                 # [B,H,T,T]
        y1h = ov_images(x1, hw1).double()            # [B,H,T,d]
        y0h = y1h[:, :, 0]                           # [B,H,d]
        # u: unit vector along the (head-summed, alpha-weighted at i... ) --
        # use the simple head-summed BOS image, fixed across sequences:
        y0 = y0h[0].sum(0)                           # [d] fixed (BOS-only prefix)
        u = y0 / y0.norm()
        h1bar = cap["h1bar"].double()
        post1 = cap["post1"].double()
        mu = post1.mean(-1)
        eps = 1e-5
        sigma = (post1.var(-1, unbiased=False) + eps).sqrt()
        gamma = model.block1.ln_2.weight.double()
        beta = model.block1.ln_2.bias.double() \
            if model.block1.ln_2.bias is not None else torch.zeros_like(gamma)
        gu = gamma * u
        # exact identity: S2 = <h1bar,u> = (<post1,gu> - mu*sum(gu))/sigma + <beta,u>
        S2 = h1bar @ u                               # [B,T]
        alpha_bos1 = a1[:, :, :, 0]                  # [B,H,T]
        sig = torch.einsum("bht,bhd,d->bt", alpha_bos1, y0h, gu) / sigma
        nonbos_part = torch.einsum("bhts,bhsd,d->bt",
                                   a1[..., 1:], y1h[:, :, 1:], gu)
        content = ((nonbos_part + cap["emb"].double() @ gu
                    + (b_attn1.to(device) @ gu) - mu * gu.sum())
                   / sigma) + (beta @ u)
        S2_all.append(S2.cpu().numpy())
        sig_all.append(sig.cpu().numpy())
        content_all.append(content.cpu().numpy())
        emb_all.append(((cap["emb"].double() @ u) / sigma).cpu().numpy())
        del cap
    S2 = np.concatenate(S2_all, 0)
    sig = np.concatenate(sig_all, 0)
    content = np.concatenate(content_all, 0)

    lo, hi = fit_range(L)
    pos_fit = np.arange(lo, hi + 1)
    ident_resid = S2 - (sig + content)
    res = {
        "identity_max_abs_err": float(np.nanmax(np.abs(ident_resid))),
        "S2_mean_at": {str(k): float(np.nanmean(S2[:, k]))
                       for k in [1, 4, 16, 64, 256, L - 1] if k < L},
        "signal_mean_at": {str(k): float(np.nanmean(sig[:, k]))
                           for k in [1, 4, 16, 64, 256, L - 1] if k < L},
        "content_std_at": {str(k): float(np.nanstd(content[:, k]))
                           for k in [1, 4, 16, 64, 256, L - 1] if k < L},
        "signal_mean_slope": slope_ci(pos_fit, sig[:, pos_fit]),
        "content_rms_slope": loglog_slope(
            pos_fit, np.sqrt(np.nanmean(
                (content - np.nanmean(content, 0, keepdims=True))[:, pos_fit] ** 2, 0)))[0],
        "snr_slope": slope_ci(
            pos_fit,
            np.abs(sig[:, pos_fit]) /
            np.nanstd(content[:, pos_fit], 0, keepdims=True)),
        # variance explained of S2 across sequences by the signal term alone
        # (should be ~0; position info is in the mean, content is noise)
        "corr_S2_signal_flat": float(np.corrcoef(
            S2[:, 1:].ravel(), sig[:, 1:].ravel())[0, 1]),
    }
    np.save(out / "step1corr__S2.npy", S2)
    np.save(out / "step1corr__signal.npy", sig)
    np.save(out / "step1corr__content.npy", content)
    update_summary(out, "step1_corrected", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
