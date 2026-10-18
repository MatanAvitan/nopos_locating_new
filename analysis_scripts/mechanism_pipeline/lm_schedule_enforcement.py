"""P7: enforce the trained alpha schedule on lm6bos at unseen positions.

P6 showed that past the trained horizon (i >= 128) the anchor head's BOS
share leaves its trained 1/i law, and frozen readouts saturate while an
oracle probe still reads position. Here the trained law is surgically
re-imposed at inference: for a patched head, at positions i >= 128 the BOS
attention weight is overwritten with alpha*(i) = exp(c) * i**gamma ((c,
gamma) fit per head on the clean calibration mean curve over positions
16..64) and the non-BOS weights are rescaled so the row sums to 1. The
patched output propagates through all subsequent layers.

Variants: clean (no patch), V1 anchor head only (layer-2 head, code index
block1/head0), V2 all heads all layers, V3 control (anchor head, constant
alpha* = alpha(127)).

Readouts fit on the clean calibration split: frozen per-layer ridge probes
(positions 1..127 only) and LM cross-entropy per bin.

Usage:
  python lm_schedule_enforcement.py --lengths 512,1024,2048
Outputs results/mechanism/lm6bos_enforce_L{L}/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, build_splits, git_commit, head_weights,
                    run_dir, sha256_file, update_summary)
from lm_extrapolation import (ANCHOR_BLOCK, ANCHOR_HEAD, FIT_HI, FIT_LO,
                              TRAIN_L, binned, bins_for, targets_from_offsets)
from lm_mechanism import (Ridge, attn_scores_weights, batch_from_offsets,
                          load_lm, ov_images)
from stats import loglog_slope


@torch.no_grad()
def forward_patched(model, idx, patch=None):
    """lm_mechanism.forward_blocks with an optional attention-weight patch.

    patch: dict {layer_index: {head_index: alpha_star [L] tensor}} applied
    at positions i >= TRAIN_L; non-BOS weights rescaled to keep row sums 1.
    Returns (per-block resid_out list, block-ANCHOR alpha [B,H,L]).
    """
    x = model.transformer.wte(idx)
    resids, alpha_anchor = [], None
    L = idx.shape[1]
    for l, blk in enumerate(model.transformer.h):
        ln1 = F.layer_norm(x, blk.ln_1.weight.shape, blk.ln_1.weight,
                           blk.ln_1.bias, 1e-5)
        hw = head_weights(blk.attn)
        _, weights = attn_scores_weights(ln1, hw)
        if patch and l in patch:
            w = weights.clone()
            for h, a_star in patch[l].items():
                a_old = w[:, h, TRAIN_L:, 0]
                a_new = a_star[TRAIN_L:L][None, :].to(a_old.dtype)
                scale = (1 - a_new) / (1 - a_old).clamp_min(1e-8)
                w[:, h, TRAIN_L:, 1:] *= scale[..., None]
                w[:, h, TRAIN_L:, 0] = a_new
            weights = w
        if l == ANCHOR_BLOCK:
            alpha_anchor = weights[:, :, :, 0].clone()
        y = ov_images(ln1, hw)
        o = torch.einsum("bhts,bhsd->bhtd", weights, y).sum(1)
        bo = blk.attn.c_proj.bias
        per_head_bias = torch.einsum(
            "he,hde->hd", hw.b_v.float(), hw.W_o.float()).sum(0)
        if bo is not None:
            per_head_bias = per_head_bias + bo.float()
        o = o + per_head_bias[None, None, :]
        x_attn = x + o
        ln2 = F.layer_norm(x_attn, blk.ln_2.weight.shape, blk.ln_2.weight,
                           blk.ln_2.bias, 1e-5)
        x = x_attn + blk.mlp(ln2)
        resids.append(x)
    return resids, alpha_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nanoGPT/out-lm-6layer-fulltrain-bos/ckpt.pt")
    ap.add_argument("--lengths", default="512,1024,2048")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-cal", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=256)
    args = ap.parse_args()
    device = args.device
    ckpt_path = str(REPO_ROOT / args.ckpt)
    model, cfg, ck = load_lm(ckpt_path, device)
    n_layer, H, d = cfg.n_layer, cfg.n_head, cfg.n_embd

    for L in [int(x) for x in args.lengths.split(",")]:
        print(f"===== L={L} =====")
        out = run_dir(f"lm6bos_enforce_L{L}")
        bs = min(32, max(2, 8192 // L))
        splits = build_splits(L, seed=1234, sizes={
            "reference": 8, "calibration": args.n_cal, "evaluation": args.n_eval})
        np.savez(out / "sequence_ids.npz", **splits)
        (out / "config.json").write_text(json.dumps({
            "model": "lm6bos", "checkpoint_path": ckpt_path,
            "checkpoint_hash": sha256_file(ckpt_path),
            "context_length": L, "trained_context_length": TRAIN_L,
            "splits": {k: len(v) for k, v in splits.items()},
            "batch_size": bs, "git_commit": git_commit(),
            "run_command": " ".join(sys.argv)}, indent=2))

        def iter_split(name):
            offs = splits[name]
            for s in range(0, len(offs), bs):
                o = offs[s:s + bs]
                yield o, batch_from_offsets(o, L, device, use_bos=True)

        # ---- clean calibration: per-head alpha curves + frozen probes ----
        print("clean calibration pass...")
        frozen = [Ridge(d, device) for _ in range(n_layer)]
        tgt = torch.arange(L, dtype=torch.float32, device=device)
        alpha_sum = torch.zeros(n_layer, H, L, device=device,
                                dtype=torch.float64)
        n_cal = 0
        for _, idx in iter_split("calibration"):
            x = model.transformer.wte(idx)
            B = idx.shape[0]
            for l, blk in enumerate(model.transformer.h):
                ln1 = F.layer_norm(x, blk.ln_1.weight.shape, blk.ln_1.weight,
                                   blk.ln_1.bias, 1e-5)
                hw = head_weights(blk.attn)
                _, w = attn_scores_weights(ln1, hw)
                alpha_sum[l] += w[:, :, :, 0].double().sum(0)
                y = ov_images(ln1, hw)
                o = torch.einsum("bhts,bhsd->bhtd", w, y).sum(1)
                per_head_bias = torch.einsum(
                    "he,hde->hd", hw.b_v.float(), hw.W_o.float()).sum(0)
                if blk.attn.c_proj.bias is not None:
                    per_head_bias = per_head_bias + blk.attn.c_proj.bias.float()
                x_attn = x + o + per_head_bias[None, None, :]
                ln2 = F.layer_norm(x_attn, blk.ln_2.weight.shape,
                                   blk.ln_2.weight, blk.ln_2.bias, 1e-5)
                x = x_attn + blk.mlp(ln2)
                frozen[l].add(x[:, 1:TRAIN_L], tgt[1:TRAIN_L].expand(B, TRAIN_L - 1))
            n_cal += B
        for l in range(n_layer):
            frozen[l].solve()
        alpha_cal = (alpha_sum / n_cal).cpu().numpy()      # [n_layer,H,L]
        np.save(out / "alpha_cal_per_head.npy", alpha_cal)

        # ---- per-head trained-law fits and alpha_star schedules ----------
        i_all = np.arange(L).astype(np.float64)
        i_fit = np.arange(FIT_LO, FIT_HI + 1)
        a_star = np.zeros((n_layer, H, L))
        fits = {}
        for l in range(n_layer):
            for h in range(H):
                g, c = loglog_slope(i_fit, alpha_cal[l, h, FIT_LO:FIT_HI + 1])
                if not np.isfinite(g):
                    g, c = 0.0, np.log(max(alpha_cal[l, h, FIT_LO:FIT_HI + 1].mean(), 1e-12))
                ext = np.exp(c) * np.maximum(i_all, 1) ** g
                a_star[l, h] = np.clip(ext, 0.0, 0.9)
                fits[f"L{l}h{h}"] = {"gamma": float(g), "c": float(c)}
        anchor_fit = fits[f"L{ANCHOR_BLOCK}h{ANCHOR_HEAD}"]
        print(f"anchor trained-law fit: gamma={anchor_fit['gamma']:.4f}")
        t_star = {(l, h): torch.from_numpy(a_star[l, h]).to(device)
                  for l in range(n_layer) for h in range(H)}
        const_anchor = torch.full(
            (L,), float(alpha_cal[ANCHOR_BLOCK, ANCHOR_HEAD, TRAIN_L - 1]),
            device=device, dtype=torch.float64)
        variants = {
            "clean": None,
            "V1_anchor": {ANCHOR_BLOCK: {ANCHOR_HEAD: t_star[(ANCHOR_BLOCK, ANCHOR_HEAD)]}},
            "V2_all": {l: {h: t_star[(l, h)] for h in range(H)}
                       for l in range(n_layer)},
            "V3_const_control": {ANCHOR_BLOCK: {ANCHOR_HEAD: const_anchor}},
        }

        # ---- evaluation: each variant through the patched forward --------
        summary = {"anchor_fit": anchor_fit,
                   "bins": {k: list(v) for k, v in bins_for(L).items()},
                   "variants": {}}
        for vname, patch in variants.items():
            print(f"evaluation pass: {vname}")
            preds = {l: [] for l in range(n_layer)}
            alpha_acc = []
            ce_sum = torch.zeros(L, device=device, dtype=torch.float64)
            n_ev = 0
            for offs, idx in iter_split("evaluation"):
                resids, al = forward_patched(model, idx, patch)
                for l in range(n_layer):
                    preds[l].append(
                        frozen[l].predict(resids[l]).float().cpu().numpy())
                alpha_acc.append(
                    al[:, ANCHOR_HEAD].float().cpu().numpy())
                xf = model.transformer.ln_f(resids[-1])
                y = targets_from_offsets(offs, L, device)
                for t0 in range(0, L, 256):
                    logits = model.lm_head(xf[:, t0:t0 + 256])
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        y[:, t0:t0 + 256].reshape(-1), reduction="none")
                    ce_sum[t0:t0 + 256] += ce.reshape(len(idx), -1).double().sum(0)
                n_ev += len(idx)
                del resids
            ce_pos = (ce_sum / n_ev).cpu().numpy()
            np.save(out / f"{vname}__ce_per_position.npy", ce_pos)
            np.save(out / f"{vname}__alpha_anchor.npy",
                    np.concatenate(alpha_acc, 0))
            vsum = {"probes": {}, "lm_ce_per_bin": {}}
            for l in (1, 4):
                arr = np.concatenate(preds[l], 0)
                np.save(out / f"{vname}__L{l}_frozen_pred.npy", arr)
                vsum["probes"][f"L{l}_frozen"] = binned(arr, L)
            for name, (a, b) in bins_for(L).items():
                vsum["lm_ce_per_bin"][name] = float(np.nanmean(ce_pos[a:b + 1]))
            summary["variants"][vname] = vsum
            print(json.dumps({"L1_frozen": vsum["probes"]["L1_frozen"],
                              "lm_ce": vsum["lm_ce_per_bin"]}, indent=2))
        update_summary(out, "lm_schedule_enforcement", summary)


if __name__ == "__main__":
    main()
