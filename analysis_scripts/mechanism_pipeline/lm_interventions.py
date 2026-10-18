"""Causal intervention suite inside the trained NoPE LM (exps.md P2).

Applies the toy-model interventions at the block-1 attention output of
nanoGPT/out-lm-6layer-fulltrain-ddp (where the position probe jumps
0.634 -> 0.975 and the anchor heads live), using the per-head decomposition
o_h = alpha w_B,h + (1-alpha) w_NB,h + eta_h with reference anchors, then
lets the remaining blocks run and quantifies three ways:

  frozen probes  : ridge probes trained on CLEAN calibration activations at
                   every block output, evaluated zero-shot on intervened
                   evaluation activations (binned MAE / relMAE downstream)
  recal probes   : the same probes retrained on INTERVENED calibration
                   activations (information present vs format)
  LM loss        : next-token cross-entropy under the intervention, by bin
                   (criterion (f): functional relevance)

Interventions: E0 rebuild sanity, C1 eta->0 (+ anchor-head-only variant),
C5 mixture removed, C6 alpha<-refmean, E1 eta const-norm, E2 eta shuffled
across sequences, E3 eta<-refmean, C7 targeted (block-0 BOS value removed,
block-1 attention weights held at realized values).

C7 replacement controls (pre-registered in exps.md C7; all hold the block-1
attention weights at their realized values, so only the value/content path
moves, and all edit the same block-0 BOS OV image y_BOS,h that C7 deletes):

  C7_ctrl_random   : y_BOS,h <- fixed random direction, ||.|| matched to
                     ||y_BOS,h||   (norm-matched: is C7 just vector deletion?)
  C7_ctrl_nonbos   : y_BOS,h <- mean nonBOS OV direction, ||.|| matched to
                     ||y_BOS,h||   (removes BOS-specific direction only)
  C7_ctrl_normonly : y_BOS,h direction kept, rescaled to ||w_NB,h||
                     (removes the BOS magnitude advantage only)
  C7_ctrl_alpharef : y_BOS,h kept; its block-0 weight alpha_{b,h,t,0} is
                     replaced by the reference-split position-wise mean
                     alpha_bar_{h,t}  (keeps the deterministic position
                     schedule, removes the input-dependent part; the C7
                     analogue of E3)
  C7_ctrl_alphaconst: y_BOS,h kept; its block-0 weight is replaced by one
                     constant per head, the mean of alpha_bar_{h,t} over
                     t >= 1 (position 0 keeps alpha_bar_{h,0}); removes the
                     position schedule while keeping the vector

These stand to C7 as E1/E2/E3 stand to C1: they separate "the BOS reference
direction is load-bearing" from "a large vector was removed".

C7_ctrl_random is a single draw of a random direction, so its effect size is
partly luck: how much that draw overlaps the probe's read-out direction.
--c7-random-seeds runs several independent draws in one pass (extra arms named
C7_ctrl_random_seed<k>) so the control can be reported as a spread instead of
one number. The first listed seed keeps the plain name C7_ctrl_random, so the
default (--c7-random-seeds 0) reproduces the canonical single-draw run.

Usage:
  python lm_interventions.py --run-id lm6_L128 --context-length 128

  # seed sweep, canonical artifacts untouched (--out-id redirects writes)
  python lm_interventions.py --run-id lm6_L128 --out-id c7_seed_sweep/lm6_L128 \
      --c7-random-seeds 0,1,2,3,4,5,6,7 --no-save-preds
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, RESULTS_ROOT, attn_scores_weights, build_splits,
                    head_weights, ov_images, position_bins, update_summary)
from lm_mechanism import Ridge, batch_from_offsets, load_lm

TARGET_BLOCK = 1
ANCHOR_RHO, ANCHOR_CV = -0.95, 0.25


def block_states(model):
    return model.transformer.h if hasattr(model, "transformer") else model.blocks


@torch.no_grad()
def attn_parts(blk, x):
    """One block's attention decomposition from residual input x."""
    ln1 = F.layer_norm(x, blk.ln_1.weight.shape, blk.ln_1.weight,
                       blk.ln_1.bias, 1e-5)
    hw = head_weights(blk.attn)
    scores, weights = attn_scores_weights(ln1, hw)
    y = ov_images(ln1, hw)
    bo = blk.attn.c_proj.bias
    bias = torch.einsum("he,hde->hd", hw.b_v.float(), hw.W_o.float()).sum(0)
    if bo is not None:
        bias = bias + bo.float()
    return ln1, weights, y, bias


@torch.no_grad()
def rest_of_block(blk, x, o):
    x_attn = x + o
    if getattr(blk, "use_ln2", True):
        ln2 = F.layer_norm(x_attn, blk.ln_2.weight.shape, blk.ln_2.weight,
                           blk.ln_2.bias, 1e-5)
        return x_attn + blk.mlp(ln2)
    return x_attn + blk.mlp(x_attn)


@torch.no_grad()
def clean_block(blk, x):
    _, w, y, bias = attn_parts(blk, x)
    o = torch.einsum("bhts,bhsd->bhtd", w, y).sum(1) + bias[None, None, :]
    return rest_of_block(blk, x, o), w, y, bias


@torch.no_grad()
def lm_loss_by_pos(model, x_final, idx):
    x = model.transformer.ln_f(x_final)
    logits = model.lm_head(x)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        idx[:, 1:].reshape(-1), reduction="none"
    ).reshape(idx.shape[0], -1)                     # [B, L-1], loss at query i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nanoGPT/out-lm-6layer-fulltrain-ddp/ckpt.pt")
    ap.add_argument("--run-id", default="lm6_L128")
    ap.add_argument("--context-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-ref", type=int, default=640)
    ap.add_argument("--n-cal", type=int, default=320)
    ap.add_argument("--n-eval", type=int, default=1600)
    ap.add_argument("--c7-random-seeds", default="0",
                    help="comma-separated seeds for the C7 random-direction "
                         "control; the first keeps the name C7_ctrl_random, "
                         "the rest become C7_ctrl_random_seed<k>")
    ap.add_argument("--out-id", default=None,
                    help="write artifacts under results/mechanism/<out-id> "
                         "instead of <run-id> (anchor-head metadata is still "
                         "read from <run-id>/summary.json)")
    ap.add_argument("--no-save-preds", action="store_true",
                    help="skip the large per-sequence prediction/loss npz")
    ap.add_argument("--geometry-arms", action="store_true",
                    help="add the toy pipeline's layer-2 geometry arms: ablate "
                         "the aggregate dw direction / the per-head dw span / "
                         "5 random unit directions (centered on the attention "
                         "bias, as in interventions.py), collapse the BOS "
                         "endpoint onto w_nonBOS, permute the BOS weight curve "
                         "across positions 1..L-1, and replace the realized "
                         "nonBOS aggregate by the uniform prefix mean")
    ap.add_argument("--geometry-seed", type=int, default=0)
    args = ap.parse_args()
    L, device = args.context_length, args.device
    src = RESULTS_ROOT / args.run_id
    out = RESULTS_ROOT / (args.out_id or args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    c7_seeds = [int(s) for s in args.c7_random_seeds.split(",") if s.strip()]
    model, cfg, _ = load_lm(str(REPO_ROOT / args.ckpt), device)
    blocks = block_states(model)
    n_layer, H, d = cfg.n_layer, cfg.n_head, cfg.n_embd
    bins = position_bins(L)
    splits = build_splits(L, seed=1234, sizes={
        "reference": args.n_ref, "calibration": args.n_cal,
        "evaluation": args.n_eval})

    def iter_split(name):
        offs = splits[name]
        for s in range(0, len(offs), args.batch_size):
            yield batch_from_offsets(offs[s:s + args.batch_size], L, device,
                                     use_bos=True)

    # anchor heads from the existing lm_mechanism summary (same criterion)
    heads_meta = json.loads((src / "summary.json").read_text())[
        "lm_mechanism"]["per_layer"][f"block{TARGET_BLOCK}"]["heads"]
    anchor = torch.tensor([
        heads_meta[f"head{h}"]["alpha_per_seq_spearman_median"] < ANCHOR_RHO
        and heads_meta[f"head{h}"]["cbar_cv_median"] < ANCHOR_CV
        for h in range(H)], device=device)
    print("anchor heads:", anchor.nonzero().flatten().tolist())

    tgt_blk = blocks[TARGET_BLOCK]

    # C7 replacements for the block-0 BOS OV image, filled after the reference
    # pass (they depend on reference statistics); empty => removal only.
    c7_repl, c7_stats, ref_mode = {}, {}, [True]
    c7_alpha = {}          # name -> [H,L] replacement for alpha_{b,h,t,0}

    @torch.no_grad()
    def parts_at_target(idx):
        """Clean forward to the target block; return its decomposition parts
        plus what is needed to continue (x_in) and the C7 variants of x_in."""
        x = model.transformer.wte(idx) if hasattr(model, "transformer") \
            else model.wte(idx)
        x_c7v = {}
        for l in range(TARGET_BLOCK):
            blk = blocks[l]
            _, w, y, bias = attn_parts(blk, x)
            o = torch.einsum("bhts,bhsd->bhtd", w, y).sum(1) + bias[None, None, :]
            x_next = rest_of_block(blk, x, o)
            if l == TARGET_BLOCK - 1:              # C7 acts on the BOS value
                wb, y_bos = w[:, :, :, 0], y[:, :, 0]     # [B,H,T], [B,H,d]
                if ref_mode[0]:
                    # position 0 is BOS in every sequence, so y_bos is constant
                    # across the batch; the nonBOS mean is accumulated.
                    c7_stats.setdefault("y_bos", y_bos[0].double())
                    c7_stats["y_nb_sum"] = c7_stats.get(
                        "y_nb_sum", 0) + y[:, :, 1:].double().sum((0, 2))
                    c7_stats["y_nb_n"] = c7_stats.get("y_nb_n", 0) \
                        + idx.shape[0] * (L - 1)
                    c7_stats["alpha0_sum"] = c7_stats.get(
                        "alpha0_sum", 0) + wb.double().sum(0)      # [H,T]
                    c7_stats["alpha0_n"] = c7_stats.get("alpha0_n", 0) \
                        + idx.shape[0]
                x_c7v["C7_targeted"] = rest_of_block(
                    blk, x, o + torch.einsum("bht,bhd->btd", wb, -y_bos))
                # split the removal by position: keep position 0 intact and
                # remove at t >= 1, or remove at position 0 only
                m_keep0 = torch.ones(wb.shape[-1], device=wb.device,
                                     dtype=wb.dtype)
                m_keep0[0] = 0
                x_c7v["C7_targeted_keep0"] = rest_of_block(
                    blk, x, o + torch.einsum(
                        "bht,bhd->btd", wb * m_keep0[None, None, :], -y_bos))
                x_c7v["C7_targeted_only0"] = rest_of_block(
                    blk, x, o + torch.einsum(
                        "bht,bhd->btd", wb * (1 - m_keep0)[None, None, :],
                        -y_bos))
                for nm, r in c7_repl.items():
                    x_c7v[nm] = rest_of_block(
                        blk, x, o + torch.einsum(
                            "bht,bhd->btd", wb,
                            r[None].to(y_bos.dtype) - y_bos))
                # alpha-edited arms: same y_BOS, edited block-0 BOS weight
                for nm, a_rep in c7_alpha.items():
                    x_c7v[nm] = rest_of_block(
                        blk, x, o + torch.einsum(
                            "bht,bhd->btd",
                            a_rep[None].to(wb.dtype) - wb, y_bos))
            x = x_next
        _, w1, y1, bias1 = attn_parts(tgt_blk, x)
        alpha = w1[..., 0].double()
        m = (1 - alpha).clamp_min(1e-30)
        q = w1[..., 1:].float() / m.float()[..., None]
        g = torch.einsum("bhts,bhsd->bhtd", q, y1[:, :, 1:]).double()
        return {"x_in": x, "x_c7v": x_c7v, "w1": w1, "alpha": alpha, "m": m,
                "g": g, "bias1": bias1.double(), "y1": y1}

    # ---- reference pass: anchors + position-wise eta stats + alpha mean ---
    print("reference pass...")
    sum_y = torch.zeros(H, d, device=device, dtype=torch.float64)
    n_tok = 0
    w_b = None
    alpha_sum = torch.zeros(H, L, device=device, dtype=torch.float64)
    n_seq = 0
    for idx in iter_split("reference"):
        p = parts_at_target(idx)
        ln1, _, y1, _ = attn_parts(tgt_blk, p["x_in"])
        sum_y += y1[:, :, 1:].double().sum((0, 2))
        n_tok += idx.shape[0] * (L - 1)
        if w_b is None:
            hw = head_weights(tgt_blk.attn)
            v0 = torch.einsum("hed,d->he", hw.W_v.double(), ln1[0, 0].double())
            w_b = torch.einsum("hde,he->hd", hw.W_o.double(), v0)  # [H,d]
        alpha_sum += p["alpha"].sum(0)
        n_seq += idx.shape[0]
    w_nb = sum_y / n_tok                                          # [H,d]
    alpha_ref = alpha_sum / n_seq                                 # [H,L]

    def eta_of(p):
        return p["m"][..., None] * (p["g"] - w_nb[None, :, None, :])

    print("reference pass 2 (eta stats)...")
    norm_sum = torch.zeros(H, L, device=device, dtype=torch.float64)
    eta_sum = torch.zeros(H, L, d, device=device, dtype=torch.float64)
    for idx in iter_split("reference"):
        eta = eta_of(parts_at_target(idx))
        norm_sum += eta.norm(dim=-1).sum(0)
        eta_sum += eta.sum(0)
    eta_norm_ref = norm_sum / n_seq
    eta_mean_ref = eta_sum / n_seq

    # ---- C7 replacement controls (exps.md C7) ------------------------------
    ref_mode[0] = False
    y_bos0 = c7_stats["y_bos"]                                    # [H,d]
    w_nb0 = c7_stats["y_nb_sum"] / c7_stats["y_nb_n"]             # [H,d]
    nrm_b = y_bos0.norm(dim=-1, keepdim=True)
    nrm_nb = w_nb0.norm(dim=-1, keepdim=True)
    rnd_names = {}
    for k, s in enumerate(c7_seeds):
        gen = torch.Generator().manual_seed(s)
        rnd = torch.randn(H, d, generator=gen).to(device).double()
        nm = "C7_ctrl_random" if k == 0 else f"C7_ctrl_random_seed{s}"
        rnd_names[nm] = s
        c7_repl[nm] = rnd / rnd.norm(dim=-1, keepdim=True) * nrm_b
    c7_repl["C7_ctrl_nonbos"] = w_nb0 / nrm_nb * nrm_b
    c7_repl["C7_ctrl_normonly"] = y_bos0 / nrm_b * nrm_nb
    alpha0_ref = c7_stats["alpha0_sum"] / c7_stats["alpha0_n"]    # [H,L]
    alpha0_const = alpha0_ref.clone()
    alpha0_const[:, 1:] = alpha0_ref[:, 1:].mean(1, keepdim=True)
    c7_alpha["C7_ctrl_alpharef"] = alpha0_ref
    c7_alpha["C7_ctrl_alphaconst"] = alpha0_const
    print("block-0 BOS weight, reference mean over sequences, per head at "
          f"t=1/8/64/127: "
          f"{[[round(float(alpha0_ref[h, t]), 4) for t in (1, 8, 64, 127)] for h in range(H)]}")
    print("block-0 BOS/nonBOS OV norms per head: "
          f"{nrm_b.flatten().tolist()} / {nrm_nb.flatten().tolist()}")

    def o_from(p, alpha, eta):
        base = (alpha[..., None] * w_b[None, :, None, :]
                + (1 - alpha[..., None]) * w_nb[None, :, None, :])
        return ((base + eta).sum(1) + p["bias1"][None, None, :]).float()

    # ---- layer-2 geometry arms (mirror interventions.py) -------------------
    dw = w_nb - w_b                                               # [H,d]
    dw_sum = dw.sum(0)
    u_dwsum = (dw_sum / dw_sum.norm()).unsqueeze(1)               # [d,1]
    Q_dw = torch.linalg.qr(dw.T)[0]                               # [d,H]
    g_rng = torch.Generator(device="cpu").manual_seed(args.geometry_seed)
    rand_us = []
    for k in range(5):
        v = torch.randn(d, generator=g_rng).double().to(device)
        rand_us.append((v / v.norm()).unsqueeze(1))
    print("geometry arms: |dw_sum|=%.3f, per-head |dw|=%s" % (
        float(dw_sum.norm()), [round(float(x), 3) for x in dw.norm(dim=-1)]))

    def proj_out(o, Q):
        return o - (o @ Q) @ Q.T

    def geometry_outputs(p):
        a = p["alpha"]
        bias = p["bias1"][None, None, :]
        o_clean = torch.einsum("bhts,bhsd->bhtd", p["w1"], p["y1"]).sum(1).double() \
            + bias
        oc = o_clean - bias                                       # centered
        gsafe = torch.where((p["m"] > 1e-6)[..., None], p["g"],
                            w_nb[None, :, None, :])
        out = {
            "C3_ablate_dw_sum": (proj_out(oc, u_dwsum) + bias).float(),
            "C3_ablate_dw_span": (proj_out(oc, Q_dw) + bias).float(),
            "C4_bos_to_nonbos": ((a[..., None] * w_nb[None, :, None, :]
                                  + (1 - a[..., None]) * gsafe).sum(1)
                                 + bias).float(),
        }
        for k, u in enumerate(rand_us):
            out[f"C3_ablate_random1d_{k}"] = (proj_out(oc, u) + bias).float()
        T = a.shape[-1]
        perm = torch.randperm(T - 1, generator=g_rng).to(a.device)
        a_perm = a.clone()
        a_perm[..., 1:] = a[..., 1:][..., perm]
        a_perm = a_perm.clamp(1e-6, 1 - 1e-6)
        out["C6_alpha_permuted"] = ((a_perm[..., None] * w_b[None, :, None, :]
                                     + (1 - a_perm[..., None]) * gsafe).sum(1)
                                    + bias).float()
        y1nb = p["y1"][:, :, 1:].double()                         # [B,H,T-1,d]
        counts = torch.arange(1, T, device=a.device).double()[None, None, :, None]
        g_unif = torch.zeros_like(p["g"])
        g_unif[:, :, 1:] = y1nb.cumsum(2) / counts                # mean of y_1..y_i
        out["C6_uniform_nonbos_attn"] = ((a[..., None] * w_b[None, :, None, :]
                                          + (1 - a[..., None]) * g_unif).sum(1)
                                         + bias).float()
        return out

    GEOM_IVS = (["C3_ablate_dw_sum", "C3_ablate_dw_span"]
                + [f"C3_ablate_random1d_{k}" for k in range(5)]
                + ["C4_bos_to_nonbos", "C6_alpha_permuted",
                   "C6_uniform_nonbos_attn"]) if args.geometry_arms else []

    def iv_outputs(p):
        eta = eta_of(p)
        a = p["alpha"]
        ar = alpha_ref[None].expand_as(a)
        n = eta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        eta0 = torch.zeros_like(eta)
        out = {
            "E0_rebuild": o_from(p, a, eta),
            "C1_eta_zero": o_from(p, a, eta0),
            "C1_eta_zero_anchorheads": o_from(
                p, a, torch.where(anchor[None, :, None, None], eta0, eta)),
            "C5_mixture_removed": (eta.sum(1) + p["bias1"][None, None, :]).float(),
            # realized g is garbage where the nonBOS mass vanishes (alpha
            # saturates at 1 in some LM heads); fall back to w_NB there
            # (eta = 0 at those positions anyway) and clamp like the toys.
            "C6_alpha_refmean": o_from(
                p, ar.clamp(1e-6, 1 - 1e-6),
                (1 - ar.clamp(1e-6, 1 - 1e-6))[..., None]
                * torch.where((p["m"] > 1e-6)[..., None],
                              p["g"], w_nb[None, :, None, :])
                - (1 - ar.clamp(1e-6, 1 - 1e-6))[..., None]
                * w_nb[None, :, None, :]),
            "E1_eta_constnorm": o_from(
                p, a, eta * (eta_norm_ref[None, :, :, None] / n)),
            "E2_eta_shuffled": o_from(p, a, eta.roll(1, dims=0)),
            "E3_eta_refmean": o_from(
                p, a, eta_mean_ref[None].expand_as(eta)),
        }
        if args.geometry_arms:
            out.update(geometry_outputs(p))
        return out

    def resid_chain(x_in, o):
        """Target block with attention output o, then remaining blocks.
        Returns list of resid_out for blocks TARGET_BLOCK..n_layer-1."""
        outs = []
        x = rest_of_block(tgt_blk, x_in, o)
        outs.append(x)
        for l in range(TARGET_BLOCK + 1, n_layer):
            x, _, _, _ = clean_block(blocks[l], x)
            outs.append(x)
        return outs

    def c7_chain(p, name):
        """Targeted C7 and its replacement controls: modified value stream
        into the target block, realized attention weights."""
        x_c7 = p["x_c7v"][name]
        _, _, y_mod, _ = attn_parts(tgt_blk, x_c7)
        o = torch.einsum("bhts,bhsd->bhtd", p["w1"], y_mod).sum(1) \
            + p["bias1"][None, None, :].float()
        return resid_chain(x_c7, o)

    C7_IVS = ["C7_targeted", "C7_targeted_keep0", "C7_targeted_only0",
              *rnd_names, "C7_ctrl_nonbos",
              "C7_ctrl_normonly", "C7_ctrl_alpharef", "C7_ctrl_alphaconst"]
    IVS = ["clean", "E0_rebuild", "C1_eta_zero", "C1_eta_zero_anchorheads",
           "C5_mixture_removed", "C6_alpha_refmean", "E1_eta_constnorm",
           "E2_eta_shuffled", "E3_eta_refmean", *GEOM_IVS, *C7_IVS]
    probe_blocks = list(range(TARGET_BLOCK, n_layer))

    def all_chains(idx):
        p = parts_at_target(idx)
        _, _, y1, _ = attn_parts(tgt_blk, p["x_in"])
        o_clean = torch.einsum("bhts,bhsd->bhtd", p["w1"], y1).sum(1) \
            + p["bias1"][None, None, :].float()
        chains = {"clean": resid_chain(p["x_in"], o_clean)}
        for name, o in iv_outputs(p).items():
            chains[name] = resid_chain(p["x_in"], o)
        for nm in C7_IVS:
            chains[nm] = c7_chain(p, nm)
        return chains

    # ---- calibration: frozen probes on clean; recal probes per iv ---------
    print("calibration pass (frozen + recal probes)...")
    frozen = {l: Ridge(d, device) for l in probe_blocks}
    recal = {(iv, l): Ridge(d, device) for iv in IVS for l in probe_blocks}
    tgt_row = torch.arange(L, dtype=torch.float32, device=device)
    for idx in iter_split("calibration"):
        y = tgt_row.expand(idx.shape[0], L)
        chains = all_chains(idx)
        for k, l in enumerate(probe_blocks):
            frozen[l].add(chains["clean"][k], y)
            for iv in IVS:
                recal[(iv, l)].add(chains[iv][k], y)
    for r in frozen.values():
        r.solve()
    for r in recal.values():
        r.solve()

    # ---- evaluation --------------------------------------------------------
    print("evaluation pass...")
    preds = {}
    losses = {iv: [] for iv in IVS}

    def push(k, t):
        preds.setdefault(k, []).append(t.detach().cpu().float().numpy())

    for idx in iter_split("evaluation"):
        chains = all_chains(idx)
        for iv in IVS:
            for k, l in enumerate(probe_blocks):
                push(f"{iv}__L{l}", frozen[l].predict(chains[iv][k]))
                push(f"{iv}__L{l}__recal", recal[(iv, l)].predict(chains[iv][k]))
            losses[iv].append(
                lm_loss_by_pos(model, chains[iv][-1], idx).cpu().float().numpy())

    arrs = {k: np.concatenate(v, 0) for k, v in preds.items()}
    loss_arrs = {iv: np.concatenate(v, 0) for iv, v in losses.items()}
    if not args.no_save_preds:
        np.savez(out / "lm_intervention_preds.npz", **arrs)
        np.savez(out / "lm_intervention_loss.npz", **loss_arrs)

    # ---- summarize ---------------------------------------------------------
    t = np.arange(L)[None, :]
    summary = {}
    for iv in IVS:
        row = {}
        for l in probe_blocks:
            for tag in ["", "__recal"]:
                pp = arrs[f"{iv}__L{l}{tag}"]
                err = np.abs(pp - t)
                tt = np.broadcast_to(t, pp.shape)
                rel = err[:, 1:] / t[:, 1:]
                ss = {"mae": float(err[:, 1:].mean()),
                      "relmae_mean": float(rel.mean()),
                      "relmae_median": float(np.median(rel)),
                      "r2": float(1 - ((pp - tt) ** 2).sum()
                                  / ((tt - tt.mean()) ** 2).sum())}
                for name, (a, b) in bins.items():
                    a = max(a, 1)
                    ss[f"mae_{name}"] = float(err[:, a:b + 1].mean())
                row[f"L{l}{'_recal' if tag else '_frozen'}"] = ss
        lo = loss_arrs[iv]
        row["lm_loss"] = {"overall": float(lo.mean())}
        for name, (a, b) in bins.items():
            b = min(b, lo.shape[1] - 1)
            row["lm_loss"][name] = float(lo[:, a:b + 1].mean())
        summary[iv] = row
    update_summary(out, "lm_interventions", summary)

    # ---- C7 random-control seed sweep -------------------------------------
    if len(c7_seeds) > 1:
        key = f"L{TARGET_BLOCK}_frozen"
        per_seed = {str(s): summary[nm][key]["relmae_median"]
                    for nm, s in rnd_names.items()}
        loss_seed = {str(s): summary[nm]["lm_loss"]["overall"]
                     for nm, s in rnd_names.items()}
        vals = np.array(list(per_seed.values()))
        lvals = np.array(list(loss_seed.values()))
        tgt_rel = summary["C7_targeted"][key]["relmae_median"]
        tgt_loss = summary["C7_targeted"]["lm_loss"]["overall"]
        sweep = {
            "run_id": args.run_id, "ckpt": args.ckpt,
            "context_length": L, "probe": f"frozen block-{TARGET_BLOCK}",
            "metric": "median over positions 1..L-1 of |pred-i|/i",
            "seeds": c7_seeds,
            "relmae_median_per_seed": per_seed,
            "spread": {"min": float(vals.min()), "median": float(np.median(vals)),
                       "max": float(vals.max()), "mean": float(vals.mean()),
                       "std": float(vals.std(ddof=1))},
            "lm_loss_per_seed": loss_seed,
            "lm_loss_spread": {
                "min": float(lvals.min()), "median": float(np.median(lvals)),
                "max": float(lvals.max()), "mean": float(lvals.mean()),
                "std": float(lvals.std(ddof=1))},
            "reference": {
                nm: summary[nm][key]["relmae_median"]
                for nm in ["clean", "C7_targeted", "C7_ctrl_nonbos",
                           "C7_ctrl_normonly"]},
            "reference_lm_loss": {
                nm: summary[nm]["lm_loss"]["overall"]
                for nm in ["clean", "C7_targeted", "C7_ctrl_nonbos",
                           "C7_ctrl_normonly"]},
            "n_seeds_above_targeted": int((vals > tgt_rel).sum()),
            "n_seeds_above_targeted_lm_loss": int((lvals > tgt_loss).sum()),
        }
        (out / "c7_random_seed_sweep.json").write_text(
            json.dumps(sweep, indent=2))
        print("C7 random seed sweep (frozen L1 median relMAE):")
        for s, v in per_seed.items():
            print(f"  seed {s:>3s}: {100 * v:6.1f}%")
        print(f"  min/med/max {100 * vals.min():.1f}/"
              f"{100 * np.median(vals):.1f}/{100 * vals.max():.1f}%  "
              f"targeted {100 * summary['C7_targeted'][key]['relmae_median']:.1f}%")

    # compact print: frozen block-1 & block-4 probes + LM loss
    for iv in IVS:
        r = summary[iv]
        l1 = r[f"L{TARGET_BLOCK}_frozen"]
        l4 = r.get("L4_frozen", l1)
        print(f"{iv:26s} L1 mae={l1['mae']:7.2f} e/m/l="
              f"{l1['mae_early']:.2f}/{l1['mae_middle']:.2f}/{l1['mae_late']:.2f}"
              f" | L4 mae={l4['mae']:7.2f} | loss={r['lm_loss']['overall']:.4f}")


if __name__ == "__main__":
    main()
