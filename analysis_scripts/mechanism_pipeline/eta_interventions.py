"""Early-code interventions on the nonBOS residual eta (empirical_test items 13-14).

eta_i := (1-alpha_i)(g_i - w_nonbos) per head is the sequence-varying part of
the Layer-2 nonBOS aggregate. C5 showed eta carries most of what early code
exists; these interventions separate its *magnitude* from its *content*:

  E1_eta_constnorm : rescale eta to the position-wise reference mean norm,
                     keeping its per-sequence direction. If the early code is
                     magnitude (||eta|| ~ 1/sqrt(i) prefix-noise decay), this
                     destroys it; if content (direction), it survives.
  E2_eta_shuffled  : swap eta across sequences at the same position (batch
                     roll). Keeps the position-wise marginal distribution of
                     eta exactly, destroys sequence-specific content.
  C7_remove_alpha_fixed : targeted C7 — remove the Layer-1 BOS component from
                     the values feeding Layer 2 while holding all Layer-2
                     attention weights (alpha) at their realized values.
                     Isolates the value-content pathway of the Step-1
                     reference from its effect on alpha.

Zero-shot + recalibrated metrics, binned MAE; preds saved to
eta_intervention_preds.npz for the relMAE tables.

Usage: python eta_interventions.py --run-id attn2_1h_L1024 --model attn2_1h \
           --context-length 1024
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
from interventions import MetricAcc, finish_from_o2


@torch.no_grad()
def parts_of(cap, hw2, L, device):
    alpha = cap["attn2_w"][:, :, :, 0].double()
    y2 = ov_images(cap["x2"], hw2)
    m_mass = (1.0 - alpha).clamp_min(1e-30)
    q = cap["attn2_w"][..., 1:].float() / m_mass.float()[..., None]
    g = torch.einsum("bhts,bhsd->bhtd", q, y2[:, :, 1:]).double()
    return {"alpha": alpha, "g": g, "o2": cap["o2"]}


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

    w_b = ref["w_bos_2"].to(device)       # [H,d] f64
    w_nb = ref["w_nonbos_2"].to(device)
    b_attn = ref["b_attn_2"].to(device)

    def eta_of(p):
        m = (1 - p["alpha"])[..., None]
        return m * (p["g"] - w_nb[None, :, None, :])   # [B,H,T,d] f64

    def rebuild(p, eta_new):
        base = (p["alpha"][..., None] * w_b[None, :, None, :]
                + (1 - p["alpha"][..., None]) * w_nb[None, :, None, :])
        return (base + eta_new).sum(1) + b_attn[None, None, :]

    # ---- reference pass: position-wise mean ||eta|| and mean eta ---------
    print("reference pass (eta norms + means)...")
    norm_sum = torch.zeros(H, L, device=device, dtype=torch.float64)
    eta_sum = torch.zeros(H, L, d, device=device, dtype=torch.float64)
    n_ref = 0
    for s, idx in iter_batches(splits["reference"], L, args.batch_size, device):
        cap = forward_capture(model, idx)
        p = parts_of(cap, hw2, L, device)
        eta = eta_of(p)
        norm_sum += eta.norm(dim=-1).sum(0)
        eta_sum += eta.sum(0)
        n_ref += idx.shape[0]
        del cap, p, eta
    eta_norm_ref = norm_sum / n_ref                    # [H,T]
    eta_mean_ref = eta_sum / n_ref                     # [H,T,d]

    def e1_constnorm(p):
        eta = eta_of(p)
        n = eta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return rebuild(p, eta * (eta_norm_ref[None, :, :, None] / n))

    def e2_shuffled(p):
        return rebuild(p, eta_of(p).roll(1, dims=0))

    def e3_refmean(p):
        return rebuild(p, eta_mean_ref[None].expand(
            p["alpha"].shape[0], -1, -1, -1))

    iv = {"E1_eta_constnorm": e1_constnorm, "E2_eta_shuffled": e2_shuffled,
          "E3_eta_refmean": e3_refmean,
          "E0_eta_rebuilt_baseline": lambda p: rebuild(p, eta_of(p))}

    # ---- targeted C7: remove L1 BOS value, hold Layer-2 alpha fixed ------
    @torch.no_grad()
    def c7_alpha_fixed(idx, a2_orig):
        emb = model.wte(idx)
        x1 = F.layer_norm(emb, model.block1.ln_1.weight.shape,
                          model.block1.ln_1.weight, model.block1.ln_1.bias, 1e-5)
        _, a1 = attn_scores_weights(x1, hw1)
        y1 = ov_images(x1, hw1)
        o1 = torch.einsum("bhts,bhsd->bhtd", a1, y1).sum(1) \
            + attn_bias_vector(hw1)[None, None, :]
        delta = torch.einsum("bht,bhd->btd", a1[:, :, :, 0], -y1[:, :, 0])
        post1 = emb + o1 + delta
        ln2 = F.layer_norm(post1, model.block1.ln_2.weight.shape,
                           model.block1.ln_2.weight, model.block1.ln_2.bias, 1e-5)
        h1 = post1 + model.block1.mlp(ln2)
        x2 = F.layer_norm(h1, model.block2.ln_1.weight.shape,
                          model.block2.ln_1.weight, model.block2.ln_1.bias, 1e-5)
        y2 = ov_images(x2, hw2)
        o2 = torch.einsum("bhts,bhsd->bhtd", a2_orig, y2).sum(1) \
            + attn_bias_vector(hw2)[None, None, :]
        return finish_from_o2(model, h1, o2.double())

    names = list(iv) + ["C7_remove_alpha_fixed"]

    # ---- calibration + evaluation passes ---------------------------------
    cal_preds = {n: [] for n in names}
    accs = {n: MetricAcc(L) for n in names}
    for split, sink in [("calibration", "cal"), ("evaluation", "eval")]:
        print(f"{split} pass...")
        for s, idx in iter_batches(splits[split], L, args.batch_size, device):
            cap = forward_capture(model, idx)
            p = parts_of(cap, hw2, L, device)
            preds = {n: finish_from_o2(model, cap["h1"], fn(p))
                     for n, fn in iv.items()}
            preds["C7_remove_alpha_fixed"] = c7_alpha_fixed(
                idx, cap["attn2_w"].float())
            for n, pred in preds.items():
                if sink == "cal":
                    cal_preds[n].append(
                        pred.cpu().double().numpy() * (train_L - 1))
                else:
                    accs[n].add(pred, train_L)
            del cap, p

    results = {}
    eval_preds = {}
    for n, acc in accs.items():
        res, p_eval = acc.result(bins)
        eval_preds[n] = p_eval
        res.update(MetricAcc.recalibrate(
            np.concatenate(cal_preds[n], 0), p_eval, L))
        results[n] = res
    np.savez(out / "eta_intervention_preds.npz", **eval_preds)
    update_summary(out, "eta_interventions", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
