"""F6 of the BOS-share spec: refit-probe versions of three existing Layer-2
ablations from interventions.py (same edit functions, same reference vectors):
  C1_ideal_mixture  : eta -> 0 (the residual around the ideal mixture removed)
  C4_bos_to_nonbos  : w_BOS -> w_nonBOS
  C3_ablate_dw_sum  : the aggregate dw direction projected out
Frozen = trained pos_head. Refit = ridge on ln_f(h2), fitted on intervened
calibration activations.

Usage: python beta_f6.py --model 1h --device cuda:1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import MODELS, Ridge, bins_for, err_by_bin, make_adapter, out_dir, spec_splits
from common import RESULTS_ROOT, batch_from_offsets, head_weights, ov_images
from interventions import build_interventions

ARMS = ["baseline", "C1_ideal_mixture", "C4_bos_to_nonbos", "C3_ablate_dw_sum"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["1h", "12h"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    ref = torch.load(RESULTS_ROOT / run_id / "ref_stats.pt", map_location="cpu", weights_only=False)
    H, d = ad.H, ad.d
    iv = build_interventions(ad.model, ref, dev, H, d)
    hw2 = head_weights(ad.attn2)
    splits = spec_splits(L)
    bins = bins_for(L)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)

    @torch.no_grad()
    def parts(p):
        alpha = p["attn2_w"][:, :, :, 0].double()
        y2 = ov_images(p["x2"], hw2)
        q = p["attn2_w"][..., 1:].float() / (1.0 - alpha).clamp_min(1e-30).float()[..., None]
        g = torch.einsum("bhts,bhsd->bhtd", q, y2[:, :, 1:]).double()
        return {"alpha": alpha, "g": g, "g_unif": None, "q": q, "y2": y2, "o2": p["o2"]}

    def run(name):
        for s in range(0, len(splits[name]), args.batch_size):
            idx = batch_from_offsets(splits[name][s:s + args.batch_size], L, dev)
            p = ad.pre(idx)
            pp = parts(p)
            for a in ARMS:
                feats, fr, _ = ad.finish(p["h1"], iv[a](pp).float())
                yield a, feats, fr

    ridges = {a: Ridge(d, dev) for a in ARMS}
    for a, feats, _ in run("calibration"):
        ridges[a].add(feats, tgt.expand(feats.shape[0], L))
    for r in ridges.values():
        r.solve()
    fz = {a: [] for a in ARMS}; rf = {a: [] for a in ARMS}
    for a, feats, fr in run("evaluation"):
        fz[a].append(fr.cpu()); rf[a].append(ridges[a].predict(feats).cpu())
    S = {a: {"frozen": err_by_bin(torch.cat(fz[a]).numpy(), L, bins),
             "refit": err_by_bin(torch.cat(rf[a]).numpy(), L, bins)} for a in ARMS}
    (out_dir(run_id) / "F6.json").write_text(json.dumps(S, indent=2))
    for a, e in S.items():
        print(f"{a:20s} frozen late={e['frozen']['relmae_late']:.4f} mae={e['frozen']['mae_all']:.1f}"
              f" | refit late={e['refit']['relmae_late']:.4f} mae={e['refit']['mae_all']:.1f}")


if __name__ == "__main__":
    main()
