"""P0 numerical consistency audit (exps.md).

Recomputes, from the canonical pipeline artifacts, every number whose value
is inconsistent across the manuscript:
  - BOS-bias ratio r_h (main text 287.3 vs appendix 21.939 for ATTN2-1H;
    incompatible FULL-12H head sets)
  - BOS/nonBOS cosine (-0.93 vs -0.969)
  - final decoding R^2 (0.9999 vs 0.999 vs 0.9929 vs 0.990)
Each key is computed at every available run (model x context length) so the
provenance of each published number can be pinned to one artifact key.

Usage: python audit.py --run-ids attn2_1h_L1024 full12h_L1024 attn2_1h_L128 full12h_L128
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, update_summary


def bos_bias_ratio(alpha: np.ndarray, L: int):
    """r_h per Eq. (bos_attention_metric): E_{b,t>0}[alpha_bos] /
    E_{b,t>0}[(1-alpha_bos)/t]. alpha: [n,(H,)T]."""
    if alpha.ndim == 2:
        alpha = alpha[:, None, :]
    t = np.arange(L)
    num = np.nanmean(alpha[:, :, 1:], axis=(0, 2))
    den = np.nanmean((1 - alpha[:, :, 1:]) / t[None, None, 1:], axis=(0, 2))
    return (num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", required=True)
    args = ap.parse_args()

    audit = {}
    for run_id in args.run_ids:
        out = RESULTS_ROOT / run_id
        if not (out / "config.json").exists():
            print(f"skip {run_id}: no config.json")
            continue
        cfg = json.loads((out / "config.json").read_text())
        L = cfg["context_length"]
        entry = {"context_length": L, "n_eval": cfg["splits"]["evaluation"],
                 "checkpoint": cfg["checkpoint_path"],
                 "checkpoint_hash": cfg["checkpoint_hash"][:16]}

        alpha = np.load(out / "step3__alpha_bos.npy")
        r_h = bos_bias_ratio(alpha, L)
        entry["bos_bias_ratio_per_head"] = np.round(r_h, 3).tolist()
        entry["bos_bias_ratio_sorted_desc"] = np.round(
            np.sort(r_h)[::-1], 3).tolist()

        ref = torch.load(out / "ref_stats.pt", map_location="cpu",
                         weights_only=False)
        w_b = ref["w_bos_2"].numpy()
        w_nb = ref["w_nonbos_2"].numpy()
        cos_h = (w_b * w_nb).sum(-1) / (
            np.linalg.norm(w_b, axis=-1) * np.linalg.norm(w_nb, axis=-1))
        entry["cos_bos_nonbos_per_head"] = np.round(cos_h, 4).tolist()
        # aggregate variant: cosine between head-summed vectors (a plausible
        # source of the -0.93 vs -0.969 discrepancy)
        cs = float(w_b.sum(0) @ w_nb.sum(0) /
                   (np.linalg.norm(w_b.sum(0)) * np.linalg.norm(w_nb.sum(0))))
        entry["cos_bos_nonbos_headsummed"] = round(cs, 4)
        entry["norm_ratio_bos_over_nonbos_mean"] = np.round(
            np.linalg.norm(w_b, axis=-1)
            / np.linalg.norm(w_nb, axis=-1), 3).tolist()
        # norm ratio in the empirical-evidence sense:
        # ||w_BOS|| vs E_j ||B_OV x_j|| — needs mu_j norms as proxy lower bound
        summ = {}
        if (out / "summary.json").exists():
            summ = json.loads((out / "summary.json").read_text())
        # final decoding from probes (evaluation split, zero-shot trained head)
        pd_ = summ.get("probes_decoders", {})
        if "model_head" in pd_:
            entry["final_decoding_model_head"] = {
                k: pd_["model_head"][k] for k in
                ["r2", "pearson_r2", "mae"] if k in pd_["model_head"]}
        iv = summ.get("interventions", {})
        if "baseline" in iv:
            entry["final_decoding_intervention_baseline"] = {
                k: iv["baseline"][k] for k in
                ["r2_zeroshot", "pearson_r2", "mae"] if k in iv["baseline"]}
        audit[run_id] = entry

    # cross-run reconciliation notes
    notes = []
    for a in ["attn2_1h_L1024", "attn2_1h_L128"]:
        if a in audit:
            notes.append(
                f"{a}: r_h={audit[a]['bos_bias_ratio_per_head']} "
                f"(manuscript claims 287.3 main / 21.939 appendix)")
    for a in ["full12h_L1024", "full12h_L128"]:
        if a in audit:
            notes.append(
                f"{a}: top-4 r_h={audit[a]['bos_bias_ratio_sorted_desc'][:4]}")
    audit["_reconciliation_notes"] = notes

    out_path = RESULTS_ROOT / "p0_audit.json"
    out_path.write_text(json.dumps(audit, indent=2))
    for run_id in args.run_ids:
        if run_id in audit:
            update_summary(RESULTS_ROOT / run_id, "p0_audit", audit[run_id])
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
