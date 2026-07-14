"""Emit MAE / relative-MAE tables for every experiment from saved predictions.

Reads probe_preds__*.npy, intervention_preds.npz, and LM L*__probe_pred.npy
arrays; reports MAE by position bin and median relative error by bin.
Writes results/mechanism/mae_tables.md and prints it.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, position_bins


def rows_for(preds: np.ndarray, L: int):
    """preds: [n_seq, L] absolute-position predictions."""
    t = np.arange(L)[None, :].astype(np.float64)
    err = np.abs(preds - t)
    rel = err[:, 1:] / t[:, 1:]
    out = {
        "MAE": float(np.nanmean(err[:, 1:])),
        "relMAE_mean": float(np.nanmean(rel)),
        "relMAE_med": float(np.nanmedian(rel)),
    }
    for name, (a, b) in position_bins(L).items():
        a = max(a, 1)
        r = err[:, a:b + 1] / t[:, a:b + 1]
        out[f"MAE_{name}"] = float(np.nanmean(err[:, a:b + 1]))
        out[f"relMAE_{name}_mean"] = float(np.nanmean(r))
        out[f"relMAE_{name}_med"] = float(np.nanmedian(r))
    return out


def fmt(v, pct=False):
    return f"{100*v:.1f}%" if pct else f"{v:.1f}"


def relcell(r, key):
    return f"{fmt(r[key + '_mean'], pct=True)} ({fmt(r[key + '_med'], pct=True)})"


def table(title, entries, L):
    bins = list(position_bins(L))
    lines = [f"\n### {title} (L={L})\n",
             "| experiment | relMAE | " + " | ".join(f"relMAE {b}" for b in bins)
             + " | MAE | " + " | ".join(f"MAE {b}" for b in bins) + " |",
             "|---|" + "---|" * (2 + 2 * len(bins))]
    for name, r in entries:
        lines.append(
            f"| {name} | {relcell(r, 'relMAE')} | "
            + " | ".join(relcell(r, f'relMAE_{b}') for b in bins)
            + f" | {fmt(r['MAE'])} | "
            + " | ".join(fmt(r[f'MAE_{b}']) for b in bins) + " |")
    return "\n".join(lines)


PROBE_ORDER = [
    "probe_linear_proxy", "probe_linear_h1bar", "probe_linear_post1",
    "probe_linear_h1", "probe_linear_o2", "probe_linear_post2",
    "probe_linear_ln4", "probe_linear_mlp_hidden", "probe_linear_m2",
    "probe_linear_h2", "probe_mlp_h1", "probe_mlp_o2", "probe_mlp_post2",
    "decode_linear_alpha", "decode_analytic_alpha", "decode_mlp_alpha",
    "decode_linear_Y", "decode_mlp_Y", "decode_term_h1", "decode_term_o2",
    "decode_terms_joint", "decode_terms_all", "model_head",
]

IV_ORDER = [
    "baseline", "C1_ideal_mixture", "C5_residual_only", "C4_bos_to_nonbos",
    "C4_nonbos_to_bos", "C2_retain_mech_span", "C2_ablate_mech_span",
    "C2_retain_top2_svd", "C2_ablate_top2_svd", "C3_ablate_dw_sum",
    "C3_ablate_dw_span", "C3_ablate_top1_svd", "C6_alpha_refmean",
    "C6_alpha_permuted", "C6_alpha_clamped", "C6_uniform_nonbos_attn",
    "S5_remove_mlp2", "S5_linearize_mlp2", "S5_mlp2_readdir_only",
    "S5_patch_bos_coordinate", "C7_remove", "C7_fixed_random",
    "C7_seq_varying",
]


def main():
    parts = []
    for run in ["attn2_1h_L1024", "full12h_L1024"]:
        out = RESULTS_ROOT / run
        L = json.loads((out / "config.json").read_text())["context_length"]
        entries = []
        for name in PROBE_ORDER:
            f = out / f"probe_preds__{name}.npy"
            if f.exists():
                entries.append((name, rows_for(np.load(f), L)))
        parts.append(table(f"{run} — probes & decoders", entries, L))
        ivf = out / "intervention_preds.npz"
        if ivf.exists():
            z = np.load(ivf)
            entries = [(n, rows_for(z[n], L)) for n in IV_ORDER if n in z]
            parts.append(table(f"{run} — causal interventions", entries, L))
    for run in ["lm6_L128", "lm6_L128_nobos"]:
        out = RESULTS_ROOT / run
        if not out.exists():
            continue
        L = json.loads((out / "config.json").read_text())["context_length"]
        entries = []
        for f in sorted(out.glob("L*__probe_pred.npy")):
            block = f.stem.split("__")[0]
            entries.append((f"linear probe @ {block} residual",
                            rows_for(np.load(f), L)))
        parts.append(table(f"{run} — NoPE LM block-wise position probes",
                           entries, L))
    doc = ("# Relative-MAE / MAE for all experiments\n\n"
           "Primary metric: relMAE = |ŷ−i|/i over positions i ≥ 1, shown as "
           "mean (median). Absolute-position MAE kept as secondary columns. "
           "Position 0 excluded everywhere. Bins: early 1–15, middle 16–127, "
           "late 128–L−1.\n" + "\n".join(parts) + "\n")
    (RESULTS_ROOT / "mae_tables.md").write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
