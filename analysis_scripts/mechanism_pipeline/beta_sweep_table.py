"""Sweep summary (beta spec, sweep): one table (rows = models) and one figure
(centered share profile against j, colored by training length).

Reads, per model, beta/summary.json (beta_extract + beta_analyze) and
beta_v2/{summary.json,evaluation.npz} (beta_v2). Columns:
  profile     centered share at j = 1, 2, 4, 8, 16 and the mean over j >= 16
  B_med       median per-query slope on the centered share
  B_ml        maximum-likelihood B of the softmax-over-shares model
  gen_err     median |alpha_hat - alpha| / alpha of alpha_BOS under
              softmax(B_ml beta_j), each sequence's own shares, by bin
  t_rho       fraction of sequences with Spearman(t_i, i) < -0.9
  a, a', i*   two-level fit (only where the model has sinks)
  iso_late    relMAE of the isotonic decoder on z_i, last bin
  sep L1/L2   separation ratio (N4) at the Layer-1 and Layer-2 inputs

Usage: python beta_sweep_table.py [--keys k1 k2 ...]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import MODELS, RESULTS_ROOT, bins_for
from beta_analyze import rowwise_spearman
from beta_extras import ml_B

DEFAULT = (["1h", "12h", "lm", "1h_8192", "1h_seed1", "1h_seed2", "1h_seed3", "1h_seed4"]
           + [k for k in MODELS if k.startswith("sw_") or k.startswith("h1_")])


def general_fit(E, h, L):
    b = E["b_x2"].astype(np.float64)
    t = E["t"][:, h].astype(np.float64)
    am = E["alpha_bos"][:, h].astype(np.float64)
    Bml, _ = ml_B(t, b, np.linspace(0.05, 25, 500))
    ah = np.exp(-np.logaddexp.accumulate(Bml * (b - 1), axis=1))
    rel = np.abs(ah - am) / am
    return Bml, {n: float(np.median(rel[:, a:c + 1])) for n, (a, c) in bins_for(L).items()}


def row(key):
    _, _, L, run_id = MODELS[key]
    d1, d2 = RESULTS_ROOT / run_id / "beta", RESULTS_ROOT / run_id / "beta_v2"
    if not (d1 / "summary.json").exists() or not (d2 / "summary.json").exists():
        return None
    S1 = json.loads((d1 / "summary.json").read_text())
    S2 = json.loads((d2 / "summary.json").read_text())
    E = np.load(d2 / "evaluation.npz")
    h = S2["bos_heads"][0]
    prof = S2["N1_profile"]["b_x2"]["per_j_mean"]
    b = E["b_x2"]
    Bml, gerr = general_fit(E, h, L)
    t_rho = float((rowwise_spearman(E["t"][:, h, 1:].astype(np.float64)) < -0.9).mean())
    last = "late" if L > 128 else "middle"
    n2 = S2["N2"].get(f"h{h}", {})
    r = {"key": key, "L": L, "bos_heads": S2["bos_heads"],
         "r_h_max": float(max(S1["meta"]["r_h"])),
         "profile": {j: float(prof[str(j)]) for j in [1, 2, 4, 8, 16]},
         "share_j16on": float(b[:, 16:].mean()),
         "B_med": float(S2["B_L2"][h]), "B_ml": Bml, "gen_err": gerr, "t_rho": t_rho,
         "sinks": S2["sinks_L2"], "a": S2["a_L2"],
         "a_prime": n2.get("free_fit", {}).get("a_prime"), "i_star": n2.get("i_star"),
         "iso_last_relmae": S1["A2"]["L2_agg_w"][f"relmae_{last}"],
         "frozen_last_relmae": S1["A2"]["frozen_head"][f"relmae_{last}"],
         "sep_L1": S2["N4"]["L1"]["separation_ratio"], "sep_L2": S2["N4"]["L2"]["separation_ratio"],
         "mean_profile": b.mean(0)}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    rows = [r for r in (row(k) for k in args.keys) if r is not None]
    od = RESULTS_ROOT / "sweep"
    od.mkdir(parents=True, exist_ok=True)
    (od / "table.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "mean_profile"} for r in rows],
                                              indent=2, default=float))
    hdr = ("| model | L | β1 | β2 | β4 | β8 | β16 | B_ml | gen err (bins) | t ρ<−0.9 | a | a′ | i* | iso relMAE | frozen relMAE | sep L1/L2 |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for r in rows:
        p = r["profile"]
        ge = "/".join(f"{v:.3f}" for v in r["gen_err"].values())
        ap_ = f"{r['a_prime']:.2f}" if (r["a"] > 1 and r["a_prime"] is not None) else "–"
        ist = f"{r['i_star']:.0f}" if (r["a"] > 1 and r["i_star"] is not None) else "–"
        lines.append(f"| {r['key']} | {r['L']} | {p[1]:.2f} | {p[2]:.2f} | {p[4]:.2f} | {p[8]:.2f} | {p[16]:.2f} | "
                     f"{r['B_ml']:.2f} | {ge} | {r['t_rho']:.2f} | {r['a']} | {ap_} | {ist} | "
                     f"{r['iso_last_relmae']:.4f} | {r['frozen_last_relmae']:.4f} | {r['sep_L1']:.0f}/{r['sep_L2']:.0f} |")
    (od / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    Ls = sorted({r["L"] for r in rows})
    cmap = plt.get_cmap("viridis")
    col = {L: cmap(k / max(len(Ls) - 1, 1)) for k, L in enumerate(Ls)}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        ls = "--" if "unifL1" in r["key"] else (":" if "noMLP1" in r["key"] else "-")
        if r["key"] in ("lm", "12h") or r["key"].startswith("h1_"):
            continue
        j = np.arange(1, min(r["L"], 1024))
        ax.plot(j, r["mean_profile"][1:len(j) + 1], ls, color=col[r["L"]], lw=1, alpha=.85)
    for L in Ls:
        ax.plot([], [], color=col[L], label=f"L = {L}")
    ax.plot([], [], "k--", label="uniform Layer 1"); ax.plot([], [], "k:", label="no Layer-1 MLP")
    ax.set_xscale("log"); ax.set_xlabel("key position j"); ax.set_ylabel("centered BOS share (mean)")
    ax.axhline(0, color="k", lw=.4); ax.legend(fontsize=7)
    ax.set_title("Layer-2 input: centered BOS share by training length (1H)")
    fig.tight_layout(); fig.savefig(od / "share_profiles_by_L.png", dpi=130)


if __name__ == "__main__":
    main()
