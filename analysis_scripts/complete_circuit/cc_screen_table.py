"""Compact per-head table of the screen (primary convention: LN-bias center,
global anchor, readout dw). Validation split only.

Usage: python cc_screen_table.py [--root results/mechanism/complete_circuit] [--models ...]
"""

import argparse
import json
from pathlib import Path


def last_bin(r):
    return "late" if "late" in r["decode_refit"]["z"] else "middle"


def fmt(x, p=3):
    return "  -  " if x is None else (f"{x:.{p}f}" if abs(x) < 1e4 else f"{x:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "results/mechanism/complete_circuit"))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--center", default="ln_bias")
    ap.add_argument("--anchor", default="global")
    ap.add_argument("--readout", default="dw")
    a = ap.parse_args()
    root = Path(a.root) / "screen"
    models = a.models or sorted(p.name for p in root.iterdir() if (p / "head_metrics.jsonl").exists())
    hdr = ("model", "L", "h", "r_h", "B", "cvB", "scR2ord", "E/pr", "vR2fit", "vR2str", "sdr/lam", "cosVd", "cosKd",
           "zdec", "zVs", "zK2", "zK3", "zK3Vf", "lratio")
    print(" ".join(f"{h:>8s}" for h in hdr))
    for m in models:
        for line in open(root / m / "head_metrics.jsonl"):
            r = json.loads(line)
            if (r["center"], r["anchor"], r["readout"]) != (a.center, a.anchor, a.readout):
                continue
            lb = last_bin(r)
            sc = r["score_fit"].get(f"ord_{lb}") or r["score_fit"].get("ord_middle") or {}
            vt = r["value_fit"]["tail"]
            dr = r["decode_refit"]
            row = [m, r["layer"], r["head"], r["r_h"], r["B_const"], r["B_i"]["cv_tail"], sc.get("r2_E_omitted"),
                   sc.get("sd_E_over_sd_pred"), vt["r2_fitted"], vt["r2_structural"], vt["sd_r_over_abs_lambda"],
                   r.get("cos_value_read_vs_d"), r.get("cos_mean_key_read_vs_d"),
                   dr["z"][lb]["relmae"], dr["zVs"][lb]["relmae"], dr["zK2"][lb]["relmae"], dr["zK3"][lb]["relmae"],
                   dr["zK3Vf"][lb]["relmae"], r["local_ratio"]["z"][lb]["median"]]
            out = []
            for x in row:
                out.append(f"{x:>8s}" if isinstance(x, str) else (f"{x:>8d}" if isinstance(x, int) else f"{fmt(x):>8s}"))
            print(" ".join(out))


if __name__ == "__main__":
    main()
