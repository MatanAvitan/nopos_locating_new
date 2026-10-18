"""Summarize cc_kv.py results: four-condition terms by key group, and endpoints.

Usage: python cc_kv_table.py --split validation [--models 1h_s42 ...] [--bin late]
"""

import argparse
import json
from pathlib import Path

from cc_common import OUT_ROOT

PAIR_ORDER = ["P2", "P2s", "P2e", "P2t", "P3", "P3s", "P3e", "P3t", "P4", "P1", "P1f"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--bin", default=None)
    a = ap.parse_args()
    files = sorted((OUT_ROOT / "kv").glob(f"*/kv_*_{a.split}.json"))
    for f in files:
        r = json.loads(f.read_text())
        if a.models and r["model"] not in a.models:
            continue
        hs = r["heads"]
        b = a.bin or ("late" if "late" in next(iter(r["decomposition"][str(hs[0])].values()))["bins"] else "middle")
        print(f"\n== {r['model']} L{r['layer']} heads {hs} ({a.split}, n={r['n_eval']}, bin {b}) frozen={r.get('frozen', {}).get('loaded')}")
        for h in map(str, hs):
            print(f"  head {h}  pair: var z11 | var z00 | var K, V, I | 2cov K,V [CI] | 2cov K,I  V,I | corr K,V")
            D = r["decomposition"][h]
            for p in PAIR_ORDER:
                if p not in D:
                    continue
                x = D[p]["bins"][b]; ci = x["ci"]["two_cov_K_V"]
                print(f"   {p:4s} {x['var_z11']:8.3g} | {x['var_z00']:8.3g} | {x['var_K']:8.3g} {x['var_V']:8.3g} {x['var_I']:8.3g} | "
                      f"{x['two_cov_K_V']:8.3g} [{ci[0]:.3g},{ci[1]:.3g}] | {x['two_cov_K_I']:8.3g} {x['two_cov_V_I']:8.3g} | {x['corr_K_V']:5.2f}")
        print("  condition: z iso refit [diff CI] | z iso frozen | frozen offset-removed | frozen head [offset-removed] | ridge refit [diff CI]")
        for c, e in r["conditions"].items():
            parts = []
            for h in map(str, hs[:3]):
                z = e["z"][h]
                d = z.get("iso_refit_diff_vs_orig", {}).get(b)
                parts.append(f"h{h} {z['iso_refit'][b]['relmae']:.4f}" + (f" [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]" if d else "")
                             + f" fr {z['iso_frozen'][b]['relmae']:.4f} fr-off {z['iso_frozen_offset_removed'][b]['relmae']:.4f}")
            if "frozen_head" in e:
                parts.append(f"head {e['frozen_head'][b]['relmae']:.4f} [{e['frozen_head_offset_removed'][b]['relmae']:.4f}]")
            else:
                parts.append(f"dloss {e['lm_loss']['delta_vs_orig']:+.4f}")
            d = e.get("ridge_refit_diff_vs_orig", {}).get(b)
            parts.append(f"ridge {e['ridge_refit'][b]['relmae']:.4f}" + (f" [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]" if d else ""))
            print(f"   {c:9s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
