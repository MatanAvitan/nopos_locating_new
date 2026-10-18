"""Freeze the head selection from the validation screen (declared rule).

For each two-layer position model (primary convention: LN-bias center, global
anchor, readout dw; last position bin):
  candidates  block-1 heads with r_h > 10 and ordinary-key score R^2 (E omitted) > 0.5
  deep        the two candidates with the lowest validation relMAE of the
              isotonic decoder on z, tested one at a time
  group       all candidates, edited jointly
  negative    a block-1 head with r_h > 10 and score R^2 <= 0.5 (failing BOS
              head) if one exists, else the block-1 head with the lowest r_h
  matched     block-0 head 0 for one-head models (the same tests at layer 1)
Writes results/mechanism/complete_circuit/configs/selection.json. The test
split is never read here.

Usage: python cc_select.py [--models 1h_s42 ...]
"""

import argparse
import json

from cc_common import OUT_ROOT, SPECS, jdump

R2_MIN, RH_MIN = 0.5, 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[k for k, s in SPECS.items() if s.family in ("1h", "12h")])
    a = ap.parse_args()
    out = {"rule": __doc__.split("Writes")[0].strip(), "models": {}}
    for m in a.models:
        rows = [json.loads(l) for l in open(OUT_ROOT / "screen" / m / "head_metrics.jsonl")]
        rows = [r for r in rows if (r["center"], r["anchor"], r["readout"]) == ("ln_bias", "global", "dw") and r["layer"] == 1]
        lb = "late" if "late" in rows[0]["decode_refit"]["z"] else "middle"
        info = []
        for r in rows:
            sc = r["score_fit"].get(f"ord_{lb}", {}).get("r2_E_omitted")
            info.append({"head": r["head"], "r_h": r["r_h"], "score_r2_ord": sc, "z_relmae": r["decode_refit"]["z"][lb]["relmae"]})
        cand = [x for x in info if x["r_h"] > RH_MIN and x["score_r2_ord"] is not None and x["score_r2_ord"] > R2_MIN]
        cand.sort(key=lambda x: x["z_relmae"])
        fail = [x for x in info if x["r_h"] > RH_MIN and (x["score_r2_ord"] is None or x["score_r2_ord"] <= R2_MIN)]
        if len(info) == 1:                       # one head: the control is the block-0 head
            neg, kind = None, "block-0 head (one-head model)"
        elif fail:
            neg, kind = min(fail, key=lambda x: x["score_r2_ord"] if x["score_r2_ord"] is not None else -1e9), "failing BOS head"
        else:
            neg, kind = min(info, key=lambda x: x["r_h"]), "lowest r_h"
        sel = {"deep": [x["head"] for x in cand[:2]], "group": [x["head"] for x in cand],
               "negative": None if neg is None else neg["head"], "negative_kind": kind,
               "matched_block0": [0] if SPECS[m].family == "1h" else [], "bin": lb, "heads": info}
        out["models"][m] = sel
        print(f"{m}: deep {sel['deep']} group {sel['group']} negative {sel['negative']} ({sel['negative_kind']})")
    jdump(out, OUT_ROOT / "configs" / "selection.json")


if __name__ == "__main__":
    main()
