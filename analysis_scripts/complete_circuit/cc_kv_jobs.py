"""Write cc_kv.py job lists from configs/selection.json, split over GPUs.

Usage: python cc_kv_jobs.py --split validation --gpus 4 5
Then run each logs/cc_kv_jobs_<split>_g<gpu>.sh with bash.
"""

import argparse
import json

from cc_common import OUT_ROOT, REPO_ROOT

PY = "/home/nlp/matan_avitan/.conda/envs/py312/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--gpus", nargs="+", required=True)
    a = ap.parse_args()
    sel = json.loads((OUT_ROOT / "configs" / "selection.json").read_text())["models"]
    jobs = []
    for m, s in sel.items():
        runs = [[h] for h in s["deep"]]
        if len(s["group"]) > 1:
            runs.append(s["group"])
        if s["negative"] is not None:
            runs.append([s["negative"]])
        for r in runs:
            jobs.append((m, " ".join(f"1:{h}" for h in r), "-".join(map(str, r)), 1))
        for h in s["matched_block0"]:
            jobs.append((m, f"0:{h}", str(h), 0))
    n_eval = 160 if a.split == "validation" else 400
    for k, g in enumerate(a.gpus):
        lines = ["#!/bin/bash", f"cd {REPO_ROOT}/analysis_scripts/complete_circuit"]
        for i, (m, heads, tag, l) in enumerate(jobs):
            if i % len(a.gpus) != k:
                continue
            bs = 2 if m.startswith("12h") else 4
            log = REPO_ROOT / "logs" / f"cc_kv_{m}_L{l}_h{tag}_{a.split}.log"
            lines.append(f"echo $(date +%T) start {m} {heads}; CUDA_VISIBLE_DEVICES={g} {PY} cc_kv.py --model {m} --heads {heads} "
                         f"--split {a.split} --n-eval {n_eval} --batch-size {bs} --device cuda:0 > {log} 2>&1; echo $(date +%T) done {m} {heads} exit $?")
        lines.append("echo all done")
        p = REPO_ROOT / "logs" / f"cc_kv_jobs_{a.split}_g{g}.sh"
        p.write_text("\n".join(lines) + "\n")
        print(p, sum(1 for i in range(len(jobs)) if i % len(a.gpus) == k), "jobs")


if __name__ == "__main__":
    main()
