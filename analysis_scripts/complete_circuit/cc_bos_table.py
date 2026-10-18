"""Per-token BOS projections before and after the block-0 MLP (measured, not modeled).

Hooks (the checkpoint's own taps, eval mode, fp32 forward):
  h_before   = block1_post_attn  residual after the block-0 attention update, before ln_2
  mlp_delta  = block1_mlp        MLP output added to the residual (includes c_proj bias)
  h_after    = block1_out        residual after adding mlp_delta
Direction: w = sum_h W_O,h W_V,h xbar_0, xbar_0 = mean block-0 attention input
(ln_1 output) at position 0 over the first 160 calibration windows; u = w/|w|.
This is the cc_lncenter.py / cc_denoise.py definition; the script checks that u
matches the frozen direction in configs/frozen_denoise_<model>.pt.
Projections are fp64 dot products with u, in raw residual units:
  bos_before = <h_before, u>, mlp_correction = <mlp_delta, u>, bos_after = <h_after, u>.
Separately labeled extra column (other convention, not comparable with the raw
columns): ln_normalized_before = <(h_before - mean)/sd, u>, the LayerNorm-
normalized input without gamma and b (eps 1e-5).

Sequences: validation split, windows 0..n-1 (BOS + L-1 tokens from val.bin).

Usage: python cc_bos_table.py --model 1h_s42 --device cuda:0 [--n 160]
"""

import argparse
import gzip
import time

import numpy as np
import torch

from cc_common import EPS_LN, OUT_ROOT, SPECS, Capture, jdump, load_net, splits_for


def decoder():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda i: enc.decode([int(i)])
    except Exception:  # noqa: BLE001
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        return lambda i: tok.decode([int(i)])


def csv_text(s):
    return '"' + s.replace('"', '""') + '"'


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    dev, bs = a.device, a.batch_size
    net, path = load_net(a.model, dev)
    assert net.spec.kind == "toy"
    m_ = net.model
    L = net.spec.L
    sp, prov = splits_for(a.model)

    # ---- direction (recomputed, then compared with the frozen one) ----------------
    cal = sp["calibration"][:160]
    x0, n = 0, 0
    for s in range(0, len(cal), bs):
        idx = net.batch(cal[s:s + bs]); cap = Capture(want={("x_attn", 0)}); net.forward(idx, cap, stop_after=0)
        x0 = x0 + cap.store[("x_attn", 0)][:, 0].double().sum(0); n += idx.shape[0]
    w = torch.einsum("hdf,f->d", net.ov(0), x0 / n)
    u = w / w.norm()
    fz = OUT_ROOT / "configs" / f"frozen_denoise_{a.model}.pt"
    fz_check = None
    if fz.exists():
        uf = torch.load(fz, map_location=dev, weights_only=False)["w"].double()
        fz_check = {"cos_with_frozen": float(u @ uf), "max_abs_diff": float((u - uf).abs().max())}

    # ---- measured projections -----------------------------------------------------
    dec = decoder()
    offs = sp["validation"][:a.n]
    out_csv = OUT_ROOT / "bos_table" / f"{a.model}_validation_first{a.n}.csv.gz"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    R_b, R_m, R_a, R_n, TOK = [], [], [], [], []
    max_vec_err = 0.0; max_proj_err = 0.0
    t0 = time.time()
    with gzip.open(out_csv, "wt") as f:
        f.write("model,sequence_id,split_index,val_bin_offset,position,token_id,token_text,"
                "bos_before,mlp_correction,bos_after,addition_error,ln_normalized_before\n")
        for s in range(0, len(offs), bs):
            idx = net.batch(offs[s:s + bs])
            m_.eval()
            m_(idx, targets=None, capture_taps=True)
            taps = m_.get_all_taps()
            hb, md, ha = (taps[k].double() for k in ("block1_post_attn", "block1_mlp", "block1_out"))
            max_vec_err = max(max_vec_err, float((ha - hb - md).abs().max()))
            rb, rm, ra = hb @ u, md @ u, ha @ u
            c = hb - hb.mean(-1, keepdim=True)
            rn = (c / (c.pow(2).mean(-1, keepdim=True) + EPS_LN).sqrt()) @ u
            err = ra - rb - rm
            max_proj_err = max(max_proj_err, float(err.abs().max()))
            for k in range(idx.shape[0]):
                si = s + k
                ids = idx[k].tolist()
                cols = [x[k].cpu().numpy() for x in (rb, rm, ra, err, rn)]
                for p in range(L):
                    f.write(f"{a.model},val{si:04d}_off{int(offs[si])},{si},{int(offs[si])},{p},{ids[p]},{csv_text(dec(ids[p]))},"
                            f"{cols[0][p]:.6f},{cols[1][p]:.6f},{cols[2][p]:.6f},{cols[3][p]:.3e},{cols[4][p]:.6f}\n")
            R_b.append(rb.cpu()); R_m.append(rm.cpu()); R_a.append(ra.cpu()); R_n.append(rn.cpu()); TOK.append(idx.cpu())
    RB, RM, RA, RN, TK = (torch.cat(x).numpy() for x in (R_b, R_m, R_a, R_n, TOK))
    print(f"extracted {RB.shape} in {time.time() - t0:.0f}s; max |h_after - h_before - mlp_delta| = {max_vec_err:.2e}; "
          f"max |projection addition error| = {max_proj_err:.2e}", flush=True)

    # ---- per-position summary (SD with ddof = 1 across sequences) ---------------------
    rows = []
    for p in range(L):
        r = {"position": p, "n_sequences": int(RB.shape[0])}
        for name, X in (("bos_before", RB), ("mlp_correction", RM), ("bos_after", RA)):
            x = X[:, p]
            r[f"{name}_mean"] = float(x.mean()); r[f"{name}_sd"] = float(x.std(ddof=1))
            r[f"{name}_min"] = float(x.min()); r[f"{name}_max"] = float(x.max())
        vb, va = RB[:, p].var(ddof=1), RA[:, p].var(ddof=1)
        r["var_after_over_var_before"] = (float(va / vb) if vb > 1e-24 else None)
        r["var_before"], r["var_after"] = float(vb), float(va)
        rows.append(r)
    keys = list(rows[0])
    with open(OUT_ROOT / "bos_table" / f"{a.model}_validation_first{a.n}_by_position.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("undefined" if r[k] is None else (f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])) for k in keys) + "\n")

    # ---- reproduce the earlier aggregates -------------------------------------------
    def agg(lo, hi):
        sl = slice(lo, hi + 1)
        return {"positions": [lo, hi], "n_sequences": int(RB.shape[0]), "n_tokens": int(RB[:, sl].size),
                "bos_before_mean": float(RB[:, sl].mean()), "mlp_correction_mean": float(RM[:, sl].mean()),
                "bos_after_mean": float(RA[:, sl].mean()),
                "mean_over_positions_of_within_position_sd_before": float(RB[:, sl].std(0, ddof=0).mean()),
                "mean_over_positions_of_within_position_sd_after": float(RA[:, sl].std(0, ddof=0).mean())}
    groups = {"0": (0, 0), "1": (1, 1), "8-15": (8, 15), "16-127": (16, min(127, L - 1))}
    if L > 128:
        groups["128-%d" % (L - 1)] = (128, L - 1)
    first_ids = TK[:, 0]
    summary = {"model": a.model, "checkpoint": str(path), "split": "validation", "sequences": f"validation windows 0..{a.n - 1}",
               "split_provenance": prov, "direction": "u = w/|w|, w = sum_h W_O,h W_V,h xbar_0 (block 0, no biases), "
               "xbar_0 = mean block-0 ln_1 output at position 0 over calibration windows 0..159",
               "w_norm": float(w.norm()), "direction_check_vs_frozen": fz_check,
               "max_vector_addition_error": max_vec_err, "max_projection_addition_error": max_proj_err,
               "first_token_ids": sorted(set(first_ids.tolist())), "n_sequences_starting_with_50256": int((first_ids == 50256).sum()),
               "n_in_window_50256_after_position_0": int((TK[:, 1:] == 50256).sum()),
               "aggregates": {k: agg(*v) for k, v in groups.items()}}
    jdump(summary, OUT_ROOT / "bos_table" / f"{a.model}_validation_first{a.n}_summary.json")
    print(summary["aggregates"], flush=True)


if __name__ == "__main__":
    main()
