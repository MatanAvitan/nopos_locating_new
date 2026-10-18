"""Empirical evidence for the mechanism story, step by step, for one model.

"Projection on BOS" = projection on u, the unit direction of BOS's residual
after Layer-1 attention (BOS's post-attention residual is the same vector in
every sequence). Layer 1 / Layer 2 = the first / second block.

Step 1  Before the Layer-1 MLP. Per position: projection of the residual on u
        (mean, sd and 10-90% over sequences), split exactly into the token's
        embedding, BOS's value (times its attention), the other tokens'
        values, and the attention bias. Layer-1 attention on BOS.
Step 2  The Layer-1 MLP. Its correction (projection of the MLP output on u)
        against the projection before it; sd before / after per position;
        per-token slope of correction on pre-projection at fixed late
        positions; the common late level; BOS before / after; one example
        sequence (positions 0-8).
Step 3  Layer-2 keys and queries (each BOS head). Feature phi_j = projection of
        the Layer-2 input on the unit direction of BOS's Layer-2 input. Per
        query: R^2 of the scores on phi (keys 0..i); gap per unit phi
        B_i = q_i . W_K d / sqrt(d_h), split into the query-bias part and the
        W_Q x_i part; spread of B_i over queries and sequences.
Step 4  Counting. Score gap G = s_i0 - median over ordinary keys; number of
        BOS-like keys a; measured BOS attention against e^G / (a e^G + n).
        Free fit 1/alpha = a' + c n.
Step 5  Values (each BOS head). v_j = projection of the value (OV image) on the
        unit direction of the head's BOS value. v_BOS, v_1, mean and sd of
        ordinary v; readout formula z_i ~ a_i0 v_0 + a_i1 v_1 + (1 - a_i0 - a_i1)
        v_ord against the measured output projection z_i, per sequence.
Step 6  Why one layer cannot. Isotonic decoding error of the readout for four
        pairings of (attention weights, values): L2 weights + L2 values (the
        model), L1 weights + L1 values (Layer-1 readout), L2 weights + L1
        values, L1 weights + L2 values. All values are projected on their own
        BOS-value direction.

Usage: python story_evidence.py --model 1h --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_analyze import isotonic_decode
from beta_common import MODELS, RESULTS_ROOT, bins_for, err_by_bin, make_adapter, spec_splits
from common import attn_bias_vector, batch_from_offsets, head_weights, ov_images

N_EVAL, N_CAL = 400, 160
POS = list(range(0, 17)) + [32, 64, 127, 256, 512, 1000]


def sinks_from(phi_share):
    frac = (phi_share[:, 1:] > 0.5).mean(0)
    return [int(j) + 1 for j in np.nonzero(frac > 0.9)[0]]


@torch.no_grad()
def collect(ad, L, heads, offsets, dev, bs, ref):
    hw1, hw2 = head_weights(ad.attn1), head_weights(ad.attn2)
    u, d2u, uv2, uv1 = ref["u"], ref["d2u"], ref["uv2"], ref["uv1"]
    out = {k: [] for k in ["pre", "corr", "post", "emb", "bosval", "othval", "bias", "a1bos",
                           "phi", "share2", "r2", "Bi", "Bi_bias", "Bi_wq", "gap", "a2bos", "a2pos1",
                           "z2", "v2", "z1", "v1", "z_l2w_l1v", "z_l1w_l2v"]}
    b1 = attn_bias_vector(hw1).double()
    for s in range(0, len(offsets), bs):
        idx = batch_from_offsets(offsets[s:s + bs], L, dev)
        p = ad.pre(idx)
        post1, h1 = p["post1"].double(), p["h1"].double()
        emb = (p["post1"] - p["o1"]).double() if ad.kind == "lm" else (p["post1"] - p["o1"]).double()
        y1 = ov_images(p["x1"], hw1).double()                                   # [B,H1,T,d]
        A1 = p["attn1_w"].double()                                              # [B,H1,T,T]
        proj1 = y1 @ u                                                          # [B,H1,T]
        out["pre"].append((post1 @ u).cpu()); out["post"].append((h1 @ u).cpu())
        out["corr"].append(((h1 - post1) @ u).cpu()); out["emb"].append((emb @ u).cpu())
        out["bosval"].append((A1[..., 0] * proj1[..., :1]).sum(1).cpu())
        out["othval"].append(torch.einsum("bhij,bhj->bi", A1[..., 1:], proj1[..., 1:]).cpu())
        out["bias"].append(torch.full_like(post1[..., 0], float(b1 @ u)).cpu())
        out["a1bos"].append(A1[..., 0].mean(1).cpu())
        # ---- Layer 2 -------------------------------------------------------
        x2 = p["x2"].double()
        phi = x2 @ d2u                                                          # projection on BOS dir
        share2 = ((x2 - ref["mu2"]) @ ref["d2"]) / (ref["d2"] @ ref["d2"])
        out["phi"].append(phi.cpu()); out["share2"].append(share2.cpu())
        S2 = p["scores2"].double()[:, heads]                                    # [B,nh,T,T]
        T = S2.shape[-1]
        M = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool))
        s0 = torch.where(M, S2, torch.zeros((), device=dev, dtype=torch.float64))
        n = torch.arange(1, T + 1, device=dev, dtype=torch.float64)
        mf = torch.cumsum(phi, -1) / n; mff = torch.cumsum(phi * phi, -1) / n
        ms = s0.sum(-1) / n; mss = (s0 * s0).sum(-1) / n; msf = torch.einsum("bhij,bj->bhi", s0, phi) / n
        cov = msf - ms * mf[:, None]; vf = (mff - mf * mf)[:, None]; vs = mss - ms * ms
        r2 = cov * cov / (vf * vs); r2[..., 0] = float("nan")
        out["r2"].append(r2.cpu())
        Wq = hw2.W_q.double()[heads]; bq = hw2.b_q.double()[heads]; Wk = hw2.W_k.double()[heads]
        kd = torch.einsum("hed,d->he", Wk, d2u)                                  # W_K d (unit d)
        qx = torch.einsum("btd,hed->bhte", x2, Wq)
        Bw = torch.einsum("bhte,he->bht", qx, kd) * hw2.scale
        Bb = (bq * kd).sum(-1)[None, :, None].expand_as(Bw) * hw2.scale
        out["Bi"].append((Bw + Bb).cpu()); out["Bi_wq"].append(Bw.cpu()); out["Bi_bias"].append(Bb.cpu())
        A2 = p["attn2_w"].double()[:, heads]
        out["a2bos"].append(A2[..., 0].cpu()); out["a2pos1"].append(A2[..., 1].cpu())
        # score gap BOS vs ordinary keys (j >= 16), per query i >= 16
        ordm = M.clone(); ordm[:, :16] = False
        so = torch.where(ordm, S2, torch.full((), float("nan"), device=dev, dtype=torch.float64))
        out["gap"].append((S2[..., 0] - so.nanmedian(-1).values).cpu())
        # values and readouts
        y2 = ov_images(p["x2"], hw2).double()[:, heads]                         # [B,nh,T,d]
        v2 = torch.einsum("bhtd,hd->bht", y2, uv2)
        out["v2"].append(v2.cpu()); out["z2"].append(torch.einsum("bhij,bhj->bhi", A2, v2).cpu())
        v1 = (y1.sum(1) @ uv1)                                                  # Layer-1 values (summed heads)
        out["v1"].append(v1.cpu())
        A1m = A1.mean(1)                                                        # Layer-1 weights (head mean)
        out["z1"].append(torch.einsum("bij,bj->bi", A1m, v1).cpu())
        out["z_l2w_l1v"].append(torch.einsum("bhij,bj->bhi", A2, v1).cpu())
        out["z_l1w_l2v"].append(torch.einsum("bij,bhj->bhi", A1m, v2).cpu())
        del p, y1, y2, S2, s0, A1, A2
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def reference(ad, L, heads, offsets, dev, bs):
    hw1, hw2 = head_weights(ad.attn1), head_weights(ad.attn2)
    p = ad.pre(batch_from_offsets(offsets[:1], L, dev))
    bos_post = p["post1"][0, 0].double()
    u = bos_post / bos_post.norm()
    x2b = p["x2"][0, 0].double()
    s2 = torch.zeros_like(x2b); n = 0
    for s in range(0, len(offsets), bs):
        idx = batch_from_offsets(offsets[s:s + bs], L, dev)
        x2 = ad.pre(idx)["x2"].double(); s2 += x2[:, 1:].sum((0, 1)); n += idx.shape[0] * (L - 1)
    mu2 = s2 / n
    M2 = torch.einsum("hde,hef->hdf", hw2.W_o.double(), hw2.W_v.double())[heads]
    wb2 = torch.einsum("hdf,f->hd", M2, x2b)
    M1 = torch.einsum("hde,hef->df", hw1.W_o.double(), hw1.W_v.double())
    wb1 = M1 @ p["x1"][0, 0].double()
    return {"u": u, "d2u": x2b / x2b.norm(), "mu2": mu2, "d2": x2b - mu2,
            "uv2": wb2 / wb2.norm(dim=-1, keepdim=True), "uv1": wb1 / wb1.norm()}


def stat(X, j):
    x = X[:, j]
    return {"mean": float(x.mean()), "sd": float(x.std()), "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    meta = json.loads((RESULTS_ROOT / run_id / "beta" / "meta.json").read_text())
    heads = meta["bos_heads"] or [int(np.argmax(meta["r_h"]))]
    sp = spec_splits(L); bins = bins_for(L)
    od = RESULTS_ROOT / "story_evidence" / args.model; od.mkdir(parents=True, exist_ok=True)
    ref = reference(ad, L, heads, sp["reference"][:160], dev, args.batch_size)
    E = collect(ad, L, heads, sp["evaluation"][:N_EVAL], dev, args.batch_size, ref)
    C = collect(ad, L, heads, sp["calibration"][:N_CAL], dev, args.batch_size, ref)
    pos = [j for j in POS if j < L]
    S = {"model": args.model, "L": L, "bos_heads": heads, "n_eval_seqs": N_EVAL}

    # ---- step 1 ---------------------------------------------------------------
    S["step1_before_mlp"] = {str(j): {"projection": stat(E["pre"], j),
                                      "parts_mean": {"embedding": float(E["emb"][:, j].mean()),
                                                     "BOS_value_x_attention": float(E["bosval"][:, j].mean()),
                                                     "other_tokens_values": float(E["othval"][:, j].mean()),
                                                     "attention_bias": float(E["bias"][:, j].mean())},
                                      "layer1_attention_on_BOS_headmean": float(E["a1bos"][:, j].mean())} for j in pos}
    split_err = float(np.abs(E["pre"] - E["emb"] - E["bosval"] - E["othval"] - E["bias"]).max())
    S["step1_split_max_abs_err"] = split_err
    # ---- step 2 ---------------------------------------------------------------
    late = [j for j in range(64, L)]
    S["step2_mlp"] = {"by_position": {str(j): {"before": stat(E["pre"], j), "correction": stat(E["corr"], j),
                                               "after": stat(E["post"], j)} for j in pos},
                      "late_common_level_after_mean": float(E["post"][:, 64:].mean()),
                      "late_sd_before": float(E["pre"][:, 64:].std(0).mean()), "late_sd_after": float(E["post"][:, 64:].std(0).mean())}
    sl = []
    for j in [16, 64, 256, 1000]:
        if j < L:
            x, y = E["pre"][:, j], E["corr"][:, j]
            c = np.polyfit(x, y, 1)
            sl.append({"j": j, "slope_correction_on_pre": float(c[0]),
                       "r2": float(np.corrcoef(x, y)[0, 1] ** 2)})
    S["step2_mlp"]["per_token_slope_at_fixed_j"] = sl
    S["step2_mlp"]["example_sequence_0"] = {str(j): [float(E["pre"][0, j]), float(E["corr"][0, j]), float(E["post"][0, j])] for j in range(9)}

    # ---- steps 3-5, per BOS head ------------------------------------------------
    sinks = sinks_from(E["share2"])
    a = 1 + len(sinks)
    is_ord = np.ones(L); is_ord[0] = 0; is_ord[sinks] = 0
    nord = np.cumsum(is_ord)
    S["sinks_at_layer2_input"] = sinks
    S["heads"] = {}
    i = np.arange(L)
    for k, h in enumerate(heads):
        r2 = E["r2"][:, k]; Bi = E["Bi"][:, k, 16:]; Bb = E["Bi_bias"][:, k, 16:]; Bw = E["Bi_wq"][:, k, 16:]
        G = float(np.nanmedian(E["gap"][:, k, 16:]))
        am = E["a2bos"][:, k].mean(0)
        pa = np.exp(G) / (a * np.exp(G) + nord)
        y = 1 / am[1:]; X = np.stack([np.ones(L - 1), nord[1:]], 1)
        cf, *_ = np.linalg.lstsq(X, y, rcond=None)
        v = E["v2"][:, k]
        vord = v[:, 16:]
        v0, v1_, vo = float(v[:, 0].mean()), float(v[:, 1].mean()), float(vord.mean())
        zf = E["a2bos"][:, k] * v0 + E["a2pos1"][:, k] * v1_ + (1 - E["a2bos"][:, k] - E["a2pos1"][:, k]) * vo
        zm = E["z2"][:, k]
        res = (zm - zf)[:, 1:]
        S["heads"][f"h{h}"] = {
            "step3_scores": {"r2_scores_on_feature_median_by_bin": {b: float(np.nanmedian(r2[:, lo:hi + 1])) for b, (lo, hi) in bins.items()},
                             "B_mean": float(Bi.mean()), "B_cv_over_queries_and_seqs": float(Bi.std() / abs(Bi.mean())),
                             "B_from_query_bias_mean": float(Bb.mean()), "B_from_Wq_x_mean": float(Bw.mean()),
                             "B_from_Wq_x_sd": float(Bw.std())},
            "step4_counting": {"score_gap_G": G, "a": a, "e^G": float(np.exp(G)),
                               "relerr_measured_vs_formula_by_bin": {b: float(np.median(np.abs(pa - am)[lo:hi + 1] / am[lo:hi + 1])) for b, (lo, hi) in bins.items()},
                               "free_fit_a_prime": float(cf[0]), "free_fit_c": float(cf[1]), "predicted_c": float(np.exp(-G)),
                               "bos_attention_at": {str(j): float(am[j]) for j in [1, 16, 128, 1023] if j < L}},
            "step5_values": {"v_BOS": v0, "v_pos1": v1_, "v_ordinary_mean": vo, "v_ordinary_sd": float(vord.std()),
                             "ordinary_sd_over_gap": float(vord.std() / abs(v0 - vo)),
                             "readout_formula_r2": float(1 - res.var() / zm[:, 1:].var()),
                             "readout_formula_rel_rms": float(np.sqrt((res ** 2).mean()) / abs(v0 - vo))}}
    # ---- step 6: which part fails with one layer --------------------------------
    k0 = 0
    pairs = {"L2 weights + L2 values (model)": ("z2", True), "L1 weights + L1 values": ("z1", False),
             "L2 weights + L1 values": ("z_l2w_l1v", True), "L1 weights + L2 values": ("z_l1w_l2v", True)}
    S["step6_readout_pairs_head"] = f"h{heads[k0]}"
    S["step6_readout_pairs"] = {}
    for name, (key, perhead) in pairs.items():
        zc = C[key][:, k0] if perhead else C[key]
        ze = E[key][:, k0] if perhead else E[key]
        S["step6_readout_pairs"][name] = err_by_bin(isotonic_decode(zc, ze, L), L, bins)
    # Layer-1 values: same value check as step 5
    v1 = E["v1"]
    S["step6_layer1_values"] = {"v_BOS": float(v1[:, 0].mean()), "v_ordinary_mean": float(v1[:, 16:].mean()),
                                "v_ordinary_sd": float(v1[:, 16:].std()),
                                "ordinary_sd_over_gap": float(v1[:, 16:].std() / abs(v1[:, 0].mean() - v1[:, 16:].mean()))}
    (od / "evidence.json").write_text(json.dumps(S, indent=2))

    # ---- figures ----------------------------------------------------------------
    J = np.arange(L)
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
    for key, lab, c in [("pre", "before the MLP", "C0"), ("corr", "MLP correction", "C1"), ("post", "after the MLP", "C2")]:
        m = E[key].mean(0); lo, hi = np.quantile(E[key], .1, 0), np.quantile(E[key], .9, 0)
        ax[0].plot(J, m, color=c, lw=1.4, label=lab); ax[0].fill_between(J, lo, hi, color=c, alpha=.18)
    ax[0].set_xscale("symlog", linthresh=8); ax[0].axhline(0, color="k", lw=.4); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("position j"); ax[0].set_ylabel("projection on BOS direction"); ax[0].set_title("Steps 1-2: Layer-1 residual on the BOS direction\n(line = mean, band = 10-90% over sequences)", fontsize=9)
    jj = np.arange(16, L)
    ax[1].scatter(E["pre"][:, jj].ravel()[::13], E["corr"][:, jj].ravel()[::13], s=1, alpha=.3)
    ax[1].set_xlabel("projection before the MLP"); ax[1].set_ylabel("MLP correction"); ax[1].set_title("Step 2: correction against input, positions ≥ 16", fontsize=9)
    k = 0; h = heads[0]
    am = E["a2bos"][:, k].mean(0); G = S["heads"][f"h{h}"]["step4_counting"]["score_gap_G"]
    ax[2].plot(J[1:], am[1:], lw=1.5, label="measured (mean)")
    ax[2].plot(J[1:], (np.exp(G) / (a * np.exp(G) + nord))[1:], "--", lw=1.2, label=f"e^G/(a e^G + n), a={a}, G={G:.2f}")
    ax[2].plot(J[1:], 1 / (J[1:] + 1), ":", color="0.4", label="1/(i+1)")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].legend(fontsize=7); ax[2].set_xlabel("query position i")
    ax[2].set_title(f"Step 4: Layer-2 attention on BOS (head {h})", fontsize=9)
    for s_ in range(6):
        ax[3].plot(J[1:], E["z2"][s_, k, 1:], lw=.8)
    v0 = S["heads"][f"h{h}"]["step5_values"]["v_BOS"]; vo = S["heads"][f"h{h}"]["step5_values"]["v_ordinary_mean"]
    zf = E["a2bos"][0, k] * v0 + E["a2pos1"][0, k] * S["heads"][f"h{h}"]["step5_values"]["v_pos1"] + (1 - E["a2bos"][0, k] - E["a2pos1"][0, k]) * vo
    ax[3].plot(J[1:], zf[1:], "k--", lw=1, label="formula, sequence 0")
    ax[3].set_xscale("log"); ax[3].legend(fontsize=7); ax[3].set_xlabel("query position i")
    ax[3].set_ylabel("output on the head's BOS-value direction"); ax[3].set_title("Step 5: readout, 6 single sequences", fontsize=9)
    fig.suptitle(f"{args.model}: evidence for the mechanism", fontsize=11)
    fig.tight_layout(); fig.savefig(od / "evidence.png", dpi=130); plt.close(fig)

    # ---- short printout -----------------------------------------------------------
    print(f"== {args.model}  heads {heads}  sinks {sinks}  (split error {split_err:.1e})")
    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 64, 256, 1000]:
        if j < L:
            b = S["step2_mlp"]["by_position"][str(j)]; pt = S["step1_before_mlp"][str(j)]["parts_mean"]
            print(f"  j={j:4d} L1attnBOS={S['step1_before_mlp'][str(j)]['layer1_attention_on_BOS_headmean']:.3f} "
                  f"pre {b['before']['mean']:8.2f}±{b['before']['sd']:6.2f}  corr {b['correction']['mean']:8.2f}  post {b['after']['mean']:8.2f}±{b['after']['sd']:6.2f}"
                  f"  | parts: emb {pt['embedding']:.2f} BOSval {pt['BOS_value_x_attention']:.2f} others {pt['other_tokens_values']:.2f} bias {pt['attention_bias']:.2f}")
    print("  late level after", round(S["step2_mlp"]["late_common_level_after_mean"], 2), "late sd before/after",
          round(S["step2_mlp"]["late_sd_before"], 3), round(S["step2_mlp"]["late_sd_after"], 3), "per-token slopes", [(d_["j"], round(d_["slope_correction_on_pre"], 3), round(d_["r2"], 2)) for d_ in sl])
    for hk, d_ in S["heads"].items():
        print(" ", hk, json.dumps({k2: {k3: (round(v3, 4) if isinstance(v3, float) else v3) for k3, v3 in v2_.items() if not isinstance(v3, dict)} for k2, v2_ in d_.items()}))
        print("     r2 by bin", {b: round(v, 3) for b, v in d_["step3_scores"]["r2_scores_on_feature_median_by_bin"].items()},
              "count relerr", {b: round(v, 3) for b, v in d_["step4_counting"]["relerr_measured_vs_formula_by_bin"].items()})
    print("  step 6 (late or last bin relMAE):", {n_: round(e[f"relmae_{list(bins)[-1]}"], 4) for n_, e in S["step6_readout_pairs"].items()})
    print("  step 6 middle:", {n_: round(e["relmae_middle"], 4) for n_, e in S["step6_readout_pairs"].items()})
    print("  layer-1 values:", {k2: round(v2_, 3) for k2, v2_ in S["step6_layer1_values"].items()})


if __name__ == "__main__":
    main()
