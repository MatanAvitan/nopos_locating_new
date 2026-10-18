"""What the Layer-1 MLP does to the BOS share, position by position.

All BOS amounts are measured in residual space (before the Layer-2
LayerNorm), along one fixed direction v = post1_BOS - mu_post1:
  s_j   = <post1_j - mu_post1, v> / |v|^2       share before the MLP
  c_j   = <m_j - mu_m, v> / |v|^2               BOS added by the MLP (route A)
  s_j + c_j = <h1_j - mu_h1, v> / |v|^2         BOS after the MLP (exact)
The Layer-2 LayerNorm then divides token j by its scale sigma_j. Route B is
the norm effect, (s_j + c_j) * (sigma_0 / sigma_j) - (s_j + c_j). The measured
share beta_j at the Layer-2 input (centered, its own direction) is split as
  beta_j - s_j = c_j + normB_j + rest_j,
where rest_j is what the LayerNorm gain and centering add.

Also: norms (MLP input, MLP output, residual before and after), a heatmap of
the 50 neurons with the largest effect on c_j, and two causal edits: zero the
MLP output (i) at positions 0 and 1 only, (ii) at positions >= 2 only.

Usage: python beta_mlp1.py --model 1h --device cuda:0
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_common import (MODELS, Ridge, attn_from_inputs, bins_for, err_by_bin,
                         ln, make_adapter, out_dir, spec_splits)
from beta_v2 import free_fit, n_ordinary
from common import batch_from_offsets

JS = list(range(17)) + [64, 256, 1000]
N_TOP = 50
ARMS = ["clean", "zero_mlp_pos01", "zero_mlp_pos2on"]


def mlp_parts(ad, p):
    mlp = ad.model.block1.mlp
    hid = mlp.gelu(mlp.c_fc(p["h1bar"]))                  # the model's own activation
    return hid, mlp.c_proj(hid)


@torch.no_grad()
def arm_h1(ad, p, arm, m):
    if arm == "clean":
        return p["post1"] + m
    m = m.clone()
    if arm == "zero_mlp_pos01":
        m[:, :2] = 0
    else:
        m[:, 2:] = 0
    return p["post1"] + m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ad, L, run_id = make_adapter(args.model, dev)
    od = out_dir(run_id, "beta_v2")
    fd = od / "figures"
    J = [j for j in JS if j < L]
    splits = spec_splits(L)
    bins = bins_for(L)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)
    blk2_ln = ad.model.block2.ln_1

    def batches(name):
        offs = splits[name]
        for s in range(0, len(offs), args.batch_size):
            yield batch_from_offsets(offs[s:s + args.batch_size], L, dev)

    # ---- reference: means per stage and per arm ----------------------------
    sums = {k: torch.zeros(ad.d, device=dev, dtype=torch.float64) for k in
            ["post1", "m"] + [f"x2_{a}" for a in ARMS]}
    bos = {}
    n = 0
    for idx in batches("reference"):
        p = ad.pre(idx)
        hid, m = mlp_parts(ad, p)
        st = {"post1": p["post1"], "m": m}
        for a in ARMS:
            st[f"x2_{a}"] = ln(arm_h1(ad, p, a, m), blk2_ln)
        for k, v in st.items():
            v = v.double()
            bos.setdefault(k, v[0, 0].clone())
            sums[k] += v[:, 1:].sum((0, 1))
        n += idx.shape[0] * (L - 1)
    mu = {k: v / n for k, v in sums.items()}
    vp = bos["post1"] - mu["post1"]
    W_out = ad.model.block1.mlp.c_proj.weight.double()               # [d, 4d]
    neuron_dir = (W_out.T @ vp) / (vp @ vp)                          # BOS per unit activation

    # ---- evaluation: per-position statistics (clean) ----------------------
    rec = {k: [] for k in ["s", "c", "sigma", "beta", "n_in", "n_out", "n_pre", "n_post"]}
    hid_mean = torch.zeros(len(J), W_out.shape[1], device=dev, dtype=torch.float64)
    neff = torch.zeros_like(hid_mean)
    nseq = 0
    for idx in batches("evaluation"):
        p = ad.pre(idx)
        hid, m = mlp_parts(ad, p)
        post1, h1 = p["post1"].double(), (p["post1"] + m).double()
        md = m.double()
        rec["s"].append((((post1 - mu["post1"]) @ vp) / (vp @ vp))[:, J].cpu())
        rec["c"].append((((md - mu["m"]) @ vp) / (vp @ vp))[:, J].cpu())
        rec["sigma"].append(h1.std(-1, unbiased=False)[:, J].cpu())
        x2 = ln(h1.float(), blk2_ln).double()
        v2 = bos["x2_clean"] - mu["x2_clean"]
        rec["beta"].append((((x2 - mu["x2_clean"]) @ v2) / (v2 @ v2))[:, J].cpu())
        rec["n_in"].append(p["h1bar"].double().norm(dim=-1)[:, J].cpu())
        rec["n_out"].append(md.norm(dim=-1)[:, J].cpu())
        rec["n_pre"].append(post1.norm(dim=-1)[:, J].cpu())
        rec["n_post"].append(h1.norm(dim=-1)[:, J].cpu())
        hd = hid.double()[:, J]                                      # [B,|J|,4d]
        hid_mean += hd.sum(0)
        neff += (hd * neuron_dir).sum(0)                             # contribution to c_j
        nseq += idx.shape[0]
    R = {k: torch.cat(v).numpy() for k, v in rec.items()}
    hid_mean = (hid_mean / nseq).cpu().numpy(); neff = (neff / nseq).cpu().numpy()
    s, c, sig, beta = R["s"], R["c"], R["sigma"], R["beta"]
    bosamt = s + c
    normB = bosamt * (sig[:, :1] / sig) - bosamt
    rest = beta - s - c - normB
    ms = lambda X: {str(j): [float(X[:, k].mean()), float(X[:, k].std())] for k, j in enumerate(J)}
    S = {"model": args.model, "J": J,
         "norms_mean_sd": {"mlp_input": ms(R["n_in"]), "mlp_output": ms(R["n_out"]),
                           "residual_before_mlp": ms(R["n_pre"]), "residual_after_mlp": ms(R["n_post"])},
         "share_before_mlp_s": ms(s), "bos_added_by_mlp_c": ms(c), "ln_scale_sigma": ms(sig),
         "share_at_layer2_input_beta": ms(beta),
         "decomposition_mean": {str(j): {"beta_minus_s": float((beta - s)[:, k].mean()),
                                         "A_mlp_bos_component": float(c[:, k].mean()),
                                         "B_norm_effect": float(normB[:, k].mean()),
                                         "rest_ln_gain_centering": float(rest[:, k].mean())}
                                for k, j in enumerate(J)}}

    # neuron heatmap: 50 neurons with the largest effect on c_j
    top = np.argsort(-np.abs(neff).max(0))[:N_TOP]
    onset = np.argmax(np.abs(neff[:, top]), 0)
    order = top[np.lexsort((-np.abs(neff[:, top]).max(0), onset))]
    S["top_neurons"] = [{"neuron": int(k), "peak_j": int(J[int(np.argmax(np.abs(neff[:, k])))]),
                         "max_abs_effect_on_c": float(np.abs(neff[:, k]).max())} for k in order]

    # ---- causal: zero the MLP output at positions 0-1, or at >= 2 ---------
    ridges = {a: Ridge(ad.d, dev) for a in ARMS}
    for idx in batches("calibration"):
        p = ad.pre(idx)
        _, m = mlp_parts(ad, p)
        for a in ARMS:
            h1 = arm_h1(ad, p, a, m)
            x2 = ln(h1, blk2_ln)
            _, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, _, _ = ad.finish(h1, o2)
            ridges[a].add(feats, tgt.expand(idx.shape[0], L))
    for r in ridges.values():
        r.solve()
    out = {a: {"beta": [], "frozen": [], "refit": [], "ab": torch.zeros(L, dtype=torch.float64)} for a in ARMS}
    for idx in batches("evaluation"):
        p = ad.pre(idx)
        _, m = mlp_parts(ad, p)
        for a in ARMS:
            h1 = arm_h1(ad, p, a, m)
            x2 = ln(h1, blk2_ln)
            w, o2 = attn_from_inputs(ad.attn2, x2, x2, x2)
            feats, fr, _ = ad.finish(h1, o2)
            v2 = bos[f"x2_{a}"] - mu[f"x2_{a}"]
            out[a]["beta"].append((((x2.double() - mu[f"x2_{a}"]) @ v2) / (v2 @ v2)).cpu())
            out[a]["frozen"].append(fr.cpu()); out[a]["refit"].append(ridges[a].predict(feats).cpu())
            out[a]["ab"] += w[:, 0, :, 0].double().sum(0).cpu()
    S["causal"] = {}
    i = np.arange(L)
    for a in ARMS:
        B_ = torch.cat(out[a]["beta"]).numpy()
        frac = (B_[:, 1:] > 0.5).mean(0)
        sinks = [int(j) + 1 for j in np.nonzero(frac > 0.9)[0]]
        ns = np.ones(L, bool); ns[:16] = False; ns[sinks] = False
        am = (out[a]["ab"] / len(B_)).numpy()
        a_f, c_f = free_fit(am, n_ordinary(L, [1]))
        S["causal"][a] = {"beta_profile": {str(j): float(B_[:, j].mean()) for j in J},
                          "beta_sd": {str(j): float(B_[:, j].std()) for j in J},
                          "sinks": sinks, "a": 1 + len(sinks), "beta_bar": float(B_[:, ns].mean()),
                          "tail_sd_j16on": float(B_[:, 16:].std()),
                          "free_fit_alpha_bos": {"a_prime": a_f, "c": c_f},
                          "frozen": err_by_bin(torch.cat(out[a]["frozen"]).double().numpy(), L, bins),
                          "refit": err_by_bin(torch.cat(out[a]["refit"]).numpy(), L, bins)}
    (od / "mlp1.json").write_text(json.dumps(S, indent=2))

    # ---- figures -----------------------------------------------------------
    x = np.arange(len(J))
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    for k, lab in [("n_in", "MLP input"), ("n_out", "MLP output"), ("n_pre", "residual before MLP"), ("n_post", "residual after MLP")]:
        ax[0].errorbar(x, R[k].mean(0), R[k].std(0), label=lab, capsize=2, marker="o", ms=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels(J, fontsize=7); ax[0].set_yscale("log"); ax[0].legend(fontsize=7)
    ax[0].set_title("norms by position j")
    for arr, lab in [(s, "s_j (before MLP)"), (c, "c_j: BOS added by MLP (A)"), (normB, "norm effect (B)"),
                     (rest, "LN gain/centering"), (beta, r"$\beta_j$ at Layer-2 input")]:
        ax[1].plot(x, arr.mean(0), marker="o", ms=3, label=lab)
    ax[1].axhline(0, color="k", lw=.4); ax[1].set_xticks(x); ax[1].set_xticklabels(J, fontsize=7); ax[1].legend(fontsize=7)
    ax[1].set_title(r"$\beta_j - s_j = A + B + $ rest")
    im = ax[2].imshow(neff[:, order].T, aspect="auto", cmap="RdBu_r",
                      vmin=-np.abs(neff[:, order]).max(), vmax=np.abs(neff[:, order]).max())
    ax[2].set_xticks(x); ax[2].set_xticklabels(J, fontsize=7); ax[2].set_ylabel("neuron (sorted by peak position)")
    ax[2].set_title("top-50 neurons: contribution to c_j (activation × BOS weight)")
    plt.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(fd / "mlp1_positions.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ARMS:
        prof = [S["causal"][a]["beta_profile"][str(j)] for j in J]
        ax.plot(x, prof, marker="o", ms=3, label=f"{a}: a={S['causal'][a]['a']}")
    ax.set_xticks(x); ax.set_xticklabels(J, fontsize=7); ax.axhline(0, color="k", lw=.4); ax.legend(fontsize=7)
    ax.set_title(r"causal: $\beta_j$ profile when the Layer-1 MLP output is zeroed")
    fig.tight_layout(); fig.savefig(fd / "mlp1_causal.png", dpi=120); plt.close(fig)

    pr = lambda d_, ks=(0, 1, 2, 3, 4, 6, 8, 16, 64, 1000): {k: round(v, 4) for k, v in d_.items() if int(k) in ks}
    print("norms (mean):", {k: pr({j: v[0] for j, v in S["norms_mean_sd"][k].items()}) for k in S["norms_mean_sd"]})
    print("s:", pr({j: v[0] for j, v in S["share_before_mlp_s"].items()}))
    print("c:", pr({j: v[0] for j, v in S["bos_added_by_mlp_c"].items()}))
    print("sigma:", pr({j: v[0] for j, v in S["ln_scale_sigma"].items()}))
    print("beta:", pr({j: v[0] for j, v in S["share_at_layer2_input_beta"].items()}))
    for j in ["1", "2", "3", "4", "8", "16", "1000"]:
        if j in S["decomposition_mean"]:
            print(" decomp j=" + j, {k: round(v, 4) for k, v in S["decomposition_mean"][j].items()})
    for a in ARMS:
        cz = S["causal"][a]
        print(a, "a=", cz["a"], "sinks", cz["sinks"][:5], "beta_bar=%.4f tail_sd=%.4f" % (cz["beta_bar"], cz["tail_sd_j16on"]),
              "a'=%.2f" % cz["free_fit_alpha_bos"]["a_prime"],
              "frozen late=%.4f refit late=%.4f" % (cz["frozen"].get("relmae_late", np.nan), cz["refit"].get("relmae_late", np.nan)),
              "profile", pr(cz["beta_profile"]))


if __name__ == "__main__":
    main()
