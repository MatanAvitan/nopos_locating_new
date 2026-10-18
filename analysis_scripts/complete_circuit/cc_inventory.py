"""Task A: manifest and exact bookkeeping checks for every checkpoint in SPECS.

For each model this records the architecture, checkpoint hash, trained versus
frozen modules (distance of every weight from its seeded init), positional
parameters, BOS handling, and the earlier artifacts. On 4 calibration
sequences it then checks, in fp64 against the fp32 forward:

  1. manual forward == the checkpoint's own forward
  2. an identity edit at every stage changes nothing
  3. attention update == sum_h sum_j a_hij B_OV,h x_j + b_attn, per layer
  4. score identity s_i0 - s_ij = (1 - beta_j) B_i - E_ij, per layer and head,
     for two centers m (attention-input LN bias; batch nonBOS mean)
  5. value identity v_j = c + lambda beta_j + r_j and xi_j orthogonal to d
  6. block-0 projection split (Section 1.2): direct + BOS + nonBOS values
  7. block-0 attention: across-sequence SD of the weights, BOS weight at i = 1..5
  8. spread of x_0 across sequences at every attention input

Usage: python cc_inventory.py --device cuda:0 [--keys 1h_s42 lm ...]
"""

import argparse
import os
import time
import traceback

import numpy as np
import torch

from cc_common import (OUT_ROOT, SPECS, Capture, git_state, jdump, load_net, sha256_full,
                       splits_for, file_hash)
from common import RESULTS_ROOT, load_model

PRIOR_RUN = {
    "1h_s42": "attn2_full_L1024", "1h_s3": "seeds/1h_s3", "1h_sw_s1": "sweep/L1024_s1", "1h_sw_s2": "sweep/L1024_s2",
    "12h_s42": "full12h_L1024", "12h_s1": "seeds/12h_s1", "12h_s2": "seeds/12h_s2", "12h_s3": "seeds/12h_s3",
    "lm": "lm6_L128_nobos", "lm_bosin": "lm6_L128",
    "lmbos_s42": "seeds/lmbos_s42", "lmbos_s1": "seeds/lmbos_s1", "lmbos_s2": "seeds/lmbos_s2", "lmbos_s3": "seeds/lmbos_s3",
    "unifL1_s42": "sweep/L1024_unifL1_s42", "unifL1_s1": "sweep/L1024_unifL1_s1",
    "noMLP1_s42": "sweep/L1024_noMLP1_s42", "noMLP1_s1": "sweep/L1024_noMLP1_s1",
    "relu_s42": "sweep/L1024_relu_s42", "relu_s1": "sweep/L1024_relu_s1",
    "attn1_s42": "h1_1layer_s42", "attn1_s1": "h1_1layer_s1", "attn2_1h": "attn2_1h_L1024",
}

POS_KEYS = ("wpe", "rope", "rotary", "alibi", "pos_emb", "position_emb")


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def frozen_report(net, key):
    """Relative distance of each parameter from the seeded initialization."""
    spec = net.spec
    if spec.kind != "toy":
        return {"method": "not available for LM checkpoints (init not reproducible here)"}
    init, _ = load_model(spec.ckpt, "cpu", init_only=True)
    tr = dict(net.model.named_parameters())
    out = {}
    for n, p0 in init.named_parameters():
        if n.startswith("mlp_probe"):
            continue
        p = tr[n].detach().cpu().double()
        p0 = p0.detach().double()
        d = float((p - p0).norm())
        out[n] = {"rel_change": d / max(float(p0.norm()), 1e-30), "abs_change": d, "init_norm": float(p0.norm())}
    groups = {}
    for n, v in out.items():
        g = n.split(".")[0] if not n.startswith("block") else ".".join(n.split(".")[:2])
        groups.setdefault(g, []).append(v["rel_change"])
    return {"method": "||W - W_init|| / ||W_init|| with the checkpoint's seed",
            "groups_max_rel_change": {g: max(v) for g, v in groups.items()},
            "groups_frozen": {g: max(v) == 0.0 for g, v in groups.items()},
            "params": out}


@torch.no_grad()
def checks(net, idx):
    R = {}
    L = idx.shape[1]
    nat = net.native(idx)
    cap = Capture()
    man = net.forward(idx, cap)
    R["manual_vs_native_max_rel"] = rel(man, nat)
    R["manual_vs_native_max_abs"] = float((man.double() - nat.double()).abs().max())
    stages = {s for (s, _) in cap.store}
    ident = Capture(want=set(), edits={(s, l): (lambda t: t.clone()) for (s, l) in cap.store})
    man2 = net.forward(idx, ident)
    R["identity_edit_max_abs"] = float((man2.double() - man.double()).abs().max())
    if net.spec.kind == "lm":
        R["lm_loss_mean"] = float(net.lm_loss(idx, man).mean())
    else:
        i = torch.arange(L, device=idx.device, dtype=torch.float64)
        R["pos_mae_batch"] = float((man.double() - i).abs()[:, 1:].mean())
    R["stages"] = sorted(stages)
    per = []
    for l in range(net.n_layer):
        S = {"layer": l}
        x = cap.store[("x_attn", l)].double()
        A = cap.store[("weights", l)].double()
        s32 = cap.store[("scores", l)].double()
        M = net.ov(l)
        Y = torch.einsum("hdf,btf->bhtd", M, x)
        o64 = torch.einsum("bhts,bhsd->btd", A, Y) + net.attn_bias(l)
        o = cap.store[("attn_out", l)].double()
        S["attn_out_recon_max_rel"] = rel(o64, o)
        S["resid_mid_identity_max_abs"] = float((cap.store[("resid_mid", l)].double()
                                                 - cap.store[("resid_in", l)].double() - o).abs().max())
        if ("mlp_out", l) in cap.store:
            S["resid_out_identity_max_abs"] = float((cap.store[("resid_out", l)].double()
                                                     - cap.store[("resid_mid", l)].double()
                                                     - cap.store[("mlp_out", l)].double()).abs().max())
        x0 = x[:, 0]
        S["x0_spread_rel"] = float((x0 - x0.mean(0)).norm(dim=-1).max() / x0.norm(dim=-1).mean())
        hw = net.hw[l]
        q = torch.einsum("btd,hed->bhte", x, hw.W_q.double()) + hw.b_q.double()[None, :, None]
        ln_b = net.ln_params(net.blocks[l].ln_1)[1]
        centers = {"ln_bias": ln_b.expand_as(x[:, 0]), "batch_nonbos_mean": x[:, 1:].mean((0, 1)).expand_as(x[:, 0])}
        S["score_identity"], S["value_identity"] = {}, {}
        for cname, m in centers.items():
            d = x0 - m                                               # [B,d] per sequence
            beta = torch.einsum("btd,bd->bt", x - m[:, None], d) / (d * d).sum(-1, keepdim=True)
            xi = x - m[:, None] - beta[..., None] * d[:, None]
            orth = float((torch.einsum("btd,bd->bt", xi, d).abs() / (xi.norm(dim=-1) * d.norm(dim=-1)[:, None] + 1e-30)).max())
            kd = torch.einsum("hed,bd->bhe", hw.W_k.double(), d)
            Bi = torch.einsum("bhte,bhe->bht", q, kd) * hw.scale
            Eij = torch.einsum("bhte,bhse->bhts", q, torch.einsum("hed,bsd->bhse", hw.W_k.double(), xi)) * hw.scale
            lhs = s32[..., :1] - s32
            rhs = (1 - beta)[:, None, None, :] * Bi[..., None] - Eij
            tri = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            err = (lhs - rhs)[..., tri]
            scale = lhs[..., tri].std()
            S["score_identity"][cname] = {"max_abs_err": float(err.abs().max()), "score_diff_sd": float(scale),
                                          "beta0_max_dev": float((beta[:, 0] - 1).abs().max()),
                                          "xi_perp_d_max_cos": orth}
            vals = {}
            for h in range(net.H):
                wb = Y[:, h, 0]                                    # [B,d]
                u = wb / wb.norm(dim=-1, keepdim=True)
                v = torch.einsum("btd,bd->bt", Y[:, h], u)
                c = torch.einsum("bd,bd->b", torch.einsum("de,be->bd", M[h], m), u)
                lam = torch.einsum("bd,bd->b", torch.einsum("de,be->bd", M[h], d), u)
                r = torch.einsum("btd,bd->bt", torch.einsum("de,bte->btd", M[h], xi), u)
                vals[h] = float((v - (c[:, None] + lam[:, None] * beta + r)).abs().max() / v.abs().max())
            S["value_identity"][cname] = {"max_rel_err_over_heads": max(vals.values())}
        if l == 0:
            h0 = cap.store[("resid_in", 0)].double()
            ht = cap.store[("resid_mid", 0)].double()
            ba = net.attn_bias(0)
            split = {}
            for h in range(net.H):
                w = Y[0, h, 0]
                nn2 = w @ w
                lhs = ((ht - ba) @ w) / nn2
                direct = (h0 @ w) / nn2
                own = A[:, h, :, 0] * ((Y[:, h, 0] @ w) / nn2)[:, None]         # BOS term (=alpha if x0 fixed)
                nonb = (A[:, h, :, 1:] * (torch.einsum("bsd,d->bs", Y[:, h, 1:], w) / nn2)[:, None]).sum(-1)
                other = sum((A[:, g] * (torch.einsum("bsd,d->bs", Y[:, g], w) / nn2)[:, None]).sum(-1)
                            for g in range(net.H) if g != h) if net.H > 1 else 0 * lhs
                split[h] = float((lhs - (direct + own + nonb + other)).abs().max() / lhs.abs().max())
            S["block0_projection_split_max_rel_err"] = max(split.values())
            sd = A.std(0)                                                     # across sequences
            tri = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            S["block0_attention"] = {
                f"h{h}": {"max_across_seq_sd": float(sd[h][tri].max()),
                          "bos_weight_i1_to_5": [float(A[:, h, i, 0].mean()) for i in range(1, 6)],
                          "uniform_reference": [1 / (i + 1) for i in range(1, 6)]}
                for h in range(net.H)}
        per.append(S)
    R["layers"] = per
    return R


def manifest(net, path, key):
    cfg = {k: v for k, v in net.cfg.items() if not k.startswith("_")}
    sd_keys = list(net.model.state_dict().keys())
    pos = [k for k in sd_keys if any(p in k.lower() for p in POS_KEYS)]
    st = os.stat(path)
    spec = net.spec
    M = {"key": key, "family": spec.family, "role": spec.role, "note": spec.note,
         "checkpoint": str(path), "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
         "sha256": sha256_full(path), "iter_num": net.meta.get("iter_num"), "best_val_loss": net.meta.get("best_val_loss"),
         "n_layer": net.n_layer, "n_head": net.H, "d_model": net.d, "d_head": net.dh, "L_train": net.L_train,
         "L_eval": spec.L, "bias": bool(net.blocks[0].attn.c_attn.bias is not None),
         "ln_bias": bool(net.blocks[0].ln_1.bias is not None), "norm": cfg.get("norm_type", "layernorm"),
         "activation": type(net.blocks[0].mlp.gelu).__name__, "positional_params": pos,
         "eval_input_bos": spec.bos, "config": cfg}
    if spec.kind == "toy":
        M["train_bos"] = bool(cfg.get("use_bos", True))
        M["regime"] = cfg.get("regime")
        M["block0_uniform"] = bool(cfg.get("l1_uniform", False))
        M["block0_no_mlp"] = bool(cfg.get("l1_no_mlp", False))
        M["output"] = "pos_head(ln_f(h)) * (L_train - 1), linear regression head"
    else:
        tr = net.cfg.get("_train", {})
        M["train_flags"] = {k: tr.get(k) for k in ["train_lm_only", "use_positional_embedding", "dataset", "block_size",
                                                   "freeze_transformer", "freeze_until_first_mlp", "use_bos", "bos"]
                            if k in tr}
        M["lm_head_tied"] = bool(net.model.lm_head.weight.data_ptr() == net.wte.weight.data_ptr())
        M["output"] = "lm_head(ln_f(h)) logits; next-token CE"
    prior = RESULTS_ROOT / PRIOR_RUN.get(key, "_none_")
    M["prior_artifacts"] = sorted(str(p.relative_to(RESULTS_ROOT)) for p in prior.rglob("*.json")) if prior.exists() else []
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keys", nargs="*", default=list(SPECS))
    ap.add_argument("--n-seq", type=int, default=4)
    args = ap.parse_args()
    out = OUT_ROOT / "inventory.json"
    inv = __import__("json").loads(out.read_text()) if out.exists() else {"models": {}, "errors": {}}
    inv["git"] = git_state()
    inv["code_hash"] = {f: file_hash(__file__.replace("cc_inventory.py", f)) for f in ["cc_common.py", "cc_inventory.py"]}
    inv["torch"] = torch.__version__
    inv.setdefault("splits", {})
    for key in args.keys:
        t0 = time.time()
        try:
            net, path = load_net(key, args.device)
            sp, prov = splits_for(key)
            inv["splits"][f"L{net.spec.L}_{'bos' if net.spec.bos else 'raw'}"] = prov
            M = manifest(net, path, key)
            M["frozen"] = frozen_report(net, key)
            idx = net.batch(sp["calibration"][:args.n_seq])
            M["first_token_ids"] = idx[:, 0].tolist()
            M["checks"] = checks(net, idx)
            inv["models"][key] = M
            inv["errors"].pop(key, None)
            c = M["checks"]
            print(f"{key}: manual/native rel {c['manual_vs_native_max_rel']:.2e} identity {c['identity_edit_max_abs']:.1e} "
                  f"| frozen {M['frozen'].get('groups_frozen')} | {time.time() - t0:.0f}s", flush=True)
            for S in c["layers"]:
                si = S["score_identity"]["ln_bias"]
                print(f"   L{S['layer']}: attn recon {S['attn_out_recon_max_rel']:.1e} score id err {si['max_abs_err']:.1e}"
                      f" (sd {si['score_diff_sd']:.2f}) value id {S['value_identity']['ln_bias']['max_rel_err_over_heads']:.1e}"
                      f" x0 spread {S['x0_spread_rel']:.1e}", flush=True)
            del net
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            inv["errors"][key] = {"error": repr(e), "trace": traceback.format_exc()}
            print(f"{key}: ERROR {e!r}", flush=True)
        jdump(inv, out)


if __name__ == "__main__":
    main()
