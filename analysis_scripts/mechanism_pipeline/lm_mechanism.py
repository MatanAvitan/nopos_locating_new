"""Does the BOS-referenced two-direction mechanism emerge in a trained NoPE LM?

Model: nanoGPT/out-lm-6layer-fulltrain-ddp/ckpt.pt — 6-layer, 12-head NoPE
transformer trained purely on next-token prediction (train_lm_only=True,
OpenWebText, block 128). No position head exists, so position decoding is
measured with cross-fitted linear probes (calibration split -> evaluation
split), and the mechanism diagnostics mirror the canonical pipeline per layer
and head.

Two input conditions:
  --bos      : prepend token 50256 (mechanism premise; <|endoftext|> was seen
               in training as a document separator)
  --no-bos   : raw text chunks (BOS control per exps.md)

Outputs results/mechanism/<run_id>/summary.json with per-layer, per-head:
  step-1 uniformity, step-3 alpha/cbar stats, step-2 endpoint separation,
  step-4 residual e + cos(g,w), step-5.1 affine coordinate R^2,
  headwise ideal-mixture reconstruction, and block-wise position probes.

Usage:
  python lm_mechanism.py --ckpt nanoGPT/out-lm-6layer-fulltrain-ddp/ckpt.pt \
      --run-id lm6_L128 --context-length 128 [--no-bos]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, RESULTS_ROOT, VAL_BIN, BOS_TOKEN_ID,
                    add_nanogpt_to_path, attn_scores_weights, build_splits,
                    git_commit, head_weights, ov_images, run_dir, sha256_file,
                    update_summary)
from stats import fit_range, loglog_slope


def load_lm(ckpt_path: str, device: str):
    add_nanogpt_to_path()
    from model_position_classifier import (GPTPositionClassifier,
                                           GPTPositionClassifierConfig)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    margs = dict(ck["model_args"])
    margs["mlp_expansion_ratio"] = margs.get("mlp_expansion_ratio", 4)
    cfg = GPTPositionClassifierConfig(**{
        k: v for k, v in margs.items()
        if k in GPTPositionClassifierConfig.__dataclass_fields__})
    model = GPTPositionClassifier(cfg)
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing[:8], "unexpected:", unexpected[:8])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ck


def batch_from_offsets(offsets, L, device, use_bos):
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    seqs = []
    for off in offsets:
        if use_bos:
            toks = data[off: off + L - 1].astype(np.int64)
            seqs.append(np.concatenate([[BOS_TOKEN_ID], toks]))
        else:
            seqs.append(data[off: off + L].astype(np.int64))
    return torch.from_numpy(np.stack(seqs)).to(device)


@torch.no_grad()
def forward_blocks(model, idx):
    """Manual pre-norm forward capturing per-block states and attention.

    Returns list of dicts per block: x_in (attn input after ln_1), scores,
    weights, y (OV images), o (attention update incl c_proj bias), resid_out
    (block output), plus final embedding table output.
    """
    x = model.transformer.wte(idx) if hasattr(model, "transformer") else None
    if x is None:
        x = model.wte(idx)
    blocks = model.transformer.h if hasattr(model, "transformer") else model.blocks
    caps = []
    for blk in blocks:
        ln1 = F.layer_norm(x, blk.ln_1.weight.shape, blk.ln_1.weight,
                           blk.ln_1.bias, 1e-5)
        hw = head_weights(blk.attn)
        scores, weights = attn_scores_weights(ln1, hw)
        y = ov_images(ln1, hw)                       # [B,H,T,d]
        o = torch.einsum("bhts,bhsd->bhtd", weights, y).sum(1)
        bo = blk.attn.c_proj.bias
        bv = hw.b_v
        per_head_bias = torch.einsum("he,hde->hd", bv.float(), hw.W_o.float()).sum(0)
        if bo is not None:
            per_head_bias = per_head_bias + bo.float()
        o = o + per_head_bias[None, None, :]
        x_attn = x + o
        if getattr(blk, "use_ln2", True):
            ln2 = F.layer_norm(x_attn, blk.ln_2.weight.shape, blk.ln_2.weight,
                               blk.ln_2.bias, 1e-5)
            x = x_attn + blk.mlp(ln2)
        else:
            x = x_attn + blk.mlp(x_attn)
        caps.append({"ln1": ln1, "scores": scores, "weights": weights,
                     "y": y, "o": o, "resid_out": x})
    return caps


def compute_log_cbar(scores):
    s = scores.double()
    rel = (s - s[..., 0:1])[..., 1:]
    lse = torch.logsumexp(rel, dim=-1)
    T = s.shape[-2]
    i_idx = torch.arange(T, device=s.device, dtype=torch.float64)
    logc = lse - torch.log(i_idx.clamp(min=1))
    logc[..., 0] = float("nan")
    return logc


class Ridge:
    def __init__(self, dim, device, lam=1e-4):
        self.A = torch.zeros(dim + 1, dim + 1, device=device, dtype=torch.float64)
        self.b = torch.zeros(dim + 1, device=device, dtype=torch.float64)
        self.lam = lam

    def add(self, X, y):
        X1 = torch.cat([X.double().reshape(-1, X.shape[-1]),
                        torch.ones(X.numel() // X.shape[-1], 1,
                                   device=X.device, dtype=torch.float64)], 1)
        self.A += X1.T @ X1
        self.b += X1.T @ y.double().reshape(-1)

    def solve(self):
        n = self.A.shape[0]
        reg = self.lam * torch.diag(self.A).mean() * torch.eye(
            n, device=self.A.device, dtype=torch.float64)
        reg[-1, -1] = 0
        self.w = torch.linalg.solve(self.A + reg, self.b)

    def predict(self, X):
        X1 = torch.cat([X.double(), torch.ones(*X.shape[:-1], 1,
                        device=X.device, dtype=torch.float64)], -1)
        return X1 @ self.w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nanoGPT/out-lm-6layer-fulltrain-ddp/ckpt.pt")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--context-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bos", action="store_true")
    ap.add_argument("--n-ref", type=int, default=640)
    ap.add_argument("--n-cal", type=int, default=320)
    ap.add_argument("--n-eval", type=int, default=1600)
    args = ap.parse_args()
    L = args.context_length
    device = args.device
    use_bos = not args.no_bos
    out = run_dir(args.run_id)
    ckpt_path = str(REPO_ROOT / args.ckpt)
    model, cfg, ck = load_lm(ckpt_path, device)
    n_layer = cfg.n_layer
    H = cfg.n_head
    d = cfg.n_embd

    splits = build_splits(L, seed=1234, sizes={
        "reference": args.n_ref, "calibration": args.n_cal,
        "evaluation": args.n_eval})
    np.savez(out / "sequence_ids.npz", **splits)
    (out / "config.json").write_text(json.dumps({
        "model": "nope_lm", "checkpoint_path": ckpt_path,
        "checkpoint_hash": sha256_file(ckpt_path), "iter_num": ck.get("iter_num"),
        "n_layer": n_layer, "n_head": H, "n_embd": d,
        "context_length": L, "use_bos": use_bos,
        "splits": {k: len(v) for k, v in splits.items()},
        "git_commit": git_commit(), "run_command": " ".join(sys.argv),
        "trained_on": "openwebtext LM (train_lm_only)",
    }, indent=2))

    def iter_split(name, bs=None):
        offs = splits[name]
        bs = bs or args.batch_size
        for s in range(0, len(offs), bs):
            yield batch_from_offsets(offs[s:s + bs], L, device, use_bos)

    # ---------------- Phase A: reference vectors ------------------------
    print("reference pass...")
    sum_y = torch.zeros(n_layer, H, d, device=device, dtype=torch.float64)
    cnt = 0
    x0_l = [None] * n_layer
    for idx in iter_split("reference"):
        caps = forward_blocks(model, idx)
        for l, c in enumerate(caps):
            sum_y[l] += c["y"][:, :, 1:].double().sum((0, 2))
            if x0_l[l] is None and use_bos:
                x0_l[l] = c["ln1"][0, 0].double().clone()
        cnt += idx.shape[0] * (L - 1)
        del caps
    w_nb = (sum_y / cnt)                                       # [n_layer,H,d]
    w_b = torch.zeros_like(w_nb)
    if use_bos:
        for l in range(n_layer):
            blk = (model.transformer.h if hasattr(model, "transformer")
                   else model.blocks)[l]
            hw = head_weights(blk.attn)
            v0 = torch.einsum("hed,d->he", hw.W_v.double(), x0_l[l])
            w_b[l] = torch.einsum("hde,he->hd", hw.W_o.double(), v0)
    torch.save({"w_nonbos": w_nb.cpu(), "w_bos": w_b.cpu()}, out / "ref_stats.pt")

    # ---------------- probes: fit on calibration ------------------------
    print("calibration pass (probes)...")
    ridges = [Ridge(d, device) for _ in range(n_layer + 1)]  # +1 for embeddings? use blocks only
    tgt_row = torch.arange(L, dtype=torch.float32, device=device)
    for idx in iter_split("calibration"):
        caps = forward_blocks(model, idx)
        y = tgt_row.expand(idx.shape[0], L)
        for l, c in enumerate(caps):
            ridges[l].add(c["resid_out"], y)
        del caps
    for l in range(n_layer):
        ridges[l].solve()

    # ---------------- evaluation pass ------------------------------------
    print("evaluation pass...")
    acc = {}

    def push(k, t):
        acc.setdefault(k, []).append(t.detach().cpu().float().numpy())

    dw = w_nb - w_b
    dwn2 = dw.pow(2).sum(-1)                                   # [n_layer,H]
    w_nb_norm = w_nb.norm(dim=-1)
    i_idx = torch.arange(L, device=device, dtype=torch.float64)

    for idx in iter_split("evaluation"):
        caps = forward_blocks(model, idx)
        B = idx.shape[0]
        for l, c in enumerate(caps):
            a = c["weights"].double()                          # [B,H,T,T]
            alpha = a[..., 0]                                  # [B,H,T]
            push(f"L{l}/alpha_bos", alpha)
            logc = compute_log_cbar(c["scores"])
            push(f"L{l}/log_cbar", logc)
            # uniformity
            mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
            ac = a.clamp_min(1e-30) * mask
            ent = -(ac * ac.log().where(mask, torch.zeros_like(ac))).sum(-1)
            push(f"L{l}/norm_entropy", ent / torch.log(i_idx + 1).clamp(min=1e-9))
            # step 4 per head
            m = (1 - alpha).clamp_min(1e-30)
            q = a[..., 1:].float() / m.float()[..., None]
            g = torch.einsum("bhts,bhsd->bhtd", q, c["y"][:, :, 1:]).double()
            e = (g - w_nb[l][None, :, None, :]).norm(dim=-1) / \
                w_nb_norm[l][None, :, None]
            push(f"L{l}/e", e)
            push(f"L{l}/cos_g_w", (g * w_nb[l][None, :, None, :]).sum(-1)
                 / (g.norm(dim=-1) * w_nb_norm[l][None, :, None]).clamp_min(1e-30))
            # step 5.1 affine coordinate per head: project realized per-head
            # output (alpha*w_b + m*g) on dw
            o_head = alpha[..., None] * w_b[l][None, :, None, :] \
                + m[..., None] * g
            Y = torch.einsum("bhtd,hd->bht", o_head, dw[l])
            push(f"L{l}/Y_proj", Y)
            # probe predictions
            push(f"L{l}/probe_pred", ridges[l].predict(c["resid_out"]))
        del caps

    arrs = {k: np.concatenate(v, 0) for k, v in acc.items()}
    for k, v in arrs.items():
        np.save(out / (k.replace("/", "__") + ".npy"), v)

    # ---------------- summarize -----------------------------------------
    lo, hi = fit_range(L)
    pos_fit = np.arange(lo, hi + 1)
    pos = np.arange(1, L)
    summary = {"per_layer": {}}
    for l in range(n_layer):
        alpha = arrs[f"L{l}/alpha_bos"]
        logc = arrs[f"L{l}/log_cbar"]
        e = arrs[f"L{l}/e"]
        cosg = arrs[f"L{l}/cos_g_w"]
        Y = arrs[f"L{l}/Y_proj"]
        nent = arrs[f"L{l}/norm_entropy"]
        heads = {}
        for h in range(H):
            ah = alpha[:, h]
            t = np.arange(L)
            r_h = np.nanmean(ah[:, 1:]) / np.nanmean(
                (1 - ah[:, 1:]) / t[None, 1:])
            mean_curve = np.nanmean(ah[:, 1:], 0)
            rho_mean = spearmanr(mean_curve, pos).statistic
            rho_seq = []
            for b in range(min(len(ah), 200)):
                rho_seq.append(spearmanr(ah[b, 1:], pos).statistic)
            ch = np.exp(logc[:, h, 1:])
            cv = np.nanmedian(np.nanstd(ch, 0) / np.abs(np.nanmean(ch, 0)))
            gamma = loglog_slope(pos_fit, mean_curve[pos_fit - 1])[0]
            wb = w_b[l, h].cpu().numpy()
            wnb = w_nb[l, h].cpu().numpy()
            cos_ep = float(wb @ wnb / (np.linalg.norm(wb) * np.linalg.norm(wnb)
                                       + 1e-30))
            nsep = float(np.linalg.norm(wnb - wb) /
                         (np.linalg.norm(wb) + np.linalg.norm(wnb) + 1e-30))
            # affine coordinate fit
            c0 = float(wnb @ (wnb - wb))
            d2 = float((wnb - wb) @ (wnb - wb))
            yh = Y[:, h, 1:].reshape(-1)
            af = ah[:, 1:].reshape(-1)
            pred = c0 - af * d2
            fin = np.isfinite(yh) & np.isfinite(pred)
            ssr = ((yh[fin] - pred[fin]) ** 2).sum()
            sst = ((yh[fin] - yh[fin].mean()) ** 2).sum()
            heads[f"head{h}"] = {
                "bos_bias_ratio": float(r_h),
                "alpha_mean_curve_spearman": float(rho_mean),
                "alpha_per_seq_spearman_median": float(np.nanmedian(rho_seq)),
                "alpha_gamma_slope": float(gamma),
                "cbar_cv_median": float(cv),
                "cos_endpoints": cos_ep,
                "normalized_separation": nsep,
                "norm_w_bos": float(np.linalg.norm(wb)),
                "norm_w_nonbos": float(np.linalg.norm(wnb)),
                "e_median_early": float(np.nanmedian(e[:, h, 1:16])),
                "e_median_late": float(np.nanmedian(e[:, h, min(64, L // 2):])),
                "cos_g_w_median_late": float(np.nanmedian(
                    cosg[:, h, min(64, L // 2):])),
                "affine_r2_theoretical": float(1 - ssr / sst) if sst > 0 else float("nan"),
                "norm_entropy_median": float(np.nanmedian(nent[:, h, 1:])),
            }
        # block probe
        pp = arrs[f"L{l}/probe_pred"]
        tgt = np.arange(L)[None, :].repeat(len(pp), 0)
        ssr = ((pp - tgt) ** 2).sum()
        sst = ((tgt - tgt.mean()) ** 2).sum()
        mae = np.abs(pp - tgt).mean()
        summary["per_layer"][f"block{l}"] = {
            "probe_r2": float(1 - ssr / sst),
            "probe_mae": float(mae),
            "heads": heads,
        }
    update_summary(out, "lm_mechanism", summary)
    # compact print
    for l in range(n_layer):
        blk = summary["per_layer"][f"block{l}"]
        best = max(blk["heads"].items(),
                   key=lambda kv: kv[1]["bos_bias_ratio"])
        print(f"block{l}: probe R2={blk['probe_r2']:.3f} | max-BOS head "
              f"{best[0]} ratio={best[1]['bos_bias_ratio']:.1f} "
              f"rho_seq={best[1]['alpha_per_seq_spearman_median']:.3f} "
              f"cv={best[1]['cbar_cv_median']:.3f} "
              f"cos_ep={best[1]['cos_endpoints']:.3f} "
              f"affineR2={best[1]['affine_r2_theoretical']:.3f}")


if __name__ == "__main__":
    main()
