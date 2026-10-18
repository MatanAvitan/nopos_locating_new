"""
Why does the Layer-2 QK circuit make attention to BOS (column 0) decay
exponentially with query position?

This script, for the ATTN2-1H model (out-mechanism-R2-1024, n_head=1, d=768):

  1. Plots W_Q and W_K (block2) as heatmaps, plus the effective query-key
     matrix  M = W_Q^T W_K  (the bilinear form acting on the residual stream).
  2. Decomposes M with an SVD to show it is effectively low rank.
  3. Runs the model on real data and captures
        x = LN(r^1_out)              (= block2_ln1, the input to attn2)
        q_i = W_Q x_i ,  k_j = W_K x_j
        w_{i,j}                       (softmax attention weights of block2)
  4. Traces the BOS column:
        logit_{i,0} = q_i . k_0 / sqrt(d_head)   vs position i
        w_{i,0}                                   vs position i
     and fits  (a) an exponential   w_{i,0} ~ A * exp(-beta i)
               (b) a power law       w_{i,0} ~ A / (i+1)
     to decide which the model actually realises.
  5. Connects the curve back to the weights: the fixed BOS key k_0 defines a
     direction; the query's projection q_i . k_0 is what produces the linear-
     in-position logit, hence the exponential softmax decay.

Output: results/qk_attn2_1h/qk_attn2_1h_analysis.{png,pdf} + metrics json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import (  # noqa: E402
    TwoLayerMechanismModel,
    TwoLayerMechanismConfig,
)

BOS_TOKEN_ID = 50256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CKPT = "nanoGPT/out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt"
DEFAULT_DATA = "nanoGPT/data/openwebtext"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: Path) -> TwoLayerMechanismModel:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config_dict = ckpt.get("config", ckpt.get("model_args", {}))
    valid = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    cfg = TwoLayerMechanismConfig(**{k: v for k, v in config_dict.items() if k in valid})
    model = TwoLayerMechanismModel(cfg)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    model.block1.attn.use_flash = False
    model.block2.attn.use_flash = False
    return model


def load_data(data_dir: Path) -> np.memmap:
    return np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")


def sample_tokens(data, seq_len, batch_size) -> torch.Tensor:
    ix = torch.randint(len(data) - (seq_len - 1), (batch_size,))
    seqs = []
    for i in ix:
        seq = np.concatenate([[BOS_TOKEN_ID], data[i : i + seq_len - 1].astype(np.int64)])
        seqs.append(torch.from_numpy(seq))
    return torch.stack(seqs).to(DEVICE)


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(model, data, seq_len, batch_size, n_batches):
    """Average over data: logit_{i,0}, w_{i,0}, mean non-BOS logit, and the
    projection of the query onto the fixed BOS-key direction."""
    d = model.config.n_embd
    n_head = model.config.n_head
    head_dim = d // n_head
    scale = 1.0 / math.sqrt(head_dim)

    W = model.block2.attn.c_attn.weight  # [3d, d]
    b = model.block2.attn.c_attn.bias  # [3d]
    W_Q, W_K = W[:d], W[d : 2 * d]
    b_Q, b_K = b[:d], b[d : 2 * d]

    sum_logit_bos = torch.zeros(seq_len, device=DEVICE)
    sum_logit_other = torch.zeros(seq_len, device=DEVICE)
    sum_w_bos = torch.zeros(seq_len, device=DEVICE)
    sum_qk_proj = torch.zeros(seq_len, device=DEVICE)  # q_i . khat_0 (unit BOS key)
    sum_k0 = torch.zeros(head_dim, device=DEVICE)
    count = 0

    for _ in range(n_batches):
        tokens = sample_tokens(data, seq_len, batch_size)
        model(tokens, capture_taps=True)
        taps = model.get_all_taps()
        _, attn2 = model.get_attention_weights()  # [B, n_head, T, T]
        x = taps["block2_ln1"]  # [B, T, d] = input to attn2

        q = x @ W_Q.T + b_Q  # [B, T, d]  (single head -> head_dim == d)
        k = x @ W_K.T + b_K
        logits = (q @ k.transpose(-2, -1)) * scale  # [B, T, T]

        B, T, _ = logits.shape
        # BOS column
        logit_bos = logits[:, :, 0]  # [B, T]
        # mean over non-BOS *valid* (causal) keys for each query row
        causal = torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool))
        non_bos_mask = causal.clone()
        non_bos_mask[:, 0] = False  # exclude BOS column
        nb = non_bos_mask.unsqueeze(0).float()  # [1, T, T]
        logit_other = (logits * nb).sum(-1) / nb.sum(-1).clamp_min(1.0)  # [B, T]

        w_bos = attn2[:, 0, :, 0]  # [B, T] attention weight on BOS

        k0 = k[:, 0, :]  # [B, d] BOS key per sample
        k0_hat = torch.nn.functional.normalize(k0, dim=-1)
        qk_proj = torch.einsum("btd,bd->bt", q, k0_hat)  # [B, T]

        sum_logit_bos += logit_bos.sum(0)
        sum_logit_other += logit_other.sum(0)
        sum_w_bos += w_bos.sum(0)
        sum_qk_proj += qk_proj.sum(0)
        sum_k0 += k0.mean(0)
        count += B

    out = {
        "logit_bos": (sum_logit_bos / count).cpu().numpy(),
        "logit_other": (sum_logit_other / count).cpu().numpy(),
        "w_bos": (sum_w_bos / count).cpu().numpy(),
        "qk_proj": (sum_qk_proj / count).cpu().numpy(),
        "k0_mean": (sum_k0 / count).cpu().numpy(),
        "W_Q": W_Q.detach().cpu().numpy(),
        "W_K": W_K.detach().cpu().numpy(),
        "scale": scale,
        "head_dim": head_dim,
    }
    return out


def fit_curves(pos, y):
    """Compare three models for the BOS attention weight w_{i,0}:
        (exp)     w ~ A exp(-beta p)
        (power)   w ~ A / (p+1)^gamma
        (softmax) w = 1 / (1 + c p)   <=>  e^Delta/(e^Delta + p), c = e^{-Delta}
    All R^2 are reported in log(w) space so the curves are compared on equal
    footing. The softmax-competition model is the one implied by a constant
    BOS logit advantage Delta over the i growing non-BOS keys."""
    pos = np.asarray(pos, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (pos >= 1) & (y > 1e-6)
    p, yy = pos[mask], y[mask]
    logy = np.log(yy)
    ss_tot = np.sum((logy - logy.mean()) ** 2)

    def r2_log(pred_logw):
        return float(1 - np.sum((logy - pred_logw) ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    # exponential
    A = np.vstack([np.ones_like(p), p]).T
    a_exp, neg_beta = np.linalg.lstsq(A, logy, rcond=None)[0]
    r2_exp = r2_log(a_exp + neg_beta * p)

    # power law
    A2 = np.vstack([np.ones_like(p), np.log(p + 1.0)]).T
    c_pow, neg_gamma = np.linalg.lstsq(A2, logy, rcond=None)[0]
    r2_pow = r2_log(c_pow + neg_gamma * np.log(p + 1.0))

    # softmax-competition: fit 1/w = a + c p, then score in log space
    inv = 1.0 / yy
    a_soft, c_soft = np.linalg.lstsq(np.vstack([np.ones_like(p), p]).T, inv, rcond=None)[0]
    w_soft = 1.0 / np.clip(a_soft + c_soft * p, 1e-9, None)
    r2_soft = r2_log(np.log(np.clip(w_soft, 1e-12, None)))

    return {
        "beta": float(-neg_beta),
        "A_exp": float(np.exp(a_exp)),
        "r2_exp_logspace": r2_exp,
        "gamma": float(-neg_gamma),
        "r2_pow_logspace": r2_pow,
        "softmax_c": float(c_soft),
        "softmax_a": float(a_soft),
        "softmax_Delta": float(-np.log(c_soft)) if c_soft > 0 else float("nan"),
        "r2_softmax_logspace": r2_soft,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def diverging(ax, mat, title, cmap="RdBu_r", q=99.5):
    v = np.percentile(np.abs(mat), q)
    im = ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(0.0, -v, v), aspect="auto")
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_figure(res, fits, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    W_Q, W_K = res["W_Q"], res["W_K"]
    d = W_Q.shape[0]

    M = W_Q.T @ W_K  # [d, d] effective query-key bilinear form
    U, S, Vt = np.linalg.svd(M)

    pos = np.arange(len(res["w_bos"]))

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # Row 1: weight heatmaps
    ax = fig.add_subplot(gs[0, 0]); diverging(ax, W_Q, r"$W_Q$ (block2, $d\times d$)")
    ax.set_xlabel("input dim"); ax.set_ylabel("head dim")
    ax = fig.add_subplot(gs[0, 1]); diverging(ax, W_K, r"$W_K$ (block2, $d\times d$)")
    ax.set_xlabel("input dim"); ax.set_ylabel("head dim")
    ax = fig.add_subplot(gs[0, 2])
    diverging(ax, M, r"effective $M=W_Q^{\top}W_K$")
    ax.set_xlabel(r"key dim ($x_j$)"); ax.set_ylabel(r"query dim ($x_i$)")

    # Row 2 col 0: singular spectrum of M
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(S / S[0], "o-", ms=3)
    ax.set_yscale("log")
    ax.set_title("Singular values of $M$ (normalised)")
    ax.set_xlabel("index"); ax.set_ylabel(r"$\sigma_k/\sigma_0$")
    ax.set_xlim(-1, min(60, len(S)))
    eff_rank = float((S.sum() ** 2) / (S ** 2).sum())
    ax.text(0.5, 0.85, f"participation ratio = {eff_rank:.1f}\n(d={d})",
            transform=ax.transAxes, fontsize=9)

    # Row 2 col 1: logit to BOS vs others
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(pos, res["logit_bos"], label=r"logit$_{i,0}$ (BOS key)", color="C3")
    ax.plot(pos, res["logit_other"], label="mean logit, non-BOS keys", color="C0")
    ax.set_title("Attention logits vs query position")
    ax.set_xlabel("query position $i$"); ax.set_ylabel("logit")
    ax.legend(fontsize=8)

    # Row 2 col 2: logit gap (BOS - others) -> CONSTANT, drives the 1/i law
    ax = fig.add_subplot(gs[1, 2])
    gap = res["logit_bos"] - res["logit_other"]
    ax.plot(pos, gap, color="C2")
    gap_mean = float(np.mean(gap[50:])) if len(gap) > 60 else float(np.mean(gap[1:]))
    ax.axhline(gap_mean, color="k", lw=0.8, ls="--")
    ax.set_title(r"logit gap $\Delta_i=$ logit$_{i,0}-$mean$_{j\neq0}$logit$_{i,j}$")
    ax.set_xlabel("query position $i$"); ax.set_ylabel(r"$\Delta_i$")
    ax.text(0.4, 0.15, fr"$\Delta\approx{gap_mean:.2f}$ (constant)",
            transform=ax.transAxes, fontsize=10)

    # Row 3 col 0: BOS attention weight, linear scale
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(pos, res["w_bos"], color="C3")
    ax.set_title(r"BOS attention weight $w_{i,0}$ (linear)")
    ax.set_xlabel("query position $i$"); ax.set_ylabel(r"$w_{i,0}$")

    # Row 3 col 1: log-scale w_bos with exp / power / softmax-competition fits
    ax = fig.add_subplot(gs[2, 1])
    m = (pos >= 1) & (res["w_bos"] > 1e-6)
    pm = pos[m]
    ax.semilogy(pm, res["w_bos"][m], ".", ms=4, color="C3", label="data")
    ax.semilogy(pm, fits["A_exp"] * np.exp(-fits["beta"] * pm), "-", color="0.5",
                label=fr"exp ($R^2$={fits['r2_exp_logspace']:.3f})")
    w_soft = 1.0 / (fits["softmax_a"] + fits["softmax_c"] * pm)
    ax.semilogy(pm, w_soft, "-", color="k", lw=2,
                label=fr"$1/(1+i e^{{-\Delta}})$ ($R^2$={fits['r2_softmax_logspace']:.3f})")
    ax.set_title(r"$w_{i,0}$ (log): softmax-competition wins")
    ax.set_xlabel("query position $i$"); ax.set_ylabel(r"$\log w_{i,0}$")
    ax.legend(fontsize=8)

    # Row 3 col 2: query projection onto fixed BOS-key direction
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(pos, res["qk_proj"], color="C4")
    ax.set_title(r"$q_i\cdot\hat{k}_0$ : query along fixed BOS-key dir")
    ax.set_xlabel("query position $i$"); ax.set_ylabel(r"$q_i\cdot\hat k_0$")

    fig.suptitle(
        "ATTN2-1H Layer-2 QK circuit: a constant BOS logit advantage "
        r"$\Delta$ gives $w_{i,0}=1/(1+i\,e^{-\Delta})$",
        fontsize=14, y=0.995,
    )
    png = save_dir / "qk_attn2_1h_analysis.png"
    pdf = save_dir / "qk_attn2_1h_analysis.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "singular_values_top20": S[:20].tolist(),
        "participation_ratio_M": eff_rank,
        "M_fro_norm": float(np.linalg.norm(M)),
        "top1_energy_frac": float(S[0] ** 2 / (S ** 2).sum()),
        "top5_energy_frac": float((S[:5] ** 2).sum() / (S ** 2).sum()),
    }, str(png)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--data_dir", default=DEFAULT_DATA)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_batches", type=int, default=12)
    p.add_argument("--save_dir", default="results/qk_attn2_1h")
    args = p.parse_args()

    model = load_model(ROOT_DIR / args.ckpt)
    data = load_data(ROOT_DIR / args.data_dir)
    seq_len = min(args.seq_len, model.config.block_size)

    res = collect(model, data, seq_len, args.batch_size, args.n_batches)
    fits = fit_curves(np.arange(seq_len), res["w_bos"])
    svd_stats, png = make_figure(res, fits, ROOT_DIR / args.save_dir)

    metrics = {
        "checkpoint": args.ckpt,
        "seq_len": seq_len,
        "n_head": model.config.n_head,
        "d_model": model.config.n_embd,
        "w_bos_fits": fits,
        "svd": svd_stats,
    }
    out_json = ROOT_DIR / args.save_dir / "qk_attn2_1h_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved figure:", png)
    print("Saved metrics:", out_json)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
