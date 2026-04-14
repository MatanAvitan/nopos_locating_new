"""Extract per-head alpha_BOS(i) decay curves for ATTN2-12H across context lengths.

For each of L in {128, 1024, 8192, 32768}:
  - load ATTN2-12H checkpoint trained at that context length
  - sample N validation sequences of length block_size with BOS prepended
  - on a log-spaced grid of query positions, compute per-head
    alpha_h(i) = softmax_k( (Q_i K_k) / sqrt(d_h) )[k=0]
  - average over sequences
  - save curves + pairwise-head correlations as JSON
  - plot 4-panel figure (one panel per L) with 12 lines per panel

Also evaluates whether a LINEAR regressor on the 12-dim alpha_BOS feature (one
scalar per head) recovers position. This is the direct basis-expansion test:
if per-head schedules are identical, R^2 saturates at the single-head value;
if they differ, combining heads linearly must improve it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPTS = [
    (128,   "nanoGPT/out-mechanism-R2-12h-128/R2/375mmb9k/best_ckpt.pt"),
    (1024,  "nanoGPT/out-mechanism-R2-12h-1024/R2/pb1sxvzm/best_ckpt.pt"),
    (8192,  "nanoGPT/out-mechanism-R2-12h-8192/R2/pblr79la/best_ckpt.pt"),
    (32768, "nanoGPT/out-mechanism-R2-12h-32k/R2/8o6lqqwh/best_ckpt.pt"),
]


def load_data(split: str = "val") -> np.memmap:
    path = ROOT_DIR / f"nanoGPT/data/openwebtext/{split}.bin"
    return np.memmap(str(path), dtype=np.uint16, mode="r")


def load_model(ckpt_path: Path) -> tuple[TwoLayerMechanismModel, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    config = TwoLayerMechanismConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        dropout=0.0,
        norm_type=cfg["norm_type"],
        bias=True,
        use_regression=True,
    )
    model = TwoLayerMechanismModel(config)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    model.block1.attn.use_flash = True
    model.block2.attn.use_flash = True
    return model, ckpt


def sample_sequences(data: np.memmap, seq_len: int, n_seq: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - (seq_len - 1), size=n_seq)
    seqs = np.empty((n_seq, seq_len), dtype=np.int64)
    seqs[:, 0] = BOS_TOKEN_ID
    for i, start in enumerate(starts):
        seqs[i, 1:] = data[start : start + seq_len - 1].astype(np.int64)
    return seqs


def log_spaced_positions(seq_len: int, n_points: int) -> np.ndarray:
    if seq_len <= n_points:
        return np.arange(1, seq_len, dtype=np.int64)
    grid = np.unique(
        np.round(np.geomspace(1.0, seq_len - 1, num=n_points)).astype(np.int64)
    )
    grid = grid[grid >= 1]
    grid = grid[grid <= seq_len - 1]
    return grid


@torch.no_grad()
def compute_alpha_bos_curves(
    model: TwoLayerMechanismModel,
    seqs: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Return array of shape [S, P, H] with alpha_BOS per seq, position, head."""
    d_model = model.config.n_embd
    n_head = model.config.n_head
    head_dim = d_model // n_head
    scale = 1.0 / math.sqrt(head_dim)

    S = seqs.shape[0]
    P = len(positions)
    out = np.zeros((S, P, n_head), dtype=np.float32)

    pos_tensor = torch.from_numpy(positions).to(DEVICE)

    for s in range(S):
        tokens = torch.from_numpy(seqs[s : s + 1]).to(DEVICE)
        emb = model.wte(tokens)
        emb = model.drop(emb)
        block1_out = model.block1(emb, capture_taps=False)
        x = model.block2.ln_1(block1_out)

        qkv = model.block2.attn.c_attn(x)
        q_all, k_all, _ = qkv.split(d_model, dim=2)
        q_all = q_all.view(1, -1, n_head, head_dim).transpose(1, 2).squeeze(0)  # [H,T,hd]
        k_all = k_all.view(1, -1, n_head, head_dim).transpose(1, 2).squeeze(0)

        k_bos = k_all[:, 0, :]  # [H, hd]

        for p_i, p in enumerate(positions):
            qp = q_all[:, p, :]  # [H, hd]
            kp = k_all[:, : p + 1, :]  # [H, p+1, hd]
            scores = torch.einsum("hd,hkd->hk", qp, kp) * scale  # [H, p+1]
            alpha = torch.softmax(scores, dim=-1)  # [H, p+1]
            out[s, p_i, :] = alpha[:, 0].detach().cpu().numpy().astype(np.float32)

        del q_all, k_all, qkv, x, block1_out, emb
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    return out


def fit_linear(x_tr, y_tr, x_te):
    mu = x_tr.mean(0, keepdims=True)
    sd = x_tr.std(0, keepdims=True) + 1e-8
    xt = (x_tr - mu) / sd
    xv = (x_te - mu) / sd
    A = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    Av = np.concatenate([xv, np.ones((xv.shape[0], 1), dtype=np.float32)], axis=1)
    return Av @ w


def r2_cod(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def basis_expansion_test(alpha_SPH: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    """Does a linear reader on per-head alpha_BOS recover position?

    Fits linear regression on the 12-dim alpha feature vs position. Compares:
      (a) best single-head linear fit
      (b) all-12-heads linear fit
    Strong (b) - (a) gap => heads carry non-redundant (diverse) schedules.
    """
    S, P, H = alpha_SPH.shape
    n_train = max(1, int(round(0.7 * S)))
    n_train = min(n_train, S - 1)

    x_tr = alpha_SPH[:n_train].reshape(-1, H).astype(np.float32)
    y_tr = np.broadcast_to(positions[None, :], (n_train, P)).reshape(-1, 1).astype(np.float32)
    x_te = alpha_SPH[n_train:].reshape(-1, H).astype(np.float32)
    y_te = np.broadcast_to(positions[None, :], (S - n_train, P)).reshape(-1, 1).astype(np.float32)

    per_head_r2 = []
    for h in range(H):
        pred = fit_linear(x_tr[:, [h]], y_tr, x_te[:, [h]])
        per_head_r2.append(float(r2_cod(y_te, pred)))

    pred_all = fit_linear(x_tr, y_tr, x_te)
    r2_all = float(r2_cod(y_te, pred_all))

    return {
        "per_head_r2_cod": per_head_r2,
        "best_single_head_r2_cod": float(max(per_head_r2)),
        "all_12_heads_r2_cod": r2_all,
        "gap": r2_all - float(max(per_head_r2)),
    }


def pairwise_head_corr(alpha_SPH: np.ndarray) -> np.ndarray:
    """Correlation between per-head alpha_BOS curves, averaged over sequences."""
    S, P, H = alpha_SPH.shape
    curves = alpha_SPH.mean(axis=0)  # [P, H]
    curves = curves - curves.mean(axis=0, keepdims=True)
    std = curves.std(axis=0, keepdims=True) + 1e-12
    curves = curves / std
    corr = (curves.T @ curves) / P
    return corr.astype(np.float32)


def plot_curves(all_curves: dict[int, dict[str, Any]], save_path: Path) -> None:
    n = len(all_curves)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    cmap = plt.cm.tab20
    for ax, (L, rec) in zip(axes, all_curves.items()):
        pos = np.array(rec["positions"])
        mean_curves = np.array(rec["alpha_mean"])  # [P, H]
        H = mean_curves.shape[1]
        for h in range(H):
            ax.plot(pos, mean_curves[:, h], color=cmap(h % 20), lw=1.2, alpha=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("position $i$")
        ax.set_ylabel(r"$\alpha_h(i) = \mathrm{softmax}_k(q_i k_k / \sqrt{d_h})[k{=}0]$")
        ax.set_title(f"ATTN2-12H, $L={L}$")
        ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(r"Per-head $\alpha_{\mathrm{BOS}}$ decay across context lengths")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seq", type=int, default=16)
    parser.add_argument("--n_positions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/per_head_alpha_bos_decay",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data("val")
    save_dir = ROOT_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, Any]] = {}
    for L, rel_ckpt in CKPTS:
        print(f"[L={L}] loading {rel_ckpt}")
        ckpt_path = ROOT_DIR / rel_ckpt
        model, ckpt = load_model(ckpt_path)
        block_size = int(ckpt["config"]["block_size"])
        assert block_size == L, f"ckpt block_size {block_size} != {L}"

        positions = log_spaced_positions(block_size, args.n_positions)
        print(f"[L={L}] {len(positions)} positions from {positions[0]} to {positions[-1]}")
        seqs = sample_sequences(data, block_size, args.n_seq, args.seed)

        print(f"[L={L}] computing alpha_BOS over {args.n_seq} seqs ...")
        alpha_SPH = compute_alpha_bos_curves(model, seqs, positions)

        alpha_mean = alpha_SPH.mean(axis=0)  # [P, H]
        head_corr = pairwise_head_corr(alpha_SPH)
        offdiag = head_corr[np.triu_indices_from(head_corr, k=1)]

        basis_test = basis_expansion_test(alpha_SPH, positions)

        results[L] = {
            "ckpt": rel_ckpt,
            "n_seq": int(args.n_seq),
            "positions": positions.tolist(),
            "alpha_mean": alpha_mean.tolist(),
            "head_corr_mean": float(np.mean(offdiag)),
            "head_corr_min": float(np.min(offdiag)),
            "head_corr_max": float(np.max(offdiag)),
            "head_corr_matrix": head_corr.tolist(),
            "basis_expansion_test": basis_test,
        }
        print(
            f"[L={L}] pairwise head-curve corr: mean={results[L]['head_corr_mean']:.3f} "
            f"min={results[L]['head_corr_min']:.3f} max={results[L]['head_corr_max']:.3f}"
        )
        print(
            f"[L={L}] basis test: best single head R2={basis_test['best_single_head_r2_cod']:.3f}, "
            f"all 12 R2={basis_test['all_12_heads_r2_cod']:.3f}, gap={basis_test['gap']:.3f}"
        )

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_json = save_dir / "per_head_alpha_bos_decay.json"
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"Saved: {save_json}")

    save_png = save_dir / "per_head_alpha_bos_decay.png"
    plot_curves(results, save_png)
    print(f"Saved: {save_png}")


if __name__ == "__main__":
    main()
