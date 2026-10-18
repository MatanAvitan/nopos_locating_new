"""Head-composition and nonlinear scalar-decode analysis on long context.

This script runs two analyses requested for the R2 long-context checkpoints:

1) ATTN2-12H head-composition test:
   - decode position from per-head BOS attention features alpha_h(i)
   - decode position from per-head attention-write projections on learned head axis
   - compare best single head vs cumulative top-k heads vs all heads

2) ATTN2-1H nonlinear scalar decode test:
   - decode position from scalar s(i) = <o_i - b_attn, delta>
   - compare linear decoder vs nonlinear MLP decoder

All sampling uses OpenWebText validation sequences with prepended BOS.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor


ROOT_DIR = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    return model, ckpt


def sample_sequences(
    data: np.memmap, seq_len: int, n_seq: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - (seq_len - 1), size=n_seq)
    seqs = np.empty((n_seq, seq_len), dtype=np.int64)
    seqs[:, 0] = BOS_TOKEN_ID
    for i, start in enumerate(starts):
        seqs[i, 1:] = data[start : start + seq_len - 1].astype(np.int64)
    return seqs


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rho = spearmanr(x, y).correlation
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2_cod = 1.0 - ss_res / (ss_tot + 1e-12)
    y_true_c = y_true - np.mean(y_true)
    y_pred_c = y_pred - np.mean(y_pred)
    denom = float(np.sqrt(np.sum(y_true_c**2) * np.sum(y_pred_c**2)))
    if denom <= 1e-12:
        corr = 0.0
    else:
        corr = float(np.sum(y_true_c * y_pred_c) / denom)
    r2_corr = float(corr * corr)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "r2_cod": float(r2_cod),
        "r2_corr": float(r2_corr),
        "mae": mae,
        "rmse": rmse,
    }


def fit_linear_with_intercept(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    xmu = x_train.mean(axis=0, keepdims=True)
    xstd = x_train.std(axis=0, keepdims=True) + 1e-8
    xt = (x_train - xmu) / xstd
    xv = (x_test - xmu) / xstd
    a = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(a, y_train, rcond=None)
    av = np.concatenate([xv, np.ones((xv.shape[0], 1), dtype=np.float32)], axis=1)
    return av @ w


def fit_scalar_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    xmu = x_train.mean(axis=0, keepdims=True)
    xstd = x_train.std(axis=0, keepdims=True) + 1e-8
    ymu = y_train.mean(axis=0, keepdims=True)
    ystd = y_train.std(axis=0, keepdims=True) + 1e-8

    xt = (x_train - xmu) / xstd
    xv = (x_test - xmu) / xstd
    yt = (y_train - ymu) / ystd

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )
    mlp.fit(xt, yt.ravel())
    pred_norm = mlp.predict(xv).reshape(-1, 1)
    return pred_norm * ystd + ymu


@torch.no_grad()
def compute_delta_axis(model: TwoLayerMechanismModel, seqs: np.ndarray) -> torch.Tensor:
    d_model = model.config.n_embd
    w_v = model.block2.attn.c_attn.weight[2 * d_model :, :].detach()
    w_o = model.block2.attn.c_proj.weight.detach()

    bos_sum = torch.zeros(d_model, device=DEVICE)
    non_sum = torch.zeros(d_model, device=DEVICE)
    bos_count = 0
    non_count = 0

    for i in range(seqs.shape[0]):
        tokens = torch.from_numpy(seqs[i : i + 1]).to(DEVICE)
        emb = model.wte(tokens)
        emb = model.drop(emb)
        block1_out = model.block1(emb, capture_taps=False)
        ln2_1 = model.block2.ln_1(block1_out)
        wo_v = (ln2_1 @ w_v.T) @ w_o.T

        bos_sum += wo_v[:, 0, :].sum(dim=0)
        bos_count += wo_v.shape[0]

        non = wo_v[:, 1:, :]
        non_sum += non.sum(dim=(0, 1))
        non_count += non.shape[0] * non.shape[1]

    w_bos = bos_sum / max(bos_count, 1)
    w_non = non_sum / max(non_count, 1)
    delta = w_non - w_bos
    return F.normalize(delta, dim=0, eps=1e-12)


@torch.no_grad()
def collect_head_features(
    model: TwoLayerMechanismModel,
    seqs: np.ndarray,
    sample_pos_per_seq: int,
    seed: int,
    delta_hat: torch.Tensor,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    d_model = model.config.n_embd
    n_head = model.config.n_head
    head_dim = d_model // n_head
    scale = 1.0 / math.sqrt(head_dim)

    w_head = model.pos_head.weight.detach().squeeze().to(DEVICE)
    w_head_hat = F.normalize(w_head, dim=0, eps=1e-12)

    w_o = model.block2.attn.c_proj.weight.detach().view(d_model, n_head, head_dim)
    w_o_heads = w_o.permute(1, 2, 0).contiguous()  # [H, hd, D]

    headproj_per_seq: list[np.ndarray] = []
    alpha_per_seq: list[np.ndarray] = []
    delta_scalar_per_seq: list[np.ndarray] = []
    pos_per_seq: list[np.ndarray] = []

    for i in range(seqs.shape[0]):
        tokens = torch.from_numpy(seqs[i : i + 1]).to(DEVICE)

        emb = model.wte(tokens)
        emb = model.drop(emb)
        block1_out = model.block1(emb, capture_taps=False)
        x = model.block2.ln_1(block1_out)  # [1, T, D]

        qkv = model.block2.attn.c_attn(x)
        q, k, v = qkv.split(d_model, dim=2)
        q = q.view(1, -1, n_head, head_dim).transpose(1, 2).squeeze(0)  # [H,T,hd]
        k = k.view(1, -1, n_head, head_dim).transpose(1, 2).squeeze(0)
        v = v.view(1, -1, n_head, head_dim).transpose(1, 2).squeeze(0)

        t = q.shape[1]
        all_pos = np.arange(1, t, dtype=np.int64)
        if len(all_pos) <= sample_pos_per_seq:
            pos_idx = all_pos
        else:
            pos_idx = np.sort(
                rng.choice(all_pos, size=sample_pos_per_seq, replace=False)
            )

        n_pos = len(pos_idx)
        headproj = np.zeros((n_pos, n_head), dtype=np.float32)
        alpha_feats = np.zeros((n_pos, n_head), dtype=np.float32)
        delta_scalar = np.zeros((n_pos, 1), dtype=np.float32)

        for p_i, p in enumerate(pos_idx):
            qp = q[:, p, :]  # [H,hd]
            kp = k[:, : p + 1, :]  # [H,p+1,hd]
            vp = v[:, : p + 1, :]

            scores = torch.einsum("hd,hkd->hk", qp, kp) * scale
            alpha = F.softmax(scores, dim=-1)  # [H,p+1]

            alpha_feats[p_i, :] = alpha[:, 0].detach().cpu().numpy().astype(np.float32)

            head_out = torch.einsum("hk,hkd->hd", alpha, vp)  # [H,hd]
            contrib = torch.einsum("hd,hde->he", head_out, w_o_heads)  # [H,D]

            headproj[p_i, :] = (
                torch.einsum("hd,d->h", contrib, w_head_hat)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            attn_centered = contrib.sum(dim=0)
            delta_scalar[p_i, 0] = float(torch.dot(attn_centered, delta_hat).item())

        headproj_per_seq.append(headproj)
        alpha_per_seq.append(alpha_feats)
        delta_scalar_per_seq.append(delta_scalar)
        pos_per_seq.append(pos_idx.astype(np.float32).reshape(-1, 1))

    return {
        "headproj": np.stack(headproj_per_seq, axis=0),  # [S,P,H]
        "alpha": np.stack(alpha_per_seq, axis=0),  # [S,P,H]
        "delta_scalar": np.stack(delta_scalar_per_seq, axis=0),  # [S,P,1]
        "positions": np.stack(pos_per_seq, axis=0),  # [S,P,1]
    }


def split_by_sequence(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_seq = x.shape[0]
    n_train = max(1, int(round(train_frac * n_seq)))
    n_train = min(n_train, n_seq - 1)

    x_train = x[:n_train].reshape(-1, x.shape[-1])
    y_train = y[:n_train].reshape(-1, 1)
    x_test = x[n_train:].reshape(-1, x.shape[-1])
    y_test = y[n_train:].reshape(-1, 1)
    return x_train, y_train, x_test, y_test


def analyze_multihead_feature_set(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = split_by_sequence(x, y)
    n_head = x.shape[-1]

    single_train = []
    single_test = []
    for h in range(n_head):
        pred_train = fit_linear_with_intercept(
            x_train[:, [h]], y_train, x_train[:, [h]]
        )
        pred_test = fit_linear_with_intercept(x_train[:, [h]], y_train, x_test[:, [h]])
        single_train.append(eval_regression(y_train, pred_train))
        single_test.append(eval_regression(y_test, pred_test))

    rank_scores = [
        m["r2_corr"] if np.isfinite(m["r2_corr"]) else -1.0 for m in single_train
    ]
    rank = np.argsort(rank_scores)[::-1]

    cumulative = []
    for k in range(1, n_head + 1):
        use = rank[:k]
        pred = fit_linear_with_intercept(x_train[:, use], y_train, x_test[:, use])
        metrics = eval_regression(y_test, pred)
        cumulative.append(
            {
                "k": int(k),
                "heads": [int(h) for h in use.tolist()],
                "metrics": metrics,
            }
        )

    pred_all = fit_linear_with_intercept(x_train, y_train, x_test)
    all_metrics = eval_regression(y_test, pred_all)

    rng = np.random.default_rng(seed)
    random_subset = {}
    for k in [1, 2, 4, 8]:
        if k > n_head:
            continue
        vals = []
        for _ in range(20):
            use = rng.choice(n_head, size=k, replace=False)
            pred = fit_linear_with_intercept(x_train[:, use], y_train, x_test[:, use])
            vals.append(eval_regression(y_test, pred)["r2_corr"])
        random_subset[f"k={k}"] = {
            "mean_r2_corr": float(np.mean(vals)),
            "std_r2_corr": float(np.std(vals)),
        }

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x.reshape(-1, n_head), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "single_head_train": single_train,
        "single_head_test": single_test,
        "head_rank_by_train_r2": [int(h) for h in rank.tolist()],
        "cumulative_topk_test": cumulative,
        "all_heads_test": all_metrics,
        "random_subset_test": random_subset,
        "pairwise_head_corr_mean_abs": float(np.mean(np.abs(corr - np.eye(n_head)))),
    }


def run_attn2_12h_experiment(
    ckpt_path: Path,
    data: np.memmap,
    n_seq: int,
    sample_pos_per_seq: int,
    seed: int,
) -> dict[str, Any]:
    model, ckpt = load_model(ckpt_path)
    seq_len = int(ckpt["config"]["block_size"])
    seqs = sample_sequences(data, seq_len, n_seq, seed)

    delta_hat = compute_delta_axis(model, seqs)
    feats = collect_head_features(model, seqs, sample_pos_per_seq, seed + 1, delta_hat)

    alpha_analysis = analyze_multihead_feature_set(
        feats["alpha"], feats["positions"], seed + 2
    )
    headproj_analysis = analyze_multihead_feature_set(
        feats["headproj"], feats["positions"], seed + 3
    )

    return {
        "checkpoint": str(ckpt_path),
        "seq_len": seq_len,
        "n_seq": n_seq,
        "sample_pos_per_seq": sample_pos_per_seq,
        "alpha_bos_features": alpha_analysis,
        "head_axis_projection_features": headproj_analysis,
    }


def run_attn2_1h_scalar_decode(
    ckpt_path: Path,
    data: np.memmap,
    n_seq: int,
    sample_pos_per_seq: int,
    seed: int,
) -> dict[str, Any]:
    model, ckpt = load_model(ckpt_path)
    seq_len = int(ckpt["config"]["block_size"])
    seqs = sample_sequences(data, seq_len, n_seq, seed)

    delta_hat = compute_delta_axis(model, seqs)
    feats = collect_head_features(model, seqs, sample_pos_per_seq, seed + 1, delta_hat)

    x_train, y_train, x_test, y_test = split_by_sequence(
        feats["delta_scalar"], feats["positions"]
    )

    pred_linear = fit_linear_with_intercept(x_train, y_train, x_test)
    pred_mlp = fit_scalar_mlp(x_train, y_train, x_test, seed + 2)

    rho = safe_spearman(
        feats["delta_scalar"].reshape(-1), feats["positions"].reshape(-1)
    )

    return {
        "checkpoint": str(ckpt_path),
        "seq_len": seq_len,
        "n_seq": n_seq,
        "sample_pos_per_seq": sample_pos_per_seq,
        "scalar_delta_spearman": rho,
        "linear_decode": eval_regression(y_test, pred_linear),
        "mlp_decode": eval_regression(y_test, pred_mlp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn2_12h_ckpt",
        type=str,
        default="nanoGPT/out-mechanism-R2-12h-8192/R2/pblr79la/best_ckpt.pt",
    )
    parser.add_argument(
        "--attn2_1h_ckpt",
        type=str,
        default="nanoGPT/out-mechanism-R2-8192/R2/czld6m70/best_ckpt.pt",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--n_seq_12h", type=int, default=24)
    parser.add_argument("--n_seq_1h", type=int, default=24)
    parser.add_argument("--sample_pos_per_seq", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/head_composition_longcontext/head_composition_8192.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(args.split)

    print(f"Using device: {DEVICE}")
    print("Running ATTN2-12H head-composition analysis...")
    res_12h = run_attn2_12h_experiment(
        ckpt_path=ROOT_DIR / args.attn2_12h_ckpt,
        data=data,
        n_seq=args.n_seq_12h,
        sample_pos_per_seq=args.sample_pos_per_seq,
        seed=args.seed,
    )

    print("Running ATTN2-1H nonlinear scalar decode analysis...")
    res_1h = run_attn2_1h_scalar_decode(
        ckpt_path=ROOT_DIR / args.attn2_1h_ckpt,
        data=data,
        n_seq=args.n_seq_1h,
        sample_pos_per_seq=args.sample_pos_per_seq,
        seed=args.seed,
    )

    out = {
        "device": DEVICE,
        "split": args.split,
        "seed": args.seed,
        "attn2_12h": res_12h,
        "attn2_1h_scalar_delta": res_1h,
    }

    save_path = ROOT_DIR / args.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved results to: {save_path}")


if __name__ == "__main__":
    main()
