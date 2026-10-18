"""Compute Nseq=128 readout tables and R2 MLP+head finetune table.

Tables produced:
1) Per-sequence Spearman on attn-centered Delta for all models.
2) Decoding metrics from o_i - b_attn for all models.
   Includes linear and MLP regressors.
3) R2-only finetune where Block2 MLP and linear head are trained jointly
   for absolute position regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


ROOT_DIR = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
NSEQ = 128
SAMPLE_POS_PER_SEQ = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExpCfg:
    name: str
    checkpoint: Path
    seq_len: int
    n_head_expected: int
    proj_batch_size: int
    is_r2: bool


EXPERIMENTS: list[ExpCfg] = [
    ExpCfg(
        name="R0_128_FULL-12H",
        checkpoint=ROOT_DIR / "nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
        seq_len=128,
        n_head_expected=12,
        proj_batch_size=16,
        is_r2=False,
    ),
    ExpCfg(
        name="R2_128_ATTN2-1H",
        checkpoint=ROOT_DIR
        / "nanoGPT/out-2layer-mechanism-r2-1head-attnonly-fullblock-40k/R2/o4w7v8dv/best_ckpt.pt",
        seq_len=128,
        n_head_expected=1,
        proj_batch_size=16,
        is_r2=True,
    ),
    ExpCfg(
        name="R0_1024_FULL-12H",
        checkpoint=ROOT_DIR / "nanoGPT/out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt",
        seq_len=1024,
        n_head_expected=12,
        proj_batch_size=8,
        is_r2=False,
    ),
    ExpCfg(
        name="R2_1024_ATTN2-1H",
        checkpoint=ROOT_DIR / "nanoGPT/out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt",
        seq_len=1024,
        n_head_expected=1,
        proj_batch_size=8,
        is_r2=True,
    ),
    ExpCfg(
        name="R0_8192_FULL-12H",
        checkpoint=ROOT_DIR / "nanoGPT/out-mechanism-R0-8192/R0/lwrgnltx/best_ckpt.pt",
        seq_len=8192,
        n_head_expected=12,
        proj_batch_size=1,
        is_r2=False,
    ),
    ExpCfg(
        name="R2_8192_ATTN2-1H",
        checkpoint=ROOT_DIR / "nanoGPT/out-mechanism-R2-8192/R2/czld6m70/best_ckpt.pt",
        seq_len=8192,
        n_head_expected=1,
        proj_batch_size=1,
        is_r2=True,
    ),
]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    r = spearmanr(x, y).correlation
    if r is None or np.isnan(r):
        return 0.0
    return float(r)


def load_data() -> np.memmap:
    return np.memmap(
        str(ROOT_DIR / "nanoGPT/data/openwebtext/train.bin"), dtype=np.uint16, mode="r"
    )


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
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    )
    model.to(DEVICE).eval()
    return model, ckpt


def sample_sequences(
    data: np.memmap, seq_len: int, n_seq: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - (seq_len - 1), size=n_seq)
    seqs = np.empty((n_seq, seq_len), dtype=np.int64)
    seqs[:, 0] = BOS_TOKEN_ID
    for i, s in enumerate(starts):
        seqs[i, 1:] = data[s : s + seq_len - 1].astype(np.int64)
    return seqs


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2_cod = 1.0 - ss_res / (ss_tot + 1e-12)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2_corr = float(corr * corr)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "r2_cod": float(r2_cod),
        "r2_corr": float(r2_corr),
        "mae": mae,
        "rmse": rmse,
    }


class ScalarMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorMLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Module:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    tx = torch.from_numpy(train_x).to(DEVICE)
    ty = torch.from_numpy(train_y).to(DEVICE)
    vx = torch.from_numpy(val_x).to(DEVICE)
    vy = torch.from_numpy(val_y).to(DEVICE)

    best_state = None
    best_val = float("inf")
    patience = 20
    bad = 0

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(tx.shape[0], device=DEVICE)
        for i in range(0, tx.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            pred = model(tx[idx])
            loss = F.mse_loss(pred, ty[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(vx)
            val_loss = F.mse_loss(val_pred, vy).item()
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    return model


def fit_linear_scalar(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    xmu = x_train.mean(axis=0, keepdims=True)
    xstd = x_train.std(axis=0, keepdims=True) + 1e-8
    xt = (x_train - xmu) / xstd
    xv = (x_test - xmu) / xstd
    A = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    Av = np.concatenate([xv, np.ones((xv.shape[0], 1), dtype=np.float32)], axis=1)
    return Av @ w


def _sample_positions(rng: np.random.Generator, L: int) -> np.ndarray:
    avail = np.arange(1, L, dtype=np.int64)
    if len(avail) <= SAMPLE_POS_PER_SEQ:
        return avail
    return np.sort(rng.choice(avail, size=SAMPLE_POS_PER_SEQ, replace=False))


def collect_model_tables(
    exp: ExpCfg,
    model: TwoLayerMechanismModel,
    seqs: np.ndarray,
) -> dict[str, Any]:
    n_seq = seqs.shape[0]
    L = exp.seq_len
    d_model = model.config.n_embd

    W_V = model.block2.attn.c_attn.weight[2 * d_model :, :].detach()
    W_O = model.block2.attn.c_proj.weight.detach()
    b_O = model.block2.attn.c_proj.bias.detach()
    head_dir = F.normalize(model.pos_head.weight.detach().squeeze(), dim=0, eps=1e-12)

    # split by sequence index
    n_train_seq = int(0.8 * n_seq)
    train_ids = set(range(n_train_seq))
    test_ids = set(range(n_train_seq, n_seq))

    # pass 1: estimate w_BOS / w_nonBOS
    bos_sum = torch.zeros(d_model, device=DEVICE)
    non_sum = torch.zeros(d_model, device=DEVICE)
    bos_count = 0
    non_count = 0

    with torch.no_grad():
        for s in range(0, n_seq, exp.proj_batch_size):
            e = min(n_seq, s + exp.proj_batch_size)
            toks = torch.from_numpy(seqs[s:e]).to(DEVICE)
            model(toks, capture_taps=True)
            ln2_1 = model.get_all_taps()["block2_ln1"]
            wov = (ln2_1 @ W_V.T) @ W_O.T
            bos_sum += wov[:, 0, :].sum(dim=0)
            bos_count += wov.shape[0]
            non = wov[:, 1:, :]
            non_sum += non.sum(dim=(0, 1))
            non_count += non.shape[0] * non.shape[1]

    w_bos = bos_sum / max(1, bos_count)
    w_non = non_sum / max(1, non_count)
    delta = F.normalize(w_non - w_bos, dim=0, eps=1e-12)

    # pass 2: projections and sampled train/test features
    proj_delta = np.zeros((n_seq, L), dtype=np.float32)
    proj_head = np.zeros((n_seq, L), dtype=np.float32)

    x_train_scalar, y_train = [], []
    x_test_scalar, y_test = [], []
    x_train_vec, x_test_vec = [], []

    rng = np.random.default_rng(123 + L)

    with torch.no_grad():
        for s in range(0, n_seq, exp.proj_batch_size):
            e = min(n_seq, s + exp.proj_batch_size)
            toks = torch.from_numpy(seqs[s:e]).to(DEVICE)
            model(toks, capture_taps=True)
            attn_centered = model.get_all_taps()["block2_attn"] - b_O

            p_delta = (attn_centered @ delta).detach().cpu().numpy().astype(np.float32)
            p_head = (
                (attn_centered @ head_dir).detach().cpu().numpy().astype(np.float32)
            )
            vec = attn_centered.detach().cpu().numpy().astype(np.float32)

            proj_delta[s:e] = p_delta
            proj_head[s:e] = p_head

            for bi, seq_id in enumerate(range(s, e)):
                pos_idx = _sample_positions(rng, L)
                feats_scalar = p_delta[bi, pos_idx][:, None]
                feats_vec = vec[bi, pos_idx, :]
                targets = pos_idx.astype(np.float32)[:, None]
                if seq_id in train_ids:
                    x_train_scalar.append(feats_scalar)
                    x_train_vec.append(feats_vec)
                    y_train.append(targets)
                else:
                    x_test_scalar.append(feats_scalar)
                    x_test_vec.append(feats_vec)
                    y_test.append(targets)

    x_train_scalar = np.concatenate(x_train_scalar, axis=0)
    x_test_scalar = np.concatenate(x_test_scalar, axis=0)
    x_train_vec = np.concatenate(x_train_vec, axis=0)
    x_test_vec = np.concatenate(x_test_vec, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # table 1 stats
    pos = np.arange(L, dtype=np.float32)
    rho_delta = []
    for i in range(n_seq):
        rho_delta.append(safe_spearman(proj_delta[i, 1:], pos[1:]))
    rho_delta = np.asarray(rho_delta)

    # table 2 decoding metrics
    pred_lin = fit_linear_scalar(x_train_scalar, y_train, x_test_scalar)
    m_lin = eval_regression(y_test, pred_lin)

    # scalar-MLP
    xmu_s = x_train_scalar.mean(axis=0, keepdims=True)
    xstd_s = x_train_scalar.std(axis=0, keepdims=True) + 1e-8
    xs_tr = (x_train_scalar - xmu_s) / xstd_s
    xs_te = (x_test_scalar - xmu_s) / xstd_s

    ys_tr = y_train / float(L - 1)
    ys_te = y_test / float(L - 1)
    scalar_mlp = train_mlp(
        model=ScalarMLP(),
        train_x=xs_tr.astype(np.float32),
        train_y=ys_tr.astype(np.float32),
        val_x=xs_te.astype(np.float32),
        val_y=ys_te.astype(np.float32),
        epochs=200,
        batch_size=min(8192, xs_tr.shape[0]),
        lr=1e-3,
    )
    with torch.no_grad():
        pred_scalar_mlp = scalar_mlp(
            torch.from_numpy(xs_te.astype(np.float32)).to(DEVICE)
        ).cpu().numpy() * float(L - 1)
    m_scalar_mlp = eval_regression(y_test, pred_scalar_mlp)

    # vector-MLP on o_i - b_attn
    xmu_v = x_train_vec.mean(axis=0, keepdims=True)
    xstd_v = x_train_vec.std(axis=0, keepdims=True) + 1e-8
    xv_tr = (x_train_vec - xmu_v) / xstd_v
    xv_te = (x_test_vec - xmu_v) / xstd_v

    vector_mlp = train_mlp(
        model=VectorMLP(d_model),
        train_x=xv_tr.astype(np.float32),
        train_y=ys_tr.astype(np.float32),
        val_x=xv_te.astype(np.float32),
        val_y=ys_te.astype(np.float32),
        epochs=120,
        batch_size=min(4096, xv_tr.shape[0]),
        lr=8e-4,
    )
    with torch.no_grad():
        pred_vec_mlp = vector_mlp(
            torch.from_numpy(xv_te.astype(np.float32)).to(DEVICE)
        ).cpu().numpy() * float(L - 1)
    m_vec_mlp = eval_regression(y_test, pred_vec_mlp)

    # pooled linear R2 from full proj_delta tokens (exclude position 0)
    flat_pos = np.tile(np.arange(1, L, dtype=np.float32), n_seq)
    flat_proj = proj_delta[:, 1:].reshape(-1)
    pooled_r2 = float(np.corrcoef(flat_pos, flat_proj)[0, 1] ** 2)

    return {
        "seq_len": L,
        "n_seq": int(n_seq),
        "n_head": int(model.config.n_head),
        "delta_alignment": {
            "rho_seq_mean": float(rho_delta.mean()),
            "rho_seq_median": float(np.median(rho_delta)),
            "rho_seq_p05": float(np.percentile(rho_delta, 5)),
            "rho_seq_p95": float(np.percentile(rho_delta, 95)),
            "pooled_linear_r2": pooled_r2,
        },
        "decode_o_minus_b": {
            "linear_scalar_delta": m_lin,
            "mlp_scalar_delta": m_scalar_mlp,
            "mlp_vector_o_minus_b": m_vec_mlp,
            "n_train_tokens": int(x_train_vec.shape[0]),
            "n_test_tokens": int(x_test_vec.shape[0]),
        },
    }


def evaluate_model_outputs(
    model: TwoLayerMechanismModel, seqs: np.ndarray, batch_size: int
) -> dict[str, float]:
    L = seqs.shape[1]
    all_pred = []
    all_true = []
    with torch.no_grad():
        for s in range(0, seqs.shape[0], batch_size):
            e = min(seqs.shape[0], s + batch_size)
            toks = torch.from_numpy(seqs[s:e]).to(DEVICE)
            out, _ = model(toks, targets=None, capture_taps=False)
            pred = out.squeeze(-1).detach().cpu().numpy()
            true = np.tile(np.arange(L, dtype=np.float32), (e - s, 1))
            all_pred.append(pred)
            all_true.append(true)
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    return eval_regression(true, pred)


def finetune_r2_mlp_head(
    base_model: TwoLayerMechanismModel,
    data: np.memmap,
    seq_len: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    # clone by reloading state dict
    model = base_model

    # freeze all
    for p in model.parameters():
        p.requires_grad = False
    # train block2 MLP path and linear head
    for p in model.block2.ln_2.parameters():
        p.requires_grad = True
    for p in model.block2.mlp.parameters():
        p.requires_grad = True
    for p in model.pos_head.parameters():
        p.requires_grad = True

    if seq_len <= 128:
        steps, bs, lr = 300, 64, 5e-4
    elif seq_len <= 1024:
        steps, bs, lr = 220, 8, 3e-4
    else:
        steps, bs, lr = 70, 1, 2e-4

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    # baseline eval on fixed 128 sequences
    eval_seqs = sample_sequences(data, seq_len, NSEQ, seed=2026 + seq_len)
    eval_bs = 8 if seq_len <= 1024 else 1
    before = evaluate_model_outputs(model, eval_seqs, eval_bs)

    model.train()
    for step in range(steps):
        ix = torch.randint(len(data) - (seq_len - 1), (bs,))
        toks = []
        for i in ix:
            seq = np.concatenate(
                [[BOS_TOKEN_ID], data[i : i + seq_len - 1].astype(np.int64)]
            )
            toks.append(torch.from_numpy(seq))
        toks = torch.stack(toks).to(DEVICE)

        targets = torch.arange(seq_len, device=DEVICE).unsqueeze(0).repeat(bs, 1)
        _, loss = model(toks, targets=targets, capture_taps=False)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optim.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    step {step + 1}/{steps} loss={loss.item():.4f}")

    model.eval()
    after = evaluate_model_outputs(model, eval_seqs, eval_bs)
    meta = {"steps": steps, "batch_size": bs, "lr": lr, "eval_nseq": int(NSEQ)}
    return before, after, meta


def main() -> None:
    print(f"Device: {DEVICE}")
    data = load_data()

    out: dict[str, Any] = {
        "device": DEVICE,
        "nseq": int(NSEQ),
        "sample_pos_per_seq": int(SAMPLE_POS_PER_SEQ),
        "experiments": {},
        "tables": {},
    }

    # Tables 1 and 2
    for exp in EXPERIMENTS:
        print("\n" + "=" * 72)
        print(f"Collecting tables for {exp.name}")
        print(f"checkpoint={exp.checkpoint}")
        print("=" * 72)
        model, ckpt = load_model(exp.checkpoint)
        if ckpt["config"]["n_head"] != exp.n_head_expected:
            raise ValueError(
                f"Head mismatch for {exp.name}: got {ckpt['config']['n_head']}, expected {exp.n_head_expected}"
            )
        seqs = sample_sequences(data, exp.seq_len, NSEQ, seed=1000 + exp.seq_len)
        res = collect_model_tables(exp, model, seqs)
        out["experiments"][exp.name] = res
        d = res["delta_alignment"]
        print(
            f"rho mean/median={d['rho_seq_mean']:.4f}/{d['rho_seq_median']:.4f}, "
            f"p05/p95={d['rho_seq_p05']:.4f}/{d['rho_seq_p95']:.4f}"
        )

    # Table 3: finetune R2 block2-MLP + linear head
    table3 = []
    for exp in [e for e in EXPERIMENTS if e.is_r2]:
        print("\n" + "=" * 72)
        print(f"Finetune R2 MLP+head for {exp.name}")
        print("=" * 72)
        model, _ = load_model(exp.checkpoint)
        before, after, meta = finetune_r2_mlp_head(model, data, exp.seq_len)
        table3.append(
            {
                "model": exp.name,
                "L": exp.seq_len,
                "heads": exp.n_head_expected,
                "before": before,
                "after": after,
                "train_meta": meta,
            }
        )
        print(
            f"before r2_corr={before['r2_corr']:.4f}, mae={before['mae']:.2f} | "
            f"after r2_corr={after['r2_corr']:.4f}, mae={after['mae']:.2f}"
        )

    # Build compact tables
    table1 = []
    table2 = []
    for name, e in out["experiments"].items():
        da = e["delta_alignment"]
        dec = e["decode_o_minus_b"]
        table1.append(
            {
                "model": name,
                "L": e["seq_len"],
                "heads": e["n_head"],
                "Nseq": e["n_seq"],
                "rho_seq_mean_delta": da["rho_seq_mean"],
                "rho_seq_median_delta": da["rho_seq_median"],
                "rho_seq_p05_delta": da["rho_seq_p05"],
                "rho_seq_p95_delta": da["rho_seq_p95"],
                "pooled_linear_r2_delta": da["pooled_linear_r2"],
            }
        )
        table2.append(
            {
                "model": name,
                "L": e["seq_len"],
                "heads": e["n_head"],
                "linear_delta_r2": dec["linear_scalar_delta"]["r2_corr"],
                "linear_delta_mae": dec["linear_scalar_delta"]["mae"],
                "linear_delta_rmse": dec["linear_scalar_delta"]["rmse"],
                "mlp_scalar_delta_r2": dec["mlp_scalar_delta"]["r2_corr"],
                "mlp_scalar_delta_mae": dec["mlp_scalar_delta"]["mae"],
                "mlp_scalar_delta_rmse": dec["mlp_scalar_delta"]["rmse"],
                "mlp_o_minus_b_r2": dec["mlp_vector_o_minus_b"]["r2_corr"],
                "mlp_o_minus_b_mae": dec["mlp_vector_o_minus_b"]["mae"],
                "mlp_o_minus_b_rmse": dec["mlp_vector_o_minus_b"]["rmse"],
            }
        )

    out["tables"]["table1_per_sequence_spearman"] = table1
    out["tables"]["table2_decode_from_o_minus_b"] = table2
    out["tables"]["table3_r2_finetune_block2mlp_plus_head"] = table3

    save_path = (
        ROOT_DIR
        / "results/readout_axis_hypothesis_validation/nseq128_tables_extended.json"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved:", save_path)
    print("\nTable 1")
    for r in table1:
        print(r)
    print("\nTable 2")
    for r in table2:
        print(r)
    print("\nTable 3")
    for r in table3:
        print(r)


if __name__ == "__main__":
    main()
