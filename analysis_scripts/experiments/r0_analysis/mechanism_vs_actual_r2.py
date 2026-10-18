"""Actual model R^2 vs. mechanism R^2 (single Delta-coordinate).

For each model (Attn2-1H, Attn2-12H pure-R2; Full-12H R0) at L=1024:
  - actual_r2  : R^2 of the model's own trained position head on the full final
                 representation (everything the model actually uses).
  - mech_lin   : R^2 of a LINEAR readout from the scalar mechanism coordinate
                 s_i = <o_i^(2) - b_attn, d_Delta>  (one number per token).
  - mech_poly  : R^2 of a degree-5 (monotone-ish) nonlinear readout from s_i,
                 a proxy for the Step-5 nonlinear head, since s_i is nonlinear in i.

The gap (actual_r2 - mech_*) is the part of position decoding NOT carried by the
single two-direction interpolation coordinate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nanoGPT"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "experiments" / "r0_analysis"))

from r0_attention_output_projection import load_model, load_data, sample_tokens  # noqa: E402

DATA_DIR = ROOT / "nanoGPT" / "data" / "openwebtext"
N_SEQ = 320
BATCH_SIZE = 32
SEED = 1234

MODELS = [
    ("Attn2-1H", "nanoGPT/out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt"),
    ("Attn2-12H", "nanoGPT/out-mechanism-R2-12h-1024/R2/pb1sxvzm/best_ckpt.pt"),
    ("Full-12H", "nanoGPT/out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt"),
]


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def fit_r2_poly(x: np.ndarray, y: np.ndarray, deg: int) -> float:
    """In-sample R^2 of a degree-`deg` polynomial readout x -> y."""
    coef = np.polyfit(x, y, deg)
    pred = np.polyval(coef, x)
    return r2_score(pred, y)


def evaluate(checkpoint: str, device: str) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = load_model(ROOT / checkpoint, device)
    data = load_data(DATA_DIR)
    T = model.config.block_size
    d_model = model.config.n_embd
    n_batches = (N_SEQ + BATCH_SIZE - 1) // BATCH_SIZE

    pred_chunks, attn_chunks, ov_chunks = [], [], []
    for _ in range(n_batches):
        tokens = sample_tokens(data, T, BATCH_SIZE, device)
        with torch.no_grad():
            output, _ = model(tokens, capture_taps=True)
            taps = model.get_all_taps()
        preds = output.squeeze(-1)                       # [B, T] model position preds
        W_V = model.block2.attn.c_attn.weight[2 * d_model :, :].detach()
        W_O = model.block2.attn.c_proj.weight.detach()
        b_O = model.block2.attn.c_proj.bias.detach()
        ln2_1 = taps["block2_ln1"]
        attn_out = taps["block2_attn"]
        Wo_v = (ln2_1 @ W_V.T) @ W_O.T
        pred_chunks.append(preds.detach().cpu())
        attn_chunks.append((attn_out - b_O).detach().cpu())
        ov_chunks.append(Wo_v.detach().cpu())

    preds = torch.cat(pred_chunks, 0).float()[:N_SEQ]    # [N, T]
    attn = torch.cat(attn_chunks, 0).float()[:N_SEQ]     # [N, T, D]
    ov = torch.cat(ov_chunks, 0).float()[:N_SEQ]

    bos_write = ov[:, 0, :].mean(0)
    non_bos_write = ov[:, 1:, :].mean((0, 1))
    delta_dir = F.normalize(non_bos_write - bos_write, dim=0, eps=1e-12)
    s = (attn @ delta_dir).numpy()                       # [N, T] mechanism coordinate

    positions = np.tile(np.arange(T, dtype=np.float64), N_SEQ)
    # Model head predicts position normalized to [0, 1] (targets = arange(T)/(T-1)).
    positions_norm = positions / max(T - 1, 1)
    s_flat = s.reshape(-1).astype(np.float64)
    pred_flat = preds.numpy().reshape(-1).astype(np.float64)

    return {
        "L": T,
        "actual_r2": r2_score(pred_flat, positions_norm),
        "mech_lin_r2": fit_r2_poly(s_flat, positions, 1),
        "mech_poly_r2": fit_r2_poly(s_flat, positions, 5),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'Model':<10} {'actual R2':>10} {'mech lin':>10} {'mech poly5':>11} "
          f"{'gap(lin)':>10} {'gap(poly)':>10}")
    for name, ckpt in MODELS:
        r = evaluate(ckpt, device)
        print(f"{name:<10} {r['actual_r2']:>10.4f} {r['mech_lin_r2']:>10.4f} "
              f"{r['mech_poly_r2']:>11.4f} "
              f"{r['actual_r2']-r['mech_lin_r2']:>10.4f} "
              f"{r['actual_r2']-r['mech_poly_r2']:>10.4f}")


if __name__ == "__main__":
    main()
