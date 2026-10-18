"""Decode position from the scalar mechanism coordinate via an MLP readout.

For each model (Attn2-1H, Attn2-12H pure-R2; Full-12H R0) at L=1024:
  1. Compute the scalar mechanism coordinate s_i = <o_i^(2) - b_attn, d_Delta>.
  2. Train a small MLP regressor  s_i -> position  on a train split of sequences.
  3. On a held-out test split, plot MLP-decoded position vs. true position for 20
     individual samples (one panel per model), plus the identity line.

This is the per-sample, out-of-sample version of the "MLP scalar" readout: it
shows how completely the single two-direction coordinate decodes position once a
nonlinear head inverts its (monotone, nonlinear) dependence on i.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nanoGPT"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "experiments" / "r0_analysis"))

from r0_attention_output_projection import load_model, load_data, sample_tokens  # noqa: E402

RES_DIR = ROOT / "results" / "r0_attention_output_projection"
PAPER_DIRS = [
    ROOT / "overleaf" / "nopos_nips" / "plots" / "r0_attention_output_projection",
    ROOT / "overleaf" / "nopos_nips" / "plots",
]
DATA_DIR = ROOT / "nanoGPT" / "data" / "openwebtext"

N_SEQ = 320
N_TRAIN = 256
N_PLOT = 20
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


def scalar_coordinate(checkpoint: str, device: str) -> tuple[np.ndarray, int]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = load_model(ROOT / checkpoint, device)
    data = load_data(DATA_DIR)
    T = model.config.block_size
    d_model = model.config.n_embd
    n_batches = (N_SEQ + BATCH_SIZE - 1) // BATCH_SIZE

    attn_chunks, ov_chunks = [], []
    for _ in range(n_batches):
        tokens = sample_tokens(data, T, BATCH_SIZE, device)
        with torch.no_grad():
            model(tokens, capture_taps=True)
            taps = model.get_all_taps()
        W_V = model.block2.attn.c_attn.weight[2 * d_model :, :].detach()
        W_O = model.block2.attn.c_proj.weight.detach()
        b_O = model.block2.attn.c_proj.bias.detach()
        ln2_1 = taps["block2_ln1"]
        attn_out = taps["block2_attn"]
        Wo_v = (ln2_1 @ W_V.T) @ W_O.T
        attn_chunks.append((attn_out - b_O).detach().cpu())
        ov_chunks.append(Wo_v.detach().cpu())

    attn = torch.cat(attn_chunks, 0).float()[:N_SEQ]
    ov = torch.cat(ov_chunks, 0).float()[:N_SEQ]
    # Directions estimated from the TRAIN split only (no test leakage).
    bos_write = ov[:N_TRAIN, 0, :].mean(0)
    non_bos_write = ov[:N_TRAIN, 1:, :].mean((0, 1))
    delta_dir = F.normalize(non_bos_write - bos_write, dim=0, eps=1e-12)
    s = (attn @ delta_dir).numpy()  # [N, T]
    return s, T


def fit_and_plot(ax, name: str, s: np.ndarray, T: int) -> float:
    positions = np.arange(T, dtype=np.float64)
    pos_norm = positions / max(T - 1, 1)

    s_train = s[:N_TRAIN].reshape(-1, 1)
    y_train = np.tile(pos_norm, N_TRAIN)
    s_test = s[N_TRAIN:]
    y_test_norm = np.tile(pos_norm, s_test.shape[0])

    scaler = StandardScaler().fit(s_train)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        max_iter=3000,
        random_state=0,
        early_stopping=True,
        n_iter_no_change=25,
    )
    mlp.fit(scaler.transform(s_train), y_train)

    pred_test_norm = mlp.predict(scaler.transform(s_test.reshape(-1, 1)))
    test_r2 = r2_score(pred_test_norm, y_test_norm)

    pred_test = pred_test_norm.reshape(s_test.shape) * (T - 1)  # denormalize to indices
    for i in range(min(N_PLOT, pred_test.shape[0])):
        ax.plot(positions, pred_test[i], color="#0072B2", lw=0.6, alpha=0.4)
    ax.plot([0, T - 1], [0, T - 1], color="black", lw=1.3, ls="--", label="identity")
    ax.plot(positions, pred_test[:N_PLOT].mean(0), color="#D55E00", lw=1.6,
            label=f"mean decode")
    ax.set_title(f"{name} — MLP(scalar) decode (test R²={test_r2:.3f})", fontsize=9.5)
    ax.set_xlabel("True position")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    return test_r2


def save(fig, name: str) -> None:
    pdf = RES_DIR / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    for d in PAPER_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.pdf").write_bytes(pdf.read_bytes())
        (d / f"{name}.png").write_bytes(pdf.with_suffix(".png").read_bytes())
    print(f"saved {name}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, (name, ckpt) in zip(axes, MODELS):
        s, T = scalar_coordinate(ckpt, device)
        r2 = fit_and_plot(ax, name, s, T)
        print(f"{name:<10} MLP(scalar) test R2 = {r2:.4f}")
    axes[0].set_ylabel("MLP-decoded position")
    plt.tight_layout()
    save(fig, "mlp_scalar_decode_1h_12h_full12h")
    plt.close(fig)


if __name__ == "__main__":
    main()
