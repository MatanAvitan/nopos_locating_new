"""Explain the per-sample variance in MLP-decoded position.

Hypothesis: std(decoded position at i) ~ std(s_i) / |d mean_s / di|, i.e. noise in
the scalar coordinate s_i amplified by the inverse slope of the (saturating) s(i)
curve. We verify by comparing, per position, the empirical decoded-position std to
the analytic error-propagation prediction std(s_i)/|slope|.
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
from mlp_decode_from_scalar import scalar_coordinate, MODELS, N_TRAIN, N_SEQ  # noqa: E402

RES_DIR = ROOT / "results" / "r0_attention_output_projection"
PAPER_DIRS = [
    ROOT / "overleaf" / "nopos_nips" / "plots" / "r0_attention_output_projection",
    ROOT / "overleaf" / "nopos_nips" / "plots",
]
COLORS = {"Attn2-1H": "#D55E00", "Attn2-12H": "#0072B2", "Full-12H": "#009E73"}


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
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for name, ckpt in MODELS:
        s, T = scalar_coordinate(ckpt, device)        # [N, T]
        pos = np.arange(T, dtype=np.float64)
        pos_norm = pos / (T - 1)

        # ---- MLP readout (same as decode figure) ----
        s_tr = s[:N_TRAIN].reshape(-1, 1)
        y_tr = np.tile(pos_norm, N_TRAIN)
        scaler = StandardScaler().fit(s_tr)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                           random_state=0, early_stopping=True, n_iter_no_change=25)
        mlp.fit(scaler.transform(s_tr), y_tr)
        s_te = s[N_TRAIN:]
        pred = mlp.predict(scaler.transform(s_te.reshape(-1, 1))).reshape(s_te.shape)
        decoded = pred * (T - 1)                       # [n_test, T] decoded index
        decoded_std = decoded.std(0)                   # empirical std per position

        # ---- analytic prediction: std(s_i) / |d mean_s/di| ----
        # Smooth the mean curve before differentiating (raw gradient is noisy and
        # its near-zero values make 1/slope explode).
        def smooth(a, w=31):
            k = np.ones(w) / w
            return np.convolve(np.pad(a, w // 2, mode="edge"), k, mode="valid")[: len(a)]

        mean_s = smooth(s.mean(0))
        slope = np.abs(np.gradient(mean_s, pos))
        slope = smooth(slope)
        s_std = smooth(s.std(0))
        # floor the slope at a small percentile so saturated tail stays finite
        slope = np.clip(slope, np.percentile(slope, 5), None)
        analytic_std = s_std / slope                  # index units

        c = COLORS[name]
        axes[0].plot(pos, s.std(0), color=c, lw=1.3, label=name)
        axes[1].plot(pos, slope, color=c, lw=1.3, label=name)
        axes[2].plot(pos, decoded_std, color=c, lw=1.7, label=f"{name} empirical")
        axes[2].plot(pos, analytic_std, color=c, lw=1.0, ls="--", alpha=0.85,
                     label=f"{name} std(s)/|slope|")

        for tag, q in [("median", 0.5)]:
            print(f"{name:<10} late/early decoded-std ratio "
                  f"(pos~900 / pos~50) = {decoded_std[900]/max(decoded_std[50],1e-9):.1f} | "
                  f"slope[50]/slope[900] = {slope[50]/max(slope[900],1e-12):.1f} | "
                  f"s_std[50]={s_std[50]:.3f} s_std[900]={s_std[900]:.3f}")

    axes[0].set_title("Input noise: std($s_i$) per position", fontsize=10)
    axes[1].set_title("Coordinate slope |d mean $s$/d$i$|", fontsize=10)
    axes[1].set_yscale("log")
    axes[2].set_title("Decoded-position std: empirical vs std($s$)/|slope|", fontsize=10)
    axes[2].set_ylim(0, 400)
    for ax in axes:
        ax.set_xlabel("Position")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    save(fig, "decode_variance_explanation_1h_12h_full12h")
    plt.close(fig)


if __name__ == "__main__":
    main()
