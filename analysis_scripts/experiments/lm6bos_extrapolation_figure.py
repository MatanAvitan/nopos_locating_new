"""Paper figure: lm6bos length extrapolation (P6).

One panel at L=2048 (16x the trained context): frozen (trained-range) vs
calibrated linear probe on the layer-2 residual. Ordering survives at every
unseen position, the frozen readout saturates, and the calibrated probe still
reads position from the same activations.

The earlier (a) schedule-departure and (c) schedule-enforcement panels were
dropped from the paper figure; the numbers they showed are quoted in the text.

Writes plots/lm6bos_extrapolation.{pdf,png} to the paper repo.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRAP = ROOT / "results/mechanism/lm6bos_extrap_L2048"
PAPER_PLOTS = ROOT / "overleaf/nopos_nips/plots"

COLOR_BOS = "#009E73"
COLOR_OTHERS = "#D55E00"
TRAIN_L = 128
L = 2048


def main():
    frozen = np.load(EXTRAP / "L1__frozen_pred.npy")
    calibrated = np.load(EXTRAP / "L1__oracle_pred.npy")

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.7))
    i = np.arange(1, L)

    ax.plot(i, i, color="gray", lw=0.8, ls=":", label="identity")
    ax.plot(i, np.nanmedian(frozen, 0)[1:], color=COLOR_OTHERS, lw=1.5,
            label="frozen probe (fit $i<128$)")
    ax.plot(i, np.nanmedian(calibrated, 0)[1:], color=COLOR_BOS, lw=1.5,
            label="calibrated probe")
    ax.axvline(TRAIN_L, color="gray", lw=0.8, ls=":")
    ax.text(TRAIN_L * 1.25, 800, "trained\nhorizon", fontsize=7, color="gray")
    ax.set_xlabel("true position $i$", fontsize=8)
    ax.set_ylabel("predicted position", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    PAPER_PLOTS.mkdir(parents=True, exist_ok=True)
    out = PAPER_PLOTS / "lm6bos_extrapolation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
