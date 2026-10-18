"""Step-4 attention-output projections for INDIVIDUAL samples (not averaged).

For each of Attn2-1H, Attn2-12H (both pure R2: only Layer-2 attention + linear
head trained) and Full-12H (R0: fully trained), plot the per-sequence projection
of the Layer-2 attention output o_i^(2) onto the BOS and non-BOS OV directions,
for 20 individual held-out sequences. The BOS/non-BOS directions are estimated
from all collected sequences; only individual trajectories are drawn.

Purpose: check whether the two-direction interpolation accounts for per-sample
position decoding, or whether individual samples leave residual scatter/noise
around the monotone mean trend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

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

COLOR_BOS = "#009E73"
COLOR_OTHERS = "#D55E00"

MODELS = [
    ("Attn2-1H", "nanoGPT/out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt"),
    ("Attn2-12H", "nanoGPT/out-mechanism-R2-12h-1024/R2/pb1sxvzm/best_ckpt.pt"),
    ("Full-12H", "nanoGPT/out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt"),
]

N_DIR_SEQ = 320      # sequences used to estimate the BOS / non-BOS directions
N_PLOT = 20          # individual sample trajectories to draw
BATCH_SIZE = 32
SEED = 1234


def per_sample_projections(checkpoint: str, device: str) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = load_model(ROOT / checkpoint, device)
    data = load_data(DATA_DIR)
    seq_len = model.config.block_size
    d_model = model.config.n_embd
    n_batches = (N_DIR_SEQ + BATCH_SIZE - 1) // BATCH_SIZE

    attn_chunks = []   # o_i^(2) - b_attn, per token
    ov_chunks = []     # B_OV x_j^(2), per token
    for _ in range(n_batches):
        tokens = sample_tokens(data, seq_len, BATCH_SIZE, device)
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

    attn = torch.cat(attn_chunks, dim=0).float()[:N_DIR_SEQ]   # [N, T, D]
    ov = torch.cat(ov_chunks, dim=0).float()[:N_DIR_SEQ]       # [N, T, D]

    bos_write = ov[:, 0, :].mean(0)
    non_bos_write = ov[:, 1:, :].mean((0, 1))
    bos_dir = F.normalize(bos_write, dim=0, eps=1e-12)
    non_bos_dir = F.normalize(non_bos_write, dim=0, eps=1e-12)
    # The decoding axis: difference of the two OV writes (Eq. head_projection_decomposition).
    delta_dir = F.normalize(non_bos_write - bos_write, dim=0, eps=1e-12)

    proj_bos = (attn @ bos_dir).numpy()        # [N, T]
    proj_non = (attn @ non_bos_dir).numpy()    # [N, T]
    proj_delta = (attn @ delta_dir).numpy()    # [N, T]
    positions = np.arange(seq_len)

    def sample_rhos(arr):
        return np.array([spearmanr(arr[i], positions).correlation for i in range(arr.shape[0])])

    return {
        "name": None,
        "L": seq_len,
        "proj_bos": proj_bos,
        "proj_non": proj_non,
        "proj_delta": proj_delta,
        "positions": positions,
        "rho_bos": sample_rhos(proj_bos),
        "rho_non": sample_rhos(proj_non),
        "rho_delta": sample_rhos(proj_delta),
    }


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
    results = []
    for name, ckpt in MODELS:
        print(f"=== {name} ===")
        r = per_sample_projections(ckpt, device)
        r["name"] = name
        print(
            f"  L={r['L']}  per-sample Spearman BOS: "
            f"median={np.median(r['rho_bos']):.3f} min={r['rho_bos'].min():.3f} | "
            f"non-BOS: median={np.median(r['rho_non']):.3f} min={r['rho_non'].min():.3f} | "
            f"DELTA: median={np.median(r['rho_delta']):.3f} min={r['rho_delta'].min():.3f}"
        )
        results.append(r)

    # Decoding-axis figure: per-sample projection onto d_delta (one panel per model).
    figd, axesd = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for col, r in enumerate(results):
        pos = r["positions"]
        ax = axesd[col]
        for i in range(N_PLOT):
            ax.plot(pos, r["proj_delta"][i], color="#0072B2", lw=0.6, alpha=0.35)
        ax.plot(pos, r["proj_delta"][:N_PLOT].mean(0), color="black", lw=1.6,
                label=f"mean (med ρ={np.median(r['rho_delta']):.3f}, "
                      f"min={r['rho_delta'].min():.3f})")
        ax.set_title(f"{r['name']} — Δ (decoding) axis", fontsize=10)
        ax.set_xlabel("Position")
        if col == 0:
            ax.set_ylabel(r"$\langle o_i^{(2)}-b,\ d_\Delta\rangle$")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    save(figd, "step4_individual_samples_delta_axis_1h_12h_full12h")
    plt.close(figd)

    # ------------------------------------------------------------------
    # Quantify residual scatter on the decoding (Delta) axis.
    # For each model: at every position, take the std across all 320 samples
    # around the mean Delta-curve; normalize by the curve's range (max-min).
    # Also report an aggregate "unexplained-variance fraction":
    #   1 - Var(mean-curve) / Var(all per-sample values), i.e. the share of
    #   total Delta-projection variance NOT captured by the position-mean curve.
    # ------------------------------------------------------------------
    print("\n=== Residual scatter on the decoding (Delta) axis (n=320 samples) ===")
    figr, axr = plt.subplots(1, 1, figsize=(5.2, 3.4))
    colors = {"Attn2-1H": "#D55E00", "Attn2-12H": "#0072B2", "Full-12H": "#009E73"}
    for r in results:
        d = r["proj_delta"]                       # [N, T]
        pos = r["positions"]
        mean_curve = d.mean(0)                     # [T]
        curve_range = float(mean_curve.max() - mean_curve.min())
        resid_std = d.std(0)                       # [T] std across samples per position
        norm_resid = resid_std / (curve_range + 1e-12)

        total_var = float(d.var())                # variance over all (sample, pos)
        explained_var = float(mean_curve.var())   # variance of the position-mean curve
        unexplained_frac = 1.0 - explained_var / (total_var + 1e-12)

        # tail summary (skip very early short-prefix positions)
        tail = slice(16, None)
        print(
            f"  {r['name']:<10} normalized residual std: "
            f"mean={np.mean(norm_resid[tail]):.4f} median={np.median(norm_resid[tail]):.4f} "
            f"p95={np.percentile(norm_resid[tail],95):.4f} | "
            f"unexplained-var frac={unexplained_frac:.4f}"
        )
        axr.plot(pos, norm_resid, color=colors.get(r["name"], "gray"), lw=1.3,
                 label=f"{r['name']} (uvf={unexplained_frac:.3f})")
    axr.set_title("Per-position residual scatter on Δ axis", fontsize=10)
    axr.set_xlabel("Position")
    axr.set_ylabel("std across samples / curve range")
    axr.grid(True, alpha=0.2)
    axr.legend(fontsize=8, loc="best")
    plt.tight_layout()
    save(figr, "step4_delta_residual_scatter_1h_12h_full12h")
    plt.close(figr)

    # 3 rows (models) x 2 cols (BOS axis, non-BOS axis); 20 individual samples each.
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.4))
    for row, r in enumerate(results):
        pos = r["positions"]
        for col, (proj, rho, color, axis_name) in enumerate(
            [
                (r["proj_bos"], r["rho_bos"], COLOR_BOS, "BOS"),
                (r["proj_non"], r["rho_non"], COLOR_OTHERS, "non-BOS"),
            ]
        ):
            ax = axes[row, col]
            for i in range(N_PLOT):
                ax.plot(pos, proj[i], color=color, lw=0.6, alpha=0.35)
            ax.plot(pos, proj[:N_PLOT].mean(0), color="black", lw=1.6,
                    label=f"mean (med ρ={np.median(rho):.3f})")
            ax.set_title(f"{r['name']} — {axis_name} axis", fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc="best")
            if row == 2:
                ax.set_xlabel("Position")
            if col == 0:
                ax.set_ylabel("Projection")
    plt.tight_layout()
    save(fig, "step4_individual_samples_1h_12h_full12h")
    plt.close(fig)


if __name__ == "__main__":
    main()
