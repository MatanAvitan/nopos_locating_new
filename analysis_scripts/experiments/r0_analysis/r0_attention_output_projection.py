"""Generate R0 attention output projections at L=128 and L=4096."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
COLOR_BOS = "#009E73"
COLOR_OTHERS = "#D55E00"
COLOR_DELTA = "#0072B2"


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho that is robust to degenerate inputs."""
    rho = spearmanr(x, y).correlation
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity for 1D vectors with epsilon protection."""
    a = F.normalize(a, dim=0, eps=1e-12)
    b = F.normalize(b, dim=0, eps=1e-12)
    return float(torch.dot(a, b).item())


def summarize_projection_spearman(proj_seq: np.ndarray) -> dict[str, float]:
    """Summarize monotonicity at population and single-sequence levels.

    Args:
        proj_seq: [N, T] projection values per sequence and position.
    """
    n_sequences, seq_len = proj_seq.shape
    positions = np.arange(seq_len, dtype=np.float32)

    population_token_rho = _safe_spearman(
        proj_seq.reshape(-1), np.tile(positions, n_sequences)
    )
    population_mean_curve_rho = _safe_spearman(proj_seq.mean(axis=0), positions)

    seq_rhos = np.array(
        [_safe_spearman(proj_seq[idx], positions) for idx in range(n_sequences)],
        dtype=np.float64,
    )

    return {
        "population_token_rho": population_token_rho,
        "population_mean_curve_rho": population_mean_curve_rho,
        "sample_sequence_mean_rho": float(seq_rhos.mean()),
        "sample_sequence_median_rho": float(np.median(seq_rhos)),
        "sample_sequence_std_rho": float(seq_rhos.std(ddof=1)),
        "sample_sequence_min_rho": float(seq_rhos.min()),
        "sample_sequence_max_rho": float(seq_rhos.max()),
        "n_sequences": int(n_sequences),
    }


def load_model(checkpoint_path: Path, device: str) -> TwoLayerMechanismModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TwoLayerMechanismConfig(**filtered)

    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device).eval()
    # Force manual attention so weights can be captured for analysis
    model.block1.attn.use_flash = False
    model.block2.attn.use_flash = False
    return model


def load_data(data_dir: Path) -> np.memmap:
    return np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")


def sample_tokens(
    data: np.memmap, seq_len: int, batch_size: int, device: str
) -> torch.Tensor:
    ix = torch.randint(len(data) - (seq_len - 1), (batch_size,))
    tokens = []
    for i in ix:
        seq = np.concatenate(
            [[BOS_TOKEN_ID], data[i : i + seq_len - 1].astype(np.int64)]
        )
        tokens.append(torch.from_numpy(seq))
    return torch.stack(tokens).to(device)


def compute_projection(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    device: str,
) -> dict:
    d_model = model.config.n_embd
    attn_batches = []
    write_seq_batches = []
    alpha_seq_batches = []
    bos_write_batches = []
    non_bos_write_batches = []

    proj_bos_batches = []
    proj_others_batches = []
    proj_delta_batches = []

    for _ in range(n_batches):
        tokens = sample_tokens(data, seq_len, batch_size, device)
        with torch.no_grad():
            model(tokens, capture_taps=True)
            taps = model.get_all_taps()
            _, attn2 = model.get_attention_weights()

        W_V = model.block2.attn.c_attn.weight[2 * d_model :, :].detach()
        W_O = model.block2.attn.c_proj.weight.detach()
        b_O = model.block2.attn.c_proj.bias.detach()

        ln2_1 = taps["block2_ln1"]
        attn_out = taps["block2_attn"]

        v2 = ln2_1 @ W_V.T
        Wo_v = v2 @ W_O.T

        if attn2 is None:
            raise RuntimeError(
                "Missing block2 attention weights for projection analysis"
            )
        alpha_eff = attn2.mean(dim=1).detach().cpu()  # [B, T, T]

        bos_mean = Wo_v[:, 0, :].mean(dim=0)
        others_mean = Wo_v[:, 1:, :].mean(dim=(0, 1))
        attn_out_no_bias = attn_out - b_O

        attn_batches.append(attn_out_no_bias.detach().cpu())
        write_seq_batches.append(Wo_v.detach().cpu())
        alpha_seq_batches.append(alpha_eff)
        bos_write_batches.append(bos_mean.detach().cpu())
        non_bos_write_batches.append(others_mean.detach().cpu())

    write_seq = torch.cat(write_seq_batches, dim=0).float()  # [N, T, D]
    alpha_seq = torch.cat(alpha_seq_batches, dim=0).float()  # [N, T, T]

    bos_write = write_seq[:, 0, :].mean(dim=0)
    non_bos_write = write_seq[:, 1:, :].mean(dim=(0, 1))
    delta_write = non_bos_write - bos_write

    bos_dir = F.normalize(bos_write, dim=0, eps=1e-12)
    others_dir = F.normalize(non_bos_write, dim=0, eps=1e-12)
    delta_dir = F.normalize(delta_write, dim=0, eps=1e-12)

    for attn_out_no_bias in attn_batches:
        proj_bos = (attn_out_no_bias @ bos_dir).numpy()  # [B, T]
        proj_others = (attn_out_no_bias @ others_dir).numpy()  # [B, T]
        proj_delta = (attn_out_no_bias @ delta_dir).numpy()  # [B, T]

        proj_bos_batches.append(proj_bos)
        proj_others_batches.append(proj_others)
        proj_delta_batches.append(proj_delta)

    proj_bos_seq = np.concatenate(proj_bos_batches, axis=0)
    proj_others_seq = np.concatenate(proj_others_batches, axis=0)
    proj_delta_seq = np.concatenate(proj_delta_batches, axis=0)

    proj_bos = proj_bos_seq.mean(axis=0)
    proj_others = proj_others_seq.mean(axis=0)
    proj_delta = proj_delta_seq.mean(axis=0)
    positions = np.arange(seq_len)

    corr_bos = float(np.corrcoef(proj_bos, positions)[0, 1])
    corr_non_bos = float(np.corrcoef(proj_others, positions)[0, 1])
    corr_delta = float(np.corrcoef(proj_delta, positions)[0, 1])
    spearman_bos = _safe_spearman(proj_bos, positions)
    spearman_non_bos = _safe_spearman(proj_others, positions)
    spearman_delta = _safe_spearman(proj_delta, positions)

    spearman_stats = {
        "bos": summarize_projection_spearman(proj_bos_seq),
        "non_bos": summarize_projection_spearman(proj_others_seq),
        "delta": summarize_projection_spearman(proj_delta_seq),
    }

    if hasattr(model, "pos_head") and getattr(model, "pos_head") is not None:
        w_head = model.pos_head.weight.detach().squeeze().cpu()
        cos_head_bos = _safe_cosine(w_head, bos_dir.cpu())
        cos_head_non_bos = _safe_cosine(w_head, others_dir.cpu())
        cos_head_delta = _safe_cosine(w_head, delta_dir.cpu())
    else:
        cos_head_bos = float("nan")
        cos_head_non_bos = float("nan")
        cos_head_delta = float("nan")

    cos_bos_non_bos = _safe_cosine(bos_dir.cpu(), others_dir.cpu())
    cos_non_bos_delta = _safe_cosine(others_dir.cpu(), delta_dir.cpu())
    cos_bos_delta = _safe_cosine(bos_dir.cpu(), delta_dir.cpu())

    angle_bos_non_bos = float(
        np.degrees(np.arccos(np.clip(cos_bos_non_bos, -1.0, 1.0)))
    )
    angle_non_bos_delta = float(
        np.degrees(np.arccos(np.clip(cos_non_bos_delta, -1.0, 1.0)))
    )
    angle_bos_delta = float(np.degrees(np.arccos(np.clip(cos_bos_delta, -1.0, 1.0))))

    # ---------------------------------------------------------------------
    # Step-4 concentration and projected residual diagnostics
    # ---------------------------------------------------------------------
    alpha_bos = alpha_seq[:, :, 0]  # [N, T]
    alpha_non_bos = alpha_seq[:, :, 1:]  # [N, T, T-1]
    non_bos_mass = 1.0 - alpha_bos  # [N, T]
    writes_non_bos = write_seq[:, 1:, :]  # [N, T-1, D]
    weighted_non_bos = torch.einsum("ntk,nkd->ntd", alpha_non_bos, writes_non_bos)

    delta_norm_sq = float(torch.dot(delta_write, delta_write).item())

    residual_vec = weighted_non_bos - non_bos_mass.unsqueeze(-1) * non_bos_write
    residual_proj = torch.einsum("ntd,d->nt", residual_vec, delta_write)  # [N, T]
    residual_proj_norm = residual_proj / (delta_norm_sq + 1e-12)

    conc_avg = weighted_non_bos / non_bos_mass.clamp_min(1e-9).unsqueeze(-1)
    conc_cos = torch.einsum(
        "ntd,d->nt",
        F.normalize(conc_avg, dim=-1, eps=1e-12),
        F.normalize(non_bos_write, dim=0, eps=1e-12),
    )

    valid_mask = non_bos_mass > 1e-6
    valid_mask[:, 0] = False  # i=0 has no non-BOS aggregate

    residual_proj_norm_np = residual_proj_norm.numpy()
    conc_cos_np = conc_cos.numpy()
    valid_np = valid_mask.numpy()

    residual_proj_norm_np[~valid_np] = np.nan
    conc_cos_np[~valid_np] = np.nan

    def _nan_mean_std_by_pos(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.full(arr.shape[1], np.nan, dtype=np.float64)
        std = np.full(arr.shape[1], np.nan, dtype=np.float64)
        for t in range(arr.shape[1]):
            vals = arr[:, t]
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                mean[t] = float(vals.mean())
                std[t] = float(vals.std(ddof=0))
        return mean, std

    residual_mean_by_pos, residual_std_by_pos = _nan_mean_std_by_pos(
        residual_proj_norm_np
    )
    conc_cos_mean_by_pos, conc_cos_std_by_pos = _nan_mean_std_by_pos(conc_cos_np)

    residual_abs_mean = float(np.nanmean(np.abs(residual_proj_norm_np)))
    residual_abs_median = float(np.nanmedian(np.abs(residual_proj_norm_np)))
    residual_abs_p95 = float(np.nanpercentile(np.abs(residual_proj_norm_np), 95))

    concentration_cos_mean = float(np.nanmean(conc_cos_np))
    concentration_cos_median = float(np.nanmedian(conc_cos_np))
    concentration_cos_p05 = float(np.nanpercentile(conc_cos_np, 5))
    concentration_cos_p95 = float(np.nanpercentile(conc_cos_np, 95))

    # Relative concentration error: ||avg - w_nonBOS|| / ||w_nonBOS||
    rel_err = torch.linalg.norm(conc_avg - non_bos_write, dim=-1) / (
        float(non_bos_write.norm().item()) + 1e-12
    )
    rel_err_np = rel_err.numpy()
    rel_err_np[~valid_np] = np.nan
    concentration_rel_err_mean = float(np.nanmean(rel_err_np))
    concentration_rel_err_median = float(np.nanmedian(rel_err_np))

    # More stable summary past early short-prefix positions.
    tail_start = min(16, seq_len - 1)
    concentration_cos_mean_curve_mean_ge16 = float(
        np.nanmean(conc_cos_mean_by_pos[tail_start:])
    )
    concentration_cos_mean_curve_min_ge16 = float(
        np.nanmin(conc_cos_mean_by_pos[tail_start:])
    )
    residual_abs_mean_curve_ge16 = float(
        np.nanmean(np.abs(residual_mean_by_pos[tail_start:]))
    )

    return {
        "proj_bos": proj_bos,
        "proj_non_bos": proj_others,
        "proj_delta": proj_delta,
        "corr_bos": corr_bos,
        "corr_non_bos": corr_non_bos,
        "corr_delta": corr_delta,
        "spearman_bos": spearman_bos,
        "spearman_non_bos": spearman_non_bos,
        "spearman_delta": spearman_delta,
        "spearman_stats": spearman_stats,
        "cos_bos_non_bos": cos_bos_non_bos,
        "cos_non_bos_delta": cos_non_bos_delta,
        "cos_bos_delta": cos_bos_delta,
        "angle_bos_non_bos_deg": angle_bos_non_bos,
        "angle_non_bos_delta_deg": angle_non_bos_delta,
        "angle_bos_delta_deg": angle_bos_delta,
        "cos_head_bos": cos_head_bos,
        "cos_head_non_bos": cos_head_non_bos,
        "cos_head_delta": cos_head_delta,
        "residual_proj_norm_mean_by_pos": residual_mean_by_pos,
        "residual_proj_norm_std_by_pos": residual_std_by_pos,
        "residual_proj_norm_abs_mean": residual_abs_mean,
        "residual_proj_norm_abs_median": residual_abs_median,
        "residual_proj_norm_abs_p95": residual_abs_p95,
        "concentration_cos_mean_by_pos": conc_cos_mean_by_pos,
        "concentration_cos_std_by_pos": conc_cos_std_by_pos,
        "concentration_cos_mean": concentration_cos_mean,
        "concentration_cos_median": concentration_cos_median,
        "concentration_cos_p05": concentration_cos_p05,
        "concentration_cos_p95": concentration_cos_p95,
        "concentration_rel_err_mean": concentration_rel_err_mean,
        "concentration_rel_err_median": concentration_rel_err_median,
        "concentration_cos_mean_curve_mean_ge16": concentration_cos_mean_curve_mean_ge16,
        "concentration_cos_mean_curve_min_ge16": concentration_cos_mean_curve_min_ge16,
        "residual_proj_norm_abs_mean_curve_ge16": residual_abs_mean_curve_ge16,
    }


def plot_projection_single(results: dict, seq_len: int, save_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.0))

    positions = np.arange(seq_len)
    ax.plot(
        positions,
        results["proj_bos"],
        color=COLOR_BOS,
        linewidth=1.2,
        label=f"BOS dir (\u03c1={results['spearman_bos']:.2f})",
    )
    ax.plot(
        positions,
        results["proj_non_bos"],
        color=COLOR_OTHERS,
        linewidth=1.2,
        label=f"non-BOS dir (\u03c1={results['spearman_non_bos']:.2f})",
    )
    ax.set_title("Attention Output Projection")
    ax.set_xlabel("Position")
    ax.set_ylabel("Projection")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_projection_axis_comparison(
    results: dict, seq_len: int, save_path: Path
) -> None:
    """Compare projections onto non-BOS and delta axes."""
    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.0))

    positions = np.arange(seq_len)
    ax.plot(
        positions,
        results["proj_non_bos"],
        color=COLOR_OTHERS,
        linewidth=1.4,
        label=f"non-BOS axis (rho={results['spearman_non_bos']:.2f})",
    )
    ax.plot(
        positions,
        results["proj_delta"],
        color=COLOR_DELTA,
        linewidth=1.4,
        label=f"delta axis (rho={results['spearman_delta']:.2f})",
    )

    ax.set_title("Axis Comparison")
    ax.set_xlabel("Position")
    ax.set_ylabel("Projection")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_concentration(results: dict, seq_len: int, save_path: Path) -> None:
    """Plot projected residual term and non-BOS concentration diagnostics."""
    positions = np.arange(seq_len)

    resid_mu = np.asarray(results["residual_proj_norm_mean_by_pos"], dtype=np.float64)
    resid_std = np.asarray(results["residual_proj_norm_std_by_pos"], dtype=np.float64)

    conc_mu = np.asarray(results["concentration_cos_mean_by_pos"], dtype=np.float64)
    conc_std = np.asarray(results["concentration_cos_std_by_pos"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    ax.plot(positions, resid_mu, color="#1f77b4", linewidth=1.4)
    ax.fill_between(
        positions,
        resid_mu - resid_std,
        resid_mu + resid_std,
        color="#1f77b4",
        alpha=0.18,
    )
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("Projected Residual")
    ax.set_xlabel("Position")
    ax.set_ylabel(r"$\langle r_i,\,\Delta\rangle/\|\Delta\|^2$")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(positions, conc_mu, color="#d62728", linewidth=1.4)
    ax.fill_between(
        positions,
        conc_mu - conc_std,
        conc_mu + conc_std,
        color="#d62728",
        alpha=0.18,
    )
    ax.set_title("Non-BOS Concentration")
    ax.set_xlabel("Position")
    ax.set_ylabel(r"$\cos(\bar{w}_{i,\mathrm{nonBOS}},\,w_{\mathrm{nonBOS}})$")
    ax.set_ylim(0.85, 1.01)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nanoGPT/out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/r0_attention_output_projection"
    )
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--output_tag", type=str, default="full12h")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = ROOT_DIR / args.checkpoint
    data_dir = ROOT_DIR / args.data_dir

    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_projection_subdir = paper_dir / "r0_attention_output_projection"
    paper_projection_subdir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, device)
    data = load_data(data_dir)

    seq_len = model.config.block_size
    print(f"Computing projections at L={seq_len}...")
    results = compute_projection(
        model, data, seq_len, args.batch_size, args.n_batches, device
    )

    save_path = save_dir / f"attention_output_projection_{args.output_tag}.pdf"
    plot_projection_single(results, seq_len, save_path)

    axis_compare_path = (
        save_dir / f"attention_output_axis_compare_{args.output_tag}.pdf"
    )
    plot_projection_axis_comparison(results, seq_len, axis_compare_path)

    residual_conc_path = (
        save_dir / f"attention_output_residual_concentration_{args.output_tag}.pdf"
    )
    plot_residual_concentration(results, seq_len, residual_conc_path)

    def to_serializable(value: Any):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, dict):
            return {k: to_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [to_serializable(v) for v in value]
        return value

    with open(save_dir / "projection_results.json", "w") as f:
        json.dump(
            to_serializable({f"L{seq_len}": results}),
            f,
            indent=2,
        )

    with open(save_dir / f"projection_results_{args.output_tag}.json", "w") as f:
        json.dump(
            to_serializable({f"L{seq_len}": results}),
            f,
            indent=2,
        )

    paper_path = paper_dir / save_path.name
    if save_path.exists():
        paper_path.write_bytes(save_path.read_bytes())
    paper_path_subdir = paper_projection_subdir / save_path.name
    if save_path.exists():
        paper_path_subdir.write_bytes(save_path.read_bytes())
    paper_path_png = paper_dir / save_path.with_suffix(".png").name
    if save_path.with_suffix(".png").exists():
        paper_path_png.write_bytes(save_path.with_suffix(".png").read_bytes())
    paper_path_png_subdir = paper_projection_subdir / save_path.with_suffix(".png").name
    if save_path.with_suffix(".png").exists():
        paper_path_png_subdir.write_bytes(save_path.with_suffix(".png").read_bytes())

    paper_axis_path = paper_dir / axis_compare_path.name
    if axis_compare_path.exists():
        paper_axis_path.write_bytes(axis_compare_path.read_bytes())
    paper_axis_path_subdir = paper_projection_subdir / axis_compare_path.name
    if axis_compare_path.exists():
        paper_axis_path_subdir.write_bytes(axis_compare_path.read_bytes())
    paper_axis_path_png = paper_dir / axis_compare_path.with_suffix(".png").name
    if axis_compare_path.with_suffix(".png").exists():
        paper_axis_path_png.write_bytes(
            axis_compare_path.with_suffix(".png").read_bytes()
        )
    paper_axis_path_png_subdir = (
        paper_projection_subdir / axis_compare_path.with_suffix(".png").name
    )
    if axis_compare_path.with_suffix(".png").exists():
        paper_axis_path_png_subdir.write_bytes(
            axis_compare_path.with_suffix(".png").read_bytes()
        )

    paper_resid_path = paper_dir / residual_conc_path.name
    if residual_conc_path.exists():
        paper_resid_path.write_bytes(residual_conc_path.read_bytes())
    paper_resid_path_subdir = paper_projection_subdir / residual_conc_path.name
    if residual_conc_path.exists():
        paper_resid_path_subdir.write_bytes(residual_conc_path.read_bytes())
    paper_resid_path_png = paper_dir / residual_conc_path.with_suffix(".png").name
    if residual_conc_path.with_suffix(".png").exists():
        paper_resid_path_png.write_bytes(
            residual_conc_path.with_suffix(".png").read_bytes()
        )
    paper_resid_path_png_subdir = (
        paper_projection_subdir / residual_conc_path.with_suffix(".png").name
    )
    if residual_conc_path.with_suffix(".png").exists():
        paper_resid_path_png_subdir.write_bytes(
            residual_conc_path.with_suffix(".png").read_bytes()
        )

    print("Saved:")
    print(f"  {save_path}")
    print(f"  {paper_path}")
    print(f"  {paper_path_subdir}")
    print(f"  {axis_compare_path}")
    print(f"  {paper_axis_path}")
    print(f"  {paper_axis_path_subdir}")
    print(f"  {residual_conc_path}")
    print(f"  {paper_resid_path}")
    print(f"  {paper_resid_path_subdir}")
    print("Summary:")
    stats_bos = results["spearman_stats"]["bos"]
    stats_non = results["spearman_stats"]["non_bos"]
    print(
        f"  L={seq_len} mean-curve Spearman: BOS={results['spearman_bos']:.3f}, "
        f"non-BOS={results['spearman_non_bos']:.3f}"
    )
    print(
        f"  L={seq_len} sample-level median Spearman: BOS={stats_bos['sample_sequence_median_rho']:.3f}, "
        f"non-BOS={stats_non['sample_sequence_median_rho']:.3f}"
    )
    print(
        f"  L={seq_len} mean-curve Spearman (delta axis): {results['spearman_delta']:.3f}"
    )
    print(
        f"  cos(d_BOS, d_nonBOS)={results['cos_bos_non_bos']:.3f} "
        f"(angle={results['angle_bos_non_bos_deg']:.2f} deg)"
    )
    print(
        f"  cos(d_nonBOS, d_delta)={results['cos_non_bos_delta']:.3f} "
        f"(angle={results['angle_non_bos_delta_deg']:.2f} deg)"
    )
    print(
        f"  cos(w_head, d_nonBOS)={results['cos_head_non_bos']:.3f}, "
        f"cos(w_head, d_delta)={results['cos_head_delta']:.3f}"
    )
    print(
        f"  residual |<r_i,Delta>|/||Delta||^2: mean={results['residual_proj_norm_abs_mean']:.4f}, "
        f"median={results['residual_proj_norm_abs_median']:.4f}, "
        f"p95={results['residual_proj_norm_abs_p95']:.4f}"
    )
    print(
        f"  concentration cos(mean/median/p05)="
        f"{results['concentration_cos_mean']:.4f}/"
        f"{results['concentration_cos_median']:.4f}/"
        f"{results['concentration_cos_p05']:.4f}"
    )
    print(
        f"  concentration rel-error mean/median="
        f"{results['concentration_rel_err_mean']:.4f}/"
        f"{results['concentration_rel_err_median']:.4f}"
    )
    print(
        f"  concentration mean-curve (i>=16): mean={results['concentration_cos_mean_curve_mean_ge16']:.4f}, "
        f"min={results['concentration_cos_mean_curve_min_ge16']:.4f}, "
        f"residual-abs-mean={results['residual_proj_norm_abs_mean_curve_ge16']:.4f}"
    )


if __name__ == "__main__":
    main()
