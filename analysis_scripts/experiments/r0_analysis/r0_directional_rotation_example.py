"""Directional rotation dial for R0-12head (positions 0..127)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel

BOS_TOKEN_ID = 50256
COLOR_PATH = "#0B4F8A"
COLOR_FACE = "#F6F8FB"
COLOR_TRACK = "#B8C6DB"
COLOR_STRIDE = "#1E293B"
CMAP_NAME = "turbo"


def load_model(checkpoint_path: Path, device: str) -> TwoLayerMechanismModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    config = TwoLayerMechanismConfig(
        block_size=config_dict["block_size"],
        vocab_size=config_dict["vocab_size"],
        n_embd=config_dict["n_embd"],
        n_head=config_dict["n_head"],
        dropout=0.0,
        norm_type=config_dict["norm_type"],
        bias=True,
        use_regression=True,
    )

    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_data(data_dir: Path) -> np.memmap:
    return np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")


def sample_sequences(
    data: np.memmap,
    seq_len: int,
    n_sequences: int,
    device: str,
    seed: int,
    start_idx: int | None,
) -> tuple[torch.Tensor, list[int]]:
    rng = np.random.RandomState(seed)
    max_start = len(data) - (seq_len - 1)
    if max_start <= 0:
        raise ValueError(f"Data too short for seq_len={seq_len}")

    indices = rng.randint(0, max_start, size=n_sequences).tolist()
    if start_idx is not None:
        indices[0] = int(start_idx)

    tokens = []
    for idx in indices:
        seq = np.concatenate(
            [[BOS_TOKEN_ID], data[idx : idx + seq_len - 1].astype(np.int64)]
        )
        tokens.append(torch.from_numpy(seq))

    return torch.stack(tokens).to(device), indices


def compute_rotation(model: TwoLayerMechanismModel, tokens: torch.Tensor) -> dict:
    D = model.config.n_embd
    with torch.no_grad():
        model(tokens, capture_taps=True)
        taps = model.get_all_taps()

    W_V = model.block2.attn.c_attn.weight[2 * D :, :].detach()
    W_O = model.block2.attn.c_proj.weight.detach()
    b_O = model.block2.attn.c_proj.bias.detach()

    ln2_1 = taps["block2_ln1"]
    attn_out = taps["block2_attn"]

    v2 = ln2_1 @ W_V.T
    Wo_v = v2 @ W_O.T

    bos_mean = Wo_v[:, 0, :].mean(dim=0)
    others_mean = Wo_v[:, 1:, :].mean(dim=(0, 1))
    bos_dir = F.normalize(bos_mean, dim=0)
    others_dir = F.normalize(others_mean, dim=0)

    attn_out_no_bias = attn_out - b_O
    proj_bos = (attn_out_no_bias @ bos_dir).mean(dim=0).cpu().numpy()
    proj_others = (attn_out_no_bias @ others_dir).mean(dim=0).cpu().numpy()

    angle = np.degrees(np.arctan2(proj_others, proj_bos))

    attn_unit = F.normalize(attn_out_no_bias, dim=-1)
    cos_bos = (attn_unit @ bos_dir).mean(dim=0).cpu().numpy()
    cos_others = (attn_unit @ others_dir).mean(dim=0).cpu().numpy()
    cos_bos_others = float(
        F.cosine_similarity(bos_dir.unsqueeze(0), others_dir.unsqueeze(0)).item()
    )
    bos_others_angle = float(np.degrees(np.arccos(np.clip(cos_bos_others, -1.0, 1.0))))
    theta_bos = np.degrees(np.arccos(np.clip(cos_bos, -1.0, 1.0)))
    dial_angle = theta_bos / bos_others_angle * 180.0
    dial_angle = np.clip(dial_angle, 0.0, 180.0)

    return {
        "proj_bos": proj_bos,
        "proj_others": proj_others,
        "angle_deg": angle,
        "dial_angle_deg": dial_angle,
        "cos_bos": cos_bos,
        "cos_others": cos_others,
        "bos_dir_norm": float(bos_mean.norm().item()),
        "others_dir_norm": float(others_mean.norm().item()),
        "cos_bos_others": cos_bos_others,
        "bos_others_angle_deg": bos_others_angle,
    }


def plot_rotation(
    rotation: dict, stride: int, save_path: Path, label_stride: int = 16
) -> None:
    proj_bos = rotation["proj_bos"]
    proj_others = rotation["proj_others"]
    dial_angle = rotation["dial_angle_deg"]
    positions = np.arange(len(proj_bos))
    n_positions = len(positions)
    dial_theta = np.radians(dial_angle)
    dial_x = -np.cos(dial_theta)
    dial_y = np.sin(dial_theta)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.4))
    ax.set_facecolor(COLOR_FACE)

    # Half-gauge dial frame (upper semicircle only).
    theta_top = np.linspace(0, np.pi, 400)

    # -- Gauge aesthetics: shadow arc behind main track --
    ax.plot(
        -np.cos(theta_top),
        np.sin(theta_top),
        color="#9CADC4",
        linewidth=5.0,
        alpha=0.35,
        zorder=0,
        solid_capstyle="round",
    )
    # Main track arc.
    ax.plot(
        -np.cos(theta_top),
        np.sin(theta_top),
        color=COLOR_TRACK,
        linewidth=2.5,
        alpha=0.95,
        zorder=1,
    )
    # -- Gauge aesthetics: thin inner arc for depth --
    inner_r = 0.88
    ax.plot(
        inner_r * -np.cos(theta_top),
        inner_r * np.sin(theta_top),
        color=COLOR_TRACK,
        linewidth=0.6,
        alpha=0.45,
        zorder=1,
    )

    # -- Gauge aesthetics: selected tick marks --
    tick_label_positions = [p for p in (16, 64, 80, 96, 112) if 0 <= p < n_positions]
    tick_inner = 0.93
    tick_outer = 1.00
    for p in tick_label_positions:
        x_u = dial_x[p]
        y_u = dial_y[p]
        ax.plot(
            [tick_inner * x_u, tick_outer * x_u],
            [tick_inner * y_u, tick_outer * y_u],
            color="#64748B",
            linewidth=1.0,
            alpha=0.7,
            zorder=2,
        )

    # Stride anchor positions for markers and arrows.
    major_positions = np.arange(0, n_positions, stride)
    if major_positions[-1] != n_positions - 1:
        major_positions = np.append(major_positions, n_positions - 1)

    # Gradient trajectory line interpolated along the arc so it follows the
    # semicircle instead of cutting straight chords between points.
    n_interp = 8  # sub-steps between consecutive positions
    arc_segments = []
    arc_colors = []
    for i in range(len(dial_theta) - 1):
        th = np.linspace(dial_theta[i], dial_theta[i + 1], n_interp + 1)
        xs = -np.cos(th)
        ys = np.sin(th)
        for j in range(len(th) - 1):
            arc_segments.append([(xs[j], ys[j]), (xs[j + 1], ys[j + 1])])
            arc_colors.append(i + j / (len(th) - 1))

    line = LineCollection(
        arc_segments,
        cmap=CMAP_NAME,
        norm=Normalize(positions.min(), positions.max()),
        linewidth=2.2,
        alpha=0.9,
        zorder=3,
    )
    line.set_array(np.array(arc_colors))
    ax.add_collection(line)

    sc = ax.scatter(
        dial_x,
        dial_y,
        c=positions,
        cmap=CMAP_NAME,
        s=11,
        alpha=0.92,
        edgecolors="none",
        zorder=4,
    )

    stride_positions = major_positions
    ax.scatter(
        dial_x[stride_positions],
        dial_y[stride_positions],
        s=38,
        facecolors="white",
        edgecolors=COLOR_STRIDE,
        linewidths=1.0,
        zorder=6,
    )

    # Curved direction arrows between stride anchors (arching from the outer part).
    if len(stride_positions) > 1:
        for i in range(len(stride_positions) - 1):
            start = stride_positions[i]
            end = stride_positions[i + 1]
            arrow = FancyArrowPatch(
                posA=(dial_x[start], dial_y[start]),
                posB=(dial_x[end], dial_y[end]),
                connectionstyle="arc3,rad=-0.35",
                arrowstyle="->",
                mutation_scale=9,
                color=COLOR_STRIDE,
                lw=0.9,
                alpha=0.7,
                zorder=7,
            )
            ax.add_patch(arrow)

    # Position labels along the arc at label_stride positions (decoupled from
    # arrow stride so finer labels are shown even with coarse arrows).
    label_positions = np.arange(0, n_positions, label_stride)
    if label_positions[-1] != n_positions - 1:
        label_positions = np.append(label_positions, n_positions - 1)
    for p in label_positions:
        if p == 0 or p == n_positions - 1 or p in (32, 48):
            continue
        label_r = 1.12
        lx, ly = label_r * dial_x[p], label_r * dial_y[p]
        ax.text(
            lx,
            ly,
            str(p),
            ha="center",
            va="center",
            fontsize=7,
            color="#334155",
            weight="bold",
        )

    # Gauge needle with sharp arrowhead, shown at position 48.
    needle_pos = min(48, n_positions - 1)
    hand = FancyArrowPatch(
        posA=(0.0, 0.0),
        posB=(dial_x[needle_pos], dial_y[needle_pos]),
        arrowstyle="-|>",
        mutation_scale=14,
        color="#EF4444",
        lw=1.6,
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(hand)
    # -- Gauge aesthetics: center pivot with metallic look --
    ax.scatter(
        [0.0], [0.0], s=120, c="#94A3B8", edgecolors="#475569", linewidths=1.5, zorder=8
    )
    ax.scatter([0.0], [0.0], s=30, c="#334155", zorder=9)

    # Mechanism formula placed horizontally in the empty gauge interior,
    # with a thin leader line connecting it to the needle.
    formula_text = (
        r"$\mathbf{o}^{(2)}_i \;\approx\; "
        r"\alpha^{(2)}_{i,\mathrm{BOS}}\,\mathbf{d}_{\mathrm{BOS}}"
        r"\;+\;(1 - \alpha^{(2)}_{i,\mathrm{BOS}})\,"
        r"\mathbf{d}_{\mathrm{non\text{-}BOS}}$"
    )
    # Needle midpoint (for leader line origin).
    needle_mid_x = 0.5 * dial_x[needle_pos]
    needle_mid_y = 0.5 * dial_y[needle_pos]
    # Formula position: lower-right interior of gauge (empty space).
    formula_x = 0.30
    formula_y = 0.32
    # Leader line from needle midpoint to formula.
    ax.annotate(
        "",
        xy=(needle_mid_x, needle_mid_y),
        xytext=(formula_x, formula_y + 0.04),
        arrowprops=dict(
            arrowstyle="-",
            color="#94A3B8",
            lw=0.7,
            connectionstyle="arc3,rad=0.15",
        ),
        zorder=2,
    )
    ax.text(
        formula_x,
        formula_y,
        formula_text,
        ha="center",
        va="top",
        fontsize=7.0,
        fontfamily="DejaVu Serif",
        color="#0F172A",
        clip_on=False,
        zorder=10,
    )

    # BOS / non-BOS labels at the arc endpoints, offset below the arc.
    ax.text(
        -1.0,
        -0.12,
        "BOS",
        ha="center",
        va="top",
        fontsize=9.5,
        weight="bold",
        fontfamily="DejaVu Serif",
        color="#0F172A",
        bbox=dict(facecolor=COLOR_FACE, alpha=0.85, edgecolor="none", pad=1.5),
    )
    ax.text(
        1.0,
        -0.12,
        "non-BOS",
        ha="center",
        va="top",
        fontsize=9.5,
        weight="bold",
        fontfamily="DejaVu Serif",
        color="#0F172A",
        bbox=dict(facecolor=COLOR_FACE, alpha=0.85, edgecolor="none", pad=1.5),
    )

    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.20, 1.26)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.02, shrink=0.72)
    cbar.set_label("Position", fontsize=8.5, fontweight="bold")
    cbar.set_ticks(
        [0, n_positions // 4, n_positions // 2, (3 * n_positions) // 4, n_positions - 1]
    )
    cbar.ax.tick_params(labelsize=7)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.88, top=0.96, bottom=0.06)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        save_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08
    )
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/r0_directional_rotation"
    )
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_sequences", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--label_stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = ROOT_DIR / args.checkpoint
    data_dir = ROOT_DIR / args.data_dir
    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, device)
    data = load_data(data_dir)

    tokens, start_indices = sample_sequences(
        data,
        seq_len=args.seq_len,
        n_sequences=args.n_sequences,
        device=device,
        seed=args.seed,
        start_idx=args.start_idx,
    )
    rotation = compute_rotation(model, tokens)

    save_path = save_dir / "r0_directional_rotation_example.pdf"
    plot_rotation(rotation, args.stride, save_path, label_stride=args.label_stride)

    summary = {
        "checkpoint": str(args.checkpoint),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "seed": args.seed,
        "n_sequences": args.n_sequences,
        "start_indices": start_indices,
        **rotation,
    }

    def to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, dict):
            return {k: to_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [to_serializable(v) for v in value]
        return value

    with open(save_dir / "rotation_example_summary.json", "w") as f:
        json.dump(to_serializable(summary), f, indent=2)

    paper_path = paper_dir / save_path.name
    if save_path.exists():
        paper_path.write_bytes(save_path.read_bytes())
    save_png_path = save_path.with_suffix(".png")
    paper_png_path = paper_dir / save_png_path.name
    if save_png_path.exists():
        paper_png_path.write_bytes(save_png_path.read_bytes())

    print("Saved:")
    print(f"  {save_path}")
    print(f"  {save_png_path}")
    print(f"  {paper_path}")
    print(f"  {paper_png_path}")
    print("Summary:")
    print(f"  n_sequences: {args.n_sequences}")
    print(f"  start_idx[0]: {start_indices[0]}")
    print(f"  cos(bos, others): {rotation['cos_bos_others']:.3f}")


if __name__ == "__main__":
    main()
