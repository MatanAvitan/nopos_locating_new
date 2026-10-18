"""Grid of directional rotation dials for R0-12head sequences."""

from __future__ import annotations

import json
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
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.RandomState(seed)
    max_start = len(data) - (seq_len - 1)
    if max_start <= 0:
        raise ValueError(f"Data too short for seq_len={seq_len}")

    indices = rng.randint(0, max_start, size=n_sequences).tolist()
    sequences = []
    for idx in indices:
        seq = np.concatenate(
            [[BOS_TOKEN_ID], data[idx : idx + seq_len - 1].astype(np.int64)]
        )
        sequences.append(seq)
    return np.stack(sequences, axis=0), indices


def compute_dials(model: TwoLayerMechanismModel, tokens: torch.Tensor) -> dict:
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
    attn_out_no_bias = attn_out - b_O

    dial_x_all = []
    dial_y_all = []
    dial_theta_all = []
    cos_bos_others_all = []

    for b in range(tokens.size(0)):
        bos_mean = Wo_v[b, 0, :]
        others_mean = Wo_v[b, 1:, :].mean(dim=0)
        bos_dir = F.normalize(bos_mean, dim=0)
        others_dir = F.normalize(others_mean, dim=0)

        attn_unit = F.normalize(attn_out_no_bias[b], dim=-1)
        cos_bos = (attn_unit @ bos_dir).cpu().numpy()
        cos_bos_others = float(
            F.cosine_similarity(bos_dir.unsqueeze(0), others_dir.unsqueeze(0)).item()
        )
        ref_angle = float(np.degrees(np.arccos(np.clip(cos_bos_others, -1.0, 1.0))))
        theta_bos = np.degrees(np.arccos(np.clip(cos_bos, -1.0, 1.0)))
        dial_angle = theta_bos / ref_angle * 180.0
        dial_angle = np.clip(dial_angle, 0.0, 180.0)

        dial_theta = np.radians(dial_angle)
        dial_x = -np.cos(dial_theta)
        dial_y = np.sin(dial_theta)

        dial_x_all.append(dial_x)
        dial_y_all.append(dial_y)
        dial_theta_all.append(dial_theta)
        cos_bos_others_all.append(cos_bos_others)

    return {
        "dial_x": dial_x_all,
        "dial_y": dial_y_all,
        "dial_theta": dial_theta_all,
        "cos_bos_others": cos_bos_others_all,
    }


def plot_grid(
    dials: dict,
    n_rows: int,
    n_cols: int,
    stride: int,
    save_path: Path,
    label_stride: int = 16,
    needle_positions: list[int] | None = None,
) -> None:
    positions = np.arange(len(dials["dial_x"][0]))
    stride_positions = np.arange(0, len(positions), stride)
    n_panels = n_rows * n_cols
    if needle_positions is None:
        needle_positions = [len(positions) - 1] * n_panels
    if len(needle_positions) != n_panels:
        raise ValueError(
            f"needle_positions length ({len(needle_positions)}) must equal n_rows*n_cols ({n_panels})"
        )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.8, 13.5))
    fig.patch.set_facecolor("white")
    axes = axes.reshape(n_rows, n_cols)

    theta_top = np.linspace(0, np.pi, 240)
    arc_x = -np.cos(theta_top)
    arc_y = np.sin(theta_top)

    for idx in range(n_panels):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.set_facecolor(COLOR_FACE)
        dial_x = dials["dial_x"][idx]
        dial_y = dials["dial_y"][idx]
        dial_th = dials["dial_theta"][idx]
        n_pos = len(positions)
        needle_pos = int(np.clip(needle_positions[idx], 0, n_pos - 1))

        # Half-gauge frame with gauge aesthetics.
        # Shadow arc behind main track.
        ax.plot(
            arc_x,
            arc_y,
            color="#9CADC4",
            linewidth=3.5,
            alpha=0.3,
            zorder=0,
            solid_capstyle="round",
        )
        # Main track arc.
        ax.plot(arc_x, arc_y, color=COLOR_TRACK, linewidth=1.5, alpha=0.95, zorder=1)
        # Thin inner arc for depth.
        inner_r = 0.88
        ax.plot(
            inner_r * arc_x,
            inner_r * arc_y,
            color=COLOR_TRACK,
            linewidth=0.4,
            alpha=0.4,
            zorder=1,
        )

        # Tick marks at selected positions.
        tick_lbl_pos = [p for p in (16, 64, 80, 96, 112) if 0 <= p < n_pos]
        tick_inner = 0.93
        tick_outer = 1.00
        for p in tick_lbl_pos:
            x_u = dial_x[p]
            y_u = dial_y[p]
            ax.plot(
                [tick_inner * x_u, tick_outer * x_u],
                [tick_inner * y_u, tick_outer * y_u],
                color="#64748B",
                linewidth=0.7,
                alpha=0.6,
                zorder=2,
            )

        # Gradient trajectory interpolated along the arc.
        n_interp = 6
        arc_segments = []
        arc_colors = []
        for i in range(len(dial_th) - 1):
            th = np.linspace(dial_th[i], dial_th[i + 1], n_interp + 1)
            xs = -np.cos(th)
            ys = np.sin(th)
            for j in range(len(th) - 1):
                arc_segments.append([(xs[j], ys[j]), (xs[j + 1], ys[j + 1])])
                arc_colors.append(i + j / (len(th) - 1))

        line = LineCollection(
            arc_segments,
            cmap=CMAP_NAME,
            norm=Normalize(positions.min(), positions.max()),
            linewidth=1.25,
            alpha=0.9,
            zorder=3,
        )
        line.set_array(np.array(arc_colors))
        ax.add_collection(line)

        ax.scatter(
            dial_x,
            dial_y,
            c=positions,
            cmap=CMAP_NAME,
            s=7,
            alpha=0.9,
            edgecolors="none",
            zorder=4,
        )

        ax.scatter(
            dial_x[stride_positions],
            dial_y[stride_positions],
            s=16,
            facecolors="white",
            edgecolors=COLOR_STRIDE,
            linewidths=0.6,
            zorder=6,
        )

        if len(stride_positions) > 1:
            for i in range(len(stride_positions) - 1):
                start = stride_positions[i]
                end = stride_positions[i + 1]
                arrow = FancyArrowPatch(
                    posA=(dial_x[start], dial_y[start]),
                    posB=(dial_x[end], dial_y[end]),
                    connectionstyle="arc3,rad=-0.35",
                    arrowstyle="->",
                    mutation_scale=7,
                    color=COLOR_STRIDE,
                    lw=0.6,
                    alpha=0.6,
                    zorder=7,
                )
                ax.add_patch(arrow)

        # Position labels along the arc at label_stride (decoupled from arrow stride).
        label_positions = np.arange(0, n_pos, label_stride)
        if label_positions[-1] != n_pos - 1:
            label_positions = np.append(label_positions, n_pos - 1)
        for p in label_positions:
            if p == 0 or p == n_pos - 1 or p in (32, 48):
                continue
            label_r = 1.14
            lx, ly = label_r * dial_x[p], label_r * dial_y[p]
            ax.text(
                lx,
                ly,
                str(p),
                ha="center",
                va="center",
                fontsize=4,
                color="#334155",
                weight="bold",
            )

        # Gauge needle with sharp arrowhead at a sampled position.
        hand = FancyArrowPatch(
            posA=(0.0, 0.0),
            posB=(dial_x[needle_pos], dial_y[needle_pos]),
            arrowstyle="-|>",
            mutation_scale=10,
            color="#EF4444",
            lw=1.0,
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(hand)

        ax.scatter([dial_x[0]], [dial_y[0]], s=16, c="#22C55E", zorder=7)
        ax.scatter(
            [dial_x[needle_pos]], [dial_y[needle_pos]], s=16, c="#EF4444", zorder=7
        )
        # Center pivot with metallic look.
        ax.scatter(
            [0.0],
            [0.0],
            s=50,
            c="#94A3B8",
            edgecolors="#475569",
            linewidths=0.8,
            zorder=8,
        )
        ax.scatter([0.0], [0.0], s=12, c="#334155", zorder=9)

        # BOS / non-BOS at arc endpoints, below the arc.
        ax.text(
            -1.0,
            -0.10,
            "BOS",
            ha="center",
            va="top",
            fontsize=5,
            weight="bold",
            fontfamily="DejaVu Serif",
            color="#0F172A",
            bbox=dict(facecolor=COLOR_FACE, alpha=0.85, edgecolor="none", pad=0.8),
        )
        ax.text(
            1.0,
            -0.10,
            "non-BOS",
            ha="center",
            va="top",
            fontsize=5,
            weight="bold",
            fontfamily="DejaVu Serif",
            color="#0F172A",
            bbox=dict(facecolor=COLOR_FACE, alpha=0.85, edgecolor="none", pad=0.8),
        )

        ax.text(
            0.03,
            0.95,
            f"Seq {idx + 1}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#334155",
            weight="bold",
        )

        ax.set_xlim(-1.20, 1.20)
        ax.set_ylim(-0.18, 1.20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=CMAP_NAME, norm=Normalize(0, len(positions) - 1))
    plt.subplots_adjust(wspace=0.05, hspace=0.12, right=0.89, top=0.98)
    cax = fig.add_axes((0.905, 0.21, 0.011, 0.58))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Position", fontsize=8, fontweight="bold")
    n_pos = len(positions)
    cbar.set_ticks([0, n_pos // 4, n_pos // 2, (3 * n_pos) // 4, n_pos - 1])
    cbar.ax.tick_params(labelsize=7)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
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
    parser.add_argument("--n_rows", type=int, default=6)
    parser.add_argument("--n_cols", type=int, default=2)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--label_stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    n_sequences = args.n_rows * args.n_cols

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

    sequences, start_indices = sample_sequences(
        data,
        seq_len=args.seq_len,
        n_sequences=n_sequences,
        seed=args.seed,
    )
    tokens = torch.from_numpy(sequences).to(device)

    dials = compute_dials(model, tokens)

    n_positions = sequences.shape[1]
    needle_rng = np.random.RandomState(args.seed + 11)
    needle_positions = needle_rng.randint(1, n_positions, size=n_sequences).tolist()

    save_path = save_dir / "r0_directional_rotation_grid.pdf"
    plot_grid(
        dials,
        args.n_rows,
        args.n_cols,
        args.stride,
        save_path,
        label_stride=args.label_stride,
        needle_positions=needle_positions,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "seed": args.seed,
        "n_rows": args.n_rows,
        "n_cols": args.n_cols,
        "start_indices": start_indices,
        "needle_positions": needle_positions,
        "cos_bos_others": dials["cos_bos_others"],
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

    with open(save_dir / "r0_directional_rotation_grid_summary.json", "w") as f:
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


if __name__ == "__main__":
    main()
