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

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel

BOS_TOKEN_ID = 50256
COLOR_PATH = "#0072B2"
COLOR_STRIDE = "#000000"


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
        cos_bos_others_all.append(cos_bos_others)

    return {
        "dial_x": dial_x_all,
        "dial_y": dial_y_all,
        "cos_bos_others": cos_bos_others_all,
    }


def plot_grid(
    dials: dict, n_rows: int, n_cols: int, stride: int, save_path: Path
) -> None:
    positions = np.arange(len(dials["dial_x"][0]))
    stride_positions = np.arange(0, len(positions), stride)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 13.0))
    axes = axes.reshape(n_rows, n_cols)

    theta = np.linspace(0, np.pi, 200)
    arc_x = -np.cos(theta)
    arc_y = np.sin(theta)

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        dial_x = dials["dial_x"][idx]
        dial_y = dials["dial_y"][idx]

        ax.plot(arc_x, arc_y, color="#888888", linewidth=0.6, alpha=0.6)
        ax.plot(dial_x, dial_y, color=COLOR_PATH, alpha=0.2, linewidth=0.8)
        ax.scatter(
            dial_x,
            dial_y,
            c=positions,
            cmap="viridis",
            s=6,
            alpha=0.75,
        )

        ax.scatter(
            dial_x[stride_positions],
            dial_y[stride_positions],
            s=18,
            facecolors="white",
            edgecolors=COLOR_STRIDE,
            linewidths=0.6,
            zorder=3,
        )

        if len(stride_positions) > 1:
            for i in range(len(stride_positions) - 1):
                start = stride_positions[i]
                end = stride_positions[i + 1]
                ax.annotate(
                    "",
                    xy=(dial_x[end], dial_y[end]),
                    xytext=(dial_x[start], dial_y[start]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=COLOR_STRIDE,
                        lw=0.6,
                        alpha=0.4,
                    ),
                )

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)

    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(0, len(positions) - 1)
    )
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Position")

    fig.text(0.06, 0.99, "BOS (left)", ha="left", va="top", fontsize=9)
    fig.text(0.94, 0.99, "Others (right)", ha="right", va="top", fontsize=9)

    plt.subplots_adjust(wspace=0.05, hspace=0.08, right=0.92, top=0.97)
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
    parser.add_argument("--stride", type=int, default=16)
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

    save_path = save_dir / "r0_directional_rotation_grid.pdf"
    plot_grid(dials, args.n_rows, args.n_cols, args.stride, save_path)

    summary = {
        "checkpoint": str(args.checkpoint),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "seed": args.seed,
        "n_rows": args.n_rows,
        "n_cols": args.n_cols,
        "start_indices": start_indices,
        "cos_bos_others": dials["cos_bos_others"],
    }

    def to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.float32, np.float64)):
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

    print("Saved:")
    print(f"  {save_path}")
    print(f"  {paper_path}")


if __name__ == "__main__":
    main()
