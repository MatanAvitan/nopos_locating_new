"""Generate R0 attention output projections at L=128 and L=4096."""

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


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
COLOR_BOS = "#009E73"
COLOR_OTHERS = "#D55E00"


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
    D = model.config.n_embd
    proj_bos_all = []
    proj_others_all = []

    for _ in range(n_batches):
        tokens = sample_tokens(data, seq_len, batch_size, device)
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
        proj_bos = (attn_out_no_bias @ bos_dir).mean(dim=0)
        proj_others = (attn_out_no_bias @ others_dir).mean(dim=0)

        proj_bos_all.append(proj_bos.cpu())
        proj_others_all.append(proj_others.cpu())

    proj_bos = torch.stack(proj_bos_all).mean(dim=0).numpy()
    proj_others = torch.stack(proj_others_all).mean(dim=0).numpy()
    positions = np.arange(seq_len)

    corr_bos = float(np.corrcoef(proj_bos, positions)[0, 1])
    corr_others = float(np.corrcoef(proj_others, positions)[0, 1])

    return {
        "proj_bos": proj_bos,
        "proj_others": proj_others,
        "corr_bos": corr_bos,
        "corr_others": corr_others,
    }


def plot_projections(results_128: dict, results_4096: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))

    for ax, seq_len, results in [
        (axes[0], 128, results_128),
        (axes[1], 4096, results_4096),
    ]:
        positions = np.arange(seq_len)
        ax.plot(
            positions,
            results["proj_bos"],
            color=COLOR_BOS,
            linewidth=1.2,
            label=f"BOS dir (r={results['corr_bos']:.2f})",
        )
        ax.plot(
            positions,
            results["proj_others"],
            color=COLOR_OTHERS,
            linewidth=1.2,
            label=f"Others dir (r={results['corr_others']:.2f})",
        )
        ax.set_title(f"L={seq_len}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Projection")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="best")

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
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/r0_attention_output_projection"
    )
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--batch_size_128", type=int, default=32)
    parser.add_argument("--batch_size_4096", type=int, default=1)
    parser.add_argument("--n_batches_128", type=int, default=10)
    parser.add_argument("--n_batches_4096", type=int, default=2)
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

    print("Computing projections at L=128...")
    results_128 = compute_projection(
        model, data, 128, args.batch_size_128, args.n_batches_128, device
    )

    print("Computing projections at L=4096...")
    results_4096 = compute_projection(
        model, data, 4096, args.batch_size_4096, args.n_batches_4096, device
    )

    save_path = save_dir / "r0_attention_output_projection_128_4096.pdf"
    plot_projections(results_128, results_4096, save_path)

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

    with open(save_dir / "projection_results.json", "w") as f:
        json.dump(
            to_serializable({"L128": results_128, "L4096": results_4096}),
            f,
            indent=2,
        )

    paper_path = paper_dir / save_path.name
    if save_path.exists():
        paper_path.write_bytes(save_path.read_bytes())

    print("Saved:")
    print(f"  {save_path}")
    print(f"  {paper_path}")
    print("Summary:")
    print(
        f"  L=128: corr(BOS)={results_128['corr_bos']:.3f}, "
        f"corr(others)={results_128['corr_others']:.3f}"
    )
    print(
        f"  L=4096: corr(BOS)={results_4096['corr_bos']:.3f}, "
        f"corr(others)={results_4096['corr_others']:.3f}"
    )


if __name__ == "__main__":
    main()
