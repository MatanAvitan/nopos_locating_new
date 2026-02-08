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


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho that is robust to degenerate inputs."""
    rho = spearmanr(x, y).correlation
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


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
    d_model = model.config.n_embd
    proj_bos_batches = []
    proj_others_batches = []

    for _ in range(n_batches):
        tokens = sample_tokens(data, seq_len, batch_size, device)
        with torch.no_grad():
            model(tokens, capture_taps=True)
            taps = model.get_all_taps()

        W_V = model.block2.attn.c_attn.weight[2 * d_model :, :].detach()
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
        proj_bos = (attn_out_no_bias @ bos_dir).detach().cpu().numpy()  # [B, T]
        proj_others = (attn_out_no_bias @ others_dir).detach().cpu().numpy()  # [B, T]

        proj_bos_batches.append(proj_bos)
        proj_others_batches.append(proj_others)

    proj_bos_seq = np.concatenate(proj_bos_batches, axis=0)
    proj_others_seq = np.concatenate(proj_others_batches, axis=0)

    proj_bos = proj_bos_seq.mean(axis=0)
    proj_others = proj_others_seq.mean(axis=0)
    positions = np.arange(seq_len)

    corr_bos = float(np.corrcoef(proj_bos, positions)[0, 1])
    corr_non_bos = float(np.corrcoef(proj_others, positions)[0, 1])
    spearman_bos = _safe_spearman(proj_bos, positions)
    spearman_non_bos = _safe_spearman(proj_others, positions)

    spearman_stats = {
        "bos": summarize_projection_spearman(proj_bos_seq),
        "non_bos": summarize_projection_spearman(proj_others_seq),
    }

    return {
        "proj_bos": proj_bos,
        "proj_non_bos": proj_others,
        "corr_bos": corr_bos,
        "corr_non_bos": corr_non_bos,
        "spearman_bos": spearman_bos,
        "spearman_non_bos": spearman_non_bos,
        "spearman_stats": spearman_stats,
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
    parser.add_argument("--n_batches_128", type=int, default=10)
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

    model = load_model(checkpoint_path, device)
    data = load_data(data_dir)

    print("Computing projections at L=128...")
    results_128 = compute_projection(
        model, data, 128, args.batch_size_128, args.n_batches_128, device
    )

    save_path = save_dir / f"attention_output_projection_{args.output_tag}.pdf"
    plot_projection_single(results_128, 128, save_path)

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
            to_serializable({"L128": results_128}),
            f,
            indent=2,
        )

    with open(save_dir / f"projection_results_{args.output_tag}.json", "w") as f:
        json.dump(
            to_serializable({"L128": results_128}),
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
    stats_bos = results_128["spearman_stats"]["bos"]
    stats_non = results_128["spearman_stats"]["non_bos"]
    print(
        f"  L=128 mean-curve Spearman: BOS={results_128['spearman_bos']:.3f}, "
        f"non-BOS={results_128['spearman_non_bos']:.3f}"
    )
    print(
        f"  L=128 sample-level median Spearman: BOS={stats_bos['sample_sequence_median_rho']:.3f}, "
        f"non-BOS={stats_non['sample_sequence_median_rho']:.3f}"
    )


if __name__ == "__main__":
    main()
