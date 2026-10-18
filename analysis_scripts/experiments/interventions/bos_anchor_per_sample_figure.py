"""Per-sample version of the Step-1 projection figure (Figure 3).

Instead of plotting mean ± std, this script plots individual sequences as
separate thin lines so we can verify that the monotonic trend holds
sample-by-sample and is not an artifact of averaging.

Output: bos_anchor_per_sample.pdf / .png  (saved to same results dir and
        copied to the paper plots dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel

BOS_TOKEN_ID = 50256


# ---------------------------------------------------------------------------
# Helpers copied/adapted from bos_anchor_decodability_after_ln.py
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: str) -> TwoLayerMechanismModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TwoLayerMechanismConfig(**filtered_config)
    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_data(data_dir: Path) -> np.memmap:
    return np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")


def get_batch_with_bos(
    data: np.memmap, batch_size: int, block_size: int, device: str
) -> torch.Tensor:
    ix = torch.randint(len(data) - (block_size - 1), (batch_size,))
    sequences = []
    for i in ix:
        tail = data[i : i + block_size - 1].astype(np.int64)
        seq = np.concatenate([[BOS_TOKEN_ID], tail])
        sequences.append(torch.from_numpy(seq))
    return torch.stack(sequences).to(device)


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def extract_step1_activations(
    model: TwoLayerMechanismModel, tokens: torch.Tensor
) -> torch.Tensor:
    """Return hbar_i^(1) = block1_ln2  [B, T, D]."""
    with torch.no_grad():
        model(tokens, capture_taps=True)
        taps = model.get_all_taps()
    return taps["block1_ln2"]


def estimate_stage1_directions(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    n_batches: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Estimate d_BOS and d_nonBOS from training data."""
    d_model = model.config.n_embd
    block_size = model.config.block_size

    bos_sum = np.zeros(d_model, dtype=np.float64)
    nonbos_sum = np.zeros(d_model, dtype=np.float64)
    bos_count = 0
    nonbos_count = 0

    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, block_size, device)
        x = extract_step1_activations(model, tokens).detach().cpu().numpy()
        bos_sum += x[:, 0, :].sum(axis=0)
        nonbos_sum += x[:, 1:, :].sum(axis=(0, 1))
        bos_count += x.shape[0]
        nonbos_count += x.shape[0] * (x.shape[1] - 1)

    return {
        "bos": normalize(bos_sum / max(1, bos_count)),
        "nonbos": normalize(nonbos_sum / max(1, nonbos_count)),
    }


def collect_per_sample_projections(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    directions: dict[str, np.ndarray],
    n_sequences: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Return per-sequence projections of shape [n_sequences, block_size]."""
    block_size = model.config.block_size
    direction_matrix = np.stack(
        [directions[k] for k in directions], axis=1
    )  # [D, K]
    direction_tensor = torch.from_numpy(direction_matrix).to(
        device=device, dtype=torch.float32
    )

    # Collect all sequences in one batch (or several if memory is tight)
    batch_size = min(n_sequences, 64)
    collected: dict[str, list] = {k: [] for k in directions}
    gathered = 0

    while gathered < n_sequences:
        this_batch = min(batch_size, n_sequences - gathered)
        tokens = get_batch_with_bos(data, this_batch, block_size, device)
        x = extract_step1_activations(model, tokens)          # [B, T, D]
        projs = torch.matmul(x, direction_tensor)             # [B, T, K]
        projs_np = projs.detach().cpu().numpy()               # [B, T, K]
        for idx, k in enumerate(directions):
            collected[k].append(projs_np[:, :, idx])         # [B, T]
        gathered += this_batch

    return {k: np.concatenate(collected[k], axis=0) for k in directions}
    # shape: [n_sequences, block_size]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_sample(
    per_sample: dict[str, dict[str, np.ndarray]],
    model_labels: dict[str, str],
    save_path: Path,
    n_show: int = 50,
    pos_limit: int | None = None,
) -> None:
    """4-panel figure: rows = BOS/nonBOS, columns = model.

    Each panel shows n_show individual sequence curves (thin, semi-transparent)
    plus the mean across all sequences (bold).
    """
    model_keys = list(per_sample.keys())
    direction_keys = ["bos", "nonbos"]
    direction_labels = {
        "bos": r"$\langle \bar{h}_i^{(1)},\; d_{\mathrm{BOS}}\rangle$",
        "nonbos": r"$\langle \bar{h}_i^{(1)},\; d_{\mathrm{nonBOS}}\rangle$",
    }
    panel_titles = {
        "bos": "BOS projection after LN2",
        "nonbos": "nonBOS projection after LN2",
    }
    colors = {model_keys[0]: "#0072B2", model_keys[1]: "#D55E00"}

    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 5.5), sharex=True
    )

    for row, dir_key in enumerate(direction_keys):
        for col, model_key in enumerate(model_keys):
            ax = axes[row, col]
            data = per_sample[model_key][dir_key]  # [N_seq, T]
            color = colors[model_key]

            T = data.shape[1]
            positions = np.arange(T) if pos_limit is None else np.arange(min(T, pos_limit))
            plot_data = data[:, : len(positions)]

            # Sample n_show individual curves without replacement
            rng = np.random.default_rng(0)
            idx = rng.choice(plot_data.shape[0], size=min(n_show, plot_data.shape[0]), replace=False)
            sample_curves = plot_data[idx]  # [n_show, T]

            for s in range(sample_curves.shape[0]):
                ax.plot(
                    positions,
                    sample_curves[s],
                    linewidth=0.45,
                    alpha=0.18,
                    color=color,
                )

            mean_curve = plot_data.mean(axis=0)
            ax.plot(
                positions,
                mean_curve,
                linewidth=2.0,
                color=color,
                label="mean",
                zorder=3,
            )

            ax.set_ylabel(direction_labels[dir_key], fontsize=8)
            ax.set_xlabel("Position $i$", fontsize=8)
            if row == 0:
                ax.set_title(
                    f"{model_labels[model_key]}\n{panel_titles[dir_key]}",
                    fontsize=8,
                )
            else:
                ax.set_title(panel_titles[dir_key], fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path} and {save_path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-sample Step-1 projections to verify monotonicity."
    )
    parser.add_argument(
        "--attn2_ckpt",
        type=str,
        default="nanoGPT/out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt",
    )
    parser.add_argument(
        "--full12h_ckpt",
        type=str,
        default="nanoGPT/out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/bos_anchor_decodability_after_ln",
    )
    parser.add_argument(
        "--paper_dir",
        type=str,
        default="overleaf/nopos_icml_2026/plots",
    )
    parser.add_argument("--n_sequences", type=int, default=200,
                        help="Total sequences to collect projections for")
    parser.add_argument("--n_show", type=int, default=50,
                        help="Number of individual sample curves to draw")
    parser.add_argument("--train_batches", type=int, default=20,
                        help="Batches for estimating d_BOS / d_nonBOS directions")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pos_limit", type=int, default=None,
                        help="Limit x-axis to first N positions (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(ROOT_DIR / args.data_dir)

    model_configs = {
        "attn2_1h": (ROOT_DIR / args.attn2_ckpt, "ATTN2-1H"),
        "full_12h": (ROOT_DIR / args.full12h_ckpt, "FULL-12H"),
    }

    per_sample_all: dict[str, dict[str, np.ndarray]] = {}
    model_labels: dict[str, str] = {}

    for model_key, (ckpt_path, label) in model_configs.items():
        print(f"\nLoading {label} from {ckpt_path}...")
        model = load_model(ckpt_path, device)
        model_labels[model_key] = label

        print(f"  Estimating directions ({args.train_batches} batches × {args.batch_size})...")
        directions = estimate_stage1_directions(
            model=model,
            data=data,
            n_batches=args.train_batches,
            batch_size=args.batch_size,
            device=device,
        )

        print(f"  Collecting {args.n_sequences} per-sample projections...")
        per_sample = collect_per_sample_projections(
            model=model,
            data=data,
            directions=directions,
            n_sequences=args.n_sequences,
            device=device,
        )
        per_sample_all[model_key] = per_sample

    save_path = save_dir / "bos_anchor_per_sample.pdf"
    print(f"\nPlotting (showing {args.n_show} sample curves per panel)...")
    plot_per_sample(
        per_sample=per_sample_all,
        model_labels=model_labels,
        save_path=save_path,
        n_show=args.n_show,
        pos_limit=args.pos_limit,
    )

    import shutil
    for suffix in [".pdf", ".png"]:
        src = save_path.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, paper_dir / src.name)
            print(f"Copied to {paper_dir / src.name}")


if __name__ == "__main__":
    main()
