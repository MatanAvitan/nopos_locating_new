"""Compute step-wise metrics for Attn2-1H and Full-12H evidence."""

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: Path) -> TwoLayerMechanismModel:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TwoLayerMechanismConfig(**filtered)

    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def load_data(data_dir: Path) -> np.memmap:
    return np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, batch_size: int, block_size: int) -> torch.Tensor:
    ix = torch.randint(len(data) - (block_size - 1), (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(
                np.concatenate(
                    [[BOS_TOKEN_ID], data[i : i + block_size - 1].astype(np.int64)]
                )
            )
            for i in ix
        ]
    )
    return x.to(DEVICE)


def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds, targets = preds.flatten().float(), targets.flatten().float()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.flatten().float(), y.flatten().float()
    return ((x - x.mean()) @ (y - y.mean()) / (x.std() * y.std() * len(x))).item()


def compute_block1_norms(model: TwoLayerMechanismModel, tokens: torch.Tensor) -> dict:
    with torch.no_grad():
        model(tokens, capture_taps=True)
        taps = model.get_all_taps()
        attn_out = taps["block1_attn"]

    norms = attn_out.norm(dim=-1)
    pos0_norm = norms[:, 0].mean().item()
    others_norm = norms[:, 1:].mean().item()

    return {
        "pos0_norm": pos0_norm,
        "others_norm": others_norm,
        "ratio": pos0_norm / others_norm,
    }


def compute_attention_bos(
    model: TwoLayerMechanismModel, tokens: torch.Tensor, pos: int
) -> dict:
    with torch.no_grad():
        model(tokens, capture_taps=True)
        attn1, attn2 = model.get_attention_weights()

    attn2_mean = attn2.mean(dim=(0, 1))
    bos_weight = attn2_mean[pos, 0].item()
    uniform = 1.0 / (pos + 1)

    return {
        "bos_attention": bos_weight,
        "uniform": uniform,
        "overweight": bos_weight / uniform,
    }


def compute_geometric_clock(
    model: TwoLayerMechanismModel, tokens: torch.Tensor
) -> dict:
    D = model.config.n_embd
    T = model.config.block_size

    with torch.no_grad():
        model(tokens, capture_taps=True)
        taps = model.get_all_taps()

        W_V = model.block2.attn.c_attn.weight[2 * D :, :].detach()
        W_O = model.block2.attn.c_proj.weight.detach()
        b_O = model.block2.attn.c_proj.bias.detach()
        w_head = model.pos_head.weight.detach().squeeze()

        ln2_1 = taps["block2_ln1"]
        attn_out = taps["block2_attn"]

        v2 = ln2_1 @ W_V.T
        Wo_v = v2 @ W_O.T

        pos0_Wo_v_mean = Wo_v[:, 0, :].mean(dim=0)
        others_Wo_v_mean = Wo_v[:, 1:, :].mean(dim=(0, 1))

        pos0_dir = F.normalize(pos0_Wo_v_mean.unsqueeze(0), dim=-1).squeeze()
        others_dir = F.normalize(others_Wo_v_mean.unsqueeze(0), dim=-1).squeeze()

        attn_out_no_bias = attn_out - b_O
        proj_pos0 = attn_out_no_bias @ pos0_dir
        proj_others = attn_out_no_bias @ others_dir

        positions = (
            torch.arange(T, device=DEVICE)
            .float()
            .unsqueeze(0)
            .expand(tokens.size(0), -1)
        )

        cos_pos0_others = F.cosine_similarity(
            pos0_dir.unsqueeze(0), others_dir.unsqueeze(0)
        ).item()
        corr_pos0 = corrcoef(proj_pos0, positions)
        corr_others = corrcoef(proj_others, positions)
        cos_w_pos0 = F.cosine_similarity(
            w_head.unsqueeze(0), pos0_dir.unsqueeze(0)
        ).item()
        cos_w_others = F.cosine_similarity(
            w_head.unsqueeze(0), others_dir.unsqueeze(0)
        ).item()

    return {
        "pos0_norm": pos0_Wo_v_mean.norm().item(),
        "others_norm": others_Wo_v_mean.norm().item(),
        "cos_pos0_others": cos_pos0_others,
        "corr_pos0": corr_pos0,
        "corr_others": corr_others,
        "cos_w_pos0": cos_w_pos0,
        "cos_w_others": cos_w_others,
    }


def compute_r2(
    model: TwoLayerMechanismModel, data: np.ndarray, n_batches: int, batch_size: int
) -> float:
    preds_all = []
    targets_all = []

    for _ in range(n_batches):
        tokens = get_batch(data, batch_size, model.config.block_size)
        targets = (
            torch.arange(model.config.block_size, device=DEVICE)
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        with torch.no_grad():
            output, _ = model(tokens, capture_taps=False)
            preds = output.squeeze(-1)
        preds_all.append(preds)
        targets_all.append(targets)

    return r2_score(torch.cat(preds_all, dim=0), torch.cat(targets_all, dim=0))


def plot_attention_maps(attn1, attn2, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    def strip_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    im1 = axes[0].imshow(attn1[0].cpu().numpy(), cmap="cividis", aspect="auto")
    axes[0].set_title("Block 1 Attention")
    strip_axes(axes[0])
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(attn2[0].cpu().numpy(), cmap="cividis", aspect="auto")
    axes[1].set_title("Block 2 Attention")
    strip_axes(axes[1])
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_attention_map(
    model: TwoLayerMechanismModel, data: np.ndarray, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = get_batch(data, batch_size, model.config.block_size)
    with torch.no_grad():
        model(tokens, capture_taps=True)
        attn1, attn2 = model.get_attention_weights()
    return attn1.mean(dim=0), attn2.mean(dim=0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn2_ckpt",
        type=str,
        default=(
            "nanoGPT/out-2layer-mechanism-r2-1head-attnonly-fullblock-40k/"
            "R2/o4w7v8dv/best_ckpt.pt"
        ),
    )
    parser.add_argument(
        "--full12h_ckpt",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/empirical_evidence_metrics"
    )
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--r2_batches", type=int, default=20)
    parser.add_argument("--r2_batch_size", type=int, default=32)
    parser.add_argument("--bos_pos", type=int, default=50)
    args = parser.parse_args()

    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(ROOT_DIR / args.data_dir)

    attn2_model = load_model(ROOT_DIR / args.attn2_ckpt)
    full_model = load_model(ROOT_DIR / args.full12h_ckpt)

    tokens = get_batch(data, args.batch_size, attn2_model.config.block_size)

    results = {
        "attn2": {
            "block1_norms": compute_block1_norms(attn2_model, tokens),
            "attention_bos": compute_attention_bos(attn2_model, tokens, args.bos_pos),
            "geometric": compute_geometric_clock(attn2_model, tokens),
            "r2": compute_r2(attn2_model, data, args.r2_batches, args.r2_batch_size),
        },
        "full12h": {
            "block1_norms": compute_block1_norms(full_model, tokens),
            "attention_bos": compute_attention_bos(full_model, tokens, args.bos_pos),
            "geometric": compute_geometric_clock(full_model, tokens),
            "r2": compute_r2(full_model, data, args.r2_batches, args.r2_batch_size),
        },
        "n_sequences": args.batch_size,
        "r2_sequences": args.r2_batches * args.r2_batch_size,
        "bos_pos": args.bos_pos,
    }

    attn1_full, attn2_full = generate_attention_map(full_model, data, args.batch_size)
    attention_map_path = save_dir / "attention_maps_full_12h.pdf"
    plot_attention_maps(attn1_full, attn2_full, attention_map_path)
    (paper_dir / attention_map_path.name).write_bytes(attention_map_path.read_bytes())

    with open(save_dir / "empirical_evidence_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved metrics to:", save_dir / "empirical_evidence_metrics.json")
    print("Saved Full-12H attention maps to:", attention_map_path)


if __name__ == "__main__":
    main()
