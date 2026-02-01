"""Dual-metric extrapolation plot: head vs linear probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ICML style settings (match paper Figure 6)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


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


def get_batch_with_bos(data: np.ndarray, batch_size: int, length: int) -> torch.Tensor:
    max_start = len(data) - (length - 1)
    ix = torch.randint(max_start, (batch_size,))
    tokens = torch.stack(
        [
            torch.from_numpy(
                np.concatenate(
                    [[BOS_TOKEN_ID], data[i : i + length - 1].astype(np.int64)]
                )
            )
            for i in ix
        ]
    )
    return tokens.to(DEVICE)


def evaluate_head_r2(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    context_lengths: list[int],
    n_batches: int,
    batch_size: int,
) -> dict:
    results = {}
    for length in context_lengths:
        preds_all = []
        targets_all = []

        for _ in range(n_batches):
            tokens = get_batch_with_bos(data, batch_size, length)
            targets = (
                torch.arange(length, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            with torch.no_grad():
                output, _ = model(tokens, capture_taps=False)
                preds = output.squeeze(-1)
            preds_all.append(preds)
            targets_all.append(targets)

        preds_all = torch.cat(preds_all, dim=0).flatten().cpu().numpy()
        targets_all = torch.cat(targets_all, dim=0).flatten().cpu().numpy()
        r = np.corrcoef(targets_all, preds_all)[0, 1]
        results[str(length)] = {"r2": float(r * r), "n": n_batches * batch_size}
    return results


def evaluate_head_r2_scheduled(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    context_lengths: list[int],
    batch_schedule: dict[int, int],
    target_n: int,
) -> dict:
    results = {}
    for length in context_lengths:
        batch_size = batch_schedule[length]
        n_batches = int(np.ceil(target_n / batch_size))
        preds_all = []
        targets_all = []

        for _ in range(n_batches):
            tokens = get_batch_with_bos(data, batch_size, length)
            targets = (
                torch.arange(length, device=DEVICE)
                .float()
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            with torch.no_grad():
                output, _ = model(tokens, capture_taps=False)
                preds = output.squeeze(-1)
            preds_all.append(preds)
            targets_all.append(targets)

        preds_all = torch.cat(preds_all, dim=0).flatten().cpu().numpy()
        targets_all = torch.cat(targets_all, dim=0).flatten().cpu().numpy()
        r = np.corrcoef(targets_all, preds_all)[0, 1]
        results[str(length)] = {"r2": float(r * r), "n": n_batches * batch_size}
    return results


def evaluate_linear_probe_r2(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    context_lengths: list[int],
    target_ns: dict[int, int],
    batch_schedule: dict[int, int],
) -> dict:
    results = {}

    for length in context_lengths:
        batch_size = batch_schedule[length]
        target_n = target_ns[length]
        n_batches = int(np.ceil(target_n / batch_size))
        post_attn_acts = []
        positions_list = []

        with torch.no_grad():
            for _ in range(n_batches):
                tokens = get_batch_with_bos(data, batch_size, length)
                batch_size_effective = tokens.shape[0]
                _ = model(tokens, capture_taps=True)
                post_attn_acts.append(model.block2.last_post_attn.cpu())
                positions = (
                    torch.arange(length).unsqueeze(0).expand(batch_size_effective, -1)
                )
                positions_list.append(positions)

        post_attn_acts = torch.cat(post_attn_acts, dim=0).numpy()
        positions = torch.cat(positions_list, dim=0).numpy()

        positions_flat = positions.reshape(-1)
        acts_flat = post_attn_acts.reshape(-1, post_attn_acts.shape[-1])

        X_train, X_test, y_train, y_test = train_test_split(
            acts_flat, positions_flat, test_size=0.2, random_state=42
        )
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results[str(length)] = {"r2": float(r2), "n": int(n_batches * batch_size)}

    return results


def evaluate_head_finetune_r2(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    context_lengths: list[int],
    target_ns: dict[int, int],
    batch_schedule: dict[int, int],
    lr: float,
    epochs: int,
    weight_decay: float,
) -> dict:
    results = {}
    initial_state = {
        k: v.detach().clone() for k, v in model.pos_head.state_dict().items()
    }

    for length in context_lengths:
        model.pos_head.load_state_dict(initial_state)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.pos_head.parameters():
            param.requires_grad = True

        model.train()
        optimizer = torch.optim.AdamW(
            model.pos_head.parameters(), lr=lr, weight_decay=weight_decay
        )

        batch_size = batch_schedule[length]
        n_batches = int(np.ceil(target_ns[length] / batch_size))

        for _ in range(epochs):
            for _ in range(n_batches):
                tokens = get_batch_with_bos(data, batch_size, length)
                targets = (
                    torch.arange(length, device=DEVICE)
                    .float()
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                _, loss = model(tokens, targets, capture_taps=False)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for _ in range(n_batches):
                tokens = get_batch_with_bos(data, batch_size, length)
                targets = (
                    torch.arange(length, device=DEVICE)
                    .float()
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                output, _ = model(tokens, capture_taps=False)
                preds = output.squeeze(-1)
                preds_all.append(preds)
                targets_all.append(targets)

        preds_all = torch.cat(preds_all, dim=0).flatten().cpu().numpy()
        targets_all = torch.cat(targets_all, dim=0).flatten().cpu().numpy()
        r = np.corrcoef(targets_all, preds_all)[0, 1]
        results[str(length)] = {"r2": float(r * r), "n": int(n_batches * batch_size)}

    return results


def plot_dual_metric(
    head_results: dict,
    probe_results: dict,
    context_lengths: list[int],
    save_path: Path,
    panel_b_title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))

    lengths = np.array(context_lengths)
    labels = {
        "attn2": "ATTN2-1H",
        "full12h": "FULL-12H",
    }
    colors = {
        "attn2": "#0072B2",
        "full12h": "#D55E00",
    }
    markers = {
        "attn2": "o",
        "full12h": "s",
    }

    ax = axes[0]
    for key in ["attn2", "full12h"]:
        vals = np.array([head_results[key][str(L)]["r2"] for L in context_lengths])
        ax.plot(
            lengths,
            vals,
            color=colors[key],
            marker=markers[key],
            label=labels[key],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Context length")
    ax.set_ylabel("R$^2$")
    ax.set_title("(a) Head-based decoding")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    for key in ["attn2", "full12h"]:
        vals = np.array([probe_results[key][str(L)]["r2"] for L in context_lengths])
        ax.plot(
            lengths,
            vals,
            color=colors[key],
            marker=markers[key],
            label=labels[key],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Context length")
    ax.set_ylabel("R$^2$")
    ax.set_title(panel_b_title)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=False)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
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
        "--save_dir", type=str, default="results/extrapolation_long_context"
    )
    parser.add_argument(
        "--paper_dir", type=str, default="overleaf/nopos_icml_2026/plots"
    )
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--full_target_n", type=int, default=64)
    parser.add_argument(
        "--context_lengths",
        type=str,
        default="128,256,512,1024,2048,4096",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="post_attn",
        choices=["post_attn", "head_finetune"],
    )
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--head_weight_decay", type=float, default=0.0)
    parser.add_argument("--output_tag", type=str, default="")
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(ROOT_DIR / args.data_dir)
    attn2_model = load_model(ROOT_DIR / args.attn2_ckpt)
    full_model = load_model(ROOT_DIR / args.full12h_ckpt)

    head_results = {
        "attn2": evaluate_head_r2(
            attn2_model, data, context_lengths, args.n_batches, args.batch_size
        ),
        "full12h": evaluate_head_r2_scheduled(
            full_model,
            data,
            context_lengths,
            batch_schedule={
                128: 32,
                256: 16,
                512: 8,
                1024: 4,
                2048: 2,
                4096: 1,
            },
            target_n=args.full_target_n,
        ),
    }

    probe_target_ns = {128: 256, 256: 256, 512: 128, 1024: 64, 2048: 32, 4096: 16}
    probe_batch_schedule = {128: 32, 256: 16, 512: 8, 1024: 4, 2048: 2, 4096: 1}

    if args.probe_type == "post_attn":
        probe_results = {
            "attn2": evaluate_linear_probe_r2(
                attn2_model,
                data,
                context_lengths,
                probe_target_ns,
                probe_batch_schedule,
            ),
            "full12h": evaluate_linear_probe_r2(
                full_model, data, context_lengths, probe_target_ns, probe_batch_schedule
            ),
        }
        panel_b_title = "(b) Linear probe (post-attn)"
    else:
        probe_results = {
            "attn2": evaluate_head_finetune_r2(
                attn2_model,
                data,
                context_lengths,
                probe_target_ns,
                probe_batch_schedule,
                lr=args.head_lr,
                epochs=args.head_epochs,
                weight_decay=args.head_weight_decay,
            ),
            "full12h": evaluate_head_finetune_r2(
                full_model,
                data,
                context_lengths,
                probe_target_ns,
                probe_batch_schedule,
                lr=args.head_lr,
                epochs=args.head_epochs,
                weight_decay=args.head_weight_decay,
            ),
        }
        panel_b_title = "(b) Head fine-tune (frozen backbone)"

    output_tag = f"_{args.output_tag}" if args.output_tag else ""
    results_path = save_dir / f"extrapolation_dual_metric{output_tag}_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "head": head_results,
                "probe": probe_results,
                "head_attn2_n": args.n_batches * args.batch_size,
                "head_full_target_n": args.full_target_n,
                "probe_target_ns": probe_target_ns,
                "probe_type": args.probe_type,
                "head_lr": args.head_lr if args.probe_type == "head_finetune" else None,
                "head_epochs": args.head_epochs
                if args.probe_type == "head_finetune"
                else None,
            },
            f,
            indent=2,
        )

    save_path = save_dir / f"extrapolation_dual_metric{output_tag}.pdf"
    plot_dual_metric(
        head_results, probe_results, context_lengths, save_path, panel_b_title
    )

    paper_path = paper_dir / save_path.name
    if save_path.exists():
        paper_path.write_bytes(save_path.read_bytes())

    print("Saved:", save_path)
    print("Copied to:", paper_path)
    print("Results:", results_path)


if __name__ == "__main__":
    main()
