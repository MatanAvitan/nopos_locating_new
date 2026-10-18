"""Analyze decoding vector alignment with MLP weights across checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_position_classifier import GPTPositionClassifier, GPTPositionClassifierConfig


@dataclass
class AlignmentConfig:
    """Configuration for decoding-vector alignment analysis."""

    checkpoint_dir: Path
    steps: list[int]
    layer_idx: int
    device: str
    output_dir: Path
    top_k: int
    wandb_project: str | None
    wandb_run_name: str | None


def parse_steps(steps_str: str) -> list[int]:
    """Parse comma-separated checkpoint steps."""
    return [int(val.strip()) for val in steps_str.split(",") if val.strip()]


def load_checkpoint(
    ckpt_path: Path, device: str
) -> tuple[GPTPositionClassifier, dict[str, Any]]:
    """Load a position-regression checkpoint into GPTPositionClassifier."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    config = GPTPositionClassifierConfig(
        n_layer=model_args.get("n_layer", 6),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 128),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        bias=model_args.get("bias", False),
        use_positional_embedding=model_args.get("use_positional_embedding", False),
        norm_type=model_args.get("norm_type", "layernorm"),
        use_regression=model_args.get("use_regression", True),
        compute_lm_loss=model_args.get("compute_lm_loss", False),
        use_ln2=model_args.get("use_ln2", True),
        mlp_expansion_ratio=model_args.get("mlp_expansion_ratio", 4),
    )

    model = GPTPositionClassifier(config)
    state_dict = checkpoint["model"]
    unwrapped = {}
    for key, value in state_dict.items():
        clean_key = key[10:] if key.startswith("_orig_mod.") else key
        unwrapped[clean_key] = value
    model.load_state_dict(unwrapped, strict=False)
    model.to(device)
    model.eval()

    return model, checkpoint


def compute_decoding_vectors(
    model: GPTPositionClassifier, layer_idx: int
) -> dict[str, torch.Tensor]:
    """Compute decoding vectors from W_V and W_O (block 0)."""
    block = model.transformer.h[layer_idx]
    embeddings = model.transformer.wte.weight
    ln_embeddings = block.ln_1(embeddings)
    summed = ln_embeddings.sum(dim=0)

    n_embd = model.config.n_embd
    c_attn_weight = block.attn.c_attn.weight
    w_v = c_attn_weight[2 * n_embd :, :]
    w_o = block.attn.c_proj.weight

    decoding_v = summed @ w_v.T
    decoding_v_o = decoding_v @ w_o.T

    return {
        "w_v": decoding_v,
        "w_v_wo": decoding_v_o,
    }


def cosine_to_rows(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between vector and each row in matrix."""
    vec_norm = torch.norm(vector) + 1e-8
    row_norms = torch.norm(matrix, dim=1) + 1e-8
    sims = (matrix @ vector) / (row_norms * vec_norm)
    return sims


def cosine_to_cols(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between vector and each column in matrix."""
    return cosine_to_rows(vector, matrix.T)


def summarize(similarities: torch.Tensor, top_k: int) -> dict[str, float]:
    """Summarize cosine similarities with descriptive stats."""
    values = similarities.detach().cpu().numpy()
    top_k = min(top_k, values.shape[0])
    top_vals = np.sort(np.abs(values))[-top_k:]
    return {
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "top_k_mean_abs": float(np.mean(top_vals)),
    }


def analyze_checkpoint(
    model: GPTPositionClassifier, layer_idx: int, top_k: int
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute decoding-vector alignment with MLP weights."""
    block = model.transformer.h[layer_idx]
    decoding_vectors = compute_decoding_vectors(model, layer_idx)

    c_fc = block.mlp.c_fc.weight
    c_proj = block.mlp.c_proj.weight

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for variant_name, vector in decoding_vectors.items():
        variant_results: dict[str, dict[str, Any]] = {}

        fc_row_sims = cosine_to_rows(vector, c_fc)
        proj_col_sims = cosine_to_cols(vector, c_proj)

        variant_results["c_fc_rows"] = {
            "cosine": fc_row_sims.detach().cpu().numpy().tolist(),
            "summary": summarize(fc_row_sims, top_k),
        }
        variant_results["c_proj_cols"] = {
            "cosine": proj_col_sims.detach().cpu().numpy().tolist(),
            "summary": summarize(proj_col_sims, top_k),
        }

        results[variant_name] = variant_results

    return results


def plot_metric_over_steps(
    steps: list[int],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot a scalar metric over checkpoints."""
    plt.figure(figsize=(7, 4))
    for label, values in series.items():
        plt.plot(steps, values, marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("Checkpoint step")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--steps", type=str, default="500,1000,2000,5000,10000,15000,20000"
    )
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir", type=str, default="results/decoding_vector_alignment"
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb_project", type=str, default="nope-position-regression-metrics"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="decoding-vector-mlp-alignment"
    )
    args = parser.parse_args()

    cfg = AlignmentConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        steps=parse_steps(args.steps),
        layer_idx=args.layer_idx,
        device=args.device,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.wandb_run_name if args.wandb else None,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "config": {
            "checkpoint_dir": str(cfg.checkpoint_dir),
            "steps": cfg.steps,
            "layer_idx": cfg.layer_idx,
            "top_k": cfg.top_k,
        },
        "results_by_step": {},
    }

    wandb_run = None
    if cfg.wandb_project:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    summary_series: dict[str, list[float]] = {
        "w_v/c_fc_rows_max_abs": [],
        "w_v/c_proj_cols_max_abs": [],
        "w_v_wo/c_fc_rows_max_abs": [],
        "w_v_wo/c_proj_cols_max_abs": [],
    }

    for step in cfg.steps:
        ckpt_path = cfg.checkpoint_dir / f"ckpt_{step:05d}.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint {ckpt_path} not found, skipping")
            continue

        model, _ = load_checkpoint(ckpt_path, cfg.device)
        step_results = analyze_checkpoint(model, cfg.layer_idx, cfg.top_k)
        results["results_by_step"][step] = step_results

        for variant, entries in step_results.items():
            fc_summary = entries["c_fc_rows"]["summary"]
            proj_summary = entries["c_proj_cols"]["summary"]
            if variant == "w_v":
                summary_series["w_v/c_fc_rows_max_abs"].append(
                    fc_summary["top_k_mean_abs"]
                )
                summary_series["w_v/c_proj_cols_max_abs"].append(
                    proj_summary["top_k_mean_abs"]
                )
            else:
                summary_series["w_v_wo/c_fc_rows_max_abs"].append(
                    fc_summary["top_k_mean_abs"]
                )
                summary_series["w_v_wo/c_proj_cols_max_abs"].append(
                    proj_summary["top_k_mean_abs"]
                )

        if wandb_run is not None:
            import wandb

            log_payload = {"eval/ckpt_step": step}
            for variant, entries in step_results.items():
                for matrix_name, data in entries.items():
                    summary = data["summary"]
                    prefix = f"{variant}/{matrix_name}"
                    log_payload[f"{prefix}/max"] = summary["max"]
                    log_payload[f"{prefix}/mean_abs"] = summary["mean_abs"]
                    log_payload[f"{prefix}/top_k_mean_abs"] = summary["top_k_mean_abs"]
            wandb.log(log_payload)

        del model
        if "cuda" in cfg.device:
            torch.cuda.empty_cache()

    results_path = cfg.output_dir / "decoding_vector_alignment.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    plot_metric_over_steps(
        cfg.steps,
        {
            "W_v x sum(E) vs c_fc rows": summary_series["w_v/c_fc_rows_max_abs"],
            "W_v x sum(E) vs c_proj cols": summary_series["w_v/c_proj_cols_max_abs"],
        },
        "Decoding Vector Alignment (top-k mean |cos|)",
        "Top-k mean |cos|",
        cfg.output_dir / "decoding_vector_alignment_wv",
    )

    plot_metric_over_steps(
        cfg.steps,
        {
            "W_O W_v x sum(E) vs c_fc rows": summary_series["w_v_wo/c_fc_rows_max_abs"],
            "W_O W_v x sum(E) vs c_proj cols": summary_series[
                "w_v_wo/c_proj_cols_max_abs"
            ],
        },
        "Decoding Vector Alignment with W_O (top-k mean |cos|)",
        "Top-k mean |cos|",
        cfg.output_dir / "decoding_vector_alignment_wv_wo",
    )

    if wandb_run is not None:
        import wandb

        wandb.log(
            {
                "plots/wv_alignment": wandb.Image(
                    str(cfg.output_dir / "decoding_vector_alignment_wv.png")
                )
            }
        )
        wandb.log(
            {
                "plots/wv_wo_alignment": wandb.Image(
                    str(cfg.output_dir / "decoding_vector_alignment_wv_wo.png")
                )
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
