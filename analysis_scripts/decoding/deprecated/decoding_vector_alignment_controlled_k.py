"""Decoding Vector Alignment Analysis with Controlled K Experiment.

This script measures how well the MLP weights align with the theoretical decoding vector
when using controlled sequences with exactly K unique tokens.

For each K:
1. Generate multiple sequences with format [t1, t2, ..., t_{k-1}, t0, t0, ...]
   where we have K unique tokens (including base token t0)
2. For each sequence, compute decoding vector using ONLY the K unique embeddings
3. Measure alignment (cosine similarity) with MLP input/output weights
4. Aggregate results across sequences and training steps

Usage:
    python analysis_scripts/decoding_vector_alignment_controlled_k.py \
        --checkpoint_dir nanoGPT/out-posreg-6layer-until-mlp \
        --k_values "2,5,10,20,50,100" \
        --n_sequences_per_k 100 \
        --wandb
"""

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
    """Configuration for controlled-K decoding-vector alignment analysis."""

    checkpoint_dir: Path
    steps: list[int]
    k_values: list[int]
    n_sequences_per_k: int
    layer_idx: int
    device: str
    output_dir: Path
    top_k: int
    wandb_project: str | None
    wandb_run_name: str | None
    context_length: int
    vocab_size: int
    seed: int


def parse_steps(steps_str: str) -> list[int]:
    """Parse comma-separated checkpoint steps."""
    return [int(val.strip()) for val in steps_str.split(",") if val.strip()]


def parse_k_values(k_str: str) -> list[int]:
    """Parse comma-separated K values."""
    return [int(val.strip()) for val in k_str.split(",") if val.strip()]


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


def generate_controlled_k_sequences(
    k: int,
    n_sequences: int,
    context_length: int,
    vocab_size: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Generate sequences with exactly K unique tokens.

    Format: [t1, t2, ..., t_{k-1}, t0, t0, ...] where:
    - First k-1 positions have unique prefix tokens
    - Position k onwards repeat the base token t0
    - Total K unique tokens = k-1 prefix + 1 base

    Returns:
        tokens: [n_sequences, context_length] token tensor
        unique_token_lists: list of [K unique token IDs] for each sequence
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokens = torch.zeros((n_sequences, context_length), dtype=torch.long, device=device)
    unique_token_lists = []

    for seq_idx in range(n_sequences):
        # Sample K unique tokens from vocab
        unique_tokens = torch.randperm(vocab_size, device=device)[:k].tolist()
        unique_token_lists.append(unique_tokens)

        # Base token is the last unique token
        base_token = unique_tokens[-1]

        # Prefix tokens (first k-1 positions)
        prefix_tokens = unique_tokens[:-1]

        # Build sequence: [prefix..., base, base, ...]
        for pos in range(context_length):
            if pos < len(prefix_tokens):
                tokens[seq_idx, pos] = prefix_tokens[pos]
            else:
                tokens[seq_idx, pos] = base_token

    return tokens, unique_token_lists


def compute_decoding_vector_for_sequence(
    model: GPTPositionClassifier,
    unique_tokens: list[int],
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Compute decoding vector using only the K unique embeddings in the sequence.

    Args:
        model: The position classifier model
        unique_tokens: List of K unique token IDs in this sequence
        layer_idx: Which transformer block to use

    Returns:
        Dictionary with 'w_v' and 'w_v_wo' decoding vectors
    """
    block = model.transformer.h[layer_idx]

    # Get embeddings for ONLY the unique tokens in this sequence
    embeddings = model.transformer.wte.weight  # [vocab_size, d_model]
    unique_embeddings = embeddings[unique_tokens, :]  # [K, d_model]

    # Apply LN1
    with torch.no_grad():
        ln_embeddings = block.ln_1(unique_embeddings)  # [K, d_model]

    # Sum over the K unique embeddings
    summed = ln_embeddings.sum(dim=0)  # [d_model]

    # Extract W_V and W_O
    n_embd = model.config.n_embd
    c_attn_weight = block.attn.c_attn.weight  # [3*d_model, d_model]
    w_v = c_attn_weight[2 * n_embd :, :]  # Value weights
    w_o = block.attn.c_proj.weight  # Output projection

    # Decoding vectors
    decoding_v = summed @ w_v.T  # [d_model]
    decoding_v_o = decoding_v @ w_o.T  # After output projection

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
        "max_abs": float(np.max(np.abs(values))),
    }


def analyze_k_checkpoint(
    model: GPTPositionClassifier,
    k: int,
    unique_token_lists: list[list[int]],
    layer_idx: int,
    top_k: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute decoding-vector alignment for all sequences with K unique tokens.

    For each sequence:
    1. Compute decoding vector from its K unique embeddings
    2. Measure alignment with MLP weights
    3. Aggregate statistics across all sequences
    """
    block = model.transformer.h[layer_idx]
    c_fc = block.mlp.c_fc.weight  # [4*d_model, d_model]
    c_proj = block.mlp.c_proj.weight  # [d_model, 4*d_model]

    # Collect results for each sequence
    all_fc_sims_v = []
    all_fc_sims_vo = []
    all_proj_sims_v = []
    all_proj_sims_vo = []

    for unique_tokens in unique_token_lists:
        # Compute decoding vector for this sequence
        decoding_vectors = compute_decoding_vector_for_sequence(
            model, unique_tokens, layer_idx
        )

        # Alignment with MLP weights
        fc_sims_v = cosine_to_rows(decoding_vectors["w_v"], c_fc)
        fc_sims_vo = cosine_to_rows(decoding_vectors["w_v_wo"], c_fc)
        proj_sims_v = cosine_to_cols(decoding_vectors["w_v"], c_proj)
        proj_sims_vo = cosine_to_cols(decoding_vectors["w_v_wo"], c_proj)

        all_fc_sims_v.append(fc_sims_v.detach().cpu().numpy())
        all_fc_sims_vo.append(fc_sims_vo.detach().cpu().numpy())
        all_proj_sims_v.append(proj_sims_v.detach().cpu().numpy())
        all_proj_sims_vo.append(proj_sims_vo.detach().cpu().numpy())

    # Convert to arrays and compute statistics
    # Each is [n_sequences, n_neurons]
    all_fc_sims_v = np.stack(all_fc_sims_v)
    all_fc_sims_vo = np.stack(all_fc_sims_vo)
    all_proj_sims_v = np.stack(all_proj_sims_v)
    all_proj_sims_vo = np.stack(all_proj_sims_vo)

    results: dict[str, dict[str, dict[str, Any]]] = {}

    # W_V variant
    results["w_v"] = {
        "c_fc_rows": {
            "summary": summarize(torch.tensor(all_fc_sims_v.mean(axis=0)), top_k),
            "per_sequence_max_abs": [
                float(np.abs(sims).max()) for sims in all_fc_sims_v
            ],
            "per_sequence_mean_abs": [
                float(np.abs(sims).mean()) for sims in all_fc_sims_v
            ],
        },
        "c_proj_cols": {
            "summary": summarize(torch.tensor(all_proj_sims_v.mean(axis=0)), top_k),
            "per_sequence_max_abs": [
                float(np.abs(sims).max()) for sims in all_proj_sims_v
            ],
            "per_sequence_mean_abs": [
                float(np.abs(sims).mean()) for sims in all_proj_sims_v
            ],
        },
    }

    # W_V W_O variant
    results["w_v_wo"] = {
        "c_fc_rows": {
            "summary": summarize(torch.tensor(all_fc_sims_vo.mean(axis=0)), top_k),
            "per_sequence_max_abs": [
                float(np.abs(sims).max()) for sims in all_fc_sims_vo
            ],
            "per_sequence_mean_abs": [
                float(np.abs(sims).mean()) for sims in all_fc_sims_vo
            ],
        },
        "c_proj_cols": {
            "summary": summarize(torch.tensor(all_proj_sims_vo.mean(axis=0)), top_k),
            "per_sequence_max_abs": [
                float(np.abs(sims).max()) for sims in all_proj_sims_vo
            ],
            "per_sequence_mean_abs": [
                float(np.abs(sims).mean()) for sims in all_proj_sims_vo
            ],
        },
    }

    return results


def plot_metric_over_steps_and_k(
    steps: list[int],
    k_values: list[int],
    data: dict[int, dict[int, float]],  # data[k][step] = value
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot a metric as heatmap or line plot over K and steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in k_values:
        values = [data[k].get(step, np.nan) for step in steps]
        ax.plot(steps, values, marker="o", linewidth=2, label=f"K={k}")

    ax.set_title(title)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--steps", type=str, default="500,1000,2000,5000,10000,15000,20000"
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="2,5,10,20,50,100",
        help="Comma-separated K values (number of unique tokens)",
    )
    parser.add_argument(
        "--n_sequences_per_k",
        type=int,
        default=100,
        help="Number of sequences to generate for each K",
    )
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/decoding_vector_alignment_controlled_k",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb_project", type=str, default="nope-position-regression-metrics"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="decoding-vector-controlled-k"
    )
    args = parser.parse_args()

    cfg = AlignmentConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        steps=parse_steps(args.steps),
        k_values=parse_k_values(args.k_values),
        n_sequences_per_k=args.n_sequences_per_k,
        layer_idx=args.layer_idx,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=args.device,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        seed=args.seed,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.wandb_run_name if args.wandb else None,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Controlled-K Decoding Vector Alignment Analysis")
    print(f"{'=' * 70}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"K values: {cfg.k_values}")
    print(f"Sequences per K: {cfg.n_sequences_per_k}")
    print(f"Steps: {cfg.steps}")
    print(f"Layer: {cfg.layer_idx}")
    print(f"Context length: {cfg.context_length}")
    print(f"Output: {cfg.output_dir}")
    print(f"{'=' * 70}\n")

    results: dict[str, Any] = {
        "config": {
            "checkpoint_dir": str(cfg.checkpoint_dir),
            "steps": cfg.steps,
            "k_values": cfg.k_values,
            "n_sequences_per_k": cfg.n_sequences_per_k,
            "layer_idx": cfg.layer_idx,
            "context_length": cfg.context_length,
            "top_k": cfg.top_k,
            "seed": cfg.seed,
        },
        "results_by_k_and_step": {},
    }

    wandb_run = None
    if cfg.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                "k_values": cfg.k_values,
                "n_sequences_per_k": cfg.n_sequences_per_k,
                "steps": cfg.steps,
                "layer_idx": cfg.layer_idx,
            },
        )

    # Data for plotting: metric[k][step] = value
    max_abs_wv_fc: dict[int, dict[int, float]] = {k: {} for k in cfg.k_values}
    max_abs_wvwo_fc: dict[int, dict[int, float]] = {k: {} for k in cfg.k_values}
    top_k_mean_wv_fc: dict[int, dict[int, float]] = {k: {} for k in cfg.k_values}
    top_k_mean_wvwo_fc: dict[int, dict[int, float]] = {k: {} for k in cfg.k_values}

    for step in cfg.steps:
        ckpt_path = cfg.checkpoint_dir / f"ckpt_{step:05d}.pt"
        if not ckpt_path.exists():
            print(f"⚠ Checkpoint {ckpt_path} not found, skipping")
            continue

        print(f"\n{'─' * 70}")
        print(f"Processing step {step}")
        print(f"{'─' * 70}")

        model, _ = load_checkpoint(ckpt_path, cfg.device)

        for k in cfg.k_values:
            print(
                f"  K={k:3d}: Generating {cfg.n_sequences_per_k} sequences... ",
                end="",
                flush=True,
            )

            # Generate controlled sequences with K unique tokens
            tokens, unique_token_lists = generate_controlled_k_sequences(
                k=k,
                n_sequences=cfg.n_sequences_per_k,
                context_length=cfg.context_length,
                vocab_size=cfg.vocab_size,
                seed=cfg.seed + step + k,  # Different seed for each K and step
                device=cfg.device,
            )

            print(f"Analyzing alignment... ", end="", flush=True)

            # Analyze alignment for this K
            k_results = analyze_k_checkpoint(
                model, k, unique_token_lists, cfg.layer_idx, cfg.top_k
            )

            # Store results
            if k not in results["results_by_k_and_step"]:
                results["results_by_k_and_step"][k] = {}
            results["results_by_k_and_step"][k][step] = k_results

            # Extract metrics for plotting
            max_abs_wv_fc[k][step] = k_results["w_v"]["c_fc_rows"]["summary"]["max_abs"]
            max_abs_wvwo_fc[k][step] = k_results["w_v_wo"]["c_fc_rows"]["summary"][
                "max_abs"
            ]
            top_k_mean_wv_fc[k][step] = k_results["w_v"]["c_fc_rows"]["summary"][
                "top_k_mean_abs"
            ]
            top_k_mean_wvwo_fc[k][step] = k_results["w_v_wo"]["c_fc_rows"]["summary"][
                "top_k_mean_abs"
            ]

            print(f"✓ (max |cos| W_V: {max_abs_wv_fc[k][step]:.4f})")

            # Log to WandB
            if wandb_run is not None:
                wandb.log(
                    {
                        "eval/ckpt_step": step,
                        "eval/k": k,
                        f"k{k}/w_v_c_fc_max_abs": max_abs_wv_fc[k][step],
                        f"k{k}/w_v_wo_c_fc_max_abs": max_abs_wvwo_fc[k][step],
                        f"k{k}/w_v_c_fc_top_k_mean_abs": top_k_mean_wv_fc[k][step],
                        f"k{k}/w_v_wo_c_fc_top_k_mean_abs": top_k_mean_wvwo_fc[k][step],
                    }
                )

        del model
        if "cuda" in cfg.device:
            torch.cuda.empty_cache()

    # Save results
    results_path = cfg.output_dir / "controlled_k_alignment_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Generate plots
    print(f"\nGenerating plots...")

    plot_metric_over_steps_and_k(
        cfg.steps,
        cfg.k_values,
        max_abs_wv_fc,
        "Max |cos| W_V · sum(E_k) vs MLP c_fc rows",
        "Max |cos|",
        cfg.output_dir / "max_abs_wv_fc_by_k",
    )

    plot_metric_over_steps_and_k(
        cfg.steps,
        cfg.k_values,
        max_abs_wvwo_fc,
        "Max |cos| W_O W_V · sum(E_k) vs MLP c_fc rows",
        "Max |cos|",
        cfg.output_dir / "max_abs_wvwo_fc_by_k",
    )

    plot_metric_over_steps_and_k(
        cfg.steps,
        cfg.k_values,
        top_k_mean_wv_fc,
        "Top-k mean |cos| W_V · sum(E_k) vs MLP c_fc rows",
        "Top-k mean |cos|",
        cfg.output_dir / "top_k_mean_wv_fc_by_k",
    )

    plot_metric_over_steps_and_k(
        cfg.steps,
        cfg.k_values,
        top_k_mean_wvwo_fc,
        "Top-k mean |cos| W_O W_V · sum(E_k) vs MLP c_fc rows",
        "Top-k mean |cos|",
        cfg.output_dir / "top_k_mean_wvwo_fc_by_k",
    )

    print(f"✓ Plots saved to {cfg.output_dir}")

    if wandb_run is not None:
        # Log plots to WandB
        for name in [
            "max_abs_wv_fc_by_k",
            "max_abs_wvwo_fc_by_k",
            "top_k_mean_wv_fc_by_k",
            "top_k_mean_wvwo_fc_by_k",
        ]:
            wandb.log(
                {f"plots/{name}": wandb.Image(str(cfg.output_dir / f"{name}.png"))}
            )
        wandb.finish()
        print(f"✓ Logged to WandB")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
