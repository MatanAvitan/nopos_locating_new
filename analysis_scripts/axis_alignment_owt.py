"""
Axis Alignment Experiment on OpenWebText with Frozen Position Regression Model

This script computes axis alignment (absolute cosine similarity) using:
1. The FROZEN 6-layer position regression checkpoints (out-posreg-6layer-until-mlp/)
2. Real OpenWebText validation data (not synthetic sequences)

Key insight: Since early layers are frozen (embeddings + attention + LN), the post_attn
and post_ln2 activations should be CONSTANT across training epochs. Only the MLP changes.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python axis_alignment_owt.py --n_samples 1000
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json

# Parse args early for GPU setting
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
    help="GPU(s) to use (if not set via CUDA_VISIBLE_DEVICES)",
)
parser.add_argument(
    "--n_samples", type=int, default=500, help="Number of sequences to sample"
)
parser.add_argument(
    "--seq_len", type=int, default=128, help="Sequence length (matches training)"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="nanoGPT/out-posreg-6layer-until-mlp",
    help="Path to checkpoint directory",
)
parser.add_argument(
    "--output_suffix",
    type=str,
    default="",
    help="Suffix to add to output files (e.g., '_1layer')",
)
parser.add_argument(
    "--layers",
    type=str,
    default="post_mlp",
    help="Comma-separated list of layers to analyze",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Log plots and metrics to Weights & Biases",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="nope-position-regression-metrics",
    help="WandB project name",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="WandB run name (defaults to axis-alignment-owt{suffix})",
)
args = parser.parse_args()

# Only override CUDA_VISIBLE_DEVICES if explicitly set via --gpu
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig

# ─── Configuration ───────────────────────────────────────────────────────────

RESULTS_DIR = Path("results/axis_alignment_owt")
PLOTS_DIR = Path("overleaf/nopos_icml_2026/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint directory - set via command line argument
CKPT_DIR = Path(args.checkpoint_dir)

# OpenWebText validation data
VAL_DATA_PATH = Path("nanoGPT/data/openwebtext/val.bin")


@dataclass
class Config:
    """Configuration for axis alignment experiment."""

    n_samples: int = 500
    seq_len: int = 128  # Match training block_size
    batch_size: int = 32
    seed: int = 42

    # Checkpoints to analyze (position regression runs to 20K)
    checkpoint_steps: List[int] = field(
        default_factory=lambda: [500, 1000, 2000, 5000, 10000, 15000, 20000]
    )

    # Layers to analyze
    layers: List[str] = field(default_factory=lambda: ["post_mlp"])

    # K bins for grouping results by number of unique tokens
    k_bins: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 10),
            (10, 25),
            (25, 50),
            (50, 75),
            (75, 100),
            (100, 128),
        ]
    )


def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[GPT, dict]:
    """Load a model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    gptconf = GPTConfig(
        n_layer=model_args.get("n_layer", 6),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 128),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=model_args.get("use_positional_embedding", False),
        norm_type=model_args.get("norm_type", "layernorm"),
        bias=model_args.get("bias", False),
    )

    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwrapped = {
        (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()
    }
    # Filter out pos_head (position regression head) - not part of base GPT
    unwrapped = {k: v for k, v in unwrapped.items() if not k.startswith("pos_head")}
    model.load_state_dict(unwrapped, strict=False)
    model.eval()
    model.to(device)

    return model, {"step": checkpoint.get("iter_num", 0), "config": model_args}


def load_owt_sequences(n_samples: int, seq_len: int, seed: int = 42) -> torch.Tensor:
    """
    Load sequences from OpenWebText validation data.

    Returns:
        tokens: [n_samples, seq_len] token indices
    """
    np.random.seed(seed)

    # Memory-map the validation data
    val_data = np.memmap(str(VAL_DATA_PATH), dtype=np.uint16, mode="r")
    total_tokens = len(val_data)

    print(f"Loaded OpenWebText val.bin: {total_tokens:,} tokens")

    # Sample random starting positions
    max_start = total_tokens - seq_len - 1
    start_indices = np.random.randint(0, max_start, size=n_samples)

    # Extract sequences
    sequences = []
    for start_idx in start_indices:
        seq = val_data[start_idx : start_idx + seq_len].astype(np.int64)
        sequences.append(seq)

    tokens = torch.tensor(np.array(sequences), dtype=torch.long, device="cpu")
    return tokens


def count_unique_tokens_per_position(tokens: torch.Tensor) -> torch.Tensor:
    """
    For each position, count unique tokens seen up to and including that position.

    Args:
        tokens: [n_samples, seq_len]

    Returns:
        unique_counts: [n_samples, seq_len] - K value at each position
    """
    n_samples, seq_len = tokens.shape
    unique_counts = torch.zeros(
        n_samples, seq_len, dtype=torch.long, device=tokens.device
    )

    for i in range(n_samples):
        seen = set()
        for j in range(seq_len):
            seen.add(tokens[i, j].item())
            unique_counts[i, j] = len(seen)

    return unique_counts


def get_activations(
    model: GPT, tokens: torch.Tensor, batch_size: int = 32
) -> Dict[str, torch.Tensor]:
    """
    Extract activations for the FIRST block only.

    Available outputs:
    - post_attn: attention output (before residual)
    - post_ln2: output after LN2 (before MLP)
    - mlp_hidden: hidden MLP state after GELU
    - post_mlp: residual output after MLP
    """
    model.eval()
    n_samples = tokens.shape[0]

    post_attn_list = []
    post_ln2_list = []
    mlp_hidden_list = []
    post_mlp_list = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = tokens[i : i + batch_size].to(device)

            # Token embeddings (no positional embedding in NoPE)
            tok_emb = model.transformer.wte(batch)
            x = model.transformer.drop(tok_emb)

            # Process through FIRST block only (block 0)
            block = model.transformer.h[0]

            # LN1 -> Attention
            ln1_out = block.ln_1(x)
            attn_out = block.attn(ln1_out)
            post_attn_list.append(attn_out.cpu())

            # Residual after attention
            x = x + attn_out

            # LN2
            if hasattr(block, "ln_2") and not getattr(block, "skip_ln2", False):
                ln2_out = block.ln_2(x)
            else:
                ln2_out = x
            post_ln2_list.append(ln2_out.cpu())

            # MLP hidden + output
            mlp_hidden = block.mlp.gelu(block.mlp.c_fc(ln2_out))
            mlp_hidden_list.append(mlp_hidden.cpu())
            mlp_out = block.mlp.c_proj(mlp_hidden)
            mlp_out = block.mlp.dropout(mlp_out)

            x = x + mlp_out
            post_mlp_list.append(x.cpu())

    return {
        "post_attn": torch.cat(post_attn_list, dim=0),
        "post_ln2": torch.cat(post_ln2_list, dim=0),
        "mlp_hidden": torch.cat(mlp_hidden_list, dim=0),
        "post_mlp": torch.cat(post_mlp_list, dim=0),
    }


def compute_projections(
    activations: torch.Tensor,  # [n_samples, seq_len, d_model]
    tokens: torch.Tensor,  # [n_samples, seq_len]
    basis_matrix: torch.Tensor,  # [vocab_size, d_model]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast computation of cosine alignment onto a token embedding basis.

    For each sequence, computes |cos(h, e)| for all unique tokens e in that sequence.

    Returns:
        max_projections: [n_samples, seq_len]
        mean_other_projections: [n_samples, seq_len]
    """
    n_samples, seq_len, d_model = activations.shape

    max_projections = torch.zeros(n_samples, seq_len)
    mean_other_projections = torch.zeros(n_samples, seq_len)

    for i in range(n_samples):
        # Get ALL unique tokens in this sequence
        unique_tokens = tokens[i].unique()
        k = len(unique_tokens)

        # Get embeddings for unique tokens
        basis = basis_matrix[unique_tokens]  # [k, d_model]

        # Compute norms
        basis_norms = torch.norm(basis, dim=1, keepdim=True)  # [k, 1]

        # Get activations for all positions: [seq_len, d_model]
        acts = activations[i]
        acts_norms = torch.norm(acts, dim=1, keepdim=True)  # [seq_len, 1]

        # Compute cosine similarities: (U . E) / (||U|| * ||E||)
        # acts @ basis.T -> [seq_len, k]
        dot_products = torch.matmul(acts, basis.T)
        denom = acts_norms * basis_norms.T + 1e-8
        projections = dot_products / denom
        abs_proj = torch.abs(projections)

        # Max |cos| per position
        max_vals, _ = torch.max(abs_proj, dim=1)  # [seq_len]
        max_projections[i] = max_vals

        # Mean of other projections (sum - max) / (k-1)
        if k > 1:
            sum_proj = abs_proj.sum(dim=1)
            mean_other_projections[i] = (sum_proj - max_vals) / (k - 1)

    return max_projections, mean_other_projections


def analyze_checkpoint(
    model: GPT,
    tokens: torch.Tensor,
    unique_counts: torch.Tensor,
    cfg: Config,
) -> Dict:
    """Analyze axis alignment for a checkpoint."""

    # Get embeddings
    embedding_matrix = model.transformer.wte.weight.data.cpu()

    # Precompute MLP hidden basis if needed
    mlp_hidden_basis = None
    if "mlp_hidden" in cfg.layers:
        block = model.transformer.h[0]
        weight = block.mlp.c_fc.weight.detach().cpu()
        bias = (
            block.mlp.c_fc.bias.detach().cpu()
            if block.mlp.c_fc.bias is not None
            else None
        )
        mlp_hidden_basis = F.linear(embedding_matrix, weight, bias)
        mlp_hidden_basis = F.gelu(mlp_hidden_basis)

    # Get activations
    activations = get_activations(model, tokens, cfg.batch_size)

    results = {}

    for layer_name in cfg.layers:
        acts = activations[layer_name]
        basis_matrix = (
            mlp_hidden_basis
            if layer_name == "mlp_hidden" and mlp_hidden_basis is not None
            else embedding_matrix
        )

        # Compute projections
        max_proj, mean_other = compute_projections(acts, tokens.cpu(), basis_matrix)

        # Aggregate by K bins
        layer_results = {
            "overall": {
                "max_projection_mean": max_proj.mean().item(),
                "mean_other_projection_mean": mean_other.mean().item(),
                "ratio": max_proj.mean().item() / (mean_other.mean().item() + 1e-8),
            },
            "by_k_bin": {},
        }

        for k_min, k_max in cfg.k_bins:
            mask = ((unique_counts >= k_min) & (unique_counts < k_max)).cpu()
            if mask.sum() > 0:
                layer_results["by_k_bin"][f"{k_min}-{k_max}"] = {
                    "max_projection_mean": max_proj[mask].mean().item(),
                    "mean_other_projection_mean": mean_other[mask].mean().item(),
                    "ratio": max_proj[mask].mean().item()
                    / (mean_other[mask].mean().item() + 1e-8),
                    "count": mask.sum().item(),
                }

        results[layer_name] = layer_results

    return results


def run_experiment(cfg: Config) -> Dict:
    """Run the full experiment across checkpoints."""
    print("\n" + "=" * 70)
    print("AXIS ALIGNMENT EXPERIMENT - OpenWebText + Frozen Position Regression")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_samples: {cfg.n_samples}")
    print(f"  seq_len: {cfg.seq_len}")
    print(f"  Checkpoints: {cfg.checkpoint_steps}")
    print(f"  Device: {device}")
    print(f"  Checkpoint dir: {CKPT_DIR}")

    # Load OpenWebText sequences (same for all checkpoints)
    print("\nLoading OpenWebText validation sequences...")
    tokens = load_owt_sequences(cfg.n_samples, cfg.seq_len, cfg.seed)

    # Count unique tokens per position
    print("Counting unique tokens per position...")
    unique_counts = count_unique_tokens_per_position(tokens)

    # Print K distribution
    final_k = unique_counts[:, -1]  # K at final position
    print(
        f"Unique tokens at final position - min: {final_k.min()}, max: {final_k.max()}, mean: {final_k.float().mean():.1f}"
    )

    all_results = {
        "config": {
            "n_samples": cfg.n_samples,
            "seq_len": cfg.seq_len,
            "checkpoint_steps": cfg.checkpoint_steps,
            "layers": cfg.layers,
            "k_bins": cfg.k_bins,
            "checkpoint_dir": str(CKPT_DIR),
        },
        "k_distribution": {
            "min": final_k.min().item(),
            "max": final_k.max().item(),
            "mean": final_k.float().mean().item(),
        },
        "results_by_step": {},
    }

    for step in tqdm(cfg.checkpoint_steps, desc="Analyzing checkpoints"):
        ckpt_path = CKPT_DIR / f"ckpt_{step:05d}.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint {ckpt_path} not found, skipping...")
            continue

        model, meta = load_checkpoint(str(ckpt_path), device)
        step_results = analyze_checkpoint(model, tokens, unique_counts, cfg)
        all_results["results_by_step"][step] = {"step": step, "results": step_results}

        del model
        torch.cuda.empty_cache()

    return all_results


def create_plots(results: Dict, output_dir: Path, suffix: str = "") -> Dict[str, Path]:
    """Create publication-quality plots."""
    print("\nCreating plots...")

    saved_paths: Dict[str, Path] = {}
    config = results["config"]
    checkpoint_steps = sorted([int(s) for s in results["results_by_step"].keys()])
    layers = config["layers"]
    k_bins = config["k_bins"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # ─── Plot 1: Overall alignment across training ───
    layer_titles = {
        "post_attn": "Post-Attention (Block 0)",
        "post_ln2": "Post-LN2 (Block 0)",
        "mlp_hidden": "MLP Hidden (Block 0)",
        "post_mlp": "Post-MLP (Block 0)",
    }
    subplot_titles = [layer_titles.get(layer, layer) for layer in layers]
    fig1 = make_subplots(
        rows=1,
        cols=len(layers),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
    )

    for layer_idx, layer_name in enumerate(layers):
        col = layer_idx + 1

        max_projs = []
        mean_others = []
        steps_list = []

        for step in checkpoint_steps:
            step_data = results["results_by_step"].get(step, {})
            layer_data = (
                step_data.get("results", {}).get(layer_name, {}).get("overall", {})
            )
            if layer_data:
                max_projs.append(layer_data["max_projection_mean"])
                mean_others.append(layer_data["mean_other_projection_mean"])
                steps_list.append(step)

        if max_projs:
            fig1.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=max_projs,
                    name="Max |cos|" if layer_idx == 0 else None,
                    mode="lines+markers",
                    line=dict(color=colors[0], width=3),
                    marker=dict(size=10),
                    showlegend=(layer_idx == 0),
                ),
                row=1,
                col=col,
            )

            fig1.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=mean_others,
                    name="Mean other |cos|" if layer_idx == 0 else None,
                    mode="lines+markers",
                    line=dict(color=colors[1], width=3, dash="dot"),
                    marker=dict(size=8, symbol="diamond"),
                    showlegend=(layer_idx == 0),
                ),
                row=1,
                col=col,
            )

        fig1.update_xaxes(title_text="Training Step", row=1, col=col)
        fig1.update_yaxes(
            title_text="Mean |cos(h,e)| (pos, seq)" if col == 1 else "", row=1, col=col
        )

    fig1.update_layout(
        title=dict(
            text="Post-MLP Axis Alignment Across Training",
            font=dict(size=20, family="Serif"),
        ),
        template="plotly_white",
        height=460,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=150, t=80, l=70, r=40),
        font=dict(size=14, family="Serif"),
    )

    training_png = output_dir / f"axis_alignment_owt{suffix}_training.png"
    training_pdf = output_dir / f"axis_alignment_owt{suffix}_training.pdf"
    fig1.write_image(str(training_png), width=1000, height=450, scale=3)
    fig1.write_image(str(training_pdf))
    saved_paths["training_png"] = training_png
    saved_paths["training_pdf"] = training_pdf
    print(f"  Saved: axis_alignment_owt{suffix}_training.png/pdf")

    # ─── Plot 2: Alignment by K bins (final checkpoint) ───
    if checkpoint_steps:
        final_step = max(checkpoint_steps)
        final_data = results["results_by_step"].get(final_step, {}).get("results", {})

        subplot_titles = [layer_titles.get(layer, layer) for layer in layers]
        fig2 = make_subplots(
            rows=1,
            cols=len(layers),
            subplot_titles=subplot_titles,
            horizontal_spacing=0.12,
        )

        for layer_idx, layer_name in enumerate(layers):
            col = layer_idx + 1
            layer_data = final_data.get(layer_name, {}).get("by_k_bin", {})

            k_labels = []
            max_projs = []
            mean_others = []

            for k_min, k_max in k_bins:
                bin_key = f"{k_min}-{k_max}"
                if bin_key in layer_data:
                    k_labels.append(bin_key)
                    max_projs.append(layer_data[bin_key]["max_projection_mean"])
                    mean_others.append(
                        layer_data[bin_key]["mean_other_projection_mean"]
                    )

            if max_projs:
                fig2.add_trace(
                    go.Bar(
                        x=k_labels,
                        y=max_projs,
                        name="Max |cos|" if layer_idx == 0 else None,
                        marker_color=colors[0],
                        showlegend=(layer_idx == 0),
                    ),
                    row=1,
                    col=col,
                )

                fig2.add_trace(
                    go.Bar(
                        x=k_labels,
                        y=mean_others,
                        name="Mean other |cos|" if layer_idx == 0 else None,
                        marker_color=colors[1],
                        showlegend=(layer_idx == 0),
                    ),
                    row=1,
                    col=col,
                )

            fig2.update_xaxes(title_text="K (unique tokens)", row=1, col=col)
            fig2.update_yaxes(
                title_text="Mean |cos(h,e)| (pos, seq)" if col == 1 else "",
                row=1,
                col=col,
            )

        fig2.update_layout(
            title=dict(
                text=f"Post-MLP Axis Alignment by K Bins (Step {final_step})",
                font=dict(size=20, family="Serif"),
            ),
            template="plotly_white",
            height=460,
            width=1000,
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.28,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(b=150, t=80, l=70, r=40),
            font=dict(size=14, family="Serif"),
        )

        by_k_png = output_dir / f"axis_alignment_owt{suffix}_by_k.png"
        by_k_pdf = output_dir / f"axis_alignment_owt{suffix}_by_k.pdf"
        fig2.write_image(str(by_k_png), width=1000, height=450, scale=3)
        fig2.write_image(str(by_k_pdf))
        saved_paths["by_k_png"] = by_k_png
        saved_paths["by_k_pdf"] = by_k_pdf
        print(f"  Saved: axis_alignment_owt{suffix}_by_k.png/pdf")

    print("All plots saved!")
    return saved_paths


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    checkpoint_steps = sorted([int(s) for s in results["results_by_step"].keys()])

    if not checkpoint_steps:
        print("No results available.")
        return

    print(f"\nK distribution in OpenWebText sequences:")
    print(f"  Min: {results['k_distribution']['min']}")
    print(f"  Max: {results['k_distribution']['max']}")
    print(f"  Mean: {results['k_distribution']['mean']:.1f}")

    print(f"\nAlignment across training steps (should be CONSTANT for frozen layers):")

    for layer in results["config"]["layers"]:
        print(f"\n  {layer.upper()}:")
        print(f"  {'Step':>8} | {'Max |cos|':>12} | {'Mean |cos|':>12} | {'Ratio':>8}")
        print(f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 8}")

        for step in checkpoint_steps:
            step_data = results["results_by_step"].get(step, {})
            layer_data = step_data.get("results", {}).get(layer, {}).get("overall", {})

            if layer_data:
                print(
                    f"  {step:>8} | {layer_data['max_projection_mean']:>12.4f} | "
                    f"{layer_data['mean_other_projection_mean']:>12.4f} | "
                    f"{layer_data['ratio']:>8.2f}"
                )


def main():
    """Main entry point."""
    setup_dirs()

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or f"axis-alignment-owt{args.output_suffix}"
        wandb_run = wandb.init(
            project=args.wandb_project, name=run_name, config=vars(args)
        )

    layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]

    cfg = Config(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        layers=layers,
    )

    # Run experiment
    results = run_experiment(cfg)

    # Save results
    suffix = args.output_suffix
    results_path = RESULTS_DIR / f"axis_alignment_owt{suffix}_results.json"

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Create plots
    saved_paths = create_plots(results, PLOTS_DIR, suffix)

    if wandb_run is not None:
        import wandb

        for key, path in saved_paths.items():
            if path.suffix == ".png":
                wandb.log({f"plots/{key}": wandb.Image(str(path))})
        wandb.log({"results_path": str(results_path)})
        wandb_run.finish()

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
