"""
Axis Alignment Experiment: Optimized Version

Vectorized computation of cosine alignment of representations onto token embedding basis.

This script analyzes how MLP activations align with token embedding basis:
1. Max absolute cosine similarity onto the basis of unique token embeddings in the sequence
2. Mean absolute cosine similarity onto the remaining axes (excluding the max)


Usage:
    python axis_alignment_experiment_optimized.py --gpu "2,3" --n_samples 500
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json

# Parse args early for GPU setting
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0", help="GPU(s) to use")
parser.add_argument(
    "--n_samples", type=int, default=200, help="Number of samples per K"
)
parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument(
    "--layers",
    type=str,
    default="post_attn,post_ln2,mlp_hidden,post_mlp",
    help="Comma-separated layers to analyze",
)
parser.add_argument(
    "--plots_dir",
    type=str,
    default="overleaf/nopos_icml_2026/plots",
    help="Directory for plot outputs",
)
args = parser.parse_args()

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

RESULTS_DIR = Path("results/axis_alignment")
PLOTS_DIR = Path(args.plots_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint directories
LN_CKPT_DIR = Path("nanoGPT/out-nope-1layer-ln")


@dataclass
class AxisAlignmentConfig:
    """Configuration for axis alignment experiment."""

    n_samples: int = 200
    seq_len: int = 64
    batch_size: int = 32
    seed: int = 42

    # K values (number of unique tokens in the sequence, including base token)
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    # Key checkpoints for training dynamics (reduced for speed)
    checkpoint_steps: List[int] = field(
        default_factory=lambda: [250, 500, 1000, 2000, 3000, 4000, 5000]
    )

    # Layers to analyze
    layers: List[str] = field(
        default_factory=lambda: ["post_attn", "post_ln2", "mlp_hidden", "post_mlp"]
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
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 256),
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
    model.load_state_dict(unwrapped)
    model.eval()
    model.to(device)

    return model, {"step": checkpoint.get("iter_num", 0), "config": model_args}


def generate_controlled_sequences(
    n_samples: int,
    seq_len: int,
    k_unique: int,
    vocab_size: int = 50304,
    seed: int = 42,
    base_token: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequences with exactly k unique tokens, including base_token.

    Controlled pattern: [t1, t2, ..., t_{k-1}, t0, t0, ...] where t0 is base_token.

    Returns:
        tokens: [n_samples, seq_len] token indices
        unique_tokens: [n_samples, k_unique] the unique tokens per sample
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokens = torch.full(
        (n_samples, seq_len), base_token, dtype=torch.long, device=device
    )
    unique_tokens = torch.zeros(n_samples, k_unique, dtype=torch.long, device=device)

    n_extra = max(k_unique - 1, 0)
    for i in range(n_samples):
        if n_extra > 0:
            sampled = np.random.choice(vocab_size - 1, size=n_extra, replace=False)
            sampled = np.where(sampled >= base_token, sampled + 1, sampled)
            unique_toks = np.concatenate(([base_token], sampled))
        else:
            unique_toks = np.array([base_token])
        unique_tokens[i] = torch.tensor(unique_toks, device=device)

        # First (k-1) positions are unique tokens, rest are base_token
        for pos in range(min(n_extra, seq_len)):
            tokens[i, pos] = unique_toks[1 + pos]

    return tokens, unique_tokens


def get_activations(
    model: GPT, tokens: torch.Tensor, batch_size: int = 32
) -> Dict[str, torch.Tensor]:
    """Extract activations for the first block (post_attn, post_ln2, mlp_hidden, post_mlp)."""
    model.eval()
    n_samples = tokens.shape[0]

    post_attn_list = []
    post_ln2_list = []
    mlp_hidden_list = []
    post_mlp_list = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = tokens[i : i + batch_size]

            # Token embeddings
            tok_emb = model.transformer.wte(batch)
            x = model.transformer.drop(tok_emb)

            # Process through single block
            block = model.transformer.h[0]

            # LN1 -> Attention
            ln1_out = block.ln_1(x)
            attn_out = block.attn(ln1_out)
            post_attn_list.append(attn_out)

            # Residual after attention
            x = x + attn_out

            # LN2
            if hasattr(block, "ln_2") and not getattr(block, "skip_ln2", False):
                ln2_out = block.ln_2(x)
            else:
                ln2_out = x
            post_ln2_list.append(ln2_out)

            # MLP hidden + output
            mlp_hidden = block.mlp.gelu(block.mlp.c_fc(ln2_out))
            mlp_hidden_list.append(mlp_hidden)
            mlp_out = block.mlp.c_proj(mlp_hidden)
            mlp_out = block.mlp.dropout(mlp_out)

            x = x + mlp_out
            post_mlp_list.append(x)

    return {
        "post_attn": torch.cat(post_attn_list, dim=0),
        "post_ln2": torch.cat(post_ln2_list, dim=0),
        "mlp_hidden": torch.cat(mlp_hidden_list, dim=0),
        "post_mlp": torch.cat(post_mlp_list, dim=0),
    }


def compute_projections_vectorized(
    activations: torch.Tensor,  # [n_samples, seq_len, d_model]
    basis_matrix: torch.Tensor,  # [vocab_size, d_model]
    unique_tokens: torch.Tensor,  # [n_samples, k]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized computation of cosine alignment.

    For simplicity, uses the full set of k unique tokens as basis for all positions.

    Returns:
        max_projections: [n_samples, seq_len] max absolute cosine per position
        mean_other_projections: [n_samples, seq_len] mean absolute cosine of others
    """
    n_samples, seq_len, d_model = activations.shape
    k = unique_tokens.shape[1]

    # Get basis embeddings for each sample: [n_samples, k, d_model]
    basis_embeddings = basis_matrix[unique_tokens]

    # Normalize basis vectors
    basis_norms = torch.norm(basis_embeddings, dim=2, keepdim=True)
    basis_normalized = basis_embeddings / (basis_norms + 1e-8)

    # Compute cosine similarities: [n_samples, seq_len, k]
    act_norms = torch.norm(activations, dim=2, keepdim=True)
    projections = torch.einsum("nsd,nkd->nsk", activations, basis_normalized)
    projections = projections / (act_norms + 1e-8)

    # Take absolute values
    abs_projections = torch.abs(projections)

    # Max |cos| per position
    max_projections, _ = torch.max(abs_projections, dim=2)  # [n_samples, seq_len]

    # Mean of other projections (excluding max)
    # Sum all, subtract max, divide by (k-1)
    sum_projections = abs_projections.sum(dim=2)  # [n_samples, seq_len]
    if k > 1:
        mean_other_projections = (sum_projections - max_projections) / (k - 1)
    else:
        mean_other_projections = torch.zeros_like(max_projections)

    return max_projections, mean_other_projections


def analyze_checkpoint(model: GPT, cfg: AxisAlignmentConfig) -> Dict:
    """Analyze axis alignment for a checkpoint."""
    results = {}

    # Get embeddings
    embedding_matrix = model.transformer.wte.weight.data

    # Precompute MLP hidden basis if needed
    mlp_hidden_basis = None
    if "mlp_hidden" in cfg.layers:
        block = model.transformer.h[0]
        mlp_hidden_basis = F.linear(
            embedding_matrix, block.mlp.c_fc.weight, block.mlp.c_fc.bias
        )
        mlp_hidden_basis = F.gelu(mlp_hidden_basis)

    for k_value in cfg.k_values:
        print(f"    K={k_value} unique tokens")

        # Generate controlled sequences
        tokens, unique_tokens = generate_controlled_sequences(
            n_samples=cfg.n_samples,
            seq_len=cfg.seq_len,
            k_unique=k_value,
            seed=cfg.seed,
        )

        # Get activations
        activations = get_activations(model, tokens, cfg.batch_size)

        # Compute projections for each layer
        results[k_value] = {}
        for layer_name in cfg.layers:
            acts = activations[layer_name]
            basis_matrix = (
                mlp_hidden_basis
                if layer_name == "mlp_hidden" and mlp_hidden_basis is not None
                else embedding_matrix
            )
            max_proj, mean_other = compute_projections_vectorized(
                acts, basis_matrix, unique_tokens
            )

            # Store results
            results[k_value][layer_name] = {
                "max_projection_mean": max_proj.mean().item(),
                "mean_other_projection_mean": mean_other.mean().item(),
                "ratio": max_proj.mean().item() / (mean_other.mean().item() + 1e-8),
            }

    return results


def run_experiment(cfg: AxisAlignmentConfig) -> Dict:
    """Run the full experiment across checkpoints."""
    print("\n" + "=" * 70)
    print("AXIS ALIGNMENT EXPERIMENT (OPTIMIZED)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_samples: {cfg.n_samples}")
    print(f"  seq_len: {cfg.seq_len}")
    print(f"  K values: {cfg.k_values}")
    print(f"  Checkpoints: {cfg.checkpoint_steps}")
    print(f"  Device: {device}")

    all_results = {
        "config": {
            "n_samples": cfg.n_samples,
            "seq_len": cfg.seq_len,
            "k_values": cfg.k_values,
            "checkpoint_steps": cfg.checkpoint_steps,
            "layers": cfg.layers,
        },
        "results_by_step": {},
    }

    for step in tqdm(cfg.checkpoint_steps, desc="Analyzing checkpoints"):
        ckpt_path = LN_CKPT_DIR / f"ckpt_{step:05d}.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint {ckpt_path} not found, skipping...")
            continue

        model, meta = load_checkpoint(str(ckpt_path), device)
        step_results = analyze_checkpoint(model, cfg)
        all_results["results_by_step"][step] = {"step": step, "by_k": step_results}

        del model
        torch.cuda.empty_cache()

    return all_results


def create_paper_ready_plots(results: Dict, output_dir: Path):
    """Create publication-quality plots."""
    print("\nCreating paper-ready plots...")

    config = results["config"]
    k_values = config["k_values"]
    checkpoint_steps = sorted([int(s) for s in results["results_by_step"].keys()])
    layers = config["layers"]

    layer_titles = {
        "post_attn": "Post-Attention",
        "post_ln2": "Post-LayerNorm",
        "mlp_hidden": "MLP Hidden",
        "post_mlp": "Post-MLP",
    }

    # High-contrast color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # ─── Main Figure: Per-layer plots with all K values ──────────────────────────
    for layer_name in layers:
        fig = go.Figure()

        for k_idx, k_value in enumerate(k_values):
            max_projs = []
            mean_other_projs = []
            steps_list = []

            for step in checkpoint_steps:
                step_data = results["results_by_step"].get(step, {})
                k_data = step_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                if k_data:
                    max_projs.append(k_data["max_projection_mean"])
                    mean_other_projs.append(k_data["mean_other_projection_mean"])
                    steps_list.append(step)

            if max_projs:
                color = colors[k_idx % len(colors)]

                # Max |cos| - solid line
                fig.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=max_projs,
                        name=f"K={k_value} (max)",
                        mode="lines+markers",
                        line=dict(color=color, width=3),
                        marker=dict(size=8),
                    )
                )

                # Mean other |cos| - dashed line
                fig.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=mean_other_projs,
                        name=f"K={k_value} (others)",
                        mode="lines+markers",
                        line=dict(color=color, width=2, dash="dot"),
                        marker=dict(size=5, symbol="diamond"),
                    )
                )

        layer_display = layer_titles.get(layer_name, layer_name)
        fig.update_layout(
            title=dict(
                text=f"Axis Alignment ({layer_display}): Cosine Alignment",
                font=dict(size=20, family="Serif"),
            ),
            xaxis_title=dict(text="Training Step", font=dict(size=16, family="Serif")),
            yaxis_title=dict(
                text="Mean |cos(h,e)|", font=dict(size=16, family="Serif")
            ),
            template="plotly_white",
            height=550,
            width=900,
            legend=dict(
                title=dict(text="Unique Tokens (K)", font=dict(size=14)),
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(b=140),
            font=dict(size=14, family="Serif"),
        )

        fig.write_image(
            str(output_dir / f"axis_alignment_main_{layer_name}.png"),
            width=900,
            height=550,
            scale=3,
        )
        fig.write_image(str(output_dir / f"axis_alignment_main_{layer_name}.pdf"))
        print(f"  Saved: axis_alignment_main_{layer_name}.png/pdf")

    # ─── Combined figure (all layers side-by-side) ───────────────────────────────
    if len(layers) > 1:
        subplot_titles = [layer_titles.get(layer, layer) for layer in layers]
        fig_combined = make_subplots(
            rows=1,
            cols=len(layers),
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
        )

        for layer_idx, layer_name in enumerate(layers):
            col = layer_idx + 1

            for k_idx, k_value in enumerate(k_values):
                max_projs = []
                mean_other_projs = []
                steps_list = []

                for step in checkpoint_steps:
                    step_data = results["results_by_step"].get(step, {})
                    k_data = (
                        step_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                    )
                    if k_data:
                        max_projs.append(k_data["max_projection_mean"])
                        mean_other_projs.append(k_data["mean_other_projection_mean"])
                        steps_list.append(step)

                if max_projs:
                    color = colors[k_idx % len(colors)]

                    fig_combined.add_trace(
                        go.Scatter(
                            x=steps_list,
                            y=max_projs,
                            name=f"K={k_value} (max)" if layer_idx == 0 else None,
                            mode="lines+markers",
                            line=dict(color=color, width=2.5),
                            marker=dict(size=7),
                            legendgroup=f"k{k_value}",
                            showlegend=(layer_idx == 0),
                        ),
                        row=1,
                        col=col,
                    )

                    fig_combined.add_trace(
                        go.Scatter(
                            x=steps_list,
                            y=mean_other_projs,
                            name=f"K={k_value} (others)" if layer_idx == 0 else None,
                            mode="lines+markers",
                            line=dict(color=color, width=1.5, dash="dot"),
                            marker=dict(size=4, symbol="diamond"),
                            legendgroup=f"k{k_value}",
                            showlegend=(layer_idx == 0),
                        ),
                        row=1,
                        col=col,
                    )

            fig_combined.update_xaxes(title_text="Training Step", row=1, col=col)
            fig_combined.update_yaxes(
                title_text="Mean |cos(h,e)|" if col == 1 else "", row=1, col=col
            )

        fig_combined.update_layout(
            title=dict(
                text="Axis Alignment: Max vs Mean |cos|",
                font=dict(size=20, family="Serif"),
            ),
            template="plotly_white",
            height=500,
            width=300 + 250 * len(layers),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(b=120),
            font=dict(size=14, family="Serif"),
        )

        fig_combined.write_image(
            str(output_dir / "axis_alignment_combined.png"),
            width=300 + 250 * len(layers),
            height=500,
            scale=3,
        )
        fig_combined.write_image(str(output_dir / "axis_alignment_combined.pdf"))
        print("  Saved: axis_alignment_combined.png/pdf")

    # ─── Ratio plot (max / mean other) ───────────────────────────────────────────
    subplot_titles = [layer_titles.get(layer, layer) for layer in layers]
    fig_ratio = make_subplots(
        rows=1,
        cols=len(layers),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    for layer_idx, layer_name in enumerate(layers):
        col = layer_idx + 1

        for k_idx, k_value in enumerate(k_values):
            ratios = []
            steps_list = []

            for step in checkpoint_steps:
                step_data = results["results_by_step"].get(step, {})
                k_data = step_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                if k_data:
                    ratios.append(k_data["ratio"])
                    steps_list.append(step)

            if ratios:
                fig_ratio.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=ratios,
                        name=f"K={k_value}",
                        mode="lines+markers",
                        line=dict(color=colors[k_idx % len(colors)], width=2.5),
                        marker=dict(size=7),
                        showlegend=(layer_idx == 0),
                    ),
                    row=1,
                    col=col,
                )

        fig_ratio.update_xaxes(title_text="Training Step", row=1, col=col)
        fig_ratio.update_yaxes(
            title_text="Ratio (Max / Mean Other)" if col == 1 else "", row=1, col=col
        )

    fig_ratio.update_layout(
        title=dict(
            text="Axis Alignment: Ratio of Max to Mean Other",
            font=dict(size=20, family="Serif"),
        ),
        template="plotly_white",
        height=470,
        width=300 + 250 * len(layers),
        legend=dict(
            title=dict(text="K (unique tokens)", font=dict(size=12)),
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=120),
        font=dict(size=14, family="Serif"),
    )

    fig_ratio.write_image(
        str(output_dir / "axis_alignment_ratio.png"),
        width=300 + 250 * len(layers),
        height=450,
        scale=3,
    )
    fig_ratio.write_image(str(output_dir / "axis_alignment_ratio.pdf"))
    print("  Saved: axis_alignment_ratio.png/pdf")

    # ─── By position plot (final checkpoint) ─────────────────────────────────────
    if checkpoint_steps:
        final_step = max(checkpoint_steps)
        final_data = results["results_by_step"].get(final_step, {})
        seq_len = config["seq_len"]
        positions = list(range(seq_len))

        for layer_name in layers:
            fig_pos = go.Figure()
            has_position_data = False

            for k_idx, k_value in enumerate(k_values):
                k_data = final_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                if "max_projection_by_pos" in k_data:
                    has_position_data = True
                    fig_pos.add_trace(
                        go.Scatter(
                            x=positions,
                            y=k_data["max_projection_by_pos"],
                            name=f"K={k_value} (max)",
                            mode="lines",
                            line=dict(color=colors[k_idx % len(colors)], width=2),
                        )
                    )
                    fig_pos.add_trace(
                        go.Scatter(
                            x=positions,
                            y=k_data["mean_other_projection_by_pos"],
                            name=f"K={k_value} (others)",
                            mode="lines",
                            line=dict(
                                color=colors[k_idx % len(colors)], width=1.5, dash="dot"
                            ),
                        )
                    )

            if not has_position_data:
                print(
                    f"  Skipping by-position plot for {layer_name} (no per-position data)"
                )
                continue

            layer_display = layer_titles.get(layer_name, layer_name)
            fig_pos.update_layout(
                title=dict(
                    text=f"Alignment by Position ({layer_display}, Step {final_step})",
                    font=dict(size=18, family="Serif"),
                ),
                xaxis_title="Position",
                yaxis_title="Mean |cos(h,e)|",
                template="plotly_white",
                height=470,
                width=900,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(b=120),
                font=dict(size=14, family="Serif"),
            )

            fig_pos.write_image(
                str(output_dir / f"axis_alignment_by_position_{layer_name}.png"),
                width=900,
                height=470,
                scale=3,
            )
            fig_pos.write_image(
                str(output_dir / f"axis_alignment_by_position_{layer_name}.pdf")
            )
            print(f"  Saved: axis_alignment_by_position_{layer_name}.png/pdf")

    print("\nAll plots saved!")


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    config = results["config"]
    k_values = config["k_values"]
    checkpoint_steps = sorted([int(s) for s in results["results_by_step"].keys()])

    if not checkpoint_steps:
        print("No results available.")
        return

    final_step = max(checkpoint_steps)
    first_step = min(checkpoint_steps)

    print(f"\nComparing step {first_step} to step {final_step}:")

    for layer in config["layers"]:
        print(f"\n  {layer.upper()}:")
        print(
            f"  {'K':>4} | {'Max(first)':>12} | {'Max(final)':>12} | {'Others(final)':>14} | {'Ratio(final)':>12}"
        )
        print(f"  {'-' * 4}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 14}-+-{'-' * 12}")

        for k in k_values:
            first_data = (
                results["results_by_step"]
                .get(first_step, {})
                .get("by_k", {})
                .get(k, {})
                .get(layer, {})
            )
            final_data = (
                results["results_by_step"]
                .get(final_step, {})
                .get("by_k", {})
                .get(k, {})
                .get(layer, {})
            )

            first_max = first_data.get("max_projection_mean", 0)
            final_max = final_data.get("max_projection_mean", 0)
            final_others = final_data.get("mean_other_projection_mean", 0)
            final_ratio = final_data.get("ratio", 0)

            print(
                f"  {k:>4} | {first_max:>12.4f} | {final_max:>12.4f} | {final_others:>14.4f} | {final_ratio:>12.2f}"
            )


def main():
    """Main entry point."""
    setup_dirs()

    layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]

    cfg = AxisAlignmentConfig(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        layers=layers,
    )

    # Run experiment
    results = run_experiment(cfg)

    # Save results
    results_path = RESULTS_DIR / "axis_alignment_results.json"

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Create plots
    create_paper_ready_plots(results, PLOTS_DIR)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
