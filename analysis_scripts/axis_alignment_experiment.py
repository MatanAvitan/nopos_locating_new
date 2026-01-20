"""
Axis Alignment Experiment: Projection of Representations onto Token Embedding Basis

This script analyzes how well post-attention and post-LN2 representations align with
the token embedding basis. For each position, we compute:
1. Max absolute cosine similarity onto the basis of unique token embeddings in the sequence
2. Mean absolute cosine similarity onto the remaining axes (excluding the max)


The experiment is run across training checkpoints and for different numbers of unique
tokens (K) in the sequence.

Key insight: If NoPE transformers encode position through averaging token embeddings,
we expect activations to project strongly onto the subspace spanned by tokens in the context.

Usage:
    python axis_alignment_experiment.py --gpu 0 --n_samples 1000 --seq_len 64
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
    "--gpu", type=str, default="0", help="GPU(s) to use (e.g., '0' or '2,3')"
)
parser.add_argument(
    "--n_samples", type=int, default=500, help="Number of samples per K"
)
parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig

# ─── Configuration ───────────────────────────────────────────────────────────

RESULTS_DIR = Path("results/axis_alignment")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint directories
LN_CKPT_DIR = Path("nanoGPT/out-nope-1layer-ln")


@dataclass
class AxisAlignmentConfig:
    """Configuration for axis alignment experiment."""

    n_samples: int = 500  # Samples per K value
    seq_len: int = 64
    batch_size: int = 64
    seed: int = 42

    # K values to test (number of unique tokens in the sequence, including base token)
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    # Checkpoints to analyze (every 250 steps as available)
    checkpoint_steps: List[int] = field(
        default_factory=lambda: [
            250,
            500,
            750,
            1000,
            1250,
            1500,
            1750,
            2000,
            2250,
            2500,
            2750,
            3000,
            3250,
            3500,
            3750,
            4000,
            4250,
            4500,
            4750,
            5000,
        ]
    )

    # Layers to analyze
    layers: List[str] = field(default_factory=lambda: ["post_attn", "post_ln2"])


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

    # Handle torch.compile prefix
    state_dict = checkpoint["model"]
    unwrapped = {
        (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(unwrapped)
    model.eval()
    model.to(device)

    meta = {
        "step": checkpoint.get("iter_num", 0),
        "config": model_args,
    }

    return model, meta


def generate_controlled_sequences(
    n_samples: int,
    seq_len: int,
    k_unique_tokens: int,
    vocab_size: int = 50304,
    seed: int = 42,
    base_token: int = 1000,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Generate sequences with exactly k unique tokens in the full sequence.

    Controlled pattern: [t1, t2, ..., t_{k-1}, t0, t0, ...] where t0 is base_token.
    The unique token set always includes base_token, plus (k-1) random tokens.

    Returns:
        tokens: [n_samples, seq_len] token indices
        unique_tokens_per_sample: List of lists, each containing the unique tokens in that sample
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokens = torch.full(
        (n_samples, seq_len), base_token, dtype=torch.long, device=device
    )
    unique_tokens_per_sample = []

    n_extra = max(k_unique_tokens - 1, 0)
    for i in range(n_samples):
        if n_extra > 0:
            sampled = np.random.choice(vocab_size - 1, size=n_extra, replace=False)
            sampled = np.where(sampled >= base_token, sampled + 1, sampled)
            unique_toks = np.concatenate(([base_token], sampled))
        else:
            unique_toks = np.array([base_token])
        unique_tokens_per_sample.append(list(unique_toks))

        # First (k-1) positions are unique tokens, rest are base_token
        for pos in range(min(n_extra, seq_len)):
            tokens[i, pos] = unique_toks[1 + pos]

    return tokens, unique_tokens_per_sample


def get_activations(
    model: GPT, tokens: torch.Tensor, batch_size: int = 64
) -> Dict[str, torch.Tensor]:
    """
    Extract activations at post_attn and post_ln2 layers.

    Returns:
        Dict with keys 'post_attn' and 'post_ln2', each [n_samples, seq_len, d_model]
    """
    model.eval()
    n_samples = tokens.shape[0]

    activations = {"post_attn": [], "post_ln2": []}

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = tokens[i : i + batch_size]

            # Token embeddings (no positional embedding)
            tok_emb = model.transformer.wte(batch)  # [B, T, d_model]
            x = model.transformer.drop(tok_emb)

            # Process through the single block
            block = model.transformer.h[0]

            # LayerNorm 1 -> Attention
            ln1_out = block.ln_1(x)
            attn_out = block.attn(ln1_out)
            activations["post_attn"].append(attn_out.cpu())

            # Residual after attention
            x = x + attn_out

            # LayerNorm 2
            if hasattr(block, "ln_2") and not block.skip_ln2:
                ln2_out = block.ln_2(x)
                activations["post_ln2"].append(ln2_out.cpu())
            else:
                # If no LN2, use post-residual
                activations["post_ln2"].append(x.cpu())

    # Concatenate batches
    for key in activations:
        activations[key] = torch.cat(activations[key], dim=0)

    return activations


def compute_projection_onto_basis(
    activation: torch.Tensor,  # [d_model]
    basis_embeddings: torch.Tensor,  # [k, d_model] - embeddings of unique tokens
) -> Tuple[float, float, int]:
    """
    Compute cosine alignment of activation onto each basis vector (token embedding).

    The basis vectors are the token embeddings of unique tokens in the sequence.
    We compute |cos(h, e)| for each basis direction.

    Returns:
        max_projection: Maximum absolute cosine similarity
        mean_other_projections: Mean of cosine similarity on non-max axes
        max_idx: Index of the max projection basis vector
    """
    # Normalize basis vectors and activation
    basis_norms = torch.norm(basis_embeddings, dim=1, keepdim=True)
    act_norm = torch.norm(activation) + 1e-8
    basis_normalized = basis_embeddings / (basis_norms + 1e-8)

    # Compute cosine similarities with each normalized basis vector
    projections = torch.matmul(basis_normalized, activation) / act_norm  # [k]

    # Take absolute values (we care about magnitude, not sign)
    abs_projections = torch.abs(projections)

    # Find max and compute mean of others
    max_val, max_idx = torch.max(abs_projections, dim=0)

    if len(abs_projections) > 1:
        # Exclude max and compute mean of rest
        mask = torch.ones_like(abs_projections, dtype=torch.bool)
        mask[max_idx] = False
        mean_others = abs_projections[mask].mean()
    else:
        mean_others = torch.tensor(0.0)

    return max_val.item(), mean_others.item(), max_idx.item()


def analyze_axis_alignment_for_checkpoint(
    model: GPT,
    cfg: AxisAlignmentConfig,
    k_value: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Analyze axis alignment for a specific K value.

    For each position i in each sequence:
    - Get the set of unique tokens seen so far (positions 0 to i)
    - This includes the token at position i itself
    - Compute projections onto this basis

    Returns dict with results for each layer.
    """
    # Generate sequences
    tokens, unique_tokens_list = generate_controlled_sequences(
        n_samples=cfg.n_samples,
        seq_len=cfg.seq_len,
        k_unique_tokens=k_value,
        seed=cfg.seed,
    )

    # Get embeddings
    embedding_matrix = model.transformer.wte.weight.data  # [vocab_size, d_model]

    # Get activations
    activations = get_activations(model, tokens, cfg.batch_size)

    results = {}

    for layer_name in cfg.layers:
        acts = activations[layer_name]  # [n_samples, seq_len, d_model]

        max_projections_by_pos = [[] for _ in range(cfg.seq_len)]
        mean_other_projections_by_pos = [[] for _ in range(cfg.seq_len)]

        for sample_idx in range(cfg.n_samples):
            sample_tokens = tokens[sample_idx].cpu().numpy()
            sample_acts = acts[sample_idx]  # [seq_len, d_model]

            for pos in range(cfg.seq_len):
                # Get unique tokens seen up to and including this position
                tokens_seen = set(sample_tokens[: pos + 1].tolist())
                token_list = list(tokens_seen)

                # K is the size of the basis (unique tokens seen so far)
                # This naturally includes the constant token at this position
                basis_size = len(token_list)

                # Get embeddings for these tokens
                basis_embeddings = embedding_matrix[token_list]  # [basis_size, d_model]

                # Get activation at this position
                activation = sample_acts[pos]  # [d_model]

                # Compute projections
                max_proj, mean_other, _ = compute_projection_onto_basis(
                    activation.to(device), basis_embeddings.to(device)
                )

                max_projections_by_pos[pos].append(max_proj)
                mean_other_projections_by_pos[pos].append(mean_other)

        # Aggregate: mean across samples
        results[layer_name] = {
            "max_projection_by_pos": [np.mean(p) for p in max_projections_by_pos],
            "mean_other_projection_by_pos": [
                np.mean(p) for p in mean_other_projections_by_pos
            ],
            "max_projection_std_by_pos": [np.std(p) for p in max_projections_by_pos],
            "mean_other_projection_std_by_pos": [
                np.std(p) for p in mean_other_projections_by_pos
            ],
            # Aggregated across all positions
            "max_projection_mean": np.mean(
                [np.mean(p) for p in max_projections_by_pos]
            ),
            "mean_other_projection_mean": np.mean(
                [np.mean(p) for p in mean_other_projections_by_pos]
            ),
            "ratio_max_to_mean_other": (
                np.mean([np.mean(p) for p in max_projections_by_pos])
                / (np.mean([np.mean(p) for p in mean_other_projections_by_pos]) + 1e-8)
            ),
        }

    return results


def run_axis_alignment_experiment(cfg: AxisAlignmentConfig) -> Dict:
    """Run the full axis alignment experiment across checkpoints and K values."""
    print("\n" + "=" * 70)
    print("AXIS ALIGNMENT EXPERIMENT")
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

        print(f"\n  Loading checkpoint step {step}...")
        model, meta = load_checkpoint(str(ckpt_path), device)

        step_results = {"step": step, "by_k": {}}

        for k_value in cfg.k_values:
            print(f"    Analyzing K={k_value}...")
            k_results = analyze_axis_alignment_for_checkpoint(model, cfg, k_value)
            step_results["by_k"][k_value] = k_results

        all_results["results_by_step"][step] = step_results

        # Clean up
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

    # High-contrast color palette for publication
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    # ─── Plot 0: MAIN PAPER FIGURE - Aggregated by K across training ────────────
    # This is the most important figure: shows max projection and mean other
    # as a function of training steps, with separate lines for each K

    for layer_name in layers:
        fig_main = go.Figure()

        for k_idx, k_value in enumerate(k_values):
            max_projs = []
            mean_other_projs = []
            steps_list = []

            for step in checkpoint_steps:
                step_data = results["results_by_step"].get(step, {})
                if not step_data:
                    continue
                k_data = step_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                if k_data:
                    max_projs.append(k_data["max_projection_mean"])
                    mean_other_projs.append(k_data["mean_other_projection_mean"])
                    steps_list.append(step)

            if max_projs:
                color = colors[k_idx % len(colors)]

                # Max |cos| - solid thick line
                fig_main.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=max_projs,
                        name=f"K={k_value} (max)",
                        mode="lines+markers",
                        line=dict(color=color, width=3),
                        marker=dict(size=8, symbol="circle"),
                    )
                )

                # Mean other |cos| - dashed thin line
                fig_main.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=mean_other_projs,
                        name=f"K={k_value} (others)",
                        mode="lines+markers",
                        line=dict(color=color, width=2, dash="dot"),
                        marker=dict(size=5, symbol="diamond"),
                    )
                )

        layer_display = "Post-Attention" if layer_name == "post_attn" else "Post-LN2"
        fig_main.update_layout(
            title=dict(
                text=f"Axis Alignment ({layer_display}): Max vs Mean Projection onto Token Embedding Basis",
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

        fig_main.write_image(
            str(output_dir / f"axis_alignment_main_{layer_name}.png"),
            width=900,
            height=550,
            scale=3,
        )
        fig_main.write_image(str(output_dir / f"axis_alignment_main_{layer_name}.pdf"))
        print(f"  Saved: axis_alignment_main_{layer_name}.png/pdf")

    # ─── Plot 1: Aggregated results vs training epochs (main figure) ────────────
    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Post-Attention", "Post-LN2"],
        horizontal_spacing=0.12,
    )

    for layer_idx, layer_name in enumerate(layers):
        col = layer_idx + 1

        for k_idx, k_value in enumerate(k_values):
            max_projs = []
            mean_other_projs = []
            steps_list = []

            for step in checkpoint_steps:
                step_data = results["results_by_step"].get(step, {})
                if not step_data:
                    continue
                k_data = step_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                if k_data:
                    max_projs.append(k_data["max_projection_mean"])
                    mean_other_projs.append(k_data["mean_other_projection_mean"])
                    steps_list.append(step)

            if max_projs:
                # Max |cos| line (solid)
                fig1.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=max_projs,
                        name=f"K={k_value} (max)" if layer_idx == 0 else None,
                        mode="lines+markers",
                        line=dict(color=colors[k_idx % len(colors)], width=2),
                        marker=dict(size=6),
                        legendgroup=f"k{k_value}",
                        showlegend=(layer_idx == 0),
                    ),
                    row=1,
                    col=col,
                )

                # Mean other |cos| line (dashed)
                fig1.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=mean_other_projs,
                        name=f"K={k_value} (mean other)" if layer_idx == 0 else None,
                        mode="lines+markers",
                        line=dict(
                            color=colors[k_idx % len(colors)], width=2, dash="dash"
                        ),
                        marker=dict(size=4, symbol="diamond"),
                        legendgroup=f"k{k_value}",
                        showlegend=(layer_idx == 0),
                    ),
                    row=1,
                    col=col,
                )

        fig1.update_xaxes(title_text="Training Step", row=1, col=col)
        fig1.update_yaxes(
            title_text="Mean |cos(h,e)|" if col == 1 else "", row=1, col=col
        )

    fig1.update_layout(
        title=dict(
            text="Axis Alignment: Max vs Mean Other Projections Across Training",
            font=dict(size=18, family="Serif"),
        ),
        template="plotly_white",
        height=450,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=100),
    )

    fig1.write_image(
        str(output_dir / "axis_alignment_training_dynamics.png"),
        width=1000,
        height=450,
        scale=2,
    )
    fig1.write_image(str(output_dir / "axis_alignment_training_dynamics.pdf"))
    print(f"  Saved: axis_alignment_training_dynamics.png/pdf")

    # ─── Plot 2: Ratio of max to mean other projections ─────────────────────────
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Post-Attention", "Post-LN2"],
        horizontal_spacing=0.12,
    )

    for layer_idx, layer_name in enumerate(layers):
        col = layer_idx + 1

        for k_idx, k_value in enumerate(k_values):
            ratios = []
            steps_list = []

            for step in checkpoint_steps:
                step_data = results["results_by_step"].get(step, {})
                if not step_data:
                    continue
                k_data = step_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                if k_data:
                    ratios.append(k_data["ratio_max_to_mean_other"])
                    steps_list.append(step)

            if ratios:
                fig2.add_trace(
                    go.Scatter(
                        x=steps_list,
                        y=ratios,
                        name=f"K={k_value}",
                        mode="lines+markers",
                        line=dict(color=colors[k_idx % len(colors)], width=2),
                        marker=dict(size=6),
                        showlegend=(layer_idx == 0),
                    ),
                    row=1,
                    col=col,
                )

        fig2.update_xaxes(title_text="Training Step", row=1, col=col)
        fig2.update_yaxes(
            title_text="Ratio (max/mean other)" if col == 1 else "", row=1, col=col
        )

    fig2.update_layout(
        title=dict(
            text="Ratio of Max to Mean Other Projections Across Training",
            font=dict(size=18, family="Serif"),
        ),
        template="plotly_white",
        height=400,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=100),
    )

    fig2.write_image(
        str(output_dir / "axis_alignment_ratio.png"), width=1000, height=400, scale=2
    )
    fig2.write_image(str(output_dir / "axis_alignment_ratio.pdf"))
    print(f"  Saved: axis_alignment_ratio.png/pdf")

    # ─── Plot 3: By position at final checkpoint ────────────────────────────────
    final_step = max(checkpoint_steps)

    fig3 = make_subplots(
        rows=2,
        cols=len(k_values),
        subplot_titles=[f"K={k}" for k in k_values] * 2,
        row_titles=["Post-Attention", "Post-LN2"],
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
    )

    final_data = results["results_by_step"].get(final_step, {})
    seq_len = config["seq_len"]
    positions = list(range(seq_len))

    for layer_idx, layer_name in enumerate(layers):
        row = layer_idx + 1

        for k_idx, k_value in enumerate(k_values):
            col = k_idx + 1
            k_data = final_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})

            if k_data:
                max_by_pos = k_data["max_projection_by_pos"]
                mean_other_by_pos = k_data["mean_other_projection_by_pos"]

                fig3.add_trace(
                    go.Scatter(
                        x=positions,
                        y=max_by_pos,
                        name="Max proj",
                        mode="lines",
                        line=dict(color="blue", width=1.5),
                        showlegend=(layer_idx == 0 and k_idx == 0),
                    ),
                    row=row,
                    col=col,
                )

                fig3.add_trace(
                    go.Scatter(
                        x=positions,
                        y=mean_other_by_pos,
                        name="Mean other",
                        mode="lines",
                        line=dict(color="red", width=1.5, dash="dash"),
                        showlegend=(layer_idx == 0 and k_idx == 0),
                    ),
                    row=row,
                    col=col,
                )

            if row == 2:
                fig3.update_xaxes(title_text="Position", row=row, col=col)

    fig3.update_layout(
        title=dict(
            text=f"Projections by Position at Step {final_step}",
            font=dict(size=18, family="Serif"),
        ),
        template="plotly_white",
        height=520,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=120),
    )

    fig3.write_image(
        str(output_dir / "axis_alignment_by_position.png"),
        width=1200,
        height=500,
        scale=2,
    )
    fig3.write_image(str(output_dir / "axis_alignment_by_position.pdf"))
    print(f"  Saved: axis_alignment_by_position.png/pdf")

    # ─── Plot 4: Heatmap of ratios (K vs Step) for each layer ───────────────────
    fig4 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Post-Attention", "Post-LN2"],
        horizontal_spacing=0.1,
    )

    for layer_idx, layer_name in enumerate(layers):
        col = layer_idx + 1

        ratio_matrix = []
        for k_value in k_values:
            row_data = []
            for step in checkpoint_steps:
                step_data = results["results_by_step"].get(step, {})
                k_data = step_data.get("by_k", {}).get(k_value, {}).get(layer_name, {})
                if k_data:
                    row_data.append(k_data["ratio_max_to_mean_other"])
                else:
                    row_data.append(np.nan)
            ratio_matrix.append(row_data)

        fig4.add_trace(
            go.Heatmap(
                z=ratio_matrix,
                x=checkpoint_steps,
                y=[f"K={k}" for k in k_values],
                colorscale="Viridis",
                colorbar=dict(title="Ratio", x=1.02 if col == 2 else 0.45),
                showscale=(col == 2),
            ),
            row=1,
            col=col,
        )

        fig4.update_xaxes(title_text="Training Step", row=1, col=col)
        fig4.update_yaxes(title_text="", row=1, col=col)

    fig4.update_layout(
        title=dict(
            text="Ratio (Max/Mean Other) Heatmap: K vs Training Step",
            font=dict(size=18, family="Serif"),
        ),
        template="plotly_white",
        height=350,
        width=1000,
    )

    fig4.write_image(
        str(output_dir / "axis_alignment_heatmap.png"), width=1000, height=350, scale=2
    )
    fig4.write_image(str(output_dir / "axis_alignment_heatmap.pdf"))
    print(f"  Saved: axis_alignment_heatmap.png/pdf")

    # ─── Plot 5: Aggregated summary figure (main paper figure) ──────────────────
    # Single figure showing evolution of mean projections across training
    # with error bands

    fig5 = go.Figure()

    # Use final layer (post_ln2) for main figure
    layer_name = "post_ln2"

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
            # Max |cos|
            fig5.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=max_projs,
                    name=f"K={k_value} (max)",
                    mode="lines+markers",
                    line=dict(color=colors[k_idx % len(colors)], width=2.5),
                    marker=dict(size=8),
                )
            )

            # Mean other |cos| (thinner, dashed)
            fig5.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=mean_other_projs,
                    name=f"K={k_value} (others)",
                    mode="lines+markers",
                    line=dict(color=colors[k_idx % len(colors)], width=1.5, dash="dot"),
                    marker=dict(size=5, symbol="diamond"),
                )
            )

    fig5.update_layout(
        title=dict(
            text="Post-LN2 Axis Alignment: Projection onto Token Embedding Basis",
            font=dict(size=20, family="Serif"),
        ),
        xaxis_title="Training Step",
        yaxis_title="Mean |cos(h,e)|",
        template="plotly_white",
        height=520,
        width=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=120),
        font=dict(size=14, family="Serif"),
    )

    fig5.write_image(
        str(output_dir / "axis_alignment_summary.png"), width=800, height=500, scale=2
    )
    fig5.write_image(str(output_dir / "axis_alignment_summary.pdf"))
    print(f"  Saved: axis_alignment_summary.png/pdf")

    print("\nAll plots saved!")


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("AXIS ALIGNMENT RESULTS SUMMARY")
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
            f"  {'K':>4} | {'Max Proj (first)':>16} | {'Max Proj (final)':>16} | {'Ratio (final)':>14}"
        )
        print(f"  {'-' * 4}-+-{'-' * 16}-+-{'-' * 16}-+-{'-' * 14}")

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
            final_ratio = final_data.get("ratio_max_to_mean_other", 0)

            print(
                f"  {k:>4} | {first_max:>16.4f} | {final_max:>16.4f} | {final_ratio:>14.2f}"
            )


def main():
    """Main entry point."""
    setup_dirs()

    cfg = AxisAlignmentConfig(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    # Run experiment
    results = run_axis_alignment_experiment(cfg)

    # Save results
    results_path = RESULTS_DIR / "axis_alignment_results.json"

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
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
