"""
Generate visualization of activation norms over positions.
Shows:
1. Mean norm with std bands across positions
2. Individual sample norm trajectories
3. Comparison between random and trained models

Usage:
    python analysis_scripts/visualize_norm_over_positions_flexible.py \
        --checkpoint-dir out-posreg-6layer-until-mlp \
        --output-suffix _6layer
"""

import torch
import numpy as np
import sys
import argparse
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, "nanoGPT")
from model_position_classifier import (
    GPTPositionClassifier as GPT,
    GPTPositionClassifierConfig as GPTConfig,
)

# Settings
device = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 500
n_individual_samples = 20  # Number of individual trajectories to show
n_ctx = 64
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)

SAVE_DIR = Path("results/norm_over_positions")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def create_random_model(config, n_ctx=64):
    """Create a randomly initialized NoPE model with same config."""
    model = GPT(config)
    model.eval()
    model.to(device)
    return model


def load_trained_model(checkpoint_path):
    """Load trained NoPE model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 6),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 128),
        bias=model_args.get("bias", False),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=model_args.get("use_positional_embedding", False),
        norm_type=model_args.get("norm_type", "layernorm"),
        use_regression=model_args.get("use_regression", True),
        compute_lm_loss=model_args.get("compute_lm_loss", False),
        use_ln2=model_args.get("use_ln2", True),
    )

    model = GPT(config)

    # Handle torch.compile prefix
    state_dict = checkpoint["model"]
    new_state_dict = {}
    unwanted_prefix = "_orig_mod."
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            new_state_dict[k[len(unwanted_prefix) :]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model, config


def get_activations_with_norms(model, tokens):
    """Get activations and compute norms at each layer."""
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()

        return hook

    hooks = []

    # Register hooks
    # After attention (before residual)
    # Note: Accessing model.transformer.h[0] directly
    hooks.append(
        model.transformer.h[0].attn.register_forward_hook(hook_fn("post_attn"))
    )
    # After LN2 (before MLP)
    if model.transformer.h[0].use_ln2:
        hooks.append(
            model.transformer.h[0].ln_2.register_forward_hook(hook_fn("post_ln2"))
        )
    else:
        # Fallback if no LN2
        pass

    with torch.no_grad():
        model.transformer(tokens)  # Run transformer forward

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute norms
    norms = {}
    for name, act in activations.items():
        # act shape: [batch, seq, hidden]
        norms[name] = torch.norm(act, dim=-1).cpu().numpy()  # [batch, seq]

    return norms


def collect_norm_statistics(model, n_samples, n_ctx, vocab_size):
    """Collect norm statistics across many samples."""
    all_norms = {"post_attn": [], "post_ln2": []}

    batch_size = 50
    n_batches = n_samples // batch_size

    for _ in range(n_batches):
        tokens = torch.randint(0, vocab_size, (batch_size, n_ctx), device=device)
        norms = get_activations_with_norms(model, tokens)

        for key in all_norms:
            if key in norms:
                all_norms[key].append(norms[key])

    # Concatenate
    for key in all_norms:
        if all_norms[key]:
            all_norms[key] = np.concatenate(
                all_norms[key], axis=0
            )  # [n_samples, n_ctx]
        else:
            all_norms[key] = np.zeros((n_samples, n_ctx))

    return all_norms


def create_norm_visualization(random_norms, trained_norms, n_ctx, suffix=""):
    """Create the visualization figure."""
    positions = np.arange(n_ctx)

    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Random Model: Post-Attention Norms",
            "Trained Model: Post-Attention Norms",
            "Random Model: Post-LN2 Norms",
            "Trained Model: Post-LN2 Norms",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Color schemes
    colors = {
        "random": {
            "mean": "rgb(31, 119, 180)",
            "band": "rgba(31, 119, 180, 0.2)",
            "individual": "rgba(31, 119, 180, 0.3)",
        },
        "trained": {
            "mean": "rgb(255, 127, 14)",
            "band": "rgba(255, 127, 14, 0.2)",
            "individual": "rgba(255, 127, 14, 0.3)",
        },
    }

    def add_norm_plot(norms, row, col, color_scheme, show_legend=False):
        """Add norm plot to subplot."""
        if np.all(norms == 0):
            return  # Skip if no data

        mean_norm = np.mean(norms, axis=0)
        std_norm = np.std(norms, axis=0)

        # Add individual sample trajectories (thin lines)
        n_show = min(n_individual_samples, norms.shape[0])
        indices = np.random.choice(norms.shape[0], n_show, replace=False)

        for i, idx in enumerate(indices):
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=norms[idx],
                    mode="lines",
                    line=dict(color=color_scheme["individual"], width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        # Add std band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([positions, positions[::-1]]),
                y=np.concatenate([mean_norm + std_norm, (mean_norm - std_norm)[::-1]]),
                fill="toself",
                fillcolor=color_scheme["band"],
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=mean_norm,
                mode="lines",
                line=dict(color=color_scheme["mean"], width=3),
                name="Mean ± Std",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

    # Random model plots
    add_norm_plot(random_norms["post_attn"], 1, 1, colors["random"], show_legend=False)
    add_norm_plot(random_norms["post_ln2"], 2, 1, colors["random"], show_legend=False)

    # Trained model plots
    add_norm_plot(
        trained_norms["post_attn"], 1, 2, colors["trained"], show_legend=False
    )
    add_norm_plot(trained_norms["post_ln2"], 2, 2, colors["trained"], show_legend=False)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Activation Norms Across Positions: Random vs Trained{suffix}",
            font=dict(size=20, family="Serif"),
            x=0.5,
        ),
        width=1200,
        height=800,
        template="plotly_white",
        font=dict(family="Serif", size=12),
        showlegend=False,
    )

    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Position", row=row, col=col)
            fig.update_yaxes(title_text="L2 Norm", row=row, col=col)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize Norm Over Positions")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--output-suffix", type=str, default="", help="Suffix for output files"
    )
    parser.add_argument(
        "--step", type=int, default=20000, help="Checkpoint step to analyze"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING NORM VISUALIZATION")
    print("=" * 60)

    checkpoint_dir = Path("nanoGPT") / args.checkpoint_dir
    ckpt_path = checkpoint_dir / f"ckpt_{args.step:05d}.pt"

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return

    # Load trained model
    trained_model, config = load_trained_model(str(ckpt_path))

    # Create random model with same config
    print("\nCreating random model...")
    random_model = create_random_model(config, n_ctx=n_ctx)

    # Collect statistics
    print(f"\nCollecting norm statistics ({n_samples} samples)...")
    print("  Random model...")
    random_norms = collect_norm_statistics(
        random_model, n_samples, n_ctx, config.vocab_size
    )

    print("  Trained model...")
    trained_norms = collect_norm_statistics(
        trained_model, n_samples, n_ctx, config.vocab_size
    )

    # Print statistics
    print("\n" + "=" * 60)
    print("NORM STATISTICS")
    print("=" * 60)

    for model_name, norms in [("Random", random_norms), ("Trained", trained_norms)]:
        print(f"\n{model_name} Model:")
        for layer_name in ["post_attn", "post_ln2"]:
            if np.all(norms[layer_name] == 0):
                continue

            mean_by_pos = np.mean(norms[layer_name], axis=0)
            std_by_pos = np.std(norms[layer_name], axis=0)
            corr = np.corrcoef(np.arange(n_ctx), mean_by_pos)[0, 1]
            print(f"  {layer_name}:")
            print(
                f"    Mean norm range: [{mean_by_pos.min():.2f}, {mean_by_pos.max():.2f}]"
            )
            print(
                f"    Std norm range: [{std_by_pos.min():.2f}, {std_by_pos.max():.2f}]"
            )
            print(f"    Position correlation: r = {corr:.4f}")

    # Create visualizations
    print("\nGenerating figures...")

    # Full 4-panel figure
    fig_full = create_norm_visualization(
        random_norms, trained_norms, n_ctx, suffix=args.output_suffix
    )
    output_path = SAVE_DIR / f"norm_over_positions{args.output_suffix}.png"
    fig_full.write_image(str(output_path), width=1200, height=800, scale=2)
    fig_full.write_image(str(SAVE_DIR / f"norm_over_positions{args.output_suffix}.pdf"))
    print(f"  Saved: {output_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
