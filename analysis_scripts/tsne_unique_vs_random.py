"""
t-SNE Visualization: Unique Prefix vs Random Sequences

Compares two experimental setups side-by-side:
1. Unique Prefix: Each sample i has i unique prefix tokens
   - Sample 0: [1000, 1000, 1000, ...]
   - Sample 1: [1001, 1000, 1000, ...]
   - Sample 2: [1001, 1002, 1000, ...]
   - ...
   - Sample 23: [1001, 1002, ..., 1023, 1000, ...]
2. Random: Completely random sequences (different for each sample)

t-SNE visualizes ALL token positions (n_samples × seq_len points)
Default: 24 samples × 128 positions = 3072 points

Plots 3 layers:
- post_ln2: After second LayerNorm (frozen)
- mlp_hidden: After GELU activation inside first MLP (trainable)
- post_mlp: After whole first MLP (trainable)

Usage:
    CUDA_VISIBLE_DEVICES=4 python analysis_scripts/tsne_unique_vs_random.py \
        --checkpoint-dir out-posreg-6layer-until-mlp \
        --experiment-name nope-6layer-until-first-mlp \
        --n-samples 24
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_position_classifier import (
    GPTPositionClassifier as GPT,
    GPTPositionClassifierConfig as GPTConfig,
)

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Plots will only be saved locally.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent

# Checkpoints to visualize (selected key points for faster execution)
CHECKPOINT_STEPS = [500, 2000, 5000, 10000, 15000, 20000]

# Layers to visualize
LAYERS_TO_PLOT = ["post_ln2", "mlp_hidden", "post_mlp"]


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[GPT, dict]:
    """Load a model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    model_args = checkpoint.get("model_args", {})

    # Create config
    gptconf = GPTConfig(
        n_layer=model_args.get("n_layer", 6),
        n_head=model_args.get("n_head", 1),
        n_embd=model_args.get("n_embd", 7),
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

    # Create and load model
    model = GPT(gptconf)

    # Handle state dict
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Extract training metadata
    meta = {
        "step": checkpoint.get("iter_num", 0),
        "train_loss": checkpoint.get("best_val_loss", None),
        "config": model_args,
    }

    return model, meta


def get_activations(
    model: GPT,
    tokens: torch.Tensor,
    layer_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Extract activations at key points including MLP hidden layer.

    Returns dict with:
    - post_ln2: After second LayerNorm
    - mlp_hidden: After GELU activation inside MLP (4*n_embd dimensional)
    - post_mlp: After whole MLP (before residual)
    """
    activations = {}

    with torch.no_grad():
        # Embedding
        tok_emb = model.transformer.wte(tokens)
        if "wpe" in model.transformer and model.config.use_positional_embedding:
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        x = model.transformer.drop(x)

        # Get the specified block
        block = model.transformer.h[layer_idx]

        # Post-LN1 and Attention
        x_ln1 = block.ln_1(x)
        attn_out = block.attn(x_ln1)

        # After first residual connection
        x = x + attn_out

        # Post-LN2 (BEFORE MLP)
        if block.use_ln2:
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.clone()
            mlp_input = x

        # Inside MLP - extract hidden activation after GELU
        mlp = block.mlp
        mlp_fc = mlp.c_fc(mlp_input)  # First linear layer
        mlp_after_gelu = mlp.gelu(mlp_fc)  # After activation
        activations["mlp_hidden"] = mlp_after_gelu.clone()

        # Complete MLP forward
        mlp_proj = mlp.c_proj(mlp_after_gelu)
        mlp_out = mlp.dropout(mlp_proj)
        activations["post_mlp"] = mlp_out.clone()

    return activations


def generate_unique_prefix_sequences(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
    base_token: int = 1000,
) -> torch.Tensor:
    """
    Generate sequences where EACH sample i has i unique prefix tokens.

    The sequence structure:
    - Sample 0: [1000, 1000, 1000, ...] (all base_token)
    - Sample 1: [1001, 1000, 1000, ...] (1 unique, rest base_token)
    - Sample 2: [1001, 1002, 1000, ...] (2 unique, rest base_token)
    - Sample i: [1001, ..., 1000+i, 1000, ...] (i unique, rest base_token)

    This creates varying levels of token diversity across samples.

    Args:
        n_samples: Number of sequences to generate
        seq_len: Sequence length (block_size)
        vocab_size: Vocabulary size (unused, kept for compatibility)
        device: Device to create tensors on
        base_token: Token to fill non-prefix positions

    Returns:
        tokens: (n_samples, seq_len) tensor - each row has different prefix length
    """
    # Create all sequences filled with base_token
    sequences = torch.full(
        (n_samples, seq_len), base_token, dtype=torch.long, device=device
    )

    # For each sample i, fill first i positions with unique tokens
    for sample_idx in range(n_samples):
        n_unique = min(sample_idx, seq_len)  # Sample i has i unique tokens
        for pos in range(n_unique):
            sequences[sample_idx, pos] = base_token + 1 + pos

    return sequences


def generate_random_sequences(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate completely random sequences."""
    return torch.randint(0, vocab_size, (n_samples, seq_len), device=device)


def collect_activations(
    model: GPT,
    tokens: torch.Tensor,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Collect activations from given token sequences.

    Args:
        tokens: (n_samples, seq_len) tensor

    Returns:
        activations: Dict[layer_name, (n_samples*seq_len, d_model)]
        positions: (n_samples*seq_len,) position indices
        sample_ids: (n_samples*seq_len,) sample ID (number of unique prefix tokens)
    """
    n_samples, seq_len = tokens.shape

    all_activations = {layer: [] for layer in LAYERS_TO_PLOT}

    with torch.no_grad():
        # Process in small batches
        batch_size = 10
        for i in range(0, n_samples, batch_size):
            batch_tokens = tokens[i : i + batch_size]

            for j in range(len(batch_tokens)):
                acts = get_activations(model, batch_tokens[j : j + 1])

                for layer in LAYERS_TO_PLOT:
                    if layer in acts:
                        all_activations[layer].append(acts[layer][0].cpu().numpy())

    # Stack and reshape: (n_samples, seq_len, d_model) -> (n_samples*seq_len, d_model)
    for layer in LAYERS_TO_PLOT:
        if all_activations[layer]:
            all_activations[layer] = np.array(all_activations[layer])
            all_activations[layer] = all_activations[layer].reshape(
                -1, all_activations[layer].shape[-1]
            )

    # Positions: repeat 0, 1, 2, ..., seq_len-1 for each sample
    positions = np.tile(np.arange(seq_len), n_samples)

    # Sample IDs: for sample i, it has i unique prefix tokens (0, 1, 2, ..., n_samples-1)
    sample_ids = np.repeat(np.arange(n_samples), seq_len)

    return all_activations, positions, sample_ids


def create_comparison_tsne(
    unique_acts: np.ndarray,
    random_acts: np.ndarray,
    positions: np.ndarray,
    sample_ids: np.ndarray,
    title: str,
    n_buckets: int = 8,
    perplexity: int = 30,
) -> plt.Figure:
    """
    Create side-by-side t-SNE comparison: unique prefix vs random.

    Dual encoding: Color = position bucket, Marker shape = prefix diversity group

    Args:
        unique_acts: (N, D) activations from unique prefix sequences
        random_acts: (N, D) activations from random sequences
        positions: (N,) position indices
        sample_ids: (N,) sample ID (number of unique prefix tokens: 0-23)
        title: Plot title
        n_buckets: Number of position buckets for coloring
        perplexity: t-SNE perplexity parameter
    """
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    # Bin positions for coloring
    max_pos = positions.max() + 1
    bucket_size = max_pos / n_buckets
    position_buckets = np.clip((positions / bucket_size).astype(int), 0, n_buckets - 1)

    # Bin sample IDs into prefix diversity groups for marker shapes
    # Groups: 0-5, 6-11, 12-17, 18-23
    prefix_groups = np.clip(sample_ids // 6, 0, 3)
    markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond
    marker_labels = ["0-5 unique", "6-11 unique", "12-17 unique", "18-23 unique"]

    cmap = plt.cm.get_cmap("viridis", n_buckets)

    for idx, (acts, subtitle) in enumerate(
        [
            (unique_acts, "Unique Prefix (varying)"),
            (random_acts, "Random Sequences"),
        ]
    ):
        ax = fig.add_subplot(gs[idx])

        # Run t-SNE with barnes_hut but higher quality settings
        print(f"    Running t-SNE for {subtitle} ({len(acts)} samples)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(acts) // 3),
            random_state=42,
            method="barnes_hut",  # Faster than exact
            angle=0.2,  # Lower angle = higher precision
            n_iter=1000,  # More iterations for better convergence
            verbose=0,
        )
        embeddings = tsne.fit_transform(acts)

        # Plot each combination of position bucket and prefix group
        for prefix_group in range(4):
            for bucket in range(n_buckets):
                mask = (position_buckets == bucket) & (prefix_groups == prefix_group)
                if mask.sum() > 0:
                    start_pos = int(bucket * bucket_size)
                    end_pos = int((bucket + 1) * bucket_size) - 1

                    # Only add label for first occurrence to avoid duplicate legend entries
                    if bucket == 0 and prefix_group == 0:
                        label = marker_labels[prefix_group]
                    elif prefix_group == 0 and bucket < n_buckets:
                        label = f"pos {start_pos}-{end_pos}"
                    elif bucket == 0 and prefix_group > 0:
                        label = marker_labels[prefix_group]
                    else:
                        label = None

                    ax.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        c=[cmap(bucket)],
                        marker=markers[prefix_group],
                        label=label,
                        alpha=0.6,
                        s=20 + prefix_group * 5,
                        edgecolors="black",
                        linewidths=0.3,
                    )

        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.set_title(subtitle, fontsize=12, fontweight="bold")

        # Create two separate legends: one for colors (position), one for markers (diversity)
        handles, labels = ax.get_legend_handles_labels()

        # Position legend (colors)
        pos_handles = [h for h, l in zip(handles, labels) if "pos" in l]
        pos_labels = [l for l in labels if "pos" in l]

        # Diversity legend (markers)
        div_handles = [h for h, l in zip(handles, labels) if "unique" in l]
        div_labels = [l for l in labels if "unique" in l]

        # Add both legends
        if pos_handles:
            leg1 = ax.legend(
                pos_handles,
                pos_labels,
                loc="upper left",
                fontsize=7,
                title="Position",
                ncol=2,
                framealpha=0.9,
            )
            ax.add_artist(leg1)
        if div_handles:
            ax.legend(
                div_handles,
                div_labels,
                loc="upper right",
                fontsize=7,
                title="Prefix Diversity",
                framealpha=0.9,
            )

        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    return fig


def analyze_checkpoints(
    checkpoint_dir: Path,
    checkpoint_steps: List[int],
    n_samples: int = 24,
    use_wandb: bool = True,
    experiment_name: str = "experiment",
):
    """Analyze all checkpoints and create comparison t-SNE visualizations.

    Each sample i has i unique prefix tokens, creating varying prefix diversity.
    """

    # Set up results directories
    results_dir = PROJECT_ROOT / "results" / f"tsne_comparison_{experiment_name}"
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"t-SNE Comparison: Unique Prefix vs Random - {experiment_name}")
    print(f"{'=' * 70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Checkpoints: {checkpoint_steps}")
    print(f"Layers: {LAYERS_TO_PLOT}")
    print(f"Samples: {n_samples} (each with varying prefix diversity)")

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="nope-position-regression-tsne",
            name=f"comparison_{experiment_name}",
            config={
                "n_samples": n_samples,
                "checkpoint_steps": checkpoint_steps,
                "layers": LAYERS_TO_PLOT,
                "experiment": experiment_name,
                "comparison": "unique_vs_random",
                "note": "Each sample i has i unique prefix tokens",
            },
        )
        print(
            f"\nWandB initialized: nope-position-regression-tsne/comparison_{experiment_name}"
        )

    # Process each checkpoint
    for step in tqdm(checkpoint_steps, desc="Analyzing checkpoints"):
        ckpt_path = checkpoint_dir / f"ckpt_{step:05d}.pt"

        if not ckpt_path.exists():
            print(f"\nWarning: Checkpoint {ckpt_path} not found, skipping...")
            continue

        print(f"\n--- Checkpoint {step} ---")

        try:
            # Load model
            model, meta = load_checkpoint(str(ckpt_path), DEVICE)
            vocab_size = model.config.vocab_size
            seq_len = model.config.block_size
            print(f"  Loaded model (n_embd={model.config.n_embd}, seq_len={seq_len})")
            print(
                f"  Total t-SNE points: {n_samples} × {seq_len} = {n_samples * seq_len}"
            )

            # Generate unique prefix sequences (each sample i has i unique tokens)
            print(f"  Generating unique prefix sequences...")
            unique_tokens = generate_unique_prefix_sequences(
                n_samples, seq_len, vocab_size, DEVICE
            )

            # Generate random sequences
            print(f"  Generating random sequences...")
            random_tokens = generate_random_sequences(
                n_samples, seq_len, vocab_size, DEVICE
            )

            # Collect activations
            print(f"  Collecting activations (unique prefix)...")
            unique_acts, positions, sample_ids = collect_activations(
                model, unique_tokens
            )

            print(f"  Collecting activations (random)...")
            random_acts, _, _ = collect_activations(model, random_tokens)

            # Create comparison plots for each layer
            for layer in LAYERS_TO_PLOT:
                if layer not in unique_acts or layer not in random_acts:
                    continue

                print(f"  Creating comparison t-SNE for {layer}...")
                layer_title = layer.replace("_", " ").title()
                fig = create_comparison_tsne(
                    unique_acts[layer],
                    random_acts[layer],
                    positions,
                    sample_ids,
                    f"Step {step} - {layer_title}",
                    n_buckets=8,
                )

                # Save locally
                save_path = plots_dir / f"step{step:05d}_{layer}_comparison.png"
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"    Saved: {save_path.name}")

                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            f"comparison/{layer}": wandb.Image(fig),
                            "checkpoint_step": step,
                        }
                    )

                plt.close(fig)

            # Clean up
            del model, unique_acts, random_acts, unique_tokens, random_tokens
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error processing checkpoint {step}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Finish wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("\nWandB run finished.")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Plots saved to: {plots_dir}")
    print(f"{'=' * 70}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create side-by-side t-SNE: unique prefix vs random sequences"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=24,
        help="Number of samples per checkpoint (each with varying prefix diversity)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=CHECKPOINT_STEPS,
        help="Checkpoint steps to analyze",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory (e.g., out-posreg-6layer-until-mlp)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for wandb",
    )
    args = parser.parse_args()

    # Set up checkpoint directory
    checkpoint_dir = PROJECT_ROOT / "nanoGPT" / args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist!")
        return

    # Set experiment name
    experiment_name = args.experiment_name or args.checkpoint_dir

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Run analysis
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    analyze_checkpoints(
        checkpoint_dir=checkpoint_dir,
        checkpoint_steps=args.steps,
        n_samples=args.n_samples,
        use_wandb=use_wandb,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    main()
