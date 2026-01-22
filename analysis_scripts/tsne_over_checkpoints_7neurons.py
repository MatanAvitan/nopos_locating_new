"""
t-SNE Visualization Over Training Checkpoints for 7-Neuron Model

This script creates t-SNE visualizations of activations at different training
checkpoints for the 7-neuron position regression model. It shows how the
activation structure emerges during training.

Usage:
    python analysis_scripts/tsne_over_checkpoints_7neurons.py --wandb
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
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
LAYERS_TO_PLOT = ["embed", "post_ln1", "post_attn", "pre_ln2", "post_ln2", "post_mlp"]


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

    # Handle state dict with _orig_mod prefix (from torch.compile)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
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
    Extract activations at key points.

    Returns dict with:
    - embed: Token embeddings
    - post_ln1: After first LayerNorm
    - post_attn: Attention output
    - pre_ln2: After first residual (before LN2)
    - post_ln2: After second LayerNorm
    - post_mlp: After MLP (before final residual)
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
        activations["embed"] = x.clone()

        # Get the specified block
        block = model.transformer.h[layer_idx]

        # Post-LN1 (BEFORE attention)
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.clone()

        # Post-Attention output (BEFORE adding residual)
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.clone()

        # After first residual connection (input to LN2)
        x = x + attn_out
        activations["pre_ln2"] = x.clone()

        # Post-LN2 (BEFORE MLP)
        if block.use_ln2:
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.clone()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.clone()
            mlp_input = x

        # Post-MLP output (BEFORE final residual)
        mlp_out = block.mlp(mlp_input)
        activations["post_mlp"] = mlp_out.clone()

    return activations


def generate_unique_prefix_sequences(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate sequences where each position has unique prefixes across samples.

    For position i, each of the n_samples sequences has a unique sequence
    of tokens [0:i] (unique prefix).

    Returns:
        tokens: (n_samples, seq_len) tensor
    """
    sequences = []

    for _ in range(n_samples):
        # Generate a random sequence
        seq = torch.randint(0, vocab_size, (seq_len,), device=device)
        sequences.append(seq)

    # Convert to tensor
    tokens = torch.stack(sequences)  # (n_samples, seq_len)

    # Ensure unique prefixes at each position by shuffling
    # For each position i, we want all prefixes [:i+1] to be unique
    for pos in range(1, seq_len):
        # Get all prefixes up to this position
        prefixes = tokens[:, : pos + 1]

        # Convert to tuples for uniqueness check
        prefix_tuples = [tuple(p.cpu().numpy()) for p in prefixes]

        # If we have duplicates, replace them with new random sequences
        seen = set()
        for idx, prefix in enumerate(prefix_tuples):
            if prefix in seen:
                # Generate a new random token for this position
                new_token = torch.randint(0, vocab_size, (1,), device=device).item()
                tokens[idx, pos] = new_token
                # Update the prefix tuple
                prefix_tuples[idx] = tuple(tokens[idx, : pos + 1].cpu().numpy())
            seen.add(prefix_tuples[idx])

    return tokens


def collect_activations(
    model: GPT,
    n_samples: int = 100,
    seq_len: int = 128,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Collect activations from sequences with unique prefixes.

    For each position, all samples have unique prefix histories.
    """
    vocab_size = model.config.vocab_size

    # Generate sequences with unique prefixes
    print(f"  Generating {n_samples} sequences with unique prefixes...")
    tokens = generate_unique_prefix_sequences(n_samples, seq_len, vocab_size, DEVICE)

    all_activations = {layer: [] for layer in LAYERS_TO_PLOT}

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 10
        for i in range(0, n_samples, batch_size):
            batch_tokens = tokens[i : i + batch_size]

            for j in range(len(batch_tokens)):
                acts = get_activations(model, batch_tokens[j : j + 1])

                for layer in LAYERS_TO_PLOT:
                    if layer in acts:
                        all_activations[layer].append(acts[layer][0].cpu().numpy())

    # Stack all samples: (n_samples, seq_len, d_model)
    for layer in LAYERS_TO_PLOT:
        if all_activations[layer]:
            all_activations[layer] = np.array(all_activations[layer])

    # Create position array: for each sample, positions 0 to seq_len-1
    # But we want to flatten it: (n_samples * seq_len,)
    for layer in LAYERS_TO_PLOT:
        if layer in all_activations and len(all_activations[layer]) > 0:
            # Reshape from (n_samples, seq_len, d_model) to (n_samples * seq_len, d_model)
            all_activations[layer] = all_activations[layer].reshape(
                -1, all_activations[layer].shape[-1]
            )

    # Positions: repeat 0, 1, 2, ..., seq_len-1 for each sample
    positions = np.tile(np.arange(seq_len), n_samples)

    return all_activations, positions


def create_tsne_visualization(
    activations: np.ndarray,
    positions: np.ndarray,
    title: str,
    n_buckets: int = 8,
    n_samples_tsne: int = 1000,
    perplexity: int = 30,
) -> plt.Figure:
    """
    Create t-SNE visualization with position buckets colored.

    Args:
        activations: (N, D) activation vectors
        positions: (N,) position indices
        title: Plot title
        n_buckets: Number of position buckets for coloring
        n_samples_tsne: Max samples for t-SNE (for speed)
        perplexity: t-SNE perplexity parameter
    """
    # Subsample if needed
    if len(positions) > n_samples_tsne:
        idx = np.random.choice(len(positions), n_samples_tsne, replace=False)
        activations = activations[idx]
        positions = positions[idx]

    # Bin positions for coloring
    max_pos = positions.max() + 1
    bucket_size = max_pos / n_buckets
    position_buckets = np.clip((positions / bucket_size).astype(int), 0, n_buckets - 1)

    # Run t-SNE with optimizations
    print(f"    Running t-SNE ({len(positions)} samples)...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_jobs=-1,
        method="barnes_hut",  # Faster than exact
        angle=0.5,  # Trade-off between speed and accuracy
        n_iter=500,  # Reduce from default 1000
        verbose=0,
    )
    embeddings = tsne.fit_transform(activations)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map
    cmap = plt.cm.get_cmap("viridis", n_buckets)

    # Plot each bucket with its color
    for bucket in range(n_buckets):
        mask = position_buckets == bucket
        if mask.sum() > 0:
            start_pos = int(bucket * bucket_size)
            end_pos = int((bucket + 1) * bucket_size) - 1
            label = f"pos {start_pos}-{end_pos}"
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[cmap(bucket)],
                label=label,
                alpha=0.6,
                s=10,
            )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_checkpoints(
    checkpoint_dir: Path,
    checkpoint_steps: List[int],
    n_samples: int = 100,
    use_wandb: bool = True,
    experiment_name: str = "7neurons",
):
    """Analyze all checkpoints and create t-SNE visualizations."""

    # Set up results directories based on experiment name
    results_dir = PROJECT_ROOT / "results" / f"tsne_{experiment_name}"
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"t-SNE Visualization Over Training Checkpoints - {experiment_name}")
    print(f"{'=' * 70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Checkpoints to analyze: {checkpoint_steps}")
    print(f"Layers to plot: {LAYERS_TO_PLOT}")
    print(f"Samples per checkpoint: {n_samples}")

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="nope-position-regression-tsne",
            name=f"tsne_{experiment_name}",
            config={
                "n_samples": n_samples,
                "checkpoint_steps": checkpoint_steps,
                "layers": LAYERS_TO_PLOT,
                "experiment": experiment_name,
            },
        )
        print(
            f"\nWandB initialized. Project: nope-position-regression-tsne, Run: {experiment_name}"
        )

    # Process each checkpoint
    results = {}

    for step in tqdm(checkpoint_steps, desc="Analyzing checkpoints"):
        ckpt_path = checkpoint_dir / f"ckpt_{step:05d}.pt"

        if not ckpt_path.exists():
            print(f"\nWarning: Checkpoint {ckpt_path} not found, skipping...")
            continue

        print(f"\n--- Checkpoint {step} ---")

        try:
            # Load model
            model, meta = load_checkpoint(str(ckpt_path), DEVICE)
            print(f"  Loaded model (n_embd={model.config.n_embd})")

            # Collect activations
            print(f"  Collecting activations...")
            activations, positions = collect_activations(model, n_samples=n_samples)

            # Create t-SNE plots for each layer
            for layer in LAYERS_TO_PLOT:
                if layer not in activations or len(activations[layer]) == 0:
                    continue

                print(f"  Creating t-SNE for {layer}...")
                fig = create_tsne_visualization(
                    activations[layer],
                    positions,
                    f"Step {step} - {layer.replace('_', ' ').title()}",
                    n_buckets=8,
                    n_samples_tsne=min(1000, len(positions)),
                )

                # Save locally
                save_path = plots_dir / f"step{step:05d}_{layer}.png"
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            f"tsne/{layer}": wandb.Image(fig),
                            "checkpoint_step": step,
                        }
                    )

                plt.close(fig)

            # Store metadata
            results[step] = {
                "step": meta["step"],
                "train_loss": meta.get("train_loss"),
            }

            # Clean up
            del model, activations
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error processing checkpoint {step}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Finish wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("\nWandB run finished. Check dashboard for t-SNE animations.")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete!")
    print(f"Plots saved to: {plots_dir}")
    print(f"{'=' * 70}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create t-SNE visualizations over training checkpoints for position regression models"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per checkpoint",
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
        help="Checkpoint directory (e.g., out-posreg-6layer-until-mlp-7dim)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for wandb and results directory (defaults to checkpoint dir name)",
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
