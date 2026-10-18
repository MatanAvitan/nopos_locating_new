"""
Generate t-SNE visualizations of post-LN embeddings across training checkpoints.

Creates t-SNE plots colored by number of unique prefix tokens, uploaded to wandb
with the same title for slideshow view.
"""

import os
import sys
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_position_classifier import GPTPositionClassifier, GPTPositionClassifierConfig


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    config = GPTPositionClassifierConfig(**model_args)
    model = GPTPositionClassifier(config)

    state_dict = checkpoint['model']
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            unwrapped_state_dict[k[len('_orig_mod.'):]] = v
        else:
            unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model = model.to(device)
    model.eval()

    iter_num = checkpoint.get('iter_num', 0)
    return model, config, iter_num


def create_random_model(config, device='cuda'):
    """Create a fresh randomly initialized model."""
    model = GPTPositionClassifier(config)
    model = model.to(device)
    model.eval()
    return model


def create_prefix_diversity_sequences(n_prefix_unique, block_size, base_token=1000, n_sequences=20):
    """Create sequences with specified prefix diversity."""
    sequences = torch.full((n_sequences, block_size), base_token, dtype=torch.long)
    if n_prefix_unique > 0:
        for i in range(min(n_prefix_unique, block_size)):
            sequences[:, i] = base_token + 1 + i
    return sequences


def extract_post_ln_representations(model, sequences, device='cuda'):
    """Extract post-final-LayerNorm representations."""
    model.eval()
    with torch.no_grad():
        tok_emb = model.transformer.wte(sequences)
        if model.config.use_positional_embedding and hasattr(model.transformer, 'wpe'):
            t = sequences.size(1)
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = model.transformer.wpe(pos)
            x = model.transformer.drop(tok_emb + pos_emb)
        else:
            x = model.transformer.drop(tok_emb)

        for block in model.transformer.h:
            x = block(x)
        x = model.transformer.ln_f(x)
    return x


def get_checkpoint_files(checkpoint_dir):
    """Get all checkpoint files sorted by iteration number."""
    files = os.listdir(checkpoint_dir)
    checkpoints = []
    for f in files:
        if f.startswith('ckpt_') and f.endswith('.pt'):
            match = re.match(r'ckpt_(\d+)\.pt', f)
            if match:
                iter_num = int(match.group(1))
                checkpoints.append((iter_num, os.path.join(checkpoint_dir, f)))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def create_tsne_plot(model, config, device, iter_num, n_sequences_per_prefix=20):
    """Create t-SNE plot of post-LN embeddings colored by prefix diversity."""

    prefix_lengths = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    prefix_lengths = [p for p in prefix_lengths if p <= config.block_size]

    all_embeddings = []
    all_labels = []  # prefix diversity labels
    all_positions = []

    for n_prefix in prefix_lengths:
        sequences = create_prefix_diversity_sequences(
            n_prefix_unique=n_prefix,
            block_size=config.block_size,
            base_token=1000,
            n_sequences=n_sequences_per_prefix
        ).to(device)

        reps = extract_post_ln_representations(model, sequences, device)
        reps = reps.cpu().numpy()  # (n_seq, block_size, n_embd)

        # Flatten: take all positions from all sequences
        n_seq, seq_len, n_embd = reps.shape
        reps_flat = reps.reshape(-1, n_embd)  # (n_seq * seq_len, n_embd)

        all_embeddings.append(reps_flat)
        all_labels.extend([n_prefix] * (n_seq * seq_len))
        all_positions.extend(list(range(seq_len)) * n_seq)

    # Concatenate all embeddings
    X = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    positions = np.array(all_positions)

    # Subsample if too many points (for faster t-SNE)
    max_points = 5000
    if len(X) > max_points:
        indices = np.random.choice(len(X), max_points, replace=False)
        X = X[indices]
        labels = labels[indices]
        positions = positions[indices]

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color map for prefix lengths
    unique_prefixes = sorted(set(labels))
    cmap = plt.cm.viridis
    colors = {p: cmap(i / (len(unique_prefixes) - 1)) for i, p in enumerate(unique_prefixes)}

    # Plot each prefix group
    for n_prefix in unique_prefixes:
        mask = labels == n_prefix
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=[colors[n_prefix]],
                  label=f'{n_prefix} unique',
                  alpha=0.6, s=15)

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f'Post-LN Embeddings t-SNE (iter {iter_num})', fontsize=14)
    ax.legend(title='Prefix tokens', loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='position-regression')
    parser.add_argument('--sample_every', type=int, default=1)
    parser.add_argument('--n_sequences', type=int, default=20)
    args = parser.parse_args()

    checkpoints = get_checkpoint_files(args.checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    if args.sample_every > 1:
        checkpoints = checkpoints[::args.sample_every]
        print(f"Sampling every {args.sample_every}th checkpoint: {len(checkpoints)} remaining")

    # Initialize wandb
    if args.wandb_run_id:
        wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume='must')
    else:
        wandb.init(project=args.wandb_project, name='tsne-training-dynamics')

    # First, do random initialization (iter 0)
    print("[0] Creating t-SNE for random initialization...")
    _, config, _ = load_checkpoint(checkpoints[0][1], args.device)
    random_model = create_random_model(config, args.device)

    fig = create_tsne_plot(random_model, config, args.device, iter_num=0,
                           n_sequences_per_prefix=args.n_sequences)
    wandb.log({"tsne/post_ln_embeddings": wandb.Image(fig)})
    plt.close(fig)

    del random_model
    torch.cuda.empty_cache()

    # Process each checkpoint
    for i, (iter_num, ckpt_path) in enumerate(checkpoints):
        print(f"[{i+1}/{len(checkpoints)}] Creating t-SNE for iteration {iter_num}...")

        model, config, _ = load_checkpoint(ckpt_path, args.device)
        fig = create_tsne_plot(model, config, args.device, iter_num=iter_num,
                               n_sequences_per_prefix=args.n_sequences)

        # Log with same key for slideshow view
        wandb.log({"tsne/post_ln_embeddings": wandb.Image(fig)})
        plt.close(fig)

        del model
        torch.cuda.empty_cache()

    print("\nAll t-SNE plots logged to wandb!")
    wandb.finish()


if __name__ == '__main__':
    main()
