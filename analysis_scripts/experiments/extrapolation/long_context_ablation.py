"""
Long Context Ablation Study: n_embd × n_heads × vocab_size × context_length × dataset

This script performs a comprehensive ablation study of the post-LN position encoding
hypothesis across different model configurations and context lengths.

Key dimensions:
- Context lengths: 64, 128, 256, 512, 1024, 2048, 4096, 8192
- n_embd: 256, 512, 768, 1024, 2048, 4096
- n_heads: 1, 2, 4, 8, 12, 16 (where n_embd % n_heads == 0)
- vocab_size: 1000, 10000, 50000 (to test embedding diversity)
- Datasets: synthetic (uniform random), natural language (Shakespeare)

Metrics reported:
- Norm R² (post-LN2)
- Direction R² (post-LN2)
- Full R² (post-LN2)

Usage:
    # Run single configuration
    python long_context_ablation.py --n_ctx 8192 --n_embd 768 --n_heads 12 --vocab_size 10000 --dataset synthetic

    # Run with job index (for Slurm array jobs)
    python long_context_ablation.py --job_index 0

    # Run all experiments sequentially
    python long_context_ablation.py --run_all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig
from sklearn.linear_model import Ridge

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "long_context_ablation"

# Ablation dimensions
CONTEXT_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_EMBDS = [256, 512, 768, 1024, 2048, 4096]
N_HEADS = [1, 2, 4, 8, 12, 16]
VOCAB_SIZES = [1000, 10000, 50000]
DATASETS = ["synthetic", "shakespeare"]


def is_valid_config(n_embd: int, n_heads: int) -> bool:
    """Check if n_embd is divisible by n_heads."""
    return n_embd % n_heads == 0


def get_all_configs():
    """Generate all valid configurations for the ablation study."""
    configs = []
    for dataset in DATASETS:
        for n_ctx in CONTEXT_LENGTHS:
            for n_embd in N_EMBDS:
                for n_heads in N_HEADS:
                    if is_valid_config(n_embd, n_heads):
                        for vocab_size in VOCAB_SIZES:
                            configs.append(
                                {
                                    "dataset": dataset,
                                    "n_ctx": n_ctx,
                                    "n_embd": n_embd,
                                    "n_heads": n_heads,
                                    "vocab_size": vocab_size,
                                }
                            )
    return configs


def get_n_samples(n_ctx: int, n_embd: int) -> int:
    """Determine number of samples based on memory constraints."""
    if n_ctx >= 4096 and n_embd >= 2048:
        return 30
    elif n_ctx >= 4096 or n_embd >= 4096:
        return 50
    elif n_ctx >= 2048 or n_embd >= 2048:
        return 100
    elif n_ctx >= 1024:
        return 150
    return 200


def create_model(n_ctx: int, n_embd: int, n_heads: int, vocab_size: int):
    """Create a randomly initialized NoPE model."""
    config = GPTConfig(
        n_layer=1,
        n_head=n_heads,
        n_embd=n_embd,
        block_size=n_ctx,
        vocab_size=vocab_size,
        dropout=0.0,
        use_positional_embedding=False,
        norm_type="layernorm",
    )
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model, config


def get_post_ln2_activations(model, tokens):
    """Get post-LN2 activations."""
    with torch.no_grad():
        tok_emb = model.transformer.wte(tokens)
        block = model.transformer.h[0]

        # Through first LN and attention
        x = block.ln_1(tok_emb)
        attn_out = block.attn(x)

        # Residual and second LN
        x = tok_emb + attn_out
        x_ln2 = block.ln_2(x)

    return x_ln2


def load_shakespeare_data(n_ctx: int, n_samples: int, vocab_size: int):
    """Load Shakespeare data for natural language experiments."""
    data_path = (
        Path(__file__).parent.parent.parent.parent / "nanoGPT" / "data" / "shakespeare" / "train.bin"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Shakespeare data not found at {data_path}")

    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Sample sequences
    tokens_list = []
    for _ in range(n_samples):
        start_idx = np.random.randint(0, max(1, len(data) - n_ctx))
        seq = data[start_idx : start_idx + n_ctx].astype(np.int64)
        # Clamp to vocab_size if needed
        seq = np.clip(seq, 0, vocab_size - 1)
        tokens_list.append(torch.tensor(seq, dtype=torch.long, device=DEVICE))

    return torch.stack(tokens_list)


def generate_synthetic_data(n_ctx: int, n_samples: int, vocab_size: int):
    """Generate synthetic uniform random tokens."""
    return torch.randint(0, vocab_size, (n_samples, n_ctx), device=DEVICE)


def compute_r2_metrics(activations: np.ndarray, positions: np.ndarray):
    """
    Compute R² metrics for norm, direction, and full activations.

    Args:
        activations: (n_samples * n_ctx, n_embd) array
        positions: (n_samples * n_ctx,) array of position indices

    Returns:
        dict with norm_r2, direction_r2, full_r2
    """
    # Compute norms and directions
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    directions = activations / (norms + 1e-8)
    norms = norms.squeeze()

    # Train/test split
    n_train = int(0.8 * len(positions))
    idx = np.random.permutation(len(positions))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def fit_probe(X_train, y_train, X_test, y_test):
        """Fit ridge regression probe and return R²."""
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        return max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Compute R² for each representation
    norm_r2 = fit_probe(
        norms[train_idx], positions[train_idx], norms[test_idx], positions[test_idx]
    )

    direction_r2 = fit_probe(
        directions[train_idx],
        positions[train_idx],
        directions[test_idx],
        positions[test_idx],
    )

    full_r2 = fit_probe(
        activations[train_idx],
        positions[train_idx],
        activations[test_idx],
        positions[test_idx],
    )

    # Also compute norm-position correlation
    norm_corr = np.corrcoef(norms, positions)[0, 1]

    return {
        "norm_r2": float(norm_r2),
        "direction_r2": float(direction_r2),
        "full_r2": float(full_r2),
        "norm_position_corr": float(norm_corr),
    }


def run_single_experiment(
    dataset: str,
    n_ctx: int,
    n_embd: int,
    n_heads: int,
    vocab_size: int,
) -> dict:
    """
    Run a single ablation experiment.

    Returns:
        dict with configuration and metrics
    """
    print(f"\n{'=' * 60}")
    print(
        f"Config: dataset={dataset}, n_ctx={n_ctx}, n_embd={n_embd}, n_heads={n_heads}, vocab={vocab_size}"
    )
    print(f"{'=' * 60}")

    # Determine sample count based on memory constraints
    n_samples = get_n_samples(n_ctx, n_embd)
    print(f"Using n_samples={n_samples}")

    # Create model
    try:
        model, config = create_model(n_ctx, n_embd, n_heads, vocab_size)
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

    # Generate/load data
    try:
        if dataset == "synthetic":
            tokens = generate_synthetic_data(n_ctx, n_samples, vocab_size)
        else:  # shakespeare
            tokens = load_shakespeare_data(n_ctx, n_samples, vocab_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Get activations
    try:
        all_activations = []
        all_positions = []

        # Process in batches to manage memory
        batch_size = min(32, n_samples)
        for i in range(0, n_samples, batch_size):
            batch_tokens = tokens[i : i + batch_size]
            activations = get_post_ln2_activations(model, batch_tokens)

            # Flatten batch and sequence dimensions
            batch_act = activations.cpu().numpy().reshape(-1, n_embd)
            batch_pos = np.tile(np.arange(n_ctx), len(batch_tokens))

            all_activations.append(batch_act)
            all_positions.append(batch_pos)

        activations = np.vstack(all_activations)
        positions = np.concatenate(all_positions)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM error, skipping config")
            torch.cuda.empty_cache()
            return None
        raise

    # Compute metrics
    metrics = compute_r2_metrics(activations, positions)

    print(
        f"Results: norm_r2={metrics['norm_r2']:.4f}, dir_r2={metrics['direction_r2']:.4f}, full_r2={metrics['full_r2']:.4f}"
    )

    # Clean up
    del model, tokens, activations
    torch.cuda.empty_cache()

    return {
        "dataset": dataset,
        "n_ctx": n_ctx,
        "n_embd": n_embd,
        "n_heads": n_heads,
        "vocab_size": vocab_size,
        "n_samples": n_samples,
        **metrics,
    }


def save_result(result: dict, results_dir: Path):
    """Save a single result to a JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from config
    filename = f"result_{result['dataset']}_ctx{result['n_ctx']}_emb{result['n_embd']}_heads{result['n_heads']}_vocab{result['vocab_size']}.json"
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {filepath}")


def collect_all_results(results_dir: Path) -> list:
    """Collect all individual result files into a single list."""
    results = []
    for filepath in results_dir.glob("result_*.json"):
        with open(filepath) as f:
            results.append(json.load(f))
    return results


def generate_latex_table(results: list, output_path: Path):
    """Generate LaTeX table from results."""

    # Sort results
    results = sorted(
        results,
        key=lambda x: (
            x["dataset"],
            x["n_ctx"],
            x["n_embd"],
            x["n_heads"],
            x["vocab_size"],
        ),
    )

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{n\_ctx} & \textbf{n\_embd} & \textbf{n\_heads} & \textbf{vocab} & \textbf{Norm R²} & \textbf{Dir R²} & \textbf{Full R²} & \textbf{Norm Corr} \\",
        r"\midrule",
    ]

    current_dataset = None
    for r in results:
        if r["dataset"] != current_dataset:
            if current_dataset is not None:
                lines.append(r"\midrule")
            current_dataset = r["dataset"]

        lines.append(
            f"{r['dataset']} & {r['n_ctx']} & {r['n_embd']} & {r['n_heads']} & {r['vocab_size']} & "
            f"{r['norm_r2']:.3f} & {r['direction_r2']:.3f} & {r['full_r2']:.3f} & {r['norm_position_corr']:.3f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Long context ablation study: Position encoding metrics across model configurations.}",
            r"\label{tab:long-context-ablation}",
            r"\end{table*}",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to: {output_path}")


def generate_summary_tables(results: list, output_dir: Path):
    """Generate condensed summary tables by different axes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dataset
    for dataset in DATASETS:
        dataset_results = [r for r in results if r["dataset"] == dataset]
        if not dataset_results:
            continue

        # Create pivot table: n_ctx vs n_embd (averaged over n_heads and vocab)
        print(f"\n=== {dataset.upper()} Dataset Summary ===")
        print(
            "\nNorm R² by Context Length and Embedding Dimension (averaged over heads/vocab):"
        )
        print(f"{'n_ctx':>8}", end="")
        for n_embd in N_EMBDS:
            print(f" | {n_embd:>6}", end="")
        print()
        print("-" * (8 + len(N_EMBDS) * 9))

        for n_ctx in CONTEXT_LENGTHS:
            print(f"{n_ctx:>8}", end="")
            for n_embd in N_EMBDS:
                matching = [
                    r
                    for r in dataset_results
                    if r["n_ctx"] == n_ctx and r["n_embd"] == n_embd
                ]
                if matching:
                    avg_norm_r2 = np.mean([r["norm_r2"] for r in matching])
                    print(f" | {avg_norm_r2:>6.3f}", end="")
                else:
                    print(f" |    N/A", end="")
            print()


def main():
    parser = argparse.ArgumentParser(description="Long context ablation study")
    parser.add_argument("--job_index", type=int, help="Job array index (0-based)")
    parser.add_argument("--n_ctx", type=int, help="Context length")
    parser.add_argument("--n_embd", type=int, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--dataset", choices=["synthetic", "shakespeare"], default="synthetic"
    )
    parser.add_argument(
        "--run_all", action="store_true", help="Run all configurations sequentially"
    )
    parser.add_argument(
        "--collect", action="store_true", help="Collect results and generate tables"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Mode 1: Collect results and generate tables
    if args.collect:
        print("Collecting results...")
        results = collect_all_results(RESULTS_DIR)
        print(f"Found {len(results)} result files")

        if results:
            # Save combined results
            with open(RESULTS_DIR / "all_results.json", "w") as f:
                json.dump(results, f, indent=2)

            # Generate tables
            generate_latex_table(results, RESULTS_DIR / "ablation_table.tex")
            generate_summary_tables(results, RESULTS_DIR)
        return

    # Mode 2: Run by job index (for Slurm array jobs)
    if args.job_index is not None:
        configs = get_all_configs()

        if args.job_index >= len(configs):
            print(
                f"Job index {args.job_index} >= total configs {len(configs)}, nothing to do"
            )
            return

        config = configs[args.job_index]
        print(f"Running job {args.job_index}/{len(configs)}")

        result = run_single_experiment(**config)
        if result:
            save_result(result, RESULTS_DIR)
        return

    # Mode 3: Run all configurations
    if args.run_all:
        configs = get_all_configs()
        print(f"Running all {len(configs)} configurations...")

        for i, config in enumerate(configs):
            print(f"\n[{i + 1}/{len(configs)}]")
            result = run_single_experiment(**config)
            if result:
                save_result(result, RESULTS_DIR)

        # Collect and generate tables
        results = collect_all_results(RESULTS_DIR)
        if results:
            with open(RESULTS_DIR / "all_results.json", "w") as f:
                json.dump(results, f, indent=2)
            generate_latex_table(results, RESULTS_DIR / "ablation_table.tex")
            generate_summary_tables(results, RESULTS_DIR)
        return

    # Mode 4: Run single configuration from CLI args
    if args.n_ctx and args.n_embd and args.n_heads:
        if not is_valid_config(args.n_embd, args.n_heads):
            print(
                f"Invalid config: n_embd={args.n_embd} not divisible by n_heads={args.n_heads}"
            )
            return

        result = run_single_experiment(
            dataset=args.dataset,
            n_ctx=args.n_ctx,
            n_embd=args.n_embd,
            n_heads=args.n_heads,
            vocab_size=args.vocab_size,
        )
        if result:
            save_result(result, RESULTS_DIR)
        return

    # Print help
    print("Usage:")
    print(
        "  Run single config:  python long_context_ablation.py --n_ctx 8192 --n_embd 768 --n_heads 12 --dataset synthetic"
    )
    print("  Run by job index:   python long_context_ablation.py --job_index 0")
    print("  Run all:            python long_context_ablation.py --run_all")
    print("  Collect results:    python long_context_ablation.py --collect")
    print(f"\nTotal configurations: {len(get_all_configs())}")


if __name__ == "__main__":
    main()
