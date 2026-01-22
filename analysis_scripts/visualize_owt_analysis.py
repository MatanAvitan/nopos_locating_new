"""
OWT Analysis Visualization

Generates publication-quality figures for the paper from analysis results:
1. Direction vs Norm comparison heatmap
2. Norm over positions (trained vs random)
3. Linearization effect comparison
4. BatchNorm population statistics
5. Summary comparison

Outputs to: overleaf/nopos---claude-version/plots/
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "owt_comprehensive"
PLOTS_DIR = PROJECT_ROOT / "overleaf" / "nopos---claude-version" / "plots"


def load_results():
    """Load all analysis results."""
    results = {}

    # Comprehensive results
    comp_path = RESULTS_DIR / "owt_comprehensive_results.json"
    if comp_path.exists():
        with open(comp_path) as f:
            results["comprehensive"] = json.load(f)

    # Linearization results
    lin_path = RESULTS_DIR / "owt_linearization_results.json"
    if lin_path.exists():
        with open(lin_path) as f:
            results["linearization"] = json.load(f)

    # BatchNorm results
    bn_path = RESULTS_DIR / "owt_batchnorm_population_results.json"
    if bn_path.exists():
        with open(bn_path) as f:
            results["batchnorm"] = json.load(f)

    return results


def plot_direction_norm_heatmap(results: dict, output_path: Path):
    """
    Create heatmap comparing Direction R² vs Norm R² across experiments and layers.
    """
    if "comprehensive" not in results:
        print("No comprehensive results found, skipping heatmap")
        return

    data = results["comprehensive"]

    experiments = list(data.keys())
    layers = ["post_attn", "post_ln2", "post_mlp_residual"]
    layer_labels = ["Post-Attn", "Post-LN2", "Post-MLP"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    for idx, metric in enumerate(["direction_r2", "norm_r2", "full_r2"]):
        ax = axes[idx]

        # Build matrix
        matrix = np.zeros((len(experiments), len(layers)))
        for i, exp in enumerate(experiments):
            for j, layer in enumerate(layers):
                if "trained" in data[exp] and layer in data[exp]["trained"]:
                    matrix[i, j] = data[exp]["trained"][layer][metric]

        # Create heatmap
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=0.6)

        # Labels
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels([e.replace(" + ", "\n") for e in experiments])

        # Annotate
        for i in range(len(experiments)):
            for j in range(len(layers)):
                text = f"{matrix[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                       color="white" if matrix[i, j] > 0.3 else "black")

        title_map = {"direction_r2": "Direction R²", "norm_r2": "Norm R²", "full_r2": "Full R²"}
        ax.set_title(title_map[metric])

    plt.tight_layout()

    # Add colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(output_path, format="pdf")
    plt.savefig(output_path.with_suffix(".png"))
    plt.close()

    print(f"Saved: {output_path}")


def plot_trained_vs_random_comparison(results: dict, output_path: Path):
    """
    Bar chart comparing trained vs random models for key metrics.
    """
    if "comprehensive" not in results:
        print("No comprehensive results found, skipping comparison")
        return

    data = results["comprehensive"]

    experiments = list(data.keys())
    n_exp = len(experiments)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ("norm_position_corr", "Norm-Position Correlation", "post_ln2"),
        ("direction_r2", "Direction R² (Post-LN2)", "post_ln2"),
        ("norm_r2", "Norm R² (Post-LN2)", "post_ln2"),
    ]

    x = np.arange(n_exp)
    width = 0.35

    for ax, (metric, title, layer) in zip(axes, metrics):
        trained_vals = []
        random_vals = []

        for exp in experiments:
            if "trained" in data[exp] and layer in data[exp]["trained"]:
                trained_vals.append(data[exp]["trained"][layer][metric])
            else:
                trained_vals.append(0)

            if "random" in data[exp] and layer in data[exp]["random"]:
                random_vals.append(data[exp]["random"][layer][metric])
            else:
                random_vals.append(0)

        bars1 = ax.bar(x - width/2, trained_vals, width, label="Trained", color="#1f77b4")
        bars2 = ax.bar(x + width/2, random_vals, width, label="Random", color="#ff7f0e")

        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=8)
        ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.savefig(output_path.with_suffix(".png"))
    plt.close()

    print(f"Saved: {output_path}")


def plot_linearization_effect(results: dict, output_path: Path):
    """
    Visualize linearization effect: how LN transforms direction→norm encoding.
    """
    if "linearization" not in results:
        print("No linearization results found, skipping")
        return

    data = results["linearization"]

    experiments = list(data.keys())
    n_exp = len(experiments)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Before vs After LN2
    ax1 = axes[0]
    x = np.arange(n_exp)
    width = 0.2

    for i, exp in enumerate(experiments):
        if "trained" in data[exp]:
            t = data[exp]["trained"]
            ax1.bar(i - width*1.5, t["pre_ln2"]["direction_r2"], width,
                   label="Pre-LN2 Dir" if i == 0 else "", color="#2ca02c", alpha=0.8)
            ax1.bar(i - width*0.5, t["pre_ln2"]["norm_r2"], width,
                   label="Pre-LN2 Norm" if i == 0 else "", color="#9467bd", alpha=0.8)
            ax1.bar(i + width*0.5, t["post_ln2"]["direction_r2"], width,
                   label="Post-LN2 Dir" if i == 0 else "", color="#2ca02c")
            ax1.bar(i + width*1.5, t["post_ln2"]["norm_r2"], width,
                   label="Post-LN2 Norm" if i == 0 else "", color="#9467bd")

    ax1.set_xticks(x)
    ax1.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=8)
    ax1.set_ylabel("R²")
    ax1.set_title("Direction vs Norm R² (Before/After LN2) - Trained")
    ax1.legend(loc="upper right", fontsize=7)

    # Plot 2: Linearization effect comparison
    ax2 = axes[1]

    trained_effects = []
    random_effects = []

    for exp in experiments:
        if "trained" in data[exp]:
            trained_effects.append(data[exp]["trained"]["linearization"]["norm_r2_change"])
        else:
            trained_effects.append(0)
        if "random" in data[exp]:
            random_effects.append(data[exp]["random"]["linearization"]["norm_r2_change"])
        else:
            random_effects.append(0)

    bars1 = ax2.bar(x - 0.2, trained_effects, 0.35, label="Trained", color="#1f77b4")
    bars2 = ax2.bar(x + 0.2, random_effects, 0.35, label="Random", color="#ff7f0e")

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=8)
    ax2.set_ylabel("Norm R² Change (Post - Pre LN2)")
    ax2.set_title("Linearization Effect")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.savefig(output_path.with_suffix(".png"))
    plt.close()

    print(f"Saved: {output_path}")


def plot_batchnorm_population(results: dict, output_path: Path):
    """
    Visualize BatchNorm vs LayerNorm population statistics analysis.
    """
    if "batchnorm" not in results:
        print("No batchnorm results found, skipping")
        return

    data = results["batchnorm"]

    experiments = list(data.keys())
    n_exp = len(experiments)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Probe comparison
    ax1 = axes[0]
    x = np.arange(n_exp)
    width = 0.25

    baseline_vals = []
    pop_mean_vals = []
    residual_vals = []

    for exp in experiments:
        baseline_vals.append(data[exp]["baseline_probe"]["r2"])
        pop_mean_vals.append(data[exp]["population_mean_probe"]["r2"])
        residual_vals.append(data[exp]["residual_probe"]["r2"])

    bars1 = ax1.bar(x - width, baseline_vals, width, label="Baseline", color="#1f77b4")
    bars2 = ax1.bar(x, pop_mean_vals, width, label="Population Mean", color="#2ca02c")
    bars3 = ax1.bar(x + width, residual_vals, width, label="Residual (h-μ)", color="#d62728")

    ax1.set_xticks(x)
    ax1.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=9)
    ax1.set_ylabel("R²")
    ax1.set_title("Population Statistics Probing")
    ax1.legend()

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    # Plot 2: Mode difference (inference vs training)
    ax2 = axes[1]

    baseline_diffs = []
    pop_mean_diffs = []

    for exp in experiments:
        if "mode_difference" in data[exp]:
            baseline_diffs.append(data[exp]["mode_difference"]["baseline_r2_diff"])
            pop_mean_diffs.append(data[exp]["mode_difference"]["pop_mean_r2_diff"])
        else:
            baseline_diffs.append(0)
            pop_mean_diffs.append(0)

    bars1 = ax2.bar(x - 0.2, baseline_diffs, 0.35, label="Baseline", color="#1f77b4")
    bars2 = ax2.bar(x + 0.2, pop_mean_diffs, 0.35, label="Population Mean", color="#2ca02c")

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=9)
    ax2.set_ylabel("R² Difference (Training - Inference)")
    ax2.set_title("Mode Difference (Train vs Eval)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.savefig(output_path.with_suffix(".png"))
    plt.close()

    print(f"Saved: {output_path}")


def plot_summary_figure(results: dict, output_path: Path):
    """
    Create comprehensive summary figure.
    """
    if "comprehensive" not in results:
        print("No comprehensive results found, skipping summary")
        return

    data = results["comprehensive"]

    experiments = list(data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Direction vs Norm dominance (post-LN2)
    ax1 = axes[0, 0]
    dir_vals = []
    norm_vals = []
    for exp in experiments:
        if "trained" in data[exp] and "post_ln2" in data[exp]["trained"]:
            dir_vals.append(data[exp]["trained"]["post_ln2"]["direction_r2"])
            norm_vals.append(data[exp]["trained"]["post_ln2"]["norm_r2"])
        else:
            dir_vals.append(0)
            norm_vals.append(0)

    x = np.arange(len(experiments))
    ax1.bar(x - 0.2, dir_vals, 0.35, label="Direction R²", color="#2ca02c")
    ax1.bar(x + 0.2, norm_vals, 0.35, label="Norm R²", color="#9467bd")
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=8)
    ax1.set_ylabel("R²")
    ax1.set_title("Direction vs Norm (Post-LN2, Trained)")
    ax1.legend()

    # 2. Attention uniformity
    ax2 = axes[0, 1]
    trained_unif = []
    random_unif = []
    for exp in experiments:
        if "trained" in data[exp]:
            trained_unif.append(data[exp]["trained"].get("attention_uniformity", {}).get("mean_uniformity", 0))
        else:
            trained_unif.append(0)
        if "random" in data[exp]:
            random_unif.append(data[exp]["random"].get("attention_uniformity", {}).get("mean_uniformity", 0))
        else:
            random_unif.append(0)

    ax2.bar(x - 0.2, trained_unif, 0.35, label="Trained", color="#1f77b4")
    ax2.bar(x + 0.2, random_unif, 0.35, label="Random", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=8)
    ax2.set_ylabel("Uniformity Score")
    ax2.set_title("Attention Uniformity")
    ax2.legend()

    # 3. Norm-Position Correlation
    ax3 = axes[1, 0]
    trained_corr = []
    random_corr = []
    for exp in experiments:
        if "trained" in data[exp] and "post_ln2" in data[exp]["trained"]:
            trained_corr.append(data[exp]["trained"]["post_ln2"]["norm_position_corr"])
        else:
            trained_corr.append(0)
        if "random" in data[exp] and "post_ln2" in data[exp]["random"]:
            random_corr.append(data[exp]["random"]["post_ln2"]["norm_position_corr"])
        else:
            random_corr.append(0)

    ax3.bar(x - 0.2, trained_corr, 0.35, label="Trained", color="#1f77b4")
    ax3.bar(x + 0.2, random_corr, 0.35, label="Random", color="#ff7f0e")
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([e.replace(" + ", "\n") for e in experiments], fontsize=8)
    ax3.set_ylabel("Pearson r")
    ax3.set_title("Norm-Position Correlation (Post-LN2)")
    ax3.legend()

    # 4. Full R² across layers (trained models)
    ax4 = axes[1, 1]
    layers = ["post_attn", "post_ln2", "post_mlp_residual"]
    layer_labels = ["Post-Attn", "Post-LN2", "Post-MLP"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for i, exp in enumerate(experiments):
        full_r2 = []
        for layer in layers:
            if "trained" in data[exp] and layer in data[exp]["trained"]:
                full_r2.append(data[exp]["trained"][layer]["full_r2"])
            else:
                full_r2.append(0)
        ax4.plot(range(len(layers)), full_r2, 'o-', label=exp, color=colors[i])

    ax4.set_xticks(range(len(layers)))
    ax4.set_xticklabels(layer_labels)
    ax4.set_ylabel("Full R²")
    ax4.set_title("Position Decodability Across Layers")
    ax4.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.savefig(output_path.with_suffix(".png"))
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("OWT ANALYSIS VISUALIZATION")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results()

    if not results:
        print("No results found. Please run analysis scripts first.")
        return

    print(f"\nLoaded results: {list(results.keys())}")

    # Generate figures
    plot_direction_norm_heatmap(
        results,
        PLOTS_DIR / "owt_direction_norm_comparison.pdf"
    )

    plot_trained_vs_random_comparison(
        results,
        PLOTS_DIR / "owt_trained_vs_random_summary.pdf"
    )

    plot_linearization_effect(
        results,
        PLOTS_DIR / "owt_linearization_effect.pdf"
    )

    plot_batchnorm_population(
        results,
        PLOTS_DIR / "owt_batchnorm_population.pdf"
    )

    plot_summary_figure(
        results,
        PLOTS_DIR / "owt_comprehensive_summary.pdf"
    )

    print(f"\n{'=' * 70}")
    print(f"All figures saved to: {PLOTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
