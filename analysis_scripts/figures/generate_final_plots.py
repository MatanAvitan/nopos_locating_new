"""
Generate final plots from saved results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "linear_probe_post_ln"
PLOTS_DIR = RESULTS_DIR / "plots"

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def create_summary_bar_plot(all_results, use_wandb=False):
    """Create summary bar plot comparing all models and layers."""
    layers_to_plot = ["post_ln1", "post_ln2", "post_attn", "post_mlp"]
    layer_labels = ["Post-LN1", "Post-LN2", "Post-Attn", "Post-MLP"]

    fig, axes = plt.subplots(
        1, len(layers_to_plot), figsize=(4 * len(layers_to_plot), 5)
    )

    bar_width = 0.35
    colors = {"random": "#3498db", "trained": "#e74c3c"}

    model_names = list(all_results.keys())
    x = np.arange(len(model_names))

    for idx, (layer, layer_label) in enumerate(zip(layers_to_plot, layer_labels)):
        ax = axes[idx]

        random_accs = []
        trained_accs = []

        for model_name in model_names:
            result = all_results[model_name]
            random_acc = result.get("random", {}).get(layer, {}).get("full_accuracy", 0)
            trained_acc = (
                result.get("trained", {}).get(layer, {}).get("full_accuracy", 0)
            )
            random_accs.append(random_acc)
            trained_accs.append(trained_acc)

        ax.bar(
            x - bar_width / 2,
            random_accs,
            bar_width,
            label="Random",
            color=colors["random"],
            alpha=0.8,
        )
        ax.bar(
            x + bar_width / 2,
            trained_accs,
            bar_width,
            label="Trained",
            color=colors["trained"],
            alpha=0.8,
        )

        ax.axhline(
            y=1 / 32, color="gray", linestyle="--", alpha=0.7, label="Chance (3.1%)"
        )
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(layer_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        short_names = ["LN" if "LayerNorm" in m else "RMS" for m in model_names]
        ax.set_xticklabels(short_names, rotation=0, ha="center", fontsize=10)
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3, axis="y")

        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Linear Probe Accuracy on Post-LN Activations (Before Residual)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = PLOTS_DIR / "summary_linear_probe_accuracy.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # Also save PDF
    fig.savefig(
        PLOTS_DIR / "summary_linear_probe_accuracy.pdf", dpi=300, bbox_inches="tight"
    )

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary/linear_probe_accuracy": wandb.Image(fig)})

    plt.close(fig)
    return fig


def create_probing_line_plot(all_results, use_wandb=False):
    """Create line plot showing accuracy across layers."""
    layers_order = ["embed", "post_ln1", "post_attn", "pre_ln2", "post_ln2", "post_mlp"]
    layer_labels = ["Embed", "Post-LN1", "Post-Attn", "Pre-LN2", "Post-LN2", "Post-MLP"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "NoPE + LayerNorm": ("#1f77b4", "o"),
        "NoPE + RMSNorm": ("#ff7f0e", "s"),
    }

    for ax_idx, init_type in enumerate(["random", "trained"]):
        ax = axes[ax_idx]

        for model_name in all_results.keys():
            result = all_results[model_name].get(init_type, {})
            if not result:
                continue

            color, marker = colors.get(model_name, ("#2ca02c", "d"))

            full_accs = []
            dir_accs = []
            norm_accs = []
            for layer in layers_order:
                layer_data = result.get(layer, {})
                full_accs.append(layer_data.get("full_accuracy", 0))
                dir_accs.append(layer_data.get("direction_accuracy", 0))
                norm_accs.append(layer_data.get("norm_accuracy", 0))

            short_name = "LN" if "LayerNorm" in model_name else "RMS"
            ax.plot(
                range(len(layers_order)),
                full_accs,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=8,
                label=f"{short_name} (Full)",
            )
            ax.plot(
                range(len(layers_order)),
                dir_accs,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=8,
                linestyle="--",
                alpha=0.6,
                label=f"{short_name} (Dir)",
            )

        ax.axhline(y=1 / 32, color="gray", linestyle=":", alpha=0.7, label="Chance")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(
            f"{init_type.capitalize()} Initialization", fontsize=13, fontweight="bold"
        )
        ax.set_xticks(range(len(layers_order)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(-0.02, 0.45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Linear Probe Position Classification: LayerNorm vs RMSNorm",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = PLOTS_DIR / "probing_line_plot.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # Also save PDF
    fig.savefig(PLOTS_DIR / "probing_line_plot.pdf", dpi=300, bbox_inches="tight")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary/probing_line_plot": wandb.Image(fig)})

    plt.close(fig)
    return fig


def create_norm_direction_comparison(all_results):
    """Create comparison of norm vs direction accuracy."""
    layers_order = ["embed", "post_ln1", "post_attn", "pre_ln2", "post_ln2", "post_mlp"]
    layer_labels = ["Embed", "Post-LN1", "Post-Attn", "Pre-LN2", "Post-LN2", "Post-MLP"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, model_name in enumerate(all_results.keys()):
        for col, init_type in enumerate(["random", "trained"]):
            ax = axes[row, col]
            result = all_results[model_name].get(init_type, {})

            dir_accs = [
                result.get(l, {}).get("direction_accuracy", 0) for l in layers_order
            ]
            norm_accs = [
                result.get(l, {}).get("norm_accuracy", 0) for l in layers_order
            ]

            x = np.arange(len(layers_order))
            width = 0.35

            ax.bar(
                x - width / 2,
                dir_accs,
                width,
                label="Direction",
                color="#3498db",
                alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                norm_accs,
                width,
                label="Norm",
                color="#e74c3c",
                alpha=0.8,
            )

            ax.axhline(y=1 / 32, color="gray", linestyle="--", alpha=0.7)
            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)

            short_name = "LayerNorm" if "LayerNorm" in model_name else "RMSNorm"
            ax.set_title(
                f"{short_name} ({init_type.capitalize()})",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 0.4)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=8)

    fig.suptitle(
        "Position Info: Direction vs Norm Component",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = PLOTS_DIR / "direction_vs_norm_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(
        PLOTS_DIR / "direction_vs_norm_comparison.pdf", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {save_path}")
    plt.close(fig)


def print_summary_table(all_results):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY: Linear Probe Position Classification (32 bins, chance = 3.1%)")
    print("=" * 100)

    key_layers = ["post_ln1", "post_attn", "pre_ln2", "post_ln2", "post_mlp"]

    for init_type in ["random", "trained"]:
        print(f"\n--- {init_type.upper()} ---")
        print(f"{'Model':<20}", end="")
        for layer in key_layers:
            print(f" {layer:>15}", end="")
        print()
        print("-" * 100)

        for name, results in all_results.items():
            data = results.get(init_type, {})
            if not data:
                continue

            short_name = "LayerNorm" if "LayerNorm" in name else "RMSNorm"
            print(f"{short_name:<20}", end="")
            for layer in key_layers:
                m = data.get(layer, {})
                acc = m.get("full_accuracy", 0)
                print(f" {acc * 100:>14.1f}%", end="")
            print()


def main():
    # Load results
    results_path = RESULTS_DIR / "linear_probe_results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    print("Loaded results from:", results_path)

    # Initialize wandb if available
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="nope-linear-probing",
            name="final_plots",
            config={"task": "generate_plots"},
        )

    # Generate plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    create_summary_bar_plot(all_results, use_wandb)
    create_probing_line_plot(all_results, use_wandb)
    create_norm_direction_comparison(all_results)

    # Print summary
    print_summary_table(all_results)

    if use_wandb:
        wandb.finish()

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
