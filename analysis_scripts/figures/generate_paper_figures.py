"""
Generate publication-quality figures for the NoPE paper.

This script creates:
1. Information flow diagram (TikZ code for the paper)
2. Trained vs Random comparison figure
3. Attention uniformity comparison
4. Direction vs Norm decomposition across layers
5. Long context analysis
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set publication style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "overleaf" / "nopos---claude-version" / "plots"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load all relevant results files."""
    results = {}

    # Trained model results
    trained_path = RESULTS_DIR / "trained_model_analysis" / "trained_model_results.json"
    if trained_path.exists():
        with open(trained_path) as f:
            results["trained"] = json.load(f)

    # Attention comparison
    attn_path = (
        RESULTS_DIR / "attention_comparison" / "attention_comparison_results.json"
    )
    if attn_path.exists():
        with open(attn_path) as f:
            results["attention"] = json.load(f)

    # Long context analysis
    long_ctx_path = (
        RESULTS_DIR / "long_context_analysis" / "long_context_ln_vs_rms_results.json"
    )
    if long_ctx_path.exists():
        with open(long_ctx_path) as f:
            results["long_context"] = json.load(f)

    return results


def fig1_trained_vs_random_comparison(results):
    """
    Create comparison figure showing how training changes position encoding.

    Key message: Training DESTROYS the norm-based position encoding mechanism.
    """
    if "trained" not in results:
        print("Skipping trained vs random figure - no data")
        return

    data = results["trained"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Norm-Position Correlation
    ax = axes[0, 0]
    models = ["Random Init", "LayerNorm (trained)", "RMSNorm (trained)"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    post_attn_corrs = [data[m]["post_attn_norm_position_corr"] for m in models]
    post_ln2_corrs = [data[m]["post_ln2_norm_position_corr"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        post_attn_corrs,
        width,
        label="Post-Attention",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        post_ln2_corrs,
        width,
        label="Post-LN2",
        color="#e74c3c",
        alpha=0.8,
    )

    ax.set_ylabel("Norm-Position Correlation")
    ax.set_title("A) Norm-Position Correlation: Training Destroys Signal")
    ax.set_xticks(x)
    ax.set_xticklabels(["Random", "Trained\n(LN)", "Trained\n(RMS)"], fontsize=9)
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(-1.1, 0.3)

    # Add annotation
    ax.annotate(
        "Training inverts\nthe correlation!",
        xy=(1.5, 0.15),
        fontsize=9,
        ha="center",
        style="italic",
        color="#e74c3c",
    )

    # Panel B: Linear Probe R² (Norm only)
    ax = axes[0, 1]

    post_attn_r2 = [data[m]["post_attn_norm_r2"] for m in models]
    post_ln2_r2 = [data[m]["post_ln2_norm_r2"] for m in models]

    bars1 = ax.bar(
        x - width / 2,
        post_attn_r2,
        width,
        label="Post-Attention",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, post_ln2_r2, width, label="Post-LN2", color="#e74c3c", alpha=0.8
    )

    ax.set_ylabel("Norm-based Probe R²")
    ax.set_title("B) Position Decodability from Norm")
    ax.set_xticks(x)
    ax.set_xticklabels(["Random", "Trained\n(LN)", "Trained\n(RMS)"], fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)

    # Add annotation for random model
    ax.annotate(
        "R²=0.94",
        xy=(0.175, 0.95),
        fontsize=9,
        ha="center",
        color="#e74c3c",
        fontweight="bold",
    )
    ax.annotate("R²=0.02", xy=(1.175, 0.05), fontsize=9, ha="center", color="#e74c3c")

    # Panel C: Direction R² comparison
    ax = axes[1, 0]

    post_attn_dir = [data[m]["post_attn_direction_r2"] for m in models]
    post_ln2_dir = [data[m]["post_ln2_direction_r2"] for m in models]

    bars1 = ax.bar(
        x - width / 2,
        post_attn_dir,
        width,
        label="Post-Attention",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, post_ln2_dir, width, label="Post-LN2", color="#e74c3c", alpha=0.8
    )

    ax.set_ylabel("Direction-based Probe R²")
    ax.set_title("C) Position in Directional Structure")
    ax.set_xticks(x)
    ax.set_xticklabels(["Random", "Trained\n(LN)", "Trained\n(RMS)"], fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.35)

    # Panel D: Summary table
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = """
KEY FINDINGS: Random vs Trained Models

┌────────────────────┬──────────┬──────────┐
│ Metric             │  Random  │ Trained  │
├────────────────────┼──────────┼──────────┤
│ Norm-Pos Corr      │  -0.97   │  +0.15   │
│ (post-LN2)         │          │          │
├────────────────────┼──────────┼──────────┤
│ Norm R² (post-LN2) │   0.94   │   0.02   │
├────────────────────┼──────────┼──────────┤
│ Direction R²       │   0.23   │   0.18   │
│ (post-attn)        │          │          │
└────────────────────┴──────────┴──────────┘

CONCLUSION:
• The norm-based position encoding is an 
  emergent property of RANDOM initialization
• Training DESTROYS this mechanism
• Trained models must use a different 
  (unknown) mechanism for position awareness
"""

    ax.text(
        0.1,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "trained_vs_random_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "trained_vs_random_comparison.pdf", bbox_inches="tight")
    print(f"Saved: trained_vs_random_comparison.png/pdf")
    plt.close()


def fig2_attention_uniformity(results):
    """
    Create figure showing attention patterns remain uniform in both trained and random models.
    """
    if "attention" not in results:
        print("Skipping attention figure - no data")
        return

    data = results["attention"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    positions = np.arange(256)

    # Panel A: Entropy vs theoretical
    ax = axes[0]
    theoretical = np.log(positions + 1)

    ax.plot(
        positions,
        data["trained"]["entropy_by_position"],
        "b-",
        label="Trained",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        positions,
        data["random"]["entropy_by_position"],
        "r-",
        label="Random",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        positions, theoretical, "k--", label="Theory: log(i+1)", alpha=0.5, linewidth=1
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Attention Entropy")
    ax.set_title(
        f"A) Entropy follows theory\n(Trained r={data['trained']['entropy_correlation_with_theory']:.4f})"
    )
    ax.legend(loc="lower right")
    ax.set_xlim(0, 255)

    # Panel B: Uniformity score
    ax = axes[1]

    ax.plot(
        positions,
        data["trained"]["uniformity_by_position"],
        "b-",
        label="Trained",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        positions,
        data["random"]["uniformity_by_position"],
        "r-",
        label="Random",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect uniform")

    ax.set_xlabel("Position")
    ax.set_ylabel("Uniformity (actual/theoretical)")
    ax.set_title(
        f"B) Both models near-uniform\n(Trained: {data['trained']['mean_uniformity']:.2%}, Random: {data['random']['mean_uniformity']:.2%})"
    )
    ax.legend(loc="lower right")
    ax.set_xlim(0, 255)
    ax.set_ylim(0.85, 1.02)

    # Panel C: First token attention
    ax = axes[2]

    theoretical_first = 1.0 / (positions + 1)

    ax.plot(
        positions,
        data["trained"]["first_token_attention"],
        "b-",
        label="Trained",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        positions,
        data["random"]["first_token_attention"],
        "r-",
        label="Random",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        positions,
        theoretical_first,
        "k--",
        label="Theory: 1/(i+1)",
        alpha=0.5,
        linewidth=1,
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Attention to First Token")
    ax.set_title("C) First-token attention decay")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 255)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "attention_uniformity_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "attention_uniformity_comparison.pdf", bbox_inches="tight")
    print(f"Saved: attention_uniformity_comparison.png/pdf")
    plt.close()


def fig3_information_flow_diagram():
    """
    Generate description for information flow TikZ diagram.
    This outputs the key numbers to be used in the paper figure.
    """
    print("\n" + "=" * 60)
    print("INFORMATION FLOW DIAGRAM DATA")
    print("=" * 60)
    print("""
For the TikZ diagram in the paper, use these values:

RANDOM MODEL - Synthetic-Small Config (from direction_norm_independence_results.json):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Embed     │  Post-Attn  │  Post-LN2   │  Post-MLP   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Full R²: 0  │ Full: 0.04  │ Full: 0.19  │ Full: 0.22  │
│ Norm R²: 0  │ Norm: 0.56  │ Norm: 0.88  │ Norm: 0.39  │
│ Dir R²:  0  │ Dir:  0.39  │ Dir:  0.19  │ Dir:  0.35  │
└─────────────┴─────────────┴─────────────┴─────────────┘

Key insight: Position is encoded in BOTH norm and direction post-attention,
then LayerNorm LINEARIZES it into norm (R²=0.88), direction drops to 0.19.
After MLP, direction recovers (0.35 vs 0.39 norm).

TRAINED MODEL (from trained_model_results.json on Shakespeare):
┌─────────────┬─────────────┬─────────────┐
│   Embed     │  Post-Attn  │  Post-LN2   │
├─────────────┼─────────────┼─────────────┤
│ Full R²: 0  │ Full: 0.15  │ Full: 0.18  │
│ Norm R²: 0  │ Norm: 0.02  │ Norm: 0.02  │
│ Dir R²:  0  │ Dir:  0.18  │ Dir:  0.18  │
└─────────────┴─────────────┴─────────────┘

Training destroys the norm-based encoding!
""")


def fig4_linearization_effect():
    """
    Create figure showing the LayerNorm linearization effect.
    Before LN: high direction R², low full R²
    After LN: norm dominates

    Values from direction_norm_independence_results.json (Synthetic-Small config):
    - post_attn: full=0.04, norm=0.56, direction=0.39
    - post_ln2: full=0.19, norm=0.88, direction=0.19
    - post_mlp_residual: full=0.22, norm=0.39, direction=0.35
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    layers = ["Embedding", "Post-Attention", "Post-LN2", "Post-MLP"]

    # Random model data - UPDATED from direction_norm_independence_results.json
    full_r2 = [0, 0.04, 0.19, 0.22]
    norm_r2 = [0, 0.56, 0.88, 0.39]
    direction_r2 = [0, 0.39, 0.19, 0.35]

    x = np.arange(len(layers))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        full_r2,
        width,
        label="Full Activation R²",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax.bar(x, norm_r2, width, label="Norm Only R²", color="#e74c3c", alpha=0.8)
    bars3 = ax.bar(
        x + width,
        direction_r2,
        width,
        label="Direction Only R²",
        color="#2ecc71",
        alpha=0.8,
    )

    ax.set_ylabel("Linear Probe R²")
    ax.set_title(
        "LayerNorm Linearizes Position Encoding\n(Random Initialization, Synthetic-Small Config)",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.0)

    # Add annotations
    ax.annotate(
        "Position in\nNORM + DIR",
        xy=(1, 0.65),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="#9b59b6", alpha=0.3),
    )

    ax.annotate(
        "LN linearizes\ninto NORM",
        xy=(2, 0.95),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="#e74c3c", alpha=0.3),
    )

    # Draw arrow showing transformation
    ax.annotate(
        "",
        xy=(2, 0.88),
        xytext=(1.3, 0.60),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "layernorm_linearization.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "layernorm_linearization.pdf", bbox_inches="tight")
    print(f"Saved: layernorm_linearization.png/pdf")
    plt.close()


def main():
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    results = load_results()
    print(f"Loaded results: {list(results.keys())}")

    # Generate figures
    fig1_trained_vs_random_comparison(results)
    fig2_attention_uniformity(results)
    fig3_information_flow_diagram()
    fig4_linearization_effect()

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
