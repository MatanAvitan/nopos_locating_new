"""
Create a combined comparison figure for subspace analysis across models.

This script generates a publication-ready figure comparing:
- 12-head R2 model
- 1-head R0 model (full training)
- 1-head R2 model (attention-only)
"""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ICML style settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "text.usetex": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Load results
results_paths = {
    "12-head R2": "out-2layer-mechanism/subspace_analysis/out-2layer-mechanism_R2/subspace_analysis_results.json",
    "1-head R0": "out-2layer-mechanism-1head/subspace_analysis/out-2layer-mechanism-1head_R0/subspace_analysis_results.json",
    "1-head R2": "out-2layer-mechanism-1head/subspace_analysis/out-2layer-mechanism-1head_R2/subspace_analysis_results.json",
}

results = {}
for name, path in results_paths.items():
    with open(path, "r") as f:
        results[name] = json.load(f)

# Colors for each model
colors = {
    "12-head R2": "#2ecc71",  # Green
    "1-head R0": "#3498db",  # Blue
    "1-head R2": "#e74c3c",  # Red
}

# Create figure: 2 rows x 2 columns
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# =============================================================================
# (a) Per-component R² comparison (top 30 components)
# =============================================================================
ax = axes[0, 0]
n_components = 30
x = np.arange(n_components)
width = 0.25

for i, (name, data) in enumerate(results.items()):
    r2_values = np.array(data["test6"]["per_component_r2"][:n_components])
    ax.bar(x + i * width, r2_values, width, label=name, color=colors[name], alpha=0.8)

ax.set_xlabel("SVD Component Index")
ax.set_ylabel("Position R²")
ax.set_title("(a) Per-Component Position R²")
ax.legend(loc="upper right", fontsize=7)
ax.set_xlim(-0.5, n_components)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# =============================================================================
# (b) Cumulative R² vs number of components
# =============================================================================
ax = axes[0, 1]

for name, data in results.items():
    cum_r2 = data["test6"]["cumulative_r2"]
    ks = [c["k"] for c in cum_r2]
    r2s = [c["r2"] for c in cum_r2]
    ax.plot(ks, r2s, "o-", color=colors[name], markersize=6, linewidth=2, label=name)

ax.set_xlabel("Number of Top Components (k)")
ax.set_ylabel("Cumulative Position R²")
ax.set_title("(b) Cumulative R² from Top-k Components")
ax.set_xscale("log")
ax.legend(loc="lower right", fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)

# =============================================================================
# (c) Retention R² (keeping only top-r subspace)
# =============================================================================
ax = axes[1, 0]

for name, data in results.items():
    original = data["test7"]["original_r2"]
    retained = data["test7"]["retained_r2"]
    rs = [r["r"] for r in retained]
    r2s = [r["r2"] for r in retained]

    # Plot retention curve
    ax.plot(
        rs, r2s, "s-", color=colors[name], markersize=5, linewidth=2, label=f"{name}"
    )
    # Plot original as horizontal line
    ax.axhline(y=original, color=colors[name], linestyle=":", linewidth=1, alpha=0.5)

ax.set_xlabel("Retained Subspace Dimension (r)")
ax.set_ylabel("Position R²")
ax.set_title("(c) Retention: Keep Only Top-r Subspace")
ax.set_xscale("log")
ax.legend(loc="lower right", fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.7, 1.0)

# =============================================================================
# (d) Ablation R² (removing top-r subspace)
# =============================================================================
ax = axes[1, 1]

for name, data in results.items():
    original = data["test7"]["original_r2"]
    ablated = data["test7"]["ablated_r2"]
    rs = [r["r"] for r in ablated]
    r2s = [r["r2"] for r in ablated]

    # Plot ablation curve
    ax.plot(
        rs, r2s, "o-", color=colors[name], markersize=5, linewidth=2, label=f"{name}"
    )
    # Plot original as horizontal line
    ax.axhline(y=original, color=colors[name], linestyle=":", linewidth=1, alpha=0.5)

ax.set_xlabel("Ablated Subspace Dimension (r)")
ax.set_ylabel("Position R² After Ablation")
ax.set_title("(d) Ablation: Remove Top-r Subspace")
ax.set_xscale("log")
ax.legend(loc="lower left", fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.6, 1.0)

plt.tight_layout()

# Save
output_dir = Path("out-2layer-mechanism/subspace_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

for fmt in ["png", "pdf"]:
    plt.savefig(
        output_dir / f"subspace_comparison_all_models.{fmt}",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {output_dir}/subspace_comparison_all_models.{fmt}")

plt.close()

# =============================================================================
# Create a summary table
# =============================================================================
print("\n" + "=" * 80)
print("SUBSPACE ANALYSIS SUMMARY")
print("=" * 80)

header = f"{'Model':<15} {'Heads':>6} {'Orig R²':>10} {'Max Comp R²':>12} {'Best Comp':>10} {'Top-10 R²':>10} {'Ret r=10':>10}"
print(header)
print("-" * 80)

for name, data in results.items():
    n_head = data["n_head"]
    orig_r2 = data["test7"]["original_r2"]
    max_comp_r2 = data["test6"]["max_single_component_r2"]
    best_comp = data["test6"]["argmax_component"]
    top10_r2 = data["test6"]["cumulative_r2"][2]["r2"]  # k=10
    ret_r2 = next(r["r2"] for r in data["test7"]["retained_r2"] if r["r"] == 10)

    row = f"{name:<15} {n_head:>6} {orig_r2:>10.4f} {max_comp_r2:>12.4f} {best_comp:>10} {top10_r2:>10.4f} {ret_r2:>10.4f}"
    print(row)

print("=" * 80)

# Key insights
print("\nKEY INSIGHTS:")
print("-" * 80)
print(
    "1. 12-head R2: Position distributed across components (max R²=0.77), multi-head provides redundancy"
)
print(
    "2. 1-head R0 (full train): Position concentrated in top 2 components (comp 1 R²=0.87)"
)
print(
    "3. 1-head R2 (attn-only): Position diffuse (max R²=0.37 at comp 47), struggles without MLP/LN training"
)
print(
    "4. All models: Top-10 subspace captures most position info (R²=0.87-0.89 cumulative)"
)
print("5. Retention test: Keeping top-10 nearly recovers full R² for 12-head/1-head R0")
