"""
Publication-quality extrapolation plots for ICML 2026.
Two panels: (a) Position decoding R², (b) Mean channel correlation.
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

# ICML-quality settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Load results
with open(
    "results/extrapolation_long_context/extrapolation_extended_results.json", "r"
) as f:
    results = json.load(f)


# Extract data
def get_data(model_name):
    lengths = []
    r2_values = []
    mean_corr = []

    for L_str, data in sorted(results[model_name].items(), key=lambda x: int(x[0])):
        L = int(L_str)
        if data.get("success", True) and "linear_probe_r2" in data:
            lengths.append(L)
            r2_values.append(data["linear_probe_r2"])
            # Handle different key names in the data
            if "mean_abs_corr" in data:
                mean_corr.append(data["mean_abs_corr"])
            elif "mean_abs_corr_neuron" in data:
                mean_corr.append(data["mean_abs_corr_neuron"])

    return lengths, r2_values, mean_corr


r0_lengths, r0_r2, r0_corr = get_data("R0")
r2_lengths, r2_r2, r2_corr = get_data("R2")

# Colors (colorblind-friendly)
COLOR_R0 = "#0072B2"  # Blue
COLOR_R2 = "#D55E00"  # Vermillion

# Create figure - ICML single column is 3.25in, use 2-panel layout
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))

# Panel (a): Position Decoding Accuracy
ax = axes[0]
ax.plot(
    r0_lengths,
    r0_r2,
    "o-",
    color=COLOR_R0,
    label="R0 (full training)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.plot(
    r2_lengths,
    r2_r2,
    "s-",
    color=COLOR_R2,
    label="R2 (attention-only)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.axvline(x=128, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Context length")
ax.set_ylabel("Linear probe $R^2$")
ax.set_xscale("log", base=2)
ax.set_xlim(90, 12000)
ax.set_ylim(0.6, 1.02)
ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["128", "256", "512", "1K", "2K", "4K", "8K"])
ax.legend(loc="lower left", frameon=False)
ax.text(
    0.02,
    0.98,
    "(a)",
    transform=ax.transAxes,
    fontweight="bold",
    va="top",
    ha="left",
    fontsize=10,
)

# Panel (b): Mean Channel Correlation
ax = axes[1]
ax.plot(
    r0_lengths,
    r0_corr,
    "o-",
    color=COLOR_R0,
    label="R0 (full training)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.plot(
    r2_lengths,
    r2_corr,
    "s-",
    color=COLOR_R2,
    label="R2 (attention-only)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.axvline(x=128, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Context length")
ax.set_ylabel("Mean $|r|$ with position")
ax.set_xscale("log", base=2)
ax.set_xlim(90, 12000)
ax.set_ylim(0.5, 0.85)
ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["128", "256", "512", "1K", "2K", "4K", "8K"])
ax.legend(loc="lower left", frameon=False)
ax.text(
    0.02,
    0.98,
    "(b)",
    transform=ax.transAxes,
    fontweight="bold",
    va="top",
    ha="left",
    fontsize=10,
)

plt.tight_layout(w_pad=2.0)

# Save
save_dir = "results/extrapolation_long_context"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(os.path.join(save_dir, "extrapolation_icml.pdf"))
plt.savefig(os.path.join(save_dir, "extrapolation_icml.png"))
plt.close()

# Also save to paper directory
plt.figure(figsize=(6.5, 2.4))
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))

# Recreate for paper directory
ax = axes[0]
ax.plot(
    r0_lengths,
    r0_r2,
    "o-",
    color=COLOR_R0,
    label="R0 (full training)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.plot(
    r2_lengths,
    r2_r2,
    "s-",
    color=COLOR_R2,
    label="R2 (attention-only)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.axvline(x=128, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Context length")
ax.set_ylabel("Linear probe $R^2$")
ax.set_xscale("log", base=2)
ax.set_xlim(90, 12000)
ax.set_ylim(0.6, 1.02)
ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["128", "256", "512", "1K", "2K", "4K", "8K"])
ax.legend(loc="lower left", frameon=False)
ax.text(
    0.02,
    0.98,
    "(a)",
    transform=ax.transAxes,
    fontweight="bold",
    va="top",
    ha="left",
    fontsize=10,
)

ax = axes[1]
ax.plot(
    r0_lengths,
    r0_corr,
    "o-",
    color=COLOR_R0,
    label="R0 (full training)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.plot(
    r2_lengths,
    r2_corr,
    "s-",
    color=COLOR_R2,
    label="R2 (attention-only)",
    markerfacecolor="white",
    markeredgewidth=1.2,
)
ax.axvline(x=128, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Context length")
ax.set_ylabel("Mean $|r|$ with position")
ax.set_xscale("log", base=2)
ax.set_xlim(90, 12000)
ax.set_ylim(0.5, 0.85)
ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["128", "256", "512", "1K", "2K", "4K", "8K"])
ax.legend(loc="lower left", frameon=False)
ax.text(
    0.02,
    0.98,
    "(b)",
    transform=ax.transAxes,
    fontweight="bold",
    va="top",
    ha="left",
    fontsize=10,
)

plt.tight_layout(w_pad=2.0)
plt.savefig("overleaf/nopos_icml_2026/plots/extrapolation.pdf")
plt.close()

print("Saved publication-quality figures:")
print("  - results/extrapolation_long_context/extrapolation_icml.pdf")
print("  - overleaf/nopos_icml_2026/plots/extrapolation.pdf")

# Print summary for verification
print("\nData summary:")
print(f"R0: L={r0_lengths}, R²={[f'{x:.3f}' for x in r0_r2]}")
print(f"R2: L={r2_lengths}, R²={[f'{x:.3f}' for x in r2_r2]}")
