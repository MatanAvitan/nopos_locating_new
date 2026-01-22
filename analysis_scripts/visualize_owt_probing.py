"""
Visualization for OWT Direction vs Norm Probing Results
"""

import json
from pathlib import Path
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_PATH = (
    Path(__file__).parent.parent
    / "results"
    / "owt_direction_norm_probing"
    / "owt_direction_norm_results.json"
)
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "owt_direction_norm_probing"

with open(RESULTS_PATH) as f:
    results = json.load(f)

layers = ["post_attn", "post_attn_residual", "post_ln2", "post_mlp_residual"]
layer_labels = ["Post-Attn", "Post-Attn Residual", "Post-LN2", "Post-MLP Residual"]

colors = {
    "NoPE + LayerNorm": "#1f77b4",
    "NoPE + BatchNorm2": "#ff7f0e",
    "Baseline + PE": "#2ca02c",
}

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Full Activation R² (Information Flow)",
        "Direction vs Norm Encoding",
    ),
    horizontal_spacing=0.15,
)

for exp_name, exp_data in results.items():
    full_r2 = [exp_data[layer]["full_r2"] for layer in layers]
    dir_r2 = [exp_data[layer]["direction_r2"] for layer in layers]
    norm_r2 = [exp_data[layer]["norm_r2"] for layer in layers]

    fig.add_trace(
        go.Scatter(
            x=layer_labels,
            y=full_r2,
            name=f"{exp_name} (Full)",
            mode="lines+markers",
            line=dict(color=colors[exp_name], width=2, dash="solid"),
            marker=dict(size=8),
        ),
        row=1,
        col=1,
    )

fig.update_layout(
    title=dict(text="Positional Information Flow Through Network", font=dict(size=18)),
    template="plotly_white",
    height=450,
    width=1100,
    legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.99),
    font=dict(family="Serif", size=12),
)

fig.update_xaxes(title_text="Layer", row=1, col=1)
fig.update_yaxes(title_text="R² (Position Decoding)", row=1, col=1, range=[0, 0.6])

fig.add_trace(
    go.Bar(
        name="Direction R²",
        x=[(exp, layer) for exp in results.keys() for layer in layer_labels],
        y=[
            exp_data[layer]["direction_r2"]
            for exp_data in results.values()
            for layer in layers
        ],
        marker_color="#3498db",
        opacity=0.8,
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Bar(
        name="Norm R²",
        x=[(exp, layer) for exp in results.keys() for layer in layer_labels],
        y=[
            exp_data[layer]["norm_r2"]
            for exp_data in results.values()
            for layer in layers
        ],
        marker_color="#e74c3c",
        opacity=0.8,
    ),
    row=1,
    col=2,
)

fig.update_xaxes(
    tickvals=[(exp, layer_labels[1]) for exp in results.keys()],
    ticktext=list(results.keys()),
    row=1,
    col=2,
)

fig.update_layout(
    barmode="group", legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01)
)

fig.write_image(
    OUTPUT_DIR / "owt_direction_norm_probing.png", width=1100, height=500, scale=2
)
fig.write_image(OUTPUT_DIR / "owt_direction_norm_probing.pdf")

print(f"Saved visualization to {OUTPUT_DIR}")
