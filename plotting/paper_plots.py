"""
Enhanced plotting utilities for generating publication-quality figures.
Extends base utilities from utils.py with specialized functions for paper figures.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from transformer_lens.utils import to_numpy
import torch
from scipy import stats
from scipy.optimize import curve_fit

# Import base utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import line, imshow


def save_publication_figure(fig, filename, plot_dir="/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots"):
    """
    Save figure in both PNG and PDF formats for publication.

    Args:
        fig: Plotly figure object
        filename: Base filename (without extension)
        plot_dir: Directory to save plots
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save PNG at high resolution
    fig.write_image(str(plot_dir / f"{filename}.png"), width=800, height=600, scale=2)  # 300 DPI

    # Save PDF for vector graphics
    fig.write_image(str(plot_dir / f"{filename}.pdf"))

    print(f"✓ Saved {filename}.png and {filename}.pdf to {plot_dir}")


def create_comparison_plot(data_dict, title="", xaxis="", yaxis="", plot_dir=None):
    """
    Create side-by-side comparison plots.

    Args:
        data_dict: Dict of {label: tensor} for each comparison
        title: Overall title
        xaxis, yaxis: Axis labels
        plot_dir: If provided, save to this directory

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for label, data in data_dict.items():
        data = to_numpy(data)
        fig.add_trace(go.Scatter(
            y=data if data.ndim == 1 else data.mean(axis=0),
            mode='lines+markers',
            name=label,
            line=dict(width=3),
            marker=dict(size=8, symbol='circle')
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family="Serif")),
        xaxis=dict(title=xaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title=yaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        legend=dict(font=dict(size=16)),
        template="plotly_white",
        width=800,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    if plot_dir:
        save_publication_figure(fig, title.replace(" ", "_").lower(), plot_dir)

    return fig


def create_scaling_plot(vocab_sizes, min_samples, title="Vocabulary Scaling Analysis", plot_dir=None):
    """
    Create log-log scaling plot with power-law fit.

    Args:
        vocab_sizes: Array of vocabulary sizes
        min_samples: Array of minimum samples required
        title: Plot title
        plot_dir: If provided, save to this directory

    Returns:
        Plotly figure and fit parameters (coefficient, exponent, r_squared)
    """
    # Convert to numpy
    vocab_sizes = np.array(vocab_sizes)
    min_samples = np.array(min_samples)

    # Power law fit: y = a * x^b
    def power_law(x, a, b):
        return a * x ** b

    # Fit in log space for numerical stability
    log_vocab = np.log10(vocab_sizes)
    log_samples = np.log10(min_samples)

    # Linear fit in log space: log(y) = log(a) + b * log(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_vocab, log_samples)

    coefficient = 10 ** intercept
    exponent = slope
    r_squared = r_value ** 2

    # Generate fit line
    x_fit = np.logspace(np.log10(vocab_sizes.min()), np.log10(vocab_sizes.max()), 100)
    y_fit = power_law(x_fit, coefficient, exponent)

    # Create figure
    fig = go.Figure()

    # Actual data points
    fig.add_trace(go.Scatter(
        x=vocab_sizes,
        y=min_samples,
        mode='markers',
        name='Experimental Data',
        marker=dict(size=12, color='blue', symbol='circle'),
    ))

    # Fit line
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        name=f'Power Law Fit: y = {coefficient:.2f} × x^{exponent:.2f}<br>R² = {r_squared:.3f}',
        line=dict(width=3, color='red', dash='dash')
    ))

    # Log-log scale
    fig.update_xaxes(type="log", title="Vocabulary Size", title_font=dict(size=18), tickfont=dict(size=16))
    fig.update_yaxes(type="log", title="Minimum Samples Required", title_font=dict(size=18), tickfont=dict(size=16))

    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family="Serif")),
        template="plotly_white",
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(font=dict(size=14), x=0.05, y=0.95)
    )

    if plot_dir:
        save_publication_figure(fig, "vocabulary_scaling", plot_dir)

    return fig, (coefficient, exponent, r_squared)


def create_heatmap_with_marginals(data, title="", xaxis="", yaxis="", plot_dir=None):
    """
    Create heatmap with marginal distributions.

    Args:
        data: 2D array for heatmap
        title: Plot title
        xaxis, yaxis: Axis labels
        plot_dir: If provided, save to this directory

    Returns:
        Plotly figure
    """
    data = to_numpy(data)

    # Create subplots with marginals
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.2, 0.8],
        column_widths=[0.8, 0.2],
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        specs=[[{"type": "scatter"}, None],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )

    # Main heatmap
    fig.add_trace(
        go.Heatmap(
            z=data,
            colorscale="RdBu_r",
            zmid=0.0,
            showscale=True
        ),
        row=2, col=1
    )

    # Top marginal (column means)
    col_means = data.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(col_means)),
            y=col_means,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

    # Right marginal (row means)
    row_means = data.mean(axis=1)
    fig.add_trace(
        go.Scatter(
            x=row_means,
            y=np.arange(len(row_means)),
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family="Serif")),
        template="plotly_white",
        width=900,
        height=800,
        showlegend=False
    )

    # Update axes
    fig.update_xaxes(title=xaxis, row=2, col=1, title_font=dict(size=18))
    fig.update_yaxes(title=yaxis, row=2, col=1, title_font=dict(size=18))

    if plot_dir:
        save_publication_figure(fig, title.replace(" ", "_").lower(), plot_dir)

    return fig


def create_multi_panel_figure(panels_dict, title="", plot_dir=None):
    """
    Create multi-panel figure for complex visualizations.

    Args:
        panels_dict: Dict of {panel_label: (data, panel_title, plot_type)}
                    where plot_type is 'line' or 'heatmap'
        title: Overall figure title
        plot_dir: If provided, save to this directory

    Returns:
        Plotly figure
    """
    n_panels = len(panels_dict)

    # Determine layout (prefer horizontal for 2-3 panels, grid for 4+)
    if n_panels <= 3:
        rows, cols = 1, n_panels
    else:
        rows = 2
        cols = (n_panels + 1) // 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[v[1] for v in panels_dict.values()],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for idx, (label, (data, panel_title, plot_type)) in enumerate(panels_dict.items()):
        row = idx // cols + 1
        col = idx % cols + 1

        data = to_numpy(data)

        if plot_type == 'line':
            fig.add_trace(
                go.Scatter(
                    y=data if data.ndim == 1 else data.mean(axis=0),
                    mode='lines+markers',
                    name=label,
                    line=dict(width=3),
                    marker=dict(size=6)
                ),
                row=row, col=col
            )
        elif plot_type == 'heatmap':
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    colorscale="RdBu_r",
                    zmid=0.0,
                    showscale=(idx == 0)
                ),
                row=row, col=col
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=28, family="Serif")),
        template="plotly_white",
        width=1200,
        height=600 * rows,
        showlegend=False
    )

    if plot_dir:
        save_publication_figure(fig, title.replace(" ", "_").lower(), plot_dir)

    return fig


def plot_attention_patterns(attention_weights, save_path=None):
    """
    Visualize attention patterns showing near-uniform distribution within causal mask.

    Args:
        attention_weights: Tensor of shape [n_heads, n_ctx, n_ctx] or [n_ctx, n_ctx]
        save_path: Path to save figure (without extension)

    Returns:
        Plotly figure
    """
    attn = to_numpy(attention_weights)

    # Handle multi-head case
    if attn.ndim == 3:
        attn = attn.mean(axis=0)  # Average over heads

    fig = imshow(
        attn,
        xaxis="Source Position",
        yaxis="Destination Position",
        title="Attention Patterns: Near-Uniform within Causal Mask",
        save_path=save_path
    )

    return fig


def plot_variance_decay(variances, save_path=None):
    """
    Plot monotonic variance decay across positions.

    Args:
        variances: Array of variance values by position
        save_path: Path to save figure (without extension)

    Returns:
        Plotly figure
    """
    variances = to_numpy(variances)

    fig = line(
        variances,
        xaxis="Position",
        yaxis="Variance",
        title="Monotonic Variance Decay in Attention Outputs",
        save_path=save_path
    )

    return fig


def plot_layernorm_paradox(single_sample, population_avg, save_path=None):
    """
    Create side-by-side comparison showing LayerNorm paradox.

    Args:
        single_sample: Activations from single sample [n_ctx, d_model]
        population_avg: Population average activations [n_ctx, d_model]
        save_path: Path to save figure (without extension)

    Returns:
        Plotly figure
    """
    single = to_numpy(single_sample)
    pop_avg = to_numpy(population_avg)

    # Create two-panel figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Single Sample (No Pattern)", "Population Average (Clear Pattern)"),
        horizontal_spacing=0.15
    )

    # Single sample heatmap
    fig.add_trace(
        go.Heatmap(
            z=single.T,
            colorscale="RdBu_r",
            zmid=0.0,
            showscale=False,
            name="Single Sample"
        ),
        row=1, col=1
    )

    # Population average heatmap
    fig.add_trace(
        go.Heatmap(
            z=pop_avg.T,
            colorscale="RdBu_r",
            zmid=0.0,
            showscale=True,
            name="Population Average"
        ),
        row=1, col=2
    )

    fig.update_xaxes(title="Position", row=1, col=1)
    fig.update_xaxes(title="Position", row=1, col=2)
    fig.update_yaxes(title="Neuron Index", row=1, col=1)
    fig.update_yaxes(title="Neuron Index", row=1, col=2)

    fig.update_layout(
        title=dict(text="The LayerNorm Paradox", font=dict(size=28, family="Serif")),
        template="plotly_white",
        width=1400,
        height=600,
        showlegend=False
    )

    if save_path:
        fig.write_image(f"{save_path}.png", width=1400, height=600, scale=2)
        fig.write_image(f"{save_path}.pdf")
        print(f"✓ Saved layernorm_paradox figure to {save_path}")

    return fig


def plot_sample_convergence(model, dataloader, sample_sizes=[10, 50, 100, 250, 500],
                            hook_name='blocks.0.ln2.hook_normalized', save_path=None):
    """
    Show pattern emergence with increasing sample sizes.

    Args:
        model: Trained transformer model
        dataloader: Data loader for getting samples
        sample_sizes: List of sample sizes to test
        hook_name: Name of hook to extract activations
        save_path: Path to save figure (without extension)

    Returns:
        Plotly figure
    """
    device = next(model.parameters()).device

    # Collect samples
    all_activations = []
    for tokens, _ in dataloader:
        tokens = tokens.to(device)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name].detach().cpu()  # [B, N_CTX, D_MODEL]
        all_activations.append(acts)
        if len(all_activations) * acts.shape[0] >= max(sample_sizes):
            break

    all_activations = torch.cat(all_activations, dim=0)  # [N_samples, N_CTX, D_MODEL]

    # Create subplots for each sample size
    n_sizes = len(sample_sizes)
    cols = min(3, n_sizes)
    rows = (n_sizes + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{size} Samples" for size in sample_sizes],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for idx, size in enumerate(sample_sizes):
        row = idx // cols + 1
        col = idx % cols + 1

        # Average over samples
        avg_pattern = all_activations[:size].mean(dim=0).mean(dim=1)  # [D_MODEL]

        fig.add_trace(
            go.Scatter(
                y=avg_pattern.numpy(),
                mode='lines',
                line=dict(width=2, color='blue'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title="Neuron" if row == rows else "", row=row, col=col)
        fig.update_yaxes(title="Activation" if col == 1 else "", row=row, col=col)

    fig.update_layout(
        title=dict(text="Emergence of Positional Patterns with Sample Size", font=dict(size=24, family="Serif")),
        template="plotly_white",
        width=1200,
        height=400 * rows,
        showlegend=False
    )

    if save_path:
        fig.write_image(f"{save_path}.png", width=1200, height=400 * rows, scale=2)
        fig.write_image(f"{save_path}.pdf")
        print(f"✓ Saved sample_convergence figure to {save_path}")

    return fig


# Utility function to ensure kaleido is installed for image export
def check_plot_dependencies():
    """
    Check if required plotting dependencies are installed.
    """
    try:
        import kaleido
        print("✓ Kaleido installed - PNG/PDF export available")
        return True
    except ImportError:
        print("⚠ Warning: kaleido not installed. Install with: pip install kaleido")
        print("  Without kaleido, figure export will not work.")
        return False


if __name__ == "__main__":
    print("Paper plotting utilities loaded successfully")
    check_plot_dependencies()
