"""Common visualization utilities for analysis scripts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def setup_plot_style():
    """Setup consistent plot style for paper figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def save_figure(fig, output_path: str, formats=('png', 'pdf')):
    """Save figure in multiple formats.

    Args:
        fig: matplotlib figure
        output_path: Base path without extension
        formats: Tuple of formats to save (default: png and pdf)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(f"{output_path}.{fmt}", format=fmt, bbox_inches='tight')
    plt.close(fig)


def get_results_dir():
    """Get path to results directory."""
    return Path(__file__).parent.parent.parent / "results"
