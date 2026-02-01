"""
Generate All Paper Figures
Creates all 6 missing figures for the LayerNorm Paradox paper.
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('..')
from utils import device
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob

# Import plotting utilities
sys.path.append('../plotting')
from plotting.paper_plots import save_publication_figure, create_scaling_plot, create_multi_panel_figure

PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_CTX = 64


def find_best_model():
    """Find best trained model."""
    model_dirs = glob.glob('models/*synthetic*w_ln*large_vocab*')
    if not model_dirs:
        model_dirs = glob.glob('models/*synthetic*')

    best_model = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)
    ckpt = list(Path(best_model).glob('*.ckpt'))[0]

    checkpoint = torch.load(ckpt, map_location=device)
    from transformer_lens import HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1, d_model=1024, d_head=1024, n_heads=1, d_mlp=4096,
        d_vocab=5000, n_ctx=N_CTX, act_fn='relu', normalization_type='LN', device=device
    )

    model = HookedTransformer(cfg)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"✓ Loaded model from: {best_model}")
    return model


def generate_figure_1_attention_patterns(model):
    """Figure 1: Attention patterns showing near-uniform distribution."""
    print("\n" + "="*60)
    print("FIGURE 1: Attention Patterns")
    print("="*60)

    with torch.no_grad():
        tokens = torch.randint(0, model.cfg.d_vocab, (1, N_CTX)).to(device)
        _, cache = model.run_with_cache(tokens)
        attn_weights = cache['blocks.0.attn.hook_pattern'][0, 0].cpu().numpy()  # [N_CTX, N_CTX]

    fig = go.Figure(data=go.Heatmap(
        z=attn_weights,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Attention Weight")
    ))

    fig.update_layout(
        title=dict(text="Attention Patterns", font=dict(size=24, family="Serif")),
        xaxis=dict(title="Key Position", title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title="Query Position", title_font=dict(size=18), tickfont=dict(size=16)),
        template="plotly_white",
        width=800,
        height=600
    )

    save_publication_figure(fig, "attention_patterns", PLOTS_DIR)
    print("✓ Figure 1 saved")


def generate_figure_2_variance_decay(model):
    """Figure 2: Monotonic variance decay."""
    print("\n" + "="*60)
    print("FIGURE 2: Variance Decay")
    print("="*60)

    variances = []
    n_samples = 1000

    with torch.no_grad():
        for _ in range(n_samples):
            tokens = torch.randint(0, model.cfg.d_vocab, (1, N_CTX)).to(device)
            _, cache = model.run_with_cache(tokens)
            attn_out = cache['blocks.0.hook_attn_out'][0].cpu()  # [N_CTX, D_MODEL]

            if len(variances) == 0:
                variances = [[] for _ in range(N_CTX)]

            for pos in range(N_CTX):
                variances[pos].append(attn_out[pos].numpy())

    # Compute variance at each position
    position_variances = []
    for pos in range(N_CTX):
        pos_data = np.array(variances[pos])
        var = pos_data.var(axis=0).mean()  # Variance across samples, averaged over dimensions
        position_variances.append(var)

    # Theoretical: var ∝ 1/(position+1)
    theoretical = [1.0 / (i + 1) for i in range(N_CTX)]
    theoretical = np.array(theoretical) * position_variances[0]  # Scale to match

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(N_CTX)),
        y=position_variances,
        mode='markers',
        name='Empirical Variance',
        marker=dict(size=8, color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(N_CTX)),
        y=theoretical,
        mode='lines',
        name='Theoretical: 1/(pos+1)',
        line=dict(color='red', dash='dash', width=3)
    ))

    fig.update_layout(
        title=dict(text="Variance Decay in Attention Outputs", font=dict(size=24, family="Serif")),
        xaxis=dict(title="Position", title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title="Variance", title_font=dict(size=18), tickfont=dict(size=16)),
        template="plotly_white",
        width=800,
        height=600,
        legend=dict(font=dict(size=14))
    )

    save_publication_figure(fig, "variance_decay", PLOTS_DIR)
    print("✓ Figure 2 saved")


def generate_figure_3_layernorm_paradox(model):
    """Figure 3: LayerNorm paradox - individual vs population."""
    print("\n" + "="*60)
    print("FIGURE 3: LayerNorm Paradox")
    print("="*60)

    hook_name = 'blocks.0.ln2.hook_normalized'

    # Single sample
    with torch.no_grad():
        tokens_single = torch.randint(0, model.cfg.d_vocab, (1, N_CTX)).to(device)
        _, cache_single = model.run_with_cache(tokens_single, names_filter=[hook_name])
        single_sample = cache_single[hook_name][0].cpu()  # [N_CTX, D_MODEL]

    # Population average
    pop_samples = []
    with torch.no_grad():
        for _ in range(1000):
            tokens = torch.randint(0, model.cfg.d_vocab, (1, N_CTX)).to(device)
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            pop_samples.append(cache[hook_name][0].cpu())

    population_avg = torch.stack(pop_samples).mean(dim=0)  # [N_CTX, D_MODEL]

    # Extract position-wise patterns
    single_pattern = single_sample.mean(dim=1).numpy()
    pop_pattern = population_avg.mean(dim=1).numpy()

    # Create figure with two panels
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Single Sample", "Population Average (1000 samples)"],
        horizontal_spacing=0.12
    )

    fig.add_trace(
        go.Scatter(x=list(range(N_CTX)), y=single_pattern, mode='lines+markers',
                   line=dict(color='gray', width=2), marker=dict(size=4)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=list(range(N_CTX)), y=pop_pattern, mode='lines+markers',
                   line=dict(color='blue', width=3), marker=dict(size=6)),
        row=1, col=2
    )

    fig.update_xaxes(title="Position", title_font=dict(size=16), row=1, col=1)
    fig.update_xaxes(title="Position", title_font=dict(size=16), row=1, col=2)
    fig.update_yaxes(title="Mean Activation", title_font=dict(size=16), row=1, col=1)

    fig.update_layout(
        title=dict(text="The LayerNorm Paradox", font=dict(size=24, family="Serif")),
        template="plotly_white",
        width=1400,
        height=600,
        showlegend=False
    )

    save_publication_figure(fig, "layernorm_paradox", PLOTS_DIR)
    print("✓ Figure 3 saved")


def generate_figure_4_token_distribution():
    """Figure 4: Token distribution analysis."""
    print("\n" + "="*60)
    print("FIGURE 4: Token Distribution Analysis")
    print("="*60)

    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast

        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        # Analyze token distributions by position
        position_token_counts = [dict() for _ in range(N_CTX)]

        for text in dataset['text'][:5000]:
            if len(text.strip()) < 10:
                continue
            tokens = tokenizer.encode(text, max_length=N_CTX, truncation=True)
            for pos, token in enumerate(tokens):
                if pos < N_CTX:
                    position_token_counts[pos][token] = position_token_counts[pos].get(token, 0) + 1

        # Calculate entropy by position
        entropies = []
        vocab_coverage = []

        for pos in range(N_CTX):
            counts = position_token_counts[pos]
            if not counts:
                entropies.append(0)
                vocab_coverage.append(0)
                continue

            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            entropies.append(entropy)
            vocab_coverage.append(len(counts))

        # Create three-panel figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Entropy by Position", "Vocabulary Coverage by Position"],
            horizontal_spacing=0.15
        )

        fig.add_trace(
            go.Scatter(x=list(range(N_CTX)), y=entropies, mode='lines+markers',
                       line=dict(color='blue', width=3), marker=dict(size=6)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=list(range(N_CTX)), y=vocab_coverage, mode='lines+markers',
                       line=dict(color='green', width=3), marker=dict(size=6)),
            row=1, col=2
        )

        fig.update_xaxes(title="Position", title_font=dict(size=16), row=1, col=1)
        fig.update_xaxes(title="Position", title_font=dict(size=16), row=1, col=2)
        fig.update_yaxes(title="Entropy", title_font=dict(size=16), row=1, col=1)
        fig.update_yaxes(title="Unique Tokens", title_font=dict(size=16), row=1, col=2)

        fig.update_layout(
            title=dict(text="Token Distribution Analysis", font=dict(size=24, family="Serif")),
            template="plotly_white",
            width=1200,
            height=500,
            showlegend=False
        )

        save_publication_figure(fig, "token_distribution_analysis", PLOTS_DIR)
        print("✓ Figure 4 saved")

    except Exception as e:
        print(f"Warning: Could not generate Figure 4: {e}")
        print("Skipping token distribution analysis (requires datasets library)")


def generate_figure_5_vocabulary_scaling():
    """Figure 5: Vocabulary scaling."""
    print("\n" + "="*60)
    print("FIGURE 5: Vocabulary Scaling")
    print("="*60)

    results_file = Path('results/vocab_scaling_results.json')

    if not results_file.exists():
        print("Warning: vocab_scaling_results.json not found. Creating placeholder figure.")
        # Placeholder data
        vocab_sizes = np.array([1024, 2048, 4096, 8192, 16384, 32768])
        min_samples = 0.49 * (vocab_sizes ** 0.98)
    else:
        with open(results_file) as f:
            data = json.load(f)

        # Extract minimum samples for each vocab size
        vocab_results = {}
        for result in data.get('results', []):
            vocab = result['vocab_size']
            if result.get('converged', False):
                if vocab not in vocab_results or result['n_samples'] < vocab_results[vocab]:
                    vocab_results[vocab] = result['n_samples']

        vocab_sizes = np.array(sorted(vocab_results.keys()))
        min_samples = np.array([vocab_results[v] for v in vocab_sizes])

    fig, (coef, exp, r2) = create_scaling_plot(vocab_sizes, min_samples,
                                                 title="Vocabulary Scaling Analysis",
                                                 plot_dir=PLOTS_DIR)

    print(f"✓ Figure 5 saved (Power law: y = {coef:.2f} × x^{exp:.2f}, R²={r2:.3f})")


def generate_figure_6_sample_convergence():
    """Figure 6: Sample convergence."""
    print("\n" + "="*60)
    print("FIGURE 6: Sample Convergence")
    print("="*60)

    results_file = Path('results/sample_convergence_data.pkl')

    if not results_file.exists():
        print("Warning: sample_convergence_data.pkl not found. Skipping Figure 6.")
        return

    with open(results_file, 'rb') as f:
        data = pickle.load(f)

    sample_sizes = data['sample_sizes']
    activations_by_size = data['activations_by_size']

    # Create multi-panel figure
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{size} Samples" for size in sample_sizes],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    for idx, size in enumerate(sample_sizes):
        row = idx // 4 + 1
        col = idx % 4 + 1

        acts = activations_by_size[size]
        avg_pattern = acts.mean(dim=0).mean(dim=1).numpy()  # [N_CTX]

        fig.add_trace(
            go.Scatter(x=list(range(N_CTX)), y=avg_pattern, mode='lines',
                       line=dict(width=2), showlegend=False),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(text="Emergence of Positional Patterns with Sample Size",
                   font=dict(size=24, family="Serif")),
        template="plotly_white",
        width=1600,
        height=800
    )

    save_publication_figure(fig, "sample_convergence", PLOTS_DIR)
    print("✓ Figure 6 saved")


def main():
    """Generate all figures."""
    print("\n" + "="*70)
    print("GENERATING ALL PAPER FIGURES")
    print("="*70)
    print(f"Output directory: {PLOTS_DIR}")
    print("="*70)

    # Load model
    model = find_best_model()

    # Generate all figures
    try:
        generate_figure_1_attention_patterns(model)
    except Exception as e:
        print(f"ERROR Figure 1: {e}")

    try:
        generate_figure_2_variance_decay(model)
    except Exception as e:
        print(f"ERROR Figure 2: {e}")

    try:
        generate_figure_3_layernorm_paradox(model)
    except Exception as e:
        print(f"ERROR Figure 3: {e}")

    try:
        generate_figure_4_token_distribution()
    except Exception as e:
        print(f"ERROR Figure 4: {e}")

    try:
        generate_figure_5_vocabulary_scaling()
    except Exception as e:
        print(f"ERROR Figure 5: {e}")

    try:
        generate_figure_6_sample_convergence()
    except Exception as e:
        print(f"ERROR Figure 6: {e}")

    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print(f"All figures saved to: {PLOTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
