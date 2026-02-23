"""
Decoding Vector Experiments for OWT Models (Random Init vs Trained)

Applies the theoretical decoding vector w = W_V · Σ_j LN(E_j) at all layers
for both randomly initialized and trained NoPE models.

The decoding vector exploits near-uniform attention and embedding orthogonality
to decode position from activations. Theory predicts:
    decoded(i) = Σ_{j≤i} (w · v_j) ≈ i · c

This script tests whether this mechanism holds for:
1. Random initialization (where it should work well)
2. Trained models (where training may modify the representation)
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))

from model_nope import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "decoding_vector_owt"
CHECKPOINT_DIR = PROJECT_ROOT / "nanoGPT"
PLOTS_DIR = PROJECT_ROOT / "overleaf" / "nopos---claude-version" / "plots"


@dataclass
class ExperimentConfig:
    name: str
    short_name: str
    checkpoint_path: str
    use_positional_embedding: bool
    use_batchnorm_ln2: bool
    skip_ln2: bool


EXPERIMENTS = [
    ExperimentConfig(
        name="NoPE + LayerNorm",
        short_name="NoPE+LN",
        checkpoint_path="out-nope-owt-ln/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=False,
        skip_ln2=False,
    ),
    ExperimentConfig(
        name="NoPE + BatchNorm2",
        short_name="NoPE+BN2",
        checkpoint_path="out-nope-owt-bn2/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=True,
        skip_ln2=False,
    ),
    ExperimentConfig(
        name="NoPE + No LN2",
        short_name="NoPE+NoLN2",
        checkpoint_path="out-nope-owt-no-ln2/ckpt.pt",
        use_positional_embedding=False,
        use_batchnorm_ln2=False,
        skip_ln2=True,
    ),
    ExperimentConfig(
        name="Baseline + PE",
        short_name="Baseline+PE",
        checkpoint_path="out-baseline-owt-pe/ckpt.pt",
        use_positional_embedding=True,
        use_batchnorm_ln2=False,
        skip_ln2=False,
    ),
]

LAYERS = [
    "embed",
    "post_ln1",
    "post_attn",
    "post_attn_residual",
    "post_ln2",
    "post_mlp_residual",
]
LAYER_NAMES = {
    "embed": "Embedding",
    "post_ln1": "Post-LN1",
    "post_attn": "Post-Attn",
    "post_attn_residual": "Attn+Res",
    "post_ln2": "Post-LN2",
    "post_mlp_residual": "MLP+Res",
}


def create_random_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Create a randomly initialized model."""
    config = GPTConfig(
        n_layer=1,
        n_head=12,
        n_embd=768,
        block_size=512,
        vocab_size=50304,
        dropout=0.0,
        use_positional_embedding=exp.use_positional_embedding,
        norm_type="layernorm",
        bias=False,
        skip_ln2=exp.skip_ln2,
        use_batchnorm_ln2=exp.use_batchnorm_ln2,
    )
    model = GPT(config)
    model.eval()
    model.to(DEVICE)
    return model, config


def load_trained_model(exp: ExperimentConfig) -> Tuple[GPT, GPTConfig]:
    """Load a trained model from checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / exp.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint.get("model_args", {})

    config = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 512),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=exp.use_positional_embedding,
        norm_type=model_args.get("norm_type", "layernorm"),
        bias=model_args.get("bias", False),
        skip_ln2=exp.skip_ln2,
        use_batchnorm_ln2=exp.use_batchnorm_ln2,
    )

    model = GPT(config)
    state_dict = checkpoint["model"]
    unwrapped = {
        (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(unwrapped)
    model.eval()
    model.to(DEVICE)
    return model, config


def compute_decoding_vectors(model: GPT) -> Dict[str, torch.Tensor]:
    """
    Compute layer-appropriate decoding vectors.

    The decoding vector formula depends on the layer:
    - embed, post_ln1: w = Σ_j E_j (or Σ_j LN(E_j) for post_ln1)
      Just the sum of embeddings, no attention transforms yet
    - post_attn and beyond: w = W_O @ W_V @ Σ_j LN(E_j)
      After attention, need to account for value and output projections

    Returns:
        dict mapping layer name to normalized decoding vector
    """
    with torch.no_grad():
        # Get token embeddings
        E = model.transformer.wte.weight.detach()  # [vocab_size, n_embd]

        # Sum of raw embeddings (for embed layer)
        sum_E = E.sum(dim=0)  # [n_embd]

        # Apply LN1 to embeddings
        ln1 = model.transformer.h[0].ln_1

        # Manual LayerNorm application (to handle potential bias)
        E_centered = E - E.mean(dim=-1, keepdim=True)
        E_std = E.std(dim=-1, keepdim=True)
        E_ln = E_centered / (E_std + 1e-5)

        # Apply LN gain (and bias if present)
        E_ln = E_ln * ln1.weight
        if hasattr(ln1, "bias") and ln1.bias is not None:
            E_ln = E_ln + ln1.bias

        # Sum of LN'd embeddings (for post_ln1)
        sum_ln_E = E_ln.sum(dim=0)  # [n_embd]

        # Get W_V and W_O from attention
        attn = model.transformer.h[0].attn
        n_embd = model.config.n_embd

        # W_V is the last n_embd rows of c_attn.weight
        W_V = attn.c_attn.weight[2 * n_embd :, :].detach()  # [n_embd, n_embd]
        W_O = attn.c_proj.weight.detach()  # [n_embd, n_embd]

        # Compute post-attention decoding vector: w = W_O @ W_V @ sum_ln_E
        w_post_attn = W_O @ W_V @ sum_ln_E  # [n_embd]

        # Normalize all vectors
        def normalize(v):
            return v / (torch.norm(v) + 1e-8)

        return {
            "embed": normalize(sum_E),
            "post_ln1": normalize(sum_ln_E),
            "post_attn": normalize(w_post_attn),
            "post_attn_residual": normalize(w_post_attn),  # Same space as post_attn
            "post_ln2": normalize(w_post_attn),  # Still in same space
            "post_mlp_residual": normalize(w_post_attn),  # Still in same space
        }


def get_activations_with_decoding(
    model: GPT,
    tokens: torch.Tensor,
    decoding_vectors: Dict[str, torch.Tensor],
    skip_ln2: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get activations at each layer and compute decoding vector projection.

    Uses layer-appropriate decoding vectors:
    - embed: sum of embeddings
    - post_ln1: sum of LN'd embeddings
    - post_attn onwards: W_O @ W_V @ sum of LN'd embeddings

    Returns dict with:
        - activations: raw activations at each layer
        - projections: projection onto decoding vector at each layer
    """
    activations = {}
    projections = {}

    with torch.no_grad():
        # Token embeddings
        tok_emb = model.transformer.wte(tokens)

        # Add positional embeddings if available
        if hasattr(model.transformer, "wpe") and model.config.use_positional_embedding:
            B, T = tokens.shape
            pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = model.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        # Embedding layer - use embedding-space decoding vector
        activations["embed"] = x.detach()
        w_embed = decoding_vectors["embed"]
        projections["embed"] = (x * w_embed).sum(dim=-1).detach()

        block = model.transformer.h[0]

        # Post LN1 - use LN'd embedding-space decoding vector
        x_ln1 = block.ln_1(x)
        activations["post_ln1"] = x_ln1.detach()
        w_ln1 = decoding_vectors["post_ln1"]
        projections["post_ln1"] = (x_ln1 * w_ln1).sum(dim=-1).detach()

        # Post attention (before residual) - use post-attention decoding vector
        attn_out = block.attn(x_ln1)
        activations["post_attn"] = attn_out.detach()
        w_attn = decoding_vectors["post_attn"]
        projections["post_attn"] = (attn_out * w_attn).sum(dim=-1).detach()

        # Post attention residual
        x = x + attn_out
        activations["post_attn_residual"] = x.detach()
        w_res = decoding_vectors["post_attn_residual"]
        projections["post_attn_residual"] = (x * w_res).sum(dim=-1).detach()

        # Post LN2 (if not skipped)
        if not skip_ln2 and hasattr(block, "ln_2"):
            x_ln2 = block.ln_2(x)
            activations["post_ln2"] = x_ln2.detach()
            w_ln2 = decoding_vectors["post_ln2"]
            projections["post_ln2"] = (x_ln2 * w_ln2).sum(dim=-1).detach()
            mlp_input = x_ln2
        else:
            activations["post_ln2"] = x.detach()
            w_ln2 = decoding_vectors["post_ln2"]
            projections["post_ln2"] = (x * w_ln2).sum(dim=-1).detach()
            mlp_input = x

        # Post MLP residual
        mlp_out = block.mlp(mlp_input)
        x = x + mlp_out
        activations["post_mlp_residual"] = x.detach()
        w_mlp = decoding_vectors["post_mlp_residual"]
        projections["post_mlp_residual"] = (x * w_mlp).sum(dim=-1).detach()

    return {"activations": activations, "projections": projections}


def analyze_decoding_vector(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    skip_ln2: bool = False,
    n_samples: int = 100,
) -> Dict:
    """
    Analyze decoding vector performance at all layers.

    Uses layer-appropriate decoding vectors:
    - embed: w = Σ_j E_j (sum of embeddings)
    - post_ln1: w = Σ_j LN(E_j) (sum of LN'd embeddings)
    - post_attn onwards: w = W_O @ W_V @ Σ_j LN(E_j) (with attention transforms)

    Returns:
        results: dict with correlation and projection statistics per layer
    """
    ctx = config.block_size
    vocab_size = config.vocab_size

    print(f"\n  Analyzing: {model_name}")

    # Compute layer-appropriate decoding vectors
    decoding_vectors = compute_decoding_vectors(model)

    # Collect projections across samples
    all_projections = {layer: [] for layer in LAYERS}

    for i in range(n_samples):
        tokens = torch.randint(0, vocab_size, (1, ctx), device=DEVICE)
        result = get_activations_with_decoding(
            model, tokens, decoding_vectors, skip_ln2=skip_ln2
        )

        for layer in LAYERS:
            all_projections[layer].append(result["projections"][layer][0].cpu().numpy())

    # Stack projections: [n_samples, ctx]
    for layer in LAYERS:
        all_projections[layer] = np.stack(all_projections[layer])

    # Compute statistics
    positions = np.arange(ctx)
    results = {"model_name": model_name, "n_samples": n_samples, "context": ctx}

    for layer in LAYERS:
        projs = all_projections[layer]  # [n_samples, ctx]

        # Mean projection at each position
        mean_proj = projs.mean(axis=0)  # [ctx]
        std_proj = projs.std(axis=0)

        # Overall correlation (flatten all samples)
        proj_flat = projs.flatten()
        pos_flat = np.tile(positions, n_samples)
        overall_corr, overall_p = pearsonr(proj_flat, pos_flat)

        # Per-sample correlations
        per_sample_corrs = [pearsonr(projs[i], positions)[0] for i in range(n_samples)]
        per_sample_mean = np.mean(per_sample_corrs)
        per_sample_std = np.std(per_sample_corrs)

        # Correlation of MEAN projection with position (this is what the paper reports)
        # When averaged across random tokens, the position signal emerges
        mean_corr, _ = pearsonr(mean_proj, positions)

        # R² from mean projection (how well does mean follow linear trend)
        slope, intercept = np.polyfit(positions, mean_proj, 1)
        predicted = slope * positions + intercept
        ss_res = np.sum((mean_proj - predicted) ** 2)
        ss_tot = np.sum((mean_proj - mean_proj.mean()) ** 2)
        r2_mean = 1 - ss_res / (ss_tot + 1e-8)

        results[layer] = {
            "overall_correlation": float(overall_corr),
            "overall_p_value": float(overall_p),
            "per_sample_corr_mean": float(per_sample_mean),
            "per_sample_corr_std": float(per_sample_std),
            "mean_projection_correlation": float(mean_corr),  # KEY METRIC
            "r2_mean_projection": float(r2_mean),
            "mean_projection": mean_proj.tolist(),
            "std_projection": std_proj.tolist(),
            "slope": float(slope),
        }

    return results


def create_layerwise_correlation_figure(all_results: Dict, output_path: Path):
    """Create figure showing decoding vector correlation at each layer.

    Shows both per-sample correlation and mean projection correlation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "NoPE + LayerNorm": "#1f77b4",
        "NoPE + BatchNorm2": "#ff7f0e",
        "NoPE + No LN2": "#2ca02c",
        "Baseline + PE": "#d62728",
    }
    markers = {
        "NoPE + LayerNorm": "o",
        "NoPE + BatchNorm2": "s",
        "NoPE + No LN2": "^",
        "Baseline + PE": "D",
    }

    x_positions = np.arange(len(LAYERS))
    x_labels = [LAYER_NAMES[l] for l in LAYERS]

    metrics = [
        ("overall_correlation", "Per-Sample Correlation"),
        ("mean_projection_correlation", "Mean Projection Correlation (Paper Metric)"),
    ]

    for row, (metric_key, metric_name) in enumerate(metrics):
        for col, init_type in enumerate(["random", "trained"]):
            ax = axes[row, col]

            for exp_name, data in all_results.items():
                if init_type not in data:
                    continue

                init_data = data[init_type]
                y_values = [init_data[layer].get(metric_key, 0) for layer in LAYERS]

                ax.plot(
                    x_positions,
                    y_values,
                    marker=markers[exp_name],
                    color=colors[exp_name],
                    linewidth=2,
                    markersize=8,
                    label=exp_name,
                )

            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Correlation", fontsize=12)
            init_label = "Random Init" if init_type == "random" else "Trained"
            ax.set_title(
                f"{metric_name}\n({init_label})", fontsize=12, fontweight="bold"
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
            ax.set_ylim(-0.5, 1.05)
            ax.grid(True, alpha=0.3)
            if col == 1:
                ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()

    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path.with_suffix('.pdf')}")


def create_projection_vs_position_figure(
    all_results: Dict, output_path: Path, layer: str = "post_attn"
):
    """Create figure showing mean projection vs position for a specific layer."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    exp_names = [e.name for e in EXPERIMENTS]

    for col, exp_name in enumerate(exp_names):
        for row, init_type in enumerate(["random", "trained"]):
            ax = axes[row, col]

            if exp_name not in all_results or init_type not in all_results[exp_name]:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            data = all_results[exp_name][init_type]
            mean_proj = np.array(data[layer]["mean_projection"])
            std_proj = np.array(data[layer]["std_projection"])
            positions = np.arange(len(mean_proj))
            corr = data[layer]["overall_correlation"]

            # Plot mean with std band
            ax.fill_between(
                positions,
                mean_proj - std_proj,
                mean_proj + std_proj,
                alpha=0.3,
                color="blue",
            )
            ax.plot(positions, mean_proj, color="blue", linewidth=1.5)

            # Add correlation annotation
            ax.text(
                0.05,
                0.95,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            if row == 0:
                ax.set_title(
                    exp_name.replace(" + ", "\n"), fontsize=11, fontweight="bold"
                )
            if col == 0:
                init_label = "Random Init" if init_type == "random" else "Trained"
                ax.set_ylabel(f"{init_label}\nProjection", fontsize=11)
            if row == 1:
                ax.set_xlabel("Position", fontsize=11)

            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Decoding Vector Projection vs Position ({LAYER_NAMES[layer]})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def create_comprehensive_figure(all_results: Dict, output_path: Path):
    """Create comprehensive figure for paper showing all key results."""
    fig = plt.figure(figsize=(16, 12))

    # Layout: 3 rows
    # Row 1: Correlation by layer (random vs trained)
    # Row 2: Projection vs position for post_attn (all 4 models, random)
    # Row 3: Projection vs position for post_attn (all 4 models, trained)

    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    colors = {
        "NoPE + LayerNorm": "#1f77b4",
        "NoPE + BatchNorm2": "#ff7f0e",
        "NoPE + No LN2": "#2ca02c",
        "Baseline + PE": "#d62728",
    }
    markers = {
        "NoPE + LayerNorm": "o",
        "NoPE + BatchNorm2": "s",
        "NoPE + No LN2": "^",
        "Baseline + PE": "D",
    }

    # Row 1: Correlation plots
    for col, init_type in enumerate(["random", "trained"]):
        ax = fig.add_subplot(gs[0, col * 2 : (col + 1) * 2])

        x_positions = np.arange(len(LAYERS))

        for exp_name, data in all_results.items():
            if init_type not in data:
                continue
            init_data = data[init_type]
            y_values = [init_data[layer]["overall_correlation"] for layer in LAYERS]
            ax.plot(
                x_positions,
                y_values,
                marker=markers[exp_name],
                color=colors[exp_name],
                linewidth=2,
                markersize=8,
                label=exp_name,
            )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Correlation", fontsize=11)
        title = "Random Init" if init_type == "random" else "Trained"
        ax.set_title(
            f"Decoding Vector Correlation ({title})", fontsize=12, fontweight="bold"
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [LAYER_NAMES[l] for l in LAYERS], rotation=45, ha="right", fontsize=9
        )
        ax.set_ylim(-0.3, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 1:
            ax.legend(loc="lower right", fontsize=8)

    # Rows 2-3: Projection vs position
    exp_names = [e.name for e in EXPERIMENTS]
    layer = "post_attn"

    for row_idx, init_type in enumerate(["random", "trained"]):
        for col, exp_name in enumerate(exp_names):
            ax = fig.add_subplot(gs[row_idx + 1, col])

            if exp_name not in all_results or init_type not in all_results[exp_name]:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            data = all_results[exp_name][init_type]
            mean_proj = np.array(data[layer]["mean_projection"])
            std_proj = np.array(data[layer]["std_projection"])
            positions = np.arange(len(mean_proj))
            corr = data[layer]["overall_correlation"]

            ax.fill_between(
                positions,
                mean_proj - std_proj,
                mean_proj + std_proj,
                alpha=0.3,
                color=colors[exp_name],
            )
            ax.plot(positions, mean_proj, color=colors[exp_name], linewidth=1.5)

            ax.text(
                0.05,
                0.95,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            if row_idx == 0:
                short_name = exp_name.split(" + ")[1] if " + " in exp_name else exp_name
                ax.set_title(short_name, fontsize=11, fontweight="bold")
            if col == 0:
                label = "Random" if init_type == "random" else "Trained"
                ax.set_ylabel(f"{label}\nProjection", fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Position", fontsize=10)

            ax.grid(True, alpha=0.3)

            # Subsample x-axis labels for readability
            ax.set_xticks([0, 128, 256, 384, 511])

    plt.suptitle(
        "Decoding Vector Analysis: w = W_V · Σ LN(E_j)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    print("=" * 70)
    print("DECODING VECTOR EXPERIMENTS: RANDOM vs TRAINED")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'#' * 60}")
        print(f"# {exp.name}")
        print(f"{'#' * 60}")

        all_results[exp.name] = {}

        # Random init
        print("\n  [Random Initialization]")
        random_model, config = create_random_model(exp)
        all_results[exp.name]["random"] = analyze_decoding_vector(
            random_model,
            config,
            f"{exp.name} (random)",
            skip_ln2=exp.skip_ln2,
            n_samples=100,
        )
        del random_model
        torch.cuda.empty_cache()

        # Trained model
        print("\n  [Trained Model]")
        try:
            trained_model, config = load_trained_model(exp)
            all_results[exp.name]["trained"] = analyze_decoding_vector(
                trained_model,
                config,
                f"{exp.name} (trained)",
                skip_ln2=exp.skip_ln2,
                n_samples=100,
            )
            del trained_model
            torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"  Skipping trained: {e}")

    # Save results
    output_path = RESULTS_DIR / "decoding_vector_owt_results.json"

    # Convert numpy arrays for JSON serialization
    serializable = {}
    for exp_name, exp_data in all_results.items():
        serializable[exp_name] = {}
        for init_type, init_data in exp_data.items():
            serializable[exp_name][init_type] = {}
            for key, value in init_data.items():
                if isinstance(value, dict):
                    serializable[exp_name][init_type][key] = {
                        k: (v if not isinstance(v, np.ndarray) else v.tolist())
                        for k, v in value.items()
                    }
                else:
                    serializable[exp_name][init_type][key] = value

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 110)
    print("SUMMARY: Decoding Vector Correlation by Layer")
    print("=" * 110)

    for init_type in ["random", "trained"]:
        print(f"\n{'─' * 110}")
        print(f"  {init_type.upper()} - Per-Sample Correlation")
        print(f"{'─' * 110}")

        header = f"{'Model':<20}"
        for layer in LAYERS:
            header += f" {LAYER_NAMES[layer]:>12}"
        print(header)
        print("-" * 110)

        for name in [e.name for e in EXPERIMENTS]:
            if init_type in all_results.get(name, {}):
                data = all_results[name][init_type]
                row = f"{name:<20}"
                for layer in LAYERS:
                    corr = data[layer]["overall_correlation"]
                    row += f" {corr:>12.3f}"
                print(row)

        print(f"\n{'─' * 110}")
        print(f"  {init_type.upper()} - Mean Projection Correlation (KEY METRIC)")
        print(f"{'─' * 110}")

        header = f"{'Model':<20}"
        for layer in LAYERS:
            header += f" {LAYER_NAMES[layer]:>12}"
        print(header)
        print("-" * 110)

        for name in [e.name for e in EXPERIMENTS]:
            if init_type in all_results.get(name, {}):
                data = all_results[name][init_type]
                row = f"{name:<20}"
                for layer in LAYERS:
                    corr = data[layer].get("mean_projection_correlation", 0)
                    row += f" {corr:>12.3f}"
                print(row)

        # Also show mean projection correlation (key metric)
        print(f"\n  {init_type.upper()} - Mean Projection Correlation (KEY METRIC)")
        print("-" * 100)

        for name in [e.name for e in EXPERIMENTS]:
            if init_type in all_results.get(name, {}):
                data = all_results[name][init_type]
                row = f"{name:<20}"
                for layer in LAYERS:
                    corr = data[layer].get("mean_projection_correlation", 0)
                    row += f" {corr:>12.3f}"
                print(row)

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    create_layerwise_correlation_figure(
        all_results, PLOTS_DIR / "decoding_vector_correlation"
    )

    create_projection_vs_position_figure(
        all_results,
        PLOTS_DIR / "decoding_vector_projection_post_attn",
        layer="post_attn",
    )

    create_comprehensive_figure(
        all_results, PLOTS_DIR / "decoding_vector_comprehensive"
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
