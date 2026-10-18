"""
Training Dynamics Analysis: How Positional Information Emerges During Training

This script analyzes how positional information develops during training by probing
checkpoints saved at regular intervals. We track:
1. Linear probe accuracy at each activation point across training
2. Attention pattern evolution (uniformity, entropy)
3. LayerNorm statistics (variance, kurtosis)
4. Decoding vector correlation emergence
5. Loss vs positional decodability correlation

Key Question: Does positional information emerge gradually or suddenly?
Does it correlate with training loss improvements?

Usage:
    python training_dynamics_analysis.py --gpu 7
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

# Parse args early for GPU setting
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=7, help="GPU to use")
parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples")
parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, kurtosis, skew
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig

# ─── Configuration ───────────────────────────────────────────────────────────

RESULTS_DIR = Path("results/training_dynamics")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint directories
LN_CKPT_DIR = Path("nanoGPT/out-nope-1layer-ln")
RMS_CKPT_DIR = Path("nanoGPT/out-nope-1layer-rms")


@dataclass
class DynamicsConfig:
    """Configuration for training dynamics analysis."""

    n_samples: int = 10000
    seq_len: int = 64
    batch_size: int = 256
    train_ratio: float = 0.8
    seed: int = 42

    # Checkpoints to analyze (every 250 steps from 250 to 5000)
    checkpoint_steps: List[int] = field(
        default_factory=lambda: list(range(250, 5001, 250))
    )


def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[GPT, dict]:
    """Load a model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Get model config from checkpoint
    model_args = checkpoint.get("model_args", {})

    # Create config
    gptconf = GPTConfig(
        n_layer=model_args.get("n_layer", 1),
        n_head=model_args.get("n_head", 1),
        n_embd=model_args.get("n_embd", 1024),
        block_size=model_args.get("block_size", 1024),
        bias=model_args.get("bias", False),
        vocab_size=model_args.get("vocab_size", 50304),
        dropout=0.0,
        use_positional_embedding=model_args.get("use_positional_embedding", False),
        norm_type=model_args.get("norm_type", "layernorm"),
    )

    # Create and load model
    model = GPT(gptconf)

    # Handle state dict with _orig_mod prefix (from torch.compile)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Extract training metadata
    meta = {
        "step": checkpoint.get("iter_num", 0),
        "train_loss": checkpoint.get("best_val_loss", None),
        "config": model_args,
    }

    return model, meta


def get_activations_nanogpt(
    model: GPT, tokens: torch.Tensor, batch_size: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Extract activations from nanoGPT model at key points.

    Returns dict with:
    - raw_embed: Token embeddings
    - post_ln1: After first LayerNorm
    - post_attn: Attention output
    - post_attn_residual: After attention residual
    - post_ln2: After second LayerNorm
    - post_mlp: MLP output
    - post_mlp_residual: Final block output
    - attention_weights: Attention patterns [batch, heads, seq, seq]
    """
    model.eval()
    n_samples, seq_len = tokens.shape
    d_model = model.config.n_embd

    activations = {
        "raw_embed": [],
        "post_ln1": [],
        "post_attn": [],
        "post_attn_residual": [],
        "post_ln2": [],
        "post_mlp": [],
        "post_mlp_residual": [],
        "attention_weights": [],
    }

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = tokens[i : i + batch_size]
            B, T = batch.shape

            # Token embeddings (no positional embedding)
            tok_emb = model.transformer.wte(batch)  # [B, T, d_model]
            activations["raw_embed"].append(tok_emb.cpu())

            x = tok_emb

            # Process through the single block
            block = model.transformer.h[0]

            # LayerNorm 1
            ln1_out = block.ln_1(x)
            activations["post_ln1"].append(ln1_out.cpu())

            # Attention (need to access internal computation for weights)
            # Get attention output and weights
            attn_module = block.attn

            # Compute Q, K, V
            B, T, C = ln1_out.shape
            q = attn_module.c_attn_q(ln1_out)
            k = attn_module.c_attn_k(ln1_out)
            v = attn_module.c_attn_v(ln1_out)

            n_head = attn_module.n_head
            head_dim = C // n_head

            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, n_head, head_dim).transpose(1, 2)

            # Compute attention weights
            att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=batch.device), diagonal=1
            ).bool()
            att = att.masked_fill(causal_mask, float("-inf"))

            att_weights = torch.softmax(att, dim=-1)
            activations["attention_weights"].append(att_weights.cpu())

            # Attention output
            y = att_weights @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            attn_out = attn_module.c_proj(y)
            activations["post_attn"].append(attn_out.cpu())

            # Residual after attention
            x = x + attn_out
            activations["post_attn_residual"].append(x.cpu())

            # LayerNorm 2
            ln2_out = block.ln_2(x)
            activations["post_ln2"].append(ln2_out.cpu())

            # MLP
            mlp_out = block.mlp(ln2_out)
            activations["post_mlp"].append(mlp_out.cpu())

            # Final residual
            x = x + mlp_out
            activations["post_mlp_residual"].append(x.cpu())

    # Concatenate batches
    for key in activations:
        activations[key] = torch.cat(activations[key], dim=0)

    return activations


def train_linear_probe(
    activations: torch.Tensor,  # [n_samples, seq_len, d_model]
    positions: torch.Tensor,  # [n_samples, seq_len]
    train_ratio: float = 0.8,
) -> Dict[str, float]:
    """Train linear probe to predict position from activations."""
    n_samples, seq_len, d_model = activations.shape

    # Flatten
    X = activations.reshape(-1, d_model).numpy()
    y = positions.reshape(-1).numpy()

    # Split
    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Train Ridge regression
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)

    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "corr": pearsonr(y_test, y_pred)[0],
    }


def compute_attention_metrics(att_weights: torch.Tensor) -> Dict[str, float]:
    """
    Compute attention pattern metrics.

    att_weights: [n_samples, n_heads, seq_len, seq_len]
    """
    n_samples, n_heads, seq_len, _ = att_weights.shape

    metrics = {}

    # Average over samples and heads
    mean_att = att_weights.mean(dim=(0, 1))  # [seq_len, seq_len]

    # Uniformity score: how close is attention to uniform distribution?
    # For position i, uniform would be 1/(i+1) for all positions j <= i
    uniformity_scores = []
    for i in range(seq_len):
        uniform = torch.ones(i + 1) / (i + 1)
        actual = mean_att[i, : i + 1]
        # KL divergence from uniform
        kl = (actual * (actual / uniform).log()).sum().item()
        uniformity_scores.append(kl)

    metrics["attention_kl_from_uniform_mean"] = np.mean(uniformity_scores)
    metrics["attention_kl_from_uniform_std"] = np.std(uniformity_scores)

    # Entropy of attention distribution
    entropies = []
    for i in range(1, seq_len):  # Skip position 0 (only one option)
        att_dist = mean_att[i, : i + 1]
        entropy = -(att_dist * att_dist.log()).sum().item()
        max_entropy = np.log(i + 1)
        entropies.append(entropy / max_entropy)  # Normalized entropy

    metrics["attention_entropy_normalized_mean"] = np.mean(entropies)
    metrics["attention_entropy_normalized_std"] = np.std(entropies)

    # Diagonal bias (does attention focus on recent tokens?)
    diag_weights = []
    for i in range(seq_len):
        diag_weights.append(mean_att[i, i].item())
    metrics["diagonal_attention_mean"] = np.mean(diag_weights)

    return metrics


def compute_activation_statistics(
    activations: Dict[str, torch.Tensor],
    seq_len: int,
) -> Dict[str, Dict[str, List[float]]]:
    """Compute statistics of activations by position."""
    stats = {}

    for name, act in activations.items():
        if name == "attention_weights":
            continue

        # act: [n_samples, seq_len, d_model]
        stats[name] = {
            "variance_by_pos": [],
            "mean_norm_by_pos": [],
            "kurtosis_by_pos": [],
            "skewness_by_pos": [],
        }

        for pos in range(seq_len):
            pos_act = act[:, pos, :].numpy()  # [n_samples, d_model]

            # Variance across samples (average over dimensions)
            var = np.var(pos_act, axis=0).mean()
            stats[name]["variance_by_pos"].append(var)

            # Mean L2 norm
            norms = np.linalg.norm(pos_act, axis=1).mean()
            stats[name]["mean_norm_by_pos"].append(norms)

            # Kurtosis (average over dimensions)
            kurt = kurtosis(pos_act, axis=0).mean()
            stats[name]["kurtosis_by_pos"].append(kurt)

            # Skewness
            sk = skew(pos_act, axis=0).mean()
            stats[name]["skewness_by_pos"].append(sk)

    return stats


def analyze_checkpoint(
    ckpt_path: str,
    cfg: DynamicsConfig,
    tokens: torch.Tensor,
    positions: torch.Tensor,
) -> Dict:
    """Analyze a single checkpoint."""
    print(f"  Loading {ckpt_path}...")
    model, meta = load_checkpoint(ckpt_path, device)

    print(f"  Extracting activations...")
    activations = get_activations_nanogpt(model, tokens, cfg.batch_size)

    results = {
        "step": meta["step"],
        "train_loss": meta.get("train_loss"),
        "probe_results": {},
        "attention_metrics": {},
        "activation_stats": {},
    }

    # Train probes at each activation point
    print(f"  Training probes...")
    for name, act in activations.items():
        if name == "attention_weights":
            continue
        results["probe_results"][name] = train_linear_probe(
            act, positions, cfg.train_ratio
        )

    # Attention metrics
    print(f"  Computing attention metrics...")
    results["attention_metrics"] = compute_attention_metrics(
        activations["attention_weights"]
    )

    # Activation statistics
    print(f"  Computing activation statistics...")
    results["activation_stats"] = compute_activation_statistics(
        activations, cfg.seq_len
    )

    # Clean up
    del model, activations
    torch.cuda.empty_cache()

    return results


def analyze_model_checkpoints(
    ckpt_dir: Path,
    cfg: DynamicsConfig,
    model_name: str,
) -> List[Dict]:
    """Analyze all checkpoints for a model."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing {model_name} checkpoints")
    print(f"{'=' * 60}")

    # Generate random tokens once
    torch.manual_seed(cfg.seed)
    vocab_size = 50304  # Default for nanoGPT
    tokens = torch.randint(0, vocab_size, (cfg.n_samples, cfg.seq_len), device=device)
    positions = (
        torch.arange(cfg.seq_len, device=device).unsqueeze(0).expand(cfg.n_samples, -1)
    )

    results = []

    for step in tqdm(cfg.checkpoint_steps, desc=f"Analyzing {model_name}"):
        ckpt_path = ckpt_dir / f"ckpt_{step:05d}.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint {ckpt_path} not found, skipping...")
            continue

        try:
            result = analyze_checkpoint(str(ckpt_path), cfg, tokens, positions)
            results.append(result)
        except Exception as e:
            print(f"  Error analyzing {ckpt_path}: {e}")
            continue

    return results


def create_training_dynamics_plots(
    ln_results: List[Dict],
    rms_results: List[Dict],
    output_dir: Path,
):
    """Create visualizations of training dynamics."""
    print("\nCreating plots...")

    # Extract data for plotting
    ln_steps = [r["step"] for r in ln_results]
    rms_steps = [r["step"] for r in rms_results]

    # 1. Probe R² across training for different activation points
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[
            "Raw Embed",
            "Post-LN1",
            "Post-Attn",
            "Post-Attn-Resid",
            "Post-LN2",
            "Post-MLP",
            "Post-MLP-Resid",
            "Post-Final-LN",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    activation_points = [
        "raw_embed",
        "post_ln1",
        "post_attn",
        "post_attn_residual",
        "post_ln2",
        "post_mlp",
        "post_mlp_residual",
        "post_mlp_residual",
    ]

    for idx, act_name in enumerate(activation_points):
        row = idx // 4 + 1
        col = idx % 4 + 1

        # LayerNorm
        ln_r2 = [r["probe_results"].get(act_name, {}).get("r2", 0) for r in ln_results]
        fig.add_trace(
            go.Scatter(
                x=ln_steps,
                y=ln_r2,
                name="LayerNorm",
                mode="lines+markers",
                line=dict(color="blue"),
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        # RMSNorm
        rms_r2 = [
            r["probe_results"].get(act_name, {}).get("r2", 0) for r in rms_results
        ]
        fig.add_trace(
            go.Scatter(
                x=rms_steps,
                y=rms_r2,
                name="RMSNorm",
                mode="lines+markers",
                line=dict(color="red"),
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text="Training Step" if row == 2 else "", row=row, col=col
        )
        fig.update_yaxes(title_text="R²" if col == 1 else "", row=row, col=col)

    fig.update_layout(
        title="Position Probe R² Across Training at Different Activation Points",
        height=600,
        width=1400,
        template="plotly_white",
    )
    fig.write_image(str(output_dir / "training_dynamics_probe_r2.png"), scale=2)
    fig.write_image(str(output_dir / "training_dynamics_probe_r2.pdf"))

    # 2. Attention uniformity across training
    fig2 = go.Figure()

    ln_uniformity = [
        r["attention_metrics"]["attention_kl_from_uniform_mean"] for r in ln_results
    ]
    rms_uniformity = [
        r["attention_metrics"]["attention_kl_from_uniform_mean"] for r in rms_results
    ]

    fig2.add_trace(
        go.Scatter(
            x=ln_steps,
            y=ln_uniformity,
            name="LayerNorm",
            mode="lines+markers",
            line=dict(color="blue"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=rms_steps,
            y=rms_uniformity,
            name="RMSNorm",
            mode="lines+markers",
            line=dict(color="red"),
        )
    )

    fig2.update_layout(
        title="Attention KL Divergence from Uniform Distribution Across Training",
        xaxis_title="Training Step",
        yaxis_title="KL Divergence (lower = more uniform)",
        template="plotly_white",
        height=400,
        width=800,
    )
    fig2.write_image(
        str(output_dir / "training_dynamics_attention_uniformity.png"), scale=2
    )
    fig2.write_image(str(output_dir / "training_dynamics_attention_uniformity.pdf"))

    # 3. Summary plot: Post-LN1 R² vs Training Loss
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    ln_post_ln1_r2 = [r["probe_results"]["post_ln1"]["r2"] for r in ln_results]
    ln_losses = [r.get("train_loss", None) for r in ln_results]

    fig3.add_trace(
        go.Scatter(
            x=ln_steps,
            y=ln_post_ln1_r2,
            name="Post-LN1 R²",
            mode="lines+markers",
            line=dict(color="blue"),
        ),
        secondary_y=False,
    )

    if any(l is not None for l in ln_losses):
        ln_losses_clean = [l for l in ln_losses if l is not None]
        ln_steps_clean = [s for s, l in zip(ln_steps, ln_losses) if l is not None]
        fig3.add_trace(
            go.Scatter(
                x=ln_steps_clean,
                y=ln_losses_clean,
                name="Val Loss",
                mode="lines+markers",
                line=dict(color="orange", dash="dash"),
            ),
            secondary_y=True,
        )

    fig3.update_layout(
        title="Position Probe R² vs Training Progress (LayerNorm Model)",
        template="plotly_white",
        height=400,
        width=700,
    )
    fig3.update_xaxes(title_text="Training Step")
    fig3.update_yaxes(title_text="R²", secondary_y=False)
    fig3.update_yaxes(title_text="Validation Loss", secondary_y=True)

    fig3.write_image(str(output_dir / "training_dynamics_r2_vs_loss.png"), scale=2)
    fig3.write_image(str(output_dir / "training_dynamics_r2_vs_loss.pdf"))

    # 4. Variance by position evolution
    fig4 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Step 250", "Step 1000", "Step 2500", "Step 5000"],
    )

    steps_to_plot = [250, 1000, 2500, 5000]

    for idx, step in enumerate(steps_to_plot):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Find result for this step
        ln_result = next((r for r in ln_results if r["step"] == step), None)
        if ln_result:
            var_by_pos = ln_result["activation_stats"]["post_ln1"]["variance_by_pos"]
            positions = list(range(len(var_by_pos)))
            fig4.add_trace(
                go.Scatter(
                    x=positions,
                    y=var_by_pos,
                    name=f"Step {step}",
                    mode="lines",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )

    fig4.update_layout(
        title="Post-LN1 Variance by Position at Different Training Steps (LayerNorm)",
        template="plotly_white",
        height=500,
        width=700,
    )
    fig4.write_image(
        str(output_dir / "training_dynamics_variance_evolution.png"), scale=2
    )
    fig4.write_image(str(output_dir / "training_dynamics_variance_evolution.pdf"))

    print("  Plots saved!")


def main():
    """Main entry point."""
    setup_dirs()

    cfg = DynamicsConfig(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
    )

    print(f"Configuration:")
    print(f"  n_samples: {cfg.n_samples}")
    print(f"  seq_len: {cfg.seq_len}")
    print(f"  checkpoints: {cfg.checkpoint_steps[0]} to {cfg.checkpoint_steps[-1]}")
    print(f"  device: {device}")

    # Analyze LayerNorm model
    ln_results = analyze_model_checkpoints(LN_CKPT_DIR, cfg, "LayerNorm")

    # Analyze RMSNorm model
    rms_results = analyze_model_checkpoints(RMS_CKPT_DIR, cfg, "RMSNorm")

    # Save raw results
    results = {
        "layernorm": ln_results,
        "rmsnorm": rms_results,
        "config": {
            "n_samples": cfg.n_samples,
            "seq_len": cfg.seq_len,
            "checkpoint_steps": cfg.checkpoint_steps,
        },
    }

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    results_json = convert_for_json(results)

    with open(RESULTS_DIR / "training_dynamics_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Create plots
    create_training_dynamics_plots(ln_results, rms_results, PLOTS_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING DYNAMICS SUMMARY")
    print("=" * 60)

    if ln_results:
        print("\nLayerNorm Model:")
        first_r2 = ln_results[0]["probe_results"]["post_ln1"]["r2"]
        last_r2 = ln_results[-1]["probe_results"]["post_ln1"]["r2"]
        print(f"  Post-LN1 R² at step {ln_results[0]['step']}: {first_r2:.4f}")
        print(f"  Post-LN1 R² at step {ln_results[-1]['step']}: {last_r2:.4f}")
        print(f"  Change: {last_r2 - first_r2:+.4f}")

    if rms_results:
        print("\nRMSNorm Model:")
        first_r2 = rms_results[0]["probe_results"]["post_ln1"]["r2"]
        last_r2 = rms_results[-1]["probe_results"]["post_ln1"]["r2"]
        print(f"  Post-LN1 R² at step {rms_results[0]['step']}: {first_r2:.4f}")
        print(f"  Post-LN1 R² at step {rms_results[-1]['step']}: {last_r2:.4f}")
        print(f"  Change: {last_r2 - first_r2:+.4f}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
