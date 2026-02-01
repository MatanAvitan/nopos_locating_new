"""
Probe-based analysis: Trace positional information through the network.

Train linear probes at each activation point to measure how much positional
information is present. This tests whether LayerNorm actually destroys
positional information or preserves it.

Activation points:
1. Raw embeddings (no positional info expected)
2. Post-LN1 (after first LayerNorm, before attention)
3. Post-attention (after attention + residual)
4. Post-LN2 (after second LayerNorm, before MLP)
5. Post-MLP (after MLP + residual)
6. Post-final-LN (after final LayerNorm)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    config.log_attention_stats = False
    model = GPT(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def extract_activations(model, input_ids, device="cuda"):
    """
    Extract activations at each stage of the network.

    Returns dict with activations at each point.
    """
    model.eval()
    activations = {}

    with torch.no_grad():
        # 1. Raw embeddings
        tok_emb = model.transformer.wte(input_ids)
        activations["1_raw_embed"] = tok_emb.cpu()

        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[0]

        # 2. Post-LN1 (before attention)
        x_ln1 = block.ln_1(x)
        activations["2_post_ln1"] = x_ln1.cpu()

        # 3. Post-attention (after attention + residual)
        attn_out = block.attn(x_ln1)
        x_attn = x + attn_out
        activations["3_post_attn"] = x_attn.cpu()

        # 4. Post-LN2 (before MLP)
        x_ln2 = block.ln_2(x_attn)
        activations["4_post_ln2"] = x_ln2.cpu()

        # 5. Post-MLP (after MLP + residual)
        mlp_out = block.mlp(x_ln2)
        x_mlp = x_attn + mlp_out
        activations["5_post_mlp"] = x_mlp.cpu()

        # 6. Post-final-LN
        x_final = model.transformer.ln_f(x_mlp)
        activations["6_post_final_ln"] = x_final.cpu()

    return activations


def train_linear_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """
    Train a Ridge regression probe to predict position from activations.

    Returns: train_r2, test_r2, train_corr, test_corr, train_mae, test_mae
    """
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)

    y_train_pred = probe.predict(X_train)
    y_test_pred = probe.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_corr, _ = pearsonr(y_train, y_train_pred)
    test_corr, _ = pearsonr(y_test, y_test_pred)

    train_mae = np.abs(y_train - y_train_pred).mean()
    test_mae = np.abs(y_test - y_test_pred).mean()

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_corr": train_corr,
        "test_corr": test_corr,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "probe": probe,
    }


def run_probe_analysis(
    checkpoint_path, n_train=5000, n_test=1000, seq_len=256, device="cuda"
):
    """Run complete probe analysis at all activation points."""

    print(f"Loading model from {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)

    norm_type = model.config.norm_type
    val_loss = checkpoint.get("best_val_loss", float("nan"))
    if torch.is_tensor(val_loss):
        val_loss = val_loss.cpu().item()
    perplexity = np.exp(val_loss)

    print(f"\nModel: {norm_type.upper()}")
    print(f"Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

    vocab_size = model.config.vocab_size

    # Generate random sequences for train and test
    print(
        f"\nGenerating {n_train} train and {n_test} test sequences of length {seq_len}..."
    )

    torch.manual_seed(42)
    train_ids = torch.randint(0, vocab_size, (n_train, seq_len), device=device)
    test_ids = torch.randint(0, vocab_size, (n_test, seq_len), device=device)

    # Extract activations
    print("Extracting activations...")

    # Process in batches to avoid OOM
    batch_size = 100

    train_activations = {
        k: []
        for k in [
            "1_raw_embed",
            "2_post_ln1",
            "3_post_attn",
            "4_post_ln2",
            "5_post_mlp",
            "6_post_final_ln",
        ]
    }
    test_activations = {k: [] for k in train_activations.keys()}

    for i in range(0, n_train, batch_size):
        batch = train_ids[i : i + batch_size]
        acts = extract_activations(model, batch, device)
        for k, v in acts.items():
            train_activations[k].append(v)

    for i in range(0, n_test, batch_size):
        batch = test_ids[i : i + batch_size]
        acts = extract_activations(model, batch, device)
        for k, v in acts.items():
            test_activations[k].append(v)

    # Concatenate
    for k in train_activations:
        train_activations[k] = torch.cat(train_activations[k], dim=0)
        test_activations[k] = torch.cat(test_activations[k], dim=0)

    # Create position labels
    # Shape: [n_samples, seq_len] -> flatten to [n_samples * seq_len]
    train_positions = (
        torch.arange(seq_len).unsqueeze(0).expand(n_train, -1).flatten().numpy()
    )
    test_positions = (
        torch.arange(seq_len).unsqueeze(0).expand(n_test, -1).flatten().numpy()
    )

    print(f"\nTraining probes at each activation point...")
    print("=" * 80)

    results = {}

    for name, train_act in train_activations.items():
        test_act = test_activations[name]

        # Reshape: [n_samples, seq_len, n_embd] -> [n_samples * seq_len, n_embd]
        X_train = train_act.reshape(-1, train_act.shape[-1]).numpy()
        X_test = test_act.reshape(-1, test_act.shape[-1]).numpy()

        # Train probe
        result = train_linear_probe(X_train, train_positions, X_test, test_positions)
        results[name] = result

        print(f"\n{name}:")
        print(
            f"  Train: R²={result['train_r2']:.4f}, r={result['train_corr']:.4f}, MAE={result['train_mae']:.2f}"
        )
        print(
            f"  Test:  R²={result['test_r2']:.4f}, r={result['test_corr']:.4f}, MAE={result['test_mae']:.2f}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Activation Point':<20} {'Test R²':>10} {'Test r':>10} {'Test MAE':>10}")
    print("-" * 55)
    for name, result in results.items():
        print(
            f"{name:<20} {result['test_r2']:>10.4f} {result['test_corr']:>10.4f} {result['test_mae']:>10.2f}"
        )

    return results, norm_type


def plot_results(results_ln, results_rms, output_path):
    """Plot comparison of probe results for LayerNorm vs RMSNorm."""

    stages = list(results_ln.keys())
    stage_labels = [s.split("_", 1)[1] for s in stages]

    ln_r2 = [results_ln[s]["test_r2"] for s in stages]
    rms_r2 = [results_rms[s]["test_r2"] for s in stages]

    ln_corr = [results_ln[s]["test_corr"] for s in stages]
    rms_corr = [results_rms[s]["test_corr"] for s in stages]

    ln_mae = [results_ln[s]["test_mae"] for s in stages]
    rms_mae = [results_rms[s]["test_mae"] for s in stages]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(stages))
    width = 0.35

    # R² plot
    axes[0].bar(x - width / 2, ln_r2, width, label="LayerNorm", color="blue", alpha=0.7)
    axes[0].bar(
        x + width / 2, rms_r2, width, label="RMSNorm", color="orange", alpha=0.7
    )
    axes[0].set_ylabel("Test R²")
    axes[0].set_title("Position Probe R² at Each Stage")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stage_labels, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # Correlation plot
    axes[1].bar(
        x - width / 2, ln_corr, width, label="LayerNorm", color="blue", alpha=0.7
    )
    axes[1].bar(
        x + width / 2, rms_corr, width, label="RMSNorm", color="orange", alpha=0.7
    )
    axes[1].set_ylabel("Test Correlation (r)")
    axes[1].set_title("Position Probe Correlation at Each Stage")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stage_labels, rotation=45, ha="right")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    # MAE plot
    axes[2].bar(
        x - width / 2, ln_mae, width, label="LayerNorm", color="blue", alpha=0.7
    )
    axes[2].bar(
        x + width / 2, rms_mae, width, label="RMSNorm", color="orange", alpha=0.7
    )
    axes[2].set_ylabel("Test MAE (positions)")
    axes[2].set_title("Position Probe MAE at Each Stage")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(stage_labels, rotation=45, ha="right")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_ln",
        type=str,
        default="nanoGPT/out-nope-1layer-ln/ckpt.pt",
        help="Path to LayerNorm model checkpoint",
    )
    parser.add_argument(
        "--checkpoint_rms",
        type=str,
        default="nanoGPT/out-nope-1layer-rms/ckpt.pt",
        help="Path to RMSNorm model checkpoint",
    )
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    # Run analysis on LayerNorm model
    print("\n" + "=" * 80)
    print("LAYERNORM MODEL")
    print("=" * 80)
    results_ln, _ = run_probe_analysis(
        args.checkpoint_ln, n_train=args.n_train, n_test=args.n_test, device=args.device
    )

    # Run analysis on RMSNorm model
    print("\n" + "=" * 80)
    print("RMSNORM MODEL")
    print("=" * 80)
    results_rms, _ = run_probe_analysis(
        args.checkpoint_rms,
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
    )

    # Plot comparison
    plot_path = Path(args.output_dir) / "probe_positional_info.png"
    plot_results(results_ln, results_rms, plot_path)

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: LayerNorm vs RMSNorm")
    print("=" * 80)
    print(f"{'Stage':<20} {'LN R²':>8} {'RMS R²':>8} {'LN MAE':>8} {'RMS MAE':>8}")
    print("-" * 60)
    for stage in results_ln.keys():
        ln = results_ln[stage]
        rms = results_rms[stage]
        print(
            f"{stage:<20} {ln['test_r2']:>8.4f} {rms['test_r2']:>8.4f} {ln['test_mae']:>8.2f} {rms['test_mae']:>8.2f}"
        )
