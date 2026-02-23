"""
Write Bottleneck Curves for ICML 2026 Paper

Generates publication-quality plots showing:
- Retention intervention: o_i <- P_r o_i (project onto top-r singular directions)
- Ablation intervention: o_i <- (I - P_r) o_i (remove top-r directions)

Computes r_95: minimal rank achieving 95% of baseline R².

Usage:
    python analysis_scripts/generate_write_bottleneck_plot.py
"""

import os
import sys
import json
import argparse
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style settings
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

# Colors (colorblind-friendly)
COLOR_R0 = "#0072B2"  # Blue (FULL-12H)
COLOR_R2 = "#D55E00"  # Vermillion (ATTN2-1H)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOS_TOKEN_ID = 50256


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TwoLayerMechanismConfig(**filtered)
    model = TwoLayerMechanismModel(config)

    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            unwrapped_state_dict[k[len("_orig_mod.") :]] = v
        else:
            unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    model.eval()

    return model, config


def load_owt_data(data_dir: str = "nanoGPT/data/openwebtext"):
    """Load OpenWebText validation data."""
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
    force_bos: bool = False,
    bos_token_id: int = BOS_TOKEN_ID,
):
    """Get a batch of sequences, optionally forcing BOS at position 0."""
    if force_bos:
        tokens_needed = block_size - 1
        ix = torch.randint(len(data) - tokens_needed, (batch_size,))
        sequences = []
        for i in ix:
            after_bos = data[i : i + tokens_needed].astype(np.int64)
            seq = np.concatenate([[bos_token_id], after_bos])
            sequences.append(torch.from_numpy(seq))
        x = torch.stack(sequences)
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
    return x.to(device)


def get_block2_write_map_svd(model: TwoLayerMechanismModel):
    """
    Compute SVD of B = W_O @ W_V from Block 2 attention.
    Returns U, S, Vt matrices.
    """
    attn = model.block2.attn
    c_attn_weight = attn.c_attn.weight  # [3*d_model, d_model]
    d_model = c_attn_weight.shape[1]

    # Extract W_V (last third of c_attn)
    W_V = c_attn_weight[2 * d_model :, :]  # [d_model, d_model]
    W_O = attn.c_proj.weight  # [d_model, d_model]
    B = W_O @ W_V  # [d_model, d_model]

    # Compute SVD
    U, S, Vt = torch.linalg.svd(B, full_matrices=True)

    return U, S, Vt


def forward_with_write_intervention(
    model: TwoLayerMechanismModel,
    tokens: torch.Tensor,
    U: torch.Tensor,
    rank: int,
    intervention_type: str = "retention",
    basis_indices: Optional[List[int]] = None,
):
    """
    Forward pass with intervention on Block 2 attention output.

    Args:
        model: The model
        tokens: Input tokens [B, T]
        U: Left singular vectors of B [d, d]
        rank: Rank for projection
        intervention_type: "retention" (keep top-r) or "ablation" (remove top-r)

    Returns:
        predictions: Position predictions [B, T]
    """
    B, T = tokens.shape
    d_model = model.config.n_embd
    n_head = model.config.n_head
    head_dim = d_model // n_head

    with torch.no_grad():
        # Embedding
        e = model.wte(tokens)

        # Block 1 forward
        ln1_out = model.block1.ln_1(e)
        attn1 = model.block1.attn

        qkv1 = attn1.c_attn(ln1_out)
        q1, k1, v1 = qkv1.split(d_model, dim=2)
        q1 = q1.view(B, T, n_head, head_dim).transpose(1, 2)
        k1 = k1.view(B, T, n_head, head_dim).transpose(1, 2)
        v1 = v1.view(B, T, n_head, head_dim).transpose(1, 2)

        att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        causal_mask = torch.triu(
            torch.ones(T, T, device=tokens.device), diagonal=1
        ).bool()
        att1 = att1.masked_fill(causal_mask, float("-inf"))
        att1 = F.softmax(att1, dim=-1)
        y1 = (att1 @ v1).transpose(1, 2).contiguous().view(B, T, d_model)
        attn_out1 = attn1.c_proj(y1)

        r1_attn = e + attn_out1
        ln2_out_b1 = model.block1.ln_2(r1_attn)
        mlp_out1 = model.block1.mlp(ln2_out_b1)
        r1 = r1_attn + mlp_out1

        # Block 2 forward with intervention
        ln1_out_b2 = model.block2.ln_1(r1)
        attn2 = model.block2.attn

        qkv2 = attn2.c_attn(ln1_out_b2)
        q2, k2, v2 = qkv2.split(d_model, dim=2)
        q2 = q2.view(B, T, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(B, T, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(B, T, n_head, head_dim).transpose(1, 2)

        att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
        att2 = att2.masked_fill(causal_mask, float("-inf"))
        att2 = F.softmax(att2, dim=-1)
        y2 = (att2 @ v2).transpose(1, 2).contiguous().view(B, T, d_model)
        attn_out2 = attn2.c_proj(y2)  # This is o_i before residual

        # Apply write subspace intervention on attn_out2
        # P_r = U[:, idx] @ U[:, idx].T
        if basis_indices is None:
            basis_indices = list(range(rank))
        U_r = U[:, basis_indices]  # [d, r]

        if intervention_type == "retention":
            # o_i <- P_r @ o_i = U_r @ U_r.T @ o_i
            attn_out2_intervened = attn_out2 @ U_r @ U_r.T
        elif intervention_type == "ablation":
            # o_i <- (I - P_r) @ o_i
            attn_out2_intervened = attn_out2 - attn_out2 @ U_r @ U_r.T
        else:
            attn_out2_intervened = attn_out2

        r2_attn = r1 + attn_out2_intervened
        ln2_out_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_out_b2)
        r2 = r2_attn + mlp_out2

        # Final layer norm and prediction head
        final = model.ln_f(r2)
        pred = model.pos_head(final).squeeze(-1)  # [B, T]

    return pred


def compute_r2_at_rank(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    U: torch.Tensor,
    rank: int,
    intervention_type: str,
    n_batches: int = 50,
    batch_size: int = 32,
    block_size: int = 128,
    basis_indices: Optional[List[int]] = None,
):
    """Compute R² for a given rank intervention."""
    all_preds = []
    all_positions = []

    for _ in range(n_batches):
        tokens = get_batch(data, batch_size, block_size, DEVICE, force_bos=True)
        preds = forward_with_write_intervention(
            model,
            tokens,
            U,
            rank,
            intervention_type,
            basis_indices=basis_indices,
        )

        positions = (
            torch.arange(block_size, device=DEVICE)
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        all_preds.append(preds.cpu())
        all_positions.append(positions.cpu())

    all_preds = torch.cat(all_preds, dim=0).flatten().numpy()
    all_positions = torch.cat(all_positions, dim=0).flatten().numpy()

    # Compute R²
    r = float(np.corrcoef(all_positions, all_preds)[0, 1])
    r2 = r * r

    return r2


def run_write_bottleneck_experiment(
    model: TwoLayerMechanismModel,
    data: np.ndarray,
    model_name: str,
    ranks: list,
    n_batches: int = 50,
    batch_size: int = 32,
    rank1_override_indices: Optional[List[int]] = None,
):
    """Run full write bottleneck experiment for a model."""
    print(f"\n{'=' * 60}")
    print(f"Running write bottleneck experiment for {model_name}")
    print(f"{'=' * 60}")

    # Get SVD of write map
    U, S, Vt = get_block2_write_map_svd(model)
    block_size = model.config.block_size

    # Compute baseline R² (no intervention)
    baseline_r2 = compute_r2_at_rank(
        model,
        data,
        U,
        768,
        "retention",  # Full rank = no intervention
        n_batches,
        batch_size,
        block_size,
    )
    print(f"Baseline R²: {baseline_r2:.4f}")

    # Compute R² for each rank
    retention_r2s = []
    ablation_r2s = []

    for rank in tqdm(ranks, desc=f"{model_name} ranks"):
        basis_indices = None
        if rank == 1 and rank1_override_indices is not None:
            basis_indices = rank1_override_indices
        ret_r2 = compute_r2_at_rank(
            model,
            data,
            U,
            rank,
            "retention",
            n_batches,
            batch_size,
            block_size,
            basis_indices=basis_indices,
        )
        abl_r2 = compute_r2_at_rank(
            model,
            data,
            U,
            rank,
            "ablation",
            n_batches,
            batch_size,
            block_size,
            basis_indices=basis_indices,
        )
        retention_r2s.append(ret_r2)
        ablation_r2s.append(abl_r2)

    # Find r_95
    r_95 = None
    threshold_95 = 0.95 * baseline_r2
    for i, r2 in enumerate(retention_r2s):
        if r2 >= threshold_95:
            r_95 = ranks[i]
            break

    print(f"r_95 (95% of baseline): {r_95}")

    results = {
        "baseline_r2": float(baseline_r2),
        "ranks": ranks,
        "retention_r2s": [float(x) for x in retention_r2s],
        "ablation_r2s": [float(x) for x in ablation_r2s],
        "r_95": r_95,
        "singular_values": S.detach().cpu().numpy().tolist(),
    }

    if rank1_override_indices is not None:
        rank1_retention = compute_r2_at_rank(
            model,
            data,
            U,
            1,
            "retention",
            n_batches,
            batch_size,
            block_size,
            basis_indices=rank1_override_indices,
        )
        rank1_ablation = compute_r2_at_rank(
            model,
            data,
            U,
            1,
            "ablation",
            n_batches,
            batch_size,
            block_size,
            basis_indices=rank1_override_indices,
        )
        results["rank1_override_indices"] = rank1_override_indices
        results["rank1_override_retention_r2"] = float(rank1_retention)
        results["rank1_override_ablation_r2"] = float(rank1_ablation)

    return results


def create_main_text_plot(results_r0: dict, results_r2: dict, save_path: str):
    """Create the main text write bottleneck figure."""
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))

    ranks = results_r0["ranks"]

    # FULL-12H curves (no markers for readability)
    ax.plot(
        ranks,
        results_r0["retention_r2s"],
        "-",
        color=COLOR_R0,
        label="FULL-12H retention",
        linewidth=1.5,
    )
    ax.plot(
        ranks,
        results_r0["ablation_r2s"],
        "--",
        color=COLOR_R0,
        label="FULL-12H ablation",
        linewidth=1.5,
        alpha=0.7,
    )

    # ATTN2-1H curves (no markers for readability)
    ax.plot(
        ranks,
        results_r2["retention_r2s"],
        "-",
        color=COLOR_R2,
        label="ATTN2-1H retention",
        linewidth=1.5,
    )
    ax.plot(
        ranks,
        results_r2["ablation_r2s"],
        "--",
        color=COLOR_R2,
        label="ATTN2-1H ablation",
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_xlabel("Rank $r$")
    ax.set_ylabel("Position $R^2$")
    ax.set_xlim(0, max(ranks))
    ax.set_ylim(0, 1.05)

    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved main text figure to {save_path}")


def create_appendix_plot(all_results: dict, save_path: str):
    """Create the appendix figure with R0 and R2 (12-head only)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    models = ["R0_12head", "R2_12head"]
    titles = ["FULL-12H", "ATTN2-1H"]
    colors = [COLOR_R0, COLOR_R2]

    for i, (model_key, title, color) in enumerate(zip(models, titles, colors)):
        ax = axes[i]

        if model_key not in all_results:
            ax.text(
                0.5,
                0.5,
                f"{title}\n(not available)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(title)
            continue

        results = all_results[model_key]
        ranks = results["ranks"]

        # No markers for readability
        ax.plot(
            ranks,
            results["retention_r2s"],
            "-",
            color=color,
            label="Retention",
            linewidth=1.5,
        )
        ax.plot(
            ranks,
            results["ablation_r2s"],
            "--",
            color=color,
            label="Ablation",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_xlabel("Rank $r$")
        ax.set_ylabel("Position $R^2$")
        ax.set_title(title)
        ax.set_xlim(0, max(ranks))
        ax.set_ylim(0, 1.05)

        ax.legend(loc="lower right", fontsize=6, frameon=True)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved appendix figure to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r0_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument(
        "--r2_checkpoint",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R2/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument("--save_dir", type=str, default="results/write_bottleneck")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    val_data = load_owt_data(args.data_dir)

    # Define ranks to test
    ranks = list(range(1, 21)) + list(
        range(25, 101, 5)
    )  # 1-20 dense, then 25, 30, ..., 100

    all_results = {}

    # R0 (12-head)
    print("\nLoading R0 model...")
    model_r0, config_r0 = load_model(args.r0_checkpoint, DEVICE)
    results_r0 = run_write_bottleneck_experiment(
        model_r0,
        val_data,
        "R0",
        ranks,
        args.n_batches,
        args.batch_size,
        rank1_override_indices=[1],
    )
    all_results["R0_12head"] = results_r0
    del model_r0
    torch.cuda.empty_cache()

    # R2 (12-head)
    print("\nLoading R2 model...")
    model_r2, config_r2 = load_model(args.r2_checkpoint, DEVICE)
    results_r2 = run_write_bottleneck_experiment(
        model_r2, val_data, "R2", ranks, args.n_batches, args.batch_size
    )
    all_results["R2_12head"] = results_r2
    del model_r2
    torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(args.save_dir, "write_bottleneck_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Create main text figure
    create_main_text_plot(
        results_r0,
        results_r2,
        os.path.join(args.save_dir, "write_bottleneck_curves.pdf"),
    )

    # Create appendix figure (just with available models for now)
    create_appendix_plot(
        all_results, os.path.join(args.save_dir, "write_bottleneck_curves_all.pdf")
    )

    # Copy to paper directory
    import shutil

    paper_dir = "overleaf/nopos_icml_2026/plots"
    shutil.copy(
        os.path.join(args.save_dir, "write_bottleneck_curves.pdf"),
        os.path.join(paper_dir, "write_bottleneck_curves.pdf"),
    )
    shutil.copy(
        os.path.join(args.save_dir, "write_bottleneck_curves_all.pdf"),
        os.path.join(paper_dir, "write_bottleneck_curves_all_models.pdf"),
    )
    print(f"\nCopied figures to {paper_dir}")

    # Print table data for paper
    print("\n" + "=" * 60)
    print("TABLE DATA FOR PAPER (tab:r95)")
    print("=" * 60)
    print(f"{'Model':<15} {'Baseline R²':<15} {'r_95':<10}")
    print("-" * 40)
    for key, results in all_results.items():
        print(f"{key:<15} {results['baseline_r2']:.4f}         {results['r_95']}")


if __name__ == "__main__":
    main()
