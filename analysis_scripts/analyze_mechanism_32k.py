"""
Analyze geometric clock mechanism in 32K-context trained models.

Verifies that the 5-step mechanism (BOS norm advantage → directional separation
→ BOS over-attention → rotation → readout) emerges at 32K context, matching
the pattern observed at 128-context models.

Produces:
  - Position regression R² (baseline)
  - d_BOS / d_nonBOS directions and cosine similarity
  - Attention output projection curves (Spearman with position)
  - BOS attention mass (r_h ratio from Eq. 13)
  - Write bottleneck curves (SVD of B = W_O W_V)
  - Dial rotation visualization

Usage:
    python analysis_scripts/analyze_mechanism_32k.py \
        --checkpoint nanoGPT/out-mechanism-R0-32k/R0/<run_id>/best_ckpt.pt \
        --model_name FULL-12H-32K \
        --out_dir results/mechanism_32k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

# ICML style
plt.rcParams.update({
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
})

COLOR_BOS = "#009E73"
COLOR_OTHERS = "#CC79A7"
COLOR_R0 = "#0072B2"
COLOR_R2 = "#D55E00"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_TOKEN_ID = 50256


# =============================================================================
# Loading
# =============================================================================

def load_model(checkpoint_path: str, device: str = DEVICE):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    config = TwoLayerMechanismConfig(
        block_size=cfg.get("block_size", 128),
        vocab_size=cfg.get("vocab_size", 50304),
        n_embd=cfg.get("n_embd", 768),
        n_head=cfg.get("n_head", 12),
        dropout=0.0,
        norm_type=cfg.get("norm_type", "layernorm"),
        bias=True,
        use_regression=True,
        use_flash=False,  # Disable flash for analysis (need weight capture)
    )

    model = TwoLayerMechanismModel(config)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    regime = checkpoint.get("regime", "unknown")
    print(f"Loaded {regime} model: block_size={config.block_size}, n_head={config.n_head}")
    return model, config, checkpoint


def load_owt_data():
    return np.memmap(
        ROOT / "nanoGPT/data/openwebtext/val.bin", dtype=np.uint16, mode="r"
    )


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - (block_size - 1), (batch_size,))
    x = torch.stack([
        torch.from_numpy(
            np.concatenate([[BOS_TOKEN_ID], data[i : i + block_size - 1].astype(np.int64)])
        )
        for i in ix
    ])
    return x.to(device)


def r2_score(preds, targets):
    preds, targets = preds.flatten().float(), targets.flatten().float()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


# =============================================================================
# 1. Baseline position regression
# =============================================================================

def evaluate_position_regression(model, data, n_batches=20, batch_size=2):
    """Evaluate R², MAE of the position head at full 32K context.

    The model predicts in [0,1] (normalized targets). We scale predictions
    back to absolute positions for interpretable metrics.
    """
    block_size = model.config.block_size
    scale = max(block_size - 1, 1)
    positions = np.arange(block_size, dtype=np.float32)

    y_true_all, y_pred_all = [], []
    for _ in tqdm(range(n_batches), desc="Position regression"):
        tokens = get_batch(data, batch_size, block_size, DEVICE)
        with torch.no_grad():
            # Manual forward to avoid attention weight storage at 32K
            e = model.wte(tokens)
            x = model.block1(e, capture_taps=False)
            x = model.block2(x, capture_taps=False)
            x = model.ln_f(x)
            preds = model.pos_head(x).squeeze(-1) * scale  # scale back

        y_pred_all.append(preds.cpu().numpy().reshape(-1))
        y_true_all.append(np.tile(positions, batch_size))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_cod = 1.0 - ss_res / ss_tot
    r2_corr = float(np.corrcoef(y_true, y_pred)[0, 1]) ** 2
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {"r2_cod": r2_cod, "r2_corr": r2_corr, "mae": mae, "rmse": rmse,
            "n_samples": len(y_true)}


# =============================================================================
# 2. Geometric gauge: d_BOS, d_nonBOS, projections
# =============================================================================

def analyze_geometric_gauge(model, data, n_batches=10, batch_size=2,
                            eval_context=None):
    """Analyze d_BOS / d_nonBOS directions and attention output projections.

    For memory reasons at 32K, we can optionally evaluate at a shorter
    eval_context while still using the trained weights.
    """
    D = model.config.n_embd
    T = eval_context or model.config.block_size
    n_head = model.config.n_head
    head_dim = D // n_head

    W_V = model.block2.attn.c_attn.weight[2 * D:, :]
    W_O = model.block2.attn.c_proj.weight
    b_O = model.block2.attn.c_proj.bias

    all_proj_bos, all_proj_others = [], []
    all_attn_to_bos = []

    for _ in tqdm(range(n_batches), desc="Geometric gauge"):
        tokens = get_batch(data, batch_size, T, DEVICE)

        with torch.no_grad():
            e = model.wte(tokens)
            # Block 1 — full forward
            r1 = model.block1(e, capture_taps=False)

            # Block 2 — manual for taps
            ln1_b2 = model.block2.ln_1(r1)

            # Compute attention manually at this context length
            qkv = model.block2.attn.c_attn(ln1_b2)
            q, k, v = qkv.split(D, dim=2)
            B = tokens.shape[0]
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, n_head, head_dim).transpose(1, 2)

            # Use flash attention if T is large, manual if small enough
            if T <= 4096:
                att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
                mask = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.bool), diagonal=1)
                att = att.masked_fill(mask, float("-inf"))
                att = F.softmax(att, dim=-1)
                y2 = (att @ v).transpose(1, 2).contiguous().view(B, T, D)

                # BOS attention mass per head
                attn_to_bos = att[:, :, :, 0].mean(dim=0)  # [n_head, T]
                all_attn_to_bos.append(attn_to_bos.cpu())
            else:
                # Flash attention — no explicit weight matrix
                y2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                y2 = y2.transpose(1, 2).contiguous().view(B, T, D)

            attn_out = model.block2.attn.c_proj(y2)

            # Compute Wo @ Wv projections for direction estimation
            Wo_v = ln1_b2 @ W_V.T @ W_O.T  # [B, T, D]

            # BOS and non-BOS write directions
            bos_write_mean = Wo_v[:, 0, :].mean(dim=0)
            others_write_mean = Wo_v[:, 1:, :].mean(dim=(0, 1))

            d_bos = F.normalize(bos_write_mean.unsqueeze(0), dim=-1).squeeze()
            d_others = F.normalize(others_write_mean.unsqueeze(0), dim=-1).squeeze()

            # Project attention output onto these directions
            attn_out_no_bias = attn_out - b_O
            proj_bos = (attn_out_no_bias @ d_bos).mean(dim=0)   # [T]
            proj_others = (attn_out_no_bias @ d_others).mean(dim=0)

            all_proj_bos.append(proj_bos.cpu())
            all_proj_others.append(proj_others.cpu())

    proj_bos = torch.stack(all_proj_bos).mean(dim=0).numpy()
    proj_others = torch.stack(all_proj_others).mean(dim=0).numpy()

    positions = np.arange(T)
    spearman_bos = stats.spearmanr(proj_bos, positions).statistic
    spearman_others = stats.spearmanr(proj_others, positions).statistic

    # Direction properties
    bos_norm = bos_write_mean.norm().item()
    others_norm = others_write_mean.norm().item()
    cos_bos_others = F.cosine_similarity(
        d_bos.unsqueeze(0), d_others.unsqueeze(0)
    ).item()

    # Head alignment
    w_head = model.pos_head.weight.detach().squeeze().cpu()
    cos_head_bos = F.cosine_similarity(w_head.unsqueeze(0), d_bos.cpu().unsqueeze(0)).item()
    cos_head_others = F.cosine_similarity(w_head.unsqueeze(0), d_others.cpu().unsqueeze(0)).item()

    # BOS attention ratio (r_h from Eq. 13)
    bos_ratio = None
    if all_attn_to_bos:
        attn_to_bos_avg = torch.stack(all_attn_to_bos).mean(dim=0)  # [n_head, T]
        # r_h = mean attn to BOS / mean attn to nonBOS  (for positions > 0)
        bos_weight = attn_to_bos_avg[:, 1:].mean(dim=-1)  # [n_head]
        # Average non-BOS weight at each head
        uniform_weight = 1.0 / (positions[1:] + 1)  # expected uniform
        bos_ratio = (bos_weight / (1.0 / T)).numpy().tolist()

    return {
        "proj_bos": proj_bos,
        "proj_others": proj_others,
        "spearman_bos": spearman_bos,
        "spearman_others": spearman_others,
        "bos_write_norm": bos_norm,
        "others_write_norm": others_norm,
        "norm_ratio": bos_norm / (others_norm + 1e-9),
        "cos_bos_others": cos_bos_others,
        "cos_head_bos": cos_head_bos,
        "cos_head_others": cos_head_others,
        "bos_ratio_per_head": bos_ratio,
        "eval_context": T,
    }


# =============================================================================
# 3. Write bottleneck (SVD of B = W_O W_V)
# =============================================================================

def get_block2_write_map_svd(model):
    D = model.config.n_embd
    W_V = model.block2.attn.c_attn.weight[2 * D:, :]
    W_O = model.block2.attn.c_proj.weight
    B = W_O @ W_V
    U, S, Vt = torch.linalg.svd(B, full_matrices=True)
    return U, S, Vt


def forward_with_write_intervention(model, tokens, U, rank, intervention_type):
    B, T = tokens.shape
    D = model.config.n_embd

    with torch.no_grad():
        e = model.wte(tokens)
        r1 = model.block1(e, capture_taps=False)

        # Block 2 attention via flash
        ln1_b2 = model.block2.ln_1(r1)
        n_head = model.config.n_head
        head_dim = D // n_head
        qkv = model.block2.attn.c_attn(ln1_b2)
        q, k, v = qkv.split(D, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        y2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y2 = y2.transpose(1, 2).contiguous().view(B, T, D)
        attn_out2 = model.block2.attn.c_proj(y2)

        # Intervention on attention output
        U_r = U[:, :rank]
        if intervention_type == "retention":
            attn_out2 = attn_out2 @ U_r @ U_r.T
        elif intervention_type == "ablation":
            attn_out2 = attn_out2 - attn_out2 @ U_r @ U_r.T

        r2_attn = r1 + attn_out2
        ln2_b2 = model.block2.ln_2(r2_attn)
        mlp_out2 = model.block2.mlp(ln2_b2)
        r2 = r2_attn + mlp_out2
        x = model.ln_f(r2)
        preds = model.pos_head(x).squeeze(-1)

    return preds


def run_write_bottleneck(model, data, ranks, n_batches=10, batch_size=2,
                         model_name=""):
    print(f"\nWrite bottleneck: {model_name}")
    U, S, Vt = get_block2_write_map_svd(model)
    block_size = model.config.block_size

    scale = max(block_size - 1, 1)

    # Baseline
    all_preds, all_pos = [], []
    for _ in range(n_batches):
        tokens = get_batch(data, batch_size, block_size, DEVICE)
        with torch.no_grad():
            e = model.wte(tokens)
            x = model.block1(e, capture_taps=False)
            x = model.block2(x, capture_taps=False)
            x = model.ln_f(x)
            preds = model.pos_head(x).squeeze(-1) * scale
        positions = torch.arange(block_size, device=DEVICE).float().unsqueeze(0).expand(batch_size, -1)
        all_preds.append(preds.cpu())
        all_pos.append(positions.cpu())

    baseline_r2 = r2_score(torch.cat(all_preds), torch.cat(all_pos))
    print(f"  Baseline R²: {baseline_r2:.4f}")

    retention_r2s, ablation_r2s = [], []
    for rank in tqdm(ranks, desc="Write bottleneck ranks"):
        for intervention, result_list in [("retention", retention_r2s), ("ablation", ablation_r2s)]:
            all_preds, all_pos = [], []
            for _ in range(n_batches):
                tokens = get_batch(data, batch_size, block_size, DEVICE)
                preds = forward_with_write_intervention(model, tokens, U, rank, intervention) * scale
                positions = torch.arange(block_size, device=DEVICE).float().unsqueeze(0).expand(batch_size, -1)
                all_preds.append(preds.cpu())
                all_pos.append(positions.cpu())
            result_list.append(r2_score(torch.cat(all_preds), torch.cat(all_pos)))

    r_95 = None
    threshold = 0.95 * baseline_r2
    for i, r2 in enumerate(retention_r2s):
        if r2 >= threshold:
            r_95 = ranks[i]
            break

    print(f"  r_95 = {r_95}")

    return {
        "baseline_r2": float(baseline_r2),
        "ranks": ranks,
        "retention_r2s": [float(x) for x in retention_r2s],
        "ablation_r2s": [float(x) for x in ablation_r2s],
        "r_95": r_95,
        "singular_values": S.detach().cpu().numpy().tolist()[:20],
    }


# =============================================================================
# 4. Dial rotation visualization
# =============================================================================

def compute_dial(model, data, n_batches=10, batch_size=2, eval_context=None):
    """Compute the directional rotation dial at the given context length."""
    D = model.config.n_embd
    T = eval_context or model.config.block_size
    n_head = model.config.n_head
    head_dim = D // n_head

    W_V = model.block2.attn.c_attn.weight[2 * D:, :]
    W_O = model.block2.attn.c_proj.weight
    b_O = model.block2.attn.c_proj.bias

    # First pass: estimate d_BOS and d_nonBOS
    bos_accum = torch.zeros(D, device=DEVICE)
    others_accum = torch.zeros(D, device=DEVICE)
    n_bos, n_others = 0, 0

    for _ in range(min(5, n_batches)):
        tokens = get_batch(data, batch_size, T, DEVICE)
        with torch.no_grad():
            e = model.wte(tokens)
            r1 = model.block1(e, capture_taps=False)
            ln1_b2 = model.block2.ln_1(r1)
            Wo_v = ln1_b2 @ W_V.T @ W_O.T

            bos_accum += Wo_v[:, 0, :].sum(dim=0)
            others_accum += Wo_v[:, 1:, :].sum(dim=(0, 1))
            n_bos += batch_size
            n_others += batch_size * (T - 1)

    d_bos = F.normalize((bos_accum / n_bos).unsqueeze(0), dim=-1).squeeze()
    d_others = F.normalize((others_accum / n_others).unsqueeze(0), dim=-1).squeeze()

    # Second pass: project attention outputs
    all_proj_bos, all_proj_others = [], []

    for _ in range(n_batches):
        tokens = get_batch(data, batch_size, T, DEVICE)
        with torch.no_grad():
            e = model.wte(tokens)
            r1 = model.block1(e, capture_taps=False)
            ln1_b2 = model.block2.ln_1(r1)

            qkv = model.block2.attn.c_attn(ln1_b2)
            q, k, v = qkv.split(D, dim=2)
            B = tokens.shape[0]
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, n_head, head_dim).transpose(1, 2)

            y2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y2 = y2.transpose(1, 2).contiguous().view(B, T, D)
            attn_out = model.block2.attn.c_proj(y2) - b_O

            proj_b = (attn_out @ d_bos).mean(dim=0)
            proj_o = (attn_out @ d_others).mean(dim=0)
            all_proj_bos.append(proj_b.cpu())
            all_proj_others.append(proj_o.cpu())

    proj_bos = torch.stack(all_proj_bos).mean(dim=0).numpy()
    proj_others = torch.stack(all_proj_others).mean(dim=0).numpy()

    # Normalize to unit circle for dial
    norms = np.sqrt(proj_bos**2 + proj_others**2)
    dial_x = proj_bos / (norms + 1e-9)
    dial_y = proj_others / (norms + 1e-9)
    dial_theta = np.arctan2(dial_y, dial_x)

    return {
        "proj_bos": proj_bos,
        "proj_others": proj_others,
        "dial_x": dial_x,
        "dial_y": dial_y,
        "dial_theta": dial_theta,
        "cos_bos_others": F.cosine_similarity(d_bos.unsqueeze(0), d_others.unsqueeze(0)).item(),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_projection_curves(gauge, save_path):
    T = len(gauge["proj_bos"])
    positions = np.arange(T)
    # Subsample for readability at 32K
    stride = max(1, T // 500)
    idx = np.arange(0, T, stride)

    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.plot(positions[idx], gauge["proj_bos"][idx],
            label=f"BOS dir (ρ={gauge['spearman_bos']:.3f})", color=COLOR_BOS, linewidth=1.2)
    ax.plot(positions[idx], gauge["proj_others"][idx],
            label=f"non-BOS dir (ρ={gauge['spearman_others']:.3f})", color=COLOR_OTHERS, linewidth=1.2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Projection")
    ax.set_title("Attention Output Projection")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_write_bottleneck(wb, save_path, model_name=""):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ranks = wb["ranks"]
    ax.plot(ranks, wb["retention_r2s"], "o-", color=COLOR_R0, markersize=3,
            linewidth=1.2, label="Retention")
    ax.plot(ranks, wb["ablation_r2s"], "s-", color=COLOR_R2, markersize=3,
            linewidth=1.2, label="Ablation")
    ax.axhline(wb["baseline_r2"], ls="--", color="gray", alpha=0.5, label="Baseline")
    ax.set_xlabel("Rank r")
    ax.set_ylabel("Position R²")
    ax.set_title(f"Write Bottleneck ({model_name})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_dial(dial_data, save_path, model_name="", marker_stride=2048):
    T = len(dial_data["dial_x"])
    positions = np.arange(T)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Draw arc
    theta = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgray", linewidth=0.5)

    # Color by position
    colors = plt.cm.viridis(positions / T)
    stride = max(1, T // 500)
    idx = np.arange(0, T, stride)

    for i in range(len(idx) - 1):
        ax.plot([dial_data["dial_x"][idx[i]], dial_data["dial_x"][idx[i+1]]],
                [dial_data["dial_y"][idx[i]], dial_data["dial_y"][idx[i+1]]],
                color=colors[idx[i]], linewidth=1.5)

    # Markers at regular intervals
    marker_idx = np.arange(0, T, marker_stride)
    ax.scatter(dial_data["dial_x"][marker_idx], dial_data["dial_y"][marker_idx],
               c=positions[marker_idx], cmap="viridis", s=20, zorder=5,
               vmin=0, vmax=T)

    ax.set_xlabel("BOS direction")
    ax.set_ylabel("non-BOS direction")
    ax.set_title(f"Dial ({model_name})")
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, T))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Position", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_ckpt.pt")
    parser.add_argument("--model_name", default="32K", help="Label for plots")
    parser.add_argument("--out_dir", default="results/mechanism_32k")
    parser.add_argument("--eval_context", type=int, default=None,
                        help="Override context for gauge analysis (saves memory)")
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--wb_batches", type=int, default=10,
                        help="Batches for write bottleneck")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, config, ckpt = load_model(args.checkpoint)
    data = load_owt_data()
    block_size = config.block_size

    results = {"model_name": args.model_name, "block_size": block_size,
               "n_head": config.n_head, "checkpoint": args.checkpoint}

    # 1. Position regression
    print("\n=== Position Regression ===")
    reg = evaluate_position_regression(model, data, n_batches=args.n_batches,
                                       batch_size=args.batch_size)
    results["position_regression"] = reg
    print(f"  R²(CoD)={reg['r2_cod']:.4f}  R²(corr)={reg['r2_corr']:.4f}  "
          f"MAE={reg['mae']:.2f}  RMSE={reg['rmse']:.2f}")

    # 2. Geometric gauge (use shorter context if needed for attention analysis)
    eval_ctx = args.eval_context or min(block_size, 4096)
    print(f"\n=== Geometric Gauge (eval_context={eval_ctx}) ===")
    gauge = analyze_geometric_gauge(model, data, n_batches=args.n_batches,
                                    batch_size=args.batch_size, eval_context=eval_ctx)
    results["geometric_gauge"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                  for k, v in gauge.items()}
    print(f"  Spearman(proj_BOS, pos) = {gauge['spearman_bos']:.4f}")
    print(f"  Spearman(proj_others, pos) = {gauge['spearman_others']:.4f}")
    print(f"  cos(d_BOS, d_nonBOS) = {gauge['cos_bos_others']:.4f}")
    print(f"  ||w_BOS|| / ||w_others|| = {gauge['norm_ratio']:.2f}")
    print(f"  cos(w_head, d_nonBOS) = {gauge['cos_head_others']:.4f}")

    plot_projection_curves(gauge, os.path.join(args.out_dir, f"projection_{args.model_name}.png"))

    # 3. Write bottleneck
    print("\n=== Write Bottleneck ===")
    ranks = list(range(1, min(101, config.n_embd)))
    wb = run_write_bottleneck(model, data, ranks, n_batches=args.wb_batches,
                              batch_size=args.batch_size, model_name=args.model_name)
    results["write_bottleneck"] = wb
    plot_write_bottleneck(wb, os.path.join(args.out_dir, f"write_bottleneck_{args.model_name}.png"),
                          model_name=args.model_name)

    # 4. Dial visualization (at shorter eval context for memory)
    print(f"\n=== Dial Visualization (context={eval_ctx}) ===")
    dial = compute_dial(model, data, n_batches=args.n_batches,
                        batch_size=args.batch_size, eval_context=eval_ctx)
    results["dial"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in dial.items()}
    print(f"  cos(d_BOS, d_nonBOS) = {dial['cos_bos_others']:.4f}")

    plot_dial(dial, os.path.join(args.out_dir, f"dial_{args.model_name}.png"),
              model_name=args.model_name, marker_stride=max(1, eval_ctx // 8))

    # Also do a full-context dial using flash attention (no explicit attn weights)
    if block_size > eval_ctx:
        print(f"\n=== Full-Context Dial ({block_size}) ===")
        dial_full = compute_dial(model, data, n_batches=min(5, args.n_batches),
                                 batch_size=1, eval_context=block_size)
        results["dial_full_context"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in dial_full.items()
        }
        plot_dial(dial_full, os.path.join(args.out_dir, f"dial_full_{args.model_name}.png"),
                  model_name=f"{args.model_name} (full {block_size})",
                  marker_stride=max(1, block_size // 8))

    # Save results
    json_path = os.path.join(args.out_dir, f"results_{args.model_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {args.model_name}")
    print(f"{'=' * 60}")
    print(f"  Block size:                {block_size}")
    print(f"  Position R² (CoD):         {reg['r2_cod']:.4f}")
    print(f"  Position MAE:              {reg['mae']:.2f}")
    print(f"  cos(d_BOS, d_nonBOS):      {gauge['cos_bos_others']:.4f}")
    print(f"  ||w_BOS|| / ||w_others||:  {gauge['norm_ratio']:.2f}")
    print(f"  Spearman(proj_BOS, pos):   {gauge['spearman_bos']:.4f}")
    print(f"  Spearman(proj_others, pos):{gauge['spearman_others']:.4f}")
    print(f"  Write bottleneck r_95:     {wb['r_95']}")
    print(f"  cos(w_head, d_nonBOS):     {gauge['cos_head_others']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
