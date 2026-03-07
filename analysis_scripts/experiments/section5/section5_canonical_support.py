"""Canonical Section 5 support artifact generator.

Produces a single JSON covering all five mechanism steps plus final decoding
and write-subspace interventions for both model_minimal (ATTN2-1H) and
model_full (FULL-12H), at the exact n_seq sizes declared in the paper.

Metrics produced (per model):
  - Final R^2 at n_seq=1600 (25 batches x 64)
  - Step 1: population and sample-level Spearman, probe R^2
  - Step 2: write-separation norms, ratio, cosine
  - Step 3: per-head BOS-bias ratio r_h, per-position BOS mass curve,
            Spearman of BOS mass curve with position
  - Step 4: population and sample Spearman (BOS/nonBOS/delta)
  - Step 5: cos(w_head, d_nonBOS), cos(w_head, d_delta)
  - Write interventions: baseline, top-1 retention, top-2 retention,
                          top-1 ablation R^2 at n_seq=1600

Usage:
    CUDA_VISIBLE_DEVICES=2 python analysis_scripts/experiments/section5/section5_canonical_support.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel

# ── constants ────────────────────────────────────────────────────────────
BOS_TOKEN_ID = 50256
SEED = 42

# Default checkpoint / data paths (repo-relative; overridable via CLI)
# Try repo-local model_backups/ first, fall back to nanoGPT/ subdirs.
_DEFAULT_ATTN2_CKPT = str(ROOT / "model_backups" / "R2_ATTN2-1H" / "best_ckpt.pt")
_DEFAULT_FULL12H_CKPT = str(ROOT / "model_backups" / "R0_FULL-12H" / "best_ckpt.pt")
_DEFAULT_DATA_DIR = str(ROOT / "nanoGPT" / "data" / "openwebtext")

ATTN2_CKPT = Path(_DEFAULT_ATTN2_CKPT)
FULL12H_CKPT = Path(_DEFAULT_FULL12H_CKPT)
DATA_DIR = Path(_DEFAULT_DATA_DIR)

SAVE_DIR = ROOT / "results" / "section5_support"

# Paper-declared evaluation sizes
FINAL_N_BATCHES = 25
FINAL_BATCH_SIZE = 64  # n_seq = 25*64 = 1600

WRITE_N_BATCHES = 50
WRITE_BATCH_SIZE = 32  # n_seq = 50*32 = 1600

STEP1_TRAIN_BATCHES = 20
STEP1_TEST_BATCHES = 10
STEP1_BATCH_SIZE = 64

STEP4_N_BATCHES = 10
STEP4_BATCH_SIZE = 32

STEP2_N_BATCHES = 25
STEP2_BATCH_SIZE = 64  # n_seq = 1600

STEP3_N_BATCHES = 25
STEP3_BATCH_SIZE = 64  # n_seq = 1600


# ── helpers ──────────────────────────────────────────────────────────────


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe_spearman(x, y) -> float:
    rho = spearmanr(x, y).correlation
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _to_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


# ── model / data loading ────────────────────────────────────────────────


def load_model(ckpt: Path, device: str) -> TwoLayerMechanismModel:
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid = set(TwoLayerMechanismConfig.__dataclass_fields__)
    config = TwoLayerMechanismConfig(**{k: v for k, v in cfg.items() if k in valid})
    model = TwoLayerMechanismModel(config)
    sd = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def load_val_data() -> np.memmap:
    return np.memmap(str(DATA_DIR / "val.bin"), dtype=np.uint16, mode="r")


def get_batch_bos(
    data: np.memmap, batch_size: int, block_size: int, device: str
) -> torch.Tensor:
    ix = torch.randint(len(data) - (block_size - 1), (batch_size,))
    seqs = []
    for i in ix:
        tail = data[i : i + block_size - 1].astype(np.int64)
        seqs.append(torch.from_numpy(np.concatenate([[BOS_TOKEN_ID], tail])))
    return torch.stack(seqs).to(device)


# ── metric functions ─────────────────────────────────────────────────────


def compute_final_r2(model, data, device) -> float:
    """R^2 at n_seq=1600."""
    bs = model.config.block_size
    positions = np.arange(bs, dtype=np.float32)
    y_true, y_pred = [], []
    for _ in range(FINAL_N_BATCHES):
        tokens = get_batch_bos(data, FINAL_BATCH_SIZE, bs, device)
        with torch.no_grad():
            out, _ = model(tokens, capture_taps=False)
        preds = out.squeeze(-1).cpu().numpy()
        y_pred.append(preds.reshape(-1))
        y_true.append(np.tile(positions, FINAL_BATCH_SIZE))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def compute_step2(model, data, device) -> dict:
    """Step 2: write-separation norms, ratio, cosine."""
    d = model.config.n_embd
    W_V = model.block2.attn.c_attn.weight[2 * d :, :].detach()
    W_O = model.block2.attn.c_proj.weight.detach()
    B = W_O @ W_V  # [d, d]

    bos_writes, nbos_writes = [], []
    for _ in range(STEP2_N_BATCHES):
        tokens = get_batch_bos(data, STEP2_BATCH_SIZE, model.config.block_size, device)
        with torch.no_grad():
            model(tokens, capture_taps=True)
        taps = model.get_all_taps()
        x2 = taps["block2_ln1"]  # [B, T, d]
        Bx = x2 @ B.T  # [B, T, d]
        bos_writes.append(Bx[:, 0, :].cpu())  # [B, d]
        nbos_writes.append(Bx[:, 1:, :].cpu())  # [B, T-1, d]

    bos_all = torch.cat(bos_writes, dim=0)  # [N, d]
    nbos_all = torch.cat(nbos_writes, dim=0)  # [N, T-1, d]

    w_bos = bos_all.mean(dim=0)  # [d]
    w_nbos = nbos_all.mean(dim=(0, 1))  # [d]

    norm_bos = float(w_bos.norm().item())
    norm_nbos = float(w_nbos.norm().item())
    ratio = norm_bos / (norm_nbos + 1e-12)
    cos = _safe_cosine(w_bos, w_nbos)

    return {
        "bos_norm": round(norm_bos, 1),
        "nbos_norm": round(norm_nbos, 1),
        "ratio": round(ratio, 2),
        "cos_bos_nbos": round(cos, 3),
        "n_seq": STEP2_N_BATCHES * STEP2_BATCH_SIZE,
    }


def compute_step3(model, data, device) -> dict:
    """Step 3: per-head BOS-bias ratio and BOS mass monotonicity."""
    n_heads = model.config.n_head
    bs = model.config.block_size

    # accumulators
    bos_mass_sum = None  # [n_heads, T]  sum over batches of mean-over-seqs BOS mass
    nbos_mass_sum = None  # [n_heads, T]  sum of mean-nbos mass
    n_total = 0

    for _ in range(STEP3_N_BATCHES):
        tokens = get_batch_bos(data, STEP3_BATCH_SIZE, bs, device)
        with torch.no_grad():
            model(tokens, capture_taps=True)
        _, attn2 = model.get_attention_weights()  # [B, H, T, T]
        attn2 = attn2.float().cpu()

        # BOS mass: attn2[:, h, t, 0] for t > 0
        bos_mass = attn2[:, :, :, 0]  # [B, H, T]

        # mean non-BOS mass per head per position:
        # for each query t>0, average attention to keys j=1..t
        # nbos_avg(b,h,t) = (1/t) * sum_{j=1}^{t} attn2(b,h,t,j)  for t>=1
        T = bs
        nbos_avg = torch.zeros_like(bos_mass)
        for t in range(1, T):
            if t == 1:
                # only key j=1 available as non-BOS
                nbos_avg[:, :, t] = attn2[:, :, t, 1]
            else:
                nbos_avg[:, :, t] = attn2[:, :, t, 1 : t + 1].mean(dim=-1)

        if bos_mass_sum is None:
            bos_mass_sum = bos_mass.sum(dim=0)  # [H, T]
            nbos_mass_sum = nbos_avg.sum(dim=0)
        else:
            bos_mass_sum += bos_mass.sum(dim=0)
            nbos_mass_sum += nbos_avg.sum(dim=0)
        n_total += STEP3_BATCH_SIZE

    bos_mass_mean = bos_mass_sum / n_total  # [H, T]
    nbos_mass_mean = nbos_mass_sum / n_total  # [H, T]

    # BOS-bias ratio per head: E_{t>0}[bos_mass] / E_{t>0}[nbos_avg]
    # (averaged over positions t>0)
    r_h = {}
    bos_mass_spearman = {}
    positions = np.arange(1, bs)

    for h in range(n_heads):
        mean_bos = float(bos_mass_mean[h, 1:].mean().item())
        mean_nbos = float(nbos_mass_mean[h, 1:].mean().item())
        r_val = mean_bos / (mean_nbos + 1e-12)
        r_h[f"H{h}"] = round(r_val, 3)

        # Spearman of BOS mass curve with position
        curve = bos_mass_mean[h, 1:].numpy()
        rho = _safe_spearman(curve, positions)
        bos_mass_spearman[f"H{h}"] = round(rho, 4)

    return {
        "bos_bias_ratio": r_h,
        "bos_mass_spearman": bos_mass_spearman,
        "bos_mass_curves": {f"H{h}": bos_mass_mean[h].tolist() for h in range(n_heads)},
        "n_seq": n_total,
    }


def compute_step4_and_5(model, data, device) -> dict:
    """Steps 4+5: projection Spearman and head alignment."""
    # Import the reusable compute_projection from r0 script
    proj_mod = _load_module(
        "r0_proj",
        ROOT
        / "analysis_scripts/experiments/r0_analysis/r0_attention_output_projection.py",
    )
    # Use validation data consistently (matches paper reporting convention)
    result = proj_mod.compute_projection(
        model,
        data,
        128,
        STEP4_BATCH_SIZE,
        STEP4_N_BATCHES,
        device,
    )

    return {
        "step4_pop_bos": result["spearman_bos"],
        "step4_pop_nonbos": result["spearman_non_bos"],
        "step4_pop_delta": result["spearman_delta"],
        "step4_sample_median_bos": result["spearman_stats"]["bos"][
            "sample_sequence_median_rho"
        ],
        "step4_sample_median_nonbos": result["spearman_stats"]["non_bos"][
            "sample_sequence_median_rho"
        ],
        "step4_sample_median_delta": result["spearman_stats"]["delta"][
            "sample_sequence_median_rho"
        ],
        "step5_cos_head_nonbos": result["cos_head_non_bos"],
        "step5_cos_head_delta": result["cos_head_delta"],
        "step5_cos_head_bos": result["cos_head_bos"],
        "cos_bos_nonbos": result["cos_bos_non_bos"],
        "cos_nonbos_delta": result["cos_non_bos_delta"],
        "angle_bos_nonbos_deg": result["angle_bos_non_bos_deg"],
        "angle_nonbos_delta_deg": result["angle_non_bos_delta_deg"],
        # Step 4 diagnostics
        "residual_abs_mean": result["residual_proj_norm_abs_mean"],
        "concentration_cos_mean_ge16": result["concentration_cos_mean_curve_mean_ge16"],
        "step4_n_seq": STEP4_N_BATCHES * STEP4_BATCH_SIZE,
    }


def compute_write_interventions(model, data, U, device, model_name) -> dict:
    """Write-subspace intervention at n_seq=1600."""
    wb_mod = _load_module(
        "wb",
        ROOT / "analysis_scripts/figures/generate_write_bottleneck_plot.py",
    )

    # For FULL-12H, rank-1 uses basis_indices=[1] per paper convention
    rank1_indices = [1] if "full" in model_name.lower() else None

    baseline = wb_mod.compute_r2_at_rank(
        model,
        data,
        U,
        768,
        "retention",
        n_batches=WRITE_N_BATCHES,
        batch_size=WRITE_BATCH_SIZE,
        block_size=model.config.block_size,
    )
    ret1 = wb_mod.compute_r2_at_rank(
        model,
        data,
        U,
        1,
        "retention",
        n_batches=WRITE_N_BATCHES,
        batch_size=WRITE_BATCH_SIZE,
        block_size=model.config.block_size,
        basis_indices=rank1_indices,
    )
    ret2 = wb_mod.compute_r2_at_rank(
        model,
        data,
        U,
        2,
        "retention",
        n_batches=WRITE_N_BATCHES,
        batch_size=WRITE_BATCH_SIZE,
        block_size=model.config.block_size,
    )
    abl1 = wb_mod.compute_r2_at_rank(
        model,
        data,
        U,
        1,
        "ablation",
        n_batches=WRITE_N_BATCHES,
        batch_size=WRITE_BATCH_SIZE,
        block_size=model.config.block_size,
        basis_indices=rank1_indices,
    )

    return {
        "baseline_r2": round(float(baseline), 4),
        "top1_retention_r2": round(float(ret1), 4),
        "top2_retention_r2": round(float(ret2), 4),
        "top1_ablation_r2": round(float(abl1), 4),
        "n_seq": WRITE_N_BATCHES * WRITE_BATCH_SIZE,
    }


def compute_step1(model, data, device, model_name) -> dict:
    """Step 1 via existing bos_anchor_decodability module."""
    bos_mod = _load_module(
        "bos_decodability",
        ROOT
        / "analysis_scripts/experiments/interventions/bos_anchor_decodability_after_ln.py",
    )
    probe_out = bos_mod.run_probe_for_model(
        model_name=model_name,
        model=model,
        data=data,
        train_batches=STEP1_TRAIN_BATCHES,
        test_batches=STEP1_TEST_BATCHES,
        batch_size=STEP1_BATCH_SIZE,
        seed=SEED,
        device=device,
        inverse_eps=0.1,
    )
    bos_stats = probe_out["metrics"]["bos_b1ln"]["projection_spearman"]
    non_stats = probe_out["metrics"]["nonbos_b1ln"]["projection_spearman"]
    full_lin = probe_out["full_linear_probe"]

    return {
        "pop_bos": bos_stats["population_mean_curve_rho"],
        "pop_nonbos": non_stats["population_mean_curve_rho"],
        "sample_median_bos": bos_stats["sample_sequence_median_rho"],
        "sample_median_nonbos": non_stats["sample_sequence_median_rho"],
        "full_probe_r2": full_lin["test_metrics"]["r2"],
        "random_label_r2": full_lin["random_label_test_metrics"]["r2"],
        "n_seq_train": STEP1_TRAIN_BATCHES * STEP1_BATCH_SIZE,
        "n_seq_test": STEP1_TEST_BATCHES * STEP1_BATCH_SIZE,
    }


def compute_step5_mlp_zero_control(model, data, device) -> dict:
    """Step 5 MLP-zero control: zero Layer-2 MLP at inference and report R².

    Uses model.set_post_attn_head(True) to skip MLP2, computes R² on the
    same held-out data, then restores the model.  Cosine alignments (Step 5)
    are weight-level and unchanged by this intervention.
    """
    # Enable MLP-zero mode (skips MLP2, uses only attn output + residual)
    model.set_post_attn_head(True)
    try:
        r2_no_mlp = round(compute_final_r2(model, data, device), 4)
    finally:
        model.set_post_attn_head(False)

    return {
        "r2_no_mlp2": r2_no_mlp,
        "note": (
            "R^2 when Layer-2 MLP output is zeroed at inference. "
            "Cosine alignments (cos(w_head, d_nonBOS), cos(w_head, d_delta)) "
            "are weight-level and unchanged by this intervention."
        ),
        "n_seq": FINAL_N_BATCHES * FINAL_BATCH_SIZE,
    }


# ── CLI argument parser ─────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical Section 5 support artifact generator."
    )
    parser.add_argument(
        "--attn2-ckpt",
        type=str,
        default=_DEFAULT_ATTN2_CKPT,
        help="Path to ATTN2-1H checkpoint (default: repo-relative model_backups/R2_ATTN2-1H/best_ckpt.pt)",
    )
    parser.add_argument(
        "--full12h-ckpt",
        type=str,
        default=_DEFAULT_FULL12H_CKPT,
        help="Path to FULL-12H checkpoint (default: repo-relative model_backups/R0_FULL-12H/best_ckpt.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=_DEFAULT_DATA_DIR,
        help="Path to OpenWebText data directory (default: repo-relative nanoGPT/data/openwebtext)",
    )
    args = parser.parse_args()

    # Validate that paths exist, with clear error messages
    for label, path in [
        ("ATTN2-1H checkpoint (--attn2-ckpt)", args.attn2_ckpt),
        ("FULL-12H checkpoint (--full12h-ckpt)", args.full12h_ckpt),
    ]:
        if not Path(path).exists():
            parser.error(
                f"{label} not found at: {path}\n"
                f"  Provide the correct path via the corresponding CLI flag."
            )
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        parser.error(
            f"Data directory (--data-dir) not found at: {data_dir}\n"
            f"  Run nanoGPT/data/openwebtext/prepare.py first, or provide the correct path."
        )
    if not (data_dir / "val.bin").exists():
        parser.error(
            f"val.bin not found in data directory: {data_dir}\n"
            f"  Run nanoGPT/data/openwebtext/prepare.py first."
        )

    return args


# ── main ─────────────────────────────────────────────────────────────────


def main():
    # Parse CLI args and resolve paths
    args = parse_args()
    global ATTN2_CKPT, FULL12H_CKPT, DATA_DIR
    ATTN2_CKPT = Path(args.attn2_ckpt).resolve()
    FULL12H_CKPT = Path(args.full12h_ckpt).resolve()
    DATA_DIR = Path(args.data_dir).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {device}")
    print(f"Loading models ...")

    attn_model = load_model(ATTN2_CKPT, device)
    full_model = load_model(FULL12H_CKPT, device)

    val_data = load_val_data()

    # Write-map SVD (reuse across steps)
    wb_mod = _load_module(
        "wb_svd",
        ROOT / "analysis_scripts/figures/generate_write_bottleneck_plot.py",
    )
    U_attn, _, _ = wb_mod.get_block2_write_map_svd(attn_model)
    U_full, _, _ = wb_mod.get_block2_write_map_svd(full_model)
    owt_data = wb_mod.load_owt_data(str(DATA_DIR))

    results: dict[str, Any] = {
        "metadata": {
            "seed": SEED,
            "final_n_seq": FINAL_N_BATCHES * FINAL_BATCH_SIZE,
            "write_n_seq": WRITE_N_BATCHES * WRITE_BATCH_SIZE,
            "step1_train_n_seq": STEP1_TRAIN_BATCHES * STEP1_BATCH_SIZE,
            "step1_test_n_seq": STEP1_TEST_BATCHES * STEP1_BATCH_SIZE,
            "step2_n_seq": STEP2_N_BATCHES * STEP2_BATCH_SIZE,
            "step3_n_seq": STEP3_N_BATCHES * STEP3_BATCH_SIZE,
            "step4_n_seq": STEP4_N_BATCHES * STEP4_BATCH_SIZE,
            "block_size": 128,
            "attn2_ckpt": str(ATTN2_CKPT),  # resolved absolute path
            "full12h_ckpt": str(FULL12H_CKPT),  # resolved absolute path
            "data_dir": str(DATA_DIR),  # resolved absolute path
        }
    }

    for model_key, model, model_name, U in [
        ("attn2_1h", attn_model, "ATTN2-1H", U_attn),
        ("full_12h", full_model, "FULL-12H", U_full),
    ]:
        print(f"\n{'=' * 60}\n  Processing {model_name}\n{'=' * 60}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        entry: dict[str, Any] = {}

        # Final R^2
        print("  Computing final R^2 ...")
        entry["final_r2"] = round(compute_final_r2(model, val_data, device), 4)
        print(f"    Final R^2 = {entry['final_r2']}")

        # Step 1
        print("  Computing Step 1 ...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        entry["step1"] = compute_step1(model, val_data, device, model_name)
        print(
            f"    Step 1 pop BOS/nonBOS: {entry['step1']['pop_bos']:.4f} / {entry['step1']['pop_nonbos']:.4f}"
        )

        # Step 2
        print("  Computing Step 2 ...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        entry["step2"] = compute_step2(model, val_data, device)
        print(
            f"    Step 2 norms: {entry['step2']['bos_norm']} / {entry['step2']['nbos_norm']} ({entry['step2']['ratio']}x)"
        )

        # Step 3
        print("  Computing Step 3 ...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        entry["step3"] = compute_step3(model, val_data, device)
        print(f"    Step 3 BOS-bias ratios: {entry['step3']['bos_bias_ratio']}")
        print(f"    Step 3 BOS mass Spearman: {entry['step3']['bos_mass_spearman']}")

        # Steps 4 + 5
        print("  Computing Steps 4 + 5 ...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        entry["step4_5"] = compute_step4_and_5(model, val_data, device)
        print(
            f"    Step 4 pop Spearman BOS/nonBOS: {entry['step4_5']['step4_pop_bos']} / {entry['step4_5']['step4_pop_nonbos']}"
        )
        print(
            f"    Step 5 cos(w_head, d_nonBOS): {entry['step4_5']['step5_cos_head_nonbos']:.4f}"
        )

        # Write interventions
        print("  Computing write interventions ...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        entry["write"] = compute_write_interventions(
            model, owt_data, U, device, model_name
        )
        print(
            f"    Write: baseline={entry['write']['baseline_r2']}, ret1={entry['write']['top1_retention_r2']}, ret2={entry['write']['top2_retention_r2']}, abl1={entry['write']['top1_ablation_r2']}"
        )

        # Step 5 MLP-zero control (P1-1 causal isolation)
        print("  Computing Step 5 MLP-zero control ...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        entry["step5_mlp_zero_control"] = compute_step5_mlp_zero_control(
            model, val_data, device
        )
        print(
            f"    Step 5 MLP-zero R^2 = {entry['step5_mlp_zero_control']['r2_no_mlp2']}"
        )

        results[model_key] = entry

    # ── save ─────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / "section5_canonical.json"
    out_path.write_text(json.dumps(_to_json(results), indent=2))
    print(f"\nSaved: {out_path}")

    # ── claim provenance ─────────────────────────────────────────────────
    provenance = build_provenance(results)
    prov_path = SAVE_DIR / "claim_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"Saved: {prov_path}")

    return results


def build_provenance(results: dict) -> dict:
    """Build a mapping from each paper claim to its backing artifact.

    Line references target overleaf/nopos_icml_2026/main.tex (1304 lines).
    """
    a = results.get("attn2_1h", {})
    f = results.get("full_12h", {})

    def _r(x, d=3):
        if isinstance(x, float):
            return round(x, d)
        return x

    claims = [
        # ── modelminimal final R^2 ──────────────────────────────────────
        {
            "claim": "modelminimal final R^2 = 0.990",
            "paper_line": "~680",
            "artifact_value": _r(a.get("final_r2")),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.final_r2",
            "n_seq": results["metadata"]["final_n_seq"],
        },
        # ── modelfull final R^2 ─────────────────────────────────────────
        {
            "claim": "modelfull final R^2 = 0.999",
            "paper_line": "~830",
            "artifact_value": _r(f.get("final_r2")),
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.final_r2",
            "n_seq": results["metadata"]["final_n_seq"],
        },
        # ── modelminimal Step 1 ─────────────────────────────────────────
        {
            "claim": "modelminimal Step 1 pop Spearman BOS/nonBOS = -0.998/+0.999",
            "paper_line": "~714-716",
            "artifact_value": f"{_r(a.get('step1', {}).get('pop_bos'), 4)}/{_r(a.get('step1', {}).get('pop_nonbos'), 4)}",
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step1",
        },
        {
            "claim": "modelminimal Step 1 sample median = -0.658/+0.707",
            "paper_line": "~718",
            "artifact_value": f"{_r(a.get('step1', {}).get('sample_median_bos'), 3)}/{_r(a.get('step1', {}).get('sample_median_nonbos'), 3)}",
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step1",
        },
        {
            "claim": "modelminimal Step 1 probe R^2 = 0.554",
            "paper_line": "~737",
            "artifact_value": _r(a.get("step1", {}).get("full_probe_r2")),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step1.full_probe_r2",
        },
        # ── modelminimal Step 2 ─────────────────────────────────────────
        {
            "claim": "modelminimal Step 2 norms 394.9/89.6 (4.41x), cos=-0.658",
            "paper_line": "~746-753",
            "artifact_value": {
                "bos_norm": a.get("step2", {}).get("bos_norm"),
                "nbos_norm": a.get("step2", {}).get("nbos_norm"),
                "ratio": a.get("step2", {}).get("ratio"),
                "cos": a.get("step2", {}).get("cos_bos_nbos"),
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step2",
        },
        # ── modelminimal Step 3 ─────────────────────────────────────────
        {
            "claim": "modelminimal Step 3 BOS-bias ratio r_h=21.9",
            "paper_line": "~771",
            "artifact_value": a.get("step3", {}).get("bos_bias_ratio", {}).get("H0"),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step3.bos_bias_ratio.H0",
        },
        {
            "claim": "modelminimal Step 3 BOS mass Spearman = -1.0",
            "paper_line": "~773-774",
            "artifact_value": a.get("step3", {}).get("bos_mass_spearman", {}).get("H0"),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step3.bos_mass_spearman.H0",
        },
        # ── modelminimal Step 4 ─────────────────────────────────────────
        {
            "claim": "modelminimal Step 4 pop Spearman = -1/+1",
            "paper_line": "~782-784",
            "artifact_value": f"{a.get('step4_5', {}).get('step4_pop_bos')}/{a.get('step4_5', {}).get('step4_pop_nonbos')}",
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step4_5",
        },
        {
            "claim": "modelminimal Step 4 sample median = -0.999/+0.999",
            "paper_line": "~786",
            "artifact_value": f"{_r(a.get('step4_5', {}).get('step4_sample_median_bos'), 4)}/{_r(a.get('step4_5', {}).get('step4_sample_median_nonbos'), 4)}",
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step4_5",
        },
        # ── modelminimal Step 5 ─────────────────────────────────────────
        {
            "claim": "modelminimal Step 5 cos(w_head, d_nonBOS) = 0.731",
            "paper_line": "~805",
            "artifact_value": _r(a.get("step4_5", {}).get("step5_cos_head_nonbos"), 3),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step4_5.step5_cos_head_nonbos",
        },
        {
            "claim": "modelminimal Step 5 cos(w_head, d_delta) = 0.155",
            "paper_line": "~808",
            "artifact_value": _r(a.get("step4_5", {}).get("step5_cos_head_delta"), 3),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step4_5.step5_cos_head_delta",
        },
        # ── modelminimal write interventions ────────────────────────────
        {
            "claim": "modelminimal write: baseline=0.990, ret1=0.809, ret2=0.991, abl1=0.125",
            "paper_line": "~961-965",
            "artifact_value": a.get("write"),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.write",
        },
        # ── modelfull Step 1 ────────────────────────────────────────────
        {
            "claim": "modelfull Step 1 pop Spearman = -0.982/+0.975",
            "paper_line": "~908-910",
            "artifact_value": f"{_r(f.get('step1', {}).get('pop_bos'), 4)}/{_r(f.get('step1', {}).get('pop_nonbos'), 4)}",
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step1",
        },
        {
            "claim": "modelfull Step 1 sample median = -0.397/+0.487",
            "paper_line": "~910",
            "artifact_value": f"{_r(f.get('step1', {}).get('sample_median_bos'), 3)}/{_r(f.get('step1', {}).get('sample_median_nonbos'), 3)}",
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step1",
        },
        {
            "claim": "modelfull Step 1 probe R^2 = 0.560",
            "paper_line": "~911",
            "artifact_value": _r(f.get("step1", {}).get("full_probe_r2")),
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step1.full_probe_r2",
        },
        # ── modelfull Step 2 (appendix) ─────────────────────────────────
        {
            "claim": "modelfull Step 2 norms 1847.7/743.6 (2.48x), cos=-0.969",
            "paper_line": "~1166-1174",
            "artifact_value": {
                "bos_norm": f.get("step2", {}).get("bos_norm"),
                "nbos_norm": f.get("step2", {}).get("nbos_norm"),
                "ratio": f.get("step2", {}).get("ratio"),
                "cos": f.get("step2", {}).get("cos_bos_nbos"),
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step2",
        },
        # ── modelfull Step 3 ────────────────────────────────────────────
        {
            "claim": "modelfull Step 3 BOS-bias ratios H7=100.2, H5=76.5, H11=62.6, H10=33.2",
            "paper_line": "~844, ~1181-1182",
            "artifact_value": {
                h: f.get("step3", {}).get("bos_bias_ratio", {}).get(h)
                for h in ["H7", "H5", "H11", "H10"]
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step3.bos_bias_ratio",
        },
        {
            "claim": "modelfull Step 3 BOS mass Spearman=-1.0 for H7,H5,H11,H10",
            "paper_line": "~847",
            "artifact_value": {
                h: f.get("step3", {}).get("bos_mass_spearman", {}).get(h)
                for h in ["H7", "H5", "H11", "H10"]
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step3.bos_mass_spearman",
        },
        # ── modelfull Step 4 ────────────────────────────────────────────
        {
            "claim": "modelfull Step 4 pop Spearman = -1/+1",
            "paper_line": "~1214-1216",
            "artifact_value": f"{f.get('step4_5', {}).get('step4_pop_bos')}/{f.get('step4_5', {}).get('step4_pop_nonbos')}",
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step4_5",
        },
        {
            "claim": "modelfull Step 4 sample median = -0.999/+0.999",
            "paper_line": "~1217-1218",
            "artifact_value": f"{_r(f.get('step4_5', {}).get('step4_sample_median_bos'), 4)}/{_r(f.get('step4_5', {}).get('step4_sample_median_nonbos'), 4)}",
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step4_5",
        },
        # ── modelfull Step 5 ────────────────────────────────────────────
        {
            "claim": "modelfull Step 5 cos(w_head, d_nonBOS)=0.318, cos(w_head, d_delta)=0.200",
            "paper_line": "~919-920, ~1256-1259",
            "artifact_value": {
                "cos_nonbos": _r(f.get("step4_5", {}).get("step5_cos_head_nonbos"), 3),
                "cos_delta": _r(f.get("step4_5", {}).get("step5_cos_head_delta"), 3),
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step4_5",
        },
        {
            "claim": "modelfull write: baseline=0.999, ret1=0.460, ret2=0.951, abl1=0.625",
            "paper_line": "~966-971",
            "artifact_value": f.get("write"),
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.write",
        },
        # ── cross-model diagnostics ─────────────────────────────────────
        {
            "claim": "cos(d_nonBOS, d_delta) = 0.762 (minimal), 0.984 (full)",
            "paper_line": "~799 (minimal), ~862 (full)",
            "artifact_value": {
                "minimal": _r(a.get("step4_5", {}).get("cos_nonbos_delta"), 3),
                "full": _r(f.get("step4_5", {}).get("cos_nonbos_delta"), 3),
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> *.step4_5.cos_nonbos_delta",
        },
        {
            "claim": "cos(d_BOS, d_nonBOS) approx -0.97 (full)",
            "paper_line": "~838",
            "artifact_value": _r(f.get("step4_5", {}).get("cos_bos_nonbos"), 3),
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step4_5.cos_bos_nonbos",
        },
        {
            "claim": "Residual cancellation: 0.045 (minimal), 0.022 (full)",
            "paper_line": "~901",
            "artifact_value": {
                "minimal": _r(a.get("step4_5", {}).get("residual_abs_mean"), 3),
                "full": _r(f.get("step4_5", {}).get("residual_abs_mean"), 3),
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> *.step4_5.residual_abs_mean",
        },
        {
            "claim": "Concentration cos mean ge16: 0.901 (minimal), 0.938 (full)",
            "paper_line": "~902",
            "artifact_value": {
                "minimal": _r(
                    a.get("step4_5", {}).get("concentration_cos_mean_ge16"), 3
                ),
                "full": _r(f.get("step4_5", {}).get("concentration_cos_mean_ge16"), 3),
            },
            "artifact_path": "results/section5_support/section5_canonical.json -> *.step4_5.concentration_cos_mean_ge16",
        },
        # ── Step 5 MLP-zero control (P1-1) ──────────────────────────────
        {
            "claim": "modelminimal Step 5 MLP-zero control R^2",
            "paper_line": "~925-929 (MLP-zero paragraph, line 927)",
            "artifact_value": _r(
                a.get("step5_mlp_zero_control", {}).get("r2_no_mlp2"), 4
            ),
            "artifact_path": "results/section5_support/section5_canonical.json -> attn2_1h.step5_mlp_zero_control.r2_no_mlp2",
            "n_seq": results["metadata"]["final_n_seq"],
        },
        {
            "claim": "modelfull Step 5 MLP-zero control R^2",
            "paper_line": "~925-929 (MLP-zero paragraph, line 928)",
            "artifact_value": _r(
                f.get("step5_mlp_zero_control", {}).get("r2_no_mlp2"), 4
            ),
            "artifact_path": "results/section5_support/section5_canonical.json -> full_12h.step5_mlp_zero_control.r2_no_mlp2",
            "n_seq": results["metadata"]["final_n_seq"],
        },
    ]

    return {
        "generated_by": "analysis_scripts/experiments/section5/section5_canonical_support.py",
        "artifact": "results/section5_support/section5_canonical.json",
        "claims": claims,
    }


if __name__ == "__main__":
    main()
