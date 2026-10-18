"""Compute MAE for all paper-reported R² setups.

Matches the exact evaluation protocol from multiseed_position_mechanism_stats.py
and generate_write_bottleneck_plot.py. Uses the same checkpoints, data splits,
batch sizes, and random seeds.

Output: JSON and markdown table with R² (re-computed) and MAE for every setup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nanoGPT"))


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed creating import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Evaluation helpers ────────────────────────────────────────────────────────


def evaluate_final_decoding(
    model,
    data,
    n_batches: int,
    batch_size: int,
    device: str,
    bos_batch_fn,
) -> dict[str, float]:
    """Evaluate final position head (pos_head) decoding.

    Returns CoD R², correlation², MAE, and RMSE.
    """
    block_size = model.config.block_size
    positions = np.arange(block_size, dtype=np.float32)

    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    for _ in range(n_batches):
        tokens = bos_batch_fn(data, batch_size, block_size, device)
        with torch.no_grad():
            output, _ = model(tokens, targets=None, capture_taps=False)
        preds = output.squeeze(-1).detach().cpu().numpy()

        y_pred_chunks.append(preds.reshape(-1))
        y_true_chunks.append(np.tile(positions, batch_size))

    y_true = np.concatenate(y_true_chunks)
    y_pred = np.concatenate(y_pred_chunks)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_cod = 1.0 - (ss_res / ss_tot)

    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2_corr = r * r

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "r2_cod": r2_cod,
        "r2_corr": r2_corr,
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(y_true),
    }


def evaluate_ov_intervention(
    model,
    data,
    U: torch.Tensor,
    rank: int,
    intervention_type: str,
    n_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    forward_fn,
    get_batch_fn,
    basis_indices: Optional[list[int]] = None,
) -> dict[str, float]:
    """Evaluate OV intervention, returning both correlation² (as used in paper) and MAE."""
    all_preds = []
    all_positions = []

    for _ in range(n_batches):
        tokens = get_batch_fn(data, batch_size, block_size, device, force_bos=True)
        preds = forward_fn(
            model,
            tokens,
            U,
            rank,
            intervention_type,
            basis_indices=basis_indices,
        )
        positions = (
            torch.arange(block_size, device=device)
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        all_preds.append(preds.cpu())
        all_positions.append(positions.cpu())

    all_preds = torch.cat(all_preds, dim=0).flatten().numpy()
    all_positions = torch.cat(all_positions, dim=0).flatten().numpy()

    r = float(np.corrcoef(all_positions, all_preds)[0, 1])
    r2_corr = r * r

    ss_res = float(np.sum((all_positions - all_preds) ** 2))
    ss_tot = float(np.sum((all_positions - all_positions.mean()) ** 2))
    r2_cod = 1.0 - (ss_res / ss_tot)

    spearman_r = float(spearmanr(all_positions, all_preds).correlation)

    mae = float(np.mean(np.abs(all_positions - all_preds)))
    rmse = float(np.sqrt(np.mean((all_positions - all_preds) ** 2)))

    return {
        "r2_corr": r2_corr,
        "r2_cod": r2_cod,
        "spearman_r": spearman_r,
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(all_preds),
    }


def evaluate_step1_probe(
    model,
    data,
    train_batches: int,
    test_batches: int,
    batch_size: int,
    seed: int,
    device: str,
    inverse_eps: float,
    bos_mod,
) -> dict[str, float]:
    """Evaluate Step 1 linear probe and return R² and MAE from the full probe."""
    probe_out = bos_mod.run_probe_for_model(
        model_name="eval",
        model=model,
        data=data,
        train_batches=train_batches,
        test_batches=test_batches,
        batch_size=batch_size,
        seed=seed,
        device=device,
        inverse_eps=inverse_eps,
    )
    test_m = probe_out["full_linear_probe"]["test_metrics"]
    return {
        "r2_cod": test_m["r2"],
        "mae": test_m["mae"],
        "pearson_r": test_m["pearson_r"],
        "spearman_rho": test_m["spearman_rho"],
    }


def evaluate_1layer_baseline(
    checkpoint_path: Path,
    data: np.memmap,
    n_batches: int,
    batch_size: int,
    device: str,
    bos_batch_fn,
) -> dict[str, float]:
    """Evaluate 1-layer MLP-only baseline model."""
    from model_1layer_mechanism import OneLayerMechanismModel, OneLayerMechanismConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(OneLayerMechanismConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = OneLayerMechanismConfig(**filtered)
    model = OneLayerMechanismModel(config)

    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    block_size = config.block_size
    positions = np.arange(block_size, dtype=np.float32)

    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    for _ in range(n_batches):
        tokens = bos_batch_fn(data, batch_size, block_size, device)
        with torch.no_grad():
            preds, _ = model(tokens, targets=None)
        preds_np = preds.squeeze(-1).detach().cpu().numpy()
        y_pred_chunks.append(preds_np.reshape(-1))
        y_true_chunks.append(np.tile(positions, batch_size))

    y_true = np.concatenate(y_true_chunks)
    y_pred = np.concatenate(y_pred_chunks)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_cod = 1.0 - (ss_res / ss_tot)

    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2_corr = r * r

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "r2_cod": r2_cod,
        "r2_corr": r2_corr,
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(y_true),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compute MAE for all paper-reported R² setups"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--attn2_ckpt",
        type=str,
        default="model_backups/R2_ATTN2-1H/best_ckpt.pt",
    )
    parser.add_argument(
        "--full12h_ckpt",
        type=str,
        default="model_backups/R0_FULL-12H/best_ckpt.pt",
    )
    parser.add_argument(
        "--onelayer_ckpt",
        type=str,
        default="nanoGPT/out-1layer-mlp-only/0ql0hif7/best_ckpt.pt",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="nanoGPT/data/openwebtext",
    )
    # Match multiseed defaults
    parser.add_argument("--final_n_batches", type=int, default=10)
    parser.add_argument("--final_batch_size", type=int, default=64)
    parser.add_argument("--step1_train_batches", type=int, default=20)
    parser.add_argument("--step1_test_batches", type=int, default=10)
    parser.add_argument("--step1_batch_size", type=int, default=64)
    parser.add_argument("--write_n_batches", type=int, default=20)
    parser.add_argument("--write_batch_size", type=int, default=32)
    parser.add_argument("--inverse_eps", type=float, default=0.1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mae_all_setups",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load helper modules (same as multiseed script)
    bos_mod = _load_module(
        "bos_anchor_decodability_after_ln",
        ROOT
        / "analysis_scripts/experiments/interventions/bos_anchor_decodability_after_ln.py",
    )
    wb_mod = _load_module(
        "generate_write_bottleneck_plot",
        ROOT / "analysis_scripts/figures/generate_write_bottleneck_plot.py",
    )

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load models
    attn_ckpt = ROOT / args.attn2_ckpt
    full_ckpt = ROOT / args.full12h_ckpt
    onelayer_ckpt = ROOT / args.onelayer_ckpt

    for ckpt, name in [
        (attn_ckpt, "ATTN2-1H"),
        (full_ckpt, "FULL-12H"),
        (onelayer_ckpt, "1-layer"),
    ]:
        if not ckpt.exists():
            print(f"ERROR: Checkpoint not found: {ckpt} ({name})")
            sys.exit(1)

    print(f"Loading ATTN2-1H from {attn_ckpt}")
    attn_model = bos_mod.load_model(attn_ckpt, device)
    print(f"Loading FULL-12H from {full_ckpt}")
    full_model = bos_mod.load_model(full_ckpt, device)

    # Load data
    val_data = bos_mod.load_data(ROOT / args.val_data_dir)
    wb_data = wb_mod.load_owt_data(str(ROOT / args.val_data_dir))

    # Precompute SVD bases
    print("Computing SVD bases for write interventions...")
    U_attn, _, _ = wb_mod.get_block2_write_map_svd(attn_model)
    U_full, _, _ = wb_mod.get_block2_write_map_svd(full_model)

    results: dict[str, Any] = {"seed": args.seed, "setups": {}}

    # ── 1. Final decoding R² (Section 5) ────────────────────────────────────
    print("\n=== Final Decoding (Section 5) ===")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    r = evaluate_final_decoding(
        model=attn_model,
        data=val_data,
        n_batches=args.final_n_batches,
        batch_size=args.final_batch_size,
        device=device,
        bos_batch_fn=bos_mod.get_batch_with_bos,
    )
    results["setups"]["final_decoding_attn2_1h"] = r
    print(
        f"  ATTN2-1H: R²(CoD)={r['r2_cod']:.6f}, MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    r = evaluate_final_decoding(
        model=full_model,
        data=val_data,
        n_batches=args.final_n_batches,
        batch_size=args.final_batch_size,
        device=device,
        bos_batch_fn=bos_mod.get_batch_with_bos,
    )
    results["setups"]["final_decoding_full_12h"] = r
    print(
        f"  FULL-12H: R²(CoD)={r['r2_cod']:.6f}, MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}"
    )

    # ── 2. Step-1 linear probes (Section 5) ─────────────────────────────────
    print("\n=== Step-1 Linear Probes (Section 5) ===")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    r = evaluate_step1_probe(
        model=attn_model,
        data=val_data,
        train_batches=args.step1_train_batches,
        test_batches=args.step1_test_batches,
        batch_size=args.step1_batch_size,
        seed=args.seed,
        device=device,
        inverse_eps=args.inverse_eps,
        bos_mod=bos_mod,
    )
    results["setups"]["step1_probe_attn2_1h"] = r
    print(f"  ATTN2-1H: R²(CoD)={r['r2_cod']:.6f}, MAE={r['mae']:.4f}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    r = evaluate_step1_probe(
        model=full_model,
        data=val_data,
        train_batches=args.step1_train_batches,
        test_batches=args.step1_test_batches,
        batch_size=args.step1_batch_size,
        seed=args.seed,
        device=device,
        inverse_eps=args.inverse_eps,
        bos_mod=bos_mod,
    )
    results["setups"]["step1_probe_full_12h"] = r
    print(f"  FULL-12H: R²(CoD)={r['r2_cod']:.6f}, MAE={r['mae']:.4f}")

    # ── 3. OV Interventions (Section 6) ─────────────────────────────────────
    print("\n=== OV Interventions (Section 6) ===")

    ov_configs = [
        # (model, model_name, U, intervention_type, rank, basis_indices, label)
        (attn_model, "attn2_1h", U_attn, "retention", 768, None, "baseline"),
        (attn_model, "attn2_1h", U_attn, "retention", 1, None, "top1_retention"),
        (attn_model, "attn2_1h", U_attn, "retention", 2, None, "top2_retention"),
        (attn_model, "attn2_1h", U_attn, "ablation", 1, None, "top1_ablation"),
        (full_model, "full_12h", U_full, "retention", 768, None, "baseline"),
        (full_model, "full_12h", U_full, "retention", 1, [1], "top1_retention"),
        (full_model, "full_12h", U_full, "retention", 2, None, "top2_retention"),
        (full_model, "full_12h", U_full, "ablation", 1, [1], "top1_ablation"),
    ]

    for model, model_name, U, itype, rank, basis_idx, label in ov_configs:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        r = evaluate_ov_intervention(
            model=model,
            data=wb_data,
            U=U,
            rank=rank,
            intervention_type=itype,
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=model.config.block_size,
            device=device,
            forward_fn=wb_mod.forward_with_write_intervention,
            get_batch_fn=wb_mod.get_batch,
            basis_indices=basis_idx,
        )
        key = f"ov_{label}_{model_name}"
        results["setups"][key] = r
        print(
            f"  {model_name} {label}: R²(CoD)={r['r2_cod']:.6f}, "
            f"Spearman={r['spearman_r']:.6f}, MAE={r['mae']:.4f}"
        )

    # ── 4. 1-Layer Baseline (Appendix A.2) ──────────────────────────────────
    print("\n=== 1-Layer Baseline (Appendix A.2) ===")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    r = evaluate_1layer_baseline(
        checkpoint_path=onelayer_ckpt,
        data=val_data,
        n_batches=args.final_n_batches,
        batch_size=args.final_batch_size,
        device=device,
        bos_batch_fn=bos_mod.get_batch_with_bos,
    )
    results["setups"]["onelayer_baseline"] = r
    print(
        f"  1-Layer: R²(CoD)={r['r2_cod']:.6f}, MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}"
    )

    # ── Save results ────────────────────────────────────────────────────────
    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    json_path = save_dir / "mae_all_setups.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON: {json_path}")

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY TABLE")
    print("=" * 110)
    header = f"{'Setup':<45} {'Paper R²':<12} {'R²(CoD)':<12} {'Spearman':<12} {'MAE':<10} {'RMSE':<10}"
    print(header)
    print("-" * 110)

    # Paper-reported values (from main.tex)
    paper_r2 = {
        "final_decoding_attn2_1h": ("0.990", "CoD"),
        "final_decoding_full_12h": ("0.999", "CoD"),
        "step1_probe_attn2_1h": ("0.554", "CoD"),
        "step1_probe_full_12h": ("0.560", "CoD"),
        "ov_baseline_attn2_1h": ("0.990", "corr²"),
        "ov_top1_retention_attn2_1h": ("0.810", "corr²"),
        "ov_top2_retention_attn2_1h": ("0.991", "corr²"),
        "ov_top1_ablation_attn2_1h": ("0.125", "corr²"),
        "ov_baseline_full_12h": ("0.999", "corr²"),
        "ov_top1_retention_full_12h": ("0.460", "corr²"),
        "ov_top2_retention_full_12h": ("0.951", "corr²"),
        "ov_top1_ablation_full_12h": ("0.625", "corr²"),
        "onelayer_baseline": ("0.927", "CoD"),
    }

    for key, metrics in results["setups"].items():
        paper_val, paper_type = paper_r2.get(key, ("—", "—"))
        r2_cod = metrics.get("r2_cod", float("nan"))
        spearman = metrics.get("spearman_r", metrics.get("spearman_rho", float("nan")))
        mae = metrics.get("mae", float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        print(
            f"{key:<45} {paper_val:<12} {r2_cod:<12.6f} {spearman:<12.6f} {mae:<10.4f} {rmse:<10.4f}"
        )

    # Write markdown table
    md_lines = [
        "# Metrics for All Paper-Reported R² Setups",
        "",
        f"Seed: {args.seed}",
        "",
        "| Setup | Paper R² | R²(CoD) | Spearman r | MAE | RMSE |",
        "|-------|----------|---------|------------|-----|------|",
    ]
    for key, metrics in results["setups"].items():
        paper_val, paper_type = paper_r2.get(key, ("—", "—"))
        r2_cod = metrics.get("r2_cod", float("nan"))
        spearman = metrics.get("spearman_r", metrics.get("spearman_rho", float("nan")))
        mae = metrics.get("mae", float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        md_lines.append(
            f"| {key} | {paper_val} ({paper_type}) | {r2_cod:.6f} | {spearman:.6f} | {mae:.4f} | {rmse:.4f} |"
        )
    md_lines.append("")
    md_lines.append(
        "**Note**: Paper Section 6 (OV interventions) currently uses corr² in code but claims CoD in text. "
        "Spearman r is recommended for Section 6 to measure information retention."
    )

    md_path = save_dir / "mae_all_setups.md"
    md_path.write_text("\n".join(md_lines))
    print(f"\nSaved markdown: {md_path}")


if __name__ == "__main__":
    main()
