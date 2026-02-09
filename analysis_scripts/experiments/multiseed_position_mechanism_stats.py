"""Compute multi-seed robustness statistics for paper-reported mechanism metrics.

This script re-evaluates key quantities used in the paper across multiple random
seeds and reports mean +- std over seeds.

Metrics covered:
1) Step-1 decodability after succeeding LayerNorm (Section 5):
   - population mean-curve Spearman (BOS/non-BOS)
   - sample-level median Spearman (BOS/non-BOS)
   - full linear probe R^2 and random-label baseline R^2
2) Step-4 directional monotonicity at Layer-2 attention output:
   - population mean-curve Spearman (BOS/non-BOS)
   - sample-level median Spearman (BOS/non-BOS)
3) Write-subspace intervention (Section 6):
   - baseline R^2
   - top-1 retention R^2
   - top-2 retention R^2
   - top-1 ablation R^2
   - r_95 over ranks {1,2}

The script saves:
- JSON: results/paper_multiseed_stats/multiseed_mechanism_stats.json
- Markdown summary: results/paper_multiseed_stats/multiseed_mechanism_stats.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed creating import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean_std(values: list[float]) -> dict[str, float | list[float]]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "values": []}
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0, "values": [float(arr[0])]}
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "values": [float(v) for v in arr.tolist()],
        }
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)),
        "values": [float(v) for v in arr.tolist()],
    }


def _summarize_scalars(metric_dict: dict[str, list[float]]) -> dict[str, Any]:
    return {k: _mean_std(v) for k, v in metric_dict.items()}


def _compute_r95_from_r1_r2(
    baseline_r2: float,
    top1_retention_r2: float,
    top2_retention_r2: float,
) -> int | None:
    threshold = 0.95 * baseline_r2
    if top1_retention_r2 >= threshold:
        return 1
    if top2_retention_r2 >= threshold:
        return 2
    return None


def _evaluate_final_r2(
    model,
    data,
    n_batches: int,
    batch_size: int,
    device: str,
    bos_batch_fn,
) -> float:
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
    return 1.0 - (ss_res / ss_tot)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run(args: argparse.Namespace) -> dict[str, Any]:
    bos_mod = _load_module(
        "bos_anchor_decodability_after_ln",
        ROOT
        / "analysis_scripts/experiments/interventions/bos_anchor_decodability_after_ln.py",
    )
    proj_mod = _load_module(
        "r0_attention_output_projection",
        ROOT
        / "analysis_scripts/experiments/r0_analysis/r0_attention_output_projection.py",
    )
    wb_mod = _load_module(
        "generate_write_bottleneck_plot",
        ROOT / "analysis_scripts/figures/generate_write_bottleneck_plot.py",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    attn_ckpt = ROOT / args.attn2_ckpt
    full_ckpt = ROOT / args.full12h_ckpt

    print(f"Using device: {device}")
    print(f"Loading checkpoints:\n  ATTN2-1H: {attn_ckpt}\n  FULL-12H: {full_ckpt}")

    # Load models once.
    attn_model = bos_mod.load_model(attn_ckpt, device)
    full_model = bos_mod.load_model(full_ckpt, device)

    # Data sources used by original scripts.
    val_data = bos_mod.load_data(ROOT / args.val_data_dir)
    train_data = proj_mod.load_data(ROOT / args.train_data_dir)
    wb_data = wb_mod.load_owt_data(str(ROOT / args.val_data_dir))

    # Precompute SVD bases for write interventions.
    U_attn, _, _ = wb_mod.get_block2_write_map_svd(attn_model)
    U_full, _, _ = wb_mod.get_block2_write_map_svd(full_model)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    print(f"Running seeds: {seeds}")

    raw: dict[str, dict[str, list[float]]] = {
        "attn2_1h": {
            "final_r2": [],
            "step1_pop_bos": [],
            "step1_pop_nonbos": [],
            "step1_sample_median_bos": [],
            "step1_sample_median_nonbos": [],
            "step1_full_probe_r2": [],
            "step1_random_label_r2": [],
            "step4_pop_bos": [],
            "step4_pop_nonbos": [],
            "step4_sample_median_bos": [],
            "step4_sample_median_nonbos": [],
            "write_baseline_r2": [],
            "write_top1_retention_r2": [],
            "write_top2_retention_r2": [],
            "write_top1_ablation_r2": [],
            "write_r95": [],
        },
        "full_12h": {
            "final_r2": [],
            "step1_pop_bos": [],
            "step1_pop_nonbos": [],
            "step1_sample_median_bos": [],
            "step1_sample_median_nonbos": [],
            "step1_full_probe_r2": [],
            "step1_random_label_r2": [],
            "step4_pop_bos": [],
            "step4_pop_nonbos": [],
            "step4_sample_median_bos": [],
            "step4_sample_median_nonbos": [],
            "write_baseline_r2": [],
            "write_top1_retention_r2": [],
            "write_top2_retention_r2": [],
            "write_top1_ablation_r2": [],
            "write_r95": [],
        },
    }

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Final decoding R^2 (same coefficient-of-determination metric as paper text).
        attn_final_r2 = _evaluate_final_r2(
            model=attn_model,
            data=val_data,
            n_batches=args.final_n_batches,
            batch_size=args.final_batch_size,
            device=device,
            bos_batch_fn=bos_mod.get_batch_with_bos,
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        full_final_r2 = _evaluate_final_r2(
            model=full_model,
            data=val_data,
            n_batches=args.final_n_batches,
            batch_size=args.final_batch_size,
            device=device,
            bos_batch_fn=bos_mod.get_batch_with_bos,
        )
        raw["attn2_1h"]["final_r2"].append(float(attn_final_r2))
        raw["full_12h"]["final_r2"].append(float(full_final_r2))

        print(
            f"  Final decoding R^2 (ATTN2-1H/FULL-12H): "
            f"{attn_final_r2:.4f} / {full_final_r2:.4f}"
        )

        # Step-1 decodability metrics.
        attn_probe = bos_mod.run_probe_for_model(
            model_name="ATTN2-1H",
            model=attn_model,
            data=val_data,
            train_batches=args.step1_train_batches,
            test_batches=args.step1_test_batches,
            batch_size=args.step1_batch_size,
            seed=seed,
            device=device,
            inverse_eps=args.inverse_eps,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        full_probe = bos_mod.run_probe_for_model(
            model_name="FULL-12H",
            model=full_model,
            data=val_data,
            train_batches=args.step1_train_batches,
            test_batches=args.step1_test_batches,
            batch_size=args.step1_batch_size,
            seed=seed,
            device=device,
            inverse_eps=args.inverse_eps,
        )

        for model_key, probe_out in [
            ("attn2_1h", attn_probe),
            ("full_12h", full_probe),
        ]:
            bos_stats = probe_out["metrics"]["bos_b1ln"]["projection_spearman"]
            non_stats = probe_out["metrics"]["nonbos_b1ln"]["projection_spearman"]
            full_lin = probe_out["full_linear_probe"]

            raw[model_key]["step1_pop_bos"].append(
                float(bos_stats["population_mean_curve_rho"])
            )
            raw[model_key]["step1_pop_nonbos"].append(
                float(non_stats["population_mean_curve_rho"])
            )
            raw[model_key]["step1_sample_median_bos"].append(
                float(bos_stats["sample_sequence_median_rho"])
            )
            raw[model_key]["step1_sample_median_nonbos"].append(
                float(non_stats["sample_sequence_median_rho"])
            )
            raw[model_key]["step1_full_probe_r2"].append(
                float(full_lin["test_metrics"]["r2"])
            )
            raw[model_key]["step1_random_label_r2"].append(
                float(full_lin["random_label_test_metrics"]["r2"])
            )

        # Step-4 projection monotonicity metrics.
        torch.manual_seed(seed)
        np.random.seed(seed)
        attn_proj = proj_mod.compute_projection(
            attn_model,
            train_data,
            128,
            args.step4_batch_size,
            args.step4_n_batches,
            device,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        full_proj = proj_mod.compute_projection(
            full_model,
            train_data,
            128,
            args.step4_batch_size,
            args.step4_n_batches,
            device,
        )

        for model_key, proj_out in [("attn2_1h", attn_proj), ("full_12h", full_proj)]:
            raw[model_key]["step4_pop_bos"].append(float(proj_out["spearman_bos"]))
            raw[model_key]["step4_pop_nonbos"].append(
                float(proj_out["spearman_non_bos"])
            )
            raw[model_key]["step4_sample_median_bos"].append(
                float(proj_out["spearman_stats"]["bos"]["sample_sequence_median_rho"])
            )
            raw[model_key]["step4_sample_median_nonbos"].append(
                float(
                    proj_out["spearman_stats"]["non_bos"]["sample_sequence_median_rho"]
                )
            )

        # Write intervention metrics.
        # ATTN2-1H (no rank-1 override).
        torch.manual_seed(seed)
        np.random.seed(seed)
        attn_baseline = wb_mod.compute_r2_at_rank(
            attn_model,
            wb_data,
            U_attn,
            768,
            "retention",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=attn_model.config.block_size,
        )
        attn_ret1 = wb_mod.compute_r2_at_rank(
            attn_model,
            wb_data,
            U_attn,
            1,
            "retention",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=attn_model.config.block_size,
        )
        attn_ret2 = wb_mod.compute_r2_at_rank(
            attn_model,
            wb_data,
            U_attn,
            2,
            "retention",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=attn_model.config.block_size,
        )
        attn_abl1 = wb_mod.compute_r2_at_rank(
            attn_model,
            wb_data,
            U_attn,
            1,
            "ablation",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=attn_model.config.block_size,
        )
        attn_r95 = _compute_r95_from_r1_r2(attn_baseline, attn_ret1, attn_ret2)

        raw["attn2_1h"]["write_baseline_r2"].append(float(attn_baseline))
        raw["attn2_1h"]["write_top1_retention_r2"].append(float(attn_ret1))
        raw["attn2_1h"]["write_top2_retention_r2"].append(float(attn_ret2))
        raw["attn2_1h"]["write_top1_ablation_r2"].append(float(attn_abl1))
        raw["attn2_1h"]["write_r95"].append(
            float(attn_r95) if attn_r95 is not None else float("nan")
        )

        # FULL-12H (keep rank-1 override = index 1 to match paper setup).
        torch.manual_seed(seed)
        np.random.seed(seed)
        full_baseline = wb_mod.compute_r2_at_rank(
            full_model,
            wb_data,
            U_full,
            768,
            "retention",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=full_model.config.block_size,
        )
        full_ret1 = wb_mod.compute_r2_at_rank(
            full_model,
            wb_data,
            U_full,
            1,
            "retention",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=full_model.config.block_size,
            basis_indices=[1],
        )
        full_ret2 = wb_mod.compute_r2_at_rank(
            full_model,
            wb_data,
            U_full,
            2,
            "retention",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=full_model.config.block_size,
        )
        full_abl1 = wb_mod.compute_r2_at_rank(
            full_model,
            wb_data,
            U_full,
            1,
            "ablation",
            n_batches=args.write_n_batches,
            batch_size=args.write_batch_size,
            block_size=full_model.config.block_size,
            basis_indices=[1],
        )
        full_r95 = _compute_r95_from_r1_r2(full_baseline, full_ret1, full_ret2)

        raw["full_12h"]["write_baseline_r2"].append(float(full_baseline))
        raw["full_12h"]["write_top1_retention_r2"].append(float(full_ret1))
        raw["full_12h"]["write_top2_retention_r2"].append(float(full_ret2))
        raw["full_12h"]["write_top1_ablation_r2"].append(float(full_abl1))
        raw["full_12h"]["write_r95"].append(
            float(full_r95) if full_r95 is not None else float("nan")
        )

    summary = {
        "seeds": seeds,
        "config": {
            "step1_train_batches": args.step1_train_batches,
            "step1_test_batches": args.step1_test_batches,
            "step1_batch_size": args.step1_batch_size,
            "step4_n_batches": args.step4_n_batches,
            "step4_batch_size": args.step4_batch_size,
            "write_n_batches": args.write_n_batches,
            "write_batch_size": args.write_batch_size,
            "final_n_batches": args.final_n_batches,
            "final_batch_size": args.final_batch_size,
            "inverse_eps": args.inverse_eps,
            "attn2_ckpt": str(attn_ckpt),
            "full12h_ckpt": str(full_ckpt),
        },
        "attn2_1h": _summarize_scalars(raw["attn2_1h"]),
        "full_12h": _summarize_scalars(raw["full_12h"]),
    }

    return {"raw": raw, "summary": summary}


def _format_pm(mean_value: float, std_value: float, digits: int = 3) -> str:
    return f"{mean_value:.{digits}f} +- {std_value:.{digits}f}"


def write_markdown(summary: dict[str, Any], save_path: Path) -> None:
    attn = summary["attn2_1h"]
    full = summary["full_12h"]

    lines = [
        "# Multi-Seed Mechanism Statistics",
        "",
        f"Seeds: {summary['seeds']}",
        "",
        "## Section 5 (Step-1 and Step-4)",
        "",
        "### ATTN2-1H",
        "",
        f"- Step-1 population Spearman (BOS/non-BOS): {_format_pm(attn['step1_pop_bos']['mean'], attn['step1_pop_bos']['std'])} / {_format_pm(attn['step1_pop_nonbos']['mean'], attn['step1_pop_nonbos']['std'])}",
        f"- Final decoding R^2: {_format_pm(attn['final_r2']['mean'], attn['final_r2']['std'])}",
        f"- Step-1 sample median Spearman (BOS/non-BOS): {_format_pm(attn['step1_sample_median_bos']['mean'], attn['step1_sample_median_bos']['std'])} / {_format_pm(attn['step1_sample_median_nonbos']['mean'], attn['step1_sample_median_nonbos']['std'])}",
        f"- Step-1 full probe R^2: {_format_pm(attn['step1_full_probe_r2']['mean'], attn['step1_full_probe_r2']['std'])}",
        f"- Step-1 random-label baseline R^2: {_format_pm(attn['step1_random_label_r2']['mean'], attn['step1_random_label_r2']['std'])}",
        f"- Step-4 population Spearman (BOS/non-BOS): {_format_pm(attn['step4_pop_bos']['mean'], attn['step4_pop_bos']['std'])} / {_format_pm(attn['step4_pop_nonbos']['mean'], attn['step4_pop_nonbos']['std'])}",
        f"- Step-4 sample median Spearman (BOS/non-BOS): {_format_pm(attn['step4_sample_median_bos']['mean'], attn['step4_sample_median_bos']['std'])} / {_format_pm(attn['step4_sample_median_nonbos']['mean'], attn['step4_sample_median_nonbos']['std'])}",
        "",
        "### FULL-12H",
        "",
        f"- Step-1 population Spearman (BOS/non-BOS): {_format_pm(full['step1_pop_bos']['mean'], full['step1_pop_bos']['std'])} / {_format_pm(full['step1_pop_nonbos']['mean'], full['step1_pop_nonbos']['std'])}",
        f"- Final decoding R^2: {_format_pm(full['final_r2']['mean'], full['final_r2']['std'])}",
        f"- Step-1 sample median Spearman (BOS/non-BOS): {_format_pm(full['step1_sample_median_bos']['mean'], full['step1_sample_median_bos']['std'])} / {_format_pm(full['step1_sample_median_nonbos']['mean'], full['step1_sample_median_nonbos']['std'])}",
        f"- Step-1 full probe R^2: {_format_pm(full['step1_full_probe_r2']['mean'], full['step1_full_probe_r2']['std'])}",
        f"- Step-1 random-label baseline R^2: {_format_pm(full['step1_random_label_r2']['mean'], full['step1_random_label_r2']['std'])}",
        f"- Step-4 population Spearman (BOS/non-BOS): {_format_pm(full['step4_pop_bos']['mean'], full['step4_pop_bos']['std'])} / {_format_pm(full['step4_pop_nonbos']['mean'], full['step4_pop_nonbos']['std'])}",
        f"- Step-4 sample median Spearman (BOS/non-BOS): {_format_pm(full['step4_sample_median_bos']['mean'], full['step4_sample_median_bos']['std'])} / {_format_pm(full['step4_sample_median_nonbos']['mean'], full['step4_sample_median_nonbos']['std'])}",
        "",
        "## Section 6 (Write-Subspace Intervention)",
        "",
        "### ATTN2-1H",
        "",
        f"- Baseline R^2: {_format_pm(attn['write_baseline_r2']['mean'], attn['write_baseline_r2']['std'])}",
        f"- Top-1 retention R^2: {_format_pm(attn['write_top1_retention_r2']['mean'], attn['write_top1_retention_r2']['std'])}",
        f"- Top-2 retention R^2: {_format_pm(attn['write_top2_retention_r2']['mean'], attn['write_top2_retention_r2']['std'])}",
        f"- Top-1 ablation R^2: {_format_pm(attn['write_top1_ablation_r2']['mean'], attn['write_top1_ablation_r2']['std'])}",
        f"- r_95 mean +- std: {_format_pm(attn['write_r95']['mean'], attn['write_r95']['std'])}",
        "",
        "### FULL-12H",
        "",
        f"- Baseline R^2: {_format_pm(full['write_baseline_r2']['mean'], full['write_baseline_r2']['std'])}",
        f"- Top-1 retention R^2: {_format_pm(full['write_top1_retention_r2']['mean'], full['write_top1_retention_r2']['std'])}",
        f"- Top-2 retention R^2: {_format_pm(full['write_top2_retention_r2']['mean'], full['write_top2_retention_r2']['std'])}",
        f"- Top-1 ablation R^2: {_format_pm(full['write_top1_ablation_r2']['mean'], full['write_top1_ablation_r2']['std'])}",
        f"- r_95 mean +- std: {_format_pm(full['write_r95']['mean'], full['write_r95']['std'])}",
        "",
    ]
    save_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument(
        "--attn2_ckpt", type=str, default="model_backups/R2_ATTN2-1H/best_ckpt.pt"
    )
    parser.add_argument(
        "--full12h_ckpt", type=str, default="model_backups/R0_FULL-12H/best_ckpt.pt"
    )
    parser.add_argument("--val_data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--train_data_dir", type=str, default="nanoGPT/data/openwebtext"
    )
    parser.add_argument("--step1_train_batches", type=int, default=20)
    parser.add_argument("--step1_test_batches", type=int, default=10)
    parser.add_argument("--step1_batch_size", type=int, default=64)
    parser.add_argument("--step4_n_batches", type=int, default=10)
    parser.add_argument("--step4_batch_size", type=int, default=32)
    parser.add_argument("--write_n_batches", type=int, default=20)
    parser.add_argument("--write_batch_size", type=int, default=32)
    parser.add_argument("--final_n_batches", type=int, default=10)
    parser.add_argument("--final_batch_size", type=int, default=64)
    parser.add_argument("--inverse_eps", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="results/paper_multiseed_stats")
    args = parser.parse_args()

    out = run(args)

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "multiseed_mechanism_stats.json"
    md_path = save_dir / "multiseed_mechanism_stats.md"

    json_path.write_text(json.dumps(_to_jsonable(out), indent=2))
    write_markdown(out["summary"], md_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
