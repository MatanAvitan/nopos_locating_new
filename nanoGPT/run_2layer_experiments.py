#!/usr/bin/env python3
"""
Complete Experiment Pipeline for 2-Layer NoPE Mechanism Study

This script runs the full experimental pipeline:
1. Experiment 1: Train all freezing regimes (R0-R4)
2. Experiments 2-6: Run comprehensive analysis on trained models

All results are logged to WandB project: nope-2layer-mechanism

Usage:
    # Run complete pipeline
    ./run_2layer_experiments.py --wandb --device cuda:0

    # Run only training (Exp 1)
    ./run_2layer_experiments.py --train_only --wandb

    # Run only analysis (Exp 2-6) on existing checkpoints
    ./run_2layer_experiments.py --analyze_only --checkpoint_dir out-2layer-mechanism --wandb
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_experiment1_training(args):
    """Run Experiment 1: All freezing regimes."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: MECHANISM DISSECTION VIA FREEZE/TRAIN MATRIX")
    print("=" * 80)

    regimes = ["R0", "R1", "R2", "R3", "R4_linear", "R4_mlp"]

    for regime in regimes:
        print(f"\n{'=' * 60}")
        print(f"Training Regime: {regime}")
        print(f"{'=' * 60}")

        cmd = [
            "python",
            "train_2layer_mechanism.py",
            "--regime",
            regime,
            "--max_iters",
            str(args.max_iters),
            "--batch_size",
            str(args.batch_size),
            "--block_size",
            str(args.block_size),
            "--seed",
            str(args.seed),
            "--out_dir",
            args.out_dir,
            "--device",
            args.device,
        ]

        if args.wandb:
            cmd.append("--wandb")

        if args.no_compile:
            cmd.append("--no_compile")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        if result.returncode != 0:
            print(f"ERROR: Regime {regime} failed!")
            if not args.continue_on_error:
                return False

    return True


def run_experiments_2_6_analysis(args, checkpoint_dir: str):
    """Run Experiments 2-6 on trained models."""
    print("\n" + "=" * 80)
    print("EXPERIMENTS 2-6: MECHANISM ANALYSIS")
    print("=" * 80)

    # Find best checkpoints for each regime
    regimes = ["R0", "R1", "R2", "R3", "R4_linear", "R4_mlp"]

    for regime in regimes:
        ckpt_path = os.path.join(checkpoint_dir, regime, "best_ckpt.pt")

        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found for {regime}: {ckpt_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Analyzing Regime: {regime}")
        print(f"{'=' * 60}")

        out_analysis_dir = os.path.join(checkpoint_dir, regime, "analysis")

        cmd = [
            "python",
            "analyze_2layer_mechanism.py",
            "--checkpoint",
            ckpt_path,
            "--out_dir",
            out_analysis_dir,
            "--device",
            args.device,
        ]

        if args.wandb:
            cmd.append("--wandb")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        if result.returncode != 0:
            print(f"WARNING: Analysis for {regime} failed!")
            if not args.continue_on_error:
                return False

    return True


def create_summary_report(args, checkpoint_dir: str):
    """Create comprehensive summary report."""
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORT")
    print("=" * 80)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "experiment1_results": {},
        "experiments_2_6_results": {},
        "hypothesis_conclusions": {},
    }

    regimes = ["R0", "R1", "R2", "R3", "R4_linear", "R4_mlp"]

    # Load Experiment 1 results
    for regime in regimes:
        ckpt_path = os.path.join(checkpoint_dir, regime, "best_ckpt.pt")
        if os.path.exists(ckpt_path):
            import torch

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "best_metrics" in ckpt:
                summary["experiment1_results"][regime] = {
                    "val_mae": ckpt["best_metrics"].get("val_mae"),
                    "val_r2": ckpt["best_metrics"].get("val_r2"),
                    "val_loss": ckpt["best_metrics"].get("val_loss"),
                }

    # Load Experiments 2-6 results
    for regime in regimes:
        analysis_path = os.path.join(
            checkpoint_dir, regime, "analysis", "analysis_results.json"
        )
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                summary["experiments_2_6_results"][regime] = json.load(f)

    # Derive hypothesis conclusions
    if summary["experiment1_results"]:
        r0_r2 = summary["experiment1_results"].get("R0", {}).get("val_r2", 0)
        r1_r2 = summary["experiment1_results"].get("R1", {}).get("val_r2", 0)
        r2_r2 = summary["experiment1_results"].get("R2", {}).get("val_r2", 0)
        r3_r2 = summary["experiment1_results"].get("R3", {}).get("val_r2", 0)
        r4l_r2 = summary["experiment1_results"].get("R4_linear", {}).get("val_r2", 0)
        r4m_r2 = summary["experiment1_results"].get("R4_mlp", {}).get("val_r2", 0)

        conclusions = []

        # H1/H2: Attention mechanism
        if r2_r2 > 0.7:
            if r3_r2 < 0.3:
                conclusions.append(
                    "STRONG SUPPORT for H1/H2: Attn2 is necessary (R2 >> R3)"
                )
            else:
                conclusions.append(
                    "PARTIAL SUPPORT for H1/H2: Attn2 helps but MLP2 also works"
                )
        else:
            conclusions.append(
                "WEAK SUPPORT for H1/H2: Attn2-only does not achieve high performance"
            )

        # H3: Shortcut
        if r4l_r2 > 0.5 or r4m_r2 > 0.5:
            conclusions.append(
                "WARNING: H3 possible - head-only probe achieves reasonable R²"
            )
        else:
            conclusions.append("H3 unlikely: head-only probe fails")

        # Compare R1 vs R2 vs R3
        conclusions.append(f"R1 (Block2-only) R² = {r1_r2:.4f}")
        conclusions.append(f"R2 (Attn2-only) R² = {r2_r2:.4f}")
        conclusions.append(f"R3 (MLP2-only) R² = {r3_r2:.4f}")

        summary["hypothesis_conclusions"]["from_exp1"] = conclusions

    # Add conclusions from Exp 2-6 if available
    for regime in ["R1", "R2"]:
        if regime in summary["experiments_2_6_results"]:
            exp_results = summary["experiments_2_6_results"][regime]

            if "experiment2" in exp_results:
                exp2 = exp_results["experiment2"]
                if "summary" in exp2 and "hypothesis_support" in exp2["summary"]:
                    summary["hypothesis_conclusions"]["exp2_" + regime] = exp2[
                        "summary"
                    ]["hypothesis_support"]

            if "experiment3" in exp_results:
                exp3 = exp_results["experiment3"]
                if "summary" in exp3 and "hypothesis_support" in exp3["summary"]:
                    summary["hypothesis_conclusions"]["exp3_" + regime] = exp3[
                        "summary"
                    ]["hypothesis_support"]

            if "experiment6" in exp_results:
                exp6 = exp_results["experiment6"]
                if "summary" in exp6 and "hypothesis_support" in exp6["summary"]:
                    summary["hypothesis_conclusions"]["exp6_" + regime] = exp6[
                        "summary"
                    ]["hypothesis_support"]

    # Save summary
    summary_path = os.path.join(checkpoint_dir, "full_experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Print conclusions
    print("\n" + "=" * 80)
    print("HYPOTHESIS CONCLUSIONS")
    print("=" * 80)

    for key, value in summary["hypothesis_conclusions"].items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="2-Layer NoPE Mechanism Experiment Pipeline"
    )

    # Mode selection
    parser.add_argument(
        "--train_only", action="store_true", help="Only run training (Experiment 1)"
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only run analysis (Experiments 2-6)",
    )

    # WandB
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")

    # Training config
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20000,
        help="Max training iterations per regime",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # Paths
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out-2layer-mechanism",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory with existing checkpoints (for analyze_only)",
    )

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue even if a regime fails",
    )

    args = parser.parse_args()

    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = args.out_dir

    # Create output directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("2-LAYER NOPE MECHANISM STUDY")
    print("=" * 80)
    print(f"\nWandB Project: nope-2layer-mechanism")
    print(f"Output Directory: {checkpoint_dir}")
    print(f"Device: {args.device}")
    print(f"Max Iterations: {args.max_iters}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Block Size: {args.block_size}")

    success = True

    # Run training if requested
    if not args.analyze_only:
        success = run_experiment1_training(args)
        if not success:
            print("\nERROR: Training failed!")
            return 1

    # Run analysis if requested
    if not args.train_only:
        success = run_experiments_2_6_analysis(args, checkpoint_dir)
        if not success:
            print("\nWARNING: Some analyses failed!")

    # Create summary report
    summary = create_summary_report(args, checkpoint_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {checkpoint_dir}/")
    print(f"Summary: {checkpoint_dir}/full_experiment_summary.json")

    if args.wandb:
        print(
            f"\nView WandB dashboard at: https://wandb.ai/[your-entity]/nope-2layer-mechanism"
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
