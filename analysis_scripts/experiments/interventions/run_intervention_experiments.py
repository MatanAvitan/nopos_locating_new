"""
Master Runner for Intervention Experiments

This script runs all intervention experiments to verify whether the model
uses the proposed mechanisms for position decoding:

1. Mechanism 1 (Decoding Vector): w = W_V · Σ_j LN(E_j)
2. Mechanism 2 (Population Mean): E[LN(h_i)] differs by position

Experiments:
- B4: Train probe on mean-subtracted activations (PRIORITY)
- B1: Position-specific mean subtraction ablation
- A1: Decoding vector direction ablation
- B2: Cross-position mean patching
- B3: Mean-only prediction
- A2: Value vector corruption
- A3: Orthogonality check

Usage:
    python run_intervention_experiments.py --setting synthetic
    python run_intervention_experiments.py --setting natural_language
    python run_intervention_experiments.py --all
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
ARTIFACTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/artifacts")
PLOTS_DIR = Path("/home/nlp/matan_avitan/git/nopos_locating_new/overleaf/nopos---claude-version/plots")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_population_means(setting, d_model, d_vocab, n_ctx, n_samples=5000):
    """Run compute_population_means.py"""
    print("\n" + "=" * 70)
    print(f"Step 1: Computing Population Means ({setting})")
    print("=" * 70)

    from compute_population_means import create_synthetic_model, compute_population_means, analyze_population_means

    model = create_synthetic_model(d_model, d_vocab, n_ctx)
    pop_means, pop_stds = compute_population_means(model, n_samples, n_ctx, d_vocab)
    analysis = analyze_population_means(pop_means, pop_stds)

    # Save
    save_path = ARTIFACTS_DIR / f"population_means_{setting}.pt"
    torch.save({
        'pop_means': pop_means,
        'pop_stds': pop_stds,
        'analysis': analysis,
        'config': {'setting': setting, 'd_model': d_model, 'd_vocab': d_vocab, 'n_ctx': n_ctx}
    }, save_path)

    print(f"\nPopulation mean analysis:")
    print(f"  Pearson r with position: {analysis['overall_pearson_r']:.4f}")
    print(f"  Saved to: {save_path}")

    return pop_means, analysis


def run_experiment_b4(setting, d_model, d_vocab, n_ctx, n_train, n_test, epochs):
    """Run Experiment B4: Mean-Subtracted Probe Training"""
    print("\n" + "=" * 70)
    print("Step 2: Experiment B4 - Mean-Subtracted Probe Training")
    print("=" * 70)

    from train_mean_subtracted_probe import run_experiment
    import argparse

    args = argparse.Namespace(
        setting=setting, d_model=d_model, d_vocab=d_vocab, n_ctx=n_ctx,
        n_train=n_train, n_test=n_test, epochs=epochs, seed=42, skip_baseline=False
    )

    results = run_experiment(args)
    return results


def run_experiment_b_series(setting, d_model, d_vocab, n_ctx, n_train, n_test):
    """Run Experiments B1, B2, B3"""
    print("\n" + "=" * 70)
    print("Step 3: Experiments B1, B2, B3 - Mean Ablation Series")
    print("=" * 70)

    from mean_subtraction_ablation import run_experiments
    import argparse

    args = argparse.Namespace(
        setting=setting, d_model=d_model, d_vocab=d_vocab, n_ctx=n_ctx,
        n_train=n_train, n_test=n_test, seed=42
    )

    results = run_experiments(args)
    return results


def run_experiment_a_series(setting, d_model, d_vocab, n_ctx, n_train, n_test):
    """Run Experiments A1, A2, A3"""
    print("\n" + "=" * 70)
    print("Step 4: Experiments A1, A2, A3 - Decoding Vector Ablation Series")
    print("=" * 70)

    from decoding_vector_ablation import run_experiments
    import argparse

    args = argparse.Namespace(
        setting=setting, d_model=d_model, d_vocab=d_vocab, n_ctx=n_ctx,
        n_train=n_train, n_test=n_test, seed=42
    )

    results = run_experiments(args)
    return results


def generate_summary_table(all_results, settings):
    """Generate summary table of all results."""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 70)

    for setting in settings:
        if setting not in all_results:
            continue

        results = all_results[setting]

        print(f"\n{'='*30} {setting.upper()} {'='*30}")

        # B4 Results
        if 'b4' in results:
            b4 = results['b4']
            if 'baseline' in b4 and 'residual' in b4:
                baseline_acc = b4['baseline']['accuracy']
                residual_acc = b4['residual']['accuracy']
                drop = (baseline_acc - residual_acc) / baseline_acc * 100

                print(f"\n[B4] Mean-Subtracted Probe:")
                print(f"  Baseline accuracy:  {baseline_acc:.4f}")
                print(f"  Residual accuracy:  {residual_acc:.4f}")
                print(f"  Relative drop:      {drop:.1f}%")

        # B-series Results
        if 'b_series' in results:
            b = results['b_series']

            if 'B1' in b:
                print(f"\n[B1] Mean Subtraction Ablation:")
                print(f"  Baseline: {b['B1']['baseline_accuracy']:.4f}")
                print(f"  Ablated:  {b['B1']['ablated_accuracy']:.4f}")
                print(f"  Drop:     {b['B1']['relative_drop']:.1f}%")

            if 'B2' in b:
                print(f"\n[B2] Cross-Position Patching:")
                print(f"  Shift rate: {b['B2']['shift_rate']:.4f}")

            if 'B3' in b:
                print(f"\n[B3] Mean-Only Prediction:")
                print(f"  Accuracy: {b['B3']['accuracy']:.4f}")

        # A-series Results
        if 'a_series' in results:
            a = results['a_series']

            if 'A1' in a:
                print(f"\n[A1] Decoding Vector Ablation:")
                print(f"  Baseline: {a['A1']['baseline_accuracy']:.4f}")
                print(f"  Ablated:  {a['A1']['ablated_accuracy']:.4f}")
                print(f"  Drop:     {a['A1']['relative_drop']:.1f}%")

            if 'A1b' in a:
                print(f"\n[A1b] Decoding vs Random Direction:")
                print(f"  Ratio: {a['A1b']['ratio']:.2f}x")

    # Final interpretation
    print("\n" + "=" * 70)
    print("MECHANISM INTERPRETATION")
    print("=" * 70)

    for setting in settings:
        if setting not in all_results:
            continue

        results = all_results[setting]
        print(f"\n{setting.upper()}:")

        # Interpret B4
        if 'b4' in results and 'baseline' in results['b4'] and 'residual' in results['b4']:
            baseline = results['b4']['baseline']['accuracy']
            residual = results['b4']['residual']['accuracy']
            drop = (baseline - residual) / baseline * 100

            if drop < 5:
                print("  Mechanism 1 (Decoding Vector): SUFFICIENT")
                print("  Mechanism 2 (Population Mean): NOT NECESSARY")
            elif drop < 20:
                print("  Both mechanisms contribute, Mechanism 1 is PRIMARY")
            else:
                print("  Mechanism 2 (Population Mean): NECESSARY")


def generate_summary_plot(all_results, settings):
    """Generate summary visualization."""

    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 5))
    if len(settings) == 1:
        axes = [axes]

    for idx, setting in enumerate(settings):
        if setting not in all_results:
            continue

        ax = axes[idx]
        results = all_results[setting]

        experiments = []
        accuracies = []
        colors = []

        # B4 baseline and residual
        if 'b4' in results:
            if 'baseline' in results['b4']:
                experiments.append('B4\nBaseline')
                accuracies.append(results['b4']['baseline']['accuracy'])
                colors.append('steelblue')
            if 'residual' in results['b4']:
                experiments.append('B4\nResidual')
                accuracies.append(results['b4']['residual']['accuracy'])
                colors.append('lightblue')

        # B1
        if 'b_series' in results and 'B1' in results['b_series']:
            experiments.append('B1\nMean Abl.')
            accuracies.append(results['b_series']['B1']['ablated_accuracy'])
            colors.append('coral')

        # A1
        if 'a_series' in results and 'A1' in results['a_series']:
            experiments.append('A1\nVec Abl.')
            accuracies.append(results['a_series']['A1']['ablated_accuracy'])
            colors.append('mediumpurple')

        if experiments:
            bars = ax.bar(experiments, accuracies, color=colors, edgecolor='black')
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'{setting.replace("_", " ").title()}', fontsize=14)
            ax.set_ylim(0, 1.1)
            ax.axhline(y=1/64, color='gray', linestyle='--', alpha=0.5, label='Random chance')

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = PLOTS_DIR / 'intervention_experiments_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSummary plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Run All Intervention Experiments')
    parser.add_argument('--setting', type=str, default='synthetic',
                        choices=['synthetic', 'natural_language'],
                        help='Data setting to run')
    parser.add_argument('--all', action='store_true',
                        help='Run on all settings')
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--d_vocab', type=int, default=5000)
    parser.add_argument('--n_ctx', type=int, default=64)
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--skip_population_means', action='store_true')
    parser.add_argument('--skip_b4', action='store_true')
    parser.add_argument('--skip_b_series', action='store_true')
    parser.add_argument('--skip_a_series', action='store_true')
    args = parser.parse_args()

    # Determine which settings to run
    if args.all:
        settings = ['synthetic', 'natural_language']
    else:
        settings = [args.setting]

    print("=" * 70)
    print("INTERVENTION EXPERIMENTS FOR POSITION DECODING MECHANISMS")
    print("=" * 70)
    print(f"\nSettings to run: {settings}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    for setting in settings:
        print(f"\n{'#' * 70}")
        print(f"# Running experiments for: {setting}")
        print(f"{'#' * 70}")

        torch.manual_seed(42)
        np.random.seed(42)

        results = {}

        # Step 1: Population means
        if not args.skip_population_means:
            pop_means, pop_analysis = run_population_means(
                setting, args.d_model, args.d_vocab, args.n_ctx
            )
            results['population_means'] = pop_analysis

        # Step 2: B4 (Priority experiment)
        if not args.skip_b4:
            b4_results = run_experiment_b4(
                setting, args.d_model, args.d_vocab, args.n_ctx,
                args.n_train, args.n_test, args.epochs
            )
            results['b4'] = {
                'baseline': b4_results.get('baseline', {}),
                'residual': b4_results.get('residual', {})
            }

        # Step 3: B-series (B1, B2, B3)
        if not args.skip_b_series:
            b_series_results = run_experiment_b_series(
                setting, args.d_model, args.d_vocab, args.n_ctx,
                args.n_train, args.n_test
            )
            results['b_series'] = b_series_results

        # Step 4: A-series (A1, A2, A3)
        if not args.skip_a_series:
            a_series_results = run_experiment_a_series(
                setting, args.d_model, args.d_vocab, args.n_ctx,
                args.n_train, args.n_test
            )
            results['a_series'] = a_series_results

        all_results[setting] = results

    # Generate summary
    generate_summary_table(all_results, settings)

    # Generate plot
    generate_summary_plot(all_results, settings)

    # Save all results
    save_path = ARTIFACTS_DIR / f"all_intervention_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'results': all_results,
        'settings': settings,
        'config': vars(args),
        'timestamp': datetime.now().isoformat()
    }, save_path)
    print(f"\nAll results saved to: {save_path}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
