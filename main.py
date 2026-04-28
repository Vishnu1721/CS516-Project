# main.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.model import train_baseline, train_gb
from src.preprocess import load_and_clean
from src.model import train_baseline
from src.deployment import run_deployment
from config import RESULTS_DIR, PLOTS_DIR, METRICS_LOG

def main():

    # Create output folders if they don't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Step 1: Load and clean data ──────────────────────────
    print("\n=== STEP 1: Loading Data ===")
    df = load_and_clean()

    # ── Step 2: Train baseline model ─────────────────────────
    print("\n=== STEP 2: Training Baseline Model ===")
    model_lr, scaler_lr, X_test, y_test, s_test = train_baseline(df)
    model_gb, scaler_gb, _, _, _ = train_gb(df)

    # ── Step 3-7: Run deployment simulation ──────────────────
    print("\n=== STEP 3-7: Running Deployment Simulation ===")
    results_lr = run_deployment(model_lr, scaler_lr, X_test, y_test, s_test)
    results_lr['model'] = 'LogReg'

    results_gb = run_deployment(model_gb, scaler_gb, X_test, y_test, s_test)
    results_gb['model'] = 'GradientBoost'

    results = pd.concat([results_lr, results_gb])
    print(results.columns) 

    # ── Step 8: Save results ──────────────────────────────────
    results.to_csv(METRICS_LOG, index=False)
    print(f"\n[main] Results saved to {METRICS_LOG}")

    # ── Step 9: Plot ──────────────────────────────────────────
    print("\n=== STEP 8: Generating Plots ===")
    plot_results(results)
    print("Done!")


def plot_results(results):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adaptive Fairness Monitoring Framework — Results',
                 fontsize=14, fontweight='bold')

    # ─────────────────────────────
    # Plot 1: Fairness metrics
    # ─────────────────────────────
    ax = axes[0, 0]

    for model in results['model'].unique():
        subset = results[results['model'] == model]

        ax.plot(subset['window'], subset['dpd'],
                marker='o', label=f'DPD ({model})')

        ax.plot(subset['window'], subset['eod'],
                marker='s', linestyle='--', label=f'EOD ({model})')

    ax.axhline(0.10, color='gray', linestyle='--', label='Tolerance')
    ax.set_title('Fairness Metrics Over Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Gap')
    ax.legend()

    # ─────────────────────────────
    # Plot 2: Accuracy
    # ─────────────────────────────
    ax = axes[0, 1]

    for model in results['model'].unique():
        subset = results[results['model'] == model]

        ax.plot(subset['window'], subset['accuracy'],
                marker='o', label=f'Accuracy ({model})')

        ax.plot(subset['window'], subset['auc'],
                marker='s', linestyle='--', label=f'AUC ({model})')

    ax.set_title('Performance Over Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Score')
    ax.legend()

    # ─────────────────────────────
    # Plot 3: Error gaps
    # ─────────────────────────────
    ax = axes[1, 0]

    for model in results['model'].unique():
        subset = results[results['model'] == model]

        ax.plot(subset['window'], subset['fpr_gap'],
                marker='o', label=f'FPR ({model})')

        ax.plot(subset['window'], subset['fnr_gap'],
                marker='s', linestyle='--', label=f'FNR ({model})')

    ax.axhline(0.10, color='gray', linestyle='--')
    ax.set_title('Error Gaps')
    ax.set_xlabel('Window')
    ax.set_ylabel('Gap')
    ax.legend()

    # ─────────────────────────────
    # Plot 4: Thresholds (only for adaptive)
    # ─────────────────────────────
    ax = axes[1, 1]

    subset = results[results['model'] == 'LogReg']

    ax.plot(subset['window'], subset['threshold_g0'],
            marker='o', label='Group 0')

    ax.plot(subset['window'], subset['threshold_g1'],
            marker='s', label='Group 1')

    ax.set_title('Adaptive Thresholds (LogReg)')
    ax.set_xlabel('Window')
    ax.set_ylabel('Threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/plots/deployment_results.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    main()