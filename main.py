# main.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
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
    model, scaler, X_test, y_test, s_test = train_baseline(df)

    # ── Step 3-7: Run deployment simulation ──────────────────
    print("\n=== STEP 3-7: Running Deployment Simulation ===")
    results = run_deployment(model, scaler, X_test, y_test, s_test)

    # ── Step 8: Save results ──────────────────────────────────
    results.to_csv(METRICS_LOG, index=False)
    print(f"\n[main] Results saved to {METRICS_LOG}")

    # ── Step 9: Plot ──────────────────────────────────────────
    print("\n=== STEP 8: Generating Plots ===")
    plot_results(results)
    print("Done!")


def plot_results(results):

    interventions = results[results['intervention_triggered']]['window']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adaptive Fairness Monitoring Framework — COMPAS Results',
                 fontsize=14, fontweight='bold')

    # Plot 1: Fairness metrics over windows
    ax = axes[0, 0]
    ax.plot(results['window'], results['dpd'],
            marker='o', label='Demographic Parity Diff', color='blue')
    ax.plot(results['window'], results['eod'],
            marker='s', label='Equalized Odds Diff', color='red')
    ax.axhline(0.10, color='gray', linestyle='--', label='Tolerance (0.10)')
    for w in interventions:
        ax.axvline(w, color='orange', alpha=0.3)
    ax.set_title('Fairness Metrics Over Deployment Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Fairness Gap')
    ax.legend()

    # Plot 2: Accuracy over windows
    ax = axes[0, 1]
    ax.plot(results['window'], results['accuracy'],
            marker='o', color='green', label='Accuracy')
    ax.plot(results['window'], results['auc'],
            marker='s', color='purple', label='AUC')
    for w in interventions:
        ax.axvline(w, color='orange', alpha=0.3,
                   label='Intervention' if w == interventions.iloc[0] else '')
    ax.set_title('Predictive Performance Over Deployment Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Score')
    ax.legend()

    # Plot 3: FPR and FNR gaps
    ax = axes[1, 0]
    ax.plot(results['window'], results['fpr_gap'],
            marker='o', label='FPR Gap', color='darkorange')
    ax.plot(results['window'], results['fnr_gap'],
            marker='s', label='FNR Gap', color='darkblue')
    ax.axhline(0.10, color='gray', linestyle='--', label='Tolerance')
    ax.set_title('FPR and FNR Gaps Over Deployment Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Gap')
    ax.legend()

    # Plot 4: Group-specific thresholds
    ax = axes[1, 1]
    ax.plot(results['window'], results['threshold_g0'],
            marker='o', label='Threshold — Caucasian', color='steelblue')
    ax.plot(results['window'], results['threshold_g1'],
            marker='s', label='Threshold — African-American', color='tomato')
    ax.set_title('Adaptive Thresholds Over Deployment Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Decision Threshold')
    ax.legend()

    plt.tight_layout()
    save_path = PLOTS_DIR + 'deployment_results.png'
    plt.savefig(save_path, dpi=150)
    print(f"[plot] Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    main()