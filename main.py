# main.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.preprocess import load_and_clean
from src.model import train_baseline, train_gb
from src.deployment import run_deployment
from src.pareto_plot import plot_pareto_frontier
from src.drift import (simulate_demographic_shift,
                        simulate_feature_drift,
                        simulate_label_noise)
from config import (RESULTS_DIR, PLOTS_DIR, METRICS_LOG,
                    DEMO_SHIFT_SCHEDULE, FEATURE_DRIFT_SCHEDULE,
                    LABEL_NOISE_SCHEDULE)


def main():

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Step 1: Load and clean data ──────────────────────────
    print("\n=== STEP 1: Loading Data ===")
    df = load_and_clean()

    # ── Step 2: Train baseline models ────────────────────────
    print("\n=== STEP 2: Training Baseline Models ===")
    model_lr, scaler_lr, X_test, y_test, s_test = train_baseline(df)
    model_gb, scaler_gb, _, _, _ = train_gb(df)

    # ── Step 3-7: Run deployment simulation ──────────────────
    print("\n=== STEP 3-7: Running Deployment Simulation ===")
    results_lr = run_deployment(model_lr, scaler_lr, X_test, y_test, s_test)
    results_lr['model'] = 'LogReg'

    results_gb = run_deployment(model_gb, scaler_gb, X_test, y_test, s_test)
    results_gb['model'] = 'GradientBoost'

    results = pd.concat([results_lr, results_gb])

    # ── Step 8: Save results ─────────────────────────────────
    results.to_csv(METRICS_LOG, index=False)
    print(f"\n[main] Results saved to {METRICS_LOG}")

    # ── Step 9: Main plots ───────────────────────────────────
    print("\n=== STEP 9: Generating Main 2x2 Plot ===")
    plot_results(results)

    # ── Step 10: Pareto frontier diagnostic plot ─────────────
    print("\n=== STEP 10: Generating Pareto Frontier Plot (window 5) ===")
    w_demo = 5
    X_demo = X_test.copy()
    y_demo = y_test.copy()
    s_demo = s_test.copy()

    X_demo, y_demo, s_demo = simulate_demographic_shift(
        X_demo, y_demo, s_demo,
        DEMO_SHIFT_SCHEDULE[w_demo], random_state=w_demo)
    X_demo = simulate_feature_drift(
        X_demo, FEATURE_DRIFT_SCHEDULE[w_demo], random_state=w_demo)
    y_demo = simulate_label_noise(
        y_demo, LABEL_NOISE_SCHEDULE[w_demo], random_state=w_demo)

    scores_demo = model_lr.predict_proba(scaler_lr.transform(X_demo))[:, 1]

    plot_pareto_frontier(
        scores_demo, y_demo.values, s_demo.values,
        metric_x_idx=0, metric_y_idx=3,
        metric_x_name='DPD',
        metric_y_name='Predictive Parity Diff'
    )

    # ── Step 11: Transfer-gap analysis (the key finding) ─────
    print("\n=== STEP 11: Transfer Gap Analysis ===")
    plot_accuracy_vs_fairness_transfer(results)
    generate_transfer_table(results)

    print("\nDone!")


def plot_results(results):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adaptive Fairness Monitoring Framework — Results',
                 fontsize=14, fontweight='bold')

    # ─── Plot 1: Fairness metrics ────────────────────────────
    ax = axes[0, 0]
    for model in results['model'].unique():
        subset = results[results['model'] == model]
        ax.plot(subset['window'], subset['dpd'],
                marker='o', label=f'DPD ({model})')
        ax.plot(subset['window'], subset['eod'],
                marker='s', linestyle='--', label=f'EOD ({model})')
    ax.axhline(0.03, color='gray', linestyle=':', label='DPD Tol (0.03)')
    ax.axhline(0.05, color='gray', linestyle='--', label='EOD Tol (0.05)')
    ax.set_title('Fairness Metrics Over Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Gap')
    ax.legend(fontsize=8)

    # ─── Plot 2: Performance ─────────────────────────────────
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
    ax.legend(fontsize=8)

    # ─── Plot 3: Error gaps ──────────────────────────────────
    ax = axes[1, 0]
    for model in results['model'].unique():
        subset = results[results['model'] == model]
        ax.plot(subset['window'], subset['fpr_gap'],
                marker='o', label=f'FPR Gap ({model})')
        ax.plot(subset['window'], subset['fnr_gap'],
                marker='s', linestyle='--', label=f'FNR Gap ({model})')
    ax.axhline(0.05, color='gray', linestyle='--')
    ax.set_title('Error Gaps Over Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Gap')
    ax.legend(fontsize=8)

    # ─── Plot 4: Thresholds (LR only) ────────────────────────
    ax = axes[1, 1]
    subset = results[results['model'] == 'LogReg']
    ax.plot(subset['window'], subset['threshold_g0'],
            marker='o', label='Group 0 (Male)')
    ax.plot(subset['window'], subset['threshold_g1'],
            marker='s', label='Group 1 (Female)')
    ax.set_title('Adaptive Thresholds (LogReg)')
    ax.set_xlabel('Window')
    ax.set_ylabel('Threshold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'deployment_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"[plot] Saved to {save_path}")
    plt.show()

def plot_accuracy_vs_fairness_transfer(results,
                                         save_path=None):
    """
    Demonstrates the core finding: under deployment drift, accuracy
    transfers relatively well while fairness metrics degrade and
    fluctuate. Uses the STATIC strategy (no intervention) to show
    what would happen in a deployed system without monitoring.

    Two y-axes:
      - Left (blue):  Accuracy — relatively stable
      - Right (red):  Fairness gaps — volatile, frequently violating
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'transfer_gap.png')

    lr = results[results['model'] == 'LogReg'].sort_values('window')
    baseline_acc = lr['acc_static'].iloc[0]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # ── Left axis: Accuracy ───────────────────────────────────
    color_acc = 'steelblue'
    ax1.set_xlabel('Deployment Window', fontsize=12)
    ax1.set_ylabel('Accuracy (no intervention)',
                    color=color_acc, fontsize=12, fontweight='bold')
    ax1.plot(lr['window'], lr['acc_static'],
              marker='o', linewidth=2.8, markersize=9,
              color=color_acc, label='Accuracy', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(baseline_acc, color=color_acc, linestyle=':',
                 alpha=0.5, linewidth=1.5,
                 label=f'Baseline accuracy ({baseline_acc:.3f})')
    ax1.grid(True, alpha=0.25)

    # ── Right axis: Fairness gaps ─────────────────────────────
    ax2 = ax1.twinx()
    color_fair = 'crimson'
    ax2.set_ylabel('Fairness Gap (no intervention)',
                    color=color_fair, fontsize=12, fontweight='bold')
    ax2.plot(lr['window'], lr['dpd_static'],
              marker='s', linewidth=2, markersize=8,
              color='crimson', label='DPD', zorder=2)
    ax2.plot(lr['window'], lr['eod_static'],
              marker='^', linewidth=2, markersize=8,
              color='darkorange', label='EOD', zorder=2)
    ax2.tick_params(axis='y', labelcolor=color_fair)
    ax2.set_ylim(0, max(0.30, lr['eod_static'].max() * 1.2))
    ax2.axhline(0.05, color='gray', linestyle='--',
                 alpha=0.6, linewidth=1.2,
                 label='Tolerance (0.05)')

    # ── Title and legend ──────────────────────────────────────
    plt.title('Fairness Degrades Faster Than Accuracy Under Distribution Shift\n'
              '(Logistic Regression on ACS Income, no intervention applied)',
              fontsize=12, fontweight='bold', pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
                loc='upper left', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[plot] Saved transfer-gap plot to {save_path}")
    plt.show()

def generate_transfer_table(results,
                              save_path=None):
    """
    Generates a quantitative table showing how each metric changed
    from window 0 (baseline) to the worst window observed.
    Saves CSV and prints a markdown-formatted version for the writeup.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'transfer_table.csv')

    lr = results[results['model'] == 'LogReg'].sort_values('window').reset_index(drop=True)

    # ── Define metrics to track ─────────────────────────────────
    # For performance metrics: lower is worse → use minimum
    # For fairness gaps: higher is worse → use maximum
    metric_specs = [
        ('Accuracy',          'acc_static',  'lower_is_worse', '{:.3f}'),
        ('AUC',                'auc_static',  'lower_is_worse', '{:.3f}'),
        ('DPD',                'dpd_static',  'higher_is_worse', '{:.3f}'),
        ('EOD',                'eod_static',  'higher_is_worse', '{:.3f}'),
        ('Predictive Parity Diff', 'pp_diff', 'higher_is_worse', '{:.3f}'),
        ('FPR Gap',            'fpr_gap',     'higher_is_worse', '{:.3f}'),
        ('FNR Gap',            'fnr_gap',     'higher_is_worse', '{:.3f}'),
        ('Accuracy Gap',       'acc_gap',     'higher_is_worse', '{:.3f}'),
    ]

    rows = []
    for label, col, direction, fmt in metric_specs:
        if col not in lr.columns:
            continue

        baseline = lr[col].iloc[0]
        if direction == 'lower_is_worse':
            worst_value = lr[col].min()
            worst_window = int(lr[col].idxmin())
        else:
            worst_value = lr[col].max()
            worst_window = int(lr[col].idxmax())

        # Relative change (signed; negative = degraded for performance metrics,
        # positive = degraded for fairness gaps)
        if baseline != 0:
            rel_change = (worst_value - baseline) / baseline * 100
        else:
            rel_change = float('inf') if worst_value > 0 else 0

        # Severity label
        if direction == 'lower_is_worse':
            degraded = worst_value < baseline
        else:
            degraded = worst_value > baseline

        rows.append({
            'Metric':           label,
            'Baseline (W0)':    fmt.format(baseline),
            'Worst Value':      fmt.format(worst_value),
            'Worst Window':     f'W{worst_window}',
            'Relative Change':  f'{rel_change:+.1f}%',
            'Degraded?':        '✗' if degraded else '✓ stable',
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(save_path, index=False)
    print(f"[table] Saved transfer table to {save_path}\n")

    # ── Print to terminal in readable format ────────────────────
    print("=" * 90)
    print("TRANSFER GAP TABLE — Logistic Regression (no intervention)")
    print("=" * 90)
    print(table_df.to_string(index=False))
    print("=" * 90)

    # ── Also print as markdown for direct copy into writeup ─────
    print("\nMARKDOWN VERSION (copy into your report):\n")
    print(table_df.to_markdown(index=False))
    print()

    return table_df


if __name__ == '__main__':
    main()