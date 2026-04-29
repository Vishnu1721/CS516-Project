# src/plots.py

"""
Generate the headline figures from deployment results.

Functions:
  plot_deployment_results_before()      → 4-panel using BEFORE-intervention
                                           fairness (shows the problem)
  plot_deployment_results_after()       → 4-panel using AFTER-intervention
                                           fairness (shows the fix)
  plot_deployment_results_compare()     → fairness panel with BOTH before
                                           & after overlaid (shows what
                                           the method does)
  plot_transfer_gap()                    → twin-axis: accuracy vs fairness
                                           degradation under shift,
                                           NO intervention applied.
  make_pareto_plot_for_window(...)       → wrapper for src/pareto_plot.py
  run_deployment_no_intervention(...)    → helper to produce data for
                                           the transfer-gap plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_slice, measure_shift
from src.metrics import compute_all_metrics
from config import (DEPLOYMENT_SCHEDULE, WINDOW_SAMPLE_SIZE,
                    FEATURE_COLS, TARGET_COL,
                    DPD_TOLERANCE, EOD_TOLERANCE)


# ──────────────────────────────────────────────────────────────
# SHARED 4-PANEL HELPER
# ──────────────────────────────────────────────────────────────

def _plot_4panel(df_lr, df_gb, fairness_suffix, title_suffix, save_path):
    """
    Internal helper. Builds the 4-panel figure using the columns
    'dpd{suffix}', 'eod{suffix}', etc.

    fairness_suffix:  ''       → before-intervention metrics
                      '_after' → after-intervention metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Adaptive Fairness Monitoring — {title_suffix}',
                 fontsize=16, fontweight='bold', y=1.00)

    windows = df_lr['window'].values

    dpd_col = f'dpd{fairness_suffix}'
    eod_col = f'eod{fairness_suffix}'
    fpr_col = 'fpr_gap'   # error gaps logged before intervention only
    fnr_col = 'fnr_gap'
    acc_col = f'acc{fairness_suffix}' if fairness_suffix else 'accuracy'

    # ── Panel 1: Fairness metrics ────────────────────────────
    ax = axes[0, 0]
    ax.plot(windows, df_lr[dpd_col], 'o-', label='DPD (LogReg)',
            color='C0', linewidth=2)
    ax.plot(windows, df_lr[eod_col], 's--', label='EOD (LogReg)',
            color='C1', linewidth=2)
    if df_gb is not None:
        ax.plot(windows, df_gb[dpd_col], 'o-', label='DPD (GradientBoost)',
                color='C2', linewidth=2)
        ax.plot(windows, df_gb[eod_col], 's--', label='EOD (GradientBoost)',
                color='C3', linewidth=2)
    ax.axhline(DPD_TOLERANCE, ls=':',  color='gray',
               label=f'DPD Tol ({DPD_TOLERANCE})')
    ax.axhline(EOD_TOLERANCE, ls='--', color='gray',
               label=f'EOD Tol ({EOD_TOLERANCE})')
    ax.set_title(f'Fairness Metrics Over Windows ({title_suffix})')
    ax.set_xlabel('Window'); ax.set_ylabel('Gap')
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3)

    # ── Panel 2: Performance ─────────────────────────────────
    ax = axes[0, 1]
    ax.plot(windows, df_lr[acc_col], 'o-', label='Accuracy (LogReg)',
            color='C0', linewidth=2)
    ax.plot(windows, df_lr['auc'],   's--', label='AUC (LogReg)',
            color='C1', linewidth=2)
    if df_gb is not None:
        ax.plot(windows, df_gb[acc_col], 'o-', label='Accuracy (GradientBoost)',
                color='C2', linewidth=2)
        ax.plot(windows, df_gb['auc'],   's--', label='AUC (GradientBoost)',
                color='C3', linewidth=2)
    ax.set_title('Performance Over Windows')
    ax.set_xlabel('Window'); ax.set_ylabel('Score')
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3)

    # ── Panel 3: Error gaps (FPR / FNR — before intervention) ──
    ax = axes[1, 0]
    ax.plot(windows, df_lr[fpr_col], 'o-', label='FPR Gap (LogReg)',
            color='C0', linewidth=2)
    ax.plot(windows, df_lr[fnr_col], 's--', label='FNR Gap (LogReg)',
            color='C1', linewidth=2)
    if df_gb is not None:
        ax.plot(windows, df_gb[fpr_col], 'o-', label='FPR Gap (GradientBoost)',
                color='C2', linewidth=2)
        ax.plot(windows, df_gb[fnr_col], 's--', label='FNR Gap (GradientBoost)',
                color='C3', linewidth=2)
    ax.axhline(0.05, ls='--', color='gray')
    ax.set_title('Error Gaps Over Windows')
    ax.set_xlabel('Window'); ax.set_ylabel('Gap')
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3)

    # ── Panel 4: Adaptive thresholds (LogReg) ────────────────
    ax = axes[1, 1]
    ax.plot(windows, df_lr['threshold_g0'], 'o-',
            label='Group 0 (Male)',   color='C0', linewidth=2)
    ax.plot(windows, df_lr['threshold_g1'], 's-',
            label='Group 1 (Female)', color='C1', linewidth=2)
    ax.set_title('Adaptive Thresholds (LogReg)')
    ax.set_xlabel('Window'); ax.set_ylabel('Threshold')
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[plots] Saved {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 1a) BEFORE INTERVENTION
# ──────────────────────────────────────────────────────────────

def plot_deployment_results_before(df_lr, df_gb=None,
                                    save_path='results/plots/deployment_before.png'):
    """4-panel summary using BEFORE-intervention fairness."""
    _plot_4panel(df_lr, df_gb,
                 fairness_suffix='',
                 title_suffix='Before Intervention',
                 save_path=save_path)


# ──────────────────────────────────────────────────────────────
# 1b) AFTER INTERVENTION
# ──────────────────────────────────────────────────────────────

def plot_deployment_results_after(df_lr, df_gb=None,
                                   save_path='results/plots/deployment_after.png'):
    """4-panel summary using AFTER-intervention fairness."""
    _plot_4panel(df_lr, df_gb,
                 fairness_suffix='_after',
                 title_suffix='After Intervention',
                 save_path=save_path)


# ──────────────────────────────────────────────────────────────
# 1c) BEFORE vs AFTER OVERLAY (most informative single figure)
# ──────────────────────────────────────────────────────────────

def plot_deployment_results_compare(df_lr, df_gb=None,
                                     save_path='results/plots/deployment_compare.png'):
    """
    Single fairness panel showing before AND after on the same axes.
    Solid = before intervention, dashed = after.
    Readers see both shift-induced unfairness and how much is recovered.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('Fairness Before vs After Pareto Adaptation\n'
                 'Solid = before intervention,  Dashed = after',
                 fontsize=13, fontweight='bold')

    windows = df_lr['window'].values

    ax.plot(windows, df_lr['dpd'],       'o-',  color='C0', linewidth=2.5,
            label='DPD before (LogReg)')
    ax.plot(windows, df_lr['dpd_after'], 'o--', color='C0', linewidth=2,
            alpha=0.6, label='DPD after  (LogReg)')

    ax.plot(windows, df_lr['eod'],       's-',  color='C1', linewidth=2.5,
            label='EOD before (LogReg)')
    ax.plot(windows, df_lr['eod_after'], 's--', color='C1', linewidth=2,
            alpha=0.6, label='EOD after  (LogReg)')

    if df_gb is not None:
        ax.plot(windows, df_gb['dpd'],       'o-',  color='C2', linewidth=2.5,
                label='DPD before (GB)')
        ax.plot(windows, df_gb['dpd_after'], 'o--', color='C2', linewidth=2,
                alpha=0.6, label='DPD after  (GB)')
        ax.plot(windows, df_gb['eod'],       's-',  color='C3', linewidth=2.5,
                label='EOD before (GB)')
        ax.plot(windows, df_gb['eod_after'], 's--', color='C3', linewidth=2,
                alpha=0.6, label='EOD after  (GB)')

    ax.axhline(DPD_TOLERANCE, ls=':',  color='gray',
               label=f'DPD Tol ({DPD_TOLERANCE})')
    ax.axhline(EOD_TOLERANCE, ls='--', color='gray',
               label=f'EOD Tol ({EOD_TOLERANCE})')

    ax.set_xlabel('Window'); ax.set_ylabel('Fairness Gap')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[plots] Saved {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 2) NO-INTERVENTION DEPLOYMENT (for transfer-gap plot)
# ──────────────────────────────────────────────────────────────

def run_deployment_no_intervention(model, scaler, df_train, threshold=0.5):
    """
    Deploy with a fixed 0.5 threshold across all windows. No adaptation.
    Used to produce the transfer-gap plot.
    """
    rows = []
    for w, (state, year, label) in enumerate(DEPLOYMENT_SCHEDULE):
        df_w = load_slice(state, year, sample_size=WINDOW_SAMPLE_SIZE,
                          verbose=False)
        X_w = df_w[FEATURE_COLS]
        y_w = df_w[TARGET_COL]
        s_w = df_w['sex_binary']

        shift = measure_shift(df_train, df_w)
        scores_w = model.predict_proba(scaler.transform(X_w))[:, 1]
        y_pred = (scores_w >= threshold).astype(int)
        m = compute_all_metrics(y_w.values, y_pred, scores_w, s_w.values)

        rows.append({
            'window': w, 'state': state, 'year': year, 'label': label,
            'accuracy': m['accuracy'], 'auc': m['auc'],
            'dpd': m['dpd'], 'eod': m['eod'],
            'fpr_gap': m['fpr_gap'], 'fnr_gap': m['fnr_gap'],
            **shift,
        })
        print(f"  W{w:02d} {state} {year}: "
              f"acc={m['accuracy']:.3f} dpd={m['dpd']:.3f} eod={m['eod']:.3f}")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 3) TRANSFER-GAP PLOT
# ──────────────────────────────────────────────────────────────

def plot_transfer_gap(df_no_intervention, baseline_acc,
                       save_path='results/plots/transfer_gap.png'):
    """Twin-axis plot: accuracy vs fairness degradation, no intervention."""
    fig, ax1 = plt.subplots(figsize=(13, 6))
    fig.suptitle('Fairness Degrades Faster Than Accuracy Under Distribution Shift\n'
                 '(Logistic Regression on ACS Income, no intervention applied)',
                 fontsize=13, fontweight='bold')

    windows = df_no_intervention['window'].values

    color_acc = '#1f77b4'
    ax1.plot(windows, df_no_intervention['accuracy'],
             'o-', color=color_acc, linewidth=2.5, markersize=8,
             label='Accuracy')
    ax1.axhline(baseline_acc, ls=':', color=color_acc, alpha=0.6,
                label=f'Baseline accuracy ({baseline_acc:.3f})')
    ax1.set_xlabel('Deployment Window', fontsize=12)
    ax1.set_ylabel('Accuracy (no intervention)',
                   color=color_acc, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color_dpd = '#d62728'
    color_eod = '#ff7f0e'
    ax2.plot(windows, df_no_intervention['dpd'],
             's-', color=color_dpd, linewidth=2.5, markersize=8,
             label='DPD')
    ax2.plot(windows, df_no_intervention['eod'],
             '^-', color=color_eod, linewidth=2.5, markersize=8,
             label='EOD')
    ax2.axhline(EOD_TOLERANCE, ls='--', color='gray', alpha=0.6,
                label=f'Tolerance ({EOD_TOLERANCE})')
    ax2.set_ylabel('Fairness Gap (no intervention)',
                   color=color_dpd, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_dpd)
    ax2.set_ylim(0, max(0.30, df_no_intervention[['dpd','eod']].max().max() * 1.1))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[plots] Saved {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: PARETO FRONTIER
# ──────────────────────────────────────────────────────────────

def make_pareto_plot_for_window(model, scaler, window_idx=10,
                                  save_path='results/plots/pareto_frontier.png'):
    """Pareto-frontier scatter for a single deployment window."""
    from src.pareto_plot import plot_pareto_frontier
    state, year, label = DEPLOYMENT_SCHEDULE[window_idx]
    df_w = load_slice(state, year, sample_size=WINDOW_SAMPLE_SIZE)
    X_w = df_w[FEATURE_COLS]
    y_w = df_w[TARGET_COL]
    s_w = df_w['sex_binary']
    scores_w = model.predict_proba(scaler.transform(X_w))[:, 1]

    print(f"[plots] Building Pareto frontier for window {window_idx} "
          f"({state} {year}, {label})...")
    return plot_pareto_frontier(
        scores_w, y_w.values, s_w.values, save_path=save_path
    )