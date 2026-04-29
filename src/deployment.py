# src/deployment.py

import numpy as np
import pandas as pd

from src.metrics import compute_all_metrics
from src.intervention import (apply_group_thresholds,
                               adapt_thresholds,           # baseline (DPD-only)
                               adapt_thresholds_pareto)    # improved (multi-metric)
from src.data_loader import load_slice, measure_shift
from config import (DEPLOYMENT_SCHEDULE, WINDOW_SAMPLE_SIZE,
                    FEATURE_COLS, TARGET_COL,
                    DEFAULT_THRESHOLD,
                    DPD_TOLERANCE, EOD_TOLERANCE,
                    PP_TOLERANCE, ACC_GAP_TOLERANCE)


def apply_global_threshold(scores, threshold=0.5):
    return (scores >= threshold).astype(int)


def run_deployment(model, scaler, df_train):
    """
    Main deployment simulation loop using REAL distribution shift.

    Train once on (TRAIN_STATE, TRAIN_YEAR). Then for each window
    in DEPLOYMENT_SCHEDULE:
      1. Load that real (state, year) slice
      2. Measure shift magnitude vs training distribution
      3. Run frozen model on the new slice
      4. Apply three strategies (static, global, group-specific)
      5. Compute fairness metrics for all three
      6. If violation detected, adapt thresholds via Pareto frontier
      7. Log everything

    Args:
        model:    frozen classifier trained on df_train
        scaler:   StandardScaler fit on df_train
        df_train: training dataframe — used only to measure shift magnitude

    Returns:
        DataFrame with one row per deployment window.
    """
    results = []

    # Start with equal thresholds for both groups
    thresholds = {0: DEFAULT_THRESHOLD, 1: DEFAULT_THRESHOLD}

    for w, (state, year, label) in enumerate(DEPLOYMENT_SCHEDULE):
        print(f"\n--- Window {w:02d}: {state} {year} ({label}) ---")

        # ── Step 1: Load real slice ─────────────────────────
        df_w = load_slice(state, year, sample_size=WINDOW_SAMPLE_SIZE)
        X_w = df_w[FEATURE_COLS]
        y_w = df_w[TARGET_COL]
        s_w = df_w['sex_binary']

        # ── Step 2: Measure shift vs training ───────────────
        shift = measure_shift(df_train, df_w)

        # ── Step 3: Score with frozen model ─────────────────
        X_w_sc = scaler.transform(X_w)
        scores_w = model.predict_proba(X_w_sc)[:, 1]

        # ── Step 4: Apply three strategies ──────────────────
        y_pred_static = apply_global_threshold(scores_w, 0.5)
        y_pred_global = apply_global_threshold(scores_w, thresholds[0])
        y_pred_group  = apply_group_thresholds(scores_w, s_w.values, thresholds)

        # ── Step 5: Compute metrics for each strategy ───────
        metrics_static = compute_all_metrics(
            y_w.values, y_pred_static, scores_w, s_w.values)
        metrics_global = compute_all_metrics(
            y_w.values, y_pred_global, scores_w, s_w.values)
        metrics_group = compute_all_metrics(
            y_w.values, y_pred_group, scores_w, s_w.values)

        # ── Build the results dict ──────────────────────────
        metrics = {
            'window': w,
            'state':  state,
            'year':   year,
            'label':  label,

            # Static baseline (threshold = 0.5 always)
            'dpd_static': metrics_static['dpd'],
            'eod_static': metrics_static['eod'],
            'acc_static': metrics_static['accuracy'],
            'auc_static': metrics_static['auc'],

            # Global threshold (same for both groups, but adapts)
            'dpd_global': metrics_global['dpd'],
            'eod_global': metrics_global['eod'],
            'acc_global': metrics_global['accuracy'],
            'auc_global': metrics_global['auc'],

            # Group adaptive (main method) — all 6 fairness metrics
            'dpd':       metrics_group['dpd'],
            'eod':       metrics_group['eod'],
            'fpr_gap':   metrics_group['fpr_gap'],
            'fnr_gap':   metrics_group['fnr_gap'],
            'pp_diff':   metrics_group['pp_diff'],
            'acc_gap':   metrics_group['acc_gap'],
            'auc_gap':   metrics_group['auc_gap'],
            'accuracy':  metrics_group['accuracy'],
            'auc':       metrics_group['auc'],

            'threshold_g0': thresholds[0],
            'threshold_g1': thresholds[1],

            # Real shift magnitudes (replace synthetic schedules)
            'pos_rate_shift':    shift['pos_rate_shift'],
            'female_rate_shift': shift['female_rate_shift'],
            'agep_mean_shift':   shift['agep_mean_shift'],
            'wkhp_mean_shift':   shift['wkhp_mean_shift'],

            'intervention_triggered': False
        }

        # ── Diagnostics: which metric breaks, likely cause ──
        metric_values = {
            'dpd':     metrics['dpd'],
            'fpr_gap': metrics['fpr_gap'],
            'fnr_gap': metrics['fnr_gap'],
            'pp_diff': metrics['pp_diff'],
            'acc_gap': metrics['acc_gap'],
        }
        metrics['worst_metric'] = max(metric_values, key=metric_values.get)

        # Classify the dominant shift type from the measured magnitudes.
        # Thresholds here are heuristic — tune them based on what you
        # actually observe in your shift columns.
        if shift['female_rate_shift'] > 0.05:
            metrics['likely_cause'] = 'demographic_shift'
        elif shift['wkhp_mean_shift'] > 0.30 or shift['agep_mean_shift'] > 0.30:
            metrics['likely_cause'] = 'feature_drift'
        elif shift['pos_rate_shift'] > 0.05:
            metrics['likely_cause'] = 'label_distribution_shift'
        else:
            metrics['likely_cause'] = 'baseline'

        print(f"  Shift: pos={shift['pos_rate_shift']:.3f} "
              f"female={shift['female_rate_shift']:.3f} "
              f"agep={shift['agep_mean_shift']:.2f} "
              f"wkhp={shift['wkhp_mean_shift']:.2f}")
        print(f"  Before intervention: "
              f"DPD={metrics['dpd']:.3f}  EOD={metrics['eod']:.3f}  "
              f"PP={metrics['pp_diff']:.3f}  AccGap={metrics['acc_gap']:.3f}  "
              f"ACC={metrics['accuracy']:.3f}")

        # ── Step 6: Intervene if any metric violates ────────
        violation = (metrics['dpd']     > DPD_TOLERANCE or
                     metrics['eod']     > EOD_TOLERANCE or
                     metrics['pp_diff'] > PP_TOLERANCE or
                     metrics['acc_gap'] > ACC_GAP_TOLERANCE)

        if violation:
            print(f"  ⚠ Violation detected! Running Pareto adaptation...")
            metrics['intervention_triggered'] = True

            thresholds = adapt_thresholds_pareto(
                scores_w, y_w.values, s_w.values
            )

            y_pred_adapted = apply_group_thresholds(
                scores_w, s_w.values, thresholds
            )
            metrics_after = compute_all_metrics(
                y_w.values, y_pred_adapted, scores_w, s_w.values
            )

            metrics['dpd_after']     = metrics_after['dpd']
            metrics['eod_after']     = metrics_after['eod']
            metrics['pp_diff_after'] = metrics_after['pp_diff']
            metrics['acc_gap_after'] = metrics_after['acc_gap']
            metrics['acc_after']     = metrics_after['accuracy']

            # Update logged thresholds to the adapted ones
            metrics['threshold_g0'] = thresholds[0]
            metrics['threshold_g1'] = thresholds[1]

            print(f"  After  intervention: "
                  f"DPD={metrics_after['dpd']:.3f}  "
                  f"EOD={metrics_after['eod']:.3f}  "
                  f"PP={metrics_after['pp_diff']:.3f}  "
                  f"AccGap={metrics_after['acc_gap']:.3f}  "
                  f"ACC={metrics_after['accuracy']:.3f}")
        else:
            metrics['dpd_after']     = metrics['dpd']
            metrics['eod_after']     = metrics['eod']
            metrics['pp_diff_after'] = metrics['pp_diff']
            metrics['acc_gap_after'] = metrics['acc_gap']
            metrics['acc_after']     = metrics['accuracy']

        results.append(metrics)

    return pd.DataFrame(results)


def run_deployment_baseline_hillclimb(model, scaler, df_train):
    """
    Same loop, but uses the DPD-only hill-climbing intervention.
    Run this in addition to run_deployment() to compare methods.
    """
    results = []
    thresholds = {0: DEFAULT_THRESHOLD, 1: DEFAULT_THRESHOLD}

    for w, (state, year, label) in enumerate(DEPLOYMENT_SCHEDULE):
        print(f"\n[hillclimb] Window {w:02d}: {state} {year} ({label})")

        df_w = load_slice(state, year, sample_size=WINDOW_SAMPLE_SIZE,
                          verbose=False)
        X_w = df_w[FEATURE_COLS]
        y_w = df_w[TARGET_COL]
        s_w = df_w['sex_binary']

        shift = measure_shift(df_train, df_w)

        X_w_sc = scaler.transform(X_w)
        scores_w = model.predict_proba(X_w_sc)[:, 1]

        y_pred_group = apply_group_thresholds(scores_w, s_w.values, thresholds)
        m = compute_all_metrics(y_w.values, y_pred_group, scores_w, s_w.values)

        violation = (m['dpd']     > DPD_TOLERANCE or
                     m['eod']     > EOD_TOLERANCE or
                     m['pp_diff'] > PP_TOLERANCE or
                     m['acc_gap'] > ACC_GAP_TOLERANCE)

        if violation:
            thresholds = adapt_thresholds(
                scores_w, y_w.values, s_w.values, thresholds
            )
            y_pred_adapted = apply_group_thresholds(
                scores_w, s_w.values, thresholds
            )
            m_after = compute_all_metrics(
                y_w.values, y_pred_adapted, scores_w, s_w.values
            )
        else:
            m_after = m

        results.append({
            'window': w, 'state': state, 'year': year, 'label': label,
            'method': 'hill_climb_dpd',
            'dpd_before':     m['dpd'],
            'eod_before':     m['eod'],
            'pp_diff_before': m['pp_diff'],
            'acc_gap_before': m['acc_gap'],
            'acc_before':     m['accuracy'],
            'dpd_after':      m_after['dpd'],
            'eod_after':      m_after['eod'],
            'pp_diff_after':  m_after['pp_diff'],
            'acc_gap_after':  m_after['acc_gap'],
            'acc_after':      m_after['accuracy'],
            'threshold_g0':   thresholds[0],
            'threshold_g1':   thresholds[1],
            'intervention_triggered': violation,
            **shift,
        })

    return pd.DataFrame(results)