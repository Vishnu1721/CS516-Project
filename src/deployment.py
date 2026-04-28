# src/deployment.py

import numpy as np
import pandas as pd
from src.metrics import compute_all_metrics
from src.drift import (simulate_demographic_shift,
                        simulate_feature_drift,
                        simulate_label_noise)
from src.intervention import (apply_group_thresholds,
                               adapt_thresholds,           # baseline (DPD-only)
                               adapt_thresholds_pareto)    # improved (multi-metric)
from config import (N_WINDOWS, DEFAULT_THRESHOLD,
                    DPD_TOLERANCE, EOD_TOLERANCE,
                    PP_TOLERANCE, ACC_GAP_TOLERANCE,
                    DEMO_SHIFT_SCHEDULE, FEATURE_DRIFT_SCHEDULE,
                    LABEL_NOISE_SCHEDULE)


def apply_global_threshold(scores, threshold=0.5):
    return (scores >= threshold).astype(int)


def run_deployment(model, scaler, X_test, y_test, s_test):
    """
    Main deployment simulation loop.

    For each window:
      1. Apply drift to get this window's data
      2. Run frozen model on drifted data
      3. Apply three strategies (static, global, group-specific)
      4. Compute fairness metrics for all three
      5. If violation detected, adapt thresholds via Pareto frontier
      6. Log everything

    Returns a DataFrame with one row per window.
    """
    results = []

    # Start with equal thresholds for both groups
    thresholds = {0: DEFAULT_THRESHOLD, 1: DEFAULT_THRESHOLD}

    for w in range(N_WINDOWS):
        print(f"\n--- Window {w:02d} ---")

        # ── Step 1: Apply drift ──────────────────────────────
        X_w = X_test.copy()
        y_w = y_test.copy()
        s_w = s_test.copy()

        X_w, y_w, s_w = simulate_demographic_shift(
            X_w, y_w, s_w,
            target_ratio=DEMO_SHIFT_SCHEDULE[w],
            random_state=w
        )

        X_w = simulate_feature_drift(
            X_w,
            drift_magnitude=FEATURE_DRIFT_SCHEDULE[w],
            random_state=w
        )

        y_w = simulate_label_noise(
            y_w,
            noise_rate=LABEL_NOISE_SCHEDULE[w],
            random_state=w
        )

        # ── Step 2: Score with frozen model ─────────────────
        X_w_sc = scaler.transform(X_w)
        scores_w = model.predict_proba(X_w_sc)[:, 1]

        # ── Step 3: Apply three strategies ──────────────────
        y_pred_static = apply_global_threshold(scores_w, 0.5)
        y_pred_global = apply_global_threshold(scores_w, thresholds[0])
        y_pred_group  = apply_group_thresholds(scores_w, s_w.values, thresholds)

        # ── Step 4: Compute metrics for each strategy ───────
        metrics_static = compute_all_metrics(
            y_w.values, y_pred_static, scores_w, s_w.values)
        metrics_global = compute_all_metrics(
            y_w.values, y_pred_global, scores_w, s_w.values)
        metrics_group = compute_all_metrics(
            y_w.values, y_pred_group, scores_w, s_w.values)

        # ── Build the results dict ──────────────────────────
        metrics = {
            'window': w,

            # Static baseline
            'dpd_static': metrics_static['dpd'],
            'eod_static': metrics_static['eod'],
            'acc_static': metrics_static['accuracy'],
            'auc_static': metrics_static['auc'],

            # Global threshold
            'dpd_global': metrics_global['dpd'],
            'eod_global': metrics_global['eod'],
            'acc_global': metrics_global['accuracy'],
            'auc_global': metrics_global['auc'],

            # Group adaptive (main method) — all 6 fairness metrics
            'dpd':       metrics_group['dpd'],
            'eod':       metrics_group['eod'],
            'fpr_gap':   metrics_group['fpr_gap'],
            'fnr_gap':   metrics_group['fnr_gap'],
            'pp_diff':   metrics_group['pp_diff'],     # NEW
            'acc_gap':   metrics_group['acc_gap'],     # NEW
            'auc_gap':   metrics_group['auc_gap'],     # NEW
            'accuracy':  metrics_group['accuracy'],
            'auc':       metrics_group['auc'],

            'threshold_g0': thresholds[0],
            'threshold_g1': thresholds[1],

            'demo_shift':    DEMO_SHIFT_SCHEDULE[w],
            'feature_drift': FEATURE_DRIFT_SCHEDULE[w],
            'label_noise':   LABEL_NOISE_SCHEDULE[w],

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

        if metrics['demo_shift'] > 0.6:
            metrics['likely_cause'] = 'demographic_shift'
        elif metrics['feature_drift'] > 0.15:
            metrics['likely_cause'] = 'feature_drift'
        elif metrics['label_noise'] > 0.1:
            metrics['likely_cause'] = 'label_noise'
        else:
            metrics['likely_cause'] = 'baseline'

        print(f"  Before intervention: "
              f"DPD={metrics['dpd']:.3f}  EOD={metrics['eod']:.3f}  "
              f"PP={metrics['pp_diff']:.3f}  AccGap={metrics['acc_gap']:.3f}  "
              f"ACC={metrics['accuracy']:.3f}")

        # ── Step 5: Intervene if any metric violates ────────
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