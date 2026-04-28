# src/intervention.py

import numpy as np
from src.metrics import (demographic_parity_difference,
                          equalized_odds_difference,
                          predictive_parity_difference,
                          accuracy_gap,
                          auc_gap)


def apply_group_thresholds(scores, sensitive, thresholds):
    """
    Applies a different decision threshold per group.
    Model scores are NOT changed — only the cutoff changes.
    """
    y_pred = np.zeros(len(scores), dtype=int)
    for g in np.unique(sensitive):
        mask = (sensitive == g)
        y_pred[mask] = (scores[mask] >= thresholds[g]).astype(int)
    return y_pred


# ─── BASELINE: HILL CLIMBING (DPD-only) ────────────────────────

def adapt_thresholds(scores, y_true, sensitive,
                     current_thresholds, step=0.02, max_iter=50):
    """
    BASELINE METHOD: Hill-climbing search to minimize DPD only.

    LIMITATION: Optimizes a single metric (DPD). When other fairness
    metrics matter (EOD, predictive parity, etc.), this method may
    reduce DPD while worsening other metrics. We retain it as a
    baseline and compare against the multi-metric Pareto approach.
    """
    best_thresholds = current_thresholds.copy()
    y_pred_current = apply_group_thresholds(scores, sensitive, best_thresholds)
    best_dpd = demographic_parity_difference(y_pred_current, sensitive)

    for iteration in range(max_iter):
        improved = False
        for g in np.unique(sensitive):
            for delta in [-step, +step]:
                trial = best_thresholds.copy()
                trial[g] = np.clip(trial[g] + delta, 0.2, 0.8)

                y_pred_trial = apply_group_thresholds(scores, sensitive, trial)
                dpd_trial = demographic_parity_difference(y_pred_trial, sensitive)

                if dpd_trial < best_dpd:
                    best_dpd = dpd_trial
                    best_thresholds = trial.copy()
                    improved = True

        if not improved:
            print(f"[hill_climb] Converged after {iteration+1} iterations. "
                  f"DPD: {best_dpd:.4f}")
            break

    return best_thresholds


# ─── IMPROVED METHOD: PARETO FRONTIER + TIEBREAKER ─────────────

def compute_metric_vector(scores, y_true, sensitive, thresholds):
    """
    For a candidate threshold pair, compute the vector of fairness
    metrics we want to jointly minimize.
    """
    y_pred = apply_group_thresholds(scores, sensitive, thresholds)
    fpr_gap, fnr_gap, _ = equalized_odds_difference(y_true, y_pred, sensitive)

    return np.array([
        demographic_parity_difference(y_pred, sensitive),
        fpr_gap,
        fnr_gap,
        predictive_parity_difference(y_true, y_pred, sensitive),
        accuracy_gap(y_true, y_pred, sensitive),
        auc_gap(y_true, scores, sensitive),
    ])


def is_dominated(point, others):
    """
    Returns True if `point` is dominated by any point in `others`.
    A point is dominated if some other point is <= on every metric
    AND strictly < on at least one metric.
    """
    for other in others:
        if np.all(other <= point) and np.any(other < point):
            return True
    return False


def adapt_thresholds_pareto(scores, y_true, sensitive,
                              grid_step=0.02,
                              weights=None):
    """
    IMPROVED METHOD: Pareto frontier search over threshold pairs.

    HOW IT WORKS:
    1. Evaluate every (t0, t1) pair on a fine grid in [0.2, 0.8].
    2. Compute a vector of 6 fairness metrics for each pair.
    3. Identify Pareto-optimal pairs — those NOT dominated by any
       other pair (no other pair is better on every metric AND strictly
       better on at least one).
    4. Among Pareto-optimal pairs, select the one minimizing weighted
       sum of metrics as a final tiebreaker.

    WHY THIS IS BETTER THAN HILL CLIMBING:
    - Considers all 6 metrics simultaneously, not just DPD
    - Globally optimal within grid resolution (no local minima)
    - Pareto filtering ensures we never sacrifice one metric needlessly

    Args:
        weights: dict mapping metric index to weight for tiebreaker.
                 If None, equal weights are used.
    """
    # ── Step 1: Evaluate all candidates on the grid ──────────
    candidates_t = np.arange(0.2, 0.8 + grid_step, grid_step)
    all_thresholds = []
    all_vectors = []

    for t0 in candidates_t:
        for t1 in candidates_t:
            thresholds = {0: float(t0), 1: float(t1)}
            vec = compute_metric_vector(scores, y_true, sensitive, thresholds)
            all_thresholds.append(thresholds)
            all_vectors.append(vec)

    all_vectors = np.array(all_vectors)
    n = len(all_vectors)

    # ── Step 2: Find Pareto-optimal candidates ──────────────
    # Vectorized domination check: point i is dominated if any j
    # satisfies (all metrics <=) and (at least one <)
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        diff = all_vectors - all_vectors[i]   # (n, 6)
        dominated_by = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
        if dominated_by.any():
            pareto_mask[i] = False

    pareto_indices = np.where(pareto_mask)[0]
    pareto_vectors = all_vectors[pareto_indices]

    # ── Step 3: Tiebreaker — weighted sum among Pareto set ──
    if weights is None:
        weights = np.ones(pareto_vectors.shape[1])
    else:
        weights = np.array(weights)

    weighted_scores = pareto_vectors @ weights
    best_idx_in_pareto = np.argmin(weighted_scores)
    best_global_idx = pareto_indices[best_idx_in_pareto]

    best_thresholds = all_thresholds[best_global_idx]
    best_vector = all_vectors[best_global_idx]

    print(f"[pareto] Evaluated {n} candidates, "
          f"{len(pareto_indices)} on Pareto frontier. "
          f"Selected: t0={best_thresholds[0]:.2f}, t1={best_thresholds[1]:.2f}")
    print(f"[pareto] Metrics: DPD={best_vector[0]:.3f}  "
          f"FPRgap={best_vector[1]:.3f}  FNRgap={best_vector[2]:.3f}  "
          f"PP={best_vector[3]:.3f}  AccGap={best_vector[4]:.3f}  "
          f"AUCgap={best_vector[5]:.3f}")

    return best_thresholds