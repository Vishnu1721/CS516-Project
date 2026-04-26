# src/intervention.py

import numpy as np
from src.metrics import demographic_parity_difference

def apply_group_thresholds(scores, sensitive, thresholds):
    """
    Applies a different decision threshold per group.
    The model scores are NOT changed — only the cutoff changes.

    This is the core idea: same frozen model, different cutoffs.
    """
    y_pred = np.zeros(len(scores), dtype=int)

    for g in np.unique(sensitive):
        mask = (sensitive == g)
        y_pred[mask] = (scores[mask] >= thresholds[g]).astype(int)

    return y_pred


def adapt_thresholds(scores, y_true, sensitive,
                     current_thresholds, step=0.02, max_iter=50):
    """
    Hill-climbing search over threshold pairs to minimize
    demographic parity difference.

    HOW IT WORKS:
    1. Try nudging group 0 threshold up by step
    2. Try nudging group 0 threshold down by step
    3. Try nudging group 1 threshold up by step
    4. Try nudging group 1 threshold down by step
    5. Keep whichever change reduced DPD the most
    6. Repeat until no improvement found or max_iter reached

    IMPORTANT: The model is never touched.
    Only the numbers in the thresholds dict change.
    """
    best_thresholds = current_thresholds.copy()

    # Compute current DPD with current thresholds
    y_pred_current = apply_group_thresholds(
        scores, sensitive, best_thresholds
    )
    best_dpd = demographic_parity_difference(y_pred_current, sensitive)

    for iteration in range(max_iter):
        improved = False

        for g in np.unique(sensitive):
            for delta in [-step, +step]:

                # Try this change
                trial = best_thresholds.copy()
                trial[g] = np.clip(trial[g] + delta, 0.2, 0.8)

                # Evaluate it
                y_pred_trial = apply_group_thresholds(
                    scores, sensitive, trial
                )
                dpd_trial = demographic_parity_difference(
                    y_pred_trial, sensitive
                )

                # Keep if better
                if dpd_trial < best_dpd:
                    best_dpd = dpd_trial
                    best_thresholds = trial.copy()
                    improved = True

        # Stop early if no improvement in this full pass
        if not improved:
            print(f"[intervention] Converged after {iteration+1} iterations. "
                  f"New DPD: {best_dpd:.4f}")
            break

    return best_thresholds