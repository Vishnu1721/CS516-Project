# src/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def demographic_parity_difference(y_pred, sensitive):
    """
    |P(ŷ=1 | group=1) - P(ŷ=1 | group=0)|
    Measures gap in positive prediction rates between groups.
    """
    groups = np.unique(sensitive)
    rates = [y_pred[sensitive == g].mean() for g in groups]
    return abs(rates[0] - rates[1])


def false_positive_rate(y_true, y_pred):
    """Among true negatives, fraction wrongly flagged as positive."""
    neg_mask = (y_true == 0)
    if neg_mask.sum() == 0:
        return 0.0
    return y_pred[neg_mask].mean()


def false_negative_rate(y_true, y_pred):
    """Among true positives, fraction missed by the model."""
    pos_mask = (y_true == 1)
    if pos_mask.sum() == 0:
        return 0.0
    return (1 - y_pred[pos_mask]).mean()


def equalized_odds_difference(y_true, y_pred, sensitive):
    """
    Returns FPR gap, FNR gap, and their sum (EOD).
    """
    groups = np.unique(sensitive)

    fpr = [false_positive_rate(y_true[sensitive == g], y_pred[sensitive == g])
           for g in groups]
    fnr = [false_negative_rate(y_true[sensitive == g], y_pred[sensitive == g])
           for g in groups]

    fpr_gap = abs(fpr[0] - fpr[1])
    fnr_gap = abs(fnr[0] - fnr[1])

    return fpr_gap, fnr_gap, fpr_gap + fnr_gap


# ─── NEW METRICS ────────────────────────────────────────────────

def precision_score_safe(y_true, y_pred):
    """
    Precision = TP / (TP + FP). Returns 0 if no positive predictions.
    """
    pred_pos_mask = (y_pred == 1)
    if pred_pos_mask.sum() == 0:
        return 0.0
    return y_true[pred_pos_mask].mean()


def predictive_parity_difference(y_true, y_pred, sensitive):
    """
    |Precision(group=0) - Precision(group=1)|
    Asks: When the model predicts positive, is it equally accurate
    across groups? This is the metric Northpointe used to defend COMPAS
    against ProPublica's FPR-based critique.
    """
    groups = np.unique(sensitive)
    precisions = [precision_score_safe(
                      y_true[sensitive == g],
                      y_pred[sensitive == g])
                  for g in groups]
    return abs(precisions[0] - precisions[1])


def accuracy_gap(y_true, y_pred, sensitive):
    """
    |Accuracy(group=0) - Accuracy(group=1)|
    Asks: Does the model perform equally well on both groups overall?
    """
    groups = np.unique(sensitive)
    accs = [accuracy_score(y_true[sensitive == g], y_pred[sensitive == g])
            for g in groups]
    return abs(accs[0] - accs[1])


def auc_gap(y_true, scores, sensitive):
    """
    |AUC(group=0) - AUC(group=1)|
    Asks: Is the model's ranking ability equal across groups?
    Threshold-independent — captures model quality, not cutoff choice.
    Returns 0 if either group has only one class present.
    """
    groups = np.unique(sensitive)
    aucs = []
    for g in groups:
        mask = (sensitive == g)
        y_g = y_true[mask]
        s_g = scores[mask]
        if len(np.unique(y_g)) < 2:
            aucs.append(0.5)  # undefined; treat as random
        else:
            aucs.append(roc_auc_score(y_g, s_g))
    return abs(aucs[0] - aucs[1])


# ─── AGGREGATOR ─────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, scores, sensitive):
    """
    Computes all metrics for one deployment window.
    """
    fpr_gap, fnr_gap, eod = equalized_odds_difference(
        y_true, y_pred, sensitive
    )

    return {
        'accuracy'    : accuracy_score(y_true, y_pred),
        'auc'         : roc_auc_score(y_true, scores),
        'dpd'         : demographic_parity_difference(y_pred, sensitive),
        'fpr_gap'     : fpr_gap,
        'fnr_gap'     : fnr_gap,
        'eod'         : eod,
        'pp_diff'     : predictive_parity_difference(y_true, y_pred, sensitive),
        'acc_gap'     : accuracy_gap(y_true, y_pred, sensitive),
        'auc_gap'     : auc_gap(y_true, scores, sensitive),
    }