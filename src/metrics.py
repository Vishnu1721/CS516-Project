# src/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def demographic_parity_difference(y_pred, sensitive):
    """
    |P(ŷ=1 | group=1) - P(ŷ=1 | group=0)|
    Measures gap in positive prediction rates between groups.
    Should be 0 for a perfectly fair model.
    """
    groups = np.unique(sensitive)
    rates = [y_pred[sensitive == g].mean() for g in groups]
    return abs(rates[0] - rates[1])


def false_positive_rate(y_true, y_pred):
    """
    Among people who did NOT recidivate (y=0),
    what fraction did the model incorrectly flag (ŷ=1)?
    This is the ProPublica metric — they showed it was
    higher for Black defendants in COMPAS.
    """
    neg_mask = (y_true == 0)
    if neg_mask.sum() == 0:
        return 0.0
    return y_pred[neg_mask].mean()


def false_negative_rate(y_true, y_pred):
    """
    Among people who DID recidivate (y=1),
    what fraction did the model miss (ŷ=0)?
    """
    pos_mask = (y_true == 1)
    if pos_mask.sum() == 0:
        return 0.0
    return (1 - y_pred[pos_mask]).mean()


def equalized_odds_difference(y_true, y_pred, sensitive):
    """
    FPR gap + FNR gap across groups.
    Both need to be equal across groups for equalized odds.
    Returns fpr_gap, fnr_gap, and total eod separately
    so you can track which component is driving the violation.
    """
    groups = np.unique(sensitive)

    fpr = [false_positive_rate(
                y_true[sensitive == g],
                y_pred[sensitive == g])
           for g in groups]

    fnr = [false_negative_rate(
                y_true[sensitive == g],
                y_pred[sensitive == g])
           for g in groups]

    fpr_gap = abs(fpr[0] - fpr[1])
    fnr_gap = abs(fnr[0] - fnr[1])

    return fpr_gap, fnr_gap, fpr_gap + fnr_gap


def compute_all_metrics(y_true, y_pred, scores, sensitive):
    """
    Computes all metrics for one deployment window.
    Returns a dictionary.
    """
    fpr_gap, fnr_gap, eod = equalized_odds_difference(
        y_true, y_pred, sensitive
    )

    return {
        'accuracy' : accuracy_score(y_true, y_pred),
        'auc'      : roc_auc_score(y_true, scores),
        'dpd'      : demographic_parity_difference(y_pred, sensitive),
        'fpr_gap'  : fpr_gap,
        'fnr_gap'  : fnr_gap,
        'eod'      : eod
    }