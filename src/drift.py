# src/drift.py

import numpy as np
import pandas as pd

def simulate_demographic_shift(X, y, sensitive,
                                target_ratio, random_state=42):
    """
    Resamples the window so that fraction of group=1
    equals target_ratio.

    WHY THIS MATTERS:
    A model trained in one county may be deployed in another
    county with a different racial composition. The model
    never saw this composition during training.

    target_ratio = 0.70 means 70% of this window is group=1
    """
    rng = np.random.RandomState(random_state)

    idx_g1 = np.where(sensitive == 1)[0]  # African-American indices
    idx_g0 = np.where(sensitive == 0)[0]  # Caucasian indices

    n_total    = len(sensitive)
    n_g1_want  = int(n_total * target_ratio)
    n_g0_want  = n_total - n_g1_want

    # Cap at available samples, sample with replacement if needed
    chosen_g1 = rng.choice(idx_g1, n_g1_want, replace=True)
    chosen_g0 = rng.choice(idx_g0, n_g0_want, replace=True)

    chosen = np.concatenate([chosen_g1, chosen_g0])
    rng.shuffle(chosen)

    return (X.iloc[chosen].reset_index(drop=True),
            y.iloc[chosen].reset_index(drop=True),
            sensitive.iloc[chosen].reset_index(drop=True))


def simulate_feature_drift(X, drift_magnitude=0.2, random_state=42):
    """
    Adds Gaussian noise proportional to each feature's std.

    WHY THIS MATTERS:
    Over time, how prior counts are recorded may change,
    or the age distribution shifts. The model's features
    now look slightly different from what it was trained on.

    drift_magnitude=0.2 means noise std = 20% of feature std
    """
    rng = np.random.RandomState(random_state)   
    X_drifted = X.copy().astype(float)          

    for col in X.columns:
        # Only apply to continuous features
        if col in ['AGEP', 'WKHP']:
            noise = rng.normal(
                loc=0,
                scale=drift_magnitude * X[col].std(),
                size=len(X)
            )
            X_drifted[col] += noise

    return X_drifted 


def simulate_label_noise(y, noise_rate=0.1, random_state=42):
    """
    Randomly flips a fraction of labels.

    WHY THIS MATTERS:
    In recidivism prediction, the label (re-arrested within 2 years)
    depends on police activity. If policing patterns change,
    or the follow-up period shortens, some labels become unreliable.
    This simulates that degraded outcome reliability.

    noise_rate=0.10 means 10% of labels are randomly flipped
    """
    rng = np.random.RandomState(random_state)
    y_noisy = y.copy()

    n_flip = int(len(y) * noise_rate)
    flip_idx = rng.choice(len(y), n_flip, replace=False)
    y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]

    return y_noisy