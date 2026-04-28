# src/preprocess.py

import pandas as pd
from folktables import ACSDataSource, ACSIncome
from config import (ACS_STATE, ACS_YEAR, ACS_SAMPLE_SIZE,
                    FEATURE_COLS, TARGET_COL, RANDOM_STATE)


def load_and_clean():
    """
    Loads ACS Income data via folktables, encodes sensitive attribute,
    optionally subsamples for speed, and returns a clean dataframe.

    Target: PINCP (1 = income > $50k, 0 otherwise) — already binarized
            by folktables' ACSIncome task.
    Sensitive: sex_binary (1 = Female, 0 = Male)
    """
    # ── Step 1: Download / load raw ACS data ─────────────────
    print(f"[preprocess] Loading ACS {ACS_YEAR} data for state={ACS_STATE}...")
    data_source = ACSDataSource(
        survey_year=ACS_YEAR,
        horizon='1-Year',
        survey='person'
    )
    acs_data = data_source.get_data(states=[ACS_STATE], download=True)

    # ── Step 2: Apply ACSIncome task definition ──────────────
    # This handles filtering (e.g., age >= 16, working hours > 0)
    # and produces a binary target.
    features, label, group = ACSIncome.df_to_pandas(acs_data)

    # ── Step 3: Build clean dataframe ────────────────────────
    df = features.copy()
    # df[TARGET_COL] = label.astype(int)
    df['income_binary'] = label.astype(int)

    # Encode sensitive attribute: SEX is coded 1=Male, 2=Female in ACS
    df['sex_binary'] = (features['SEX'] == 2).astype(int)

    # ── Step 4: Subsample if configured ──────────────────────
    if ACS_SAMPLE_SIZE is not None and len(df) > ACS_SAMPLE_SIZE:
        df = df.sample(n=ACS_SAMPLE_SIZE, random_state=RANDOM_STATE)
        df = df.reset_index(drop=True)

    # ── Step 5: Keep only relevant columns and drop NaNs ─────
    keep_cols = FEATURE_COLS + [TARGET_COL, 'sex_binary']
    df = df[keep_cols].dropna().reset_index(drop=True)

    # ── Step 6: Report ───────────────────────────────────────
    print(f"[preprocess] Loaded {len(df)} rows after cleaning.")
    print(f"[preprocess] Class balance: {df[TARGET_COL].value_counts().to_dict()}")
    print(f"[preprocess] Group balance: {df['sex_binary'].value_counts().to_dict()}")

    return df