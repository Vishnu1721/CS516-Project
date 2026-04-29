# src/data_loader.py

import pandas as pd
from folktables import ACSDataSource, ACSIncome
from config import FEATURE_COLS, TARGET_COL, RANDOM_STATE


def load_slice(state, year, sample_size=None, verbose=True):
    """
    Load one (state, year) slice of ACS Income data.

    Returns a clean dataframe with FEATURE_COLS + target + sex_binary.
    Used both for training (one slice) and for deployment windows
    (different slices, real distribution shift).
    """
    if verbose:
        print(f"[data_loader] Loading ACS {year} for state={state}...")

    data_source = ACSDataSource(
        survey_year=year,
        horizon='1-Year',
        survey='person'
    )
    acs_data = data_source.get_data(states=[state], download=True)

    # ACSIncome handles age >= 16, hours > 0 filtering and binarizes target
    features, label, _ = ACSIncome.df_to_pandas(acs_data)

    df = features.copy()
    df[TARGET_COL] = label.astype(int)

    # SEX is 1=Male, 2=Female in ACS — map to sex_binary (1=Female)
    df['sex_binary'] = (features['SEX'] == 2).astype(int)

    # Subsample to keep windows comparable in size
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        df = df.reset_index(drop=True)

    keep_cols = FEATURE_COLS + [TARGET_COL, 'sex_binary']
    df = df[keep_cols].dropna().reset_index(drop=True)

    if verbose:
        print(f"  → {len(df)} rows | "
              f"pos rate: {df[TARGET_COL].mean():.3f} | "
              f"female rate: {df['sex_binary'].mean():.3f}")

    return df


def measure_shift(df_train, df_window):
    """
    Post-hoc measurement of how much a window has shifted from training.

    Returns a dict of shift magnitudes — used to (a) classify likely_cause
    in the results log, and (b) correlate shift magnitude with how much
    the intervention helped.
    """
    return {
        # Label base rate shift — captures economic / temporal shift
        'pos_rate_shift':    abs(df_window[TARGET_COL].mean()
                                 - df_train[TARGET_COL].mean()),

        # Demographic shift — group ratio change
        'female_rate_shift': abs(df_window['sex_binary'].mean()
                                 - df_train['sex_binary'].mean()),

        # Continuous feature shifts, normalized by training std
        'agep_mean_shift':   abs(df_window['AGEP'].mean()
                                 - df_train['AGEP'].mean()) / df_train['AGEP'].std(),
        'wkhp_mean_shift':   abs(df_window['WKHP'].mean()
                                 - df_train['WKHP'].mean()) / df_train['WKHP'].std(),
    }