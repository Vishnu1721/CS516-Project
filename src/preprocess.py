# src/preprocess.py

import pandas as pd
from config import DATA_PATH, FEATURE_COLS, TARGET_COL

def load_and_clean():
    """
    Loads COMPAS CSV, applies ProPublica standard filters,
    encodes categorical columns, returns clean dataframe.
    """
    df = pd.read_csv(DATA_PATH)

    # Standard ProPublica filters
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']

    # Keep only Black and White defendants for binary sensitive attribute
    df = df[df['race'].isin(['African-American', 'Caucasian'])]

    # Encode sensitive attribute: 1 = African-American, 0 = Caucasian
    df['race_binary'] = (df['race'] == 'African-American').astype(int)

    # Encode charge degree: 1 = Felony, 0 = Misdemeanor
    df['c_charge_degree'] = (df['c_charge_degree'] == 'F').astype(int)

    # Keep only relevant columns
    keep_cols = FEATURE_COLS + [TARGET_COL, 'race_binary']
    df = df[keep_cols].dropna()

    print(f"[preprocess] Loaded {len(df)} rows after cleaning.")
    print(f"[preprocess] Class balance: {df[TARGET_COL].value_counts().to_dict()}")
    print(f"[preprocess] Group balance: {df['race_binary'].value_counts().to_dict()}")

    return df