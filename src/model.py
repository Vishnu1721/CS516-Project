# src/model.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from config import (FEATURE_COLS, TARGET_COL, SENSITIVE_COL,
                    TEST_SIZE, RANDOM_STATE)
import pandas as pd


def train_baseline(df):

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    s = df[SENSITIVE_COL]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    scores = model.predict_proba(X_test_sc)[:, 1]

    print(f"[model] Baseline Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"[model] Baseline AUC      : {roc_auc_score(y_test, scores):.4f}")

    return model, scaler, X_test.reset_index(drop=True), y_test.reset_index(drop=True), s_test.reset_index(drop=True)


def train_gb(df):

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    s = df[SENSITIVE_COL]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_sc, y_train)

    print("[GB] Accuracy:", accuracy_score(y_test, model.predict(X_test_sc)))

    return model, scaler, X_test.reset_index(drop=True), y_test.reset_index(drop=True), s_test.reset_index(drop=True)