# make_plots.py

"""
Generates all figures.

Run AFTER run_experiment.py.

Produces:
  results/plots/deployment_before.png    (4-panel, BEFORE intervention)
  results/plots/deployment_after.png     (4-panel, AFTER intervention)
  results/plots/deployment_compare.png   (overlay: before vs after)
  results/plots/transfer_gap.png         (no-intervention degradation)
  results/plots/pareto_frontier.png      (single-window trade-off)
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data_loader import load_slice
from src.model import train_baseline, train_gb
from src.deployment import run_deployment
from src.plots import (plot_deployment_results_before,
                        plot_deployment_results_after,
                        plot_deployment_results_compare,
                        plot_transfer_gap,
                        run_deployment_no_intervention,
                        make_pareto_plot_for_window)
from config import TRAIN_STATE, TRAIN_YEAR, ACS_SAMPLE_SIZE


def main():
    os.makedirs('results/plots', exist_ok=True)

    # ── Step 1: Load training slice ─────────────────────────
    print("Loading training slice...")
    df_train = load_slice(TRAIN_STATE, TRAIN_YEAR,
                          sample_size=ACS_SAMPLE_SIZE)

    # ── Step 2: Train both models ───────────────────────────
    print("\n" + "=" * 60)
    print("Training LogReg...")
    print("=" * 60)
    lr_model, lr_scaler, lr_X_test, lr_y_test, _ = train_baseline(df_train)
    lr_baseline_acc = accuracy_score(
        lr_y_test, lr_model.predict(lr_scaler.transform(lr_X_test))
    )

    print("\n" + "=" * 60)
    print("Training GradientBoost...")
    print("=" * 60)
    gb_model, gb_scaler, _, _, _ = train_gb(df_train)

    # ── Step 3: Reuse / run LogReg deployment ───────────────
    if os.path.exists('results/deployment_pareto.csv'):
        print("\n[plots] Loading existing results/deployment_pareto.csv")
        df_lr = pd.read_csv('results/deployment_pareto.csv')
    else:
        print("\n[plots] Running LogReg deployment (Pareto)...")
        df_lr = run_deployment(lr_model, lr_scaler, df_train)
        df_lr.to_csv('results/deployment_pareto.csv', index=False)

    # ── Step 4: Run / load GB deployment ────────────────────
    if os.path.exists('results/deployment_pareto_gb.csv'):
        print("[plots] Loading existing results/deployment_pareto_gb.csv")
        df_gb = pd.read_csv('results/deployment_pareto_gb.csv')
    else:
        print("\n[plots] Running GradientBoost deployment (Pareto)...")
        df_gb = run_deployment(gb_model, gb_scaler, df_train)
        df_gb.to_csv('results/deployment_pareto_gb.csv', index=False)

    # ── Step 5: All three 4-panel variants ──────────────────
    print("\n[plots] 4-panel BEFORE intervention...")
    plot_deployment_results_before(df_lr, df_gb)

    print("\n[plots] 4-panel AFTER intervention...")
    plot_deployment_results_after(df_lr, df_gb)

    print("\n[plots] Before-vs-after overlay...")
    plot_deployment_results_compare(df_lr, df_gb)

    # ── Step 6: Transfer-gap plot (no intervention) ─────────
    if os.path.exists('results/deployment_no_intervention.csv'):
        print("\n[plots] Loading existing no-intervention results...")
        df_noint = pd.read_csv('results/deployment_no_intervention.csv')
    else:
        print("\n[plots] Running no-intervention deployment...")
        df_noint = run_deployment_no_intervention(lr_model, lr_scaler, df_train)
        df_noint.to_csv('results/deployment_no_intervention.csv', index=False)
    plot_transfer_gap(df_noint, baseline_acc=lr_baseline_acc)

    # ── Step 7: Pareto frontier for one window ──────────────
    print("\n[plots] Pareto frontier (window 10, PR 2018)...")
    make_pareto_plot_for_window(lr_model, lr_scaler, window_idx=10)

    print("\n" + "=" * 60)
    print("All plots saved to results/plots/")
    print("=" * 60)


if __name__ == '__main__':
    main()