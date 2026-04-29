# run_experiment.py

"""
Main entry point: train on one (state, year) slice, deploy across many.

Compares two intervention methods on the same real-data deployment stream:
  1. Pareto frontier (multi-metric, our method)
  2. Hill-climbing on DPD only (baseline)

Outputs:
  results/deployment_pareto.csv
  results/deployment_hillclimb.csv
  results/comparison_summary.csv
"""

import os
import pandas as pd

from src.data_loader import load_slice
from src.model import train_baseline
from src.deployment import (run_deployment,
                             run_deployment_baseline_hillclimb)
from config import TRAIN_STATE, TRAIN_YEAR, ACS_SAMPLE_SIZE


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # ── Step 1: Load training slice ─────────────────────────
    print("=" * 60)
    print(f"TRAINING on {TRAIN_STATE} {TRAIN_YEAR}")
    print("=" * 60)
    df_train = load_slice(TRAIN_STATE, TRAIN_YEAR,
                          sample_size=ACS_SAMPLE_SIZE)

    # ── Step 2: Train frozen model ──────────────────────────
    model, scaler, X_test, y_test, s_test = train_baseline(df_train)

    # ── Step 3: Run deployment with Pareto intervention ─────
    print("\n" + "=" * 60)
    print("DEPLOYMENT — Pareto frontier method (ours)")
    print("=" * 60)
    df_pareto = run_deployment(model, scaler, df_train)
    df_pareto.to_csv('results/deployment_pareto.csv', index=False)
    print(f"\n[saved] results/deployment_pareto.csv")

    # ── Step 4: Run deployment with hill-climb baseline ─────
    # Reuse the same trained model/scaler — only the intervention differs
    print("\n" + "=" * 60)
    print("DEPLOYMENT — Hill-climbing DPD-only (baseline)")
    print("=" * 60)
    df_hc = run_deployment_baseline_hillclimb(model, scaler, df_train)
    df_hc.to_csv('results/deployment_hillclimb.csv', index=False)
    print(f"\n[saved] results/deployment_hillclimb.csv")

    # ── Step 5: Side-by-side comparison ─────────────────────
    summary = pd.DataFrame({
        'window': df_pareto['window'],
        'state':  df_pareto['state'],
        'year':   df_pareto['year'],
        'label':  df_pareto['label'],

        'dpd_pareto':     df_pareto['dpd_after'],
        'dpd_hillclimb':  df_hc['dpd_after'],

        'eod_pareto':     df_pareto['eod_after'],
        'eod_hillclimb':  df_hc['eod_after'],

        'pp_pareto':      df_pareto['pp_diff_after'],
        'pp_hillclimb':   df_hc['pp_diff_after'],

        'accgap_pareto':    df_pareto['acc_gap_after'],
        'accgap_hillclimb': df_hc['acc_gap_after'],

        'acc_pareto':     df_pareto['acc_after'],
        'acc_hillclimb':  df_hc['acc_after'],
    })
    summary.to_csv('results/comparison_summary.csv', index=False)
    print(f"[saved] results/comparison_summary.csv")

    # ── Step 6: Quick aggregate report ──────────────────────
    print("\n" + "=" * 60)
    print("AGGREGATE COMPARISON (mean across windows)")
    print("=" * 60)
    print(f"{'Metric':<12} {'Pareto':>10} {'HillClimb':>12} {'Δ (P-H)':>10}")
    print("-" * 46)
    for metric in ['dpd', 'eod', 'pp', 'accgap', 'acc']:
        p_mean = summary[f'{metric}_pareto'].mean()
        h_mean = summary[f'{metric}_hillclimb'].mean()
        diff = p_mean - h_mean
        print(f"{metric:<12} {p_mean:>10.4f} {h_mean:>12.4f} {diff:>+10.4f}")

    print("\n(For fairness metrics, lower is better. "
          "For accuracy, higher is better.)")


if __name__ == '__main__':
    main()