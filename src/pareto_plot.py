# src/pareto_plot.py

import numpy as np
import matplotlib.pyplot as plt
from src.intervention import compute_metric_vector


def plot_pareto_frontier(scores, y_true, sensitive,
                          metric_x_idx=0, metric_y_idx=3,
                          metric_x_name='DPD',
                          metric_y_name='Predictive Parity Diff',
                          grid_step=0.02,
                          save_path='results/plots/pareto_frontier.png'):
    """
    Visualize the Pareto frontier on two chosen fairness metrics.

    Plots all candidate threshold pairs as dots in (metric_x, metric_y) space,
    highlights the Pareto-optimal ones in red, and marks the selected
    candidate (minimum weighted sum) with a star.

    This visually demonstrates the impossibility theorem — typically you
    cannot push both metrics to zero simultaneously; only the trade-off
    frontier is reachable.

    Default: DPD (idx 0) on x-axis, Predictive Parity (idx 3) on y-axis.
    These two are known to be incompatible (Chouldechova, 2017).
    """
    # ── Step 1: Evaluate the full grid ──────────────────────
    candidates_t = np.arange(0.2, 0.8 + grid_step, grid_step)
    all_thresholds = []
    all_vectors = []

    for t0 in candidates_t:
        for t1 in candidates_t:
            thr = {0: float(t0), 1: float(t1)}
            vec = compute_metric_vector(scores, y_true, sensitive, thr)
            all_thresholds.append(thr)
            all_vectors.append(vec)

    all_vectors = np.array(all_vectors)
    n = len(all_vectors)

    # ── Step 2: Find Pareto-optimal points (full 6D check) ──
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        diff = all_vectors - all_vectors[i]
        dominated_by = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
        if dominated_by.any():
            pareto_mask[i] = False

    # ── Step 3: Pick the selected one (min weighted sum) ────
    weighted_scores = all_vectors[pareto_mask].sum(axis=1)
    pareto_indices = np.where(pareto_mask)[0]
    selected_idx = pareto_indices[np.argmin(weighted_scores)]

    # ── Step 4: Plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    # All candidates (gray)
    ax.scatter(all_vectors[~pareto_mask, metric_x_idx],
                all_vectors[~pareto_mask, metric_y_idx],
                alpha=0.25, s=18, color='lightgray',
                label=f'Dominated candidates ({(~pareto_mask).sum()})')

    # Pareto-optimal (red)
    ax.scatter(all_vectors[pareto_mask, metric_x_idx],
                all_vectors[pareto_mask, metric_y_idx],
                alpha=0.85, s=45, color='crimson',
                edgecolor='darkred', linewidth=0.5,
                label=f'Pareto-optimal ({pareto_mask.sum()})')

    # Selected point (gold star)
    ax.scatter(all_vectors[selected_idx, metric_x_idx],
                all_vectors[selected_idx, metric_y_idx],
                s=350, marker='*', color='gold',
                edgecolor='black', linewidth=1.5, zorder=5,
                label='Selected (min weighted sum)')

    ax.set_xlabel(metric_x_name, fontsize=12)
    ax.set_ylabel(metric_y_name, fontsize=12)
    ax.set_title('Pareto Frontier of Fairness Trade-offs\n'
                 'No threshold pair can minimize both metrics simultaneously',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[pareto_plot] Saved to {save_path}")
    plt.show()

    # Return the selected thresholds for inspection
    return all_thresholds[selected_idx], all_vectors[selected_idx]