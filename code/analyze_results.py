#!/usr/bin/env python3
"""plot_regret_curves helper used by analyze_results_surrogate.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OPTIMIZER_COLORS, OPTIMIZERS


def plot_regret_curves(raw: pd.DataFrame, headline_objectives: list[str], size_key: str, figures_dir,
                        x_axis: str = "time") -> None:
    assert x_axis in ("time", "evals")
    n_points = 60
    n = len(headline_objectives)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")

    for ax, objective in zip(axes, headline_objectives):
        sub = raw[raw["search_objective"] == objective]
        if x_axis == "time":
            x_grid = np.linspace(0, sub["st_tuner_time"].max(), n_points)
        else:
            x_grid = np.arange(1, sub.groupby(["optimizer", "seed"]).size().max() + 1)

        for optimizer in OPTIMIZERS:
            opt_sub = sub[sub["optimizer"] == optimizer]
            seed_curves = []
            for seed, g in opt_sub.groupby("seed"):
                g = g.sort_values("st_tuner_time")
                best_so_far = g[objective].cummax().to_numpy()
                x_obs = g["st_tuner_time"].to_numpy() if x_axis == "time" else np.arange(1, len(best_so_far) + 1)
                curve = np.interp(x_grid, x_obs, best_so_far, left=np.nan, right=best_so_far[-1])
                seed_curves.append(curve)
            arr = np.array(seed_curves)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0, ddof=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(min=1))
            color = OPTIMIZER_COLORS[optimizer]
            x_plot = x_grid / 3600 if x_axis == "time" else x_grid
            ax.plot(x_plot, mean, color=color, linewidth=2, label=optimizer)
            ax.fill_between(x_plot, mean - sem, mean + sem, color=color, alpha=0.15)

        ax.set_xlabel("Simulated wallclock time (hours, 2 workers)" if x_axis == "time"
                      else "Cumulative blackbox evaluations\n(config x checkpoint queries)", fontsize=9)
        ax.set_ylabel(f"Best {objective} found so far", fontsize=9)
        ax.set_title(f"Search objective: {objective}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    budget_desc = "simulated budget" if x_axis == "time" else "number of evaluations"
    fig.suptitle(
        f"{size_key} DPO-AO — optimizer comparison — mean best-value-so-far vs. {budget_desc} "
        "(25 seeds, shaded = ±1 SEM)",
        fontsize=11,
    )
    fig.tight_layout()
    filename = "regret_curves.png" if x_axis == "time" else "regret_curves_by_evals.png"
    out = figures_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
