"""Incumbent selection that does not reward extra intermediate reports.

The old rule took argmax of the search metric over every checkpoint. Methods
that emit more intermediate observations then get more chances to post an
unusually high z-score (often at step 200, before ASHA's first rung).

Correct rule, applied independently to each (search_objective, optimizer, seed):

  1. If any trial reached ``final_fidelity``, recommend the trial with the
     best search-metric value *at that final fidelity*.
  2. Otherwise recommend the trial with the best search-metric value among
     those evaluated at the highest fidelity present in the run (so every
     candidate is scored at the same checkpoint).

The recommended ``(lr, beta)`` is then scored at true final fidelity by the
summarizer, same as before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import FIDELITY_ATTR

GROUP_COLS = ["search_objective", "optimizer", "seed"]


def infer_final_fidelity(raw: pd.DataFrame) -> int:
    return int(raw[FIDELITY_ATTR].max())


def select_incumbents(raw: pd.DataFrame, final_fidelity: int | None = None) -> pd.DataFrame:
    """One recommended row per (search_objective, optimizer, seed)."""
    if raw.empty:
        return raw.copy()
    if final_fidelity is None:
        final_fidelity = infer_final_fidelity(raw)

    rows = []
    for (objective, optimizer, seed), g in raw.groupby(GROUP_COLS, sort=False):
        finished = g[g[FIDELITY_ATTR] == final_fidelity]
        if len(finished):
            pool = finished
            rule = "finished"
        else:
            highest = int(g[FIDELITY_ATTR].max())
            pool = g[g[FIDELITY_ATTR] == highest]
            rule = "highest_common"
        pool = pool.sort_values("st_tuner_time")
        per_trial = pool.groupby("trial_id", as_index=False).tail(1)
        best = per_trial.loc[per_trial[objective].idxmax()].copy()
        best["selection_rule"] = rule
        best["selection_fidelity"] = int(best[FIDELITY_ATTR])
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def incumbent_lr_beta_over_time(
    g: pd.DataFrame, metric: str, final_fidelity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Running recommendation after each time-sorted report in ``g``.

    Same rule as :func:`select_incumbents`, restricted to the prefix of reports
    available at that simulated time. Used by anytime regret plots so the
    end-of-budget incumbent matches ``best_found.csv``.
    """
    g = g.sort_values("st_tuner_time")
    n = len(g)
    out_lr = np.empty(n, dtype=float)
    out_beta = np.empty(n, dtype=float)
    steps = g[FIDELITY_ATTR].to_numpy()
    scores = g[metric].to_numpy()
    lrs = g["lr"].to_numpy()
    betas = g["beta"].to_numpy()

    best_finished = -np.inf
    fin_lr = fin_beta = None
    highest = -1
    best_at_high = -np.inf
    high_lr = high_beta = None

    for i in range(n):
        step = int(steps[i])
        score = float(scores[i])
        if step == final_fidelity:
            if score > best_finished:
                best_finished = score
                fin_lr, fin_beta = float(lrs[i]), float(betas[i])
        elif fin_lr is None:
            if step > highest:
                highest = step
                best_at_high = score
                high_lr, high_beta = float(lrs[i]), float(betas[i])
            elif step == highest and score > best_at_high:
                best_at_high = score
                high_lr, high_beta = float(lrs[i]), float(betas[i])

        if fin_lr is not None:
            out_lr[i], out_beta[i] = fin_lr, fin_beta
        else:
            out_lr[i], out_beta[i] = high_lr, high_beta
    return out_lr, out_beta
