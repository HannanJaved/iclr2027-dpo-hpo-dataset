#!/usr/bin/env python3
"""Regret curves for one (size, surrogate, budget) of the informed study."""
from __future__ import annotations

import argparse

import pandas as pd

from analyze_results import plot_regret_curves
from build_blackbox_surrogate import get_surrogate_results_paths
from common import DEFAULT_FAMILY, FAMILIES, FAMILY_LABELS, SEARCH_OBJECTIVES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=FAMILIES)
    parser.add_argument("--surrogate", default="gp", choices=["knn1", "knn3", "knn5", "gp"])
    parser.add_argument("--budget-tag", default="generous", choices=["generous", "tight"])
    args = parser.parse_args()

    paths = get_surrogate_results_paths(
        args.size, family=args.family, surrogate=args.surrogate, budget_tag=args.budget_tag,
    )
    family_label = FAMILY_LABELS.get(args.family, args.family.capitalize())
    budget_note = "" if args.budget_tag == "generous" else f", {args.budget_tag}"
    label = f"{family_label}-{args.size} (GP surrogate{budget_note})" if args.surrogate == "gp" \
        else f"{family_label}-{args.size} (surrogate {args.surrogate}{budget_note})"

    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(paths["simulation_raw_csv"])

    search_objectives = [o for o in SEARCH_OBJECTIVES if o in raw["search_objective"].unique()]
    ordered_objectives = [o for o in SEARCH_OBJECTIVES if o in search_objectives]

    plot_regret_curves(raw, headline_objectives=ordered_objectives, size_key=label,
                        figures_dir=paths["figures_dir"], x_axis="time")
    plot_regret_curves(raw, headline_objectives=ordered_objectives, size_key=label,
                        figures_dir=paths["figures_dir"], x_axis="evals")


if __name__ == "__main__":
    main()
