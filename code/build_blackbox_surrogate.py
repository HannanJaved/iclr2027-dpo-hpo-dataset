#!/usr/bin/env python3
"""
build_blackbox_surrogate.py

Fits a continuous (log10 lr, log10 beta) regression surrogate over the
cleaned per-size DPO grid and pickles a syne-tune BlackboxSurrogate.

The informed study default is ``gp``. knn1/knn3/knn5 remain available for
ablations but are not submitted by submit_all.sh -- knn1 in particular is
locally constant and under-serves TPE/CQR.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import syne_tune.config_space as sp
from syne_tune.blackbox_repository import add_surrogate
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular

from build_blackbox import add_composite_zscores, drop_fidelity_incomplete_configs, fill_gaps, usable_benchmarks
from common import COMPOSITE_OBJECTIVES, DEFAULT_FAMILY, FAMILIES, FIDELITY_ATTR, TIME_OBJECTIVE, get_paths

SURROGATE_CHOICES = ["knn1", "knn3", "knn5", "gp"]
DEFAULT_SURROGATE = "gp"


def make_surrogate_model(name: str):
    if name == "knn1":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=1)
    if name == "knn3":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=3, weights="distance")
    if name == "knn5":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    if name == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    raise ValueError(name)


def surrogate_path(size_key: str, family: str, surrogate: str):
    size_dir = get_paths(size_key, family=family).size_dir
    out_dir = size_dir / "blackbox_surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{family}-{size_key}-dpo-ao-surrogate-{surrogate}.pkl"


def get_surrogate_results_paths(size_key: str, family: str, surrogate: str = DEFAULT_SURROGATE, budget_tag: str = "generous"):
    size_dir = get_paths(size_key, family=family).size_dir
    results_dirname = f"results_surrogate_{surrogate}" if budget_tag == "generous" else f"results_surrogate_{surrogate}_{budget_tag}"
    results_dir = size_dir / results_dirname
    return {
        "results_dir": results_dir,
        "simulation_raw_csv": results_dir / "simulation_raw.csv",
        "best_found_csv": results_dir / "best_found.csv",
        "figures_dir": results_dir / "figures",
    }


def build(size_key: str, family: str = DEFAULT_FAMILY, surrogate: str = DEFAULT_SURROGATE):
    paths = get_paths(size_key, family=family)
    df = pd.read_csv(paths.grid_csv)
    df = drop_fidelity_incomplete_configs(df, size_key)
    benchmarks = usable_benchmarks(df, size_key)
    df = fill_gaps(df, benchmarks, size_key)
    df = add_composite_zscores(df, benchmarks)
    objectives = benchmarks + COMPOSITE_OBJECTIVES + [TIME_OBJECTIVE]

    df["log_lr"] = np.log10(df["lr"])
    df["log_beta"] = np.log10(df["beta"])

    fidelity_values = np.sort(df["dpo_step"].unique())
    configs = df[["log_lr", "log_beta"]].drop_duplicates().sort_values(["log_lr", "log_beta"]).reset_index(drop=True)

    n_evals, n_seeds, n_fidelities, n_objectives = (
        len(configs), 1, len(fidelity_values), len(objectives),
    )
    objectives_evaluations = np.full((n_evals, n_seeds, n_fidelities, n_objectives), np.nan)

    fidelity_index = {step: i for i, step in enumerate(fidelity_values)}
    config_index = {(row.log_lr, row.log_beta): i for i, row in enumerate(configs.itertuples())}

    for _, row in df.iterrows():
        ci = config_index[(row["log_lr"], row["log_beta"])]
        fi = fidelity_index[row["dpo_step"]]
        objectives_evaluations[ci, 0, fi, :] = [row[obj] for obj in objectives]

    assert not np.isnan(objectives_evaluations).any(), "every (config, fidelity) cell must be filled"

    discrete_configuration_space = {
        "log_lr": sp.choice(sorted(configs["log_lr"].unique().tolist())),
        "log_beta": sp.choice(sorted(configs["log_beta"].unique().tolist())),
    }
    fidelity_space = {FIDELITY_ATTR: sp.randint(0, int(fidelity_values.max()))}
    placeholder = BlackboxTabular(
        hyperparameters=configs[["log_lr", "log_beta"]],
        configuration_space=discrete_configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objectives_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objectives,
    )

    log_lr_min, log_lr_max = float(configs["log_lr"].min()), float(configs["log_lr"].max())
    log_beta_min, log_beta_max = float(configs["log_beta"].min()), float(configs["log_beta"].max())
    continuous_configuration_space = {
        "log_lr": sp.uniform(log_lr_min, log_lr_max),
        "log_beta": sp.uniform(log_beta_min, log_beta_max),
    }

    surrogate_blackbox = add_surrogate(
        placeholder,
        surrogate=make_surrogate_model(surrogate),
        configuration_space=continuous_configuration_space,
        predict_curves=True,
    )
    return surrogate_blackbox, (log_lr_min, log_lr_max), (log_beta_min, log_beta_max)


def sanity_check(blackbox, log_lr_range: tuple[float, float], log_beta_range: tuple[float, float]) -> None:
    print(blackbox)
    fidelity_values = blackbox.fidelity_values
    log_lr_mid = (log_lr_range[0] + log_lr_range[1]) / 2
    log_beta_mid = (log_beta_range[0] + log_beta_range[1]) / 2
    cfg = {"log_lr": log_lr_mid, "log_beta": log_beta_mid}

    # Work around BlackboxSurrogate fidelity-index bug: query the full curve.
    curve = blackbox.objective_function(cfg)
    for step in (fidelity_values[0], fidelity_values[-1]):
        fi = int(np.where(fidelity_values == step)[0][0])
        result = dict(zip(blackbox.objectives_names, curve[fi]))
        print(f"Off-grid spot check lr={10**log_lr_mid:.2e} beta={10**log_beta_mid:.4f} @ step {step}: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=FAMILIES)
    parser.add_argument("--surrogate", default=DEFAULT_SURROGATE, choices=SURROGATE_CHOICES)
    args = parser.parse_args()

    blackbox, log_lr_range, log_beta_range = build(args.size, family=args.family, surrogate=args.surrogate)
    sanity_check(blackbox, log_lr_range, log_beta_range)

    out_path = surrogate_path(args.size, args.family, args.surrogate)
    with open(out_path, "wb") as fh:
        pickle.dump(blackbox, fh)
    print(f"Saved {args.surrogate} surrogate blackbox to: {out_path}")
    print(f"Continuous search space: lr in [{10**log_lr_range[0]:.2e}, {10**log_lr_range[1]:.2e}], "
          f"beta in [{10**log_beta_range[0]:.4f}, {10**log_beta_range[1]:.4f}] (both log-uniform)")


if __name__ == "__main__":
    main()
