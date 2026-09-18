#!/usr/bin/env python3
"""
run_simulations_surrogate.py

Continuous-space HPO against a GP (or knn) surrogate: RandomSearch / TPE /
CQR / ASHA / BOHB x Z-Dynamic / Z-All / Z-Macro x 25 seeds.

ASHA/BOHB first rung is ASHA_GRACE_PERIOD (400). Search space is continuous
log_lr x log_beta, not discrete cfg_id.

End-of-run incumbent (best_found.csv): best finished trial on the search
metric, or if none finished the best trial at the highest common fidelity.
See select_incumbent.py. Do not argmax over every checkpoint report.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing
import os
import pickle
import time

RESULTS_ROOT = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default="qwen3", choices=["qwen3", "llama"])
    parser.add_argument("--surrogate", default="gp", choices=["knn1", "knn3", "knn5", "gp"])
    parser.add_argument("--budget-tag", default="generous", choices=["generous", "tight"])
    parser.add_argument("--objectives", default=None,
                         help="Comma-separated subset of SEARCH_OBJECTIVES (default: all). "
                              "Merges into an existing simulation_raw.csv.")
    parser.add_argument("--optimizers", default=None,
                         help="Comma-separated subset of OPTIMIZERS (default: all). "
                              "Merges into an existing simulation_raw.csv.")
    parser.add_argument("--n-procs", type=int, default=None,
                         help="Worker processes for the outer simulation loop (default: os.cpu_count()).")
    return parser.parse_args()


_args = _parse_args()


def _setup_results_root(size_key: str, family: str, surrogate: str, budget_tag: str) -> dict:
    global RESULTS_ROOT
    from build_blackbox_surrogate import get_surrogate_results_paths

    paths = get_surrogate_results_paths(size_key, family=family, surrogate=surrogate, budget_tag=budget_tag)
    RESULTS_ROOT = paths["results_dir"] / "syne_tune_runs"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["SYNETUNE_FOLDER"] = str(RESULTS_ROOT)
    return paths


_paths = _setup_results_root(_args.size, _args.family, _args.surrogate, _args.budget_tag)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from syne_tune import StoppingCriterion, Tuner  # noqa: E402
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback  # noqa: E402
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend  # noqa: E402
from syne_tune.experiments import load_experiment  # noqa: E402
from build_blackbox_surrogate import surrogate_path  # noqa: E402
from common import (  # noqa: E402
    ASHA_GRACE_PERIOD,
    ASHA_REDUCTION_FACTOR,
    BUDGET_REGIMES,
    FIDELITY_ATTR,
    N_SEEDS,
    N_WORKERS,
    OPTIMIZERS,
    SEARCH_OBJECTIVES,
    TIME_OBJECTIVE,
    get_paths,
)
from schedulers import make_scheduler  # noqa: E402
from select_incumbent import select_incumbents  # noqa: E402

_WORKER_BLACKBOX = None


def _worker_init(blackbox_path: str) -> None:
    global _WORKER_BLACKBOX
    with open(blackbox_path, "rb") as fh:
        _WORKER_BLACKBOX = pickle.load(fh)


def run_one(objective: str, optimizer: str, seed: int, max_t: int, min_t: int, budget: float) -> pd.DataFrame:
    blackbox = _WORKER_BLACKBOX
    scheduler = make_scheduler(optimizer, blackbox.configuration_space, objective, max_t, min_t, seed)
    backend = UserBlackboxBackend(blackbox=blackbox, elapsed_time_attr=TIME_OBJECTIVE)
    tuner_name = f"{objective}-{optimizer}-s{seed}"[:60].replace(" ", "")
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=budget),
        n_workers=N_WORKERS,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        tuner_name=tuner_name,
        suffix_tuner_name=True,
        save_tuner=False,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        tuner.run()
    exp = load_experiment(tuner.name)
    df = exp.results.copy()
    df["search_objective"] = objective
    df["optimizer"] = optimizer
    df["seed"] = seed
    return df


def _run_one_task(task: tuple[str, str, int, int, int, float]) -> tuple[str, str, int, pd.DataFrame | None, str | None]:
    objective, optimizer, seed, max_t, min_t, budget = task
    try:
        return objective, optimizer, seed, run_one(objective, optimizer, seed, max_t, min_t, budget), None
    except Exception as exc:  # noqa: BLE001
        return objective, optimizer, seed, None, str(exc)


def main() -> None:
    size_key = _args.size
    family = _args.family
    n_procs = _args.n_procs or os.cpu_count()

    surrogate = _args.surrogate
    blackbox_path = str(surrogate_path(size_key, family, surrogate))
    with open(blackbox_path, "rb") as fh:
        blackbox = pickle.load(fh)
    max_t = int(blackbox.fidelity_values.max())
    fidelity_values = np.asarray(blackbox.fidelity_values)
    if ASHA_GRACE_PERIOD not in fidelity_values:
        raise SystemExit(
            f"{size_key}: ASHA_GRACE_PERIOD={ASHA_GRACE_PERIOD} is not in this "
            f"surrogate's fidelities {sorted(fidelity_values.tolist())}"
        )
    min_t = ASHA_GRACE_PERIOD

    grid_df = pd.read_csv(get_paths(size_key, family=family).grid_csv)
    real_point = {"log_lr": float(np.log10(grid_df["lr"].iloc[0])), "log_beta": float(np.log10(grid_df["beta"].iloc[0]))}

    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    curve = blackbox.objective_function(real_point)
    full_run_cost = float(curve[-1, time_idx])
    budget_tag = _args.budget_tag
    budget = BUDGET_REGIMES[budget_tag] * full_run_cost
    print(f"{family}-{size_key} (surrogate={surrogate}), budget regime {budget_tag!r} "
          f"({BUDGET_REGIMES[budget_tag]}x): full-fidelity run cost ~{full_run_cost:.0f}s "
          f"-> budget {budget:.0f}s/run; ASHA/BOHB grace_period={ASHA_GRACE_PERIOD}, "
          f"reduction_factor={ASHA_REDUCTION_FACTOR}")

    requested = _args.objectives.split(",") if _args.objectives else SEARCH_OBJECTIVES
    requested = [o.strip() for o in requested if o.strip()]
    objectives = [o for o in requested if o in blackbox.objectives_names]
    skipped = [o for o in requested if o not in objectives]
    if skipped:
        print(f"[WARN] {size_key}: skipping search objectives not available for this size: {skipped}")

    requested_opts = _args.optimizers.split(",") if _args.optimizers else list(OPTIMIZERS)
    optimizers = [o.strip() for o in requested_opts if o.strip()]
    unknown_opts = [o for o in optimizers if o not in OPTIMIZERS]
    if unknown_opts:
        raise SystemExit(f"Unknown optimizer(s) {unknown_opts}; choose from {OPTIMIZERS}")

    tasks = [
        (objective, optimizer, seed, max_t, min_t, budget)
        for objective in objectives
        for optimizer in optimizers
        for seed in range(N_SEEDS)
    ]
    print(f"Running {len(tasks)} simulations "
          f"({len(objectives)} objectives x {len(optimizers)} optimizers x {N_SEEDS} seeds) "
          f"across {n_procs} worker processes")

    all_frames = []
    t0 = time.time()
    n_failed = 0
    with multiprocessing.Pool(processes=n_procs, initializer=_worker_init, initargs=(blackbox_path,)) as pool:
        for i, (objective, optimizer, seed, df, err) in enumerate(pool.imap_unordered(_run_one_task, tasks), 1):
            if err is not None:
                n_failed += 1
                print(f"[WARN] run {objective}/{optimizer}/seed={seed} failed: {err}")
            else:
                all_frames.append(df)
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  {i}/{len(tasks)} done ({elapsed:.0f}s elapsed, {n_failed} failed)", flush=True)

    raw = pd.concat(all_frames, ignore_index=True)
    keep_cols = [
        "search_objective", "optimizer", "seed", "trial_id",
        "config_log_lr", "config_log_beta", FIDELITY_ATTR, "st_tuner_time",
    ] + blackbox.objectives_names
    raw = raw[keep_cols].rename(columns={"config_log_lr": "log_lr", "config_log_beta": "log_beta"})
    raw["lr"] = 10 ** raw["log_lr"]
    raw["beta"] = 10 ** raw["log_beta"]
    raw = raw.drop(columns=["log_lr", "log_beta"])

    full_grid = (set(objectives) >= set(SEARCH_OBJECTIVES)) and (set(optimizers) == set(OPTIMIZERS))
    if (not full_grid) and _paths["simulation_raw_csv"].exists():
        existing = pd.read_csv(_paths["simulation_raw_csv"])
        drop = existing["search_objective"].isin(objectives) & existing["optimizer"].isin(optimizers)
        raw = pd.concat([existing.loc[~drop], raw], ignore_index=True)

    _paths["results_dir"].mkdir(parents=True, exist_ok=True)
    raw.to_csv(_paths["simulation_raw_csv"], index=False)
    print(f"Saved raw per-report results: {_paths['simulation_raw_csv']} ({len(raw)} rows)")

    best = select_incumbents(raw, final_fidelity=int(max_t))
    best.to_csv(_paths["best_found_csv"], index=False)
    n_finished = int((best["selection_rule"] == "finished").sum())
    print(f"Saved best-found-per-run summary: {_paths['best_found_csv']} ({len(best)} rows, "
          f"{n_finished} finished / {len(best) - n_finished} highest-common-fidelity)")


if __name__ == "__main__":
    main()
