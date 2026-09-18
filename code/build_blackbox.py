#!/usr/bin/env python3
"""Grid-cleaning helpers reused by build_blackbox_surrogate.py.

This informed study does not run discrete cfg_id HPO. The functions below
(usable_benchmarks / fill_gaps / add_composite_zscores /
drop_fidelity_incomplete_configs) are the same pipeline the GP surrogate is
fit on. The BlackboxTabular builder is kept so the module stays importable.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import syne_tune.config_space as sp
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular, serialize

from common import BENCHMARKS, COMPOSITE_OBJECTIVES, DEFAULT_FAMILY, FAMILIES, FIDELITY_ATTR, TIME_OBJECTIVE, get_paths

STATIC_BENCHMARKS = ["ARC-C", "GPQA", "GSM8K", "HellaSwag", "PIQA", "TruthfulQA", "IFEval"]
DYNAMIC_BENCHMARKS = ["Arena-Hard", "MT-Bench", "AlpacaEval", "ELO"]

DROP_BENCHMARK_THRESHOLD = 0.2


def usable_benchmarks(df: pd.DataFrame, size_key: str, group_cols: list[str] | None = None) -> list[str]:
    if group_cols is None:
        group_cols = ["lr", "beta"]
    n_configs = df[group_cols].drop_duplicates().shape[0]
    fully_missing_frac = {
        b: df.groupby(group_cols)[b].apply(lambda s: s.isna().all()).mean()
        for b in BENCHMARKS
    }
    dropped = [b for b, frac in fully_missing_frac.items() if frac > DROP_BENCHMARK_THRESHOLD]
    for b in dropped:
        print(f"[WARN] {size_key}: dropping benchmark {b!r} entirely "
              f"(missing for {fully_missing_frac[b]:.0%} of {n_configs} configs)")
    return [b for b in BENCHMARKS if b not in dropped]


def fill_gaps(df: pd.DataFrame, benchmarks: list[str], size_key: str,
              group_cols: list[str] | None = None, peer_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["lr", "beta"]
    if peer_cols is None:
        peer_cols = []
    df = df.sort_values(group_cols + ["dpo_step"]).copy()
    df[benchmarks] = df.groupby(group_cols)[benchmarks].transform(
        lambda col: col.ffill().bfill()
    )
    still_missing = df.groupby(group_cols)[benchmarks].apply(lambda g: g.isna().any())
    bad_configs = still_missing[still_missing.any(axis=1)]
    for key, row in bad_configs.iterrows():
        key_tuple = key if isinstance(key, tuple) else (key,)
        missing = row[row].index.tolist()
        key_desc = ", ".join(f"{c}={v}" for c, v in zip(group_cols, key_tuple))
        print(f"[WARN] {size_key}: config {key_desc} has no data at ANY fidelity for "
              f"{missing} -> imputing from the per-step mean of {'peer' if peer_cols else 'other'} configs")
        mask = np.logical_and.reduce([df[c] == v for c, v in zip(group_cols, key_tuple)])
        for col in missing:
            peer_mean = df.groupby(peer_cols + ["dpo_step"])[col].transform("mean") if peer_cols \
                else df.groupby("dpo_step")[col].transform("mean")
            df.loc[mask, col] = df.loc[mask, col].fillna(peer_mean[mask])
    assert not df[benchmarks].isna().any().any(), "unfillable gaps remain"
    return df


def add_composite_zscores(df: pd.DataFrame, benchmarks: list[str] | None = None) -> pd.DataFrame:
    if benchmarks is None:
        benchmarks = BENCHMARKS
    static = [b for b in STATIC_BENCHMARKS if b in benchmarks]
    dynamic = [b for b in DYNAMIC_BENCHMARKS if b in benchmarks]
    df = df.copy()
    z = df.groupby("dpo_step")[benchmarks].transform(
        lambda col: (col - col.mean()) / col.std(ddof=0) if col.std(ddof=0) > 1e-12 else 0.0
    )
    df["Z-Static"] = z[static].mean(axis=1) if static else np.nan
    df["Z-Dynamic"] = z[dynamic].mean(axis=1) if dynamic else np.nan
    df["Z-All"] = z[benchmarks].mean(axis=1)
    df["Z-Macro"] = 0.5 * (df["Z-Dynamic"] + df["Z-Static"])
    return df


def drop_fidelity_incomplete_configs(df: pd.DataFrame, size_key: str) -> pd.DataFrame:
    n_rows = df.groupby(["lr", "beta"]).size()
    max_rows = n_rows.max()
    incomplete = n_rows[n_rows < max_rows]
    if len(incomplete):
        for (lr, beta), n in incomplete.items():
            print(f"[WARN] {size_key}: excluding lr={lr:.0e}, beta={beta} from the SEARCHABLE HPO grid -- "
                  f"only {n}/{max_rows} fidelities evaluated.")
        keep = pd.MultiIndex.from_frame(df[["lr", "beta"]]).isin(n_rows[n_rows == max_rows].index)
        df = df[keep]
    return df


def build(size_key: str, family: str = DEFAULT_FAMILY) -> tuple[BlackboxTabular, pd.DataFrame]:
    paths = get_paths(size_key, family=family)
    df = pd.read_csv(paths.grid_csv)
    df = drop_fidelity_incomplete_configs(df, size_key)
    benchmarks = usable_benchmarks(df, size_key)
    df = fill_gaps(df, benchmarks, size_key)
    df = add_composite_zscores(df, benchmarks)
    objectives = benchmarks + COMPOSITE_OBJECTIVES + [TIME_OBJECTIVE]

    fidelity_values = np.sort(df["dpo_step"].unique())
    configs = df[["lr", "beta"]].drop_duplicates().sort_values(["lr", "beta"]).reset_index(drop=True)
    configs["cfg_id"] = configs.index

    n_evals, n_seeds, n_fidelities, n_objectives = (
        len(configs), 1, len(fidelity_values), len(objectives),
    )
    objectives_evaluations = np.full((n_evals, n_seeds, n_fidelities, n_objectives), np.nan)

    fidelity_index = {step: i for i, step in enumerate(fidelity_values)}
    config_index = {(row.lr, row.beta): int(row.cfg_id) for row in configs.itertuples()}

    for _, row in df.iterrows():
        ci = config_index[(row["lr"], row["beta"])]
        fi = fidelity_index[row["dpo_step"]]
        objectives_evaluations[ci, 0, fi, :] = [row[obj] for obj in objectives]

    assert not np.isnan(objectives_evaluations).any(), "every (config, fidelity) cell must be filled"
    assert len(configs) == len(configs.drop_duplicates(["lr", "beta"])), "duplicate (lr, beta) rows in the grid"

    configs["_const"] = 0
    configuration_space = {"cfg_id": sp.choice(configs["cfg_id"].tolist()), "_const": sp.choice([0])}
    fidelity_space = {FIDELITY_ATTR: sp.randint(0, int(fidelity_values.max()))}

    blackbox = BlackboxTabular(
        hyperparameters=configs[["cfg_id", "_const"]],
        configuration_space=configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objectives_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objectives,
    )
    return blackbox, configs[["cfg_id", "lr", "beta"]]


def sanity_check(blackbox: BlackboxTabular, config_map: pd.DataFrame) -> None:
    print(blackbox)
    cfg = blackbox.hyperparameters.iloc[0].to_dict()
    lr, beta = config_map.loc[config_map["cfg_id"] == cfg["cfg_id"], ["lr", "beta"]].iloc[0]
    result = blackbox.objective_function(cfg, fidelity={FIDELITY_ATTR: int(blackbox.fidelity_values[0])})
    print(f"Spot check {cfg} (lr={lr}, beta={beta}) @ step {blackbox.fidelity_values[0]}: {result}")
    result_final = blackbox.objective_function(cfg, fidelity={FIDELITY_ATTR: int(blackbox.fidelity_values[-1])})
    print(f"Spot check {cfg} @ step {blackbox.fidelity_values[-1]}: {result_final}")
    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    times = blackbox.objectives_evaluations[0, 0, :, time_idx]
    assert np.all(np.diff(times) > 0), "elapsed_time_sec must be strictly increasing per config"
    print("elapsed_time_sec is monotonically increasing: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=FAMILIES)
    args = parser.parse_args()

    paths = get_paths(args.size, family=args.family)
    blackbox, config_map = build(args.size, args.family)
    sanity_check(blackbox, config_map)

    paths.blackbox_dir.parent.mkdir(parents=True, exist_ok=True)
    serialize(
        {paths.blackbox_key: blackbox},
        path=str(paths.blackbox_dir),
        metadata={"size_key": args.size, "family": args.family, "source": f"ICLR27 {args.family} DPO-AO sweep"},
    )
    config_map.to_csv(paths.config_map_csv, index=False)
    print(f"Serialized blackbox to: {paths.blackbox_dir}")
    print(f"Saved cfg_id -> (lr, beta) map to: {paths.config_map_csv}")


if __name__ == "__main__":
    main()
