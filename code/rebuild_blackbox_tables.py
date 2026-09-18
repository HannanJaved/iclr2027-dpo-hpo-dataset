#!/usr/bin/env python3
"""Rebuild blackbox tables (incl. Z-Macro) from data/*_raw.csv. Needs pandas+numpy only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_blackbox import add_composite_zscores, drop_fidelity_incomplete_configs, fill_gaps, usable_benchmarks
from common import COMPOSITE_OBJECTIVES, SIZE_KEYS_BY_FAMILY

ROOT = HERE.parent
DATA = ROOT / "data"


def main() -> None:
    frames = []
    for family, sizes in SIZE_KEYS_BY_FAMILY.items():
        for size in sizes:
            raw_path = DATA / f"{family}_{size}_raw.csv"
            raw = pd.read_csv(raw_path)
            key = f"{family}-{size}"
            df = drop_fidelity_incomplete_configs(raw.drop(columns=["family", "size"]), key)
            benches = usable_benchmarks(df, key)
            df = fill_gaps(df, benches, key)
            df = add_composite_zscores(df, benches)
            df.insert(0, "family", family)
            df.insert(1, "size", size)
            for col in COMPOSITE_OBJECTIVES:
                if col not in df.columns:
                    df[col] = pd.NA
            out = DATA / f"{family}_{size}_blackbox.csv"
            df.to_csv(out, index=False)
            frames.append(df)
            print(f"wrote {out} ({len(df)} rows)")
    pd.concat(frames, ignore_index=True).to_csv(DATA / "all_checkpoints_blackbox.csv", index=False)


if __name__ == "__main__":
    main()
