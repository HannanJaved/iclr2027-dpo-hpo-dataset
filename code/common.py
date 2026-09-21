"""Protocol constants for the informed continuous-surrogate HPO study.

Paths resolve relative to this package: grid tables live in ../data,
writable outputs go to ../work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ANON_ROOT = Path(__file__).resolve().parent.parent
HERE = ANON_ROOT / "work"
HERE.mkdir(parents=True, exist_ok=True)
SOURCE_HPO = ANON_ROOT

SIZE_KEYS = ["0.6b", "1.7b", "4b", "8b", "14b"]
SIZE_KEYS_BY_FAMILY = {
    "qwen3": SIZE_KEYS,
    "llama": ["1b", "3b", "8b"],
}
FAMILIES = list(SIZE_KEYS_BY_FAMILY)
DEFAULT_FAMILY = "qwen3"
FAMILY_LABELS = {"qwen3": "Qwen3", "llama": "Llama"}

BENCHMARKS = [
    "ARC-C", "GPQA", "GSM8K", "HellaSwag", "PIQA", "TruthfulQA", "IFEval",
    "Arena-Hard", "MT-Bench", "AlpacaEval", "ELO",
]
COMPOSITE_OBJECTIVES = ["Z-Static", "Z-Dynamic", "Z-All", "Z-Macro"]
TIME_OBJECTIVE = "elapsed_time_sec"
ALL_OBJECTIVES = BENCHMARKS + COMPOSITE_OBJECTIVES + [TIME_OBJECTIVE]
FIDELITY_ATTR = "dpo_step"
SEARCH_OBJECTIVES = ["Z-Dynamic", "Z-All", "Z-Macro"]
OPTIMIZERS = ["RandomSearch", "TPE", "CQR", "ASHA", "BOHB"]
N_SEEDS = 25
N_WORKERS = 2
ASHA_GRACE_PERIOD = 400
ASHA_REDUCTION_FACTOR = 2
BUDGET_REGIMES = {"generous": 8.0, "tight": 1.0}
DEFAULT_BUDGET_TAG = "generous"


def slug(size_key: str) -> str:
    return size_key.replace(".", "p")


@dataclass(frozen=True)
class Paths:
    size_key: str
    family: str
    budget_tag: str
    size_dir: Path
    grid_csv: Path
    blackbox_dir: Path
    blackbox_key: str
    config_map_csv: Path
    results_dir: Path
    simulation_raw_csv: Path
    best_found_csv: Path
    figures_dir: Path


def get_paths(size_key: str, budget_tag: str = DEFAULT_BUDGET_TAG, family: str = DEFAULT_FAMILY) -> Paths:
    if family not in SIZE_KEYS_BY_FAMILY:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if size_key not in SIZE_KEYS_BY_FAMILY[family]:
        raise ValueError(f"size_key for family {family!r} must be one of {SIZE_KEYS_BY_FAMILY[family]}, got {size_key!r}")
    if budget_tag not in BUDGET_REGIMES:
        raise ValueError(f"budget_tag must be one of {list(BUDGET_REGIMES)}, got {budget_tag!r}")
    sl = slug(size_key)
    size_dir = HERE / f"{family}_{sl}_dpo"
    results_dirname = "results" if budget_tag == DEFAULT_BUDGET_TAG else f"results_{budget_tag}"
    results_dir = size_dir / results_dirname
    return Paths(
        size_key=size_key,
        family=family,
        budget_tag=budget_tag,
        size_dir=size_dir,
        grid_csv=ANON_ROOT / "data" / f"{family}_{size_key}_raw.csv",
        blackbox_dir=size_dir / "blackbox" / f"{family}-{size_key}-dpo-ao",
        blackbox_key=f"{family}-{size_key}-dpo-ao",
        config_map_csv=size_dir / "blackbox" / f"{family}-{size_key}-dpo-ao-config-map.csv",
        results_dir=results_dir,
        simulation_raw_csv=results_dir / "simulation_raw.csv",
        best_found_csv=results_dir / "best_found.csv",
        figures_dir=results_dir / "figures",
    )
