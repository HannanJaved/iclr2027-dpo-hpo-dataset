"""Scheduler construction for the informed continuous-surrogate study.

ASHA/BOHB start at ASHA_GRACE_PERIOD (400) with reduction_factor 2 (rungs
400/800/1600).

Syne Tune's RandomSearcher and KernelDensityEstimator (TPE) call
``Domain.sample()`` without ``random_state``, so they draw from the process
global ``np.random``. Under a forked multiprocessing pool every worker
inherits the same RNG, and the first RandomSearch tasks then produce
identical trajectories. Wrap both searchers so ``random_seed`` actually
reaches the sampler.
"""
from __future__ import annotations

import numpy as np
from syne_tune.config_space import Domain
from syne_tune.optimizer.baselines import ASHA, BOHB, BOTorch, CQR
from syne_tune.optimizer.schedulers.searchers.kde import KernelDensityEstimator
from syne_tune.optimizer.schedulers.searchers.random_searcher import RandomSearcher
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import SingleFidelityScheduler
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler

from common import ASHA_GRACE_PERIOD, ASHA_REDUCTION_FACTOR, FIDELITY_ATTR

if ASHA_REDUCTION_FACTOR not in (2, 3):
    raise ValueError(
        f"ASHA_REDUCTION_FACTOR must be 2 or 3, got {ASHA_REDUCTION_FACTOR} "
        "(do not use the rf=100 single-rung schedule)"
    )


def _sample_config(config_space: dict, random_state: np.random.RandomState) -> dict:
    return {
        k: v.sample(random_state=random_state) if isinstance(v, Domain) else v
        for k, v in config_space.items()
    }


class SeededRandomSearcher(RandomSearcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(self.random_seed)

    def suggest(self) -> dict | None:
        new_config = self._next_points_to_evaluate()
        if new_config is None:
            new_config = _sample_config(self.config_space, self.random_state)
        return new_config


class SeededKernelDensityEstimator(KernelDensityEstimator):
    def _get_random_config(self):
        return _sample_config(self.config_space, self.random_state)


# Syne Tune KDE-TPE defaults (top_n_percent=15, num_min_data_points=#HPs) refuse to
# fit until *both* the "good" and "bad" sets have strictly more points than the
# number of hyperparameters. With 2 continuous HPs that means >=20 *completed*
# trials before any model-based suggestion. Under our generous budget most sizes
# only finish ~14-18 trials, so stock TPE is bit-identical to RandomSearch.
# Use a denser good-set quantile so the model can start after 6 completions
# (3 good + 3 bad). Tight (≈2 finished) still cannot leave the random phase.
TPE_TOP_N_PERCENT = 33
TPE_NUM_MIN_DATA_POINTS = 3


def make_scheduler(name: str, config_space: dict, metric: str, max_t: int, min_t: int, seed: int):
    del min_t  # first rung is ASHA_GRACE_PERIOD, not the blackbox minimum
    if name == "RandomSearch":
        return SingleFidelityScheduler(
            config_space=config_space,
            metrics=[metric],
            do_minimize=False,
            searcher=SeededRandomSearcher(config_space=config_space, random_seed=seed),
            random_seed=seed,
        )
    if name == "BOTorch":
        return BOTorch(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "TPE":
        return SingleObjectiveScheduler(
            config_space=config_space,
            metric=metric,
            do_minimize=False,
            searcher=SeededKernelDensityEstimator(
                config_space=config_space,
                random_seed=seed,
                top_n_percent=TPE_TOP_N_PERCENT,
                num_min_data_points=TPE_NUM_MIN_DATA_POINTS,
            ),
            random_seed=seed,
        )
    if name == "CQR":
        return CQR(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "ASHA":
        return ASHA(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=ASHA_GRACE_PERIOD,
            reduction_factor=ASHA_REDUCTION_FACTOR, do_minimize=False, random_seed=seed,
        )
    if name == "BOHB":
        return BOHB(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=ASHA_GRACE_PERIOD,
            reduction_factor=ASHA_REDUCTION_FACTOR, do_minimize=False, random_seed=seed,
        )
    raise ValueError(name)
