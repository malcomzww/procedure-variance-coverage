"""Synthetic data with known truth, and a coverage simulator.

The generator is the model the statistics assume,
``D[r, i] = delta + a_r + b_i + e_ri``. Because the truth ``delta`` is known and
fresh items are drawn for every pseudo-experiment, the true coverage of any
interval can be measured, and for the eval-only interval it can also be written
down in closed form. That closed form is the check that the simulator and the
intervals are implemented correctly, before they are pointed at real runs.

Nothing here calls a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from procedure_variance_coverage import stats


@dataclass(frozen=True)
class Scenario:
    """Variance structure of a two arm comparison."""

    delta: float
    s_procedure: float
    s_item: float
    s_error: float
    n_items: int

    @property
    def rho(self) -> float:
        """Procedure variance over the item sampling variance of one run."""
        item_var = (self.s_item**2 + self.s_error**2) / self.n_items
        return self.s_procedure**2 / item_var

    @classmethod
    def with_rho(
        cls, rho: float, *, n_items: int, s_item: float = 1.0, s_error: float = 1.0, delta: float
    ) -> Scenario:
        """Scenario whose procedure variance gives the requested ``rho``."""
        item_var = (s_item**2 + s_error**2) / n_items
        return cls(delta, math.sqrt(rho * item_var), s_item, s_error, n_items)


def draw(scn: Scenario, n_replicates: int, rng: np.random.Generator) -> np.ndarray:
    """One pseudo-experiment, an ``(R, n)`` matrix of paired differences.

    The items are fresh for every call. The truth is therefore the procedure
    level difference over the item population, ``scn.delta``.
    """
    a = rng.normal(0.0, scn.s_procedure, size=(n_replicates, 1))
    b = rng.normal(0.0, scn.s_item, size=(1, scn.n_items))
    e = rng.normal(0.0, scn.s_error, size=(n_replicates, scn.n_items))
    return scn.delta + a + b + e


@dataclass(frozen=True)
class Coverage:
    rate: float
    n_experiments: int

    @property
    def mc_se(self) -> float:
        """Monte Carlo standard error of the coverage rate."""
        return math.sqrt(self.rate * (1 - self.rate) / self.n_experiments)


def measure_coverage(
    scn: Scenario,
    *,
    n_replicates: int,
    interval: Callable[[np.ndarray], stats.Interval],
    n_experiments: int,
    rng: np.random.Generator,
) -> Coverage:
    """Fraction of pseudo-experiments whose interval covers ``scn.delta``."""
    hits = 0
    for _ in range(n_experiments):
        D = draw(scn, n_replicates, rng)
        if interval(D).covers(scn.delta):
            hits += 1
    return Coverage(hits / n_experiments, n_experiments)


def analytic_eval_only_coverage(rho: float, confidence: float = 0.95) -> float:
    """Large item count closed form for the single run eval-only interval.

    The centre of the interval has error variance ``s2_proc + item_var`` and the
    interval half width is ``z * sqrt(item_var)``, so coverage is
    ``2 * Phi(z / sqrt(1 + rho)) - 1``.
    """
    z = float(sps.norm.ppf(0.5 + confidence / 2))
    return 2 * float(sps.norm.cdf(z / math.sqrt(1 + rho))) - 1
