"""Coverage experiments on replicate pools.

A pool holds per item paired differences ``D[r, i]`` for ``R`` replicates and ``N``
items, with the sign chosen so that a positive value means arm B is better. The
protocol mirrors the synthetic one in ``simulate.py``.

* A pseudo-experiment draws ``r`` replicates and a random subset of ``n`` items.
  The items are fresh in every pseudo-experiment, so the item population is the
  whole pool of ``N`` items.
* The truth is the mean of ``D`` over the replicates that were not drawn, and over
  all ``N`` items. It is therefore itself estimated, and its noise slightly lowers
  the measured coverage. ``protocol_bias`` measures that effect on synthetic data
  with the same ``R``, ``r`` and ``N`` so the size of the artefact is known.
* The items-only intervals use the first drawn replicate, as a single run study would.
  The corrections use all ``r`` drawn replicates.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from procedure_variance_coverage import stats
from procedure_variance_coverage.simulate import Scenario, draw


@dataclass(frozen=True)
class Pool:
    """Paired differences for one pipeline, metric and contrast."""

    name: str
    D: np.ndarray                 # (R, N), positive means arm B is better
    A: np.ndarray | None = None   # (R, N) binary outcomes of arm A, if the metric is binary
    B: np.ndarray | None = None   # (R, N) binary outcomes of arm B

    @property
    def binary(self) -> bool:
        return self.A is not None

    @property
    def R(self) -> int:
        return self.D.shape[0]

    @property
    def N(self) -> int:
        return self.D.shape[1]


def newcombe_paired_interval(a_out: np.ndarray, b_out: np.ndarray, confidence: float = 0.95):
    """Newcombe's score interval (method 10) for a paired difference of proportions.

    Returns ``stats.Interval`` for ``mean(b_out) - mean(a_out)``. Uses Wilson intervals
    for the two margins and the continuity corrected phi coefficient.
    """
    a_out = np.asarray(a_out).astype(int)
    b_out = np.asarray(b_out).astype(int)
    n = a_out.size
    both = int(((a_out == 1) & (b_out == 1)).sum())
    only_a = int(((a_out == 1) & (b_out == 0)).sum())
    only_b = int(((a_out == 0) & (b_out == 1)).sum())
    neither = n - both - only_a - only_b
    z = float(sps.norm.ppf(0.5 + confidence / 2))

    def wilson(k: int) -> tuple[float, float, float]:
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return p, centre - half, centre + half

    p1, l1, u1 = wilson(both + only_a)    # arm A correct
    p2, l2, u2 = wilson(both + only_b)    # arm B correct
    delta = p2 - p1
    prod = (both + only_a) * (neither + only_b) * (both + only_b) * (neither + only_a)
    if prod == 0:
        phi = 0.0
    else:
        c = both * neither - only_a * only_b
        if c > n / 2:
            c -= n / 2
        elif c >= 0:
            c = 0.0
        phi = c / math.sqrt(prod)
    d_lo, e_lo = p2 - l2, u1 - p1
    d_hi, e_hi = u2 - p2, p1 - l1
    lo = delta - math.sqrt(max(d_lo**2 + e_lo**2 - 2 * phi * d_lo * e_lo, 0))
    hi = delta + math.sqrt(max(d_hi**2 + e_hi**2 - 2 * phi * d_hi * e_hi, 0))
    return stats.Interval(delta, lo, hi)


def variance_summary(D: np.ndarray, n_practice: int) -> dict[str, float]:
    """Variance components from the full pool and the ratio at the practitioner's ``n``."""
    vc = stats.fit_variance_components(D)
    item_var = (vc.s2_item + vc.s2_error) / n_practice
    rho = vc.s2_procedure / item_var if item_var > 0 else math.inf
    return {
        "s2_proc": vc.s2_procedure,
        "s2_item": vc.s2_item,
        "s2_err": vc.s2_error,
        "rho": rho,
        "share": rho / (1 + rho) if math.isfinite(rho) else 1.0,
    }


def rho_interval(
    D: np.ndarray, n_practice: int, *, n_boot: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for rho, resampling replicates only.

    The uncertainty in the procedure variance comes from how few replicates there
    are, and the item variance is estimated from many items. Resampling items as
    well as replicates duplicates both factors and biases the variance
    components, so only the replicates are resampled.
    """
    r = D.shape[0]
    vals = []
    for _ in range(n_boot):
        ri = rng.integers(0, r, size=r)
        vals.append(variance_summary(D[ri], n_practice)["rho"])
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return float(lo), float(hi)


def _methods(
    pool: Pool, r: int, n_boot_items: int, n_boot_multi: int, rng, include_items: bool = True
) -> dict[str, Callable]:
    """Interval constructors, each taking (replicate indices, item indices)."""
    D = pool.D

    def items_t(rep, items):
        return stats.eval_only_t_interval(D[rep[0], items])

    def items_boot(rep, items):
        return stats.eval_only_bootstrap_interval(
            D[rep[0], items], n_resamples=n_boot_items, rng=rng
        )

    out: dict[str, Callable] = {}
    if include_items:
        out["items_t"] = items_t
        out["items_boot"] = items_boot
        if pool.binary:
            out["items_newcombe"] = lambda rep, items: newcombe_paired_interval(
                pool.A[rep[0], items], pool.B[rep[0], items]
            )
    if r >= 2:
        out["crossed"] = lambda rep, items: stats.crossed_interval(D[np.ix_(rep, items)])
        out["multiboot"] = lambda rep, items: stats.multi_bootstrap_interval(
            D[np.ix_(rep, items)], n_resamples=n_boot_multi, rng=rng
        )
    return out


def coverage_experiment(
    pool: Pool,
    *,
    r: int,
    n_items: int,
    n_experiments: int,
    rng: np.random.Generator,
    n_boot_items: int = 300,
    n_boot_multi: int = 200,
    include_items: bool = True,
    only: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """Coverage and mean half width for every interval, at ``r`` drawn replicates.

    ``only`` restricts the run to the named intervals, which keeps sweeps cheap.
    """
    if r >= pool.R:
        raise ValueError("need at least one replicate left over to define the truth")
    methods = _methods(pool, r, n_boot_items, n_boot_multi, rng, include_items)
    if only is not None:
        methods = {k: v for k, v in methods.items() if k in only}
    hits = {m: 0 for m in methods}
    widths = {m: 0.0 for m in methods}
    for _ in range(n_experiments):
        perm = rng.permutation(pool.R)
        rep, rest = perm[:r], perm[r:]
        truth = float(pool.D[rest].mean())
        items = rng.choice(pool.N, size=n_items, replace=False)
        for name, fn in methods.items():
            iv = fn(rep, items)
            hits[name] += int(iv.covers(truth))
            widths[name] += iv.half_width
    out = {}
    for name in methods:
        rate = hits[name] / n_experiments
        out[name] = {
            "coverage": rate,
            "mc_se": math.sqrt(rate * (1 - rate) / n_experiments),
            "half_width": widths[name] / n_experiments,
        }
    return out


def protocol_check(
    scn: Scenario, *, R: int, r: int, N: int, n_items: int, n_pools: int, n_per_pool: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Size of the artefact from estimating the truth out of a finite pool.

    Many independent synthetic pools with the real pool's ``R`` and ``N`` are drawn.
    For every pseudo-experiment the same items interval is judged against two
    truths, the pool based one used on real data and the known ``delta``. Their
    coverage difference is the artefact, so real results can be read with it in mind.
    """
    big = Scenario(scn.delta, scn.s_procedure, scn.s_item, scn.s_error, N)
    pool_hits = known_hits = total = 0
    for _ in range(n_pools):
        D = draw(big, R, rng)
        for _ in range(n_per_pool):
            perm = rng.permutation(R)
            rep, rest = perm[:r], perm[r:]
            truth = float(D[rest].mean())
            items = rng.choice(N, size=n_items, replace=False)
            iv = stats.eval_only_t_interval(D[rep[0], items])
            pool_hits += int(iv.covers(truth))
            known_hits += int(iv.covers(scn.delta))
            total += 1
    return {"pool_truth_coverage": pool_hits / total, "known_truth_coverage": known_hits / total}


def budget_check(
    pool: Pool,
    *,
    r_pilot: int,
    n_items: int,
    n_experiments: int,
    rng: np.random.Generator,
    power: float = 0.8,
    delta_scale: float = 1.0,
) -> dict[str, float]:
    """Compare replicates needed according to a pilot with the full pool value.

    The target difference is ``delta_scale`` times the pool's own procedure level
    mean. At 1.0 the rule answers how many replicates certify the effect that was
    actually observed. A large effect needs only the minimum of two replicates, so
    smaller scales test the rule where it has something to say.
    The pool value uses all replicates and a random subset of ``n_items`` items,
    the same evaluation size the pilot uses.
    """
    delta = abs(float(pool.D.mean())) * delta_scale
    if delta == 0:
        raise ValueError("pool mean difference is exactly zero, the rule is undefined")
    # The reference value uses the components of the whole pool, which are far more
    # precise than any subset, with the item terms scaled to the practitioner's n.
    pool_vc = dataclasses.replace(stats.fit_variance_components(pool.D), n_items=n_items)
    pool_need = stats.replicates_needed(pool_vc, delta, power=power)
    ratios, undefined = [], 0
    for _ in range(n_experiments):
        rep = rng.choice(pool.R, size=r_pilot, replace=False)
        sub = rng.choice(pool.N, size=min(n_items, pool.N), replace=False)
        need = stats.replicates_needed(
            stats.fit_variance_components(pool.D[np.ix_(rep, sub)]), delta, power=power
        )
        if need is None or pool_need is None:
            undefined += 1
        else:
            ratios.append(need / pool_need)
    arr = np.array(ratios) if ratios else np.array([math.nan])
    return {
        "delta": delta,
        "pool_need": float(pool_need) if pool_need is not None else math.nan,
        "median_ratio": float(np.median(arr)) if ratios else math.nan,
        "within_1.5": float(np.mean((arr >= 1 / 1.5) & (arr <= 1.5))) if ratios else math.nan,
        "undefined_share": undefined / n_experiments,
    }
