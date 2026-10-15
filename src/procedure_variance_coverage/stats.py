"""Intervals and variance components for comparing two pipelines.

Setup. Two arms A and B are scored on the same ``n`` evaluation items. The recipe
is run ``R`` times per arm with fresh randomness (training subset, data order,
calibration draw). The input is the matrix ``D`` of shape ``(R, n)`` holding the
per-item paired difference ``score_B - score_A`` for each replicate.

Two estimands are in play.

* Artifact level. The difference between these particular trained models, over
  the population of items. An interval built from the items of one replicate
  targets this, and only this.
* Procedure level. The expected difference when the recipe is re-run with fresh
  randomness. This is what a sentence such as "method B beats method A" usually
  claims.

Treating items as the only source of randomness when the claim is procedure level
is pseudoreplication. The functions here measure how wrong that is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Interval:
    """A confidence interval for a mean difference."""

    estimate: float
    lo: float
    hi: float
    df: float | None = None

    def covers(self, truth: float) -> bool:
        return self.lo <= truth <= self.hi

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2


def _check_vector(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("need a 1-D vector of at least two item differences")
    return d


def eval_only_t_interval(d: np.ndarray, confidence: float = 0.95) -> Interval:
    """Paired t interval over the items of ONE replicate.

    This is the interval a single-run study reports. It treats the items as the
    only source of randomness.
    """
    d = _check_vector(d)
    n = d.size
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    q = float(sps.t.ppf(0.5 + confidence / 2, n - 1))
    return Interval(mean, mean - q * se, mean + q * se, df=n - 1)


def eval_only_bootstrap_interval(
    d: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    rng: np.random.Generator,
) -> Interval:
    """Percentile bootstrap over the items of ONE replicate, vectorised."""
    d = _check_vector(d)
    n = d.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = d[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return Interval(float(d.mean()), float(lo), float(hi))


@dataclass(frozen=True)
class VarianceComponents:
    """Crossed random effects fit for ``D[r, i] = delta + a_r + b_i + e_ri``.

    ``a_r`` is procedure randomness, ``b_i`` is the item effect and ``e_ri`` is
    the residual. Components are the usual expected-mean-squares estimates, with
    negative values clipped to zero.
    """

    n_replicates: int
    n_items: int
    ms_replicate: float
    ms_item: float
    ms_error: float
    s2_procedure: float
    s2_item: float
    s2_error: float

    @property
    def rho(self) -> float:
        """Procedure variance relative to the item sampling variance of one run.

        An eval-only interval for one replicate has squared standard error
        ``(s2_item + s2_error) / n``. ``rho`` is the ratio of ``s2_procedure`` to
        that. At ``rho = 0`` the eval-only interval is calibrated for the
        procedure level estimand.
        """
        item_var = (self.s2_item + self.s2_error) / self.n_items
        return self.s2_procedure / item_var if item_var > 0 else math.inf


def fit_variance_components(D: np.ndarray) -> VarianceComponents:
    """Two way ANOVA without cell replication on an ``(R, n)`` matrix."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] < 2 or D.shape[1] < 2:
        raise ValueError("need an (R, n) matrix with R >= 2 and n >= 2")
    r, n = D.shape
    grand = D.mean()
    row = D.mean(axis=1)
    col = D.mean(axis=0)
    ss_r = n * float(((row - grand) ** 2).sum())
    ss_i = r * float(((col - grand) ** 2).sum())
    resid = D - row[:, None] - col[None, :] + grand
    ss_e = float((resid**2).sum())
    ms_r = ss_r / (r - 1)
    ms_i = ss_i / (n - 1)
    ms_e = ss_e / ((r - 1) * (n - 1))
    return VarianceComponents(
        n_replicates=r,
        n_items=n,
        ms_replicate=ms_r,
        ms_item=ms_i,
        ms_error=ms_e,
        s2_procedure=max(0.0, (ms_r - ms_e) / n),
        s2_item=max(0.0, (ms_i - ms_e) / r),
        s2_error=ms_e,
    )


def crossed_interval(D: np.ndarray, confidence: float = 0.95) -> Interval:
    """Interval for the grand mean from the crossed fit, Satterthwaite df.

    ``Var(grand mean) = (MS_R + MS_I - MS_E) / (R n)``. The variance is floored at
    ``MS_E / (R n)`` so a noisy fit cannot return a degenerate interval.
    """
    vc = fit_variance_components(D)
    r, n = vc.n_replicates, vc.n_items
    scale = r * n
    var = (vc.ms_replicate + vc.ms_item - vc.ms_error) / scale
    var = max(var, vc.ms_error / scale)
    denom = (
        (vc.ms_replicate / scale) ** 2 / (r - 1)
        + (vc.ms_item / scale) ** 2 / (n - 1)
        + (vc.ms_error / scale) ** 2 / ((r - 1) * (n - 1))
    )
    df = var**2 / denom
    mean = float(np.asarray(D, dtype=float).mean())
    half = float(sps.t.ppf(0.5 + confidence / 2, df)) * math.sqrt(var)
    return Interval(mean, mean - half, mean + half, df=df)


def multi_bootstrap_interval(
    D: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    rng: np.random.Generator,
) -> Interval:
    """Two level percentile bootstrap, resampling replicates and items together.

    The Multi-Bootstrap idea from Sellam et al. (2022), vectorised. In the
    synthetic checks (``scripts/generate_results.py``) it undercovers more than
    the crossed interval at three and at five replicates, and the gap is largest
    when procedure variance is large.
    """
    D = np.asarray(D, dtype=float)
    r, n = D.shape
    ri = rng.integers(0, r, size=(n_resamples, r))
    ii = rng.integers(0, n, size=(n_resamples, n))
    means = D[ri[:, :, None], ii[:, None, :]].mean(axis=(1, 2))
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return Interval(float(D.mean()), float(lo), float(hi))


def replicates_needed(
    vc: VarianceComponents,
    delta: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int | None:
    """Replicates ``R`` needed to detect a procedure level difference ``delta``.

    Uses the fitted components and the evaluation size they were estimated at.
    ``Var(mean) = s2_proc / R + s2_item / n + s2_err / (R n)`` has a floor of
    ``s2_item / n`` that more replicates cannot remove. Returns ``None`` when the
    target standard error is below that floor, which means more items are needed,
    not more replicates.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    z = float(sps.norm.ppf(1 - alpha / 2)) + float(sps.norm.ppf(power))
    target_var = (delta / z) ** 2
    floor = vc.s2_item / vc.n_items
    if target_var <= floor:
        return None
    needed = (vc.s2_procedure + vc.s2_error / vc.n_items) / (target_var - floor)
    return max(2, math.ceil(needed))
