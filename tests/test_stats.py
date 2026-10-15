import math

import numpy as np
import pytest

from procedure_variance_coverage import stats
from procedure_variance_coverage.simulate import Scenario, draw


def test_interval_covers_and_half_width():
    iv = stats.Interval(1.0, 0.5, 1.5)
    assert iv.covers(1.0)
    assert not iv.covers(1.6)
    assert iv.half_width == pytest.approx(0.5)


def test_eval_only_t_interval_matches_textbook_formula():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    iv = stats.eval_only_t_interval(d)
    se = d.std(ddof=1) / math.sqrt(5)
    assert iv.estimate == pytest.approx(3.0)
    assert iv.half_width == pytest.approx(2.7764451 * se, rel=1e-6)
    assert iv.df == 4


def test_vectors_that_are_too_short_are_rejected():
    with pytest.raises(ValueError):
        stats.eval_only_t_interval(np.array([1.0]))
    with pytest.raises(ValueError):
        stats.fit_variance_components(np.ones((1, 5)))


def test_variance_components_are_recovered():
    scn = Scenario(delta=0.2, s_procedure=0.5, s_item=1.0, s_error=0.7, n_items=2000)
    rng = np.random.default_rng(1)
    D = draw(scn, 400, rng)
    vc = stats.fit_variance_components(D)
    assert vc.s2_procedure == pytest.approx(0.25, rel=0.2)
    assert vc.s2_item == pytest.approx(1.0, rel=0.15)
    assert vc.s2_error == pytest.approx(0.49, rel=0.05)


def test_rho_is_zero_when_there_is_no_procedure_variance():
    D = np.random.default_rng(2).normal(size=(6, 300))
    vc = stats.fit_variance_components(D - D.mean(axis=1, keepdims=True))
    assert vc.rho == pytest.approx(0.0, abs=1e-9)


def test_crossed_interval_is_wider_than_eval_only_when_procedure_variance_exists():
    scn = Scenario.with_rho(4.0, n_items=300, delta=0.0)
    D = draw(scn, 5, np.random.default_rng(3))
    crossed = stats.crossed_interval(D)
    single = stats.eval_only_t_interval(D[0])
    assert crossed.half_width > single.half_width


def test_bootstrap_intervals_are_reproducible_for_a_seed():
    D = np.random.default_rng(4).normal(size=(4, 120))
    a = stats.multi_bootstrap_interval(D, n_resamples=200, rng=np.random.default_rng(9))
    b = stats.multi_bootstrap_interval(D, n_resamples=200, rng=np.random.default_rng(9))
    assert (a.lo, a.hi) == (b.lo, b.hi)
    c = stats.eval_only_bootstrap_interval(D[0], n_resamples=200, rng=np.random.default_rng(9))
    assert c.lo < c.estimate < c.hi


def test_replicates_needed_falls_as_the_target_difference_grows():
    scn = Scenario.with_rho(4.0, n_items=300, delta=0.0)
    vc = stats.fit_variance_components(draw(scn, 30, np.random.default_rng(5)))
    small = stats.replicates_needed(vc, delta=0.3)
    large = stats.replicates_needed(vc, delta=0.6)
    assert small is None or large is None or small >= large
    assert large is not None and large >= 2


def test_replicates_needed_is_none_when_the_item_floor_blocks_it():
    vc = stats.VarianceComponents(
        n_replicates=10,
        n_items=50,
        ms_replicate=1.0,
        ms_item=60.0,
        ms_error=1.0,
        s2_procedure=0.0,
        s2_item=6.0,
        s2_error=1.0,
    )
    # floor = 6 / 50 = 0.12, so a standard error of 0.1 cannot be reached
    assert stats.replicates_needed(vc, delta=0.28) is None
    with pytest.raises(ValueError):
        stats.replicates_needed(vc, delta=0.0)
