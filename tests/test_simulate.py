"""Check the simulator and the intervals against closed forms and known truth.

These tests use fixed seeds and tolerances of several Monte Carlo standard errors.
A failure here means the statistics are wrong, not that a model behaved oddly.
"""

import numpy as np
import pytest

from procedure_variance_coverage import simulate, stats
from procedure_variance_coverage.simulate import Scenario


def _eval_only_first_row(D):
    return stats.eval_only_t_interval(D[0])


@pytest.mark.parametrize("rho", [0.0, 1.0, 4.0])
def test_eval_only_coverage_matches_the_closed_form(rho):
    scn = Scenario.with_rho(rho, n_items=400, delta=0.1)
    cov = simulate.measure_coverage(
        scn,
        n_replicates=1,
        interval=_eval_only_first_row,
        n_experiments=4000,
        rng=np.random.default_rng(11),
    )
    expected = simulate.analytic_eval_only_coverage(rho)
    assert abs(cov.rate - expected) < 4 * cov.mc_se + 0.005


def test_eval_only_coverage_falls_as_procedure_variance_grows():
    rates = []
    for rho in (0.0, 1.0, 4.0, 16.0):
        scn = Scenario.with_rho(rho, n_items=300, delta=0.0)
        rates.append(
            simulate.measure_coverage(
                scn,
                n_replicates=1,
                interval=_eval_only_first_row,
                n_experiments=1500,
                rng=np.random.default_rng(12),
            ).rate
        )
    assert rates == sorted(rates, reverse=True)
    assert rates[0] > 0.92
    assert rates[-1] < 0.45


def test_crossed_interval_reaches_nominal_coverage_with_five_replicates():
    scn = Scenario.with_rho(4.0, n_items=200, delta=0.05)
    cov = simulate.measure_coverage(
        scn,
        n_replicates=5,
        interval=stats.crossed_interval,
        n_experiments=3000,
        rng=np.random.default_rng(13),
    )
    assert 0.93 <= cov.rate <= 0.97


def test_multi_bootstrap_undercovers_more_than_crossed_at_three_replicates():
    """Measured, not assumed. At R=3 and rho=4 the gap is about 4 points.

    An earlier version of this test asserted the same at R=5 and rho=4 and failed
    on one run, because the gap there is small and within noise at that size. The
    gap is clear at R=3 and grows with rho, so the test pins that case.
    """
    scn = Scenario.with_rho(4.0, n_items=150, delta=0.0)
    rng = np.random.default_rng(14)

    def interval(D):
        return stats.multi_bootstrap_interval(D, n_resamples=300, rng=rng)

    cov = simulate.measure_coverage(
        scn, n_replicates=3, interval=interval, n_experiments=1500, rng=np.random.default_rng(15)
    )
    crossed = simulate.measure_coverage(
        scn,
        n_replicates=3,
        interval=stats.crossed_interval,
        n_experiments=1500,
        rng=np.random.default_rng(15),
    )
    assert cov.rate < crossed.rate - 0.02


def test_scenario_with_rho_round_trips():
    scn = Scenario.with_rho(3.0, n_items=250, delta=0.0)
    assert scn.rho == pytest.approx(3.0)


def test_draw_has_the_requested_shape_and_known_truth():
    scn = Scenario.with_rho(2.0, n_items=64, delta=0.7)
    D = simulate.draw(scn, 6, np.random.default_rng(16))
    assert D.shape == (6, 64)
    big = simulate.draw(
        Scenario(0.7, 0.01, 0.01, 0.01, 20000), 3, np.random.default_rng(17)
    )
    assert big.mean() == pytest.approx(0.7, abs=0.01)
