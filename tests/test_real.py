"""Check the replicate pool engine on synthetic pools where the answer is known."""

import math

import numpy as np
import pytest

from procedure_variance_coverage import real, simulate
from procedure_variance_coverage.simulate import Scenario


def _binary_pair(rng, n, p_a, p_b, corr):
    """Paired binary outcomes with a shared latent difficulty per item."""
    shared = rng.random(n)
    own_a, own_b = rng.random(n), rng.random(n)
    ua = np.where(rng.random(n) < corr, shared, own_a)
    ub = np.where(rng.random(n) < corr, shared, own_b)
    return (ua < p_a).astype(int), (ub < p_b).astype(int)


def test_newcombe_interval_covers_the_artifact_difference_at_nominal_rate():
    rng = np.random.default_rng(1)
    p_a, p_b, n, hits, reps = 0.50, 0.56, 200, 0, 2500
    for _ in range(reps):
        a, b = _binary_pair(rng, n, p_a, p_b, corr=0.6)
        hits += real.newcombe_paired_interval(a, b).covers(p_b - p_a)
    assert 0.935 <= hits / reps <= 0.97


def test_newcombe_interval_handles_degenerate_tables():
    same = np.ones(50, dtype=int)
    iv = real.newcombe_paired_interval(same, same)
    assert iv.estimate == 0.0 and iv.lo <= 0.0 <= iv.hi


def test_variance_summary_recovers_rho_at_the_practice_size():
    scn = Scenario.with_rho(4.0, n_items=200, delta=0.1)
    big = Scenario(scn.delta, scn.s_procedure, scn.s_item, scn.s_error, 3000)
    D = simulate.draw(big, 60, np.random.default_rng(2))
    out = real.variance_summary(D, n_practice=200)
    assert out["rho"] == pytest.approx(4.0, rel=0.35)
    assert out["share"] == pytest.approx(0.8, abs=0.05)


def test_coverage_experiment_matches_the_closed_form_on_a_synthetic_pool():
    scn = Scenario.with_rho(4.0, n_items=200, delta=0.0)
    big = Scenario(scn.delta, scn.s_procedure, scn.s_item, scn.s_error, 1500)
    D = simulate.draw(big, 40, np.random.default_rng(3))
    pool = real.Pool("synthetic", D)
    res = real.coverage_experiment(
        pool, r=1, n_items=200, n_experiments=1500, rng=np.random.default_rng(4)
    )
    expected = simulate.analytic_eval_only_coverage(4.0)
    assert abs(res["items_t"]["coverage"] - expected) < 0.07
    res5 = real.coverage_experiment(
        pool, r=5, n_items=200, n_experiments=600, rng=np.random.default_rng(5)
    )
    assert 0.88 <= res5["crossed"]["coverage"] <= 0.99
    assert res5["crossed"]["half_width"] > res["items_t"]["half_width"]


def test_coverage_experiment_needs_replicates_left_for_the_truth():
    pool = real.Pool("tiny", np.random.default_rng(6).normal(size=(4, 50)))
    with pytest.raises(ValueError):
        real.coverage_experiment(
            pool, r=4, n_items=20, n_experiments=5, rng=np.random.default_rng(7)
        )


def test_binary_pool_exposes_the_newcombe_interval():
    rng = np.random.default_rng(8)
    A = np.stack([_binary_pair(rng, 300, 0.5, 0.5, 0.5)[0] for _ in range(6)])
    B = np.stack([_binary_pair(rng, 300, 0.5, 0.55, 0.5)[1] for _ in range(6)])
    pool = real.Pool("bin", (B - A).astype(float), A=A, B=B)
    assert pool.binary
    res = real.coverage_experiment(
        pool, r=2, n_items=100, n_experiments=40, rng=np.random.default_rng(9)
    )
    assert {"items_t", "items_boot", "items_newcombe", "crossed", "multiboot"} <= set(res)


def test_protocol_artefact_from_estimating_the_truth_is_small_and_one_sided():
    scn = Scenario.with_rho(4.0, n_items=200, delta=0.0)
    out = real.protocol_check(
        scn, R=30, r=1, N=1000, n_items=200, n_pools=40, n_per_pool=150,
        rng=np.random.default_rng(10),
    )
    assert out["pool_truth_coverage"] <= out["known_truth_coverage"] + 0.02
    assert out["known_truth_coverage"] - out["pool_truth_coverage"] < 0.08


def test_budget_check_returns_defined_ratios_for_a_clear_effect():
    scn = Scenario(delta=0.25, s_procedure=0.15, s_item=1.0, s_error=0.8, n_items=200)
    big = Scenario(scn.delta, scn.s_procedure, scn.s_item, scn.s_error, 1000)
    pool = real.Pool("effect", simulate.draw(big, 30, np.random.default_rng(11)))
    out = real.budget_check(
        pool, r_pilot=5, n_items=200, n_experiments=100, rng=np.random.default_rng(12)
    )
    assert out["delta"] > 0
    assert math.isfinite(out["median_ratio"]) or out["undefined_share"] == 1.0
    with pytest.raises(ValueError):
        real.budget_check(
            real.Pool("zero", np.zeros((5, 40))), r_pilot=3, n_items=20, n_experiments=3,
            rng=np.random.default_rng(13),
        )


def test_smaller_target_effect_needs_more_replicates():
    scn = Scenario(delta=0.25, s_procedure=0.2, s_item=1.0, s_error=0.8, n_items=200)
    big = Scenario(scn.delta, scn.s_procedure, scn.s_item, scn.s_error, 1500)
    pool = real.Pool("effect", simulate.draw(big, 40, np.random.default_rng(21)))
    full = real.budget_check(
        pool, r_pilot=5, n_items=200, n_experiments=20, rng=np.random.default_rng(22)
    )
    small = real.budget_check(
        pool, r_pilot=5, n_items=200, n_experiments=20, rng=np.random.default_rng(22),
        delta_scale=0.25,
    )
    assert small["delta"] == pytest.approx(full["delta"] * 0.25)
    assert math.isnan(small["pool_need"]) or small["pool_need"] >= full["pool_need"]
