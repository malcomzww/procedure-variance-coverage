"""The analysis script must be deterministic and well formed, or the drift gate is meaningless."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from procedure_variance_coverage import real, simulate
from procedure_variance_coverage.simulate import Scenario

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_real.py"


@pytest.fixture(scope="module")
def analyze():
    spec = importlib.util.spec_from_file_location("analyze_real", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.N_EXPERIMENTS = 60
    mod.N_BOOT_RHO = 20
    return mod


def _pool(name="synthetic/x", binary=False):
    scn = Scenario(delta=0.2, s_procedure=0.2, s_item=1.0, s_error=0.8, n_items=200)
    big = Scenario(scn.delta, scn.s_procedure, scn.s_item, scn.s_error, 400)
    D = simulate.draw(big, 10, np.random.default_rng(1))
    if not binary:
        return real.Pool(name, D)
    rng = np.random.default_rng(2)
    a = (rng.random((10, 400)) < 0.5).astype(int)
    b = (rng.random((10, 400)) < 0.55).astype(int)
    return real.Pool(name, (b - a).astype(float), A=a, B=b)


def test_analysis_is_deterministic_for_a_pool(analyze):
    first = analyze.analyse_pool(_pool())
    second = analyze.analyse_pool(_pool())
    assert first == second


def test_analysis_output_has_every_field_the_paper_uses(analyze):
    out = analyze.analyse_pool(_pool())
    for key in ("rho", "rho_lo", "rho_hi", "share", "mean_delta", "coverage", "budget"):
        assert key in out
    assert set(out["coverage"]) == {"1", "2", "3", "5"}
    assert {"items_t", "items_boot"} <= set(out["coverage"]["1"])
    assert {"crossed", "multiboot"} <= set(out["coverage"]["3"])
    assert 0.0 <= out["coverage"]["1"]["items_t"]["coverage"] <= 1.0


def test_binary_pool_reports_the_newcombe_interval(analyze):
    out = analyze.analyse_pool(_pool("synthetic/bin", binary=True))
    assert "items_newcombe" in out["coverage"]["1"]


def test_seed_depends_only_on_the_pool_name(analyze):
    assert analyze.seed_for("a") == analyze.seed_for("a")
    assert analyze.seed_for("a") != analyze.seed_for("b")
