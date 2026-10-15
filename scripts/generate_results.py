"""Validate the intervals on synthetic data with known truth.

Writes ``results/synthetic-validation.md``. Every claim made in the README about
this table is asserted here before the file is written, so a change to the
statistics that breaks a claim fails this script instead of quietly changing a
number.

This is a check on the statistics, not on any model. No model is called. Rates
are rounded to two decimals so the committed file does not depend on the last
bits of floating point arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from procedure_variance_coverage import simulate, stats
from procedure_variance_coverage.simulate import Scenario

N_ITEMS = 200
RHOS = (0.0, 1.0, 4.0, 16.0)
REPLICATES = (2, 3, 5, 10)
SEED = 20261015
OUT = Path(__file__).resolve().parent.parent / "results" / "synthetic-validation.md"


def _scenario(rho: float) -> Scenario:
    return Scenario.with_rho(rho, n_items=N_ITEMS, delta=0.0)


def eval_only_rows() -> list[dict]:
    rows = []
    for k, rho in enumerate(RHOS):
        cov = simulate.measure_coverage(
            _scenario(rho),
            n_replicates=1,
            interval=lambda D: stats.eval_only_t_interval(D[0]),
            n_experiments=4000,
            rng=np.random.default_rng(SEED + k),
        )
        rows.append(
            {
                "rho": rho,
                "measured": cov.rate,
                "analytic": simulate.analytic_eval_only_coverage(rho),
                "mc_se": cov.mc_se,
            }
        )
    return rows


def crossed_rows() -> dict[tuple[float, int], float]:
    out = {}
    for k, rho in enumerate(RHOS):
        for j, r in enumerate(REPLICATES):
            cov = simulate.measure_coverage(
                _scenario(rho),
                n_replicates=r,
                interval=stats.crossed_interval,
                n_experiments=3000,
                rng=np.random.default_rng(SEED + 100 + 10 * k + j),
            )
            out[(rho, r)] = cov.rate
    return out


def multiboot_rows() -> dict[tuple[float, int], float]:
    out = {}
    for k, rho in enumerate((4.0, 16.0)):
        for j, r in enumerate((3, 5)):
            boot_rng = np.random.default_rng(SEED + 500 + 10 * k + j)
            cov = simulate.measure_coverage(
                _scenario(rho),
                n_replicates=r,
                interval=lambda D, g=boot_rng: stats.multi_bootstrap_interval(
                    D, n_resamples=300, rng=g
                ),
                n_experiments=1500,
                rng=np.random.default_rng(SEED + 600 + 10 * k + j),
            )
            out[(rho, r)] = cov.rate
    return out


def assert_claims(eval_only, crossed, multiboot) -> None:
    for row in eval_only:
        gap = abs(row["measured"] - row["analytic"])
        assert gap < 4 * row["mc_se"] + 0.005, f"eval-only off the closed form at {row}"
    measured = [r["measured"] for r in eval_only]
    assert measured == sorted(measured, reverse=True), "coverage must fall as rho grows"
    assert eval_only[0]["measured"] > 0.93, "at rho 0 the eval-only interval is calibrated"
    assert eval_only[-1]["measured"] < 0.4, "at rho 16 the eval-only interval fails badly"
    for rho in RHOS:
        for r in (5, 10):
            assert 0.93 <= crossed[(rho, r)] <= 0.97, f"crossed off nominal at {rho}, R={r}"
    for rho in (4.0, 16.0):
        assert crossed[(rho, 2)] < 0.95, "two replicates are not enough to reach nominal"
        for r in (3, 5):
            assert multiboot[(rho, r)] < crossed[(rho, r)], f"multi-bootstrap undercovers at R={r}"


def render(eval_only, crossed, multiboot) -> str:
    lines = [
        "# Synthetic validation of the intervals",
        "",
        "Known truth, no model. `rho` is procedure variance relative to the item",
        f"sampling variance of one run, with {N_ITEMS} items. Coverage is the share of",
        "pseudo-experiments whose 95% interval covers the procedure level difference.",
        "Monte Carlo standard error is about 0.01 or less in every cell.",
        "",
        "## Eval-only interval, one replicate",
        "",
        "| rho | measured coverage | closed form |",
        "|---|---|---|",
    ]
    for row in eval_only:
        lines.append(f"| {row['rho']:g} | {row['measured']:.2f} | {row['analytic']:.2f} |")
    lines += [
        "",
        "## Crossed random effects interval, R replicates",
        "",
        "| rho | " + " | ".join(f"R={r}" for r in REPLICATES) + " |",
        "|---|" + "---|" * len(REPLICATES),
    ]
    for rho in RHOS:
        cells = " | ".join(f"{crossed[(rho, r)]:.2f}" for r in REPLICATES)
        lines.append(f"| {rho:g} | {cells} |")
    lines += [
        "",
        "## Multi-Bootstrap compared with the crossed interval",
        "",
        "| rho | R | multi-bootstrap | crossed |",
        "|---|---|---|---|",
    ]
    for (rho, r), rate in sorted(multiboot.items()):
        lines.append(f"| {rho:g} | {r} | {rate:.2f} | {crossed[(rho, r)]:.2f} |")
    lines += [
        "",
        "Reproduce with `uv run python scripts/generate_results.py`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    eval_only = eval_only_rows()
    crossed = crossed_rows()
    multiboot = multiboot_rows()
    assert_claims(eval_only, crossed, multiboot)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(eval_only, crossed, multiboot), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
