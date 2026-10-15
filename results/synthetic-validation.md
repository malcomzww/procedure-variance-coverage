# Synthetic validation of the intervals

Known truth, no model. `rho` is procedure variance relative to the item
sampling variance of one run, with 200 items. Coverage is the share of
pseudo-experiments whose 95% interval covers the procedure level difference.
Monte Carlo standard error is about 0.01 or less in every cell.

## Eval-only interval, one replicate

| rho | measured coverage | closed form |
|---|---|---|
| 0 | 0.95 | 0.95 |
| 1 | 0.84 | 0.83 |
| 4 | 0.63 | 0.62 |
| 16 | 0.36 | 0.37 |

## Crossed random effects interval, R replicates

| rho | R=2 | R=3 | R=5 | R=10 |
|---|---|---|---|---|
| 0 | 0.95 | 0.94 | 0.95 | 0.94 |
| 1 | 0.92 | 0.94 | 0.94 | 0.95 |
| 4 | 0.88 | 0.92 | 0.94 | 0.94 |
| 16 | 0.87 | 0.94 | 0.94 | 0.95 |

## Multi-Bootstrap compared with the crossed interval

| rho | R | multi-bootstrap | crossed |
|---|---|---|---|
| 4 | 3 | 0.90 | 0.92 |
| 4 | 5 | 0.92 | 0.94 |
| 16 | 3 | 0.82 | 0.94 |
| 16 | 5 | 0.89 | 0.94 |

Reproduce with `uv run python scripts/generate_results.py`.
