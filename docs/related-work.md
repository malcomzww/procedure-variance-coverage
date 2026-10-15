# Related work

Checked 2026-10-15. Search tooling was noisy, so an absence here is weak evidence. A manual
Google Scholar and Semantic Scholar pass with citation tracking is still to do. Re-check
weekly until posting.

**Read depth** is what was actually read. Abstract or snippet rows are leads, not verified
claims.

| Work | Date | What it shows | Covers eval-only interval coverage for a cross replicate difference | Read depth |
|---|---|---|---|---|
| Singh, arXiv 2608.15046, Certifying Compressed Language Models | Aug 2026 | Churn is about five times the net delta over 1,707 paired cells. A preregistered GPTQ versus AWQ study reversed the order in 5 of 8 cells when only the calibration draw changed | No. Reports seed SD against item SE as separate components. Quantization only, no llama.cpp, SFT or RAG | Full text, appendices grepped |
| arXiv 2604.11581, Hidden Measurement Error in LLM Pipelines | Apr 2026 | Naive standard errors are 40 to 60 percent too small. Coverage falls as n grows. Variance decomposition and a replicate budget idea | Yes in simulation, but for judge, prompt and temperature variance, not training, compression or retrieval randomness | Full text, appendices grepped |
| Sellam et al., MultiBERTs, arXiv 2106.16163 | 2021 | The Multi-Bootstrap over pretraining seeds, fine-tuning seeds and test items | No coverage validation seen | Full text |
| Madaan et al., arXiv 2406.10229 | Jun 2024 | Pretraining seed variance at 7B is generally below the average bootstrap interval, with a variable ratio | Compares seed SD with interval width, not coverage of a delta interval | Abstract and results |
| Arawjo, arXiv 2609.35815, evalstats | Sep 2026 | Plain bootstrap overconfident at small N. Recommends Bonett-Price for paired binary data | Item sampling only, judge noise simulated, no training or compression replicates | Full text |
| Gonzalez et al., arXiv 2509.24086 | Sep 2025 | Repeated runs under stochastic decoding, rank instability | No coverage | Abstract |
| Miller, arXiv 2411.00640 | Nov 2024 | Error bars and power formulas that treat items as the only randomness | Defines the interval this study tests | Abstract |
| Bouthillier et al., arXiv 2103.03098 | 2021 | All sources of variation in ML benchmarks | Non LLM, unverified in detail | Snippet |
| arXiv 2609.07664, 2608.21019, 2610.04403, 2608.06564 | 2026 | Quantization fidelity, calibration data selection and flip prediction | None measure coverage | Full text for the first four |
| arXiv 2402.13991, 2407.09105 | 2024 | Packing and masking claims the SFT arm re-tests | No seed variance | Full text |
| arXiv 2608.00585 | Aug 2026 | Per-chunk filtering fails on multi-hop retrieval, single seed, eval-only intervals | None | Full text |

## What is published

The two level bootstrap over seeds and items exists. Seed variance compared with a bootstrap
interval exists at pretraining scale. A preregistered calibration draw study exists for
GPTQ and AWQ. Undercoverage of naive intervals exists for judge, prompt and temperature
variance.

## What was not found

Empirical coverage of the eval-only interval for the cross replicate difference, a variance
ratio, and a replicate budget rule, measured across SFT packing and llama.cpp quantization
with a validated correction.

## A conceptual caution

An eval-only interval is valid for the artifact level estimand. It will miss the procedure
level estimand by construction. The paper is only interesting if it measures how large the
gap is in practice, per pipeline, so the estimand is defined first and the claim is scoped to
procedure level conclusions.

## Terminology

"Pseudoreplication" is used in general statistics guides for CS papers (arXiv 2605.00428) and
in passing in one LLM evaluation stopping paper. No paper found uses it as the frame of this
question.
