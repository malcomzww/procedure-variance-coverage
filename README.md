# procedure-variance-coverage

**An evaluation interval built from test items covers the difference between two trained
models. It does not necessarily cover the difference between two training recipes.** This repo
measures how often the two come apart, on real fine-tuning and quantization replicates, with
the analysis preregistered.

The paper is *Pseudoreplication in LLM Evaluation and the Anti-Conservative Coverage of
Eval-Set Confidence Intervals*. Its source, figures and arXiv package are in the parent
workspace under `papers/procedure-variance/`.

## Headline

<!-- headline:start -->

**Supervised fine-tuning, naive against block diagonal packing, 30 replicates.** Procedure randomness is 65% of the variance of a single-replicate difference in completion loss (rho 1.86). The nominal 95% items-only interval covers the procedure level difference in 74% of pseudo-experiments (Monte Carlo standard error 0.01). For ARC-Easy accuracy on the same models the same interval covers in 96%.

Coverage falls as the evaluation grows, from 88% with 50 items to 32% with 1000.

A crossed random effects interval estimated from 3 replicates covers in 93% and from 5 replicates in 94%.

**Quantization, importance matrix against plain, Q4_K, KL divergence to the unquantized model.** SmolLM2-135M, rho 0.232, items-only coverage 96%, crossed interval from 3 replicates 99%; SmolLM2-360M, rho 0.735, items-only coverage 91%, crossed interval from 3 replicates 98%; Pythia-160M, rho 0.244, items-only coverage 96%, crossed interval from 3 replicates 99%.

<!-- headline:end -->

All of the numbers above are generated from `results/real-results.json` by
`scripts/update_readme.py`, and CI fails if they drift. The full tables are in
[`results/real-results.md`](results/real-results.md).

## The question

A paper reports that method B beats method A with a confidence interval computed over the
evaluation items. That interval treats the items as the only source of randomness. A reader
usually takes the sentence to mean something stronger, that B beats A when the recipe is
re-run. Re-running a recipe changes the training subset, the data order or the calibration
draw, and the items interval knows nothing about that. Treating items as replicates of a
procedure is pseudoreplication.

Two estimands follow.

* **Artifact level.** The difference between these particular models.
* **Procedure level.** The expected difference under fresh randomness in the recipe.

The coverage of the items-only interval for the procedure level difference is
`2 Phi(z / sqrt(1 + rho)) - 1`, where `rho` is procedure variance relative to the item sampling
variance of one run. Because that sampling variance shrinks with the number of items and
procedure variance does not, `rho` grows with the number of items and coverage falls.

## What was run

* **SFT.** SmolLM2-135M, naive against block diagonal packing, one epoch on 500 instruction
  examples. A replicate is a fresh draw of the examples and a fresh batch order. Items are a
  held out pool of 2000 Dolly examples scored by completion loss, and 1000 ARC-Easy questions
  scored by correctness.
* **Quantization.** llama.cpp k-quants at Q4_K and Q2_K on SmolLM2-135M, SmolLM2-360M and
  Pythia-160M, through the real ggml kernels. A replicate is a fresh draw of calibration
  windows. Two contrasts, an importance matrix against plain quantization, and calibration on
  code against calibration on encyclopedic text. Items are 500 evaluation windows scored by KL
  divergence to the unquantized model and by top-1 flip rate, and 500 ARC-Easy questions.
* **Pseudo-experiments.** From a pool of replicates, draw `r` replicates and 200 items, build
  the interval, and check whether it covers the mean difference over the replicates not drawn.
  The numbers of replicates per pool are in [`data/README.md`](data/README.md).

## Validation of the statistics

`results/synthetic-validation.md` checks every interval where the truth is known. The
single-replicate items interval matches its closed form, the crossed random effects interval
reaches near nominal coverage from five replicates, and the Multi-Bootstrap undercovers more
than the crossed interval at three and five replicates. `tests/` checks the pool engine, the
paired proportion interval and the analysis script.

## Preregistration

[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) fixed the hypotheses and decision rules
before any replicate was analysed, with a dated log of every later change. The decision rules
were applied exactly as written, including where they fail. The paper reports each verdict.

## Limitations

* The pipelines are small. Fine-tuning uses one model with full fine-tuning for one epoch.
  Quantization uses three models of at most 360M parameters.
* Each pipeline varies one source of randomness. Adapter initialisation and decoding are not
  varied.
* The procedure level truth is estimated from a finite pool of replicates. The size of that
  artefact was measured on synthetic pools and is reported in the paper.
* The replicate budget rule is noisy at small pilots, and its test at a smaller effect was an
  exploratory analysis added after seeing the first results.
* Per item values depend on hardware in the last bits. Committed results are rounded.
* Retrieval augmented generation is not covered.

## Reproduce

The analysis needs only numpy and scipy and reads the per item arrays in `data/`.

```bash
uv sync --extra dev
uv run python scripts/generate_results.py      # synthetic validation
uv run python scripts/analyze_real.py          # real pools
uv run python scripts/update_readme.py         # headline
uv run pytest
```

Training and quantization need a CUDA environment with PyTorch, Transformers, Datasets and
gguf, the sibling checkouts `sft-loss-masking-and-packing` and `quantization-accuracy-curves`,
and the ggml library from a `llama-cpp-python` wheel.

```bash
python scripts/run_sft_replicates.py --replicates 30
python scripts/run_quant_replicates.py --tag smollm2-135m --model HuggingFaceTB/SmolLM2-135M
```

The runs behind the paper used one laptop with an NVIDIA RTX 4060 (8 GB) and a 24 thread CPU,
in October 2026. Run these one at a time, since two jobs sharing the GPU memory slowed both
by a large factor.

## Related work

See [`docs/related-work.md`](docs/related-work.md). The closest papers are arXiv 2608.15046,
2604.11581 and the MultiBERTs paper. This repo adds the empirical coverage of the items
interval for a procedure level difference, the variance ratio per pipeline, and a replicate
budget rule, measured across fine-tuning and quantization.

## Cite

See [`CITATION.cff`](CITATION.cff).

## License

MIT
