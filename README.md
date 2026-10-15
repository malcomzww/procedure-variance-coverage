# procedure-variance-coverage

**An evaluation interval built from test items covers the difference between two trained
models. It does not necessarily cover the difference between two training recipes.** This repo
measures how often the two come apart.

> **Status.** The statistics are built and validated on synthetic data with known truth. No
> model has been run yet. Every number below comes from simulation, not from an LLM. The
> real replicate runs are specified in [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md)
> and have not started.

The working paper title is *Pseudoreplication in LLM Evaluation and the Anti-Conservative
Coverage of Eval-Set Confidence Intervals*. "Anti-conservative" is a hypothesis. The
preregistration says what result would change the wording.

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

## What is validated so far

`results/synthetic-validation.md` checks the intervals where the truth is known. The
variance structure is summarised by `rho`, procedure variance relative to the item sampling
variance of one run.

* The single run items interval matches its closed form. Its coverage is about 0.95 at
  `rho` 0 and falls to about 0.63 at `rho` 4 and about 0.36 at `rho` 16.
* A crossed random effects interval with Satterthwaite degrees of freedom reaches about 0.94
  from five replicates, and falls short with two.
* The Multi-Bootstrap undercovers more than the crossed interval at three and five
  replicates when procedure variance is large.

Those results say the statistics are implemented correctly. They say nothing about how large
`rho` is for any real pipeline. That is the empirical question, and it is not answered here.

## What is not here

* No model has been run. There is no SFT, quantization or retrieval replicate yet.
* The two source repos, `sft-loss-masking-and-packing` and `quantization-accuracy-curves`,
  currently have no real replicate axis. The first has a seed that does not change training,
  and the second has fixed inputs and an 896 token evaluation.
* Nothing here supports a claim about any real pipeline until the preregistered runs finish.

## Reproduce

```bash
uv sync --extra dev
uv run python scripts/generate_results.py
uv run pytest
```

The results are seeded and rounded, so they are identical across machines. CI regenerates
them and fails on any diff.

## Related work

See [`docs/related-work.md`](docs/related-work.md). The closest papers are arXiv 2608.15046,
2604.11581 and the MultiBERTs paper. This repo's added material is empirical coverage of the
items interval for a procedure level difference, a variance ratio and a replicate budget rule,
measured across SFT and quantization.

## License

MIT
