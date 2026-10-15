# Preregistration

Written 2026-10-15, before any real replicate has been run. The statistics module has been
validated on synthetic data only (`results/synthetic-validation.md`). Any change after this
point goes in the deviations log at the bottom with a date and a reason. Nothing here is
edited in place.

## Question

When two LLM pipelines are compared and the difference is reported with an interval built
from evaluation items alone, how often does that interval cover the procedure level
difference, meaning the expected difference when the recipe is re-run with fresh randomness?

## Estimands

* **Artifact level.** The difference between these particular trained models over the
  population of items. An eval-only interval targets this.
* **Procedure level.** The expected difference under fresh randomness in the recipe. This
  is what "method B beats method A" usually claims.

The paper measures the gap between the two in practice, per pipeline.

## Pipelines and replicates

**SFT.** `sft-loss-masking-and-packing`, contrast naive versus block diagonal packing. The
existing seed argument does not change training, so a replicate is defined as a fresh draw
of the training subset together with a shuffled batch and packing order. The held out
evaluation items are fixed. Model SmolLM2-135M first. SmolLM2-360M and Qwen2.5-0.5B only if
time allows, and reported as secondary.

**Quantization.** `quantization-accuracy-curves`, llama.cpp k-quants with an importance
matrix. A replicate is a fresh resample of calibration windows from a larger multi domain
corpus. The evaluation text is fixed and far larger than the current 896 tokens (at least
100,000 tokens), with per item multiple choice questions added. Metrics are KL divergence to
the unquantized model, top-1 flip rate, and accuracy. A second design fixes the draw and
varies the corpus.

**Out of scope for v1.** RAG. The repo has no model and a deterministic retriever.

## Design

* Pool of R replicates per arm, target R = 20, never fewer than 10.
* A pseudo-experiment draws r replicates per arm from the pool. r = 1 is the common
  practice and is the primary condition. r = 2, 3, 5 are used for the corrections.
* The truth for a pseudo-experiment is the mean difference over replicates in the pool that
  were not drawn, evaluated on a large independent item set where one exists. Where the item
  set is fixed and small, the truth is conditional on it and the paper says so.
* At least 2000 pseudo-experiments per cell. Monte Carlo standard error is reported.

## Interval comparators

Eval-only, computed on one replicate: paired t, percentile bootstrap, and for binary items a
Wilson or Bonett-Price paired interval. Corrections, computed on r replicates: the crossed
random effects interval with Satterthwaite degrees of freedom, and the Multi-Bootstrap.

## Hypotheses

* **H1.** In SFT and in quantization, single replicate eval-only 95% intervals cover the
  procedure level difference less often than 95%.
* **H2.** The crossed random effects interval with r between 3 and 5 replicates covers
  within 0.93 to 0.97.
* **H3.** The replicate count predicted from a pilot of 3 to 5 replicates is within a
  factor of 1.5 of the count predicted from the full pool.

## Decision rules

* H1 supports the word "anti-conservative" in the title for a pipeline if the coverage at
  r = 1 is below 0.90 for the primary metric and the Monte Carlo interval excludes 0.95.
  If no pipeline meets that, the title is changed to say that coverage depends on the
  pipeline, and the null is reported as a result.
* H2 and H3 are reported whether or not they hold. A correction that fails is reported as
  failing.
* If a pilot of three replicates shows that a replicate axis does not change the outcome
  beyond item noise, the axis is redefined and the redefinition is logged here before the
  full run.
* No metric, arm or pipeline is dropped after seeing results. Additional analyses are
  labelled exploratory.

## Known limits stated in advance

Tiny models and one family per pipeline. Few replicate pools mean the truth is itself
estimated. Percentile intervals are known to behave badly at small n, which is why
eval-only comparators include t and Wilson style intervals.

## Deviations log

| Date | Change | Reason |
|---|---|---|
