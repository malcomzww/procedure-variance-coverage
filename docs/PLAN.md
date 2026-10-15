# Paper plan, 2026-10-15

Approved by the owner on 2026-10-15. Status lives in `PROGRESS.md`. The preregistration in
`PREREGISTRATION.md` governs the real runs.

## Title

Pseudoreplication in LLM Evaluation and the Anti-Conservative Coverage of Eval-Set Confidence
Intervals

"Anti-conservative" is a hypothesis. Madaan et al. (arXiv 2406.10229) found seed variance mostly
below the bootstrap interval at pretraining scale. If a pipeline covers well, the paper says so
and the wording changes.

## Novelty status

Partly covered. See `related-work.md`. The closest papers are arXiv 2608.15046 (a preregistered
calibration draw study for GPTQ and AWQ), arXiv 2604.11581 (undercoverage for judge, prompt and
temperature variance) and the MultiBERTs paper (the two level bootstrap). Not found anywhere is
empirical coverage of the eval-only interval for the cross replicate difference, a variance
ratio and a replicate budget rule, across SFT and quantization. The search evidence is weak.

## The estimand

An eval-only interval is valid for the artifact level estimand, the difference between these
particular trained models. A claim that method B beats method A usually means the procedure
level estimand, the expected difference when the recipe is re-run with fresh randomness. The
paper defines both, shows which interval targets which, and measures the gap per pipeline.

## Steps

0. Preregister the analysis and decision rules. Done in `PREREGISTRATION.md`. Register the repo
   in `roadmap/slate.json` and `roadmap/ROADMAP.md` once the owner confirms.
1. Pilot of three replicates to confirm each replicate axis moves the outcome beyond item
   noise.
2. Statistics module validated on synthetic data with known truth. Done, see
   `results/synthetic-validation.md`.
3. Replicate runs. SFT, a fresh training subset draw with shuffled batch and packing order, two
   arms and about 20 replicates, roughly 13 minutes each on CPU by the earlier estimate.
   Quantization, a fresh calibration window resample, evaluated on a large held out text.
4. Analysis with asserted claims and a drift gate that regenerates from committed raw
   summaries, not by retraining in CI.
5. Paper in `C:\maworks\papers\procedure-variance\`, every number traced to a results file.
6. Release in the fixed order, repo public, then arXiv (cs.LG or cs.CL primary, stat.ME
   cross-list), then a workshop. `CITATION.cff`, a stable author name, and an updated CV and
   Scholar profile. Re-run the novelty check weekly until posting.

## Reuse

* `sft-loss-masking-and-packing`, `train_arm`, `ARMS`, `evaluate.evaluate`,
  `dataset.load_dolly`, `scripts/run_ablation.py`.
* `quantization-accuracy-curves`, `quantise_dequantise`, `quantise_model`, `restore_model`,
  `evaluate.collect_importance`, `evaluate.perplexity`, `calibrate.CORPORA`.
* `llm-eval-harness`, `paired_delta_ci`, `bootstrap_ci`, `is_reportable`, `RunMeta`. Percentile
  bootstrap only and pure Python, so used as the naive baseline, with fast versions written
  here. The harness is pinned at v0.1.0 by other repos, so it is not edited.

## Risks

* The authors of arXiv 2608.15046 may extend to llama.cpp and coverage. Preregister, date
  commits, move fast.
* The hypothesis may fail. That is still a result.
* Two pipelines on tiny models fit a strong workshop paper. A top venue needs the correction
  method, the budget rule validated, and at least three models per pipeline.
* The existing 896 token quantization evaluation is too small for any claim.

## Verification

* Tests, ruff 0.14.3 and the results drift gate pass in CI.
* The statistics are validated on synthetic data before touching real runs.
* A pilot shows the replicate axes move the outcome.
* The paper compiles and every number traces to a results file. A reviewer style pass targets
  the estimand argument and the truth definition.
