# Progress

The running record of this repo. **Read it first when a session starts, and update it in the
same commit as every completed change.**

## Where it stands

The statistics are built and validated on synthetic data with known truth. The single run items
interval matches its closed form, the crossed random effects interval reaches near nominal
coverage from five replicates, and the Multi-Bootstrap undercovers more at few replicates. No
model has been run. The real replicate runs are preregistered in `docs/PREREGISTRATION.md` and
have not started. The repo exists locally only and has not been created on GitHub.

## In flight

| Stream | Session or owner | Started | State |
|---|---|---|---|
| Step 0 and Step 2, scaffold and statistics module | owner session | 2026-10-15 | done, see Done |
| Step 1, pilot of the replicate axes | owner session | | next |

## Next

1. Owner to confirm creating the public GitHub repo `malcomzww/procedure-variance-coverage` and
   registering it as slate entry 28 in `roadmap/slate.json` and `roadmap/ROADMAP.md`, mapped to
   the 3B statistics KNOW and the SFT and quantization DOs. Nothing is pushed until then.
2. Pilot of three replicates for the SFT axis, to check the axis moves the outcome beyond item
   noise. If it does not, redefine the axis and log it in the preregistration deviations log
   before scaling.
3. Replace the 896 token quantization evaluation with a large held out text plus per item
   multiple choice, add KL to the unquantized model and top-1 flip rate, and build a larger
   multi domain calibration corpus.
4. Full replicate pools, then the coverage analysis on real data.
5. Add the Nadeau and Bengio style variance inflation as a further comparator.
6. Write the paper in `C:\maworks\papers\procedure-variance\`.

## Done

Newest first.

| Date | Change | Commit |
|---|---|---|
| 2026-10-15 | Scaffold, statistics module, synthetic validation, preregistration, related work, README | first commit |

## Decisions

- **The title is fixed, with a condition.** Pseudoreplication in LLM Evaluation and the
  Anti-Conservative Coverage of Eval-Set Confidence Intervals. "Anti-conservative" is a
  hypothesis, and the preregistration says what result would change the wording.
- **RAG is out of v1.** The repo has no model and a deterministic retriever, so it has no
  procedure randomness to measure.
- **A replicate is defined on purpose.** SFT, a fresh training subset draw with shuffled batch
  and packing order. Quantization, a fresh calibration window resample.
- **No existing repo headline is cited.** The 38.56 pp and non-monotone ladder results rest on
  a 135M model and 896 tokens with no intervals, so they are rerun first.
- **The Multi-Bootstrap claim was corrected after measurement.** Undercoverage holds at three
  and at five replicates and grows with procedure variance.
