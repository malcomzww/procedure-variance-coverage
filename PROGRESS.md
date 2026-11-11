# Progress

The running record of this repo. **Read it first when a session starts, and update it in the
same commit as every completed change.**

## Where it stands

All real replicate pools are complete and the full analysis is committed. SFT, one pool of 30
replicates; quantization, three models (SmolLM2-135M, SmolLM2-360M, Pythia-160M) at 20 replicates
each, both contrasts at Q4_K and Q2_K. `results/real-results.json` holds all 38 pools. The paper
in `C:\maworks\papers\procedure-variance` is regenerated from it, compiles, and the arXiv package
is built and verified. H1 holds for SFT completion loss (74% coverage) and, as the
preregistration allowed, does not hold for quantization KL (91% to 96%); the coverage of every
continuous pool follows the closed form in rho. The repo exists locally only and has not been
created on GitHub.

## In flight

| Stream | Session or owner | Started | State |
|---|---|---|---|
| Step 0 and Step 2, scaffold and statistics module | owner session | 2026-10-15 | done, see Done |
| Real pools, analysis and paper | owner session | 2026-11-11 | done, see Done |

## Next

1. Owner to confirm creating the public GitHub repo `malcomzww/procedure-variance-coverage` and
   registering it as slate entry 28 in `roadmap/slate.json` and `roadmap/ROADMAP.md`, mapped to
   the 3B statistics KNOW and the SFT and quantization DOs. Nothing is pushed until then.
2. arXiv submission of the paper in `C:\maworks\papers\procedure-variance`, after the repo is
   public and citable. First time submitting, so an endorsement for cs.LG is needed.
3. Add the Nadeau and Bengio style variance inflation as a further comparator.

## Done

Newest first.

| Date | Change | Commit |
|---|---|---|
| 2026-11-11 | Quantization pools for three models at R = 20, full real analysis of 38 pools, README headline, paper regenerated and arXiv package verified | this commit |
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
- **SmolLM2-360M replicates run with the expandable segments allocator.** The default PyTorch
  allocator fragmented on the 8 GB GPU and failed at replicate 12 with an out of memory error.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` changes allocation only, never a computed
  value, and the pool resumed from the twelve replicates already on disk.
