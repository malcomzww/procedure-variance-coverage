# procedure-variance-coverage

A study of how often an evaluation interval built from test items covers the procedure level
difference between two LLM pipelines. It is the backing repo for an early standalone paper.
The dissertation paper lives in `guardrail-transfer-study` and is separate.

This file is the operating manual for this repo. It travels with the repo. The workspace rules
in `C:\maworks\CLAUDE.md` and `C:\maworks\projects\CLAUDE.md` also apply when this repo sits
inside the maworks folder. The rules below apply everywhere.

## Rules that are not negotiable

**No AI attribution anywhere.** No `Co-Authored-By` trailer, no "Generated with", no mention of
any AI tool in a commit message, code comment, README or doc. The commit history must show the
author's own work.

**Record every completed change in `PROGRESS.md`** in the same commit as the change.

**Preregistration comes first.** `docs/PREREGISTRATION.md` is fixed before any real replicate
runs. Changes go in its deviations log with a date and a reason. Never edit it in place.

**Writing rules.** No em or en dashes, no colons or semicolons as sentence connectors, plain
sentences. Applies to the README, results files, docs, commit messages and the paper.

## Stack

Python 3.11 or later, `uv`, numpy, scipy, pytest, ruff pinned at 0.14.3.

## Commands

| | |
|---|---|
| Install | `uv sync --extra dev` |
| Results | `uv run python scripts/generate_results.py` (locally without uv, `PYTHONPATH=src python ...`) |
| Tests | `uv run pytest` |
| Lint | `python -m ruff check .` |

CI regenerates `results/` and fails on any diff.

## Layout

```
src/procedure_variance_coverage/   stats.py (intervals, variance components, budget), simulate.py
tests/                             closed form and known truth checks
scripts/generate_results.py        synthetic validation, asserts claims before writing
results/                           committed, machine independent, rounded
docs/                              PLAN.md, PREREGISTRATION.md, related-work.md
```

## The results contract

`generate_results.py` asserts every claim the README makes before it writes a line. Never type a
figure into a README or the paper from memory. Read it from a results file or regenerate it.
Commit ratios, orderings and invariants only, rounded so floating point drift cannot change
them. Raw per replicate outputs go to the gitignored `runs/` and `results/*-raw.*`.

## Honesty labelling

Right now this repo is synthetic. It calls no model. The README says so in its first
paragraph and must keep saying so until real replicates exist. A synthetic coverage table says
the statistics are right. It says nothing about any real pipeline.

## Known traps

- **The source repos have no replicate axis.** The SFT seed does not change training and the
  quantization inputs are fixed. A replicate has to be defined on purpose. See the
  preregistration.
- **The quantization evaluation is 896 tokens.** Too small for any claim. Replace it before
  measuring.
- **Thread count changes the last digits of float results.** Commit rounded summaries only.
- **A claim written before it was measured can be wrong.** The Multi-Bootstrap docstring once
  said it undercovers at few replicates, then the data showed the real pattern. Measure, then
  write.
- **Publication order is fixed.** Repo public, then arXiv, then workshop.
