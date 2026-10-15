"""Analyse the real replicate pools and write the committed results.

Reads the per item arrays under ``data/`` and writes

* ``results/real-results.json``, every number the paper quotes, rounded;
* ``results/real-results.md``, the same as readable tables.

Nothing here trains or quantizes a model. It is the cheap half and it is what the
drift gate regenerates. Pseudo-experiment randomness is seeded per pool from the
pool name, so a rerun reproduces the file exactly.

Usage::

    python scripts/analyze_real.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from procedure_variance_coverage import real, stats  # noqa: E402
from procedure_variance_coverage.simulate import Scenario  # noqa: E402

DATA = ROOT / "data"
OUT_JSON = ROOT / "results" / "real-results.json"
OUT_MD = ROOT / "results" / "real-results.md"

N_PRACTICE = 200
N_EXPERIMENTS = 2000
N_BOOT_RHO = 300
R_CORRECTION = (2, 3, 5)
R_PILOT = (3, 5)
EXPLORE_SCALES = (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2)
N_SWEEP = (50, 100, 200, 500, 1000)
N_SWEEP_EXPERIMENTS = 1000

QUANT_TAGS = ("smollm2-135m", "smollm2-360m", "pythia-160m")
LEVELS = ("Q4_K", "Q2_K")
SFT_TAG = "smollm2-135m"


def seed_for(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def load_sft(tag: str) -> list[real.Pool]:
    d = DATA / f"sft_{tag}"
    reps = sorted(d.glob("rep*.npz"))
    valid = np.load(d / "base.npz")["valid"].astype(bool)
    loss_n = np.stack([np.load(p)["loss_naive"][valid] for p in reps])
    loss_b = np.stack([np.load(p)["loss_block"][valid] for p in reps])
    arc_n = np.stack([np.load(p)["arc_naive"] for p in reps]).astype(int)
    arc_b = np.stack([np.load(p)["arc_block"] for p in reps]).astype(int)
    return [
        real.Pool(f"sft/{tag}/loss", loss_n - loss_b),
        real.Pool(f"sft/{tag}/accuracy", (arc_b - arc_n).astype(float), A=arc_n, B=arc_b),
    ]


def load_quant(tag: str) -> list[real.Pool]:
    d = DATA / f"quant_{tag}"
    reps = sorted(d.glob("rep*.npz"))
    plain = np.load(d / "plain.npz")
    arrays = [np.load(p) for p in reps]
    pools = []
    for lv in LEVELS:
        for contrast, a_key, b_key in (("imatrix", None, "mix"), ("corpus", "code", "wiki")):
            for metric in ("kl", "flip", "arc"):
                if a_key is None:
                    a = np.stack([plain[f"{metric}_{lv}"] for _ in arrays])
                else:
                    a = np.stack([z[f"{metric}_{lv}_{a_key}"] for z in arrays])
                b = np.stack([z[f"{metric}_{lv}_{b_key}"] for z in arrays])
                name = f"quant/{tag}/{contrast}/{lv}/{'accuracy' if metric == 'arc' else metric}"
                if metric == "arc":
                    pools.append(real.Pool(name, (b - a).astype(float), A=a.astype(int),
                                           B=b.astype(int)))
                else:
                    pools.append(real.Pool(name, a.astype(float) - b.astype(float)))
    return pools


def rnd(x: float, digits: int = 3) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return float(np.round(x, digits))


def sig(x: float, n: int = 3) -> float | None:
    if x is None or not np.isfinite(x) or x == 0:
        return None if x is None or not np.isfinite(x) else 0.0
    return float(f"{x:.{n}g}")


def analyse_pool(pool: real.Pool) -> dict:
    rng = np.random.default_rng(seed_for(pool.name))
    n_items = min(N_PRACTICE, pool.N)
    vs = real.variance_summary(pool.D, n_items)
    lo, hi = real.rho_interval(pool.D, n_items, n_boot=N_BOOT_RHO, rng=rng)
    out = {
        "name": pool.name,
        "R": pool.R,
        "N": pool.N,
        "binary": pool.binary,
        "mean_delta": sig(float(pool.D.mean())),
        "s2_proc": sig(vs["s2_proc"]),
        "s2_item": sig(vs["s2_item"]),
        "s2_err": sig(vs["s2_err"]),
        "rho": sig(vs["rho"]),
        "rho_lo": sig(lo),
        "rho_hi": sig(hi),
        "share": rnd(vs["share"], 3),
        "coverage": {},
        "budget": {},
    }
    for r in (1, *R_CORRECTION):
        res = real.coverage_experiment(
            pool, r=r, n_items=n_items, n_experiments=N_EXPERIMENTS, rng=rng,
            include_items=(r == 1),
        )
        out["coverage"][str(r)] = {
            m: {"coverage": rnd(v["coverage"], 3), "mc_se": rnd(v["mc_se"], 3),
                "half_width": sig(v["half_width"])}
            for m, v in res.items()
        }
    # Items-only coverage as the evaluation grows. Procedure variance does not shrink
    # with more items, so the variance ratio grows with n and coverage should fall.
    out["n_sweep"] = {}
    for n in N_SWEEP:
        # At n close to N the finite population correction shrinks the true item noise
        # and flatters the items-only interval, so the sweep stops at half the pool.
        if n <= pool.N // 2:
            res_n = real.coverage_experiment(
                pool, r=1, n_items=n, n_experiments=N_SWEEP_EXPERIMENTS, rng=rng,
                only=("items_t",),
            )["items_t"]
            out["n_sweep"][str(n)] = {
                "coverage": rnd(res_n["coverage"], 3), "mc_se": rnd(res_n["mc_se"], 3),
                "rho": sig(real.variance_summary(pool.D, n)["rho"]),
            }
    # Interval for the procedure level difference from the whole pool.
    full = stats.crossed_interval(pool.D)
    out["full_pool"] = {
        "estimate": sig(full.estimate), "lo": sig(full.lo), "hi": sig(full.hi),
        "df": sig(full.df),
    }
    out["budget_small"] = {}
    if abs(float(pool.D.mean())) > 0:
        for rp in R_PILOT:
            b = real.budget_check(
                pool, r_pilot=rp, n_items=n_items, n_experiments=500, rng=rng, delta_scale=1.0
            )
            out["budget"][str(rp)] = {
                k: sig(v) if isinstance(v, float) else v for k, v in b.items()
            }
        # Exploratory. A large effect needs only the minimum of two replicates, which
        # says little about the rule. Take the smallest fraction of the observed effect
        # at which the rule is defined and asks for at least four replicates.
        for scale in EXPLORE_SCALES:
            probe = real.budget_check(
                pool, r_pilot=5, n_items=n_items, n_experiments=1, rng=rng, delta_scale=scale
            )
            if np.isfinite(probe["pool_need"]) and probe["pool_need"] >= 4:
                out["budget_small_scale"] = scale
                for rp in R_PILOT:
                    b = real.budget_check(
                        pool, r_pilot=rp, n_items=n_items, n_experiments=500, rng=rng,
                        delta_scale=scale,
                    )
                    out["budget_small"][str(rp)] = {
                        k: sig(v) if isinstance(v, float) else v for k, v in b.items()
                    }
                break
    # protocol artefact for a pool with this size and this rho
    if np.isfinite(vs["rho"]) and vs["rho"] > 0:
        scn = Scenario.with_rho(float(vs["rho"]), n_items=n_items, delta=0.0)
        pc = real.protocol_check(
            scn, R=pool.R, r=1, N=pool.N, n_items=n_items, n_pools=40, n_per_pool=100, rng=rng
        )
        out["protocol_check"] = {k: rnd(v, 3) for k, v in pc.items()}
    return out


def render_md(results: list[dict]) -> str:
    """A readable table of every pool. The JSON is the source of truth."""
    lines = [
        "# Results on real replicate pools",
        "",
        "Generated by `scripts/analyze_real.py` from the per item arrays in `data/`.",
        "Coverage is the",
        "share of pseudo-experiments whose nominal 95% interval covers the procedure level",
        f"difference, with {N_EXPERIMENTS} pseudo-experiments and {N_PRACTICE} items each.",
        "`rho` is procedure variance relative to the item sampling variance of one run at",
        f"{N_PRACTICE} items, with a bootstrap interval over replicates and items.",
        "",
        "| pool | R | N | rho [95% CI] | share | items t, r=1 | items boot, r=1 | crossed r=3 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in sorted(results, key=lambda q: q["name"]):
        c1 = p["coverage"]["1"]
        c3 = p["coverage"]["3"]["crossed"]["coverage"]
        lines.append(
            f"| {p['name']} | {p['R']} | {p['N']} | {p['rho']} [{p['rho_lo']}, {p['rho_hi']}] | "
            f"{p['share']} | {c1['items_t']['coverage']} | {c1['items_boot']['coverage']} | {c3} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("sft", "quant", "all"), default="all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json-out", default=None, help="write here instead of results/")
    args = ap.parse_args()

    pools = []
    if args.only in ("sft", "all"):
        pools += load_sft(SFT_TAG)
    if args.only in ("quant", "all"):
        for tag in QUANT_TAGS:
            if (DATA / f"quant_{tag}").is_dir():
                pools += load_quant(tag)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(analyse_pool, pools))
    meta = {
        "n_practice": N_PRACTICE, "n_experiments": N_EXPERIMENTS, "n_boot_rho": N_BOOT_RHO,
        "sft": json.loads((DATA / f"sft_{SFT_TAG}" / "meta.json").read_text()),
        "quant": {t: json.loads((DATA / f"quant_{t}" / "meta.json").read_text())
                  for t in QUANT_TAGS
                  if args.only != "sft" and (DATA / f"quant_{t}" / "meta.json").is_file()},
    }
    payload = json.dumps({"meta": meta, "pools": results}, indent=1) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(payload, encoding="utf-8", newline="\n")
        print(f"wrote {args.json_out} with {len(results)} pools")
        return
    OUT_JSON.write_text(payload, encoding="utf-8", newline="\n")
    OUT_MD.write_text(render_md(results), encoding="utf-8", newline="\n")
    print(f"wrote {OUT_JSON} and {OUT_MD} with {len(results)} pools")


if __name__ == "__main__":
    main()
