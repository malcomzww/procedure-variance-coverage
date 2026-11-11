"""Write the README headline from ``results/real-results.json``.

The text between the ``headline`` markers in ``README.md`` is generated here, so a number
in the README cannot differ from the results file. CI runs this after the analysis and
fails on any difference.

Usage::

    python scripts/update_readme.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results" / "real-results.json"
START, END = "<!-- headline:start -->", "<!-- headline:end -->"

NAMES = {"smollm2-135m": "SmolLM2-135M", "smollm2-360m": "SmolLM2-360M",
         "pythia-160m": "Pythia-160M"}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def headline(pools: dict[str, dict]) -> str:
    out = []
    sft = pools["sft/smollm2-135m/loss"]
    acc = pools["sft/smollm2-135m/accuracy"]
    c = sft["coverage"]["1"]["items_t"]
    out.append(
        f"**Supervised fine-tuning, naive against block diagonal packing, {sft['R']} replicates.** "
        f"Procedure randomness is {pct(sft['share'])} of the variance of a single-replicate "
        f"difference in completion loss (rho {sft['rho']}). The nominal 95% items-only interval "
        f"covers the procedure level difference in {pct(c['coverage'])} of pseudo-experiments "
        f"(Monte Carlo standard error {c['mc_se']}). For ARC-Easy accuracy on the same models "
        f"the same interval covers in {pct(acc['coverage']['1']['items_t']['coverage'])}."
    )
    sweep = sft.get("n_sweep", {})
    if "50" in sweep and "1000" in sweep:
        out.append(
            f"Coverage falls as the evaluation grows, from {pct(sweep['50']['coverage'])} with 50 "
            f"items to {pct(sweep['1000']['coverage'])} with 1000."
        )
    cr = sft["coverage"]
    out.append(
        f"A crossed random effects interval estimated from 3 replicates covers in "
        f"{pct(cr['3']['crossed']['coverage'])} and from 5 replicates in "
        f"{pct(cr['5']['crossed']['coverage'])}."
    )
    quant_lines = []
    for tag, name in NAMES.items():
        p = pools.get(f"quant/{tag}/imatrix/Q4_K/kl")
        if p:
            quant_lines.append(
                f"{name}, rho {p['rho']}, items-only coverage "
                f"{pct(p['coverage']['1']['items_t']['coverage'])}, crossed interval from 3 "
                f"replicates {pct(p['coverage']['3']['crossed']['coverage'])}"
            )
    if quant_lines:
        out.append(
            "**Quantization, importance matrix against plain, Q4_K, KL divergence to the "
            "unquantized model.** " + "; ".join(quant_lines) + "."
        )
    return chr(10).join(line + chr(10) for line in out)


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    pools = {p["name"]: p for p in data["pools"]}
    text = README.read_text(encoding="utf-8")
    a, b = text.index(START) + len(START), text.index(END)
    new = text[:a] + chr(10) + chr(10) + headline(pools) + chr(10) + text[b:]
    README.write_text(new, encoding="utf-8", newline=chr(10))
    print("updated the README headline")


if __name__ == "__main__":
    main()
