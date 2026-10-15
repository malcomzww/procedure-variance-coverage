"""Run replicated post-training quantization comparisons and score them per item.

A replicate is a fresh draw of calibration windows. Two contrasts are measured,
each at two k-quant levels.

* ``imatrix`` contrast. Arm A is plain quantization with no importance matrix, which is
  deterministic. Arm B uses an importance matrix collected on a fresh random
  mixture of three domains. Randomness enters only through arm B.
* ``corpus`` contrast. Arm A uses an importance matrix collected on random code windows,
  arm B on random encyclopedic windows. Both arms are redrawn in every replicate.

Items are the 128 token evaluation windows of a held out Wikipedia text, scored by
KL divergence to the unquantized model and by the top-1 flip rate, and the
ARC-Easy test questions, scored by correctness. Per item scores are saved so the
analysis can resample items and replicates.

The quantization kernels are the real llama.cpp ggml kernels, called through the
sibling checkout ``quantization-accuracy-curves``. Layers are quantized in
parallel threads, which is safe because the ctypes call releases the GIL and each
call owns its buffers.

Run (CUDA environment, from the repo root)::

    .venv/Scripts/python scripts/run_quant_replicates.py --tag smollm2-135m \
        --model HuggingFaceTB/SmolLM2-135M --replicates 24
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
QUANT_SRC = ROOT.parent / "quantization-accuracy-curves" / "src"
sys.path.insert(0, str(QUANT_SRC))
sys.path.insert(0, str(ROOT / "src"))
# The quantization kernels are the ggml library bundled in the llama-cpp-python wheel
# installed for the system Python. Only that DLL directory is needed, so a minimal
# module that points ``quantise._lib`` at it stands in for ``llama_cpp``. Adding the
# whole user site to the path would shadow packages in this environment.
import types  # noqa: E402

LLAMA_DIR = (
    Path.home() / "AppData" / "Roaming" / "Python" / "Python313" / "site-packages" / "llama_cpp"
)
_stub = types.ModuleType("llama_cpp")
_stub.__file__ = str(LLAMA_DIR / "__init__.py")
sys.modules.setdefault("llama_cpp", _stub)

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from quantization_accuracy_curves.evaluate import (  # noqa: E402
    _get_module,
    quantisable_layers,
)
from quantization_accuracy_curves.quantise import BY_NAME, quantise_dequantise  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from procedure_variance_coverage.gpu_eval import arc_correct, load_arc  # noqa: E402

WINDOW = 128
LEVELS_DEFAULT = ("Q4_K", "Q2_K")


def window_ids(tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids


def load_domains(tokenizer):
    """Token streams for the three calibration domains and the held out eval text."""
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_wiki = window_ids(tokenizer, "\n".join(t for t in wiki["train"]["text"] if t.strip()))
    test_text = "\n".join(t for t in wiki["test"]["text"] if t.strip())
    eval_ids = window_ids(tokenizer, test_text)

    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    code = window_ids(tokenizer, "\n\n".join(r["text"] + "\n" + r["code"] for r in mbpp))

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    chat_text = "\n\n".join(
        f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['response']}"
        for r in dolly.select(range(0, len(dolly), 3))
    )
    chat = window_ids(tokenizer, chat_text)
    return {"wiki": train_wiki, "code": code, "chat": chat}, eval_ids


def sample_windows(stream: torch.Tensor, n: int, rng: np.random.Generator) -> torch.Tensor:
    starts = rng.integers(0, len(stream) - WINDOW, size=n)
    return torch.stack([stream[s : s + WINDOW] for s in starts])


def collect_importance(model, names, windows: torch.Tensor, device, batch: int = 16):
    """Mean squared activation entering every quantisable layer, as an imatrix."""
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(name):
        def hook(_m, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            sq = (x * x).sum(dim=0)
            sums[name] = sums.get(name, 0) + sq
            counts[name] = counts.get(name, 0) + x.shape[0]

        return hook

    handles = [_get_module(model, n).register_forward_pre_hook(make_hook(n)) for n in names]
    try:
        with torch.no_grad():
            for i in range(0, len(windows), batch):
                model(windows[i : i + batch].to(device))
    finally:
        for h in handles:
            h.remove()
    return {n: (sums[n] / counts[n]).cpu().numpy().astype(np.float32) for n in names}


def quantise_parallel(work, base_weights, names, level, importance, pool) -> None:
    """Round trip every layer from the pristine weights, layers in parallel."""
    jobs = [(n, base_weights[n].cpu().numpy(), None if importance is None else importance[n])
            for n in names]
    recon = list(pool.map(lambda j: quantise_dequantise(j[1], level, j[2]), jobs))
    with torch.no_grad():
        for (n, _, _), w in zip(jobs, recon, strict=True):
            _get_module(work, n).weight.copy_(torch.from_numpy(w))


def restore(work, base_weights, names) -> None:
    with torch.no_grad():
        for n in names:
            _get_module(work, n).weight.copy_(base_weights[n])


@torch.no_grad()
def score_windows(base, work, eval_windows: torch.Tensor, device, batch: int = 8):
    """Per window KL(base || quantized) and top-1 flip rate."""
    kl = np.zeros(len(eval_windows), dtype=np.float32)
    flip = np.zeros(len(eval_windows), dtype=np.float32)
    for i in range(0, len(eval_windows), batch):
        x = eval_windows[i : i + batch].to(device)
        lb = torch.log_softmax(base(x).logits.float(), dim=-1)
        lq = torch.log_softmax(work(x).logits.float(), dim=-1)
        k = (lb.exp() * (lb - lq)).sum(dim=-1).mean(dim=-1)
        f = (lb.argmax(dim=-1) != lq.argmax(dim=-1)).float().mean(dim=-1)
        kl[i : i + len(x)] = k.cpu().numpy()
        flip[i : i + len(x)] = f.cpu().numpy()
        del lb, lq
    torch.cuda.empty_cache()
    return kl, flip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tag", default="smollm2-135m")
    ap.add_argument("--replicates", type=int, default=24)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--levels", default=",".join(LEVELS_DEFAULT))
    ap.add_argument("--n-eval-windows", type=int, default=500)
    ap.add_argument("--n-arc", type=int, default=500)
    ap.add_argument("--n-cal", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out) if args.out else ROOT / "data" / f"quant_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [BY_NAME[n] for n in args.levels.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    domains, eval_stream = load_domains(tokenizer)
    n_eval = min(args.n_eval_windows, len(eval_stream) // WINDOW)
    eval_windows = eval_stream[: n_eval * WINDOW].reshape(n_eval, WINDOW)
    arc_items = load_arc(tokenizer, args.n_arc)

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    work = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    names = quantisable_layers(base)
    base_weights = {n: _get_module(base, n).weight.detach().cpu().clone() for n in names}
    print(f"{args.model}: {len(names)} layers, {n_eval} eval windows, {len(arc_items)} arc items, "
          f"domains {({k: len(v) for k, v in domains.items()})}", flush=True)

    meta = {
        "model": args.model, "date": date.today().isoformat(),
        "device": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "torch": torch.__version__, "window": WINDOW, "n_eval_windows": n_eval,
        "n_arc": len(arc_items), "n_cal_windows": args.n_cal, "levels": args.levels,
        "replicate_seed_base": 200000, "eval_text": "wikitext-2-raw-v1 test",
        "calibration_domains": {"wiki": "wikitext-2 train", "code": "MBPP text and code",
                                "chat": "Dolly instruction and response"},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pool = ThreadPoolExecutor(max_workers=args.threads)

    def evaluate_current():
        kl, flip = score_windows(base, work, eval_windows, device)
        return kl, flip, arc_correct(work, arc_items, device)

    plain_path = out_dir / "plain.npz"
    if not plain_path.is_file():
        arrays = {}
        b_arc = arc_correct(base, arc_items, device)
        arrays["arc_base"] = b_arc
        for lv in levels:
            quantise_parallel(work, base_weights, names, lv, None, pool)
            kl, flip, arc = evaluate_current()
            arrays[f"kl_{lv.name}"] = kl
            arrays[f"flip_{lv.name}"] = flip
            arrays[f"arc_{lv.name}"] = arc
            restore(work, base_weights, names)
        np.savez_compressed(plain_path, **arrays)
        print("plain arm scored", flush=True)

    for r in range(args.start, args.start + args.replicates):
        path = out_dir / f"rep{r:03d}.npz"
        if path.is_file():
            print(f"replicate {r}: exists", flush=True)
            continue
        t0 = time.time()
        rng = np.random.default_rng(meta["replicate_seed_base"] + r)
        n3 = args.n_cal // 3
        mix = torch.cat([
            sample_windows(domains["wiki"], n3, rng),
            sample_windows(domains["code"], n3, rng),
            sample_windows(domains["chat"], args.n_cal - 2 * n3, rng),
        ])
        wiki = sample_windows(domains["wiki"], args.n_cal, rng)
        code = sample_windows(domains["code"], args.n_cal, rng)
        imp = {"mix": collect_importance(base, names, mix, device),
               "wiki": collect_importance(base, names, wiki, device),
               "code": collect_importance(base, names, code, device)}
        arrays = {}
        for lv in levels:
            for arm in ("mix", "wiki", "code"):
                quantise_parallel(work, base_weights, names, lv, imp[arm], pool)
                kl, flip, arc = evaluate_current()
                arrays[f"kl_{lv.name}_{arm}"] = kl
                arrays[f"flip_{lv.name}_{arm}"] = flip
                arrays[f"arc_{lv.name}_{arm}"] = arc
                restore(work, base_weights, names)
        np.savez_compressed(path, **arrays)
        k = arrays[f"kl_{levels[0].name}_mix"].mean()
        print(f"replicate {r}: {levels[0].name} mix mean KL {k:.5f}  {time.time() - t0:.0f}s",
              flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
