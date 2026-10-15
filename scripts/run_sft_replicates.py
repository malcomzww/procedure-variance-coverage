"""Run replicated SFT comparisons of naive versus block diagonal packing.

This is the expensive half of the SFT arm. It trains the two arms once per
replicate and writes one compressed file per replicate holding the per item
scores on a large evaluation pool. ``scripts/analyze_real.py`` reads those files.

A replicate is a fresh draw of the training subset together with a fresh batch
and packing order. Both arms inside one replicate see the same draw and the same
order, so the difference is paired within the replicate, and only the attention
mask differs between them. Everything else is held fixed, including the base
weights, the optimiser and the evaluation items.

Two item pools are scored for every trained model.

* A Dolly held out pool, scored by per item mean completion loss. Continuous.
* The ARC-Easy test questions, scored by per item correctness of the highest
  likelihood answer. Binary.

The pretrained weights and the packing and masking code come from the sibling
checkout ``sft-loss-masking-and-packing``. Evaluation is unpacked with a plain
causal mask for both arms, as in that repo, so the eval does not inherit the
arm's training mask.

Run (CUDA environment, from the repo root)::

    .venv/Scripts/python scripts/run_sft_replicates.py --replicates 30
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SFT_SRC = ROOT.parent / "sft-loss-masking-and-packing" / "src"
sys.path.insert(0, str(SFT_SRC))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sft_loss_masking_and_packing.dataset import Record, to_example  # noqa: E402
from sft_loss_masking_and_packing.packing import pack_examples  # noqa: E402
from sft_loss_masking_and_packing.train import (  # noqa: E402
    build_attention_mask,
    build_position_ids,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from procedure_variance_coverage.gpu_eval import arc_correct, load_arc  # noqa: E402

SEQ_LEN = 512
LR = 2e-5
POOL_SEED = 0
MAX_CHARS = 1200


def load_pools(n_eval_pool: int, tokenizer):
    """Dolly split into a fixed evaluation pool and a training pool."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.filter(
        lambda r: len(r["instruction"]) + len(r["context"]) + len(r["response"]) <= MAX_CHARS
    )
    ds = ds.shuffle(seed=POOL_SEED)
    records = [
        Record(instruction=r["instruction"], context=r["context"], response=r["response"])
        for r in ds
    ]
    eval_records, train_records = records[:n_eval_pool], records[n_eval_pool:]
    eval_examples = [to_example(r, tokenizer, max_length=SEQ_LEN) for r in eval_records]
    valid = np.array([len(e.input_ids) - e.completion_start >= 2 for e in eval_examples])
    return eval_examples, valid, train_records


@torch.no_grad()
def per_item_loss(model, examples, valid, device, max_tokens: int = 1536) -> np.ndarray:
    """Mean completion cross entropy per example, unpacked, plain causal mask."""
    model.eval()
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].input_ids))
    out = np.full(len(examples), np.nan, dtype=np.float32)
    batch: list[int] = []
    longest = 0

    def flush():
        nonlocal batch, longest
        if not batch:
            return
        length = longest
        ids = torch.zeros((len(batch), length), dtype=torch.long)
        keep = torch.zeros((len(batch), length), dtype=torch.bool)
        for row, idx in enumerate(batch):
            e = examples[idx]
            ids[row, : len(e.input_ids)] = torch.tensor(e.input_ids)
            keep[row, e.completion_start : len(e.input_ids)] = True
        ids, keep = ids.to(device), keep.to(device)
        logits = model(input_ids=ids).logits[:, :-1].float()
        target = ids[:, 1:]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
        ).reshape(target.shape)
        mask = keep[:, 1:].float()
        mean = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        for row, idx in enumerate(batch):
            out[idx] = float(mean[row])
        batch, longest = [], 0

    for idx in order:
        if not valid[idx]:
            continue
        n = len(examples[idx].input_ids)
        if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
            flush()
        batch.append(idx)
        longest = max(longest, n)
    flush()
    torch.cuda.empty_cache()
    return out


def train_one(model, batches, *, correct_attention: bool, device) -> tuple[float, float]:
    """One epoch, batch of one packed sequence, as in the sibling repo."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    first = last = math.nan
    for step, b in enumerate(batches):
        mask = build_attention_mask(b.document_ids, correct=correct_attention).to(device)
        pos = build_position_ids(b.document_ids).unsqueeze(0).to(device)
        out = model(
            input_ids=b.input_ids.unsqueeze(0).to(device),
            attention_mask=mask,
            position_ids=pos,
            labels=b.labels.unsqueeze(0).to(device),
        )
        if not torch.isfinite(out.loss):
            raise RuntimeError(f"non finite loss at step {step}")
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 0:
            first = float(out.loss.detach())
        last = float(out.loss.detach())
    del opt
    torch.cuda.empty_cache()
    return first, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tag", default="smollm2-135m")
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-eval-pool", type=int, default=2000)
    ap.add_argument("--arc-n", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) if args.out else ROOT / "data" / f"sft_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eval_examples, valid, train_records = load_pools(args.n_eval_pool, tokenizer)
    arc_items = load_arc(tokenizer, args.arc_n)
    print(f"device {device}  eval pool {int(valid.sum())}/{len(valid)}  "
          f"train pool {len(train_records)}  arc items {len(arc_items)}", flush=True)

    cache: dict[int, object] = {}

    def example_for(i: int):
        if i not in cache:
            cache[i] = to_example(train_records[i], tokenizer, max_length=SEQ_LEN)
        return cache[i]

    meta = {
        "model": args.model,
        "date": date.today().isoformat(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seq_len": SEQ_LEN,
        "lr": LR,
        "epochs": 1,
        "n_train": args.n_train,
        "n_eval_pool": int(valid.sum()),
        "n_arc": len(arc_items),
        "pool_seed": POOL_SEED,
        "replicate_seed_base": 100000,
        "arms": {
            "naive": "loss masked, causal packed mask",
            "block": "loss masked, block diagonal",
        },
        "valid_eval_mask_sum": int(valid.sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_path = out_dir / "base.npz"
    if not base_path.is_file():
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
        np.savez_compressed(
            base_path,
            loss=per_item_loss(base, eval_examples, valid, device),
            arc=arc_correct(base, arc_items, device),
            valid=valid,
        )
        del base
        gc.collect()
        torch.cuda.empty_cache()
        print("base model scored", flush=True)

    for r in range(args.start, args.start + args.replicates):
        path = out_dir / f"rep{r:03d}.npz"
        if path.is_file():
            print(f"replicate {r}: exists", flush=True)
            continue
        t0 = time.time()
        rng = np.random.default_rng(meta["replicate_seed_base"] + r)
        draw = rng.choice(len(train_records), size=args.n_train, replace=False)
        examples = [example_for(int(i)) for i in draw]
        packed = pack_examples(
            examples, seq_len=SEQ_LEN, pad_token_id=tokenizer.eos_token_id, mask_prompt=True
        )
        result: dict[str, np.ndarray] = {}
        info: dict[str, float] = {"n_packed": float(len(packed))}
        for arm, correct in (("naive", False), ("block", True)):
            torch.manual_seed(r)
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
            first, last = train_one(model, packed, correct_attention=correct, device=device)
            result[f"loss_{arm}"] = per_item_loss(model, eval_examples, valid, device)
            result[f"arc_{arm}"] = arc_correct(model, arc_items, device)
            info[f"train_first_{arm}"], info[f"train_last_{arm}"] = first, last
            del model
            gc.collect()
            torch.cuda.empty_cache()
        info["seconds"] = time.time() - t0
        np.savez_compressed(path, draw=draw.astype(np.int32), **result,
                            info=np.array(json.dumps(info)))
        d = np.nanmean(result["loss_naive"] - result["loss_block"])
        print(f"replicate {r}: mean loss naive-block {d:+.4f}  arc "
              f"{result['arc_naive'].mean():.3f}/{result['arc_block'].mean():.3f}  "
              f"{info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
