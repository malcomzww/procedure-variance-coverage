"""Item level scoring shared by the SFT and quantization runners.

Imported only by the scripts that actually train or quantize a model. The
statistics in ``stats.py`` and ``simulate.py`` never import this module, so the
tests and the continuous integration run without a deep learning stack.
"""

from __future__ import annotations

import numpy as np

POOL_SEED = 0


def load_arc(tokenizer, n_items: int):
    """ARC-Easy test items as ``(context ids, [choice ids], gold index)``."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").shuffle(seed=POOL_SEED)
    items = []
    for r in ds.select(range(min(n_items, len(ds)))):
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        ctx = tokenizer("Question: " + r["question"] + "\nAnswer:", add_special_tokens=False)[
            "input_ids"
        ]
        choices = [
            tokenizer(" " + t, add_special_tokens=False)["input_ids"] for t in r["choices"]["text"]
        ]
        items.append((ctx, choices, labels.index(r["answerKey"])))
    return items


def arc_correct(model, items, device, batch_size: int = 16) -> np.ndarray:
    """1 if the answer with the highest total log likelihood is the gold one."""
    import torch

    model.eval()
    flat = [
        (i, c, ctx + ch, len(ctx))
        for i, (ctx, chs, _) in enumerate(items)
        for c, ch in enumerate(chs)
    ]
    score = np.full((len(items), max(len(chs) for _, chs, _ in items)), -np.inf)
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            chunk = flat[start : start + batch_size]
            length = max(len(x[2]) for x in chunk)
            ids = torch.zeros((len(chunk), length), dtype=torch.long)
            cont = torch.zeros((len(chunk), length), dtype=torch.bool)
            for row, (_, _, seq, c0) in enumerate(chunk):
                ids[row, : len(seq)] = torch.tensor(seq)
                cont[row, c0 : len(seq)] = True
            ids, cont = ids.to(device), cont.to(device)
            logp = torch.log_softmax(model(input_ids=ids).logits[:, :-1].float(), dim=-1)
            tok = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            total = (tok * cont[:, 1:].float()).sum(dim=1)
            for row, (i, c, _, _) in enumerate(chunk):
                score[i, c] = float(total[row])
    torch.cuda.empty_cache()
    pred = score.argmax(axis=1)
    return (pred == np.array([g for _, _, g in items])).astype(np.int8)
