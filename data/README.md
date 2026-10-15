# Raw replicate data

Per item scores for every replicate of every pipeline. These are the raw files of the study. The
analysis in `scripts/analyze_real.py` reads only these files, so anyone can rerun it exactly.

Each folder holds one compressed `repNNN.npz` per replicate, a `meta.json` with the
configuration and hardware, and a file with the scores of the fixed arms.

## `sft_smollm2-135m/`

Naive versus block diagonal packing, SmolLM2-135M, one epoch, learning rate 2e-5, 500 training
examples per replicate. A replicate is a fresh draw of the training subset and a fresh batch
order, shared by both arms.

| file | contents |
|---|---|
| `base.npz` | `loss` per item of the untrained model, `arc` correctness, `valid` mask of scoreable Dolly items |
| `repNNN.npz` | `loss_naive`, `loss_block` per Dolly item (2000, `NaN` where not scoreable), `arc_naive`, `arc_block` (1000 binary), `draw` indices of the training subset, `info` JSON string with training losses and seconds |

## `quant_<model>/`

llama.cpp k-quant quantization at Q4_K and Q2_K for SmolLM2-135M, SmolLM2-360M and
Pythia-160M. A replicate is a fresh draw of calibration windows.

| file | contents |
|---|---|
| `plain.npz` | plain quantization without an importance matrix (deterministic), `kl_<level>`, `flip_<level>`, `arc_<level>`, and `arc_base` |
| `repNNN.npz` | for each level and arm in `mix`, `wiki`, `code`, the arrays `kl_<level>_<arm>`, `flip_<level>_<arm>` per evaluation window and `arc_<level>_<arm>` per question |

KL is the mean per token KL divergence from the unquantized model to the quantized one over a
128 token window. Flip rate is the share of positions where the top-1 prediction changes.
The evaluation text is the WikiText-2 test split. Calibration windows come from the WikiText-2
train split (`wiki`), MBPP text and code (`code`), and Dolly instructions and responses
(`chat`). The `mix` arm draws equally from all three.

## Hardware

Training, scoring and the quantization kernels ran on one laptop, an RTX 4060 with 8 GB for the
models and the CPU for the ggml kernels. Per item values can differ in the last bits on other
hardware. The committed results are rounded so they do not depend on this.
