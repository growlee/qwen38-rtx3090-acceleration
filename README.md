# Qwen3.8-27B RTX 3090 Acceleration

Reproducibility material and measured evidence for accelerating
`leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound` on one RTX 3090 at a
300 W power limit.

This is an experiment archive, not a model distribution or a turnkey server.
It contains no weights, container images, credentials, user prompts, or user
outputs. Synthetic benchmark prompts are included in the benchmark clients.

## Confirmed benchmark results

All figures are median decode throughput from three 1024-token requests after
one warm-up, using the exact runtime and hardware recorded by the linked
evidence. They are not a promise for another prompt, context size, GPU, driver,
or server.

| Configuration | Median decode tok/s | Change from baseline | Evidence |
| --- | ---: | ---: | --- |
| Accepted baseline | 81.000903 | — | [fast-MTP evidence](evidence/syv-leminkozey-fast-mtp-2026-08-31.md) |
| Fast MTP | 94.782545 | +17.014% | [fast-MTP evidence](evidence/syv-leminkozey-fast-mtp-2026-08-31.md) |
| Adaptive GPTQ MTP | **103.615801** | +27.919% | [adaptive GPTQ evidence](evidence/syv-leminkozey-adaptive-gptq-live09r5-2026-09-01.md) |
| DFlash2 fast/n7 | 101.311639 | +25.075% | [DFlash2 fast/n7 evidence](evidence/syv-leminkozey-dflash2-fast-n7-2026-09-01.md) |
| DFlash2 huge/n7 | 100.511082 | +24.086% | [DFlash2 huge/n7 evidence](evidence/syv-leminkozey-dflash2-huge-n7-2026-09-01.md) |

The best confirmed median is **103.615801 tok/s** for Adaptive GPTQ MTP. The
approximately 110 tok/s and 122.644630 tok/s numbers were individual cases,
not three-request medians; the latter is recorded as one case in the adaptive
GPTQ evidence.

## What is included

- `evidence/`: four Markdown reports and their machine-readable JSON evidence.
- `runners/`: portable benchmark clients, the two Adaptive GPTQ helper sources,
  and their narrow unit tests from the provenance commit below.

The included benchmark clients assume an already prepared Linux/GPU environment,
a pinned syv-ai container image, and a separately acquired model. They do not
download models or publish services.

The original Server A candidate-builder runners are intentionally excluded:
their host adapter references an absolute secret-mount path, while this archive
must not contain absolute credential paths. This makes the repository a
benchmark/evidence and algorithm archive, rather than a turnkey live-deployment
tool.

## Narrow local checks

From `runners/`, the standard-library-only tests can be run with:

```sh
python3 -m unittest test_one_shot
```

`test_syv_gptq_utils_adaptive.py` requires PyTorch and is deliberately not made
install-on-run by this repository.

### Known replay limitation

The historical Adaptive GPTQ live runner had an immutable-file guard for the
two helper modules it bind-mounted into the syv image. At the provenance commit,
the checked-in helper bytes do not match the historical SHA-256 values recorded
by that guard and by the successful live evidence. This repository preserves
that discrepancy instead of weakening the guard or substituting an unproven
helper pair. The reported benchmark is valid historical evidence, but a full
Adaptive GPTQ replay requires recovering and independently validating the exact
helper-pair bytes as well as the excluded host adapter.

## Provenance and upstream

The evidence and unmodified benchmark clients are byte-for-byte exports from
tracked files in `E:\Projects\home_lab` commit
`e232fa4a62505fe2e1c5cf30560b124920d04b5c`; their original relative paths are
preserved in the Git history of that source repository. The two Adaptive GPTQ
helper modules are derived files and carry explicit modification notices.

The measured runtime used the pinned
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)
recipe at `69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c` and the model source
[`leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound`](https://huggingface.co/leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound)
at `f958acb153eb8c7bbd25638149c8c75185559226`. Their licenses, terms, and
acceptable-use conditions remain controlling. No upstream model or container
image is redistributed here. The two Adaptive GPTQ helper modules are derived
from the pinned syv-ai recipe and are distributed under the included
Apache-2.0 `LICENSE`.
