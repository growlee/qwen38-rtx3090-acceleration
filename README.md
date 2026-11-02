# Qwen3.8-27B RTX 3090 Acceleration

Reproducibility material and measured evidence for accelerating
`leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound` on one RTX 3090 at a
300 W power limit.

This is an experiment archive, not a model distribution or a turnkey server.
It contains no weights, container images, credentials, user prompts, or user
outputs. Synthetic benchmark prompts are included in the benchmark clients.

## Quality and context results — 2026-09-08

Start with the [consolidated results](evidence/quality-summary-2026-09-08.md):
seven measured configurations on the same frozen 200-task suite, speed/quality
trade-offs, configured context windows, and the exact historical Adaptive
candidate that remains BLOCKED. Useful 200k context has **not** been validated;
the longest tested prompt was 11,017 tokens.

Detailed reports:

- [Initial 60-task series](evidence/quality-results-2026-09-08.md).
- [Expanded 200-task controls](evidence/quality-expanded-results-2026-09-08.md).
- [INT8 and DFlash configurations](evidence/quality-remaining-results-2026-09-08.md).
- [Adaptive with new calibration](evidence/quality-adaptive-reconstruction-2026-09-08.md).
- [Existing roadmap and context audit](roadmaps/quality-degradation.md).

These reports record completed experiments, not a fresh runtime check or an
automatic production switch. Their weighted decode metric must not be compared
directly with the historical fixed-output median benchmark below. Full raw run
artifacts remain in the local evidence directories identified by each report;
they are not all embedded in this repository.

The quality client includes the frozen 60-task and 200-task suites. The local
`test_quality_suite.py` depends on a fixture in the separate `home_lab` checkout
and is not published as a portable test; its reported historical test run must
not be interpreted as a clone-independent replay command.

The expanded suite adapts selected HumanEval prompts, reference solutions and
literal test vectors from commit `6d43fb980f9fee3c892a914eda09951f772ad10d`.
Those portions retain their [OpenAI MIT license](runners/HumanEval-LICENSE).
The request wrappers and task selection are modified; this is not the official
HumanEval benchmark or score.

## Historical performance progression — through 2026-09-02

**Original accepted setup at 300 W: 37.621 tok/s. Best confirmed median:
103.615801 tok/s.** That is 2.754x the original result, an illustrative gain of
65.994801 tok/s or 175.420%. The earlier value around 33 tok/s was
**33.526 tok/s at 280 W**.

The confirmed measurements below are ordered from slowest to fastest:

| Order | Configuration | Power | Median decode | Meaning | Evidence |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | Original GGUF Q4_K_M + llama.cpp | 250 W | 27.273 tok/s | Original power-cap sweep | [original GPU evidence](evidence/phase2-qwen-gpu-300w-promotion-2026-08-23.md) |
| 2 | Original GGUF Q4_K_M + llama.cpp | 280 W | 33.526 tok/s | The remembered ~33 tok/s result | [original GPU evidence](evidence/phase2-qwen-gpu-300w-promotion-2026-08-23.md) |
| 3 | Original GGUF Q4_K_M + llama.cpp | 300 W | **37.621 tok/s** | Original accepted production setup | [original GPU evidence](evidence/phase2-qwen-gpu-300w-promotion-2026-08-23.md) |
| 4 | AutoRound W4A16 candidate baseline | 300 W | 81.000903 tok/s | Baseline of the later acceleration series | [fast-MTP evidence](evidence/syv-leminkozey-fast-mtp-2026-08-31.md) |
| 5 | Fast MTP | 300 W | 94.782545 tok/s | +17.014% within the later series | [fast-MTP evidence](evidence/syv-leminkozey-fast-mtp-2026-08-31.md) |
| 6 | DFlash2 huge/n7 | 300 W | 100.511082 tok/s | +24.086% within the later series | [DFlash2 huge/n7 evidence](evidence/syv-leminkozey-dflash2-huge-n7-2026-09-01.md) |
| 7 | DFlash2 fast/n7 | 300 W | 101.311639 tok/s | +25.075% within the later series | [DFlash2 fast/n7 evidence](evidence/syv-leminkozey-dflash2-fast-n7-2026-09-01.md) |
| 8 | Adaptive GPTQ MTP | 300 W | **103.615801 tok/s** | Best confirmed median; +27.919% within the later series | [adaptive GPTQ evidence](evidence/syv-leminkozey-adaptive-gptq-live09r5-2026-09-01.md) |

The first three rows used one warm-up and two measured 512-token completions.
The later series used one warm-up and three measured 1024-token completions,
and also changed the model packaging and runtime. Therefore the full 37.621 to
103.615801 progression is a historical before/after comparison, not a single
controlled A/B test. The controlled improvement inside the later series is
81.000903 to 103.615801 tok/s, or **+27.919%**.

The highest individual request was **122.644630 tok/s**, but it was one case,
not the stable headline result. The maximum confirmed median remains
**103.615801 tok/s** for that historical benchmark protocol.

## What is included

- `evidence/`: historical GPU/acceleration reports and JSON evidence, plus the
  September 8 quality reports and consolidated context findings.
- `runners/`: portable benchmark clients, the two Adaptive GPTQ helper sources,
  and their narrow unit tests from the provenance commit below.
- `server-a-2026-11-02/`: the Server A deployment (runtime image, compose, Studio bridge,
  weight identities) and its raw benchmark/experiment trees, collected when Unsloth Studio
  left Server A; secret-file paths redacted ([details](server-a-2026-11-02/README.md)).

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

The original historical evidence and unmodified benchmark clients are byte-for-byte exports from
tracked files in `E:\Projects\home_lab` commit
`e232fa4a62505fe2e1c5cf30560b124920d04b5c`; their original relative paths are
preserved in the Git history of that source repository. The two Adaptive GPTQ
helper modules are derived files and carry explicit modification notices.

The September 8 quality reports and roadmap are later additions with their own
run identities and evidence references; the original export claim above does
not apply to them.

The measured runtime used the pinned
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)
recipe at `69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c` and the model source
[`leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound`](https://huggingface.co/leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound)
at `f958acb153eb8c7bbd25638149c8c75185559226`. Their licenses, terms, and
acceptable-use conditions remain controlling. No upstream model or container
image is redistributed here. The two Adaptive GPTQ helper modules are derived
from the pinned syv-ai recipe and are distributed under the included
Apache-2.0 `LICENSE`.
