# Leminkozey Qwen3.8-27B DFlash2 fast/n7 benchmark

Date: 2026-09-01

Classification: valid negative result. The benchmark and restoration passed;
the hypothesis that the pinned DFlash2 `fast/n7` profile would outperform the
accepted fast-MTP `k=4` baseline by at least 5% on both decode and end-to-end
throughput was rejected.

## Immutable inputs

- Target: `leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound@f958acb153eb8c7bbd25638149c8c75185559226`.
- syv recipe: `syv-ai/qwen38-27b-rtx3090@69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`.
- Runtime image: `ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`.
- DFlash2 drafter: `syvai/Qwen3.8-27B-DFlash2-W4A16@4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.
- Target receipt SHA-256: `a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a`.
- Drafter receipt SHA-256: `4908dc8560ffa3c236c57adc579c3a3fdec3ed3df9a2c7b7ca8861356e788238`.
- Runner SHA-256: `c4071aaa9bb7ad6ec60e548fdb230101420073d3a851498e5ce42c2ef82a487e`.
- Test SHA-256: `7e5c24c7bf3daba9ff8b160c2f22c48f4971211cc608bf4572330eb2bd8b51d6`.

## Frozen matrix

- One warmup followed by three fixed requests with 1,024 output tokens each.
- Every measured request returned exactly 1,024 completion tokens with
  `finish_reason=length`.
- Profile: `SPEC=dflash2`, `CTX=fast`, `DFLASH_MAX_LEN=65536`,
  `DFLASH_TOKENS=7`, `GPU_UTIL=0.93`, `PREFIX_CACHE=0`, `VISION=0`.
- Target and drafter mounts were read-only; runtime networking was disabled.
- DFlash2 identity was proven from `DFlash2DraftModel`, the exact live process
  arguments, and non-zero speculative counters.
- Baseline: accepted adaptive GPTQ INT4 fast-MTP with four speculative tokens,
  report SHA-256
  `454ecd10b26edbfe26cd5e3cc73f5b09ae80b790c604986c858dbfe4892c64f6`.

## Result

| Metric | fast-MTP `k=4` | DFlash2 fast/n7 | Change |
|---|---:|---:|---:|
| Median decode | 103.615801 tok/s | 101.311639 tok/s | -2.224% |
| Median end-to-end | 102.115791 tok/s | 99.314940 tok/s | -2.743% |
| Median TTFT | 0.134937 s | 0.137474 s | +1.880% |
| Draft acceptance | 0.339793 | 0.244736 | -9.506 pp |

- Per-case decode: `101.311639`, `85.725069`, and `148.269582` tok/s.
- DFlash2 accepted `1,941` of `7,931` drafted tokens.
- Exact correctness marker passed; request and response text were not retained.
- Candidate load time was `281.006` seconds.

The candidate did not meet the required 5% decode and end-to-end improvement.
It was slower on both throughput measures and had lower draft acceptance. This
is a rejected performance hypothesis, not a harness or runtime failure.

## GPU and restoration evidence

- Peak utilization: `100%`.
- Peak power: `300.49 W` at a `300 W` power limit.
- Peak temperature: `70 C`.
- Peak VRAM use: `23,400 MiB`; minimum free VRAM: `728 MiB`.
- Kernel fault check passed.
- Outcome: `passed_restored`.
- Final state: admission public with an empty queue and ledger, Qwen ready,
  gateway and Open WebUI healthy, candidate absent, two gateway completions
  passed, and power limit restored to `300 W`.
- The already verified target and drafter were reused; no model was downloaded.
- The exact temporary runner bundle was removed after final live verification.

## Evidence and scope

- Source report on Server A:
  `/srv/ai-server/benchmarks/experiments/syv-leminkozey-dflash2-fast-n7-c4071aaa.json`.
- Source report SHA-256:
  `d6254e24be80d43d6d8b3af1b8e9f3a2e7cda7249e35460f6d72a86477ab2dcc`.
- Targeted runner tests: `23/23` passed; Python bytecode compilation passed.
- Content flags: `content_recorded=false`, `request_text_recorded=false`,
  `response_text_recorded=false`, and `secrets_recorded=false`.
- The adjacent JSON is a compact content-free summary derived from that exact
  source report; it is not a byte-for-byte copy of the raw report.

This result applies only to the exact pinned `fast/n7`, no-prefix-cache profile.
It does not rule out prefix caching, lookup drafting, a different speculative
block length, another model/runtime revision, or a workload dominated by
repeated long-context prefill.
