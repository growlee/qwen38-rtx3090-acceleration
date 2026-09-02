# Leminkozey Qwen3.8-27B DFlash2 huge/n7 benchmark

Date: 2026-09-01

Classification: valid negative result. The benchmark and restoration passed;
the hypothesis that this DFlash2 profile would outperform the accepted fast-MTP
`k=4` baseline was rejected.

## Immutable inputs

- Target: `leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound@f958acb153eb8c7bbd25638149c8c75185559226`.
- syv recipe: `syv-ai/qwen38-27b-rtx3090@69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`.
- Runtime image: `ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`.
- DFlash2 drafter: `syvai/Qwen3.8-27B-DFlash2-W4A16@4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.
- Target receipt SHA-256: `a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a`.
- Drafter receipt SHA-256: `4908dc8560ffa3c236c57adc579c3a3fdec3ed3df9a2c7b7ca8861356e788238`.
- Runner SHA-256: `166c25da157ed79502a494738132846a6a32577516bf00fea5311f2895f4becb`.
- Test SHA-256: `404833fced8c83b2e53efd1d2f2e02e29b432d6dd1677e368029a1e9ebafd80e`.

## Frozen matrix

- One warmup followed by three fixed requests with 1,024 output tokens each.
- Profile: `SPEC=dflash2`, `CTX=huge`, `DFLASH_MAX_LEN=245760`,
  `DFLASH_TOKENS=7`, `GPU_UTIL=0.97`, `PREFIX_CACHE=0`, `VISION=0`.
- Target and drafter mounts were read-only; runtime networking was disabled.
- Baseline: accepted adaptive GPTQ INT4 fast-MTP with four speculative tokens.

## Result

| Metric | fast-MTP `k=4` | DFlash2 huge/n7 | Change |
|---|---:|---:|---:|
| Median decode | 103.615801 tok/s | 100.511082 tok/s | -2.996% |
| Median end-to-end | 102.115791 tok/s | 98.556740 tok/s | -3.485% |
| Median TTFT | 0.134937 s | 0.120219 s | -10.907% |
| Draft acceptance | 0.339793 | 0.244826 | -9.497 pp |

- Per-case decode: `100.511082`, `83.732411`, and `118.915140` tok/s.
- Exact correctness marker passed; request and response text were not retained.
- DFlash2 runtime identity was proven from the loaded draft architecture,
  speculative configuration, and live process arguments.
- Candidate load time was `367.663` seconds.

The candidate improved time to first token, but lower acceptance made sustained
decode and end-to-end throughput slower than the accepted fast-MTP baseline.
This is a rejected performance hypothesis, not a harness or runtime failure.

## GPU and restoration evidence

- Peak utilization: `100%`.
- Peak power: `301.42 W` at a `300 W` power limit.
- Peak temperature: `70 C`.
- Peak VRAM use: `23,912 MiB`; minimum free VRAM: `216 MiB`.
- Kernel fault check passed.
- Outcome: `passed_restored`.
- Final state: admission public, Qwen ready, gateway and Open WebUI healthy,
  candidate absent, two gateway completions passed, power limit restored to
  `300 W`.
- The already downloaded drafter was reused; no model file was downloaded by
  this valid run.

## Evidence and scope

- Source report on Server A:
  `/srv/ai-server/benchmarks/experiments/syv-leminkozey-dflash2-goal02-166c25da.json`.
- Source report SHA-256:
  `ebc11eab274d6de13f84638042180e57554f7ca6cc66b6409a2e4fb82183a462`.
- Content flags: `content_recorded=false`, `request_text_recorded=false`,
  `response_text_recorded=false`, and `secrets_recorded=false`.
- The adjacent JSON is a compact content-free summary derived from that exact
  source report; it is not a byte-for-byte copy of the raw report.

This result applies only to the pinned `huge/n7` profile. It does not rule out
the newer DFlash2 `CTX=fast` profile, prefix caching, lookup drafting, or a
different speculative block length.
