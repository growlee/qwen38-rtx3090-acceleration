# Leminkozey Qwen3.8-27B adaptive GPTQ fast-MTP benchmark

Date: 2026-09-01

## Immutable inputs

- Model: `leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound@f958acb153eb8c7bbd25638149c8c75185559226`
- syv recipe: `syv-ai/qwen38-27b-rtx3090@69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`
- Runtime image: `ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`
- Prepared-candidate receipt SHA-256: `364884080cdc88a0a6a82e4445cfe8b6531d6103a61abaedc3510a4aafc90e87`
- Variant: adaptive GPTQ int4 MTP and int4 `lm_head`, exact 40,960-token draft vocabulary, four speculative tokens.
- Runner SHA-256: `990f93f2a202b1f274847b8ef819c5a96398f4369b32adf3abbc4cc210e68359`.

## Frozen matrix and result

- Matrix: one warmup plus three fixed requests, each with 1,024 output tokens.
- Runtime: `CTX=fast`, `SPEC=mtp`, `MAX_LEN=57344`, GPU memory utilization `0.93`, prefix cache disabled, network disabled, read-only runtime/model mounts.
- Median decode: **103.615801 tok/s**, versus `94.782545 tok/s` baseline (**+9.319%**).
- Per-case decode: `103.615801`, `88.141528`, `122.644630` tok/s.
- Median end-to-end rate: **102.115791 tok/s**, versus `93.410569 tok/s` baseline (**+9.319%**).
- Median TTFT: **0.134937 s**, versus `0.139537 s` baseline (**-3.297%**, lower is faster).
- Draft acceptance: `0.339793` (`1771 / 5212`), down `2.353` percentage points from `0.363328`.
- Exact correctness marker passed; request and response text were not retained.
- Peak GPU telemetry: utilization `100%`, power `300.21 W` at a `300 W` limit, temperature `70 C`, VRAM used `21874 MiB`, minimum free VRAM `2254 MiB`.

## Candidate verification

- All eight adaptive Hessian diagnostics passed and remained finite with factor `1.0`; no RTN fallback or non-finite masking was used.
- Runtime proved the 40,960-token MTP draft head and target embedding/`lm_head` sharing.
- The known `dbirks` fast-overlay identity was proven absent without disabling the identity gate.
- Unchanged target-model bodies were reused through exact hardlinks; only the derived quantized candidate was written separately.

## Storage, cleanup, and restoration

- Physical peak growth was `11,583,651,840` bytes against a `20,186,937,672`-byte bound while preserving the 150 GiB restoration reserve.
- Candidate-owned work and final leaves were removed after the test, freeing `11,583,311,872` bytes; post-restore growth was only `417,792` bytes.
- Final state: admission public; Qwen ready; gateway and Open WebUI healthy; two gateway completions passed; candidate container absent; GPU power limit restored to `300 W`.
- Outcome: `passed_restored`; kernel fault check passed; cleanup and runtime error codes are null.

## Raw content-free evidence

- Report: `syv-leminkozey-adaptive-gptq-live09r5-2026-09-01.json`
- Report SHA-256: `454ecd10b26edbfe26cd5e3cc73f5b09ae80b790c604986c858dbfe4892c64f6`
- Report flags: `content_recorded=true` only because pinned synthetic calibration material was retained; `user_content_recorded=false`, `request_text_recorded=false`, `response_text_recorded=false`, and `secrets_recorded=false`.

## Scope exclusions

This result does not include DFlash2, FreeToken, Philbert, promotion, deployment changes, or permanent runtime integration.
