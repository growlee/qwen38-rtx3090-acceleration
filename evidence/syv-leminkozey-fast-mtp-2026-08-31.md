# Leminkozey Qwen3.8-27B fast-MTP benchmark

Date: 2026-08-31

## Immutable inputs

- Model: `leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound@f958acb153eb8c7bbd25638149c8c75185559226`
- syv recipe: `syv-ai/qwen38-27b-rtx3090@69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`
- Runtime image: `ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`
- Prepared-candidate receipt SHA-256: `33ef6b4ab8d6047892dc1c123d87bcbacd22c382e85e59426a191e437ee89521`
- Variant: int8 MTP, group size 128, generic shipped 40,960-token draft vocabulary; the draft vocabulary was not derived from exact fine-tune outputs.

## Frozen matrix and result

- Matrix: one warmup plus three fixed requests, each with 1,024 output tokens.
- Runtime: `CTX=fast`, `SPEC=mtp`, `MAX_LEN=57344`, GPU memory utilization `0.93`, prefix cache disabled, network disabled, read-only runtime/model mounts.
- Median decode: **94.782545 tok/s**.
- Per-case decode: `94.782545`, `91.062543`, `109.822995` tok/s.
- Median end-to-end rate: `93.410569 tok/s`; median end-to-end latency: `10.962357 s`; median TTFT: `0.139537 s`.
- Draft acceptance: `0.363328` (`1821 / 5012`).
- Accepted baseline: `81.000903 tok/s`, acceptance `0.414002`.
- Decode improvement: **+17.014%**; acceptance change: `-5.067` percentage points.
- Peak GPU telemetry: utilization `100%`, power `300.49 W` at a `300 W` limit, temperature `69 C`, VRAM used `23072 MiB`, minimum free VRAM `1056 MiB`.
- DFlash2 was not attempted because it was outside this bounded matrix.

## Safety and restoration

- The 150 GiB restoration reserve was preserved. Candidate preparation peaked at `2,357,456,896` bytes of growth against a 4 GiB bound.
- Final state: admission public; Qwen ready; gateway and Open WebUI healthy; two gateway completions passed; candidate container absent; GPU power limit restored to `300 W`.
- Two earlier invocations were invalidated before any benchmark case ran: one because the pre-existing Qwen service was not ready, and one because the adapter did not recognize the actual runtime draft-head evidence. The evidence matcher was corrected and covered by a regression test before the single valid matrix execution.

## Raw content-free evidence

- Report: `syv-leminkozey-fast-mtp-2026-08-31.json`
- Report SHA-256: `3553ffd24706edeb1b388aa75855a288b444e0ee7f3fe401cccab53448a8d645`
- Report flags: `content_recorded=false`, `request_text_recorded=false`, `response_text_recorded=false`, `secrets_recorded=false`.
