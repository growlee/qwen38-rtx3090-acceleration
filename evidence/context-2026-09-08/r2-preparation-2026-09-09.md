# Context continuation r2: preparation

User authorized preparation and testing on2026-09-09. Fixed scope:257 untouched
coordinates plus one storage-aborted dispatch only after exact invalidation;
354 completed r1 answers, including the valid INT8 FAIL, are never repeated.

Fresh read-only Server A checks through canonical Remote SSH alias `server-a`
at10:57–11:00 UTC:

- Original r1 unit terminal failed; r2 directory absent.
- Retained r1 source, manifest and inherited controller bundle validated.
- Root available168220778496 bytes; >=154GiB staging/closure prerequisite.
- GPU1MiB,39C,300W limit; no inference owner.
- Available RAM29092308KiB; swap free7684860KiB.
- Public admission empty, serialized-99, no breaker, all AI profiles stopped.
- Original Qwen ID15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f,
  image sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0,
  model `/srv/ai-server/artifacts/extensions/qwen-syv-k4-4d4166b891489bfc/model`;
  unchanged, idle-stopped. Gateway/Open WebUI healthy. No generation sent.
- All12 r1 candidate/client IDs retained and stopped; OOMfalse. Interrupted
  Adaptive huge candidate exit137 and client exit3 match the recorded stop.
- Another agent's FACEIT resource client is running. Do not stop or modify it.
  This is not a claim of CPU/IO isolation; timestamps and resource samples must
  retain the concurrent-work qualification for timing comparisons.

Exact retained source hashes/sizes:
[r2-source-receipt-2026-09-09.json](r2-source-receipt-2026-09-09.json).
These are pre-preparation observations, not launch proof or current runtime.
Fresh safety checks remain required after package staging and before closure.
No remote files or services changed during these checks.

## Preparation and dispatch

The Remote SSH text-write command exceeded the Windows command-length bound
at the fifth file. Read-only inspection proved only the first four files had
been created; no controller was present or launched. Remaining files were
transferred in exclusive, offset-checked chunks and SHA-verified. The root-only
benchmarks parent permissions were preserved. The owned transfer directory
was verified, frozen root-owned/read-only and moved on the same filesystem.
No model, image or frozen-fixture copy was made.

The first source preflight exposed an audit/summary schema mismatch before
any runtime action. The minimal corrected package passed the same reviewer's
delta check and actual server source/selection preflight. Replaced staging
names were consumed only after establishing retained exact hardlinks to old
files; all old source content remains recoverable in preparation-history.
The source directory including retained history occupies630784 allocated bytes,
below16MiB; no user/model data was removed.

Dispatched once as `qwen-context-20260908-r2.service`, invocation
2828b0102ddc417c81c886af0e0fe554, observed MainPID4101930 active/running,
Restart=no. Final manifest89e8667d4c2031b6ce91fe0f73b206592ea9a6f4066b2e9328b4d570e209299e.
This proves controller launch, not completed inference or final restoration.

[Actual launch preflight](r2-launch-preflight-2026-09-09.json),
[dispatch and first process observation](r2-dispatch-2026-09-09.json),
[source correction receipt](r2-correction-receipt-2026-09-09.json).

## First completed case, intermediate only

Observed1788953597.9928548: the same invocation/MainPID remains active.
Adaptive huge completed context-version-start-t200000 with input200018,
client non-code verdict pass, output9 tokens. TTFT316.325791s;
decode17.972042tok/s; total316.826569s. The next case,
context-version-middle-t200000, was dispatched once.

This first restarted-profile sample is cold_continuation; live candidate logs
also recorded inference-time Triton compilation for
eagle_step_slot_mapping_metadata_kernel, _kvarn_fused_decode_stage1 and
_kvarn_build_packed_kv_kernel. No additional warmup or rerun was performed.
Nine output tokens are a short decode sample, not sustained throughput proof.
Remaining257 outcomes and final restoration were still unproven at this point.
[First captured response and live-handle observation](r2-first-result-2026-09-09.json).

## Adaptive huge continuation captured; matrix still running

Observed1788955204.8103619: all six selected Adaptive huge continuation
responses were captured and its candidate stopped. The controller advanced
to reconstructed-mtp-fast. The three version cases passed the client checks;
the three code cases remain pending_sandbox, not a final quality verdict.
No completed r1 answer was repeated.

At approximately200k input, TTFT ranged314.790316–316.325791s. Code outputs
were91–116 tokens, with decode31.082817–34.513519tok/s. Version outputs were
only8–10 tokens, so their17.972042–20.346307tok/s are short-sample timings.
All six belong to the restarted continuation window; only the first request
has the observed inference-time compilation qualification above.

The1834 resource samples recorded peak VRAM22000MiB, peak temperature76C,
maximum owned allocation1101824 bytes and minimum real root free167760191488
bytes. These are this profile's observed bounds, not future safety guarantees.
The shared-host/concurrent-work timing qualification remains applicable.

[Captured responses](r2-results/adaptive-mtp-huge.jsonl),
[resource samples](r2-results/adaptive-mtp-huge-resources.json),
[export hashes and newline normalization](r2-results/adaptive-mtp-huge-export.json).
Code execution grading and final public-access restoration remain pending.

## Reconstructed MTP fast captured; huge profile started

Observed1788956392.308838: reconstructed-mtp-fast captured all36 selected
responses. Its subsequent controller log records capture_finished/CAPTURED
and reconstructed-mtp-huge preparation/capture. The same unit invocation
remained active; the finite matrix and restoration were not yet complete.
All27 non-code client checks passed; nine code responses await sandbox grading.

Code-response measurements (three cases per context,98–158 output tokens):

| Target context | TTFT seconds, range | Decode tok/s, range |
| --- | --- | --- |
| 8192 | 6.854888–6.914094 | 173.625594–204.227337 |
| 32768 | 30.521674–30.534127 | 154.502453–167.667310 |
| 50000 | 49.781489–49.807778 | 138.006686–162.077601 |

These are short generated-code samples, not long-code qualification or a
controlled cross-profile speed verdict. The concurrent-work qualification
above remains applicable. No additional warmup or case repetition was added.

The1018 resource samples recorded peak VRAM21960MiB, peak temperature76C,
maximum owned allocation1802240 bytes and minimum real root free167823351808
bytes. Owned allocation is cumulative r2 usage observed during this profile.

[Captured responses](r2-results/reconstructed-mtp-fast.jsonl),
[resource samples](r2-results/reconstructed-mtp-fast-resources.json),
[export hashes and newline normalization](r2-results/reconstructed-mtp-fast-export.json).

## Reconstructed MTP huge captured; DFlash2 fast/n7 loading

Observed1788967633.9116964: all84 selected reconstructed-mtp-huge responses
were captured, with84 unique case IDs in the exact selection order. The same
unit invocation remained active. The controller log subsequently recorded
capture_finished/CAPTURED and dflash2-fast-n7 preparation/capture. A capture
phase marker does not itself prove that the next model has answered a request.

The raw file contains the one configured warmup,84 case_start records and84
complete case records. All63 non-code client checks passed;21 code responses
remain pending_sandbox. No extra warmup or repeated case was added. Final
code correctness, the remaining profiles and public restoration are pending.

Code-response measurements (three cases per context,92–158 output tokens):

| Target context | TTFT seconds, range | Decode tok/s, range |
| --- | --- | --- |
| 8192 | 7.030871–7.098885 | 108.162492–123.144051 |
| 32768 | 31.244218–31.262537 | 85.242397–89.633600 |
| 50000 | 50.746014–50.772581 | 71.813971–81.155049 |
| 96000 | 113.686411–113.735850 | 50.575173–57.794281 |
| 128000 | 166.807151–166.842216 | 44.193894–48.479640 |
| 160000 | 227.776075–227.846830 | 37.967028–40.032327 |
| 200000 | 314.430132–314.477680 | 32.194328–35.721088 |

These are short generated-code samples, not long-code qualification. Retain
the shared-host/concurrent-work timing qualification above. Do not rank final
quality from pending_sandbox or present short-output decode as sustained speed.

The10582 resource samples recorded peak VRAM22002MiB, peak temperature76C,
maximum cumulative r2 owned allocation2547712 bytes and minimum real root
free167789826048 bytes. These are observations during this profile, not
future guarantees or a claim about another agent's storage use.

Exports were read only after capture_finished. The resources file exceeded
the remote tool's1MiB stdout bound, so it was read in immutable SHA-checked
chunks without creating another remote file. Local raw bytes match the
remote SHA exactly. Local resources bytes match the remote SHA after removing
the single added trailing LF, explicitly recorded in the export receipt.

[Captured responses](r2-results/reconstructed-mtp-huge.jsonl),
[resource samples](r2-results/reconstructed-mtp-huge-resources.json),
[export hashes and newline normalization](r2-results/reconstructed-mtp-huge-export.json).

## DFlash2 fast/n7 captured; huge/n7 loading (historical checkpoint)

Observed1788968932.4715111: all36 selected dflash2-fast-n7 responses were
captured. The controller subsequently recorded capture_finished/CAPTURED and
dflash2-huge-n7 preparation/capture. The same unit invocation remains active;
this is not a completed matrix or final restoration claim.

The raw contains one configured warmup,36 starts and36 complete responses in
the exact selected order. All27 non-code client checks passed; nine code
responses remain pending_sandbox. No extra warmup or case repetition was added.

Code-response measurements (three cases per context,99–158 output tokens):

| Target context | TTFT seconds, range | Decode tok/s, range |
| --- | --- | --- |
| 8192 | 6.849606–6.889013 | 212.665715–263.818223 |
| 32768 | 30.402052–30.420592 | 202.378274–229.316545 |
| 50000 | 49.515003–49.536679 | 196.130885–214.265860 |

These are short generated-code samples, not sustained long-code performance
or a final quality ranking. Keep the concurrent-work qualification above.

The1006 resource samples recorded peak VRAM23860MiB, peak temperature75C,
maximum cumulative r2 owned allocation4141056 bytes and minimum real root
free167787257856 bytes. This profile used more VRAM than reconstructed MTP;
the observed bounds do not guarantee the remaining profile's capacity.

Export occurred after capture_finished. Raw bytes match the remote SHA.
Resources match after removing the single local added trailing LF, verified
against the full remote SHA and recorded in the receipt.

[Captured responses](r2-results/dflash2-fast-n7.jsonl),
[resource samples](r2-results/dflash2-fast-n7-resources.json),
[export hashes and newline normalization](r2-results/dflash2-fast-n7-export.json).

## Subsequent terminal capture event

At1788969208 the sole r2 capture controller stopped after DFlash2 huge/n7
encountered CUDA OOM on its first8192-target request. No repeat was dispatched.
The old restoration path also failed due to a reused start idempotency key;
one-shot recovery is tracked separately. This supersedes the active-unit
checkpoint above. See [terminal evidence and recovery](r2-terminal-2026-09-09.md).
