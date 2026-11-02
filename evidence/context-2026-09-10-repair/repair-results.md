# Qwen context: DFlash huge repair and remaining GGUF controls

## Scope and provenance

User authorized a minimal DFlash huge repair and continuation until tests finish
or safe completion is demonstrably impossible. This is not a production profile
promotion. Preserve the original DFlash OOM and all completed answers.

Completed new answers: 12 previously never-dispatched `original-gguf` controls at input target
8192, followed by 84 cases for a separately identified repaired DFlash huge
configuration through input target 200000. The original 612-coordinate matrix
and the repaired profile must remain distinguishable. No automatic retries.

Existing fixture SHA-256:
`2df1be2cff66b53ce57b3eefe3eed8be2d16c59d1e37b783341f176995689d94`.
Retained r1+r2 combined scores SHA-256:
`777d45c1eb2e495cc33cf1d9a031fa0eb5100dc69f6e2401aca87dc3de2a4af9`.
Existing 516 complete answers (515 PASS / 1 FAIL) are not regenerated or graded
again. The 3 new GGUF and 21 repaired-profile code answers were each graded once.

## Read-only preflight, 2026-09-10 Moscow / 2026-09-09 UTC

Canonical Remote SSH alias `server-a`; connection succeeded.
GPU: 1 MiB used of 24576 MiB, 36 C, 300 W power limit.
Root filesystem available: 167769202688 bytes at this snapshot.
Admission public, queue and ledger empty, epoch 317; no global/profile breaker.
Original Qwen is stopped after the recorded idle stop, with unchanged identity:

- Container `15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f`.
- Image `sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`.
- Model `/srv/ai-server/artifacts/extensions/qwen-syv-k4-4d4166b891489bfc/model`.

Gateway and Open WebUI healthy. Unrelated FACEIT services are running and must
not be touched. r2 remains failed under its original invocation; it was not
relaunched. The new remote output path was absent when checked:
`/srv/ai-server/benchmarks/context-20260910-repair`.

These are preparation observations, not repair acceptance or proof of a new
generation. Live preflight must be refreshed before service cutover.

## Completed result — 2026-09-10 03:54:16 UTC / 06:54:16 Moscow

- Original GGUF: all 12 remaining controls PASS; 3 code answers PASS, 22/22 behavior checks.
- Repaired DFlash huge: 84/84 responses captured, 83 PASS / 1 FAIL; all 21 code answers PASS, 154/154 behavior checks.
- All 12 cases at 200k PASS. No inference OOM occurred in the repaired profile.
- The original 612-coordinate matrix remains 527 PASS / 1 FAIL / 84 BLOCKED. Its original DFlash OOM is not overwritten.

The repair is confirmed for this frozen matrix: reduce fixed KV memory from
5261334938 to 4350000000 bytes and set max length to 204800. Actual startup
capacity is 211935 tokens. Target weights, drafter, image, KVarN dtype and n7
remain unchanged. This is a separately named experimental profile, not a
permanent production replacement or proof that the original OOM profile passed.

Exactly 21 preserved responses plus 63 newly completed responses form the
84-case union. No completed response was regenerated. All 21 code responses
were evaluated once after capture and service restoration. The sole quality
failure is `context-join-start-t32768`: input 32790, repeated `Register`,
256 output tokens, `finish_reason=length`. It preceded the backup, was not
rerun, and was not attributed to infrastructure load.

### Speed by context size

Medians use successful responses only, excluding the two documented
backup-overlap timings. Decode means generation after the first generated
token, not prompt processing. Code-only decode uses three answers per target.

| Input target, tokens | Quality PASS | First token, seconds | Decode, tokens/s | Code-only decode, tokens/s | Clean timing samples |
|---|---|---|---|---|---|
| 8192 | 12/12 | 7.1 | 171.0 | 203.8 | 12 |
| 32768 | 11/12 | 30.9 | 119.7 | 139.0 | 9 |
| 50000 | 12/12 | 50.0 | 89.4 | 105.0 | 12 |
| 96000 | 12/12 | 111.4 | 61.0 | 66.6 | 12 |
| 128000 | 12/12 | 162.5 | 50.1 | 61.0 | 12 |
| 160000 | 12/12 | 221.1 | 28.5 | 43.8 | 12 |
| 200000 | 12/12 | 305.0 | 23.9 | 40.2 | 12 |

These are short frozen answers and functions embedded in long inputs, not long
generated programs or a full coding-agent qualification. The 8k short-answer
rates are not comparable to the historical ~110 tokens/s 512/1024-token
benchmark. Output length and speculative acceptance affect these measurements.
First 200k fact-start was deliberately run cold before the other cases; the
remaining captures span the preserved and continuation windows. At target
200000, actual inputs were 200001–200027 tokens.

Sampled maximum VRAM in the continuation was 23338 MiB. The original
21-response window has no retained full resource trace, so no peak is claimed
for it. The timing table excludes `context-version-start-t32768` and
`context-version-middle-t32768` because their intervals overlap the backup;
their quality results remain included. The 32k quality FAIL is also excluded
from successful-response timing medians.

### Final restoration and evidence

Controller outcome `tests_complete`, exit 0. Final read-only check at
2026-09-10T03:56:15Z confirmed the exact original Qwen container/image/model,
healthy Qwen, gateway and Open WebUI, public admission epoch 329, empty queue
and ledger, and no active breakers. The restoration receipt includes two
thinking-disabled gateway completions. All 23 continuation-owned containers
(candidate, client and code sandboxes) are retained and stopped with exit 0.
Final free space was 167357145088 bytes; experiment-owned allocation after
grading was 3948544 bytes. No model downloads, deletions, backup changes or
permanent Qwen-profile changes were performed.

Exact raw responses, 84 scores, reports, grading receipt, live restoration
snapshot and computed metrics are in
`kv4350-continuation-terminal-evidence.json`. Full resource samples remain at
the exact remote path recorded there (976168 bytes); their SHA and per-case
aggregates are exported, not the full trace. Key SHA-256 values:

- Continuation raw 63: `0482a9f0d1957bfd67eb17c57a7b4db6e0bb5c232b82c820d89eb6aa167d529d`.
- Repaired scores 84: `3b4b2a8421c5128211abc0da3536cb64f4cfb8ec50a5e543e8dd5ec650fc4b0d`.
- Final context report: `4388f49dffc37042d6020f52de1d14e9a4bc7cd3cda6d76ea137d464ebadb561`.
- Grading summary: `81246b79f3d6c16319c7421c9b90cd77f99f115065e63a15c306399e4c7bc639`.

The original 612-score hash and preserved 21-response hash were reverified
unchanged after completion. Artifacts are saved in this local repository;
no Git commit or push was performed by this repair task.

## Earlier progress and repair history (historical snapshots)

The dated intermediate states below are retained for provenance. The completed
result above supersedes their pending or in-progress status statements.

| Profile / step | Confirmed result |
|---|---|
| Original GGUF, 12 remaining controls | 12 PASS; 3 code answers, 22/22 behavior checks |
| DFlash huge, KV 4.0 GB / maxlen 204800 | Startup capacity failure; no warmup or test request |
| DFlash huge, KV 4.35 GB / maxlen 204800 | Startup and first 200k inference succeeded; 21 responses preserved, external backup interrupted capture; 63 still needed |

The original matrix is 527 PASS / 1 FAIL / 84 BLOCKED; the original DFlash OOM
remains unchanged. Production Qwen and public admission were restored.

The KV4350 capture was interrupted by the existing Server B backup job at
00:37 UTC, not model OOM. See `backup-collision.md` and the exact source bytes
in `kv4350-storage-abort-evidence.json`. Two retained 32k timing samples
overlap backup load (`context-version-start-t32768` and
`context-version-middle-t32768`); their quality evidence is still retained.
No quality evaluator has run for these 21 responses yet.

The retained raw envelopes already contain one valid output-limit failure:
`context-join-start-t32768`, input 32790, repeated `Register`, 256 output
tokens, `finish_reason=length`, `error=finish_reason_length`. It ended at
1789000446.8097436 (before the backup), so it is not attributed to that load.
This valid failure will not be regenerated or tuned away. The other raw
statuses are 17 `pass` and 3 `pending_sandbox` code responses; pending code is
not a quality PASS until the sandbox evaluates it.

At 00:47:12 UTC the independently reviewed restore-only adapter completed:
same production Qwen container/image/model, two thinking-disabled markers,
empty public admission, 167338598400 bytes free. Exact restoration receipt
SHA `bd837831136de91466b46e4bbf0ccf28b332640c42fff413b33bf241a433f308`;
see `kv4350-restoration-evidence.json`. No backup settings or data were changed.
The next continuation uses only the missing 63 terminal responses, then grades
the exact preserved-21/new-63 union once. The partly dispatched 22nd case is
eligible because the storage guard terminated its client before any answer.

Continuation launched once with manifest
`fe7bea03234b74072f37b4a6f5ee8339dab57c4220360063bc7db466e4cfc0c3`,
invocation `c0a6c57b31aa4c41a19a3dc1a3adf2ca`. Fresh preflight verified
167337455616 bytes free, empty public admission, exact preserved prefix,
completed restoration, and a new unused output directory. The independent
review caught terminal-state assumptions before launch; the corrected
validator was checked against actual pinned raw data and frozen fixture.
See `kv4350-continuation-launch-receipt.json`. Capture and final grading are
still in progress; the valid existing quality FAIL remains preserved.

### First repair-window attempt

The first reviewed controller (`qwen-context-20260910-repair.service`, invocation
`de049eeccc634b7e8ad705a16e956fd7`) failed in the client import before any warmup
or matrix request. It restored the original Qwen and public queue successfully.
This is a proven harness failure, not a GGUF or repaired-DFlash quality result.
No models, images, caches, user data or unrelated services have been changed.

Runtime bundle manifest SHA-256:
`fc10b9dd1d5ced63b92104e16618f6aece18f851e262f0d8314b551b4e2af88b`.
Five adapter checks passed once (selection/order, manifest/AST, operation-key
uniqueness, exact parameter patch, capacity fence); five grading checks passed
once across two targeted invocations (four initial checks and one added
incomplete-dispatch check). No model-generated code was executed on PC-A.

The controller uses distinct per-instance operation keys for public preflight
and restoration, addressing the previously proven idempotency-key collision.
It accepts the unchanged original Qwen in normal ready or stopped/idle state;
the inherited managed public preflight establishes readiness before closure.
The exact process check uses `--kv-cache-memory=4000000000`, matching the
installed entrypoint's argument spelling.

Independent exact operational review: APPROVE. Staged package allocated 65536
bytes; all six files (five runtime files plus manifest) were verified remotely.
Root free before launch was 167768064000 bytes. Public admission was empty,
all GPU profiles stopped, no breakers, and the original Qwen identity unchanged.
See `launch-receipt.json`. One oversized transfer command failed locally with
`spawn ENAMETOOLONG`; read-only verification proved the remote path absent.
The same bytes were then transferred exclusively per file, without a temporary
copy or overwrite. No inference was attempted by that failed transfer.

### Client import failure and repair

`ModuleNotFoundError: No module named 'quality_context_remaining'` occurred at
client import line 13. The existing client mounts expose r1 and old sources,
not the r2 helper module. The imported helper is only a two-line assertion
function; the narrow correction is to define it in the client, not broaden
container mounts. A separately named immutable client-fix bundle is being
prepared; the failed attempt and both stopped containers remain retained.

The GGUF JSONL is exactly zero bytes (SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`),
and repaired DFlash was not attempted. The final report proves the original
container/image/model, two non-thinking gateway completions, public admission,
and no restoration error. Artifact bytes and hashes are preserved in
`failed-client-import-evidence.json`; embedded `utf8` values retain exact
source contents without newline-normalization ambiguity.

The fixed bundle manifest is
`8f8bfc60b8a84ffa6a4c2e0b30d1ad943b2cd1693445811398e7b9b2f7630571`.
Its client SHA-256 is
`6fbfabccf70cf7ca9ba3d016d31f2ede69e8357134eba9e907148ad824abf6bf`.
One targeted import-isolation check passed on Server A: root `python3 -B`
executed the trusted client definitions with `__name__` different from
`__main__` and `sys.path` containing only the retained r1 directory, old
`sources` directory, and standard-library paths (no r2). Output:
`CLIENTFIX_IMPORT_ISOLATION_PASS`, exit 0. The first unprivileged read had
returned PermissionDenied on root-owned sources; the sudo check addressed
that access boundary without launching a model or request. AST and manifest
identity checks also passed; previous test suites were not repeated.

The same independent reviewer approved the exact client-fix delta. One-shot
unit `qwen-context-20260910-repair-clientfix.service` was launched with invocation
`b3189f1f4f8f422ea5372764e38c2e06`. All remote bundle/dependency hashes and the
previous harness-failure proof were checked before launch. Package allocation
is 69632 bytes; root free 167765917696 bytes. Qwen was ready, public admission
empty at epoch 320, all other GPU profiles stopped, and no breakers open.
See `clientfix-launch-receipt.json`.

### GGUF complete; 4 GB DFlash capacity failure

The client-fix window terminated with `candidate_exit_before_ready` on DFlash,
then restored the original Qwen identity, two non-thinking gateway markers,
and public admission without error. All 12 GGUF controls passed, including
three code answers graded once (22/22 executable behavior checks passed).
Their median post-first-token generation
speed was 40.8777 tokens/s (range 37.1461–41.5317); median first-token delay
7.4911 s; sampled peak VRAM 16730 MiB. These are short frozen answers at input
target 8192, not a standalone sustained throughput benchmark.

The original 612-coordinate matrix is now 527 PASS / 1 FAIL / 84 BLOCKED.
The remaining original-profile BLOCKED coordinates belong to the original
DFlash huge failure and are not overwritten by repair tuning. Combined score
SHA-256: `4f47916f1f0444478cc408d2a0610c2781777fdbfd1579f6bd83d589e320391e`.
Exact raw GGUF responses, scores, resource samples, report and source hashes
are retained in `clientfix-terminal-evidence.json`.

DFlash with KV_MEM=4000000000 failed before readiness or a client launch:
the runtime required 3.91 GiB KV memory for max length 204800, but had 3.73 GiB;
estimated supported maximum was 191488. No DFlash raw capture file exists.
This is a valid startup-capacity failure of that repaired profile, not OOM
and not a quality verdict. See `kv4gb-capacity-failure.json`.

Next repair changes only the fixed KV pool to 4350000000 bytes, with max
length still 204800. The installed hybrid allocator rounds memory by groups
and pages, so the earlier linear estimate based on the old logical KV-token
metric was not reliable. The selected pool leaves 911334938 bytes (0.849 GiB)
more headroom than the original 5261334938-byte pool. Actual startup capacity
and real 200k inference still require validation. The next selection contains
only 84 new-profile DFlash cases; GGUF is neither regenerated nor graded again.
The 84-only grader's targeted integration test passed once (0.085 s): exact
case order, no GGUF output or evaluator calls, original 612 rows unchanged,
and a second grading attempt rejected. Grader SHA-256:
`411a87e000c81d2a22d6d8471d91750df9d495124a07b33a4a0d6fdb765197d6`.

KV4350 bundle sealed with manifest
`d422861a845fb35a200e70b9fc0c10bbf1a7d0df76e7b1ab6c383ebc5c783afd`.
The existing independent reviewer approved this exact delta. Its five runtime
files plus manifest were staged exclusively under
`/srv/ai-server/benchmarks/context-20260910-repair-kv4350`.
Another active task's FACEIT API rollout and closure finished at
2026-09-10T00:16:11Z before benchmark launch. Fresh preflight verified the
sealed bundle, prior terminal outcomes, empty public admission, all AI profiles
stopped without breakers, and 167340847104 bytes free (57344 allocated bundle
bytes). The unit was launched once, invocation
`bf113c51638a4151a9215db16b11e139`, and was active at 00:17:11Z.
See `kv4350-launch-receipt.json`. At 00:25:01Z the runtime reported
`GPU KV cache size: 211,935 tokens` and maximum concurrency `1.03x` for
204800 tokens per request. Startup passed the capacity gate. The first real
case `context-fact-start-t200000` completed with 200006 input tokens and
15 output tokens: TTFT 306.691574 s, total 307.667648 s, decode 15.367692 tok/s.
No inference OOM occurred. This first cold 200k response proves execution, not
yet final quality grading or sustained decode performance; the full matrix
remains in progress.

## Repair hypothesis selected before inference

The stopped candidate's startup log specifies a fixed KV allocation of
5261334938 bytes and says this bypasses memory profiling / ignores the
gpu-memory-utilization setting. Therefore lowering `GPU_UTIL` alone is not
the selected repair. Original runtime cache capacity was reported as 273151
tokens, with almost no working VRAM left after startup.

The installed KVarN implementation failed inside `_batched_flush` at
`K_rot.permute(0, 2, 3, 1).reshape(nB * Hk, D, G)`. The preceding float staging
and subsequent transforms require temporary buffers. The retained exception
does not state the requested allocation size, so the precise transient peak
is unknown. This supports reducing the fixed cache allocation before changing
attention code or installing another runtime.

Selected candidate: `dflash2-huge-n7-kv4gb-204800`.
Only `KV_MEM=4000000000` and `DFLASH_MAX_LEN=204800` change; target, drafter,
image, KVarN dtype, n7, GPU_UTIL and other inherited settings remain unchanged.
The installed entrypoint supports `KV_MEM` as `--kv-cache-memory`.
The cache reduction is 1261334938 bytes (~1.17 GiB). Proportional extrapolation
suggests roughly 207720 cache tokens, but hybrid allocation/alignment means
this is only a planning estimate. Actual startup must prove capacity at least
204800 before test dispatch, sufficient for the retained 200k inputs and caps.

Prefer the standard `context-fact-start-t200000` as the first of the 84 repaired
cases, then the other 83 once. This is not an additional smoke/retry. Record
the changed order and cold first request when interpreting timing comparisons.
Old OOM remains a valid result of the original profile, regardless of repair.
