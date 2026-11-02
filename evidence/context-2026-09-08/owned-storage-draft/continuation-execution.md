# Context continuation r2 — draft, not deployed

User changed the time authority on 2026-09-08: extend beyond12 hours if needed
to finish the same612 coordinates. Do not interrupt r1 or modify its immutable
sources/timer. This is one additional finite window, not a scheduler/retry loop.
It can run only after r1 is terminal, original public service restoration is
proven, and r1's one-shot grading is complete. If r1 completes the matrix,
do not launch r2. No new model/profile/length/quality case is authorized here.

## Existing controls and remaining work

Reuse r1 manifest `c71c26c5cf3124347a021489d60148b102a5ad2c35ee4b1314e37ef8eafa36e6`
and the retained quality-remaining controller/admission/sandbox bundle. Source
and receipt SHA checks precede imports, selection and inference. Retained frozen
fixture is mounted read-only directly from r1; never retokenize/freeze it again
or stage a second fixture/model tree. All settings/requests/seeds/output limits
remain the same. Rootfs, GPU ownership, FreeToken checks, temperature79C stop,
300W limit, resource guard, RAM controls and exact-ID stop/restore are inherited.

The code is local-only. The first independent Sol review rejected the draft;
the six findings are addressed locally below and require the same reviewer's
follow-up before deployment. No second review pipeline is introduced.
Before deployment bind the FINAL r1 settings, frozen fixture, context report,
grading summary, all11 scores and every existing raw capture by full SHA in
`manifest.json.source_evidence`. Missing raw files must still be absent. Bind
the exact remaining selection in `selection.json` and its SHA in the manifest.
The manifest's exact local-source filenames are enumerated by `FILES` in
`continue_context.py`; do not change their meaning during staging.
The runbook filename is exactly `continuation-execution.md`, not r1's
`execution.md`. Every selected target payload is SHA/size-verified, each used
image (including the shared SYV client/restore image) inspected, selected GGUF
verified in full, and the DFlash drafter checked through `verify_draft()`.

`quality_context_remaining.py` holds started-but-unfinished dispatches; it
does not grant automatic retry. A deadline-interrupted capture requires an
exact invalidation entry: mode/case, reason
`deadline_interruption_no_complete_capture`, evidence note, matching global
and mode deadline error, and retained exact-ID client/candidate cleanup proof.
The new runner rechecks those IDs stopped live. A complete JSONL answer, valid
FAIL, pending_sandbox or a transport-error envelope cannot be selected again.
If interruption has a different cause or evidence is ambiguous, classify it
before deployment; do not add a generic retry or replace source results.

Only selected cases are tokenized/inferred. Previously recorded warmups are
not repeated. If a partially captured profile must load into a new process,
mark its new timings `cold_continuation=true` and keep the process/window break
visible in the final performance analysis. Prefix cache remains off. There is
no attempt to pretend a restarted model is the original uninterrupted session.

One additional window is sized from the exact remaining selection, not another
arbitrary12-hour cap. Its capture budget is5520s per remaining case (up to two
tokenizer calls plus generation at1800s each,120s overhead), plus1800s per
unrecorded warmup,1800s model discovery,600s load and300s cleanup per selected
profile, plus2700s for public preflight/controller overhead. Add2700s for
restoration. This conservative bound is NOT an ETA; healthy requests finish
much sooner and the controller exits when the finite selection ends. Existing
1800s generation timeout remains unchanged. Abnormal protocol/runtime stalls
remain blockers, not authority to rerun captured answers.
The exact-ID ownership scan and original Qwen
start +two authenticated gateway completions +public/ready checks are unchanged.
Before closure, require r1's ready/public/two-completion proof and matching
container/image/model; under the existing controller locks, verify fresh public
Qwen readiness with one authenticated non-thinking marker. If already idle,
start Qwen through its existing supervisor control. Stop it only AFTER admission
is proven closed, then rerun the original all-profiles-stopped/GPU-idle check.
The marker is service-readiness evidence, not a repeated matrix case. A failed
marker does not authorize closure or a blind retry.
The retained controller's finally is adapted only for a proven original Qwen:
if it is still ready, require every active GPU PID to belong to its exact
container rather than demanding GPU-idle before restoration. Other GPU owners
or unknown identities still fence restoration. A failed closure response is
resolved by a read-only phase query, never another close call. A failed stop
response is accepted only when stopped state is proven; otherwise the existing
finally performs restoration. If Qwen stayed ready, restoration does not send
a blind start. Before any closure, a later preflight failure preserves proven
public+ready and records only confirmed markers; it is not a successful
two-marker restoration and cannot pass the continuation grader's restore gate.
The proven-marker counter starts at0 and increments only after a successful
response. Lost start or marker responses write `public-preflight-failure.json`
with0 proven completions and explicit `start_outcome_unknown` or
`public_marker_outcome_unknown`; neither operation is retried. If preflight
fails before the inherited controller assigns production_before, its finally
does not fabricate a restore result; the separate failure receipt is retained.
Use an isolated transient `qwen-context-20260908-r2.service`, Restart=no,
KillMode=mixed, RuntimeMaxSec=derived window+7200 and TimeoutStopSec2700, only AFTER exact package
and launch review. No systemd timer/property edit of r1. No repeated window
launch is built into this controller. If a real safety blocker occurs, preserve
evidence and restore; valid semantic FAIL does not cancel the remaining cases.

After restoration, grade ONLY r2 captures in the existing CPU sandbox once.
Check real free space and test-owned allocations unconditionally immediately after restoration,
before grading and before writing final success, including non-code-only runs.
Preserve the restore report even when the post-restore disk check fails.
Valid JSON without a trailing LF is retained; only a JSONDecodeError-proven
incomplete final envelope may be excluded. Interior corruption fences the run.
Source r1 scores are read unchanged and merged, not reevaluated. Existing r2
score files fence automatic grading re-entry. The combined score file must
have exactly612 unique coordinates and carry the r1/r2 result provenance.
Combined accounting is not by itself proof that every request was executed:
inspect each BLOCKED reason and actual captures, including any time boundary.

## Peak disk bound for this additional window

This is a NEW bound derived after measuring the retained first window, not a
live relaxation/reset of r1's guard. All r1 output, references, invalid r0
artifacts, transfer parts, image layers, model trees and existing backups remain
charged to measured free space. No deletions, downloads or model copies.
Before staging and again before closure require >=154GiB free:150GiB floor
plus the4GiB test-owned allocation budget for r2. Keep the cumulative
initial/r1-final/r2-baseline/r2-final measurements in the report; do not present
two window budgets as a4GiB total-task limit. If that headroom is absent, stop
before preparation or closure; time authority does not authorize cleanup.

| Additional retained item | Worst-case bound |
| --- | ---: |
| New source/manifest/selection/evidence plus retained transfer overlap |16MiB|
| Model/image/fixture download, copy, unpack or mutable staging |0|
| At most11 raw outputs32MiB +stderr1MiB |363MiB|
| At most11 candidate logs4MiB |44MiB|
| At most11 GPU candidates +11 clients +153 code sandboxes,4MiB metadata each |700MiB|
| Samples, scores, merged612-score copy and reports |1024MiB|
| Restoration/log reserve above150GiB floor |512MiB|
| **Planned test allocation plus headroom** |**2659MiB <4096MiB**|

The extended sample bound is computed at at most one256-byte row per second
plus64MiB for reports/scores. The maximum full612 selection gives3430440s capture;
sample rows plus64MiB remain below1024MiB. Row size and the derived budget are
checked in code. Actual remaining selection is smaller. Longer time does not
relax the4GiB owned-allocation stop; background disk exhaustion still stops safely.

The final selection will reduce these maxima but never add cases. No tokenizer
or trusted-reference container is created again. Existing swap is preallocated;
bounded tmpfs is released on exact-ID stop. Compare actual filesystem growth
after source preparation, each profile and restoration, but do not attribute
that global delta to this test. The r1 investigation proved overlapping FACEIT
allocations, not an Unsloth cause. No other agent's files may be removed.

### Confirmed storage policy amendment, 2026-09-09

This amendment is for the future draft only. Frozen r1 and its historical
4GiB global-growth stop remain byte-for-byte unchanged. User confirmed that
another agent must be able to write independently while the test is bounded.

`context_storage.py` measures allocated blocks (`st_blocks*512`) in the run
directory and exact receipt-bound Docker container metadata/log directories
plus each container's active and init containerd snapshots. Inodes are counted
once; descendant symlinks are not followed. Shared read-only model/fixture/image
inputs are not copied or charged again. Missing paths, replaced root identities,
unknown storage layout, writable rootfs or non-read-only bind mounts fail closed.
This is sampled physical occupancy, not cumulative IO or a filesystem quota.

Checks retain512MiB of the existing restoration/log headroom: measured owned
data+512MiB<=4GiB and actual root free>=150GiB+512MiB before new work and during
capture. The512MiB is headroom, not an exact attribution of shared Docker DB,
journal or control-plane log writes. Those shared writes and every foreign
allocation always affect the real root-space check. No synthetic free-space
value, quota increase or suppression of measurement failure is used.
The planned per-item allowances above are conservative planning inputs, not
per-container disk quotas; actual total is checked after create and before
start. An external writer can still exhaust real free space between samples;
this does not promise storage isolation from the other agent.

Only two exact storage predicates in the already hash-verified inherited
controller are replaced in memory. GPU/RAM/temperature/time guards and original
stop/finally restoration stay unchanged. Candidate and client creation IDs are
registered before start; failed registration leaves the created stopped object
retained. The grader receives the same accounting object in-process and hooks
its existing sandbox create receipt before start. Standalone grader entry is
rejected rather than losing ownership accounting. Resource rows append owned
allocated bytes and real root free bytes at indexes8 and9; prior indexes and
the enforced256-byte row bound remain intact.

No matrix was resumed for this amendment. Final source manifest, interrupted
dispatch classification and remaining selection still require reconciliation
with finalized r1 evidence before any future launch; this patch does not
authorize a replay or make the existing draft launch-ready.

## Verification status

2026-09-09 owned-storage amendment:17 targeted offline methods now pass.
The first invocation was16PASS/1FAIL: the one failure was the test's comparison
of WindowsPath with PurePosixPath, not a storage-policy failure. Corrected only
that expected path type and reran only that method:PASS. Adapter tests cover
the two exact predicate replacements, unchanged inherited source hash and real
available-byte function, create/registration ordering and failure before start.
The grader integration preserves611 prior scores and evaluates one new mocked
code task, checks storage before grading, and fences re-entry.
The same independent Sol reviewer approved this exact policy candidate with
no materialP0/P1/P2. This is not approval to deploy or run incomplete r2 inputs.
Final adapter SHA2085436ed77ec5a9f64d61568bb612254cb58158027869fc036fa666ff914cfd
passed a read-only Linux probe across12 exact retained capture/client IDs:
35995648 allocated bytes including the r1 run root;168293060608 bytes real free.
Frozen r1 source and manifest SHA remain unchanged. Gateway and Open WebUI are
healthy; original Qwen remains its unchanged idle-stopped exit0/OOMfalse state.
No inference was sent. These counts are current selected-path occupancy, not
a reconstruction of all historical or reference-container allocations.

Earlier unchanged local checks: selected-case client6PASS, launch selection
contract6PASS, score merge5PASS. After the review changes: selection9PASS,
six new controller safety checks PASS, one updated file-level grader integration PASS
(mock sandbox: only one new evaluation,611 prior scores preserved by hash,
exact612 merge, valid final JSON without LF preserved, disk failure before any
grading, second entry fenced before another sandbox call).16 targeted checks,
each executed once for the changed candidate; unchanged suites not repeated.
After the follow-up review identified a ready-Qwen/old-finally mismatch, the
three affected handoff unit checks passed for the corrected candidate. One new
file-level test runs the retained controller's actual try/finally with IO doubles
for three failures: after marker before closure; lost closure response after
closed; lost stop response while Qwen remains ready. All preserve or restore
public+ready, never repeat close/stop, and do not report restore_unknown.
The later proof-counter correction passed one affected marker unit check and
one new file-level test with lost-start-after-applied and lost-marker-response:
public+ready remains observed in both IO doubles,0 proven completions and the
matching unknown flag are recorded, with no repeated start/marker. Earlier
unchanged file-level scenarios were not repeated.
These are offline checks, not independent
operational review or permission to deploy an incomplete manifest. Final
source-selection inputs cannot be frozen until r1 finishes. No server mutation
has been made by preparing this continuation draft.
