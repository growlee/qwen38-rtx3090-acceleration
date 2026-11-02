# Context r1: storage safety stop, restoration blocked

Snapshot: 2026-09-09 00:28:15 UTC (03:28:15 MSK). This supersedes earlier live-running snapshots; the ELI5 pages remain historical cuts.

Restoration status superseded at 2026-09-09 01:07:01 UTC by the [authorized storage-accounting repair](storage-repair.md): original Qwen is ready and admission public. The capture stop and coverage below remain unchanged; no matrix cases were replayed.

Final grading is now complete in the [r1 final report](r1-final-results.md): 353 PASS, 1 FAIL, 258 BLOCKED over all 612 coordinates. Pending grading below describes this historical pre-repair snapshot only.

## Outcome

The matrix is NOT complete. Unit `qwen-context-20260908-r1.service` terminated at 00:24:11 UTC: MainPID=0, failed, ExecMainStatus=1.
The existing 4 GiB root-filesystem growth guard stopped capture during current Adaptive Huge 200k.
Original Qwen restoration was attempted once by the existing controller and rejected before runtime dispatch by the supervisor storage guard.
Qwen remains stopped; admission remains closed. No public readiness or successful restoration is claimed.

## Exact coverage (not final grading)

| Profile | Complete normal responses | Text PASS in raw | Code pending in raw | Error envelope | Not dispatched |
|---|---:|---:|---:|---:|---:|
| AutoRound fast | 36 | 27 | 9 | 0 | 0 |
| AutoRound huge | 84 | 63 | 21 | 0 | 0 |
| INT8 fast | 36 | 27 | 9 | 0 | 0 |
| INT8 huge | 84 | 63 | 21 | 0 | 0 |
| Current Adaptive fast | 36 | 27 | 9 | 0 | 0 |
| Current Adaptive huge | 78 | 60 | 18 | 1 | 5 |
| Reconstructed Adaptive fast | 0 | 0 | 0 | 0 | 36 |
| Reconstructed Adaptive huge | 0 | 0 | 0 | 0 | 84 |
| DFlash2 fast/n7 | 0 | 0 | 0 | 0 | 36 |
| DFlash2 huge/n7 | 0 | 0 | 0 | 0 | 84 |
| Original GGUF | 0 | 0 | 0 | 0 | 12 |
| Total | 354 | 267 | 87 | 1 | 257 |

All 612 coordinates are accounted for by 354 normal responses + 1 interrupted error envelope + 257 not dispatched.
This table is a capture inventory, not a substitute for canonical scores.
AutoRound's 30 code responses were already graded once on PC-A: 30/30 PASS, 220/220 vectors. They remain pending in immutable server raw.
The other 57 captured code responses are not graded.
Canonical server grading refused at its public-restoration precondition, before evaluating any code or writing scores. Do not bypass this precondition or rewrite the original failure report.

Adaptive Huge completed all six lower lengths through 160k and six text cases at 200k.
Interrupted case: `context-version-start-t200000`, actual input 200018, request SHA `5fd34e99ba0d8fda9bb7334005852019852ae7e3508545ecd03694d1da7833e7`.
It recorded `stream_incomplete`, empty content, no first token, elapsed 279.369051 s.
This is a storage-triggered infrastructure interruption, not a demonstrated quality FAIL or model capacity failure. Do not treat the incomplete 200k group as a 12-case speed median or a complete pass.
No generation was replayed; no additional long-output coding round was added.

## Two distinct storage constraints

1. Capture guard: baseline free 172198895616 B minus observed free 167902392320 B = 4296503296 B, above the frozen 4294967296 B growth bound. The 150 GiB free floor itself remained intact. Subsequent free space partially recovered; that does not invalidate or erase the one-shot safety stop. Attribution of the global-root growth remains unproven.
2. Restoration guard: idempotency key `q38-nvfp4-3985739-start-1` has a completed refusal, `operation_id: null`, `storage_guard_blocked`, reason `ai_budget_exceeded`. Projected AI usage 211.21988662611693 GiB exceeds configured 200 GiB; root free 156.0375747680664 GiB exceeds its separate 150 GiB floor.

Read-only inspection of deployed `/opt/homelab/ai-server/supervisor/service.py`, `HostStorageProbe.snapshot`, shows summation of `st_size` per path, without deduplicating hardlink inodes.
The 00:27:33 UTC audit measured:
- per-path logical bytes: 226795716825;
- unique-inode logical bytes: 126293072636;
- unique-inode physical allocation: 112964624384;
- repeated hardlink paths: 70.
Sparse allocation also contributes to logical-versus-physical differences. The figures are not interchangeable quotas.
This establishes why the supervisor refuses an otherwise spacious root; it does not authorize changing its contract.
The experiment's preflight checked root reserve/growth and stopped runtime, but did not detect this independent production startup quota. The restoration guarantee was therefore incomplete.

## Safety and next boundary

Fresh proof: all 16 r1-owned containers stopped, all OOMKilled=false, GPU compute PID list empty.
The original container/image/model identity is unchanged:
`15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f`,
SYV image `sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`,
`/srv/ai-server/artifacts/extensions/qwen-syv-k4-4d4166b891489bfc/model`.
At 00:28:15 UTC free=168251994112 B; no active experiment process, no repeated start, no deletion/download/copy of weights, no service restart, no safety-limit bypass.
The approved r2 continuation is NOT deployed: its required restored/graded r1 preconditions are false, and this was a storage stop rather than the reviewed deadline interruption.

An independently reviewed, explicitly authorized storage-accounting repair is required before attempting normal Qwen restoration again. Preserve the 150 GiB floor and existing artifacts; do not raise limits or change production silently. After repair, use existing restoration controls, prove two authenticated gateway completions and public admission, then finish canonical grading. Do not resume GPU experiments merely because the free-space reading recovered.

## Evidence

- [Original controller report](r1-window-report.json)
- [Context report](r1-context-window-report.json)
- [Controller log](r1-controller.log)
- [Immutable settings](r1-execution-settings.json)
- [Fresh stopped-state and rejected-start proof](r1-restoration-blocker.json)
- [Storage audit and final capture inventory](r1-storage-audit.json)
- [Adaptive Huge raw](adaptive-mtp-huge.jsonl)
- [Adaptive Huge resources](adaptive-mtp-huge-resources.json)
- [Export receipt with nine verified file identities](r1-stop-export-receipt.json)

Original server evidence is retained unchanged. JSON exports that gained one final LF are explicitly identified in the receipt; removing exactly that byte reproduces the server SHA. Raw JSONL and original window report are byte-identical. Files are local, uncommitted and unpushed.
