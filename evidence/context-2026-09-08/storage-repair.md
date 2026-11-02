# Authorized storage-accounting repair and Qwen restoration

Verified snapshot: 2026-09-09 01:07:01 UTC (04:07 MSK).

Subsequent [final grading report](r1-final-results.md) completes the retained-answer assessment without resuming GPU capture. Pending-grading statements below describe the repair-time snapshot.

## Result

Original Qwen, gateway and Open WebUI are running and healthy. Two authenticated
gateway completion checks succeeded. Admission is public; queue and active ledger
are empty, Qwen is ready, and no restart/reconciliation or queue error remains.
Browser playback and image/TTS generation were not tested in this repair.

The supervisor previously counted each hardlink path independently, reporting
about 211.22 GiB against its 200 GiB AI quota. The narrow repair deduplicates
regular files by `(st_dev, st_ino)` and retains logical `st_size` accounting.
Distinct copies still count separately; sparse files retain their logical size.
Symlink handling and fail-closed walk errors are unchanged. This is not physical
block accounting. The repaired reading is 117.61958956345916 GiB.

Neither the 200 GiB quota nor the 150 GiB free-space floor was changed.
Final free space: 168313749504 bytes (about 156.76 GiB).

## Change and bounded verification

- User explicitly authorized this production repair and the required supervisor restart.
- Exactly one independent Sol critic reviewed the candidate; after material corrections,
  the final candidate was approved for repair and restoration only, not matrix continuation.
- Four scoped storage probe/guard tests passed once on PC-A, including hardlink versus
  distinct-copy accounting, sparse logical size, existing symlink behavior and guard limits.
- Command: `wsl -d DockerEngine --cd /mnt/e/Projects/home_lab/projects/ai-server python3 -B -m unittest supervisor.tests.test_service.HostStorageProbeTests supervisor.tests.test_core.StorageGuardTests -v`.
- Two offline fault checks proved that a lost mutation reply is resolved read-only:
  proven application is accepted; unresolved application is fenced. Neither retries the mutation.
- Only the exact deployed service source was transformed; unrelated local dirty changes
  were not deployed. Backups were read back and synced before atomic replacement.
- Supervisor restarted once, PID 783991 to 2448831. Existing controls reconciled the
  empty ledger, started the original Qwen and reopened admission after the two checks.
- Transient unit `qwen-context-storage-repair-20260909.service` completed with exit 0.
- An earlier installer preflight rejected a malformed pasted expected SHA before root
  installation or dispatch. The actual local digest was then used; the repair unit ran once.
- No model downloads, copies, deletions, matrix reruns, reboot or permanent profile changes.

## Identities and evidence

Original Qwen container remains `15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f`.
Its image and model identity are unchanged, as recorded in the restoration receipt.

| Artifact | SHA-256 |
| --- | --- |
| Deployed service before | `cfa9d6dff5d892d65cd881e6ed7a4ea11a905622633e0e427f5d69868e66d89f` |
| Deployed service after | `14726d0e0b21b29443043936925b3a791156d79e3980048b1dcbff6ee5a967f4` |
| Root-custody repair script | `6341b58ee92f0b055e15b496bc870288cfebdd876f0fa587e874c6eeeaec7628` |
| [Restoration receipt](storage-repair-restoration.json) | `6d5e4b444d69b8caf52dd598b06de59ebc2b20ab6e204ca80f9d142518b7b9c8` |
| [Backup receipt](storage-repair-backup-receipt.json) | `203374684bcb86d7ecdbcd964015b9c2964b66511734f832d640d292bcbc8f96` |

The two exported receipts match their remote SHA-256 byte for byte.
[Fresh live proof](storage-repair-live-proof.json) records service health, queue state,
free space, deployed hashes and the absence of running context-test containers.

Root-only backups remain under `/var/lib/ai-server-cutovers/context-storage-20260909/`:
`service.py.before`, `state.json.before`, `admission.json.before`.
State/admission backups must not be blindly replayed over newer live state.
Root-custody script: `/var/lib/ai-server-cutovers/restore-storage-20260909.py`.

## Matrix remains incomplete

The [r1 safety stop](r1-safety-stop.md) is preserved, not reclassified as a model failure:
354 normal captures, 1 interrupted envelope and 257 unstarted coordinates out of 612.
The separate 4 GiB root-filesystem growth guard triggered capture termination; fixing
duplicate hardlink quota accounting does not explain or waive that growth limit.
GPU testing was not resumed. Existing AutoRound PC-A grading remains 30/30 answers,
220/220 vectors; the remaining 57 captured code answers still need grading.
