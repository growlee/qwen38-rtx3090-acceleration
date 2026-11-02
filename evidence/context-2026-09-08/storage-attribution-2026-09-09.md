# Context r1: physical disk-growth attribution

Read-only investigation on `server-a`, 2026-09-09 08:24–08:29 UTC.
[Machine-readable evidence](storage-attribution-2026-09-09.json).

## Finding

The capture guard did not confuse hardlinks or RAM with disk usage. Its input
was filesystem available space, and its arithmetic was correct: free space
fell from 172198895616 to 167902392320 B, a 4296503296 B net increase against
the frozen 4294967296 B limit. The excess was 1536000 B; the independent
150 GiB reserve still held. This remains a valid infrastructure safety stop,
not a model-quality or 200k-capacity failure.

Large unrelated provisioning happened on the same filesystem during r1.
The following distinct retained files were born after the 14:30:12 UTC
capture baseline and before the 00:24:10 UTC stop:

| Object | Born UTC, September 8 | Allocated bytes |
| --- | --- | ---: |
| `/var/lib/homelab-storage/demo-api-a.ext4` | 15:24:14 | 3221229568 |
| Guarded-rollout input `guarded-image.tar` | 20:20:47 | 230612992 |
| Guarded-rollout input `test_demo.dem` | 20:20:47 | 60604416 |
| containerd blob `f467b00c…` | 20:21:53 | 155938816 |
| containerd blob `eb2d89ad…` | 20:21:52 | 59887616 |
| **Distinct retained allocation** | | **3728273408** |

Exact paths, device/inode identities and timestamps are in the JSON. This is
about 3.47 GiB, equivalent to 86.77% of the observed net growth. It is **not**
a complete net-growth decomposition: filesystem metadata, changes to existing
files, temporary allocations and releases were not captured per directory.

The strongest historical attribution is the 3 GiB FACEIT demo backing file.
The retained provisioning receipt records `format.allocated_bytes=3221229568`,
the same allocation and inode measured now. Its provisioning source calls
`os.posix_fallocate` followed by `mke2fs` with `nodiscard`; this was an actual
allocation, not just a sparse 3 GiB logical size. The receipt is
`/var/backups/homelab/demo-storage-prestate-20260908T152352Z/evidence.json`,
SHA-256 `9984a86110bdc2a9df597ae5088a26d99864e93e4b10ff5aeedef9da30b54c34`.
The file is still mounted through `/dev/loop1` at `/srv/faceit-demo-uploads`.
It is another service's retained infrastructure, not Qwen scratch space.

For comparison, all 16 retained r1 containers have aggregate Docker `SizeRw`
1261568 B and retained log sizes 690995 B. Six model containers bind weights
read-only. Their cache/tmpfs contract is in the unchanged captured launcher.
This rules out multi-GiB retained writable layers in these containers, but
does not turn a post-run snapshot into a bound on all historical transient I/O.
The new-snapshot inventory is retained separately; do not add it blindly to
the container numbers or confuse OCI blob storage with extracted layers.

## Limits of the reconstruction

- The r1 resource samples did not retain a per-directory disk baseline.
- `sar -F` reports activity unavailable in both retained `sa08` and `sa09`.
- Journal aggregation for the window found roughly 10 MiB of message text,
  mostly SSH/session activity; message bytes are not journal physical bytes.
- The remaining net growth is not assigned to Qwen, FreeToken, Unsloth or a
  particular cache without evidence. No full byte-for-byte explanation is claimed.
- Concurrent host provisioning also means GPU-exclusive operation was not
  whole-host isolation. Its effect on timings is unquantified; no reruns were made.

## Correction boundary

No new Qwen/runtime defect requiring a code patch was demonstrated. Removing
the root-growth check, subtracting guessed foreign growth, raising its limit,
or automatically resetting its baseline mid-window would hide real disk use.
None of those changes was made. The separate hardlink-accounting fix remains
in place and is not reused as an explanation of this physical allocation.

The actionable correction is operational: any future context window must not
overlap unrelated host provisioning, image imports/builds or large downloads.
Already retained files belong in the next window's initial occupied-space
baseline, not in its new-growth allowance. Before a future launch, calculate
the selected cases' peak allocation, background-write allowance and restoration
reserve from current evidence; keep the 150 GiB floor and do not silently waive
the 4 GiB growth limit. Existing per-AI locks do not reserve disk writes by other
projects. No new global locking service or experiment lifecycle is proposed.

This investigation does not authorize stopping or modifying FACEIT deployment,
unmounting its disk, deleting its artifacts, or starting r2. Excluding those
operations during a future matrix requires coordination with their owner.
Consequently **no runtime repair is claimed**; the external overlap is identified,
but whole-host isolation for a future test window has not been established.

## Preserved access and scope

At 08:29:20 UTC, the existing read-only helper verified empty public admission.
Supervisor is active, global breaker is false, capacity mode `serialized-99`,
no heavy profile or GPU compute process is active. Qwen's original container,
image and model path are unchanged. Qwen exited normally at 01:27:41 UTC,
consistent with the documented idle unload; it was not restarted by this audit.
Gateway and Open WebUI were healthy at the initial live check. This is control-
plane/access preservation evidence, not a new generation or browser smoke.

Free space: 168247549952 B (about 156.69 GiB). Supervisor source remains
`14726d0e0b21b29443043936925b3a791156d79e3980048b1dcbff6ee5a967f4`.
No service changes, restarts, deletions, downloads, model requests, repeated
matrix cases, quota changes, commits or pushes occurred. This report and its
evidence were added locally and the existing roadmap was updated; historical
results remain unchanged.
