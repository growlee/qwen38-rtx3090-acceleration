# Future context harness: owned storage accounting

User confirmed this policy amendment on2026-09-09 after explaining that another
agent is working on Server A. Scope: prepare and verify the future harness;
do not resume the matrix, modify frozen r1, remove data or interrupt that agent.

## Change

The old4GiB guard measured net occupancy of the whole root filesystem. It
correctly stopped r1, but unrelated provisioning contributed materially to
that delta. See [the forensic evidence](storage-attribution-2026-09-09.md).

The future draft instead measures allocated blocks in its own run directory,
receipt-bound container metadata/log directories, and each container's active
and init containerd snapshots. Hardlinks are counted once; shared read-only
models/images/fixtures are not copied. Unknown or missing accounting fails
closed. The real filesystem free-space measurement is not synthesized.

- Owned allocations plus512MiB headroom must stay within4GiB.
- Real root free space must stay at least150GiB plus that512MiB headroom
  before/during further work. Initial preflight remains at least154GiB.
- Shared Docker metadata/journal/control-plane writes are not claimed to be
  exactly attributable. The retained512MiB is an allowance, not such a proof.
  These writes and all other agents' allocations affect the real free-space
  guard. Sampling does not provide a filesystem quota or protect against an
  external writer exhausting space between samples.
- Only two inherited storage predicates change in memory, with exact-match
  checks against already SHA-verified source. GPU/RAM/time and restore logic
  remain in the inherited controller; frozen files stay unchanged.
- Candidate/client/sandbox IDs are registered after create and before start.
  The existing grader receives the same ownership object in-process. Direct
  standalone grading is rejected rather than silently forgetting containers.
- Resource samples append owned allocated bytes and real root free bytes;
  existing timing/VRAM/RAM columns and the256-byte row cap remain intact.

## Verification

An initial read-only probe on Server A at epoch1788946246.7840521 loaded the
adapter from stdin without writing its source to the host. It inspected the
retained r1 directory plus one exact stopped container
`54b8dfec96b44892f446048bd2228e141fa176a8a4f347b4a7aaa8d5819d1cdd`.
The adapter identified snapshot6878(active),6877(init), and its Docker metadata
directory. Count:33529856 allocated bytes; real root free:168293281792 bytes.
Probe source SHA:`bf78ed71876f093d92aab01470ba7be9e9ef6c70aeb813da4672e092a40275e1`.
This is an initial measurement check, not a complete r1 allocation total or a
model benchmark. The later pure POSIX path parsing adjustment supports offline
Windows tests without changing Linux path semantics.

Offline verification:17 targeted methods pass. Initial invocation:16PASS and
one test-harness failure (WindowsPath vs PurePosixPath expected object). Only
that expectation was corrected; only that method was rerun, withPASS. This
was not a repeated model test. Adapter checks exercise exactly two predicate
rewrites, source hash and real available-byte function preservation, create
receipt ordering and failure before start. File-level mocked grading preserves
611 prior scores and evaluates only one new task; disk failure and re-entry
are fenced. [Raw local test output](storage-owned-accounting-tests.txt).

Final adapter SHA:`2085436ed77ec5a9f64d61568bb612254cb58158027869fc036fa666ff914cfd`.
Its read-only Linux probe at epoch1788946613.3155868 registered12 exact IDs from
r1's capture cleanup receipts, all stopped. The r1 run root plus those12
containers:35995648 allocated bytes; real free:168293060608 bytes. This is not
the four reference/fixture containers or an all-time write total.
[Exact final probe](storage-owned-accounting-probe.json).
Frozen r1 source SHA remained
`7cb2f97465a69992babc9887780477ab98d3eb548500e33052cba7eb70248816`,
manifest SHA remained
`c71c26c5cf3124347a021489d60148b102a5ad2c35ee4b1314e37ef8eafa36e6`.
Gateway and Open WebUI were healthy; original Qwen identity/image remained
unchanged and idle-stopped, exit0/OOMfalse, since01:27:41UTC. No new generation
was used to claim end-to-end user access.

The existing independent Sol reviewer returnedAPPROVE with no material
P0/P1/P2. This covers the local policy change, not deployment or incomplete
selection/manifest inputs. [Source snapshot and hashes](owned-storage-draft/source-sha256.json)
retain the reviewed code and test files in this repository.

No test inference, container creation, service restart, download,
mount operation, deletion or matrix resumption was performed for this change.

## Delivery boundary

This is a local future-harness repair, not a deployed runtime change. The r2
draft still needs its final source manifest/selection reconciled with r1's
terminal evidence before any launch.257 never-dispatched cases and one
interrupted case must not be conflated; no valid result is replayed here.
The storage policy change does not itself authorize launch or cure those
pre-existing selection prerequisites. No commit or push is included.
