# Bounded context matrix — 2026-09-08

Source: acceleration repository `roadmaps/quality-degradation.md` §8.
Preparation r0 stopped before any inference/admission mutation: installed
Transformers defaults apply_chat_template(return_dict=True), so len(mapping)
was not a token count. Source inspection proved this; r1 explicitly requests
return_dict=False and truncation=False. Failed r0 tokenizer/unit/files remain
retained. No measured case or warmup is repeated. R1 uses isolated suffix-r1
container/unit/output names; original manifests and logs are not overwritten.
Scope: 612 coordinates, 11 profiles, at most 11 excluded warmups.
Order: AutoRound fast, AutoRound huge, INT8 fast/huge, current Adaptive fast/huge,
reconstructed Adaptive fast/huge, DFlash fast/huge, GGUF 8k control.

User explicitly approved up to 12 hours of shared AI queue unavailability in this
task on 2026-09-08, with the last 15 minutes reserved for restoration. Independent
review found the inherited restore chain can exceed 15 minutes; the load phase
is therefore shortened to 11.25 hours and restoration gets **2700 seconds** within
the same 12-hour total, not an expansion of downtime authority. The old
controller starts its monotonic 43200-second deadline just before its final
preflight; tokenization and reference checks occur before that window. No new
request is allowed past 40500 seconds; the current candidate/client is stopped
by its exact created ID and remaining coordinates are BLOCKED. Per-request
timeout is 1800 seconds; model load timeout 600 seconds. Tokenization and other
preparation must finish within 3600 seconds before public closure. The transient
unit uses RuntimeMaxSec=50400 (14 hours including preparation and post-restore
grading), TimeoutStopSec=2700, Restart=no. Restore budget covers Qwen start600s,
two gateway calls480s, two owned-container stops90s, two GPU-idle waits180s,
22 final candidate/client inspections440s, four cleanup inspections80s,
three Qwen status calls180s, three identity inspections60s and five helper
admission calls50s: 2260s, with440s additional scheduling/probe allowance.
This includes the full final ownership scan, not just the start and completions.
These are ceilings,
not performance estimates or a promise that all 612 requests fit in 12 hours.

Fixtures: same 12 families/answers across seven lengths. Actual common text is
frozen using the pinned image's CPU tokenizer at target..target+256 tokens;
four weight trees have byte-identical tokenizer/config/template. Every backend
recounts its own messages before inference. Final target is >=200000 real
input tokens; huge MTP window 204800, DFlash 245760, no per-step KV/maxlen changes.
Thinking off, temp0, family seed fixed, output caps256/2048, prefixcache off,
concurrency1, power300W. MTP huge KVarN K4/V2 group128 with k3; native k4 remains
separate. GGUF window16384, DFlash native65536, MTP native57344.

Reuse: exact retained quality-remaining bundle manifest
`0a6ba209c1a9320743c9e455af3934afe47f9d856e7c01e35ff6665cb2ab35e3`, its controller,
transport, operator/admission functions and CPU sandbox. New files are fixture,
context client, bounded controller adapter and grading adapter only. No model
preparation/download function is called. All models are exact read-only binds.
Files/receipts are SHA-verified before preparation. Model/image absence is a
blocker, never permission to fetch. Global CAS/host locks and the old closure/
restoration path are unchanged. One independent Sol risk review covers this
exact package and launch. Old suites and measurements are untouched.

## Peak additional physical disk budget

Existing four trees, retained backups/CAS, GGUF, drafter and image layers stay
where they are and are already charged to filesystem usage. Added model copies,
downloads, partials, quarantine, image loads/unpacks/build caches: **zero**.

| Phase / retained output | Conservative upper bound |
| --- | ---: |
| Source package, manifests and preparation evidence | 16 MiB |
| One frozen fixture file, exclusive-created; no second server copy | 256 MiB |
| 11 captured outputs at32MiB + stderr1MiB each | 363 MiB |
| 11 candidate rotating logs at4MiB | 44 MiB |
| 181 retained container metadata records at4MiB each, including invalid r0 tokenizer | 724 MiB |
| Resource samples, grades, reports | 128 MiB |
| Unrelated host writes allowance | 1024 MiB |
| Restoration/log reserve beyond150GiB floor | 512 MiB |
| **Total bounded by enforced 4GiB growth cap** | **3067 MiB <4096 MiB** |

Container count ceiling: 11 GPU candidates+11 clients+1 CPU tokenizer+3 trusted
references+153 generated-code coordinates=179, plus invalid r0 tokenizer
(conservative ceiling181). Rootfs read-only;
cache/temp output uses bounded tmpfs, released when the owned container stops.
No retained writable weight tree, image backup or second full model copy.
Swap is existing preallocated host storage, not a new swap file. Metadata and
host-write allowance are conservative estimates, backed by measured filesystem
growth stops rather than claimed exact content sizes. Before/after preparation
growth <=512MiB; total growth <=4GiB and available disk >=150GiB throughout.

All created files and stopped exact-ID containers are retained; no deletion is
authorized. Peak disk and RAM guards are inherited. Tokenizer CPU capped2cores/
2GiB with no GPU/network; clients1GiB; GPU candidate uses old limits. Check no
FreeToken process before prep and again before the runtime window. Other user's
FreeToken artifacts/processes are never changed. During capture, infrastructure
uncertainty, OOM, lost process, temperature>=79C or resource exhaustion stops
the window; a valid semantic failure is not a reason to cancel or repeat cases.

After capture: original container/model/image checked, two gateway completions,
public admission and Qwen ready confirmed by the old controller. Only then grade
retained responses in the old CPU sandbox, exactly once. Results describe the
backend; no claim of Open WebUI 200k support (gateway1MiB cap unchanged).
