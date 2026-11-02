# Qwen context repair continuation

One transient, one-shot execution. It runs exactly two profiles in this order:

1. `original-gguf`: the 12 never-dispatched 8192-token coordinates.
2. `dflash2-huge-n7-kv4gb-204800`: 84 frozen coordinates under a distinct repaired profile. The existing 200000-token fact/start coordinate is first, then the other 83 once in frozen order.

The repaired profile changes only `KV_MEM=4000000000` and
`DFLASH_MAX_LEN=204800`. `GPU_UTIL=0.97`, DFlash n7, KVarN, max-seqs 2,
the image, target weights, drafter, prompts, output caps, and temperature remain
unchanged. Before the first matrix request, startup evidence must prove the
exact process flags and a reported GPU KV capacity of at least 204800 tokens.

The adapter SHA-pins the r1 fixture/controller and the finalized r2 combined
scores/restoration evidence. It verifies all 52 r2-owned containers remain
stopped, public admission is empty, and the exact production Qwen identity is
unchanged before handoff. A process-unique, per-Operator namespace removes the
known preflight/finally idempotency-key collision. Unknown close, stop, start,
or marker outcomes are never replayed.

No model download, build, weight copy, fixture copy, cleanup, promotion, or
automatic response retry is permitted. The inherited CAS/host locks, GPU/RAM
guards, exact-ID cleanup, and finally restoration remain active. A terminal
runtime failure stops the matrix; already completed cases are retained as
evidence and are not repeated.

## Physical disk bound

The owned-storage controller enforces a 4 GiB allocation ceiling and a real
150 GiB root-free floor with 512 MiB write headroom. The precomputed peak for
this run is 911 MiB (955252736 bytes): 16 MiB package/transfer, 66 MiB bounded
raw/stderr, 8 MiB candidate logs, 112 MiB for four candidate/client plus up to
24 code-sandbox container metadata/snapshots, 133 MiB bounded samples, 64 MiB
reports/package margin, and 512 MiB headroom. Existing weights, images, r1
fixture, and r2 evidence are
read-only retained inputs and no second full copy is made.

## Launch boundary

Transfer only the five manifest-bound runtime files into the absent directory
`/srv/ai-server/benchmarks/context-20260910-repair-clientfix`, verify remote SHA-256, and
invoke once as root with:

```text
python3 -B /srv/ai-server/benchmarks/context-20260910-repair-clientfix/repair_context.py <MANIFEST_SHA256>
```

Use the existing transient-systemd launch pattern so the caller can disconnect.
Do not relaunch if `attempt.json` or `results/` exists. Terminal outcomes are
`tests_complete`, `terminal_failure_graded`, or `terminal_failure`. In every
case the window report must prove the exact original production identity,
two non-thinking gateway markers, and public empty admission before results are
accepted.

The original 612-row matrix remains a separate provenance boundary: only its
12 `original-gguf` BLOCKED rows may be replaced. The original
`dflash2-huge-n7` 84 BLOCKED rows stay unchanged; the repaired 84-row profile is
additive and separately identified.
