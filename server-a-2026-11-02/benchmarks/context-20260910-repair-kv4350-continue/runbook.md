# KV-4350 context continuation: remaining 63 cases

This immutable one-shot bundle continues the same frozen
`dflash2-huge-n7-kv4350mb-204800` profile after a proven host-storage abort.
It dispatches only the 63 cases without a terminal envelope. The first is
`context-version-end-t32768`, whose earlier `case_start` was interrupted by
`root_storage_reserve_exhausted`; the 21 completed answers are never replayed.

The grader runs only after all 63 new terminal envelopes are captured and the
original Qwen service is restored. It then verifies that the pinned 21-answer
prefix and new 63-answer suffix form the exact 84-case matrix before grading it
once. It references the pinned 612-row original matrix and does not copy or
regrade GGUF results. Incomplete continuation evidence is preserved without
invoking the grader.

The controller reuses the existing admission, storage, client, profile,
capacity, restoration, and grading controls. It pins the interrupted raw data,
report, attempt, selection, controller, completed original matrix, and the new
restoration receipt. All 62 previously owned containers must retain exact
identity and be stopped before admission closes.

No model download, image pull/build, weight copy, cleanup, deletion, promotion,
automatic retry, or public identity change is permitted. The 4 GiB owned-data
ceiling, 150 GiB root floor, and 512 MiB write reserve remain fail-closed.
The planned peak stays bounded at 955252736 bytes.

Remote root:
`/srv/ai-server/benchmarks/context-20260910-repair-kv4350-continue`

Transfer exactly the five manifest-bound runtime files and invoke once as root:

```text
python3 -B repair_context.py MANIFEST_SHA256
```

Terminal outcomes are `tests_complete` or `terminal_failure`. Every controller
path restores exact production identity, public admission, and two
thinking-disabled gateway markers before returning.
