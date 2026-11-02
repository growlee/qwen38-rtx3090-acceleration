# Qwen context repair: KV 4.35 GB

This immutable one-shot bundle runs only the 84 frozen DFlash context cases.
The first standard case is `context-fact-start-t200000`; the other 83 retain
their frozen order. The previously captured and graded 12 GGUF cases are not
launched, copied, or graded again.

The additive profile `dflash2-huge-n7-kv4350mb-204800` changes only
`KV_MEM=4350000000` and `DFLASH_MAX_LEN=204800` from `dflash2-huge-n7`.
`GPU_UTIL=0.97`, seven draft tokens, KVarN, and `max_num_seqs=2` remain fixed.
Startup must report at least 204800 GPU KV-cache tokens before the first case.
The failed 4.0 GB profile remains separate evidence and is never relaunched.

The controller pins the completed clientfix attempt, its restored production
receipt, the 612-row combined original matrix, the 12-row GGUF scores, and the
84-row failed 4.0 GB profile. It also requires all 60 containers owned by r2,
the import-failure attempt, and clientfix to exist under their exact identities
and be stopped before admission closes.

No model download, image pull/build, weight copy, cleanup, promotion, automatic
retry, or change to public identity is permitted. The existing admission,
storage, capture, restore, and grading controls are reused. The planned peak is
955252736 bytes, below the 4 GiB owned-storage ceiling; the existing 150 GiB
root floor and 512 MiB write reserve remain fail-closed.

Remote root:
`/srv/ai-server/benchmarks/context-20260910-repair-kv4350`

After transferring exactly the five manifest-bound runtime files, invoke once
as root with the externally verified manifest SHA-256:

```text
python3 -B repair_context.py MANIFEST_SHA256
```

Terminal outcomes are `tests_complete`, `terminal_failure_graded`, or
`terminal_failure`. Restoration of the exact original Qwen identity, public
admission, and two thinking-disabled gateway markers is mandatory in all
controller paths. A failed or unknown mutation is never replayed.
