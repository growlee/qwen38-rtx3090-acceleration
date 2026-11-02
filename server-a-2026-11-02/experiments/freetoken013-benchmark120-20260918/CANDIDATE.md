# FreeToken coding120: exact finite benchmark candidate

User requested the same 120 tests used for other Qwen configurations on Forgejo.
The exact suite is the coding subset (120 of 200) of qwen38-quality-expanded-200-v1,
commit eb8932dd83130ac0330236a70c5caa5e5517ebfb. The original 60-case suite and
unrelated 600-case multi-turn matrix are excluded. 112 HumanEval-derived tasks,
8 authored repair/validation tasks, 827 fixed vectors. No model answer repairs.

FreeToken 0.1.3 commit cac247a860e316e06580d05aeb05f2e647bde214;
preetpatel/Qwen3.8-27B-Uncensored-NVFP4 revision
37b5130a2d2a1f7d4456ab3f8d05d0b2a45ea350, already agreed and installed.
Image sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134.
Existing reviewed runtime bundle b0bbaf4fddce284d2987b8dcabcc47d295c2f27686a7e4ff1ca45d246102dc95.

Runtime adapter derives from the already reviewed run_candidate.py. It retains
the original immutable image/bundle/weights/custody/configuration guards,
real host locks, native admission close/reopen, loopback binding and exact stop
and restoration. Qwen baseline is intentionally stopped. No new image/container
configuration, driver, Python installation, gateway or autostart modification.
Only the old five-request smoke client is replaced with the exact coding120
selector. One excluded historical warmup and 120 sequential streaming requests.
temperature=0, max_tokens=2048, unchanged per-case seeds and thinking OFF.
FreeToken accepts explicit max_tokens above its configured default256. Its API
does not consume seed; temperature0 selects greedy decoding. Record this runtime
limitation, without changing requests. Prefix cache remains enabled as installed.

Tokenizer-only trusted code executes inside the loaded container and enforces
the historic shared 16384-token request budget (prompt+2048+128). No generated
code executes in the client or on the host. After model stop and public admission
restoration, score120.py calls the original sandbox. Its one-line derivative
changes only the user-waived 150GiB free floor to32GiB. The existing256MiB growth
ceiling and all scoring criteria/container isolation remain unchanged.

Storage preparation budget, before any remote transfer:
- Existing image23244253967B, existing19.71GB weight CAS and prior retained
  artifacts are already present and counted in current disk usage. No download,
  pull, image load/build, model staging copy, backup or export is required.
- Source manifest binds about0.65MiB files; budget4MiB for exact source plus
  temporary JSON ingress and its finalization overlap. Ingress retained.
- Runtime persistent upper bound256MiB: up to121*256KiB generated output with
  JSON escaping reserve64MiB, rotated container logs32MiB, sampler/JIT/events16MiB,
  retained source/provenance4MiB, and metadata/overhead140MiB. Tmpfs stays in RAM.
- Scoring bound256MiB: at most120 read-only containers, each with bounded
  stdout64KiB, no logs, no host mounts and no persistent writes in container;
  budget1MiB metadata/empty upper per container plus136MiB allocator slack.
  Existing exact pinned sandbox image is shared; no extra image copy.
- Conservative combined growth516MiB including ingress overlap, generation
  evidence, and all retained score containers. Current free122283257856B gives
  about121.74GB remaining; required restoration reserve34359738368B.
  AI cap214748364800B remains applicable; current AI usage about117.51GB.
- Measure free and physical growth after staging, generation and scoring;
  no prune, cleanup or deletion. Source staging max4MiB; enforce256MiB per
  generation/scoring phase and32GiB reserve. Existing usage retained.

PASS execution:120 unique generation rows and120 scored rows with original
case IDs. Individual quality FAIL is a valid result, not grounds to tune/retry.
BLOCKED only for proven infrastructure/transport failure or failed restoration;
retain all completed cases and repair only a demonstrated harness issue before
considering any resume. No automatic retry or repeated successful request.
