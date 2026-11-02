# Final r1 evidence adapter verification

Implementation: final evidence is a separate SHA-bound input; the original r1
failed report and612 final scores are not rewritten or reevaluated. The one
zero-token storage interruption is classified explicitly. Generic transport
errors and every completed answer remain excluded from continuation.

Executor-reported targeted offline results:

- 15 affected-boundary methods PASS,0.051s.
- 3 real-evidence/prepared-selection methods PASS,0.018s: exact258 coordinates,
  6 profiles;354/257/1 source partition; zero overlap with354 completed answers;
  generic transport-error mutation rejected.
- Initial test invocation had an import-path error and one test-only mutation
  pattern mismatch; corrected the invocation/test, not production behavior.
  These were harness checks only; no model request was executed.

Sources tested and included byte-for-byte in the launch package:

| Source | SHA-256 |
| --- | --- |
| continue_context.py | a0a96fc533bb0dd962ff049a9b17fd0d950fb0d69650c97eee582e196476ef3e |
| grade_context_continuation.py | e89b034035990aa87906a84a40ea03f16f71e0fe956ef98dfde57c3b48a4261b |
| quality_context_remaining.py | d05b4125873eb7afdea27e64f47515c2ff2371c85771ec60fbaee5b1bc6e3979 |

Test sources remain in home_lab outputs/qwen-context-20260908:
test_r1_final_evidence_adapter.py, test_continue_context.py and
test_grade_context_continuation_integration.py. This report records offline
evidence only, not remote deployment, model execution or restoration proof.

The initial manifest1bd2220fc052f295c906a9387b334e93a94e0a79af8bd447b0e78aecffcc3572
failed remote check_source before launch: audit does not contain
restoration_sha256. The final summary does, and is itself audit-hash-bound.
Removed only the nonexistent audit-field comparison; kept the complete
audit→scores/summary→restoration chain. One new end-to-end check_source test
passed in0.063s, using actual final/settings/report/raw evidence and only a
mocked physical SHA for the server-held frozen file. Full actual frozen-file
SHA and check_source subsequently passed on Server A.

Corrected continue_context.py SHA:
6d2e89059388a5fef595cfc80f76a070d46f21e3a8865c38054b33691c6a5307.
Final launch manifest SHA:
89e8667d4c2031b6ce91fe0f73b206592ea9a6f4066b2e9328b4d570e209299e.
The same independent Sol reviewer approved the original operational candidate
and this material delta; no material findings. Old manifest/controller retained
under preparation-history, on PC-A and server. No inference was repeated.
Package and exact selection: [r2-launch-package](r2-launch-package/manifest.json).
