# Grading-only continuation after verified restoration

Exact source SHA-256:
`f116dc156bad0fe0475f7fd3cbbe0e16dc741d8f7623f5e0376eec78a3d22013`.
The original grader, failed report, source manifest, selection and captured
answers are immutable. Only the imported restoration-check callable is replaced
in memory with a check of the separate SHA-bound recovery receipt.

Source/report/receipt validation binds 162 normal captures, 1 error envelope,
42 code answers, 258 selected coordinates and 10 exact stopped owned containers.
The same OwnedStorage and isolated CPU sandbox remain in use. Previous r1 code
is not executed again. An exclusive attempt receipt and absent-score checks
prevent blind repetition after any partial grading attempt.

Three focused offline tests passed. The same independent Sol critic identified
one missing check before dispatch: absent integrity entries did not prove absent
capture files for unstarted Original GGUF. The minimal fix now fences every
unlisted selected-mode raw/resource path. Only its new negative test was run,
covering both files; PASS. Source compile also passed. Final review: APPROVE.

Remote staged bytes matched the exact reviewed SHA. Fresh launch checks proved
the original capture unit failed/terminal, recovery unit exited successfully,
the receipt hash matched, admission was public/empty, and original Qwen was ready.
Dispatched once as `qwen-context-20260908-r2-grade.service`, invocation
`3e26403a02f1470e8f70afa734db5c3b`, Restart=no, RuntimeMaxSec=2700.
This launches no model inference and does not reopen the benchmark window.
