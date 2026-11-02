# Final read-only data audit

No model requests, code execution, regrading or file changes were performed by
the independent data auditor. This checks the saved outputs, not runtime.

- 612 combined rows, 612 unique coordinates, 612 expected; missing=0, extra=0.
- All612 request SHA-256 and target-token values match the frozen plan.
- Windows:354 r1 +258 r2. The354 r1 rows are exactly the previously completed
  captures; after removing only added `result_window`, payload mismatches=0.
- r2:162 PASS with no error and finish_reason=stop; one BLOCKED error envelope
  (`finish_reason_missing`), plus95 BLOCKED coordinates with no dispatched capture.
- Totals:515 PASS,1 FAIL,96 BLOCKED.
- Old code87 and new graded code42 have disjoint coordinates. All42 new code
  answers PASS. Vectors: old634/638, new308/308, combined942/946.
- Combined SHA matches both summary and attempt receipt:
  `777d45c1eb2e495cc33cf1d9a031fa0eb5100dc69f6e2401aca87dc3de2a4af9`.

The auditor initially flagged the local summary hash differing from its remote
hash in the attempt receipt. This is the explicitly recorded one-trailing-LF
export normalization, not a change in grading data. The19-file export audit
verified remote SHA against exact local bytes or, only where recorded, local
bytes minus that single LF. See [final-export-receipt.json](final-export-receipt.json).

Code-only medians at200k (three PASS cases per profile):

| Huge profile | TTFT seconds | Decode tok/s |
|---|---:|---:|
| AutoRound | 314.391 | 35.036 |
| Current Adaptive | 314.800 | 33.747 |
| Reconstructed | 314.473 | 32.639 |
| INT8 | 313.312 | 32.416 |

These are short outputs with the capture-window and shared-host qualifications
in the terminal report. They are not long-code throughput measurements.
