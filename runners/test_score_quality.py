import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_quality
from quality_suite import SUITE


MODE = "candidate-test"


def _gold_row(case):
    check = case["check"]
    row = {
        "kind": "case",
        "mode_id": MODE,
        "case_id": case["case_id"],
        "outcome": "pending_sandbox" if check["type"] == "python_function_cases" else "pass",
        "finish_reason": "stop",
        "content": "def placeholder():\n    pass" if check["type"] == "python_function_cases" else "",
        "reasoning": "",
        "tool_calls": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "request_sha256": "a" * 64,
        "elapsed_seconds": 1.0,
        "ttft_first_generated_seconds": 0.1,
        "ttft_first_content_or_tool_seconds": 0.2,
        "decode_tokens_per_second": 5.5,
        "decode_rate_definition": "completion_tokens / generated seconds",
    }
    if check["type"] == "exact":
        row["content"] = check["expected"]
    elif check["type"] == "json_shape":
        row["content"] = json.dumps(check["expected"], ensure_ascii=False)
    elif check["type"] == "text_constraints":
        row["content"] = {
            "instruction-constraints": "ALPHA,BRAVO,CHARLIE",
            "instruction-three-lines": "RED\nGREEN\nBLUE",
            "instruction-prefix-suffix": "START:2048:END",
        }[case["case_id"]]
    elif check["type"] == "python_lambda_cases":
        row["content"] = "lambda x: x**2 + 2*x - 3"
    elif check["type"] == "tool_call":
        row["finish_reason"] = "tool_calls"
        row["tool_calls"] = [{
            "id": "gold-call",
            "type": "function",
            "function": {
                "name": check["function_name"],
                "arguments": json.dumps(check["arguments"], separators=(",", ":")),
            },
        }]
    return row


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


class ScoreQualityTests(unittest.TestCase):
    def test_disk_guard_blocks_code_without_running_sandbox(self):
        rows = [_gold_row(case) for case in SUITE["cases"]]
        free_bytes = 149 * 1024**3
        fake_fs = types.SimpleNamespace(f_bavail=free_bytes, f_frsize=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.jsonl", root / "scored.jsonl"
            _write_jsonl(source, rows)
            with (
                mock.patch.object(score_quality.os, "statvfs", return_value=fake_fs, create=True) as statvfs,
                mock.patch.object(score_quality, "evaluate_code") as sandbox,
                mock.patch("builtins.print"),
            ):
                result = score_quality.score(source, MODE, output, window_free_bytes=200 * 1024**3)
            scored = {item["case_id"]: item for item in map(json.loads, output.read_text(encoding="utf-8").splitlines())}

        self.assertEqual((result["PASS"], result["FAIL"], result["BLOCKED"], result["cases"]), (31, 0, 29, 60))
        self.assertEqual(statvfs.call_count, 29)
        sandbox.assert_not_called()
        self.assertEqual(scored["code-clamp"]["error"], "sandbox_stopped_after_infrastructure_failure")
        self.assertEqual(scored["code-retry-delays"]["error"], "sandbox_stopped_after_infrastructure_failure")
        self.assertEqual(scored["retrieval-multihop-owner"]["outcome"], "PASS")

    def test_mocked_sandbox_classifies_all_sixty_and_stops_later_code(self):
        rows = [_gold_row(case) for case in SUITE["cases"]]
        by_id = {row["case_id"]: row for row in rows}
        rows.remove(by_id["reasoning-trains"])
        by_id["reasoning-workers"].update(outcome="error", error="transport_error", finish_reason=None)
        by_id["reasoning-discount-tax"].update(outcome="fail", error="finish_reason_length", finish_reason="length")

        calls = []

        def sandbox(case, _content, *, mode_id):
            calls.append((case["case_id"], mode_id))
            if len(calls) == 1:
                return {"outcome": "PASS", "error": None}
            return {"outcome": "BLOCKED", "error": "sandbox_infrastructure_failure"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.jsonl", root / "scored.jsonl"
            _write_jsonl(source, rows)
            with mock.patch.object(score_quality, "evaluate_code", side_effect=sandbox), mock.patch("builtins.print"):
                result = score_quality.score(source, MODE, output)
            scored = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result, {
            "mode_id": MODE,
            "input_sha256": result["input_sha256"],
            "PASS": 29,
            "FAIL": 1,
            "BLOCKED": 30,
            "cases": 60,
        })
        self.assertEqual(calls, [("code-clamp", MODE), ("code-dedupe-preserve", MODE)])
        verdicts = {item["case_id"]: item for item in scored}
        self.assertEqual(verdicts["code-clamp"]["outcome"], "PASS")
        self.assertEqual(verdicts["reasoning-discount-tax"]["outcome"], "FAIL")
        self.assertEqual(verdicts["reasoning-trains"]["error"], "generation_not_completed")
        self.assertEqual(verdicts["reasoning-workers"]["failure_class"], "runtime_or_protocol")
        self.assertEqual(verdicts["code-dedupe-preserve"]["error"], "sandbox_infrastructure_failure")
        self.assertEqual(verdicts["code-chunked"]["error"], "sandbox_stopped_after_infrastructure_failure")
        self.assertEqual(verdicts["retrieval-multihop-owner"]["outcome"], "PASS")

    def test_wrong_mode_and_duplicate_are_rejected_before_output(self):
        base = _gold_row(SUITE["cases"][0])
        matrices = [
            [{**base, "mode_id": "wrong-mode"}],
            [base, dict(base)],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, rows in enumerate(matrices):
                source, output = root / f"input-{index}.jsonl", root / f"output-{index}.jsonl"
                _write_jsonl(source, rows)
                with self.assertRaisesRegex(ValueError, "wrong_mode_or_duplicate_case"):
                    score_quality.score(source, MODE, output)
                self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.jsonl", root / "scored.jsonl"
            _write_jsonl(source, [])
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                score_quality.score(source, MODE, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")


if __name__ == "__main__":
    unittest.main()
