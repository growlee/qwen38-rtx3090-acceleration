import json
import unittest
from unittest import mock

import quality_sandbox


CASE = {
    "case_id": "code-add-one",
    "check": {
        "type": "python_function_cases",
        "function_name": "add_one",
        "cases": [
            {"args": [1], "kwargs": {}, "expected": 2},
            {"args": ["x"], "kwargs": {}, "expected_exception": "ValueError"},
        ],
    },
}


class QualitySandboxTests(unittest.TestCase):
    def test_accepts_plain_or_single_fenced_function_only(self):
        self.assertEqual("def add_one(x):\n return x + 1", quality_sandbox._extract_source("```python\ndef add_one(x):\n return x + 1\n```"))
        for source in ("text\n```python\ndef add_one(x): return 2\n```", "def add_one(x): return 2\n\ndef helper(): return 3"):
            result = quality_sandbox.evaluate_code(CASE, source, mode_id="baseline")
            self.assertEqual("FAIL", result["outcome"])

    @mock.patch.object(quality_sandbox, "_run_docker")
    def test_builds_hardened_cpu_only_command_and_returns_pass(self, run):
        envelope = {"passed": True, "vectors": [{"index": 0, "passed": True}, {"index": 1, "passed": True}]}
        run.return_value = quality_sandbox._DockerResult(0, (quality_sandbox._RESULT_MARKER + json.dumps(envelope) + "\n").encode(), b"", container_id="a" * 64, stopped=True)
        result = quality_sandbox.evaluate_code(CASE, "def add_one(x):\n if x == 'x': raise ValueError()\n return x + 1", mode_id="adaptive-k4")
        self.assertEqual("PASS", result["outcome"])
        command, payload, name = run.call_args.args
        joined = " ".join(command)
        for required in ("docker create -i", "--pull never", "--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "NVIDIA_VISIBLE_DEVICES=void", "/app/models:size=1m", "/cache:size=1m", "--log-driver none"):
            self.assertIn(required, joined)
        self.assertNotIn("--rm", command)
        self.assertNotIn("--gpus", command)
        self.assertEqual("qwen-quality-20260908-code-adaptive-k4-code-add-one", name)
        self.assertEqual("adaptive-k4", result["mode_id"])
        self.assertEqual("add_one", json.loads(payload)["function_name"])

    @mock.patch.object(quality_sandbox, "_run_docker")
    def test_model_failure_and_infrastructure_block_are_distinct(self, run):
        failed = {"passed": False, "vectors": [{"index": 0, "passed": False, "error": "value_mismatch"}]}
        run.return_value = quality_sandbox._DockerResult(0, (quality_sandbox._RESULT_MARKER + json.dumps(failed) + "\n").encode(), b"")
        self.assertEqual("FAIL", quality_sandbox.evaluate_code(CASE, "def add_one(x): return 0", mode_id="gguf")["outcome"])
        run.return_value = quality_sandbox._DockerResult(125, b"", b"daemon error")
        blocked = quality_sandbox.evaluate_code(CASE, "def add_one(x): return 0", mode_id="gguf")
        self.assertEqual("BLOCKED", blocked["outcome"])
        self.assertEqual("docker_exit_125", blocked["error"])

        run.return_value = quality_sandbox._DockerResult(137, b"", b"", resource_failure="candidate_oom", container_id="b" * 64, stopped=True)
        limited = quality_sandbox.evaluate_code(CASE, "def add_one(x): return 0", mode_id="gguf")
        self.assertEqual("FAIL", limited["outcome"])
        self.assertTrue(limited["resource_limited"])

    def test_invalid_fixture_is_blocked_not_model_failure(self):
        broken = {"case_id": "broken", "check": {"type": "python_function_cases"}}
        self.assertEqual("BLOCKED", quality_sandbox.evaluate_code(broken, "def f(): pass")["outcome"])

    @mock.patch.object(quality_sandbox.subprocess, "Popen")
    @mock.patch.object(quality_sandbox, "_short_command")
    def test_preexisting_name_is_never_started_or_killed(self, short, popen):
        short.return_value = quality_sandbox._DockerResult(0, b"c" * 64 + b"\n", b"")
        result = quality_sandbox._run_docker(["docker", "create"], b"{}", "owned-name")
        self.assertEqual("container_name_already_exists", result.error)
        popen.assert_not_called()
        self.assertEqual("ps", short.call_args.args[0][1])

    @mock.patch.object(quality_sandbox.subprocess, "run")
    def test_kill_requires_full_owned_container_id(self, run):
        self.assertFalse(quality_sandbox._kill_owned_container("owned-name"))
        run.assert_not_called()

    def test_bool_does_not_equal_integer_in_container_comparator(self):
        self.assertIn("type(actual) is type(expected)", quality_sandbox._CONTAINER_PROGRAM)


if __name__ == "__main__":
    unittest.main()
