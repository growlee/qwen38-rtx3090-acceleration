import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_client
import quality_context_client as context_client
import quality_context_suite


class _OfflineTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get('return_dict', True):
            raise AssertionError('installed_transformers_defaults_to_mapping_not_token_ids')
        assert kwargs.get('truncation') is False
        content = "\n".join(message["content"] for message in messages)
        # Exact monotonic stand-in for tokenizer output: it tests the finite
        # bisection and bounds without pretending to model real tokenization.
        return range(8000 + content.count("[ШУМ D") * 64)


class QualityContextClientTests(unittest.TestCase):
    def test_freeze_builds_exact_order_and_enforces_200k_lower_bound(self):
        old_suite = copy.deepcopy(quality_client.SUITE)
        fake_transformers = mock.Mock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = _OfflineTokenizer()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frozen.json"
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                context_client.freeze("unused-local-model", output)
            raw = output.read_bytes()
        frozen = json.loads(raw)
        self.assertEqual(len(frozen["cases"]), 84)
        self.assertEqual(
            [(case["target_tokens"], case["family_id"]) for case in frozen["cases"]],
            [(target, family) for target in context_client.LENGTHS for family in quality_context_suite.FAMILY_IDS],
        )
        final = [case for case in frozen["cases"] if case["target_tokens"] == 200000]
        self.assertEqual(len(final), 12)
        self.assertTrue(all(200000 <= case["frozen_input_tokens"] <= 200256 for case in final))
        self.assertLessEqual(max(case["frozen_input_tokens"] for case in frozen["cases"]) + 4096 + 128, 204800)
        self.assertEqual(quality_client.SUITE, old_suite)

    def test_frozen_request_bytes_and_hash_match_exact_transport_body(self):
        fake_transformers = mock.Mock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = _OfflineTokenizer()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frozen.json"
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                context_client.freeze("unused-local-model", output)
            frozen = json.loads(output.read_bytes())
        for case in frozen["cases"]:
            body = quality_client.QualityClient("http://127.0.0.1:8080", None, 60)._body("qwen3.8-27b", case)
            encoded = context_client.canonical(body)
            self.assertEqual(case["request_bytes"], len(encoded))
            self.assertEqual(case["request_sha256"], hashlib.sha256(encoded).hexdigest())

    def test_capture_blocks_on_output_plus_128_without_dispatching_case(self):
        case = quality_context_suite.build_case(quality_context_suite.FAMILY_IDS[0], 0)
        case.update(case_id="context-fact-start-t8192", family_id="context-fact-start",
                    target_tokens=8192, frozen_input_tokens=8192, request_bytes=123)
        body = quality_client.QualityClient("http://127.0.0.1:8080", None, 60)._body("qwen3.8-27b", case)
        case["request_sha256"] = hashlib.sha256(context_client.canonical(body)).hexdigest()
        frozen = {"suite_id": "test", "cases": [case]}
        raw = context_client.canonical(frozen)

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def get(self, path):
                self_path = path
                return 200, json.dumps({"data": [{"id": "qwen3.8-27b", "max_model_len": 8447}]}).encode(), {}

        emitted = []
        run = mock.Mock(return_value=({"outcome": "pass"}, False))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture.json"
            fixture.write_bytes(raw)
            with mock.patch.object(context_client, "QualityClient", FakeClient), \
                 mock.patch.object(context_client, "tokenize", return_value=8192), \
                 mock.patch.object(context_client, "run_case", run), \
                 mock.patch.object(context_client, "_emit", side_effect=emitted.append):
                result = context_client.capture(fixture, hashlib.sha256(raw).hexdigest(), "mode", 8447, (8192,), "vllm")
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 1)  # excluded warmup only
        blocked = [row for row in emitted if row.get("kind") == "case"]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["outcome"], "blocked")
        self.assertEqual(blocked[0]["error"], "capacity_bound_exceeded")
        self.assertEqual(blocked[0]["request_sha256"], case["request_sha256"])

    def test_fixture_byte_identity_mismatch_fails_before_runtime(self):
        raw = context_client.canonical({"suite_id": "test", "cases": []})
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture.json"
            fixture.write_bytes(raw)
            with mock.patch.object(context_client, "QualityClient") as client:
                with self.assertRaisesRegex(RuntimeError, "fixture_identity_invalid"):
                    context_client.capture(fixture, "0" * 64, "mode", 204800, context_client.LENGTHS, "vllm")
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
