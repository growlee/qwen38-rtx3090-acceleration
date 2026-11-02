import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quality_client
from quality_suite import SUITE


def _event(value):
    return b"data: " + json.dumps(value, separators=(",", ":")).encode() + b"\n"


class FakeResponse:
    def __init__(self, lines, *, status=200, content_type="text/event-stream"):
        self.lines = iter(lines)
        self.status = status
        self.content_type = content_type

    def getheader(self, name, default=None):
        return self.content_type if name.lower() == "content-type" else default

    def readline(self, _limit):
        return next(self.lines, b"")

    def read(self, _limit):
        return b""


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.body = None
        self.sock = None

    def request(self, _method, _path, *, body, headers):
        self.body = body
        self.headers = headers

    def getresponse(self):
        return self.response

    def close(self):
        pass


class FakeClient(quality_client.QualityClient):
    def __init__(self, responses):
        super().__init__("http://127.0.0.1:8000", None, 10)
        self.responses = iter(responses)
        self.connections = []

    def connection(self):
        connection = FakeConnection(next(self.responses))
        self.connections.append(connection)
        return connection


class QualityClientTests(unittest.TestCase):
    def test_all_non_fullcode_gold_responses_pass(self):
        constraint_gold = {
            "instruction-constraints": "ALPHA,BRAVO,CHARLIE",
            "instruction-three-lines": "RED\nGREEN\nBLUE",
            "instruction-prefix-suffix": "START:2048:END",
        }
        checked = 0
        for case in SUITE["cases"]:
            check = case["check"]
            check_type = check["type"]
            if check_type == "python_function_cases":
                self.assertIsNone(quality_client.evaluate_noncode(case, {"content": "", "tool_calls": []}))
                continue
            if check_type == "exact":
                response = {"content": check["expected"], "tool_calls": []}
            elif check_type == "json_shape":
                response = {"content": json.dumps(check["expected"], ensure_ascii=False), "tool_calls": []}
            elif check_type == "text_constraints":
                response = {"content": constraint_gold[case["case_id"]], "tool_calls": []}
            elif check_type == "python_lambda_cases":
                response = {"content": "lambda x: x * x + 2 * x - 3", "tool_calls": []}
            elif check_type == "tool_call":
                response = {
                    "content": "",
                    "tool_calls": [{
                        "id": "gold-call",
                        "type": "function",
                        "function": {
                            "name": check["function_name"],
                            "arguments": json.dumps(check["arguments"], separators=(",", ":")),
                        },
                    }],
                }
            else:
                self.fail(f"unhandled safe check: {check_type}")
            self.assertTrue(quality_client.evaluate_noncode(case, response), case["case_id"])
            checked += 1
        self.assertEqual(checked, 31)

    def test_fixture_matrix_and_request_allowlist(self):
        self.assertEqual(len(SUITE["cases"]), 60)
        client = FakeClient([])
        case = next(item for item in SUITE["cases"] if "reference_source" in item)
        body = client._body("model", case)
        encoded = quality_client.canonical(body)
        self.assertNotIn(b"reference_source", encoded)
        self.assertNotIn(b'"check"', encoded)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["seed"], case["request"]["generation"]["seed"])

    def test_reasoning_content_and_metrics_are_separate(self):
        response = FakeResponse([
            _event({"choices":[{"delta":{"reasoning_content":"think "},"finish_reason":None}]}),
            _event({"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}),
            _event({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"completion_tokens_details":{"reasoning_tokens":1}}}),
            b"data: [DONE]\n",
        ])
        client = FakeClient([response])
        result = client.stream_quality("model", SUITE["cases"][0])
        self.assertEqual(result["reasoning"], "think ")
        self.assertEqual(result["content"], "answer")
        self.assertIsNotNone(result["ttft_first_generated_seconds"])
        self.assertIsNotNone(result["ttft_first_content_or_tool_seconds"])
        self.assertIsNotNone(result["decode_tokens_per_second"])
        self.assertEqual(result["usage"]["completion_tokens_details"], {"reasoning_tokens": 1})

    def test_tool_call_fragments_are_assembled(self):
        case = next(item for item in SUITE["cases"] if item["case_id"] == "tool-call-schema")
        response = FakeResponse([
            _event({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup_","arguments":"{\"service\":\"cedar\","}}]},"finish_reason":None}]}),
            _event({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"service","arguments":"\"region\":\"eu-west\"}"}}]},"finish_reason":"tool_calls"}]}),
            _event({"choices":[],"usage":{"completion_tokens":9}}),
            b"data: [DONE]\n",
        ])
        client = FakeClient([response])
        row, terminal = quality_client.run_case(client, "model", "mode", case)
        self.assertFalse(terminal)
        self.assertEqual(row["outcome"], "pass")
        self.assertEqual(row["tool_calls"][0]["function"]["name"], "lookup_service")

    def test_pow_lambda_and_nested_boolean_are_evaluated_strictly(self):
        lambda_case = next(item for item in SUITE["cases"] if item["case_id"] == "code-unit-expression")
        self.assertTrue(quality_client.evaluate_noncode(lambda_case, {"content": "lambda x: x**2 + 2*x - 3", "tool_calls": []}))
        self.assertFalse(quality_client.evaluate_noncode(lambda_case, {"content": "lambda x, /: x**2 + 2*x - 3", "tool_calls": []}))

        nested_case = next(item for item in SUITE["cases"] if item["case_id"] == "instruction-nested-json")
        wrong = {"user": {"name": "Ира", "active": 1}, "roles": ["admin", "reader"]}
        self.assertFalse(quality_client.evaluate_noncode(nested_case, {"content": json.dumps(wrong), "tool_calls": []}))

    def test_finish_length_is_valid_fail_not_terminal(self):
        response = FakeResponse([
            _event({"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}),
            _event({"choices":[],"usage":{"completion_tokens":3}}),
            b"data: [DONE]\n",
        ])
        client = FakeClient([response])
        row, terminal = quality_client.run_case(client, "model", "mode", SUITE["cases"][0])
        self.assertFalse(terminal)
        self.assertEqual((row["outcome"], row["error"]), ("fail", "finish_reason_length"))
        self.assertEqual(row["content"], "partial")

    def test_parse_error_preserves_bounded_partial_and_is_terminal(self):
        response = FakeResponse([
            _event({"choices":[{"delta":{"reasoning":"r"},"finish_reason":None}]}),
            _event({"choices":[{"delta":{"content":"kept"},"finish_reason":None}]}),
            b"data: {bad json}\n",
        ])
        client = FakeClient([response])
        row, terminal = quality_client.run_case(client, "model", "mode", SUITE["cases"][0])
        self.assertTrue(terminal)
        self.assertEqual((row["outcome"], row["error"]), ("error", "sse_invalid"))
        self.assertEqual((row["reasoning"], row["content"]), ("r", "kept"))

    def test_clean_eof_and_transport_failure_are_terminal(self):
        eof = FakeResponse([_event({"choices":[{"delta":{"content":"unknown"},"finish_reason":None}]})])
        row, terminal = quality_client.run_case(FakeClient([eof]), "model", "mode", SUITE["cases"][0])
        self.assertTrue(terminal)
        self.assertEqual(row["error"], "stream_incomplete")

        class BrokenResponse(FakeResponse):
            def readline(self, _limit):
                raise OSError("connection lost")

        broken = BrokenResponse([])
        row, terminal = quality_client.run_case(FakeClient([broken]), "model", "mode", SUITE["cases"][0])
        self.assertTrue(terminal)
        self.assertEqual(row["error"], "transport_error")


if __name__ == "__main__":
    unittest.main()
