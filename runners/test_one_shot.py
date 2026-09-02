from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_client


class Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    payload = (
        b'data: {"choices":[{"delta":{"reasoning":"valid output"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":8},'
        b'"timings":{"prompt_per_second":100.0,"draft_n":4,"draft_n_accepted":3}}\n\n'
        b'data: [DONE]\n\n'
    )

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b"{}"
        elif self.path == "/v1/models":
            body = b'{"data":[{"id":"qwen3.8-27b-uncensored"}]}'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers["Content-Length"])
        self.requests.append(json.loads(self.rfile.read(size)))
        payload = self.payload
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class OneShotClientTest(unittest.TestCase):
    def test_one_warmup_three_measured_and_no_content_in_report(self) -> None:
        Handler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        output = io.StringIO()
        try:
            argv = [
                "benchmark_client.py",
                "--base-url", f"http://127.0.0.1:{server.server_port}",
                "--model", "qwen3.8-27b-uncensored",
                "--mode-id", "test",
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
                self.assertEqual(benchmark_client.main(), 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)
        value = json.loads(output.getvalue())
        self.assertEqual(len(Handler.requests), 4)
        self.assertEqual(value["warmup_count"], 1)
        self.assertEqual(value["measured_request_count"], 3)
        self.assertEqual(value["metrics"]["draft_n_total"], 12)
        self.assertEqual(value["metrics"]["draft_n_accepted_total"], 9)
        self.assertEqual(value["metrics"]["draft_acceptance_ratio"], 0.75)
        self.assertNotIn("valid output", output.getvalue())
        self.assertNotIn("резерв", output.getvalue())

    def test_clean_eof_without_done_has_content_free_protocol_diagnostic(self) -> None:
        original_payload = Handler.payload
        diagnostic_payload = (
            b'data: {"choices":[{"delta":{"content":"private-response-fragment"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":10,"completion_tokens":8}}\n\n'
        )
        Handler.payload = diagnostic_payload
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = benchmark_client.Client(f"http://127.0.0.1:{server.server_port}", None, 5)
            case = {"case_id": "operations-guide-ru", "messages": [], "max_tokens": 1024}
            with self.assertRaises(benchmark_client.BenchmarkError) as caught:
                client.stream("qwen3.8-27b", case, warmup=False)
        finally:
            Handler.payload = original_payload
            server.shutdown()
            server.server_close()
            thread.join(3)
        self.assertEqual(caught.exception.code, "stream_incomplete")
        self.assertEqual(caught.exception.case_id, "operations-guide-ru")
        self.assertEqual(
            caught.exception.protocol,
            {
                "saw_first_delta": True,
                "saw_finish_reason": True,
                "finish_reason": "stop",
                "saw_done": False,
                "saw_usage": True,
                "usage_complete": True,
                "termination": "clean_eof",
                "response_bytes": len(diagnostic_payload),
                "delta_count": 1,
            },
        )
        self.assertNotIn("private-response-fragment", repr(caught.exception.protocol))

    def test_transport_error_has_protocol_state_without_response_text(self) -> None:
        line = b'data: {"choices":[{"delta":{"content":"private-response-fragment"}}]}\n\n'

        class Response:
            status = 200
            calls = 0

            def getheader(self, _name: str, _default: str = "") -> str:
                return "text/event-stream"

            def readline(self, _limit: int) -> bytes:
                self.calls += 1
                if self.calls == 1:
                    return line
                raise ConnectionResetError()

        class Connection:
            def request(self, *_args: object, **_kwargs: object) -> None:
                return

            def getresponse(self) -> Response:
                return Response()

            def close(self) -> None:
                return

        client = benchmark_client.Client("http://127.0.0.1:9", None, 5)
        case = {"case_id": "operations-guide-ru", "messages": [], "max_tokens": 1024}
        with mock.patch.object(client, "connection", return_value=Connection()):
            with self.assertRaises(benchmark_client.BenchmarkError) as caught:
                client.stream("qwen3.8-27b", case, warmup=False)
        self.assertEqual(caught.exception.code, "transport_error")
        self.assertEqual(caught.exception.protocol["termination"], "transport_error")
        self.assertTrue(caught.exception.protocol["saw_first_delta"])
        self.assertFalse(caught.exception.protocol["saw_done"])
        self.assertNotIn("private-response-fragment", repr(caught.exception.protocol))


if __name__ == "__main__":
    unittest.main()
