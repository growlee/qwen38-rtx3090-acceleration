"""One excluded warmup and three fixed streamed speed requests.

Runs inside the isolated FreeToken container.  The measured prompts are exact
requests from coding120 cases that previously passed with at least 240 output
tokens.  A request failure is terminal; there are no retries or extra cases.
"""
from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import time
import sys

MODEL = "qwen38-nvfp4-freetoken"
SUITE = Path("/speed-cases.json")
CASE_IDS = ("expanded-he-075", "expanded-he-081", "expanded-he-119")
MAX_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 120
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def get_json(path: str) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", 1919, timeout=15)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if response.status != 200 or len(body) > MAX_RESPONSE_BYTES:
            raise RuntimeError("GET_failed_" + path)
        return json.loads(body)
    finally:
        conn.close()


def selected_requests() -> list[dict]:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    by_id = {row["case_id"]: row for row in suite["cases"]}
    rows = [by_id[case_id] for case_id in CASE_IDS]
    if [row["case_id"] for row in rows] != list(CASE_IDS):
        raise RuntimeError("fixed_case_identity_changed")
    requests = []
    for row in rows:
        request = row.get("request") or {}
        generation = request.get("generation") or {}
        if (row.get("category") != "coding" or generation.get("temperature") != 0
                or not isinstance(generation.get("seed"), int)
                or not isinstance(request.get("messages"), list)):
            raise RuntimeError("fixed_case_request_invalid_" + row["case_id"])
        requests.append({"case_id": row["case_id"], "messages": request["messages"],
                         "seed": generation["seed"]})
    return requests


def stream(case_id: str, messages: list[dict], seed: int, excluded_warmup: bool) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    result = {
        "kind": "warmup" if excluded_warmup else "case",
        "case_id": case_id,
        "status": "FAIL",
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "completion_tokens": None,
        "first_content_ttft_seconds": None,
        "decode_elapsed_seconds": None,
        "decode_tokens_per_second": None,
        "elapsed_seconds": None,
        "full_output": "",
        "error": None,
        "usage": None,
        "finish_reason": None,
        "done": False,
        "reasoning_chars": 0,
        "cache_before": None,
        "cache_after": None,
        "stats_before": None,
        "stats_after": None,
    }
    conn = http.client.HTTPConnection("127.0.0.1", 1919,
                                      timeout=REQUEST_TIMEOUT_SECONDS)
    started = time.perf_counter()
    first_content_at = None
    total_bytes = 0
    try:
        result["stats_before"] = get_json("/v1/stats")
        result["cache_before"] = get_json("/v1/cache/status")
        started = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json",
                              "Accept": "text/event-stream"})
        response = conn.getresponse()
        result["http_status"] = response.status
        if (response.status != 200
                or not response.getheader("Content-Type", "").startswith("text/event-stream")):
            result["error_body"] = response.read(4096).decode(errors="replace")
            raise RuntimeError("stream_http_or_content_type")
        while True:
            line = response.readline(1024 * 1024 + 1)
            now = time.perf_counter()
            total_bytes += len(line)
            if (len(line) > 1024 * 1024 or total_bytes > MAX_RESPONSE_BYTES
                    or now - started > REQUEST_TIMEOUT_SECONDS):
                raise RuntimeError("request_bound_exceeded")
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                result["done"] = True
                break
            event = json.loads(data)
            if event.get("error"):
                result["server_error"] = event["error"]
                raise RuntimeError("stream_server_error")
            if event.get("usage") is not None:
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if not isinstance(content, str):
                    raise RuntimeError("non_text_delta")
                if first_content_at is None and content.strip():
                    first_content_at = now
                result["full_output"] += content
                result["reasoning_chars"] += len(delta.get("reasoning_content") or "")
                if delta.get("tool_calls"):
                    raise RuntimeError("unexpected_tool_call")
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
        usage = result["usage"] or {}
        completion_tokens = usage.get("completion_tokens")
        if not (result["done"] and first_content_at is not None
                and result["finish_reason"] in ("stop", "length")
                and isinstance(completion_tokens, int) and 0 < completion_tokens <= MAX_TOKENS
                and result["reasoning_chars"] == 0):
            raise RuntimeError("stream_or_usage_incomplete_or_thinking_enabled")
        result["completion_tokens"] = completion_tokens
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        ended = time.perf_counter()
        conn.close()
        result["elapsed_seconds"] = ended - started
        if first_content_at is not None:
            result["first_content_ttft_seconds"] = first_content_at - started
            result["decode_elapsed_seconds"] = ended - first_content_at
            n = result["completion_tokens"]
            if n and n > 1 and result["decode_elapsed_seconds"] > 0:
                result["decode_tokens_per_second"] = ((n - 1)
                                                       / result["decode_elapsed_seconds"])
        try:
            result["cache_after"] = get_json("/v1/cache/status")
            result["stats_after"] = get_json("/v1/stats")
        except Exception as exc:
            result["status"] = "FAIL"
            result["observation_error"] = type(exc).__name__ + ": " + str(exc)
    emit(result)
    if result["status"] != "PASS":
        raise RuntimeError("terminal_request_failure_" + case_id)
    return result


def main() -> int:
    requests = selected_requests()
    if sys.argv[1:] == ["--profile-only"]:
        request = requests[0]
        stream("excluded-profile-"+request["case_id"],request["messages"],request["seed"],True)
        return 0
    if sys.argv[1:]:
        raise RuntimeError("unexpected_arguments")
    emit({"event": "matrix_scope", "excluded_warmups": 1,
          "measured_cases": list(CASE_IDS), "max_tokens": MAX_TOKENS,
          "temperature": 0, "thinking": False,
          "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS})
    warmup = requests[0]
    stream("excluded-warmup-" + warmup["case_id"], warmup["messages"],
           warmup["seed"], True)
    for request in requests:
        stream(request["case_id"], request["messages"], request["seed"], False)
    emit({"event": "matrix_complete", "stats": get_json("/v1/stats"),
          "cache": get_json("/v1/cache/status")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
