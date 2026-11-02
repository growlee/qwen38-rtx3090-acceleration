#!/usr/bin/env python3
"""Bounded JSONL quality client for one already-running Qwen candidate.

The client transports synthetic fixtures only.  It does not retry requests,
execute generated code, write files, or manage candidate/runtime state.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import http.client
import json
import math
import re
import socket
import sys
import time
from typing import Any, Mapping

from benchmark_client import BenchmarkError, Client, canonical, rounded
from quality_suite import SUITE


MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_SSE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_GENERATED_BYTES = 256 * 1024
WARMUP = {
    "case_id": "warmup",
    "category": "warmup",
    "critical": False,
    "request": {
        "messages": [{"role": "user", "content": "Ответь только словом: готов."}],
        "generation": {"temperature": 0, "seed": 8000, "max_tokens": 32},
    },
}


class QualityStreamError(RuntimeError):
    def __init__(self, code: str, *, partial: dict[str, Any], terminal: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.partial = partial
        self.terminal = terminal


def _json_type_matches(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return False


def _matches_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    type_name = schema.get("type")
    if not isinstance(type_name, str) or not _json_type_matches(value, type_name):
        return False
    if type_name == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return False
        if not set(required).issubset(value):
            return False
        if schema.get("additionalProperties") is False and not set(value).issubset(properties):
            return False
        return all(key not in value or _matches_schema(value[key], child) for key, child in properties.items())
    if type_name == "array" and "items" in schema:
        items = schema["items"]
        return isinstance(items, dict) and all(_matches_schema(item, items) for item in value)
    return True


def _strict_equal(actual: Any, expected: Any) -> bool:
    """JSON equality that never treats booleans as the numbers 0 and 1."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) or isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and isinstance(expected, dict)
            and set(actual) == set(expected)
            and all(_strict_equal(actual[key], expected[key]) for key in actual)
        )
    if isinstance(actual, list) or isinstance(expected, list):
        return (
            isinstance(actual, list)
            and isinstance(expected, list)
            and len(actual) == len(expected)
            and all(_strict_equal(left, right) for left, right in zip(actual, expected))
        )
    return actual == expected


def _eval_arithmetic(node: ast.AST, argument: str, value: int) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        if abs(node.value) > 1_000_000:
            raise ValueError("constant_out_of_range")
        return node.value
    if isinstance(node, ast.Name) and node.id == argument:
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_arithmetic(node.operand, argument, value)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)):
        left = _eval_arithmetic(node.left, argument, value)
        right = _eval_arithmetic(node.right, argument, value)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Pow):
            if right < 0 or right > 8 or abs(left) > 1_000_000:
                raise ValueError("power_out_of_range")
            result = left**right
            if abs(result) > 1_000_000_000_000:
                raise ValueError("power_result_out_of_range")
            return result
        if right == 0:
            raise ValueError("division_by_zero")
        return left // right if isinstance(node.op, ast.FloorDiv) else left % right
    raise ValueError("unsafe_expression")


def _check_lambda(content: str, check: Mapping[str, Any]) -> bool:
    if len(content) > 4096 or "```" in content:
        return False
    try:
        expression = ast.parse(content.strip(), mode="eval").body
        argument = check["argument"]
        if (
            not isinstance(expression, ast.Lambda)
            or len(expression.args.args) != 1
            or expression.args.args[0].arg != argument
            or expression.args.posonlyargs
            or expression.args.vararg is not None
            or expression.args.kwarg is not None
            or expression.args.kwonlyargs
            or expression.args.kw_defaults
            or expression.args.defaults
        ):
            return False
        return all(
            _eval_arithmetic(expression.body, argument, vector["input"]) == vector["expected"]
            for vector in check["cases"]
        )
    except (KeyError, SyntaxError, TypeError, ValueError):
        return False


def evaluate_noncode(case: Mapping[str, Any], response: Mapping[str, Any]) -> bool | None:
    """Evaluate safe checks; return None when external code sandbox is required."""
    check = case["check"]
    check_type = check["type"]
    if check_type == "python_function_cases":
        return None
    content = response.get("content")
    tool_calls = response.get("tool_calls")
    if check_type == "tool_call":
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            return False
        call = tool_calls[0]
        if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict):
            return False
        function = call["function"]
        if function.get("name") != check["function_name"]:
            return False
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return False
        return _strict_equal(arguments, check["arguments"])
    if not isinstance(content, str):
        return False
    if check_type == "exact":
        actual, expected = content.strip(), check["expected"].strip()
        if check["casefold"]:
            actual, expected = actual.casefold(), expected.casefold()
        return actual == expected
    if check_type == "json_shape":
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return False
        return _strict_equal(value, check["expected"]) and _matches_schema(value, check["schema"])
    if check_type == "text_constraints":
        stripped = content.strip()
        return (
            len(stripped.splitlines()) <= check["max_lines"]
            and all(re.fullmatch(pattern, stripped, re.DOTALL) for pattern in check["required_regex"])
            and not any(re.search(pattern, content, re.DOTALL) for pattern in check["forbidden_regex"])
        )
    if check_type == "python_lambda_cases":
        return _check_lambda(content, check)
    raise ValueError("check_type_unsupported")


def _numeric_usage(value: Mapping[str, Any], depth: int = 0) -> dict[str, Any]:
    """Keep bounded numeric usage trees, including completion token details."""
    if depth > 4:
        return {}
    result: dict[str, Any] = {}
    for key, item in list(value.items())[:64]:
        if not isinstance(key, str) or len(key) > 128:
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[key] = item
        elif isinstance(item, dict):
            nested = _numeric_usage(item, depth + 1)
            if nested:
                result[key] = nested
    return result


class QualityClient(Client):
    def _body(self, model: str, case: Mapping[str, Any]) -> dict[str, Any]:
        request = case["request"]
        generation = request["generation"]
        body: dict[str, Any] = {
            "model": model,
            "messages": request["messages"],
            "temperature": generation["temperature"],
            "seed": generation["seed"],
            "max_tokens": generation["max_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for key in ("tools", "tool_choice"):
            if key in request:
                body[key] = request[key]
        return body

    def stream_quality(self, model: str, case: Mapping[str, Any]) -> dict[str, Any]:
        body = self._body(model, case)
        started = time.perf_counter()
        deadline = started + self.timeout
        request_sha256 = hashlib.sha256(canonical(body)).hexdigest()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        first_generated_at: float | None = None
        first_content_or_tool_at: float | None = None
        response_bytes = 0
        generated_bytes = 0
        saw_done = False
        connection = self.connection()

        def tools() -> list[dict[str, Any]]:
            return [tool_parts[index] for index in sorted(tool_parts)]

        def partial() -> dict[str, Any]:
            ended = time.perf_counter()
            first = None if first_generated_at is None else rounded(first_generated_at - started)
            first_visible = None if first_content_or_tool_at is None else rounded(first_content_or_tool_at - started)
            return {
                "request_sha256": request_sha256,
                "content": "".join(content_parts),
                "reasoning": "".join(reasoning_parts),
                "tool_calls": tools(),
                "finish_reason": finish_reason,
                "usage": usage,
                "ttft_first_generated_seconds": first,
                "ttft_first_content_or_tool_seconds": first_visible,
                "elapsed_seconds": rounded(ended - started),
                "decode_tokens_per_second": None,
                "decode_rate_definition": "completion_tokens / seconds from first generated delta through stream end",
                "sse_bytes": response_bytes,
            }

        def fail(code: str, *, terminal: bool = False) -> QualityStreamError:
            return QualityStreamError(code, partial=partial(), terminal=terminal)

        def append_text(parts: list[str], value: str, *, visible: bool) -> None:
            nonlocal generated_bytes, first_generated_at, first_content_or_tool_at
            if not value:
                return
            encoded = len(value.encode("utf-8"))
            generated_bytes += encoded
            if generated_bytes > MAX_GENERATED_BYTES:
                raise fail("generated_content_too_large", terminal=True)
            now = time.perf_counter()
            if first_generated_at is None:
                first_generated_at = now
            if visible and first_content_or_tool_at is None:
                first_content_or_tool_at = now
            parts.append(value)

        try:
            connection.request("POST", "/v1/chat/completions", body=canonical(body), headers=self.headers("text/event-stream"))
            response = connection.getresponse()
            if response.status != 200:
                response.read(MAX_SSE_TOTAL_BYTES + 1)
                raise fail(f"http_{response.status}", terminal=response.status not in {400, 422})
            if "text/event-stream" not in (response.getheader("Content-Type", "") or "").lower():
                raise fail("content_type_invalid", terminal=True)
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise fail("timeout", terminal=True)
                sock = getattr(connection, "sock", None)
                if sock is not None:
                    sock.settimeout(remaining)
                line = response.readline(MAX_SSE_LINE_BYTES + 1)
                if not line:
                    break
                response_bytes += len(line)
                if len(line) > MAX_SSE_LINE_BYTES:
                    raise fail("sse_line_too_large", terminal=True)
                if response_bytes > MAX_SSE_TOTAL_BYTES:
                    raise fail("sse_total_too_large", terminal=True)
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    saw_done = True
                    break
                try:
                    event = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise fail("sse_invalid", terminal=True) from exc
                if not isinstance(event, dict):
                    raise fail("sse_invalid", terminal=True)
                choices = event.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            reasoning = delta.get("reasoning_content")
                            if not isinstance(reasoning, str):
                                reasoning = delta.get("reasoning")
                            if isinstance(reasoning, str):
                                append_text(reasoning_parts, reasoning, visible=False)
                            content = delta.get("content")
                            if isinstance(content, str):
                                append_text(content_parts, content, visible=True)
                            calls = delta.get("tool_calls")
                            if isinstance(calls, list):
                                for call in calls:
                                    if (
                                        not isinstance(call, dict)
                                        or not isinstance(call.get("index"), int)
                                        or isinstance(call.get("index"), bool)
                                        or call["index"] < 0
                                    ):
                                        raise fail("tool_delta_invalid", terminal=True)
                                    index = call["index"]
                                    assembled = tool_parts.setdefault(index, {"id": "", "type": "", "function": {"name": "", "arguments": ""}})
                                    for key in ("id", "type"):
                                        value = call.get(key)
                                        if isinstance(value, str):
                                            append_text([], value, visible=True)
                                            assembled[key] += value
                                    function = call.get("function")
                                    if isinstance(function, dict):
                                        for key in ("name", "arguments"):
                                            value = function.get(key)
                                            if isinstance(value, str):
                                                append_text([], value, visible=True)
                                                assembled["function"][key] += value
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = _numeric_usage(event_usage)
        except QualityStreamError:
            raise
        except (socket.timeout, TimeoutError) as exc:
            raise fail("timeout", terminal=True) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise fail("transport_error", terminal=True) from exc
        finally:
            connection.close()

        if not saw_done:
            raise fail("stream_incomplete", terminal=True)
        if finish_reason is None:
            raise fail("finish_reason_missing", terminal=True)
        if first_generated_at is None:
            raise fail("empty_output", terminal=True)
        if (
            not isinstance(usage, dict)
            or not isinstance(usage.get("completion_tokens"), int)
            or isinstance(usage.get("completion_tokens"), bool)
            or usage["completion_tokens"] <= 0
        ):
            raise fail("usage_missing", terminal=True)
        result = partial()
        ended = started + result["elapsed_seconds"]
        decode_seconds = max(ended - first_generated_at, 0.000001)
        result["decode_tokens_per_second"] = rounded(usage["completion_tokens"] / decode_seconds)
        return result


def run_case(client: QualityClient, model: str, mode_id: str, case: Mapping[str, Any], *, kind: str = "case") -> tuple[dict[str, Any], bool]:
    base = {"schema": "qwen38-quality-jsonl/v1", "kind": kind, "mode_id": mode_id, "model_id": model, "case_id": case["case_id"]}
    try:
        response = client.stream_quality(model, case)
    except QualityStreamError as exc:
        return {**base, "outcome": "error", "error": exc.code, **exc.partial}, exc.terminal
    if response["finish_reason"] == "length":
        return {**base, "outcome": "fail", "error": "finish_reason_length", **response}, False
    tool_finish = response["finish_reason"] == "tool_calls" and case.get("check", {}).get("type") == "tool_call"
    if response["finish_reason"] != "stop" and not tool_finish:
        return {**base, "outcome": "error", "error": "finish_reason_unsupported", **response}, True
    if kind == "warmup":
        return {**base, "outcome": "pass", "error": None, **response}, False
    evaluated = evaluate_noncode(case, response)
    if evaluated is None:
        outcome, error = "pending_sandbox", "python_sandbox_required"
    else:
        outcome, error = ("pass", None) if evaluated else ("fail", "quality_check_failed")
    return {**base, "outcome": outcome, "error": error, **response}, False


def _emit(row: Mapping[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode-id", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 180:
        parser.error("--timeout must be > 0 and <= 180 seconds")
    client = QualityClient(args.base_url, None, args.timeout)
    try:
        health_status, _, _ = client.get("/health")
        models_status, models_raw, _ = client.get("/v1/models")
    except BenchmarkError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    if health_status != 200 or models_status != 200:
        print("preflight_failed", file=sys.stderr)
        return 2
    try:
        models = json.loads(models_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("models_invalid", file=sys.stderr)
        return 2
    data = models.get("data") if isinstance(models, dict) else None
    if not isinstance(data, list) or not any(isinstance(item, dict) and item.get("id") == args.model for item in data):
        print("model_missing", file=sys.stderr)
        return 2

    warmup, terminal = run_case(client, args.model, args.mode_id, WARMUP, kind="warmup")
    _emit(warmup)
    if terminal or warmup["outcome"] != "pass":
        return 3
    for case in SUITE["cases"]:
        row, terminal = run_case(client, args.model, args.mode_id, case)
        _emit(row)
        if terminal:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
