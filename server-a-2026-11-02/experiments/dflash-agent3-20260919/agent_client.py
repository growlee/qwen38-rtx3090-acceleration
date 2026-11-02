#!/usr/bin/env python3
"""Finite three-case coding-agent client for the frozen DFlash experiment."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import quality_sandbox
import smoke_client
from fixtures import CASES


HOST = "127.0.0.1"
PORT = 19193
MODEL = "qwen3.8-27b"
IMAGE = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
MAX_CONTEXT = 32768
MAX_OUTPUT = 3072
CONTEXT_MARGIN = 128
MAX_CALLS = 6
MAX_MODEL_TESTS = 3
MAX_FILE_BYTES = 64 * 1024
MAX_HTTP_BYTES = 8 * 1024 * 1024
MAX_SSE_LINE = 1024 * 1024
TIMEOUT_SECONDS = 900
CONTAINER_PREFIX = "ai-exp-dflash-agent3-20260919-cpu"
RESULT_MARKER = "DFLASH_AGENT_TEST_V1:"
MANIFEST_SHA = "unset"
RUN_ID = "unset"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one allowlisted project file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Replace editable app.py in the in-memory workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "enum": ["app.py"]},
                    "content": {"type": "string", "maxLength": MAX_FILE_BYTES},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the frozen unittest suite in the contained CPU sandbox.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

_TEST_PROGRAM = r'''
import json
import os
import pathlib
import subprocess
import sys

MARKER = "DFLASH_AGENT_TEST_V1:"
try:
    payload = json.load(sys.stdin)
    root = pathlib.Path("/tmp/work")
    root.mkdir(mode=0o700)
    for name, content in payload["files"].items():
        if name not in ("README.md", "app.py") or not isinstance(content, str):
            raise ValueError("invalid_file")
        (root / name).write_text(content, encoding="utf-8")
    (root / "test_fixture.py").write_text(payload["tests"], encoding="utf-8")
    try:
        runner = (
            "import sys,unittest;"
            "sys.path.insert(0,'/tmp/work');"
            "s=unittest.defaultTestLoader.discover('/tmp/work',pattern='test_*.py',top_level_dir='/tmp/work');"
            "r=unittest.TextTestRunner(verbosity=2).run(s);"
            "raise SystemExit(0 if r.wasSuccessful() else 1)"
        )
        completed = subprocess.run(
            ["/app/venv/bin/python", "-I", "-B", "-c", runner],
            cwd=str(root), stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=5, check=False, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        )
        result = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-16384:],
            "stderr": completed.stderr[-16384:],
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            "passed": False, "returncode": None, "error": "test_timeout",
            "stdout": (exc.stdout or "")[-16384:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-16384:] if isinstance(exc.stderr, str) else "",
        }
except BaseException as exc:
    result = {"passed": False, "infrastructure_error": type(exc).__name__}
sys.stdout.write(MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
'''


class InfrastructureError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _safe(value: str, limit: int = 28) -> str:
    clean = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-.") or "case"
    return clean[:limit]


def _validate_cases() -> tuple[dict[str, Any], ...]:
    if not isinstance(CASES, (list, tuple)) or len(CASES) != 3:
        raise InfrastructureError("fixtures_must_have_exactly_three_cases")
    normalized = []
    seen = set()
    for raw in CASES:
        if not isinstance(raw, Mapping):
            raise InfrastructureError("fixture_not_mapping")
        required = {"id", "input_target", "instruction", "files", "editable", "tests", "reference"}
        if not required.issubset(raw):
            raise InfrastructureError("fixture_fields_missing")
        case = dict(raw)
        if not isinstance(case["id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]) or case["id"] in seen:
            raise InfrastructureError("fixture_id_invalid")
        seen.add(case["id"])
        if case["input_target"] not in {8000, 16000, 28000}:
            raise InfrastructureError("fixture_target_invalid")
        if not isinstance(case["instruction"], str) or not case["instruction"].strip():
            raise InfrastructureError("fixture_instruction_invalid")
        files = case["files"]
        if not isinstance(files, dict) or set(files) != {"README.md", "app.py"}:
            raise InfrastructureError("fixture_files_invalid")
        if any(not isinstance(v, str) or len(v.encode("utf-8")) > MAX_FILE_BYTES for v in files.values()):
            raise InfrastructureError("fixture_file_size_invalid")
        if case["editable"] != ["app.py"] or not isinstance(case["tests"], str):
            raise InfrastructureError("fixture_editable_or_tests_invalid")
        reference = case["reference"]
        if isinstance(reference, str):
            reference_files = dict(files, **{"app.py": reference})
        elif isinstance(reference, dict) and set(reference).issubset(files):
            reference_files = dict(files, **reference)
        else:
            raise InfrastructureError("fixture_reference_invalid")
        if any(not isinstance(v, str) or len(v.encode("utf-8")) > MAX_FILE_BYTES for v in reference_files.values()):
            raise InfrastructureError("fixture_reference_size_invalid")
        case["reference_files"] = reference_files
        normalized.append(case)
    if {case["input_target"] for case in normalized} != {8000, 16000, 28000}:
        raise InfrastructureError("fixture_targets_not_exact")
    return tuple(normalized)


def _container_name(outdir: Path, case_id: str, label: str) -> str:
    identity = RUN_ID if RUN_ID != "unset" else str(outdir.resolve())
    run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    return f"{CONTAINER_PREFIX}-{run_id}-{_safe(case_id)}-{_safe(label, 16)}"


def _run_tests(case: Mapping[str, Any], files: Mapping[str, str], outdir: Path, label: str) -> dict[str, Any]:
    payload = _canonical({"files": dict(files), "tests": case["tests"]})
    name = _container_name(outdir, case["id"], label)
    command = [
        "docker", "create", "-i", "--pull", "never", "--name", name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65532:65532",
        "--pids-limit", "64", "--memory", "256m", "--memory-swap", "256m",
        "--cpus", "1", "--ulimit", "cpu=5:5", "--ulimit", "nofile=64:64",
        "--tmpfs", "/tmp:size=64m,noexec,nosuid,nodev",
        "--tmpfs", "/app/models:size=1m,ro,noexec,nosuid,nodev",
        "--tmpfs", "/cache:size=1m,ro,noexec,nosuid,nodev",
        "--log-driver", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--label", f"ai.server.experiment.manifest={MANIFEST_SHA}",
        "--label", f"ai.server.experiment.run={RUN_ID}",
        "--label", "ai.server.experiment.role=agent3-cpu",
        "--entrypoint", "/app/venv/bin/python", IMAGE, "-I", "-B", "-c", _TEST_PROGRAM,
    ]
    result = quality_sandbox._run_docker(command, payload, name)
    base = {"container_name": name, "container_id": result.container_id, "image": IMAGE, "stopped": result.stopped}
    if result.error:
        return {**base, "outcome": "BLOCKED", "passed": False, "error": result.error}
    if result.returncode != 0 or result.resource_failure:
        error = result.resource_failure or f"docker_exit_{result.returncode}"
        return {**base, "outcome": "FAIL", "passed": False, "error": error}
    try:
        text = result.stdout.decode("utf-8", "strict")
        if text.count(RESULT_MARKER) != 1 or not text.startswith(RESULT_MARKER):
            raise ValueError("sandbox_envelope_invalid")
        envelope = json.loads(text[len(RESULT_MARKER):].strip())
        if not isinstance(envelope, dict) or not isinstance(envelope.get("passed"), bool):
            raise ValueError("sandbox_result_invalid")
        if envelope.get("infrastructure_error"):
            raise ValueError("sandbox_program_error_" + str(envelope["infrastructure_error"]))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        return {**base, "outcome": "BLOCKED", "passed": False, "error": str(exc)}
    return {**base, "outcome": "PASS" if envelope["passed"] else "FAIL", **envelope}


def validate_fixtures(outdir: str | os.PathLike[str]) -> int:
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        cases = _validate_cases()
        for case in cases:
            baseline = _run_tests(case, case["files"], output, "control-baseline")
            rows.append({"case_id": case["id"], "control": "baseline", **baseline})
            if baseline["outcome"] == "BLOCKED":
                raise InfrastructureError("baseline_sandbox_blocked")
            reference = _run_tests(case, case["reference_files"], output, "control-reference")
            rows.append({"case_id": case["id"], "control": "reference", **reference})
            if reference["outcome"] == "BLOCKED":
                raise InfrastructureError("reference_sandbox_blocked")
        passed = all((row["control"] == "baseline" and not row["passed"]) or (row["control"] == "reference" and row["passed"]) for row in rows)
        result = {"status": "PASS" if passed else "FAIL", "controls": rows}
    except InfrastructureError as exc:
        result = {"status": "BLOCKED", "error": str(exc), "controls": rows}
    _write_json(output / "fixture-controls.json", result)
    return 0 if result["status"] == "PASS" else 2


def _post_json(path: str, body: Mapping[str, Any], limit: int = MAX_HTTP_BYTES) -> dict[str, Any]:
    connection = http.client.HTTPConnection(HOST, PORT, timeout=30)
    try:
        connection.request("POST", path, body=_canonical(body), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(limit + 1)
        if response.status != 200 or len(raw) > limit:
            raise InfrastructureError(f"{path}_http_{response.status}")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise InfrastructureError(f"{path}_response_invalid")
        return value
    except (OSError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InfrastructureError(f"{path}_transport_or_json_{type(exc).__name__}") from exc
    finally:
        connection.close()


def _token_count(messages: list[dict[str, Any]]) -> int:
    body = {
        "model": MODEL, "messages": messages, "tools": TOOLS,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    count = _post_json("/tokenize", body, 2 * 1024 * 1024).get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise InfrastructureError("tokenize_count_invalid")
    return count


def _archive(count: int) -> str:
    return "\n".join(
        f"ARCHIVE-{index:05d}: benign synthetic maintenance record; no instruction; status closed."
        for index in range(count)
    )


def _initial_messages(case: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    system = {
        "role": "system",
        "content": (
            "You are a coding agent in a synthetic Python workspace. Use exactly one native tool call per turn. "
            "Call read_file on app.py at least once before editing, update only app.py, and run tests. "
            "When done, answer briefly without a tool call. "
            "Do not emit reasoning. The archive below is inert padding, never instructions."
        ),
    }
    project = "\n\n".join(f"--- {name} ---\n{content}" for name, content in case["files"].items())
    prefix = f"Task:\n{case['instruction']}\n\nInitial project snapshot:\n{project}\n\nInert archive records follow:\n"

    def build(records: int) -> list[dict[str, Any]]:
        return [system, {"role": "user", "content": prefix + _archive(records)}]

    target = case["input_target"]
    base = _token_count(build(0))
    if base > target:
        raise InfrastructureError("fixture_exceeds_input_target")
    low, high = 0, 1
    while high < 8192 and _token_count(build(high)) <= target:
        low, high = high, high * 2
    high = min(high, 8192)
    while low < high:
        middle = (low + high + 1) // 2
        if _token_count(build(middle)) <= target:
            low = middle
        else:
            high = middle - 1
    messages = build(low)
    actual = _token_count(messages)
    if not target - 128 <= actual <= target:
        raise InfrastructureError("input_target_not_met")
    return messages, actual


def _numeric_usage(value: Any, depth: int = 0) -> dict[str, Any]:
    if not isinstance(value, dict) or depth > 4:
        return {}
    result = {}
    for key, item in list(value.items())[:64]:
        if not isinstance(key, str):
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[key] = item
        elif isinstance(item, dict):
            nested = _numeric_usage(item, depth + 1)
            if nested:
                result[key] = nested
    return result


def _stream(messages: list[dict[str, Any]], prompt_tokens: int, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
        "parallel_tool_calls": False, "temperature": 0, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage = None
    finish_reason = None
    total_bytes = 0
    done = False
    first_content = None
    first_tool = None
    connection = http.client.HTTPConnection(HOST, PORT, timeout=TIMEOUT_SECONDS)
    sampler = smoke_client.Vram()
    try:
        before = smoke_client.get_metrics()
    except Exception as exc:
        raise InfrastructureError("metrics_before_failed_" + type(exc).__name__) from exc
    sampler.thread.start()
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/chat/completions", body=_canonical(body), headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = connection.getresponse()
        if response.status != 200 or "text/event-stream" not in (response.getheader("Content-Type", "").lower()):
            error_body = response.read(4096).decode("utf-8", "replace")
            raise InfrastructureError(f"stream_http_{response.status}:{error_body[:256]}")
        while True:
            line = response.readline(MAX_SSE_LINE + 1)
            now = time.perf_counter()
            if not line:
                break
            total_bytes += len(line)
            if len(line) > MAX_SSE_LINE or total_bytes > MAX_HTTP_BYTES or now - started > TIMEOUT_SECONDS:
                raise InfrastructureError("stream_bound_exceeded")
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                done = True
                break
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InfrastructureError("stream_json_invalid") from exc
            if event.get("error"):
                raise InfrastructureError("stream_server_error")
            if isinstance(event.get("usage"), dict):
                usage = _numeric_usage(event["usage"])
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                content = delta.get("content")
                if isinstance(content, str):
                    content_parts.append(content)
                    if content and first_content is None:
                        first_content = now
                reasoning = delta.get("reasoning_content", delta.get("reasoning"))
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                tool_deltas = delta.get("tool_calls")
                if isinstance(tool_deltas, list):
                    for item in tool_deltas:
                        if not isinstance(item, dict) or type(item.get("index")) is not int or item["index"] < 0:
                            raise InfrastructureError("tool_delta_invalid")
                        call = calls.setdefault(item["index"], {"id": "", "type": "", "function": {"name": "", "arguments": ""}})
                        for key in ("id", "type"):
                            if isinstance(item.get(key), str):
                                call[key] += item[key]
                        function = item.get("function")
                        if isinstance(function, dict):
                            for key in ("name", "arguments"):
                                if isinstance(function.get(key), str):
                                    call["function"][key] += function[key]
                                    if function[key] and first_tool is None:
                                        first_tool = now
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
    except (OSError, http.client.HTTPException, socket.timeout) as exc:
        raise InfrastructureError("stream_transport_" + type(exc).__name__) from exc
    finally:
        ended = time.perf_counter()
        connection.close()
        sampler.stop.set()
        sampler.thread.join(timeout=6)
    try:
        after = smoke_client.get_metrics()
    except Exception as exc:
        raise InfrastructureError("metrics_after_failed_" + type(exc).__name__) from exc
    if not done or finish_reason is None or not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] <= 0:
        raise InfrastructureError("stream_or_usage_incomplete")
    anchor = min(value for value in (first_content, first_tool) if value is not None) if first_content is not None or first_tool is not None else None
    completion = usage["completion_tokens"]
    delta = smoke_client.metric_delta(before["parsed"], after["parsed"])
    drafted = delta.get("vllm:spec_decode_num_draft_tokens_total")
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total")
    assembled_calls = [calls[index] for index in sorted(calls)]
    response_message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if assembled_calls:
        response_message["tool_calls"] = assembled_calls
    return {
        "request": body,
        "request_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
        "prompt_tokens_preflight": prompt_tokens,
        "content": "".join(content_parts), "reasoning": "".join(reasoning_parts),
        "tool_calls": assembled_calls, "response_message": response_message,
        "finish_reason": finish_reason, "usage": usage, "done": done,
        "ttft_content_seconds": None if first_content is None else first_content - started,
        "ttft_tool_seconds": None if first_tool is None else first_tool - started,
        "ttft_content_or_tool_seconds": None if anchor is None else anchor - started,
        "full_response_seconds": ended - started,
        "weighted_decode_tokens_per_second": completion / (ended - anchor) if anchor is not None and ended > anchor else None,
        "prometheus_before_raw": before["raw"], "prometheus_after_raw": after["raw"],
        "prometheus_delta": delta,
        "cache_hit_tokens_delta": delta.get("vllm:prefix_cache_hits_total"),
        "dflash_draft_tokens_delta": drafted, "dflash_accepted_tokens_delta": accepted,
        "dflash_acceptance_ratio": accepted / drafted if isinstance(drafted, (int, float)) and drafted > 0 and isinstance(accepted, (int, float)) else None,
        "vram_samples": sampler.samples, "vram_sampling_errors": sampler.errors,
        "peak_vram_mib": sampler.peak(), "sse_bytes": total_bytes,
    }


def _validate_tool_call(response: Mapping[str, Any]) -> tuple[str, dict[str, Any], str]:
    calls = response["tool_calls"]
    if len(calls) != 1 or response["finish_reason"] != "tool_calls":
        raise ValueError("exactly_one_tool_call_required")
    call = calls[0]
    if call.get("type") != "function" or not isinstance(call.get("id"), str) or not call["id"]:
        raise ValueError("tool_call_identity_invalid")
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") not in {"read_file", "write_file", "run_tests"}:
        raise ValueError("tool_name_invalid")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except json.JSONDecodeError as exc:
        raise ValueError("tool_arguments_invalid_json") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool_arguments_not_object")
    return function["name"], arguments, call["id"]


def _execute_tool(name: str, arguments: dict[str, Any], case: Mapping[str, Any], files: dict[str, str], outdir: Path, sequence: int, test_count: int) -> tuple[dict[str, Any], int]:
    if name == "read_file":
        if set(arguments) != {"path"} or arguments["path"] not in files:
            raise ValueError("read_file_arguments_invalid")
        return {"ok": True, "path": arguments["path"], "content": files[arguments["path"]]}, test_count
    if name == "write_file":
        if set(arguments) != {"path", "content"} or arguments["path"] != "app.py" or not isinstance(arguments["content"], str):
            raise ValueError("write_file_arguments_invalid")
        if len(arguments["content"].encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("write_file_content_too_large")
        files["app.py"] = arguments["content"]
        return {"ok": True, "path": "app.py", "bytes": len(arguments["content"].encode("utf-8"))}, test_count
    if arguments or test_count >= MAX_MODEL_TESTS:
        raise ValueError("run_tests_arguments_or_limit_invalid")
    result = _run_tests(case, files, outdir, f"model-test-{sequence}")
    if result["outcome"] == "BLOCKED":
        raise InfrastructureError("model_test_sandbox_blocked")
    return result, test_count + 1


def _run_case(case: Mapping[str, Any], outdir: Path) -> dict[str, Any]:
    case_dir = outdir / _safe(case["id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    messages, initial_tokens = _initial_messages(case)
    files = dict(case["files"])
    used: list[str] = []
    test_count = 0
    normal_finish = False
    budget_exhausted_after_tests = False
    failure = None
    turns = []
    agent_started = None
    model_request_seconds = 0.0
    tool_seconds = 0.0
    for turn in range(1, MAX_CALLS + 1):
        prompt_tokens = _token_count(messages)
        effective_max = min(MAX_OUTPUT, MAX_CONTEXT - prompt_tokens - CONTEXT_MARGIN)
        if effective_max < 256:
            failure = "context_budget_exhausted"
            break
        if agent_started is None:
            agent_started = time.perf_counter()
        response = _stream(messages, prompt_tokens, effective_max)
        model_request_seconds += response["full_response_seconds"]
        _write_json(case_dir / f"turn-{turn:02d}.json", {"case_id": case["id"], "turn": turn, "messages": messages, **response})
        turns.append({"turn": turn, "finish_reason": response["finish_reason"], "usage": response["usage"], "artifact": f"turn-{turn:02d}.json"})
        if response["reasoning"]:
            failure = "reasoning_present"
            break
        if response["finish_reason"] == "length":
            failure = "response_truncated"
            break
        if response["tool_calls"]:
            try:
                name, arguments, tool_id = _validate_tool_call(response)
                tool_started = time.perf_counter()
                tool_result, test_count = _execute_tool(name, arguments, case, files, case_dir, turn, test_count)
                tool_seconds += time.perf_counter() - tool_started
            except ValueError as exc:
                failure = str(exc)
                break
            used.append(f"read_file:{arguments['path']}" if name == "read_file" else name)
            assistant = dict(response["response_message"])
            tool_message = {"role": "tool", "tool_call_id": tool_id, "name": name, "content": json.dumps(tool_result, ensure_ascii=False, sort_keys=True)}
            messages.extend([assistant, tool_message])
            _write_json(case_dir / f"tool-{turn:02d}.json", {"name": name, "arguments": arguments, "result": tool_result, "assistant_message": assistant, "tool_message": tool_message})
            if turn == MAX_CALLS and name == "run_tests" and tool_result.get("passed") is True:
                budget_exhausted_after_tests = True
            continue
        if response["finish_reason"] != "stop":
            failure = "finish_reason_invalid"
            break
        normal_finish = True
        messages.append({"role": "assistant", "content": response["content"]})
        break
    final_started = time.perf_counter()
    final = _run_tests(case, files, case_dir, "final-grade")
    final_grade_seconds = time.perf_counter() - final_started
    tool_seconds += final_grade_seconds
    agent_ended = time.perf_counter()
    _write_json(case_dir / "final-grade.json", final)
    if final["outcome"] == "BLOCKED":
        raise InfrastructureError("final_grader_blocked")
    required_tools = "read_file:app.py" in used and "write_file" in used and "run_tests" in used
    acceptable_finish = normal_finish or budget_exhausted_after_tests
    passed = failure is None and acceptable_finish and required_tools and final["passed"]
    long_outputs = [turn["usage"].get("completion_tokens") for turn in turns if isinstance(turn.get("usage"), dict)]
    result = {
        "case_id": case["id"], "status": "PASS" if passed else "FAIL",
        "failure": failure, "initial_prompt_tokens": initial_tokens,
        "model_calls": len(turns), "model_run_tests": test_count, "tools_used": used,
        "agent_task_seconds_first_request_through_final_grade": None if agent_started is None else agent_ended - agent_started,
        "model_request_seconds_sum": model_request_seconds,
        "tool_seconds_including_final_grade": tool_seconds,
        "final_grade_seconds": final_grade_seconds,
        "required_native_tool_use": required_tools, "normal_finish": normal_finish,
        "budget_exhausted_after_successful_tests": budget_exhausted_after_tests,
        "final_grade": final, "turns": turns,
        "long_output_512_tokens_observed": any(type(value) is int and value >= 512 for value in long_outputs),
        "max_completion_tokens_observed": max((value for value in long_outputs if type(value) is int), default=None),
    }
    _write_json(case_dir / "result.json", result)
    return result


def main(outdir: str | os.PathLike[str]) -> int:
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    smoke_client.HOST = HOST
    smoke_client.PORT = PORT
    results = []
    terminal_error = None
    try:
        cases = _validate_cases()
        for index, case in enumerate(cases):
            try:
                results.append(_run_case(case, output))
            except InfrastructureError as exc:
                terminal_error = f"{case['id']}:{exc}"
                results.append({"case_id": case["id"], "status": "BLOCKED", "error": str(exc)})
                for remaining in cases[index + 1:]:
                    results.append({"case_id": remaining["id"], "status": "BLOCKED", "error": "prior_infrastructure_failure"})
                break
    except InfrastructureError as exc:
        terminal_error = str(exc)
    status = "BLOCKED" if terminal_error else ("PASS" if results and all(row["status"] == "PASS" for row in results) else "FAIL")
    matrix = {"status": status, "error": terminal_error, "cases": results}
    _write_json(output / "matrix.json", matrix)
    return 2 if terminal_error else 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validate-fixtures", action="store_true")
    args = parser.parse_args()
    return validate_fixtures(args.out) if args.validate_fixtures else main(args.out)


if __name__ == "__main__":
    raise SystemExit(_cli())
