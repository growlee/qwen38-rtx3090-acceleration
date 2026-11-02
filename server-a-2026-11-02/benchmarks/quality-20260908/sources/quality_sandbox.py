"""Docker boundary for scoring synthetic model-generated Python functions.

The candidate source is never imported or executed by this host process.  Each
evaluation creates one stopped, retained container whose name includes the
matrix mode and case identifiers.  Callers must treat ``BLOCKED`` as missing
infrastructure evidence, not as a model failure, and must not retry a case.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from typing import Any


PINNED_IMAGE = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
MAX_SOURCE_BYTES = 64 * 1024
MAX_DOCKER_OUTPUT_BYTES = 64 * 1024
DOCKER_WALL_SECONDS = 10.0
CONTAINER_PREFIX = "qwen-quality-20260908-code"
_RESULT_MARKER = "QWEN_QUALITY_RESULT_V1:"


_CONTAINER_PROGRAM = r'''
import contextlib
import json
import os
import signal
import sys

MARKER = "QWEN_QUALITY_RESULT_V1:"

def same(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same(actual[k], expected[k]) for k in expected)
    return actual == expected

def compact(value):
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except BaseException:
        return {"type": type(value).__name__, "repr": "<not-json-serializable>"}
    if len(encoded.encode("utf-8")) > 4096:
        return {"type": type(value).__name__, "repr": "<value-too-large>"}
    return value

def main():
    def wall_timeout(_signum, _frame):
        raise TimeoutError("candidate wall timeout")
    signal.signal(signal.SIGALRM, wall_timeout)
    signal.alarm(5)
    payload = json.load(sys.stdin)
    namespace = {}
    vectors = []
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(payload["source"], "<candidate>", "exec"), namespace, namespace)
        function = namespace[payload["function_name"]]
        for index, vector in enumerate(payload["vectors"]):
            row = {"index": index}
            try:
                actual = function(*vector["args"], **vector["kwargs"])
                if "expected_exception" in vector:
                    row.update({"passed": False, "error": "expected_exception_not_raised", "actual": compact(actual)})
                else:
                    passed = same(actual, vector["expected"])
                    row.update({"passed": passed, "actual": compact(actual)})
                    if not passed:
                        row["error"] = "value_mismatch"
            except BaseException as exc:
                wanted = vector.get("expected_exception")
                passed = wanted == "ValueError" and type(exc) is ValueError
                row.update({"passed": passed, "exception": type(exc).__name__})
                if not passed:
                    row["error"] = "unexpected_exception"
            vectors.append(row)
    result = {"passed": all(row["passed"] for row in vectors), "vectors": vectors}
    sys.stdout.write(MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")

try:
    main()
except BaseException as exc:
    sys.stdout.write(MARKER + json.dumps({"passed": False, "vectors": [], "error": "candidate_runtime", "exception": type(exc).__name__}, separators=(",", ":")) + "\n")
'''


@dataclass(frozen=True)
class _DockerResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    error: str | None = None
    resource_failure: str | None = None
    container_id: str | None = None
    stopped: bool = False


def _safe_component(value: str, limit: int = 44) -> str:
    raw = str(value).lower()
    safe = re.sub(r"[^a-z0-9_.-]+", "-", raw).strip("-.") or "unknown"
    if len(safe) <= limit:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe[: limit - 9]}-{digest}"


def _container_name(mode_id: str, case_id: str) -> str:
    return f"{CONTAINER_PREFIX}-{_safe_component(mode_id)}-{_safe_component(case_id)}"


def _extract_source(content: str) -> str:
    if not isinstance(content, str):
        raise ValueError("response_content_not_string")
    if len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("response_content_too_large")
    stripped = content.strip()
    if "```" in stripped:
        match = re.fullmatch(r"```(?:python|py)?[ \t]*\r?\n([\s\S]*?)\r?\n```", stripped, re.IGNORECASE)
        if not match:
            raise ValueError("response_must_be_one_code_block_without_prose")
        stripped = match.group(1).strip()
    if not stripped:
        raise ValueError("empty_source")
    return stripped


def _validate_source(source: str, function_name: str) -> None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError("invalid_python_syntax") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("source_must_define_exactly_one_function")
    function = tree.body[0]
    if function.name != function_name:
        raise ValueError("wrong_function_name")
    if function.decorator_list:
        raise ValueError("decorators_not_allowed")


def _validate_case(case: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    try:
        case_id = str(case["case_id"])
        check = case["check"]
        function_name = str(check["function_name"])
        vectors = check["cases"]
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid_case_schema") from exc
    if check.get("type") != "python_function_cases" or not isinstance(vectors, list) or not vectors:
        raise ValueError("unsupported_case_schema")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name):
        raise ValueError("invalid_function_name")
    normalized = []
    for vector in vectors:
        if not isinstance(vector, dict) or set(vector).intersection(("expected", "expected_exception")) not in ({"expected"}, {"expected_exception"}):
            raise ValueError("vector_requires_exactly_one_expected_outcome")
        if vector.get("expected_exception", "ValueError") != "ValueError":
            raise ValueError("only_value_error_is_supported")
        args = vector.get("args", [])
        kwargs = vector.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError("invalid_vector_arguments")
        normalized.append(dict(vector, args=args, kwargs=kwargs))
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    return case_id, function_name, normalized


def _kill_owned_container(container_id: str) -> bool:
    """Contain an exact container ID created by this invocation; retain it."""
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        return False
    try:
        killed = subprocess.run(
            ["docker", "kill", container_id], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=False,
        )
        return killed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _short_command(command: list[str], timeout: float = 3.0) -> _DockerResult:
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
        return _DockerResult(result.returncode, result.stdout[:MAX_DOCKER_OUTPUT_BYTES], result.stderr[:MAX_DOCKER_OUTPUT_BYTES])
    except (OSError, subprocess.SubprocessError) as exc:
        return _DockerResult(None, b"", b"", f"docker_command:{type(exc).__name__}")


def _stopped_state(container_id: str) -> tuple[bool, dict[str, Any] | None]:
    inspected = _short_command(["docker", "inspect", "--format", "{{json .State}}", container_id])
    if inspected.error or inspected.returncode != 0:
        return False, None
    try:
        state = json.loads(inspected.stdout.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, None
    return isinstance(state, dict) and state.get("Running") is False, state


def _run_docker(command: list[str], payload: bytes, container_name: str) -> _DockerResult:
    # This read-only preflight prevents touching a pre-existing same-name object.
    preflight = _short_command([
        "docker", "ps", "-aq", "--no-trunc", "--filter", f"name=^/{container_name}$",
    ])
    if preflight.error or preflight.returncode != 0:
        return _DockerResult(preflight.returncode, b"", preflight.stderr, preflight.error or "container_preflight_failed")
    if preflight.stdout.strip():
        return _DockerResult(None, b"", b"", "container_name_already_exists")

    created = _short_command(command)
    if created.error or created.returncode != 0:
        return _DockerResult(created.returncode, created.stdout, created.stderr, created.error or "container_create_failed")
    container_id = created.stdout.decode("ascii", "ignore").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        return _DockerResult(created.returncode, created.stdout, created.stderr, "invalid_created_container_id")

    start_command = ["docker", "start", "-a", "-i", container_id]
    try:
        process = subprocess.Popen(start_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        stopped, _ = _stopped_state(container_id)
        return _DockerResult(None, b"", b"", f"docker_start:{type(exc).__name__}", container_id=container_id, stopped=stopped)
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    os.set_blocking(process.stdin.fileno(), False)
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stderr.fileno(), False)
    selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    total = 0
    sent = 0
    deadline = time.monotonic() + DOCKER_WALL_SECONDS
    error = None
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = "sandbox_wall_timeout"
                break
            for key, _ in selector.select(min(remaining, 0.25)):
                if key.data == "stdin":
                    try:
                        sent += os.write(key.fileobj.fileno(), payload[sent:sent + 8192])
                    except BlockingIOError:
                        continue
                    if sent == len(payload):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > MAX_DOCKER_OUTPUT_BYTES:
                    error = "sandbox_output_limit"
                    break
                chunks[key.data].append(chunk)
            if error:
                break
        if error:
            contained = _kill_owned_container(container_id)
            process.kill()
        else:
            contained = True
        process.wait(timeout=2)
    except (OSError, subprocess.SubprocessError):
        error = error or "docker_io_failure"
        contained = _kill_owned_container(container_id)
        process.kill()
        process.wait()
    finally:
        selector.close()
    stopped, state = _stopped_state(container_id)
    if not stopped or (error and not contained):
        return _DockerResult(process.returncode, b"".join(chunks["stdout"]), b"".join(chunks["stderr"]), "containment_unproven", container_id=container_id, stopped=stopped)
    resource_failure = error
    if not resource_failure and state and state.get("OOMKilled") is True:
        resource_failure = "candidate_oom"
    return _DockerResult(
        process.returncode, b"".join(chunks["stdout"]), b"".join(chunks["stderr"]),
        resource_failure=resource_failure, container_id=container_id, stopped=True,
    )


def evaluate_code(
    case: dict[str, Any],
    response_content: str,
    image: str = PINNED_IMAGE,
    *,
    mode_id: str = "unknown",
) -> dict[str, Any]:
    """Score one ``python_function_cases`` response in an isolated container."""
    try:
        case_id, function_name, vectors = _validate_case(case)
    except (TypeError, ValueError, OverflowError) as exc:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        return {"case_id": case_id, "mode_id": mode_id, "outcome": "BLOCKED", "passed": False, "error": f"fixture:{exc}"}
    try:
        source = _extract_source(response_content)
        _validate_source(source, function_name)
    except (TypeError, ValueError, OverflowError) as exc:
        return {"case_id": case_id, "mode_id": mode_id, "outcome": "FAIL", "passed": False, "error": str(exc)}

    container_name = _container_name(mode_id, case_id)
    payload = json.dumps(
        {"source": source, "function_name": function_name, "vectors": vectors},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    command = [
        "docker", "create", "-i", "--pull", "never", "--name", container_name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65532:65532",
        "--pids-limit", "32", "--memory", "256m", "--memory-swap", "256m",
        "--cpus", "1", "--ulimit", "cpu=5:5", "--ulimit", "nofile=64:64",
        "--tmpfs", "/tmp:size=16m,noexec,nosuid,nodev",
        "--tmpfs", "/app/models:size=1m,noexec,nosuid,nodev",
        "--tmpfs", "/cache:size=1m,noexec,nosuid,nodev",
        "--log-driver", "none",
        "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--entrypoint", "/app/venv/bin/python", image, "-I", "-S", "-c", _CONTAINER_PROGRAM,
    ]
    result = _run_docker(command, payload, container_name)
    base = {"case_id": case_id, "mode_id": mode_id, "container_name": container_name, "container_id": result.container_id, "image": image}
    if result.error:
        return {**base, "outcome": "BLOCKED", "passed": False, "error": result.error}
    if result.resource_failure:
        return {**base, "outcome": "FAIL", "passed": False, "error": result.resource_failure, "resource_limited": True}
    if result.returncode != 0:
        detail = result.error or f"docker_exit_{result.returncode}"
        return {**base, "outcome": "BLOCKED", "passed": False, "error": detail}
    try:
        output = result.stdout.decode("utf-8", "strict")
        if not output.startswith(_RESULT_MARKER) or output.count(_RESULT_MARKER) != 1:
            raise ValueError("invalid_sandbox_envelope")
        envelope = json.loads(output[len(_RESULT_MARKER):].strip())
        if not isinstance(envelope, dict) or not isinstance(envelope.get("passed"), bool) or not isinstance(envelope.get("vectors"), list):
            raise ValueError("invalid_sandbox_result")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        return {**base, "outcome": "BLOCKED", "passed": False, "error": str(exc)}
    return {**base, "outcome": "PASS" if envelope["passed"] else "FAIL", **envelope}
