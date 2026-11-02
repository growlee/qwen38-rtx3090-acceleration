"""Fail-closed, content-free smoke test for generic OpenAI-compatible candidates."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import ssl
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


RUNNER_VERSION = "1.0.0"
CONFIG_VERSION = 1
RECEIPT_VERSION = 1
MAX_CONFIG_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 16_777_216
MAX_SSE_LINE_BYTES = 1_048_576
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}$")
ALLOWED_CANDIDATES = {"syv", "freetoken"}
CAPABILITY_NAMES = {
    "responses_api",
    "constrained_response_format",
    "tools",
    "vision",
}

# Public, versioned smoke material. Text is used transiently and is never receipted.
_PROMPTS = {
    "basic": "Reply with exactly BASIC_OK.",
    "stream": "Reply with exactly STREAM_OK, emitted normally.",
    "sequential": "Reply with exactly SEQUENTIAL_OK.",
    "cancel": "Write a long numbered list from 1 to 200 without commentary.",
    "recovery": "Reply with exactly RECOVERY_OK.",
    "json": (
        "Return only valid JSON with exactly these fields and values: "
        '{"status":"ok","value":7}'
    ),
    "harmony": "Reply with exactly HARMONY_OK.",
    "format": "Return a JSON object with string field status equal to ok.",
    "tool": "Call the report_status function once with status equal to ok.",
    "vision": "Reply with exactly VISION_OK after inspecting this image.",
}
_SUITE_SHA256 = hashlib.sha256(
    json.dumps(_PROMPTS, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()
_ONE_PIXEL_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


class ConfigurationError(ValueError):
    pass


class SmokeError(RuntimeError):
    """An error whose code is safe to persist."""

    def __init__(self, code: str, http_status: int | None = None):
        super().__init__(code)
        self.code = code
        self.http_status = http_status


@dataclass(frozen=True)
class Config:
    candidate_id: str
    candidate_type: str
    runtime_id: str
    manifest_id: str
    base_url: str
    model_id: str
    auth_mode: str
    token_file: Path | None
    connect_timeout_seconds: float
    request_timeout_seconds: float
    cancel_after_deltas: int
    capabilities: Mapping[str, bool]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ConfigurationError(code)


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ConfigurationError(f"{field}_invalid")
    return value


def _bounded_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ConfigurationError(f"{field}_invalid")
    return float(value)


def load_config(path: Path) -> tuple[Config, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError("config_unreadable") from exc
    if not raw or len(raw) > MAX_CONFIG_BYTES:
        raise ConfigurationError("config_size_invalid")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("config_json_invalid") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("config_invalid")
    _exact_keys(
        value,
        {
            "schema_version",
            "candidate_id",
            "candidate_type",
            "runtime_id",
            "manifest_id",
            "base_url",
            "model_id",
            "auth_mode",
            "token_file",
            "connect_timeout_seconds",
            "request_timeout_seconds",
            "cancel_after_deltas",
            "capabilities",
        },
        "config_keys_invalid",
    )
    if value["schema_version"] != CONFIG_VERSION:
        raise ConfigurationError("config_version_unsupported")
    candidate_type = value["candidate_type"]
    if candidate_type not in ALLOWED_CANDIDATES:
        raise ConfigurationError("candidate_type_invalid")
    base_url = value["base_url"]
    if not isinstance(base_url, str) or len(base_url) > 2048:
        raise ConfigurationError("base_url_invalid")
    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("base_url_invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or base_url.endswith("/")
        or (port is not None and not 1 <= port <= 65535)
        or (parsed.scheme == "http" and parsed.hostname != "127.0.0.1")
    ):
        raise ConfigurationError("base_url_invalid")
    auth_mode = value["auth_mode"]
    if auth_mode not in {"bearer", "none_loopback"}:
        raise ConfigurationError("auth_mode_invalid")
    token_file_raw = value["token_file"]
    token_file: Path | None
    if auth_mode == "none_loopback":
        if parsed.hostname != "127.0.0.1" or token_file_raw is not None:
            raise ConfigurationError("none_loopback_scope_invalid")
        token_file = None
    else:
        if not isinstance(token_file_raw, str) or not token_file_raw or len(token_file_raw) > 4096:
            raise ConfigurationError("token_file_invalid")
        token_file = Path(token_file_raw)
    capabilities = value["capabilities"]
    if not isinstance(capabilities, dict):
        raise ConfigurationError("capabilities_invalid")
    _exact_keys(capabilities, CAPABILITY_NAMES, "capabilities_keys_invalid")
    if any(not isinstance(item, bool) for item in capabilities.values()):
        raise ConfigurationError("capabilities_invalid")
    if candidate_type == "freetoken" and capabilities["constrained_response_format"]:
        raise ConfigurationError("freetoken_response_format_forbidden")
    cancel_after_deltas = value["cancel_after_deltas"]
    if (
        not isinstance(cancel_after_deltas, int)
        or isinstance(cancel_after_deltas, bool)
        or not 1 <= cancel_after_deltas <= 32
    ):
        raise ConfigurationError("cancel_after_deltas_invalid")
    model_id = value["model_id"]
    if not isinstance(model_id, str) or not model_id or len(model_id) > 512:
        raise ConfigurationError("model_id_invalid")
    config = Config(
        candidate_id=_safe_id(value["candidate_id"], "candidate_id"),
        candidate_type=candidate_type,
        runtime_id=_safe_id(value["runtime_id"], "runtime_id"),
        manifest_id=_safe_id(value["manifest_id"], "manifest_id"),
        base_url=base_url,
        model_id=model_id,
        auth_mode=auth_mode,
        token_file=token_file,
        connect_timeout_seconds=_bounded_number(
            value["connect_timeout_seconds"], "connect_timeout_seconds", 0.1, 30
        ),
        request_timeout_seconds=_bounded_number(
            value["request_timeout_seconds"], "request_timeout_seconds", 1, 3600
        ),
        cancel_after_deltas=cancel_after_deltas,
        capabilities=dict(capabilities),
    )
    digest_input = dict(value)
    # A local secret path is operational input, not evidence identity.
    digest_input["token_file"] = None if token_file is None else "<redacted>"
    return config, hashlib.sha256(_canonical_bytes(digest_input)).hexdigest()


def _read_secret(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigurationError("token_file_unreadable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= 4096:
            raise ConfigurationError("token_file_invalid")
        if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
            raise ConfigurationError("token_file_permissions_invalid")
        raw = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    if not raw or len(raw) > 4096 or b"\n" in raw.strip() or b"\r" in raw.strip():
        raise ConfigurationError("token_file_invalid")
    try:
        token = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ConfigurationError("token_file_invalid") from exc
    if not token:
        raise ConfigurationError("token_file_invalid")
    return token


class Client:
    def __init__(self, config: Config, token: str | None):
        self.config = config
        self.token = token
        self.parsed = urlsplit(config.base_url)

    def _connection(self) -> http.client.HTTPConnection:
        port = self.parsed.port or (443 if self.parsed.scheme == "https" else 80)
        if self.parsed.scheme == "https":
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                self.parsed.hostname,
                port,
                timeout=self.config.connect_timeout_seconds,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(
                self.parsed.hostname, port, timeout=self.config.connect_timeout_seconds
            )
        return connection

    def _headers(self, *, accept: str) -> dict[str, str]:
        headers = {"Accept": accept, "Content-Type": "application/json"}
        if self.token is not None:
            headers["Authorization"] = "Bearer " + self.token
        return headers

    @staticmethod
    def _validate_response(response: http.client.HTTPResponse, expected_type: str) -> None:
        if 300 <= response.status <= 399:
            raise SmokeError("redirect_forbidden", response.status)
        if response.status != 200:
            raise SmokeError("http_status_invalid", response.status)
        encoding = response.getheader("Content-Encoding")
        if encoding and encoding.lower() != "identity":
            raise SmokeError("content_encoding_forbidden", response.status)
        content_type = (response.getheader("Content-Type") or "").lower()
        if expected_type not in content_type:
            raise SmokeError("content_type_invalid", response.status)
        length = response.getheader("Content-Length")
        if length is not None:
            try:
                parsed_length = int(length)
            except ValueError as exc:
                raise SmokeError("content_length_invalid", response.status) from exc
            if parsed_length < 0 or parsed_length > MAX_RESPONSE_BYTES:
                raise SmokeError("response_too_large", response.status)

    @staticmethod
    def _read_limited(response: http.client.HTTPResponse) -> bytes:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise SmokeError("response_too_large", response.status)
        return raw

    def json_request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], float]:
        connection = self._connection()
        payload = None if body is None else _canonical_bytes(body)
        started = time.perf_counter()
        try:
            connection.request(
                method,
                path,
                body=payload,
                headers=self._headers(accept="application/json"),
            )
            if connection.sock is not None:
                connection.sock.settimeout(self.config.request_timeout_seconds)
            response = connection.getresponse()
            self._validate_response(response, "application/json")
            raw = self._read_limited(response)
        except SmokeError:
            raise
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            raise SmokeError("transport_error") from exc
        finally:
            connection.close()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmokeError("response_json_invalid") from exc
        if not isinstance(value, dict):
            raise SmokeError("response_shape_invalid")
        return value, time.perf_counter() - started

    def stream_request(
        self, body: Mapping[str, Any], *, cancel_after: int | None = None
    ) -> dict[str, Any]:
        connection = self._connection()
        payload = _canonical_bytes(body)
        started = time.perf_counter()
        first_delta: float | None = None
        deltas = 0
        response_bytes = 0
        saw_done = False
        saw_usage = False
        saw_finish = False
        completion_tokens: int | None = None
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers=self._headers(accept="text/event-stream"),
            )
            if connection.sock is not None:
                connection.sock.settimeout(self.config.request_timeout_seconds)
            response = connection.getresponse()
            self._validate_response(response, "text/event-stream")
            while True:
                line = response.readline(MAX_SSE_LINE_BYTES + 1)
                if not line:
                    break
                response_bytes += len(line)
                if len(line) > MAX_SSE_LINE_BYTES or response_bytes > MAX_RESPONSE_BYTES:
                    raise SmokeError("response_too_large", response.status)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    saw_done = True
                    break
                try:
                    event = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SmokeError("sse_event_invalid", response.status) from exc
                if not isinstance(event, dict):
                    raise SmokeError("sse_event_invalid", response.status)
                choices = event.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"]:
                            deltas += 1
                            if first_delta is None:
                                first_delta = time.perf_counter()
                            if cancel_after is not None and deltas >= cancel_after:
                                connection.close()
                                return {
                                    "cancelled": True,
                                    "delta_count": deltas,
                                    "ttft_seconds": first_delta - started,
                                    "end_to_end_seconds": time.perf_counter() - started,
                                }
                        if choice.get("finish_reason") is not None:
                            saw_finish = True
                usage = event.get("usage")
                if isinstance(usage, dict):
                    saw_usage = True
                    count = usage.get("completion_tokens")
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        completion_tokens = count
        except SmokeError:
            raise
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            raise SmokeError("transport_error") from exc
        finally:
            connection.close()
        if deltas < 2:
            raise SmokeError("sse_incremental_deltas_missing")
        if not saw_done:
            raise SmokeError("sse_done_missing")
        if not saw_usage:
            raise SmokeError("sse_usage_missing")
        if not saw_finish:
            raise SmokeError("sse_natural_finish_missing")
        ended = time.perf_counter()
        decode_rate: float | None = None
        if completion_tokens is not None and first_delta is not None and ended > first_delta:
            decode_rate = completion_tokens / (ended - first_delta)
        return {
            "cancelled": False,
            "delta_count": deltas,
            "ttft_seconds": None if first_delta is None else first_delta - started,
            "end_to_end_seconds": ended - started,
            "decode_tokens_per_second": decode_rate,
        }


def _chat_body(config: Config, prompt: str, *, stream: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _completion_text(value: Mapping[str, Any]) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise SmokeError("completion_shape_invalid")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise SmokeError("completion_shape_invalid")
    return message["content"].strip()


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _request_result(name: str, elapsed: float) -> dict[str, Any]:
    return {"check": name, "status": "pass", "latency_seconds": _round_metric(elapsed)}


def _run_smoke(config: Config, client: Client) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    capability_results = {name: "not_declared" for name in sorted(CAPABILITY_NAMES)}

    models, elapsed = client.json_request("GET", "/v1/models")
    data = models.get("data")
    if not isinstance(data, list) or not data:
        raise SmokeError("models_shape_invalid")
    if not any(isinstance(item, dict) and item.get("id") == config.model_id for item in data):
        raise SmokeError("configured_model_missing")
    results.append(_request_result("models", elapsed))

    basic, elapsed = client.json_request(
        "POST", "/v1/chat/completions", _chat_body(config, _PROMPTS["basic"])
    )
    if _completion_text(basic) != "BASIC_OK":
        raise SmokeError("non_stream_content_invalid")
    results.append(_request_result("non_stream", elapsed))

    streamed = client.stream_request(
        _chat_body(config, _PROMPTS["stream"], stream=True)
    )
    results.append(
        {
            "check": "sse_stream",
            "status": "pass",
            "ttft_seconds": _round_metric(streamed["ttft_seconds"]),
            "end_to_end_seconds": _round_metric(streamed["end_to_end_seconds"]),
            "decode_tokens_per_second": _round_metric(
                streamed["decode_tokens_per_second"]
            ),
            "incremental_deltas_observed": streamed["delta_count"] >= 2,
            "done_observed": True,
            "usage_observed": True,
        }
    )

    for index in range(2):
        value, elapsed = client.json_request(
            "POST",
            "/v1/chat/completions",
            _chat_body(config, _PROMPTS["sequential"]),
        )
        if _completion_text(value) != "SEQUENTIAL_OK":
            raise SmokeError("sequential_content_invalid")
        results.append(_request_result(f"sequential_{index + 1}", elapsed))

    cancelled = client.stream_request(
        _chat_body(config, _PROMPTS["cancel"], stream=True),
        cancel_after=config.cancel_after_deltas,
    )
    if not cancelled["cancelled"]:
        raise SmokeError("cancel_not_observed")
    results.append(
        {
            "check": "client_disconnect",
            "status": "pass",
            "ttft_seconds": _round_metric(cancelled["ttft_seconds"]),
            "end_to_end_seconds": _round_metric(cancelled["end_to_end_seconds"]),
        }
    )
    for index in range(2):
        value, elapsed = client.json_request(
            "POST",
            "/v1/chat/completions",
            _chat_body(config, _PROMPTS["recovery"]),
        )
        if _completion_text(value) != "RECOVERY_OK":
            raise SmokeError("recovery_content_invalid")
        results.append(_request_result(f"recovery_{index + 1}", elapsed))

    value, elapsed = client.json_request(
        "POST", "/v1/chat/completions", _chat_body(config, _PROMPTS["json"])
    )
    try:
        parsed_json = json.loads(_completion_text(value))
    except json.JSONDecodeError as exc:
        raise SmokeError("prompt_json_parse_failed") from exc
    if parsed_json != {"status": "ok", "value": 7}:
        raise SmokeError("prompt_json_shape_failed")
    results.append(_request_result("prompt_json", elapsed))

    if config.capabilities["responses_api"]:
        response, elapsed = client.json_request(
            "POST",
            "/v1/responses",
            {"model": config.model_id, "input": _PROMPTS["harmony"]},
        )
        output_text = response.get("output_text")
        if output_text != "HARMONY_OK":
            output = response.get("output")
            texts: list[str] = []
            if isinstance(output, list):
                for item in output:
                    if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                        continue
                    for content in item["content"]:
                        if isinstance(content, dict) and isinstance(content.get("text"), str):
                            texts.append(content["text"])
            if "".join(texts).strip() != "HARMONY_OK":
                raise SmokeError("harmony_response_invalid")
        capability_results["responses_api"] = "pass"
        results.append(_request_result("harmony_response", elapsed))

    if config.capabilities["constrained_response_format"]:
        body = _chat_body(config, _PROMPTS["format"])
        body["response_format"] = {"type": "json_object"}
        value, elapsed = client.json_request("POST", "/v1/chat/completions", body)
        try:
            constrained = json.loads(_completion_text(value))
        except json.JSONDecodeError as exc:
            raise SmokeError("constrained_json_parse_failed") from exc
        if constrained != {"status": "ok"}:
            raise SmokeError("constrained_json_shape_failed")
        capability_results["constrained_response_format"] = "pass"
        results.append(_request_result("constrained_response_format", elapsed))

    if config.capabilities["tools"]:
        body = _chat_body(config, _PROMPTS["tool"])
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "report_status",
                    "parameters": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        body["tool_choice"] = {"type": "function", "function": {"name": "report_status"}}
        value, elapsed = client.json_request("POST", "/v1/chat/completions", body)
        choices = value.get("choices")
        try:
            tool_call = choices[0]["message"]["tool_calls"][0]
            arguments = json.loads(tool_call["function"]["arguments"])
            valid_tool = (
                tool_call["type"] == "function"
                and tool_call["function"]["name"] == "report_status"
                and arguments == {"status": "ok"}
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            valid_tool = False
        if not valid_tool:
            raise SmokeError("tool_call_invalid")
        capability_results["tools"] = "pass"
        results.append(_request_result("tools", elapsed))

    if config.capabilities["vision"]:
        body = _chat_body(config, _PROMPTS["vision"])
        body["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPTS["vision"]},
                    {"type": "image_url", "image_url": {"url": _ONE_PIXEL_PNG}},
                ],
            }
        ]
        value, elapsed = client.json_request("POST", "/v1/chat/completions", body)
        if _completion_text(value) != "VISION_OK":
            raise SmokeError("vision_content_invalid")
        capability_results["vision"] = "pass"
        results.append(_request_result("vision", elapsed))

    metrics = {
        "request_count": len(results),
        "max_latency_seconds": _round_metric(
            max(
                item.get("latency_seconds", item.get("end_to_end_seconds", 0.0))
                for item in results
            )
        ),
    }
    return results, capability_results, metrics


def execute(config_path: Path) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    token = _read_secret(config.token_file) if config.token_file is not None else None
    started = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    capabilities = {name: "not_run" for name in sorted(CAPABILITY_NAMES)}
    metrics: dict[str, Any] = {"request_count": 0, "max_latency_seconds": None}
    error_code: str | None = None
    http_status: int | None = None
    try:
        results, capabilities, metrics = _run_smoke(config, Client(config, token))
    except SmokeError as exc:
        error_code = exc.code
        http_status = exc.http_status
    ended = datetime.now(timezone.utc)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_VERSION,
        "runner_version": RUNNER_VERSION,
        "candidate_id": config.candidate_id,
        "candidate_type": config.candidate_type,
        "runtime_id": config.runtime_id,
        "manifest_id": config.manifest_id,
        "model_id": config.model_id,
        "auth_mode": config.auth_mode,
        "config_sha256": config_sha256,
        "suite_sha256": _SUITE_SHA256,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "status": "pass" if error_code is None else "fail",
        "error_code": error_code,
        "http_status": http_status,
        "results": results,
        "capabilities": capabilities,
        "metrics": metrics,
        "content_recorded": False,
        "secrets_recorded": False,
        "token_counts_recorded": False,
    }
    receipt["content_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    return receipt


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("receipt_exists")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ConfigurationError("receipt_parent_invalid")
    payload = _canonical_bytes(receipt) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o400)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count < 1:
                raise OSError("receipt_write_incomplete")
            written += count
        os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, 0o400)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if os.name == "nt":
        path.chmod(stat.S_IREAD)
    else:
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = execute(args.config)
        write_receipt(args.receipt, receipt)
    except (ConfigurationError, FileExistsError) as exc:
        parser.error(str(exc))
    print(receipt["status"])
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
