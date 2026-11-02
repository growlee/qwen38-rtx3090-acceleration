"""Content-free OpenAI benchmark runner for the syv and FreeToken candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import candidate_openai_smoke as smoke


RUNNER_VERSION = "1.0.0"
CONFIG_VERSION = 1
SUITE_VERSION = 1
REPORT_VERSION = 1
MAX_JSON_BYTES = 1_048_576
MIN_STABILITY_SECONDS = 1800
ALLOWED_TYPES = {"syv", "freetoken"}
ALLOWED_SYV_ARMS = {"mtp", "dflash2"}


class ConfigurationError(ValueError):
    pass


class BenchmarkError(RuntimeError):
    def __init__(self, code: str, http_status: int | None = None):
        super().__init__(code)
        self.code = code
        self.http_status = http_status


@dataclass(frozen=True)
class Case:
    case_id: str
    messages: tuple[Mapping[str, Any], ...]
    generation: Mapping[str, Any]


@dataclass(frozen=True)
class Suite:
    suite_id: str
    cases: tuple[Case, ...]
    freetoken_4k: Case
    freetoken_context_turns: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Config:
    candidate_id: str
    candidate_type: str
    arm: str
    runtime_id: str
    manifest_id: str
    base_url: str
    model_id: str
    auth_mode: str
    token_file: Path | None
    connect_timeout_seconds: float
    request_timeout_seconds: float
    stability_seconds: int
    cold_load_seconds: float
    safety_receipt_sha256: str


class StreamingClient(Protocol):
    def stream_request(
        self, body: Mapping[str, Any], *, cancel_after: int | None = None
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TestDependencies:
    """Dependency injection for bounded unit tests; the CLI cannot construct this."""

    client_factory: Callable[[Config, str | None], StreamingClient]
    monotonic: Callable[[], float]
    stability_seconds_override: float


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError("json_unreadable") from exc
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ConfigurationError("json_size_invalid")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("json_invalid") from exc
    return value, _sha256_bytes(raw)


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ConfigurationError(code)


def _finite_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ConfigurationError(f"{field}_invalid")
    return float(value)


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConfigurationError(f"{field}_invalid")
    return value


def load_config(path: Path) -> tuple[Config, str]:
    value, _ = _load_json(path)
    if not isinstance(value, dict):
        raise ConfigurationError("config_invalid")
    _exact_keys(
        value,
        {
            "schema_version",
            "candidate_id",
            "candidate_type",
            "arm",
            "runtime_id",
            "manifest_id",
            "base_url",
            "model_id",
            "auth_mode",
            "token_file",
            "connect_timeout_seconds",
            "request_timeout_seconds",
            "stability_seconds",
            "cold_load_seconds",
            "safety_receipt_sha256",
        },
        "config_keys_invalid",
    )
    if value["schema_version"] != CONFIG_VERSION:
        raise ConfigurationError("config_version_unsupported")
    candidate_type = value["candidate_type"]
    arm = value["arm"]
    if candidate_type not in ALLOWED_TYPES:
        raise ConfigurationError("candidate_type_invalid")
    if (candidate_type == "syv" and arm not in ALLOWED_SYV_ARMS) or (
        candidate_type == "freetoken" and arm != "freetoken"
    ):
        raise ConfigurationError("arm_invalid")

    # Reuse the hardened URL<redacted-secret-path> validation from the generic smoke.
    smoke_value = {
        "schema_version": smoke.CONFIG_VERSION,
        "candidate_id": value["candidate_id"],
        "candidate_type": candidate_type,
        "runtime_id": value["runtime_id"],
        "manifest_id": value["manifest_id"],
        "base_url": value["base_url"],
        "model_id": value["model_id"],
        "auth_mode": value["auth_mode"],
        "token_file": value["token_file"],
        "connect_timeout_seconds": value["connect_timeout_seconds"],
        "request_timeout_seconds": value["request_timeout_seconds"],
        "cancel_after_deltas": 1,
        "capabilities": {
            "responses_api": False,
            "constrained_response_format": False,
            "tools": False,
            "vision": False,
        },
    }
    # Avoid filesystem mutation: apply the same validator through a private helper
    # input encoded in memory, rather than calling smoke.load_config(Path).
    parsed = smoke.urlsplit(smoke_value["base_url"])
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("base_url_invalid") from exc
    if (
        not isinstance(smoke_value["base_url"], str)
        or len(smoke_value["base_url"]) > 2048
        or parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or smoke_value["base_url"].endswith("/")
        or (port is not None and not 1 <= port <= 65535)
        or (parsed.scheme == "http" and parsed.hostname != "127.0.0.1")
    ):
        raise ConfigurationError("base_url_invalid")
    auth_mode = value["auth_mode"]
    token_file_raw = value["token_file"]
    if auth_mode == "none_loopback":
        if parsed.hostname != "127.0.0.1" or token_file_raw is not None:
            raise ConfigurationError("none_loopback_scope_invalid")
        token_file = None
    elif auth_mode == "bearer":
        if not isinstance(token_file_raw, str) or not token_file_raw or len(token_file_raw) > 4096:
            raise ConfigurationError("token_file_invalid")
        token_file = Path(token_file_raw)
    else:
        raise ConfigurationError("auth_mode_invalid")

    stability = value["stability_seconds"]
    if (
        not isinstance(stability, int)
        or isinstance(stability, bool)
        or not MIN_STABILITY_SECONDS <= stability <= 7200
    ):
        raise ConfigurationError("stability_seconds_invalid")
    for field in ("candidate_id", "runtime_id", "manifest_id"):
        try:
            smoke._safe_id(value[field], field)
        except smoke.ConfigurationError as exc:
            raise ConfigurationError(str(exc)) from exc
    model_id = value["model_id"]
    if not isinstance(model_id, str) or not model_id or len(model_id) > 512:
        raise ConfigurationError("model_id_invalid")
    config = Config(
        candidate_id=value["candidate_id"],
        candidate_type=candidate_type,
        arm=arm,
        runtime_id=value["runtime_id"],
        manifest_id=value["manifest_id"],
        base_url=value["base_url"],
        model_id=model_id,
        auth_mode=auth_mode,
        token_file=token_file,
        connect_timeout_seconds=_finite_number(
            value["connect_timeout_seconds"], "connect_timeout_seconds", 0.1, 30
        ),
        request_timeout_seconds=_finite_number(
            value["request_timeout_seconds"], "request_timeout_seconds", 1, 3600
        ),
        stability_seconds=stability,
        cold_load_seconds=_finite_number(
            value["cold_load_seconds"], "cold_load_seconds", 0, 7200
        ),
        safety_receipt_sha256=_sha256(
            value["safety_receipt_sha256"], "safety_receipt_sha256"
        ),
    )
    # Hash a redacted canonical config so the local secret path never enters evidence.
    redacted = dict(value)
    redacted["token_file"] = None if token_file is None else "<redacted>"
    return config, _sha256_bytes(_canonical_bytes(redacted))


def _parse_case(value: Any, code: str) -> Case:
    if not isinstance(value, dict):
        raise ConfigurationError(code)
    _exact_keys(value, {"case_id", "messages", "generation"}, code)
    case_id = value["case_id"]
    if not isinstance(case_id, str) or smoke.SAFE_ID.fullmatch(case_id) is None:
        raise ConfigurationError(code)
    messages = value["messages"]
    if not isinstance(messages, list) or not messages or len(messages) > 16:
        raise ConfigurationError(code)
    checked_messages: list[Mapping[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ConfigurationError(code)
        _exact_keys(message, {"role", "content"}, code)
        if message["role"] not in {"system", "user", "assistant"}:
            raise ConfigurationError(code)
        if not isinstance(message["content"], str) or not message["content"] or len(message["content"]) > 32768:
            raise ConfigurationError(code)
        checked_messages.append(dict(message))
    generation = value["generation"]
    if not isinstance(generation, dict):
        raise ConfigurationError(code)
    _exact_keys(generation, {"temperature", "seed", "max_tokens"}, code)
    if generation["temperature"] != 0:
        raise ConfigurationError(code)
    if not isinstance(generation["seed"], int) or isinstance(generation["seed"], bool):
        raise ConfigurationError(code)
    if (
        not isinstance(generation["max_tokens"], int)
        or isinstance(generation["max_tokens"], bool)
        or not 1 <= generation["max_tokens"] <= 1024
    ):
        raise ConfigurationError(code)
    return Case(case_id, tuple(checked_messages), dict(generation))


def load_suite(path: Path) -> tuple[Suite, str]:
    value, raw_sha256 = _load_json(path)
    if not isinstance(value, dict):
        raise ConfigurationError("suite_invalid")
    _exact_keys(
        value,
        {"schema_version", "suite_id", "cases", "freetoken_4k", "freetoken_context_turns"},
        "suite_keys_invalid",
    )
    if value["schema_version"] != SUITE_VERSION or value["suite_id"] != "acceleration-benchmark-v1":
        raise ConfigurationError("suite_identity_invalid")
    if not isinstance(value["cases"], list) or len(value["cases"]) != 4:
        raise ConfigurationError("suite_cases_invalid")
    cases = tuple(_parse_case(item, "suite_case_invalid") for item in value["cases"])
    if len({case.case_id for case in cases}) != 4:
        raise ConfigurationError("suite_case_duplicate")
    four_k_source = _parse_case(value["freetoken_4k"], "suite_4k_invalid")
    source_words = sum(len(message["content"].split()) for message in four_k_source.messages)
    if not 1000 <= source_words <= 2000:
        raise ConfigurationError("suite_4k_size_invalid")
    repeat_count = math.ceil(3800 / source_words)
    expanded_messages = list(four_k_source.messages)
    expanded_messages[-1] = {
        **expanded_messages[-1],
        "content": "\n".join(
            [expanded_messages[-1]["content"]] * repeat_count
        ),
    }
    expanded_words = sum(len(message["content"].split()) for message in expanded_messages)
    if not 3500 <= expanded_words <= 5000:
        raise ConfigurationError("suite_4k_size_invalid")
    four_k = Case(
        four_k_source.case_id,
        tuple(expanded_messages),
        four_k_source.generation,
    )
    turns = value["freetoken_context_turns"]
    if not isinstance(turns, list) or len(turns) != 3:
        raise ConfigurationError("suite_context_invalid")
    checked_turns: list[Mapping[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ConfigurationError("suite_context_invalid")
        _exact_keys(turn, {"role", "content"}, "suite_context_invalid")
        if turn["role"] != "user" or not isinstance(turn["content"], str) or not turn["content"]:
            raise ConfigurationError("suite_context_invalid")
        checked_turns.append(dict(turn))
    return Suite(value["suite_id"], cases, four_k, tuple(checked_turns)), raw_sha256


def verify_safety_receipt(path: Path, expected_sha256: str) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError("safety_receipt_unreadable") from exc
    if not raw or len(raw) > MAX_JSON_BYTES or _sha256_bytes(raw) != expected_sha256:
        raise ConfigurationError("safety_receipt_sha256_mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("safety_receipt_invalid") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("safety_receipt_invalid")
    return expected_sha256


def _smoke_config(config: Config) -> smoke.Config:
    return smoke.Config(
        candidate_id=config.candidate_id,
        candidate_type=config.candidate_type,
        runtime_id=config.runtime_id,
        manifest_id=config.manifest_id,
        base_url=config.base_url,
        model_id=config.model_id,
        auth_mode=config.auth_mode,
        token_file=config.token_file,
        connect_timeout_seconds=config.connect_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        cancel_after_deltas=1,
        capabilities={name: False for name in smoke.CAPABILITY_NAMES},
    )


def _body(config: Config, messages: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]], generation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": config.model_id,
        "messages": [dict(message) for message in messages],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 6)


def _median(values: list[float]) -> float | None:
    return None if not values else round(statistics.median(values), 6)


def _record_metric(result: Mapping[str, Any], metrics: dict[str, list[float]]) -> None:
    for source, destination in (
        ("ttft_seconds", "ttft"),
        ("decode_tokens_per_second", "decode"),
        ("end_to_end_seconds", "e2e"),
    ):
        value = result.get(source)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise BenchmarkError(f"{destination}_metric_missing")
        metrics[destination].append(float(value))


def _request(
    client: StreamingClient,
    body: Mapping[str, Any],
    metrics: dict[str, list[float]],
    counters: dict[str, int],
    *,
    measure: bool = True,
) -> None:
    counters["request_count"] += 1
    try:
        result = client.stream_request(body)
    except smoke.SmokeError as exc:
        counters["error_count"] += 1
        if exc.http_status is not None and 500 <= exc.http_status <= 599:
            counters["http_5xx_count"] += 1
        if exc.code.startswith("sse_"):
            counters["incomplete_stream_count"] += 1
        raise BenchmarkError(exc.code, exc.http_status) from exc
    if result.get("cancelled") is not False:
        counters["error_count"] += 1
        counters["incomplete_stream_count"] += 1
        raise BenchmarkError("unexpected_cancel")
    sample: dict[str, list[float]] = {"ttft": [], "decode": [], "e2e": []}
    _record_metric(result, sample)
    if measure:
        for name in metrics:
            metrics[name].extend(sample[name])


def run_benchmark(
    config: Config,
    suite: Suite,
    token: str | None,
    *,
    test_dependencies: TestDependencies | None = None,
) -> tuple[dict[str, Any], str | None, int | None]:
    if test_dependencies is None:
        client: StreamingClient = smoke.Client(_smoke_config(config), token)
        monotonic = time.monotonic
        stability_seconds = float(config.stability_seconds)
    else:
        if (
            not isinstance(test_dependencies.stability_seconds_override, (int, float))
            or isinstance(test_dependencies.stability_seconds_override, bool)
            or not math.isfinite(float(test_dependencies.stability_seconds_override))
            or test_dependencies.stability_seconds_override <= 0
        ):
            raise ConfigurationError("test_stability_override_invalid")
        client = test_dependencies.client_factory(config, token)
        monotonic = test_dependencies.monotonic
        stability_seconds = float(test_dependencies.stability_seconds_override)

    metrics: dict[str, list[float]] = {"ttft": [], "decode": [], "e2e": []}
    counters = {
        "request_count": 0,
        "warmup_count": 0,
        "benchmark_count": 0,
        "stability_count": 0,
        "error_count": 0,
        "incomplete_stream_count": 0,
        "http_5xx_count": 0,
    }
    error_code: str | None = None
    http_status: int | None = None
    try:
        warmup_body = _body(config, suite.cases[0].messages, suite.cases[0].generation)
        for _ in range(3):
            _request(client, warmup_body, metrics, counters, measure=False)
            counters["warmup_count"] += 1

        if config.candidate_type == "syv":
            for case in suite.cases:
                body = _body(config, case.messages, case.generation)
                for _ in range(20):
                    _request(client, body, metrics, counters)
                    counters["benchmark_count"] += 1
        else:
            for index in range(20):
                case = suite.cases[index % len(suite.cases)]
                _request(client, _body(config, case.messages, case.generation), metrics, counters)
                counters["benchmark_count"] += 1
            _request(
                client,
                _body(config, suite.freetoken_4k.messages, suite.freetoken_4k.generation),
                metrics,
                counters,
            )
            counters["benchmark_count"] += 1
            context: list[Mapping[str, Any]] = []
            context_generation = suite.cases[0].generation
            for turn in suite.freetoken_context_turns:
                context.append(turn)
                _request(client, _body(config, context, context_generation), metrics, counters)
                counters["benchmark_count"] += 1
                # Public synthetic assistant context; response text is never retained.
                context.append({"role": "assistant", "content": "ACK"})

        stability_started = monotonic()
        index = 0
        while monotonic() - stability_started < stability_seconds:
            case = suite.cases[index % len(suite.cases)]
            _request(
                client,
                _body(config, case.messages, case.generation),
                metrics,
                counters,
                measure=False,
            )
            counters["stability_count"] += 1
            index += 1
        if counters["stability_count"] == 0:
            raise BenchmarkError("stability_request_missing")
    except BenchmarkError as exc:
        error_code = exc.code
        http_status = exc.http_status

    result = {
        **counters,
        "cold_load_seconds": round(config.cold_load_seconds, 6),
        "ttft_seconds_median": _median(metrics["ttft"]),
        "ttft_seconds_p95": _percentile(metrics["ttft"], 0.95),
        "decode_tokens_per_second_median": _median(metrics["decode"]),
        "end_to_end_seconds_p95": _percentile(metrics["e2e"], 0.95),
    }
    return result, error_code, http_status


def execute(
    config_path: Path,
    suite_path: Path,
    safety_receipt_path: Path,
    *,
    test_dependencies: TestDependencies | None = None,
) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    suite, suite_sha256 = load_suite(suite_path)
    safety_sha256 = verify_safety_receipt(
        safety_receipt_path, config.safety_receipt_sha256
    )
    token = smoke._read_secret(config.token_file) if config.token_file is not None else None
    started = datetime.now(timezone.utc)
    metrics, error_code, http_status = run_benchmark(
        config, suite, token, test_dependencies=test_dependencies
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "runner_version": RUNNER_VERSION,
        "candidate_id": config.candidate_id,
        "candidate_type": config.candidate_type,
        "arm": config.arm,
        "runtime_id": config.runtime_id,
        "manifest_id": config.manifest_id,
        "config_sha256": config_sha256,
        "suite_sha256": suite_sha256,
        "safety_receipt_sha256": safety_sha256,
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if error_code is None else "fail",
        "error_code": error_code,
        "http_status": http_status,
        "metrics": metrics,
        "content_recorded": False,
        "secrets_recorded": False,
        "token_counts_recorded": False,
        "endpoint_recorded": False,
    }
    report["content_sha256"] = _sha256_bytes(_canonical_bytes(report))
    return report


def evaluate_syv_dflash(
    mtp_report: Mapping[str, Any],
    dflash_report: Mapping[str, Any],
    *,
    quality_pass: bool,
    no_resource_regression: bool,
) -> dict[str, Any]:
    if mtp_report.get("candidate_type") != "syv" or mtp_report.get("arm") != "mtp":
        raise ConfigurationError("mtp_report_identity_invalid")
    if dflash_report.get("candidate_type") != "syv" or dflash_report.get("arm") != "dflash2":
        raise ConfigurationError("dflash_report_identity_invalid")
    for field in ("candidate_id", "runtime_id", "suite_sha256"):
        left = mtp_report.get(field)
        right = dflash_report.get(field)
        if not isinstance(left, str) or not left or left != right:
            raise ConfigurationError("comparison_coordinate_mismatch")
    if mtp_report.get("status") != "pass" or dflash_report.get("status") != "pass":
        return {"status": "fail", "reason": "benchmark_not_passed"}
    try:
        mtp_decode = float(mtp_report["metrics"]["decode_tokens_per_second_median"])
        dflash_decode = float(dflash_report["metrics"]["decode_tokens_per_second_median"])
        mtp_e2e = float(mtp_report["metrics"]["end_to_end_seconds_p95"])
        dflash_e2e = float(dflash_report["metrics"]["end_to_end_seconds_p95"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("comparison_metric_invalid") from exc
    if min(mtp_decode, dflash_decode, mtp_e2e, dflash_e2e) <= 0:
        raise ConfigurationError("comparison_metric_invalid")
    decode_ratio = dflash_decode / mtp_decode
    e2e_ratio = dflash_e2e / mtp_e2e
    passed = (
        decode_ratio >= 1.15
        and e2e_ratio <= 1.10
        and quality_pass is True
        and no_resource_regression is True
    )
    return {
        "status": "pass" if passed else "fail",
        "decode_ratio": round(decode_ratio, 6),
        "e2e_p95_ratio": round(e2e_ratio, 6),
        "quality_pass": quality_pass is True,
        "no_resource_regression": no_resource_regression is True,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("report_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = _canonical_bytes(report) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("report_write_failed")
            view = view[written:]
        os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--safety-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = execute(args.config, args.suite, args.safety_receipt)
    write_report(args.output, report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
