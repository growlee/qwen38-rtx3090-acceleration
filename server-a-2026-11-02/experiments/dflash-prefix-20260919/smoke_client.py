"""Run the bounded DFlash2 prefix-cache smoke matrix inside its container.

The client makes exactly three serial chat-completion requests and writes JSONL
only to stdout. It performs no retries and makes no filesystem writes.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import subprocess
import threading
import time


HOST = "127.0.0.1"
PORT = 18020
MODEL = "qwen3.8-27b"
TOKENIZER = "/app/models/Qwen3.8-27B-W4A16-AutoRound"
MAX_CONTEXT = 32768
MAX_OUTPUT = 256
MAX_BYTES = 8 * 1024 * 1024
SHORT_MARKER = "DFLASH_PREFIX_SMOKE_OK_20260919"


class InfrastructureError(RuntimeError):
    """A failure that invalidates later cases in this serial matrix."""


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


class Vram:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
                self.samples.append({"monotonic": time.perf_counter(), "used_mib": values})
            except Exception as exc:  # Sampling must not invalidate an inference result.
                self.errors.append(type(exc).__name__)
                break
            self.stop.wait(0.25)

    def peak(self):
        return max((value for sample in self.samples for value in sample["used_mib"]), default=None)


def get_metrics():
    connection = http.client.HTTPConnection(HOST, PORT, timeout=15)
    try:
        connection.request("GET", "/metrics", headers={"Accept": "text/plain"})
        response = connection.getresponse()
        raw = response.read(MAX_BYTES + 1)
        if response.status != 200:
            raise InfrastructureError("metrics_http_" + str(response.status))
        if len(raw) > MAX_BYTES:
            raise InfrastructureError("metrics_response_too_large")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InfrastructureError("metrics_not_utf8") from exc
        return {"raw": text, "parsed": parse_metrics(text)}
    except (OSError, http.client.HTTPException) as exc:
        raise InfrastructureError("metrics_transport_" + type(exc).__name__) from exc
    finally:
        connection.close()


def parse_metrics(text):
    """Keep exact relevant samples plus aggregates used for cache/spec deltas."""
    samples = {}
    aggregates = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        sample, raw_value = fields
        name = sample.split("{", 1)[0]
        lowered = name.lower()
        relevant = (
            "prefix_cache" in lowered
            or ("cache" in lowered and "hit" in lowered)
            or ("spec_decode" in lowered and ("draft" in lowered or "accept" in lowered))
        )
        if not relevant:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        samples[sample] = value
        aggregates[name] = aggregates.get(name, 0.0) + value
    return {"samples": dict(sorted(samples.items())), "aggregates": dict(sorted(aggregates.items()))}


def metric_delta(before, after):
    result = {}
    for name, after_value in after["aggregates"].items():
        before_value = before["aggregates"].get(name)
        if before_value is not None and name.endswith(("_total", "_count", "_sum")):
            result[name] = after_value - before_value
        else:
            result[name] = after_value
    return dict(sorted(result.items()))


def prefix_metric_evidence(delta, required_tokens):
    # These exact vLLM counters are token counters. Do not use a cumulative hit
    # rate gauge as per-request proof: it may include traffic before this case.
    query_tokens = delta.get("vllm:prefix_cache_queries_total", 0.0)
    hit_tokens = delta.get("vllm:prefix_cache_hits_total", 0.0)
    ratio = hit_tokens / query_tokens if query_tokens > 0 else None
    passed = hit_tokens >= required_tokens and ratio is not None and ratio >= 0.75
    return {
        "counter_unit": "tokens (exact vLLM counter names)",
        "hit_tokens_delta": hit_tokens,
        "query_tokens_delta": query_tokens,
        "hit_ratio_from_counter_deltas": ratio,
        "passed": passed,
    }


def stream(case_id, messages, local_prompt_tokens, validator, warm_required_tokens=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    messages_body = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    result = {
        "case": case_id,
        "status": "FAIL",
        "infrastructure_failure": False,
        "local_prompt_tokens": local_prompt_tokens,
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "messages_sha256": hashlib.sha256(messages_body).hexdigest(),
        "ttft_definition": "request start to first non-whitespace delta.content",
        "usage": None,
        "cached_tokens": None,
        "response_content": "",
        "reasoning_chars": 0,
        "first_content_seconds": None,
        "finish_reason": None,
        "done": False,
    }
    connection = http.client.HTTPConnection(HOST, PORT, timeout=900)
    sampler = Vram()
    start = None
    stream_end = None
    first = None
    metrics_before = None
    try:
        metrics_before = get_metrics()
        result["prometheus_before"] = metrics_before
        sampler.thread.start()
        start = time.perf_counter()
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        result["http_status"] = response.status
        if response.status != 200 or not response.getheader("Content-Type", "").startswith("text/event-stream"):
            result["error_body"] = response.read(4096).decode("utf-8", errors="replace")
            raise InfrastructureError("stream_http_or_content_type")
        total_bytes = 0
        while True:
            line = response.readline(1024 * 1024 + 1)
            now = time.perf_counter()
            total_bytes += len(line)
            if len(line) > 1024 * 1024 or total_bytes > MAX_BYTES or now - start > 1200:
                raise InfrastructureError("stream_bound_exceeded")
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                result["done"] = True
                break
            try:
                event = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InfrastructureError("invalid_sse_json") from exc
            if event.get("error"):
                result["server_error"] = event["error"]
                raise InfrastructureError("stream_server_error")
            if event.get("usage") is not None:
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if not isinstance(content, str) or not isinstance(reasoning, str):
                    raise InfrastructureError("non_text_delta")
                if first is None and content.strip():
                    first = now
                result["response_content"] += content
                result["reasoning_chars"] += len(reasoning)
                if delta.get("tool_calls"):
                    raise InfrastructureError("unexpected_tool_call")
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]

        stream_end = time.perf_counter()
        sampler.stop.set()
        sampler.thread.join(timeout=6)
        metrics_after = get_metrics()
        result["prometheus_after"] = metrics_after
        result["prometheus_delta"] = metric_delta(metrics_before["parsed"], metrics_after["parsed"])
        usage = result["usage"] if isinstance(result["usage"], dict) else {}
        if not (
            result["done"]
            and first is not None
            and type(usage.get("prompt_tokens")) is int
            and usage["prompt_tokens"] > 0
            and type(usage.get("completion_tokens")) is int
            and 0 < usage["completion_tokens"] <= MAX_OUTPUT
        ):
            raise InfrastructureError("stream_or_usage_incomplete")
        details = usage.get("prompt_tokens_details") or {}
        result["cached_tokens"] = details.get("cached_tokens") if isinstance(details, dict) else None

        failures = []
        if result["finish_reason"] != "stop":
            failures.append("finish_reason_not_stop")
        if result["reasoning_chars"] != 0:
            failures.append("reasoning_present")
        validation_error = validator(result["response_content"])
        if validation_error:
            failures.append(validation_error)
        if warm_required_tokens is not None:
            cached = result["cached_tokens"]
            usage_pass = type(cached) is int and cached >= warm_required_tokens
            metric_evidence = prefix_metric_evidence(result["prometheus_delta"], warm_required_tokens)
            result["warm_prefix_evidence"] = {
                "required_tokens": warm_required_tokens,
                "usage_cached_tokens_passed": usage_pass,
                "metric": metric_evidence,
                "passed": usage_pass or metric_evidence["passed"],
            }
            if not result["warm_prefix_evidence"]["passed"]:
                failures.append("substantial_prefix_hit_not_proven")
        result["validation_failures"] = failures
        if not failures:
            result["status"] = "PASS"
    except InfrastructureError as exc:
        result["error"] = str(exc)
        result["infrastructure_failure"] = True
    except (OSError, http.client.HTTPException) as exc:
        result["error"] = "stream_transport_" + type(exc).__name__
        result["infrastructure_failure"] = True
    finally:
        if stream_end is None and start is not None:
            stream_end = time.perf_counter()
        duration = None if start is None or stream_end is None else stream_end - start
        ttft = None if first is None or start is None else first - start
        result["duration_seconds"] = duration
        result["first_content_seconds"] = ttft
        connection.close()
        sampler.stop.set()
        if sampler.thread.ident is not None:
            sampler.thread.join(timeout=6)
        result["vram_samples"] = sampler.samples
        result["vram_sampling_errors"] = sampler.errors
        result["peak_vram_mib"] = sampler.peak()
        usage = result["usage"] if isinstance(result["usage"], dict) else {}
        completion_tokens = usage.get("completion_tokens")
        result["completion_tokens_per_stream_second"] = (
            completion_tokens / duration
            if type(completion_tokens) is int and duration is not None and duration > 0
            else None
        )
        result["estimated_decode_tokens_per_second"] = (
            (completion_tokens - 1) / (stream_end - first)
            if type(completion_tokens) is int
            and completion_tokens > 1
            and stream_end is not None
            and first is not None
            and stream_end > first
            else None
        )
        result["historical_weighted_decode_tokens_per_second"] = (
            completion_tokens / (duration - ttft)
            if type(completion_tokens) is int
            and duration is not None
            and ttft is not None
            and duration > ttft
            else None
        )
        result["decode_rate_note"] = (
            "Approximate: SSE content chunks may contain multiple tokens; estimated decode uses "
            "(completion_tokens - 1) / (stream_end - first_content), while the historical weighted "
            "metric uses completion_tokens / (elapsed - TTFT)."
        )
        emit(result)
    if result["infrastructure_failure"]:
        raise InfrastructureError("terminal_infrastructure_failure_" + case_id)
    return result


def exact_marker_validator(content):
    return None if content.strip() == SHORT_MARKER else "exact_marker_mismatch"


def cold_validator(content):
    return None if content.strip() else "empty_cold_response"


def json_array_validator(expected):
    def validate(content):
        try:
            value = json.loads(content.strip())
        except json.JSONDecodeError:
            return "continuation_not_only_json"
        return None if value == expected else "continuation_semantic_mismatch"

    return validate


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, trust_remote_code=False)

    def count(messages):
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )

    def execute(case_id, messages, validator, warm_required_tokens=None):
        prompt_tokens = count(messages)
        if prompt_tokens + MAX_OUTPUT > MAX_CONTEXT:
            raise InfrastructureError("context_budget_exceeded_" + case_id)
        return stream(case_id, messages, prompt_tokens, validator, warm_required_tokens)

    def fresh(target):
        prefix = "gamma: Synthetic source fixture. Read the code and describe its intent in two short sentences.\n"
        suffix = "\nGive only the brief description, without reasoning."

        def messages(function_count):
            code = "".join(
                f"def gamma_{index}(items):\n    return [x + {index % 7} for x in items if x >= 0]\n"
                for index in range(function_count)
            )
            return [{"role": "user", "content": prefix + code + suffix}]

        low, high = 1, target
        while low < high:
            middle = (low + high + 1) // 2
            if count(messages(middle)) <= target:
                low = middle
            else:
                high = middle - 1
        result = messages(low)
        actual = count(result)
        if not target - 128 <= actual <= target:
            raise InfrastructureError("fixture_token_target_not_met")
        return result, low - 1

    executed = []
    try:
        short = execute(
            "short_stream",
            [{"role": "user", "content": f"Return exactly this marker and nothing else: {SHORT_MARKER}"}],
            exact_marker_validator,
        )
        executed.append(short["case"])

        original, last_index = fresh(28000)
        cold = execute("cold_28k", original, cold_validator)
        executed.append(cold["case"])

        offset = last_index % 7
        expected = [offset, offset + 3]
        continuation = original + [
            {"role": "assistant", "content": cold["response_content"]},
            {
                "role": "user",
                "content": (
                    f"Evaluate gamma_{last_index}([-2, 0, 3]). Return ONLY the JSON array result, "
                    "with no markdown or explanation."
                ),
            },
        ]
        required = min(20000, int(cold["local_prompt_tokens"] * 0.75))
        warm = execute("warm_28k_continuation", continuation, json_array_validator(expected), required)
        executed.append(warm["case"])
    except InfrastructureError as exc:
        emit({"event": "matrix_terminal", "status": "BLOCKED", "executed_cases": executed, "error": str(exc)})
        return 2

    emit({"event": "matrix_terminal", "status": "COMPLETE", "executed_cases": executed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
