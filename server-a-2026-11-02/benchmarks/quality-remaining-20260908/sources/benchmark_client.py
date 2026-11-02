#!/usr/bin/env python3
"""One warm-up plus three content-free OpenAI streaming measurements."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_SSE_LINE_BYTES = 1024 * 1024

WARMUP = {
    "case_id": "warmup",
    "messages": [{"role": "user", "content": "Ответь одним коротким предложением: что такое резервная копия?"}],
    "max_tokens": 48,
}

CASES = (
    {
        "case_id": "concise-factual-ru",
        "messages": [
            {"role": "system", "content": "Отвечай точно и кратко."},
            {"role": "user", "content": "Объясни разницу между резервной копией и репликацией в четырёх предложениях."},
        ],
        "max_tokens": 128,
    },
    {
        "case_id": "structured-reasoning-en",
        "messages": [
            {"role": "system", "content": "Return only valid JSON with keys risks and mitigations."},
            {"role": "user", "content": "List three operational risks of running one GPU worker and one mitigation for each."},
        ],
        "max_tokens": 128,
    },
    {
        "case_id": "code-python",
        "messages": [
            {"role": "system", "content": "Write correct, dependency-free Python."},
            {"role": "user", "content": "Implement a function that returns the first duplicate string in an iterable while preserving encounter order. Include type hints and two assertions."},
        ],
        "max_tokens": 128,
    },
)


class BenchmarkError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        case_id: str | None = None,
        protocol: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.case_id = case_id
        self.protocol = protocol


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


class Client:
    def __init__(self, base_url: str, token: str | None, timeout: float) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in {"", "/"}:
            raise BenchmarkError("endpoint_not_loopback")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.token = token
        self.timeout = timeout

    def connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def headers(self, accept: str) -> dict[str, str]:
        value = {"Accept": accept, "Content-Type": "application/json"}
        if self.token is not None:
            value["Authorization"] = "Bearer " + self.token
        return value

    def get(self, path: str) -> tuple[int, bytes, str]:
        connection = self.connection()
        try:
            connection.request("GET", path, headers=self.headers("application/json"))
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise BenchmarkError("response_too_large")
            return response.status, raw, response.getheader("Content-Type", "") or ""
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            raise BenchmarkError("transport_error") from exc
        finally:
            connection.close()

    def stream(self, model: str, case: dict[str, Any], *, warmup: bool) -> dict[str, Any]:
        body = {
            "model": model,
            "messages": case["messages"],
            "temperature": 0,
            "max_tokens": case["max_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body_sha256 = hashlib.sha256(canonical(body)).hexdigest()
        connection = self.connection()
        started = time.perf_counter()
        first_delta_at: float | None = None
        completion_tokens: int | None = None
        prompt_tokens: int | None = None
        finish_reason: str | None = None
        server_prompt_tps: float | None = None
        draft_n: int | None = None
        draft_n_accepted: int | None = None
        visible_chars = 0
        delta_count = 0
        response_bytes = 0
        saw_done = False
        saw_usage = False
        termination = "pending"

        def protocol_state() -> dict[str, Any]:
            normalized_finish_reason = (
                finish_reason
                if finish_reason in {"stop", "length"}
                else "other" if finish_reason is not None else None
            )
            return {
                "saw_first_delta": first_delta_at is not None,
                "saw_finish_reason": finish_reason is not None,
                "finish_reason": normalized_finish_reason,
                "saw_done": saw_done,
                "saw_usage": saw_usage,
                "usage_complete": (
                    isinstance(prompt_tokens, int)
                    and prompt_tokens > 0
                    and isinstance(completion_tokens, int)
                    and completion_tokens > 0
                ),
                "termination": termination,
                "response_bytes": response_bytes,
                "delta_count": delta_count,
            }

        def stream_error(code: str) -> BenchmarkError:
            return BenchmarkError(code, case_id=str(case["case_id"]), protocol=protocol_state())

        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=canonical(body),
                headers=self.headers("text/event-stream"),
            )
            response = connection.getresponse()
            if response.status != 200:
                response.read(MAX_RESPONSE_BYTES + 1)
                raise BenchmarkError(f"http_{response.status}")
            if "text/event-stream" not in (response.getheader("Content-Type", "") or "").lower():
                raise BenchmarkError("content_type_invalid")
            while True:
                line = response.readline(MAX_SSE_LINE_BYTES + 1)
                if not line:
                    termination = "clean_eof"
                    break
                response_bytes += len(line)
                if len(line) > MAX_SSE_LINE_BYTES or response_bytes > MAX_RESPONSE_BYTES:
                    raise stream_error("response_too_large")
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    saw_done = True
                    termination = "done_sentinel"
                    break
                try:
                    event = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise stream_error("sse_invalid") from exc
                if not isinstance(event, dict):
                    raise stream_error("sse_invalid")
                choices = event.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            for field in ("reasoning", "reasoning_content", "content"):
                                text = delta.get(field)
                                if isinstance(text, str) and text:
                                    if first_delta_at is None:
                                        first_delta_at = time.perf_counter()
                                    visible_chars += len(text)
                                    delta_count += 1
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])
                usage = event.get("usage")
                if isinstance(usage, dict):
                    saw_usage = True
                    if isinstance(usage.get("prompt_tokens"), int):
                        prompt_tokens = usage["prompt_tokens"]
                    if isinstance(usage.get("completion_tokens"), int):
                        completion_tokens = usage["completion_tokens"]
                timings = event.get("timings")
                if isinstance(timings, dict):
                    candidate = timings.get("prompt_per_second")
                    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and math.isfinite(float(candidate)) and candidate > 0:
                        server_prompt_tps = float(candidate)
                    proposed = timings.get("draft_n")
                    accepted = timings.get("draft_n_accepted")
                    if (proposed is None) != (accepted is None):
                        raise stream_error("draft_timings_incomplete")
                    if proposed is not None:
                        if (
                            not isinstance(proposed, int)
                            or isinstance(proposed, bool)
                            or not isinstance(accepted, int)
                            or isinstance(accepted, bool)
                            or proposed < 0
                            or accepted < 0
                            or accepted > proposed
                        ):
                            raise stream_error("draft_timings_invalid")
                        draft_n, draft_n_accepted = proposed, accepted
        except BenchmarkError:
            raise
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            termination = "transport_error"
            raise stream_error("transport_error") from exc
        finally:
            connection.close()
        ended = time.perf_counter()
        if not saw_done or first_delta_at is None or finish_reason not in {"stop", "length"}:
            raise stream_error("stream_incomplete")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0 or not isinstance(completion_tokens, int) or completion_tokens <= 0:
            raise stream_error("usage_missing")
        if visible_chars <= 0 or delta_count <= 0:
            raise stream_error("empty_output")
        if not warmup and (visible_chars < 8 or completion_tokens < 4):
            raise stream_error("degenerate_output")
        ttft = first_delta_at - started
        e2e = ended - started
        decode_seconds = max(ended - first_delta_at, 0.000001)
        return {
            "case_id": case["case_id"],
            "request_sha256": body_sha256,
            "ttft_seconds": rounded(ttft),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_processing_tokens_per_second_estimate": rounded(prompt_tokens / max(ttft, 0.000001)),
            "server_prompt_tokens_per_second": rounded(server_prompt_tps),
            "decode_tokens_per_second": rounded(completion_tokens / decode_seconds),
            "end_to_end_seconds": rounded(e2e),
            "draft_n": draft_n,
            "draft_n_accepted": draft_n_accepted,
            "draft_acceptance_ratio": None if draft_n in {None, 0} else rounded(draft_n_accepted / draft_n),
            "delta_count": delta_count,
            "finish_reason": finish_reason,
            "nondegenerate": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-file")
    parser.add_argument("--timeout", type=float, default=1900.0)
    parser.add_argument("--mode-id", required=True)
    args = parser.parse_args()

    token = None
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="ascii").strip()
        if not token or len(token) > 4096:
            raise BenchmarkError("token_invalid")
    client = Client(args.base_url, token, args.timeout)
    health_status, _health_body, _health_type = client.get("/health")
    if health_status != 200:
        raise BenchmarkError("health_failed")
    models_status, models_raw, _models_type = client.get("/v1/models")
    if models_status != 200:
        raise BenchmarkError("models_failed")
    try:
        models = json.loads(models_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError("models_invalid") from exc
    data = models.get("data") if isinstance(models, dict) else None
    if not isinstance(data, list) or not any(isinstance(item, dict) and item.get("id") == args.model for item in data):
        raise BenchmarkError("model_missing")

    client.stream(args.model, WARMUP, warmup=True)
    samples = [client.stream(args.model, case, warmup=False) for case in CASES]
    draft_rows = [item for item in samples if item["draft_n"] is not None]
    metrics = {
        "ttft_seconds_median": rounded(statistics.median(item["ttft_seconds"] for item in samples)),
        "prompt_processing_tokens_per_second_estimate_median": rounded(statistics.median(item["prompt_processing_tokens_per_second_estimate"] for item in samples)),
        "decode_tokens_per_second_median": rounded(statistics.median(item["decode_tokens_per_second"] for item in samples)),
        "end_to_end_seconds_median": rounded(statistics.median(item["end_to_end_seconds"] for item in samples)),
        "draft_n_total": None if not draft_rows else sum(item["draft_n"] for item in draft_rows),
        "draft_n_accepted_total": None if not draft_rows else sum(item["draft_n_accepted"] for item in draft_rows),
    }
    metrics["draft_acceptance_ratio"] = (
        None
        if not draft_rows or metrics["draft_n_total"] == 0
        else rounded(metrics["draft_n_accepted_total"] / metrics["draft_n_total"])
    )
    print(json.dumps({
        "schema": "qwen38-nvfp4-one-shot-benchmark/v1",
        "mode_id": args.mode_id,
        "model_id": args.model,
        "health": "pass",
        "models": "pass",
        "warmup_count": 1,
        "measured_request_count": 3,
        "samples": samples,
        "metrics": metrics,
        "request_text_recorded": False,
        "response_text_recorded": False,
        "secrets_recorded": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
