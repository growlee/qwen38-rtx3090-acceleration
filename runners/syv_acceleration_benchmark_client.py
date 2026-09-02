#!/usr/bin/env python3
"""Content-free 1+3 single-user benchmark for the bounded syv MTP test."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any

import benchmark_client as base


WARMUP = {
    "case_id": "warmup",
    "messages": [
        {"role": "user", "content": "Ответь одним коротким предложением: зачем нужна резервная копия?"},
    ],
    "max_tokens": 64,
}

CASES = (
    {
        "case_id": "operations-guide-ru",
        "messages": [
            {"role": "system", "content": "Отвечай по-русски, практично и структурированно."},
            {
                "role": "user",
                "content": (
                    "Составь подробное руководство для домашнего AI-сервера с одной видеокартой: "
                    "как безопасно переключать текстовую модель и генератор изображений, контролировать "
                    "память GPU, проверять доступность сервисов и восстанавливаться после сбоя. "
                    "Добавь конкретный чек-лист и объясни причины каждого шага."
                ),
            },
        ],
        "max_tokens": 1024,
    },
    {
        "case_id": "architecture-analysis-en",
        "messages": [
            {"role": "system", "content": "Give a detailed, technically precise answer with clear sections."},
            {
                "role": "user",
                "content": (
                    "Design a reliable single-GPU local inference service for a household. Explain admission "
                    "control, model lifecycle, private-network access, failure recovery, storage discipline, "
                    "and a minimal monitoring strategy. Include trade-offs and an operator checklist."
                ),
            },
        ],
        "max_tokens": 1024,
    },
    {
        "case_id": "python-service-code",
        "messages": [
            {"role": "system", "content": "Write dependency-free Python and explain the important design choices."},
            {
                "role": "user",
                "content": (
                    "Implement a small thread-safe admission controller for one scarce worker. It must support "
                    "acquire with timeout, release, status reporting, and safe context-manager use. Include type "
                    "hints, meaningful errors, unit-test examples, and a concise explanation."
                ),
            },
        ],
        "max_tokens": 1024,
    },
)


def _prometheus_spec_metrics(raw: bytes) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return result
    for line in text.splitlines():
        if not line or line.startswith("#") or "spec_decode" not in line:
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        result[name] = result.get(name, 0.0) + value
    return {key: base.rounded(value) for key, value in sorted(result.items())}


def _metrics_snapshot(client: base.Client) -> dict[str, float]:
    try:
        status, raw, _content_type = client.get("/metrics")
    except base.BenchmarkError:
        return {}
    return _prometheus_spec_metrics(raw) if status == 200 else {}


def _metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in after.items():
        if key.endswith(("_total", "_count")) and key in before:
            result[key] = base.rounded(value - before[key])
        else:
            result[key] = value
    return result


def _spec_summary(delta: dict[str, float]) -> dict[str, float | None]:
    def find(suffixes: tuple[str, ...]) -> float | None:
        matches = [value for key, value in delta.items() if key.endswith(suffixes)]
        return None if not matches else sum(matches)

    accepted = find(("spec_decode_num_accepted_tokens_total", "spec_decode_accepted_tokens_total"))
    drafted = find(("spec_decode_num_draft_tokens_total", "spec_decode_draft_tokens_total"))
    acceptance = find(("spec_decode_draft_acceptance_rate", "spec_decode_acceptance_rate"))
    if accepted is not None and drafted not in {None, 0}:
        acceptance = accepted / drafted
    return {
        "accepted_tokens": base.rounded(accepted),
        "draft_tokens": base.rounded(drafted),
        "acceptance_ratio": base.rounded(acceptance),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=1900.0)
    parser.add_argument("--mode-id", required=True)
    args = parser.parse_args()

    client = base.Client(args.base_url, None, args.timeout)
    health_status, _health_raw, _health_type = client.get("/health")
    if health_status != 200:
        raise base.BenchmarkError("health_failed")
    models_status, models_raw, _models_type = client.get("/v1/models")
    if models_status != 200:
        raise base.BenchmarkError("models_failed")
    try:
        models = json.loads(models_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise base.BenchmarkError("models_invalid") from exc
    data = models.get("data") if isinstance(models, dict) else None
    if not isinstance(data, list) or not any(
        isinstance(item, dict) and item.get("id") == args.model for item in data
    ):
        raise base.BenchmarkError("model_missing")

    client.stream(args.model, WARMUP, warmup=True)
    spec_before = _metrics_snapshot(client)
    samples = [client.stream(args.model, case, warmup=False) for case in CASES]
    spec_after = _metrics_snapshot(client)
    spec_delta = _metric_deltas(spec_before, spec_after)

    draft_rows = [item for item in samples if item["draft_n"] is not None]
    metrics: dict[str, Any] = {
        "ttft_seconds_median": base.rounded(statistics.median(item["ttft_seconds"] for item in samples)),
        "decode_tokens_per_second_median": base.rounded(
            statistics.median(item["decode_tokens_per_second"] for item in samples)
        ),
        "end_to_end_tokens_per_second_median": base.rounded(
            statistics.median(
                item["completion_tokens"] / max(item["end_to_end_seconds"], 0.000001)
                for item in samples
            )
        ),
        "end_to_end_seconds_median": base.rounded(
            statistics.median(item["end_to_end_seconds"] for item in samples)
        ),
        "completion_tokens_total": sum(item["completion_tokens"] for item in samples),
        "draft_n_total": None if not draft_rows else sum(item["draft_n"] for item in draft_rows),
        "draft_n_accepted_total": None
        if not draft_rows
        else sum(item["draft_n_accepted"] for item in draft_rows),
    }
    metrics["draft_acceptance_ratio"] = (
        None
        if not draft_rows or metrics["draft_n_total"] == 0
        else base.rounded(metrics["draft_n_accepted_total"] / metrics["draft_n_total"])
    )

    print(
        json.dumps(
            {
                "schema": "qwen38-syv-acceleration-benchmark/v1",
                "mode_id": args.mode_id,
                "model_id": args.model,
                "health": "pass",
                "models": "pass",
                "warmup_count": 1,
                "measured_request_count": 3,
                "max_output_tokens_per_request": 1024,
                "samples": samples,
                "metrics": metrics,
                "spec_decode_metrics_before": spec_before,
                "spec_decode_metrics_after": spec_after,
                "spec_decode_metrics_delta": spec_delta,
                "spec_decode_summary": _spec_summary(spec_delta),
                "request_text_recorded": False,
                "response_text_recorded": False,
                "secrets_recorded": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def entrypoint() -> int:
    try:
        return main()
    except base.BenchmarkError as exc:
        print(
            json.dumps(
                {
                    "schema": "qwen38-syv-acceleration-diagnostic/v1",
                    "status": "FAIL",
                    "error_code": exc.code,
                    "case_id": exc.case_id,
                    "protocol": exc.protocol,
                    "request_text_recorded": False,
                    "response_text_recorded": False,
                    "secrets_recorded": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
