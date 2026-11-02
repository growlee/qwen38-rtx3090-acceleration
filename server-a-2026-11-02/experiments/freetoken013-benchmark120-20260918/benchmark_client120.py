#!/usr/bin/env python3
"""Run the frozen expanded-Qwen coding 120 once against the FreeToken endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "http://127.0.0.1:19191"
MODEL = "qwen38-nvfp4-freetoken"
MODE_ID = "freetoken013-nvfp4-coding120"
TIMEOUT_SECONDS = 180.0


def _harness_dir() -> Path:
    return Path(__file__).resolve().with_name("harness")


def _load_frozen_harness():
    harness = _harness_dir()
    if not harness.is_dir():
        raise RuntimeError(f"frozen_harness_missing:{harness}")
    sys.path.insert(0, str(harness))
    from benchmark_client import BenchmarkError  # noqa: PLC0415
    from quality_client import QualityClient, WARMUP, run_case  # noqa: PLC0415
    from quality_expanded_suite import load_suite  # noqa: PLC0415

    return BenchmarkError, QualityClient, WARMUP, run_case, load_suite


def _coding_cases(load_suite):
    suite = load_suite()
    cases = [case for case in suite["cases"] if case.get("category") == "coding"]
    case_ids = [case.get("case_id") for case in cases]
    if len(cases) != 120 or len(set(case_ids)) != 120:
        raise RuntimeError("coding120_identity_invalid")
    if any(
        case.get("request", {}).get("generation", {}).get("temperature") != 0
        or case.get("request", {}).get("generation", {}).get("max_tokens") != 2048
        or not isinstance(case.get("request", {}).get("generation", {}).get("seed"), int)
        for case in cases
    ):
        raise RuntimeError("coding120_generation_contract_invalid")
    return cases


def _write_row(stream, row: dict, started_at_utc: str) -> None:
    retained = {**row, "started_at_utc": started_at_utc}
    line = json.dumps(retained, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    print(
        json.dumps(
            {
                "case_id": row.get("case_id"),
                "kind": row.get("kind"),
                "outcome": row.get("outcome"),
                "ttft": row.get("ttft_first_generated_seconds"),
                "completion_tokens": usage.get("completion_tokens"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    stream.write(line + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    BenchmarkError, QualityClient, WARMUP, run_case, load_suite = _load_frozen_harness()
    cases = _coding_cases(load_suite)
    client = QualityClient(BASE_URL, None, TIMEOUT_SECONDS)
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
    if not isinstance(data, list) or not any(isinstance(item, dict) and item.get("id") == MODEL for item in data):
        print("model_missing", file=sys.stderr)
        return 2

    try:
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            warmup, terminal = run_case(client, MODEL, MODE_ID, WARMUP, kind="warmup")
            _write_row(output, warmup, started_at_utc)
            if terminal or warmup["outcome"] != "pass":
                return 3
            for case in cases:
                started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                row, terminal = run_case(client, MODEL, MODE_ID, case)
                _write_row(output, row, started_at_utc)
                if terminal:
                    return 3
    except FileExistsError:
        print("output_exists", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
