"""Capture the exact 84-case KV-4350 repair selection against the frozen fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from quality_context_client import (
    CASE_TIMEOUT, MAX_FIXTURE_BYTES, QualityClient, WARMUP, _emit, run_case, tokenize,
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


ALLOWED = {
    "dflash2-huge-n7-kv4350mb-204800": (8192, 32768, 50000, 96000, 128000, 160000, 200000),
}


def capture_selected(fixture, fixture_sha, mode, max_len, backend, selection):
    raw = Path(fixture).read_bytes()
    require(len(raw) <= MAX_FIXTURE_BYTES and hashlib.sha256(raw).hexdigest() == fixture_sha,
            "fixture_identity_invalid")
    suite = json.loads(raw)
    require(mode in ALLOWED, "mode_outside_repair_matrix")
    require(selection["mode_id"] == mode and selection["max_len"] == max_len
            and selection["backend"] == backend, "selection_runtime_contract_mismatch")
    require(selection["fixture_sha256"] == fixture_sha, "selection_fixture_mismatch")
    selected_ids = selection["case_ids"]
    require(isinstance(selected_ids, list) and selected_ids
            and len(selected_ids) == len(set(selected_ids)), "selection_case_set_invalid")
    expected = [case for case in suite["cases"] if case["target_tokens"] in ALLOWED[mode]]
    by_id = {case["case_id"]: case for case in expected}
    require(set(selected_ids) == set(by_id) and len(selected_ids) == len(expected),
            "selection_order_or_case_set_changed")
    require(type(selection["skip_recorded_warmup"]) is bool, "warmup_selection_invalid")

    client = QualityClient("http://127.0.0.1:8080", None, CASE_TIMEOUT)
    status, data, _ = client.get("/v1/models")
    matches = [item for item in json.loads(data).get("data", []) if item.get("id") == "qwen3.8-27b"]
    require(status == 200 and len(matches) == 1, "runtime_model_missing")
    if backend == "vllm":
        require(matches[0].get("max_model_len") == max_len, "runtime_context_limit_mismatch")
    counts = {case_id: tokenize(client, by_id[case_id], backend) for case_id in selected_ids}
    if backend == "vllm":
        require(all(counts[case_id] == by_id[case_id]["frozen_input_tokens"]
                    for case_id in selected_ids), "runtime_tokenization_differs_from_frozen")

    if not selection["skip_recorded_warmup"]:
        warmup, terminal = run_case(client, "qwen3.8-27b", mode, WARMUP, kind="warmup")
        _emit(warmup)
        if terminal or warmup["outcome"] != "pass":
            return 3
    for case_id in selected_ids:
        case = by_id[case_id]
        count = counts[case_id]
        metadata = {key: case[key] for key in ("family_id", "target_tokens", "request_bytes")}
        metadata.update(input_tokens=count, context_limit=max_len)
        if count + case["request"]["generation"]["max_tokens"] + 128 > max_len:
            _emit(dict(schema="qwen38-quality-jsonl/v1", kind="case", mode_id=mode,
                       model_id="qwen3.8-27b", case_id=case_id, outcome="blocked",
                       error="capacity_bound_exceeded", request_sha256=case["request_sha256"], **metadata))
            continue
        metadata["started_epoch"] = time.time()
        _emit(dict(kind="case_start", mode_id=mode, case_id=case_id, **metadata))
        row, terminal = run_case(client, "qwen3.8-27b", mode, case)
        metadata["ended_epoch"] = time.time()
        row.update(metadata)
        if row.get("error") == "empty_output" and row.get("finish_reason") == "stop":
            row.update(outcome="fail", error="empty_answer")
            terminal = False
        if row.get("request_sha256") != case["request_sha256"]:
            row.update(outcome="error", error="request_identity_mismatch")
            terminal = True
        usage = row.get("usage") or {}
        if row["outcome"] != "error" and usage.get("prompt_tokens") != count:
            row.update(outcome="error", error="actual_prompt_tokens_mismatch")
            terminal = True
        _emit(row)
        if terminal:
            return 3
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-sha", required=True)
    parser.add_argument("--mode", required=True, choices=tuple(ALLOWED))
    args = parser.parse_args()
    raw = Path(args.selection).read_bytes()
    require(len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == args.selection_sha,
            "selection_identity_invalid")
    selection = json.loads(raw)["modes"][args.mode]
    return capture_selected(args.fixture, args.sha, args.mode, selection["max_len"],
                            selection["backend"], selection)


if __name__ == "__main__":
    raise SystemExit(main())
