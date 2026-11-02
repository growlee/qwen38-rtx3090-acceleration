#!/usr/bin/env python3
"""Score the frozen Qwen expanded coding 120 with the reviewed 32-GiB floor."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


MODE_ID = "freetoken013-nvfp4-coding120"
MIN_FREE_BYTES = 32 * 1024**3


def _harness_dir() -> Path:
    return Path(__file__).resolve().with_name("harness")


def _load_score_dependencies():
    harness = _harness_dir()
    if not harness.is_dir():
        raise RuntimeError(f"frozen_harness_missing:{harness}")
    sys.path.insert(0, str(harness))
    from quality_expanded_suite import load_suite  # noqa: PLC0415
    from score_quality_32g import score  # noqa: PLC0415

    return load_suite, score


def _coding_suite(load_suite):
    suite = load_suite()
    cases = [case for case in suite["cases"] if case.get("category") == "coding"]
    case_ids = [case.get("case_id") for case in cases]
    if len(cases) != 120 or len(set(case_ids)) != 120:
        raise RuntimeError("coding120_identity_invalid")
    return {"schema_version": suite.get("schema_version"), "suite_id": suite.get("suite_id"), "cases": cases}


def _require_free_floor(path: Path) -> int:
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free_space_below_32g:{free}")
    return free


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    load_suite, score = _load_score_dependencies()
    suite = _coding_suite(load_suite)
    before_free = _require_free_floor(args.input.parent)
    result = score(args.input, MODE_ID, args.output, before_free, suite=suite)
    after_free = _require_free_floor(args.output.parent)
    if result.get("cases") != 120:
        raise RuntimeError(f"coding120_score_count_invalid:{result.get('cases')}")
    print(json.dumps({**result, "free_bytes_before": before_free, "free_bytes_after": after_free}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
