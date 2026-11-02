#!/usr/bin/env python3
"""Run or score the frozen coding120 benchmark against the DFlash endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


BENCH = Path("/srv/ai-server/artifacts/experiments/freetoken013-benchmark120-20260918")
BASE_URL = "http://127.0.0.1:19193"
MODEL = "qwen3.8-27b"
MODE_ID = "dflash2-fast-n7-prefix32k-coding120-20260919"


def _import_benchmark_modules():
    if not BENCH.is_dir():
        raise RuntimeError(f"benchmark_missing:{BENCH}")
    sys.path.insert(0, str(BENCH))
    import benchmark_client120  # noqa: PLC0415
    import score120  # noqa: PLC0415

    benchmark_client120.BASE_URL = BASE_URL
    benchmark_client120.MODEL = MODEL
    benchmark_client120.MODE_ID = MODE_ID
    score120.MODE_ID = MODE_ID
    return benchmark_client120, score120


def _generate(output: Path, benchmark_client120) -> int:
    _, QualityClient, _, _, load_suite = benchmark_client120._load_frozen_harness()
    suite = load_suite()
    cases = benchmark_client120._coding_cases(lambda: suite)
    coding_suite = {
        "schema_version": suite.get("schema_version"),
        "suite_id": suite.get("suite_id"),
        "cases": cases,
    }
    import quality_client  # noqa: PLC0415

    client = QualityClient(BASE_URL, None, benchmark_client120.TIMEOUT_SECONDS)
    preflight = quality_client.context_preflight(client, MODEL, coding_suite, "vllm")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "context-preflight.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(preflight, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return benchmark_client120.main(["--output", str(output / "matrix.jsonl")])


def _score(output: Path, score120) -> int:
    return score120.main(
        ["--input", str(output / "matrix.jsonl"), "--output", str(output / "scored.jsonl")]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", type=Path, metavar="OUTDIR")
    action.add_argument("--score", type=Path, metavar="OUTDIR")
    args = parser.parse_args(argv)

    benchmark_client120, score120 = _import_benchmark_modules()
    if args.generate is not None:
        return _generate(args.generate, benchmark_client120)
    return _score(args.score, score120)


if __name__ == "__main__":
    raise SystemExit(main())
