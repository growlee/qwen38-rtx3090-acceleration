"""Score retained synthetic responses once, after the GPU window is restored.

Usage: python score_quality.py INPUT.jsonl MODE OUTPUT.jsonl [WINDOW_FREE_BYTES]
Never queries a model; generated Python is sent only to quality_sandbox.
Existing output is an explicit stop, not permission to replay or overwrite.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from quality_client import evaluate_noncode
from quality_sandbox import evaluate_code
from quality_suite import SUITE


def score(input_path: Path, mode: str, output_path: Path, window_free_bytes: int | None = None, *, suite=None, max_growth_bytes=256*1024**2) -> dict:
    suite = SUITE if suite is None else suite
    raw = input_path.read_bytes()
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError('input_too_large')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    by_id = {}
    for row in rows:
        if row.get('kind') != 'case':
            continue
        if row.get('mode_id') != mode or row['case_id'] in by_id:
            raise ValueError('wrong_mode_or_duplicate_case')
        by_id[row['case_id']] = row
    allowed = {case['case_id'] for case in suite['cases']}
    if not set(by_id).issubset(allowed):
        raise ValueError('unknown_case')
    result = {'mode_id': mode, 'input_sha256': hashlib.sha256(raw).hexdigest(),
              'PASS': 0, 'FAIL': 0, 'BLOCKED': 0, 'cases': 0}
    infrastructure_blocked = False
    with output_path.open('x', encoding='utf-8') as out:
        for case in suite['cases']:
            row = by_id.get(case['case_id'])
            item = {'case_id': case['case_id'], 'category': case['category'],
                    'critical': case['critical'], 'mode_id': mode}
            if row is None:
                verdict = {'outcome': 'BLOCKED', 'error': 'generation_not_completed'}
            elif row.get('outcome') == 'error':
                verdict = {'outcome': 'BLOCKED', 'error': row.get('error'),
                           'failure_class': 'runtime_or_protocol'}
            elif row.get('error') == 'finish_reason_length':
                verdict = {'outcome': 'FAIL', 'error': 'finish_reason_length'}
            elif row.get('finish_reason') not in {'stop', 'tool_calls'}:
                verdict = {'outcome': 'BLOCKED', 'error': 'unproven_termination'}
            elif case['check']['type'] == 'python_function_cases':
                if window_free_bytes is not None:
                    fs = os.statvfs(input_path.parent)
                    free = fs.f_bavail * fs.f_frsize
                    if free < 32 * 1024**3 or window_free_bytes - free > max_growth_bytes:
                        infrastructure_blocked = True
                if infrastructure_blocked:
                    verdict = {'outcome': 'BLOCKED', 'error': 'sandbox_stopped_after_infrastructure_failure'}
                else:
                    verdict = evaluate_code(case, row.get('content', ''), mode_id=mode)
                    infrastructure_blocked = verdict['outcome'] == 'BLOCKED'
            else:
                passed = evaluate_noncode(case, row)
                verdict = {'outcome': 'PASS' if passed else 'FAIL',
                           'error': None if passed else 'quality_check_failed'}
            item.update(verdict)
            if row is not None:
                for key in ['request_sha256', 'usage', 'finish_reason', 'elapsed_seconds',
                            'ttft_first_generated_seconds', 'ttft_first_content_or_tool_seconds',
                            'decode_tokens_per_second', 'decode_rate_definition']:
                    item[key] = row.get(key)
            out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
            out.flush()
            os.fsync(out.fileno())
            result[item['outcome']] += 1
            result['cases'] += 1
            print(json.dumps({'mode': mode, 'scored': result['cases'], 'case': case['case_id'],
                              'outcome': item['outcome']}), flush=True)
    return result


if __name__ == '__main__':
    if len(sys.argv) not in {4, 5}:
        raise SystemExit('usage: score_quality.py INPUT.jsonl MODE OUTPUT.jsonl [WINDOW_FREE_BYTES]')
    baseline = int(sys.argv[4]) if len(sys.argv) == 5 else None
    print(json.dumps(score(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), baseline), sort_keys=True), flush=True)
