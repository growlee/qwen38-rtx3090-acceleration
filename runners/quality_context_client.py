"""Finite context adapter; old quality transport/scorer and 16k suites stay unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from benchmark_client import canonical, BenchmarkError
from quality_client import QualityClient, WARMUP, run_case, _emit

LENGTHS = (8192, 32768, 50000, 96000, 128000, 160000, 200000)
MAX_FIXTURE_BYTES = 256 * 1024**2
MAX_TOKENIZER_BYTES = 8 * 1024**2
CASE_TIMEOUT = 1800


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def freeze(model_path, output):
    from transformers import AutoTokenizer
    from quality_context_suite import FAMILY_IDS, build_case
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                             trust_remote_code=False)
    def count(case):
        return len(tokenizer.apply_chat_template(case['request']['messages'],
                   tokenize=True, return_dict=False, truncation=False,
                   add_generation_prompt=True, enable_thinking=False))
    cases = []
    for target in LENGTHS:
        for family in FAMILY_IDS:
            # Finite bisection of deterministic padding, not repeated inference.
            lo, hi = 0, min(32768, target // 32 + 256)
            if count(build_case(family, hi)) < target:
                raise RuntimeError('padding_bound_insufficient')
            while lo < hi:
                mid = (lo + hi) // 2
                if count(build_case(family, mid)) < target:
                    lo = mid + 1
                else:
                    hi = mid
            case = build_case(family, lo)
            actual = count(case)
            if not target <= actual <= target + 256:
                raise RuntimeError('padding_tolerance_exceeded')
            case.update(case_id=f'{family}-t{target}', family_id=family,
                        target_tokens=target, frozen_input_tokens=actual,
                        padding_count=lo)
            body = QualityClient('http://127.0.0.1:8080', None, 60)._body('qwen3.8-27b', case)
            case['request_sha256'] = digest(body)
            case['request_bytes'] = len(canonical(body))
            cases.append(case)
    value = {'suite_id': 'qwen38-context-12x7-v1', 'cases': cases,
             'max_len_mtp_huge': 204800, 'output_reserve': 4096, 'margin': 128}
    if len(cases) != 84 or max(c['frozen_input_tokens'] for c in cases) + 4096 + 128 > 204800:
        raise RuntimeError('frozen_window_invalid')
    raw = canonical(value)
    if len(raw) > MAX_FIXTURE_BYTES:
        raise RuntimeError('fixture_disk_bound_exceeded')
    with Path(output).open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'frozen_sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw), 'cases': len(cases)}), flush=True)


def post(client, path, body):
    conn = client.connection()
    try:
        conn.request('POST', path, body=canonical(body), headers=client.headers('application/json'))
        response = conn.getresponse()
        raw = response.read(MAX_TOKENIZER_BYTES + 1)
        if response.status != 200 or len(raw) > MAX_TOKENIZER_BYTES:
            raise BenchmarkError('context_tokenizer_failed')
        return json.loads(raw)
    finally:
        conn.close()


def tokenize(client, case, backend):
    body = {'model': 'qwen3.8-27b', 'messages': case['request']['messages'],
            'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
    if backend == 'llama':
        rendered = post(client, '/apply-template', body)
        if not isinstance(rendered.get('prompt'), str):
            raise BenchmarkError('context_template_invalid')
        tokens = post(client, '/tokenize', {'content': rendered['prompt'], 'add_special': True}).get('tokens')
        if not isinstance(tokens, list):
            raise BenchmarkError('context_tokenizer_invalid')
        count = len(tokens)
    else:
        count = post(client, '/tokenize', body).get('count')
    if type(count) is not int or count <= 0:
        raise BenchmarkError('context_tokenizer_invalid')
    return count


def capture(fixture, expected_sha, mode, max_len, lengths, backend):
    raw = Path(fixture).read_bytes()
    if len(raw) > MAX_FIXTURE_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError('fixture_identity_invalid')
    suite = json.loads(raw)
    cases = [c for c in suite['cases'] if c['target_tokens'] in lengths]
    client = QualityClient('http://127.0.0.1:8080', None, CASE_TIMEOUT)
    status, data, _ = client.get('/v1/models')
    models = json.loads(data).get('data', [])
    matches = [m for m in models if m.get('id') == 'qwen3.8-27b']
    if status != 200 or len(matches) != 1:
        raise RuntimeError('runtime_model_missing')
    if backend == 'vllm' and matches[0].get('max_model_len') != max_len:
        raise RuntimeError('runtime_context_limit_mismatch')
    # Freeze the actual backend counts before any generation, including warmup.
    counts = {c['case_id']: tokenize(client, c, backend) for c in cases}
    if backend == 'vllm' and any(counts[c['case_id']] != c['frozen_input_tokens'] for c in cases):
        raise RuntimeError('runtime_tokenization_differs_from_frozen')
    warmup, terminal = run_case(client, 'qwen3.8-27b', mode, WARMUP, kind='warmup')
    _emit(warmup)
    if terminal or warmup['outcome'] != 'pass':
        return 3
    for case in cases:
        count = counts[case['case_id']]
        metadata = {key: case[key] for key in ('family_id', 'target_tokens', 'request_bytes')}
        metadata.update(input_tokens=count, context_limit=max_len)
        if count + case['request']['generation']['max_tokens'] + 128 > max_len:
            _emit(dict(schema='qwen38-quality-jsonl/v1', kind='case', mode_id=mode,
                       model_id='qwen3.8-27b', case_id=case['case_id'], outcome='blocked',
                       error='capacity_bound_exceeded', request_sha256=case['request_sha256'], **metadata))
            continue
        metadata['started_epoch'] = time.time()
        _emit(dict(kind='case_start', mode_id=mode, case_id=case['case_id'], **metadata))
        row, terminal = run_case(client, 'qwen3.8-27b', mode, case)
        metadata['ended_epoch'] = time.time()
        row.update(metadata)
        # The legacy client treats empty text as terminal. A completed empty
        # answer is a valid semantic FAIL here and must not cancel this ladder.
        if row.get('error') == 'empty_output' and row.get('finish_reason') == 'stop':
            row.update(outcome='fail', error='empty_answer')
            terminal = False
        if row.get('request_sha256') != case['request_sha256']:
            row.update(outcome='error', error='request_identity_mismatch'); terminal = True
        usage = row.get('usage') or {}
        if row['outcome'] not in ('error',) and usage.get('prompt_tokens') != count:
            row.update(outcome='error', error='actual_prompt_tokens_mismatch'); terminal = True
        _emit(row)
        if terminal:
            return 3
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze-model'); p.add_argument('--output')
    p.add_argument('--fixture'); p.add_argument('--sha')
    p.add_argument('--mode'); p.add_argument('--max-len', type=int)
    p.add_argument('--lengths'); p.add_argument('--backend', choices=('vllm', 'llama'))
    args = p.parse_args()
    if args.freeze_model:
        freeze(args.freeze_model, args.output)
        return 0
    lengths = tuple(int(x) for x in args.lengths.split(','))
    if lengths not in (LENGTHS, LENGTHS[:3], LENGTHS[:1]):
        p.error('matrix_lengths_invalid')
    return capture(args.fixture, args.sha, args.mode, args.max_len, lengths, args.backend)


if __name__ == '__main__':
    sys.exit(main())
