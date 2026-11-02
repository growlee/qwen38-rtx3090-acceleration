"""Exactly five sequential streamed requests; run inside the isolated container.

Writes JSONL to stdout for a caller to retain outside the read-only container.
No retry, tuning, cache reset, extra warmup request or user data.
"""
from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import threading
import time

MODEL = 'qwen38-nvfp4-freetoken'
LIMIT = 8 * 1024 * 1024


def jit_records():
    path = Path('/cache/jit-trace.jsonl')
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise RuntimeError('jit_trace_absent_or_oversized')
    # Append records are atomic and have a trailing newline; ignore a partial
    # tail if a different process is recording at this instant.
    raw = path.read_bytes()
    lines = raw.splitlines() if raw.endswith(b'\n') else raw.splitlines()[:-1]
    records = [json.loads(line) for line in lines]
    if not any(r.get('event') == 'jit_listener_installed' for r in records):
        raise RuntimeError('jit_listener_not_proven')
    return records


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def get(path):
    c = http.client.HTTPConnection('127.0.0.1', 1919, timeout=15)
    try:
        c.request('GET', path)
        r = c.getresponse()
        body = r.read(LIMIT + 1)
        if r.status != 200 or len(body) > LIMIT:
            raise RuntimeError('GET_failed_' + path)
        return json.loads(body)
    finally:
        c.close()


class Vram:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                    '--format=csv,noheader,nounits'], capture_output=True,
                                   text=True, check=True, timeout=5)
                self.samples.append({'monotonic': time.perf_counter(),
                                     'used_mib': int(r.stdout.strip())})
            except Exception as exc:
                self.errors.append(type(exc).__name__)
                break
            self.stop.wait(0.25)


def stream(case_id, messages, input_tokens):
    jit_before = jit_records()
    payload = {'model': MODEL, 'messages': messages, 'temperature': 0,
               'max_tokens': 256, 'stream': True,
               'stream_options': {'include_usage': True},
               'chat_template_kwargs': {'enable_thinking': False}}
    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
    result = {'case': case_id, 'status': 'FAIL', 'local_prompt_tokens': input_tokens,
              'request_sha256': hashlib.sha256(body).hexdigest(),
              'ttft_definition': 'request start to first non-whitespace delta.content',
              'usage': None, 'response_content': '', 'reasoning_chars': 0,
              'first_content_seconds': None, 'finish_reason': None, 'done': False,
              'jit_separately_measured': False}
    c = http.client.HTTPConnection('127.0.0.1', 1919, timeout=600)
    sampler = Vram()
    sampler.thread.start()
    start = time.perf_counter()
    first = None
    total_bytes = 0
    try:
        c.request('POST', '/v1/chat/completions', body=body,
                  headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        r = c.getresponse()
        result['http_status'] = r.status
        if r.status != 200 or not r.getheader('Content-Type', '').startswith('text/event-stream'):
            result['error_body'] = r.read(4096).decode(errors='replace')
            raise RuntimeError('stream_http_or_content_type')
        while True:
            line = r.readline(1024 * 1024 + 1)
            now = time.perf_counter()
            total_bytes += len(line)
            if len(line) > 1024 * 1024 or total_bytes > LIMIT or now - start > 900:
                raise RuntimeError('stream_bound_exceeded')
            if not line:
                break
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                result['done'] = True
                break
            obj = json.loads(data)
            if obj.get('error'):
                result['server_error'] = obj['error']
                raise RuntimeError('stream_server_error')
            if obj.get('usage') is not None:
                result['usage'] = obj['usage']
            for choice in obj.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                if not isinstance(text, str):
                    raise RuntimeError('non_text_delta')
                if first is None and text.strip():
                    first = now
                result['response_content'] += text
                result['reasoning_chars'] += len(delta.get('reasoning_content') or '')
                if delta.get('tool_calls'):
                    raise RuntimeError('unexpected_tool_call')
                if choice.get('finish_reason'):
                    result['finish_reason'] = choice['finish_reason']
        usage = result['usage'] or {}
        if not (result['done'] and first is not None and result['finish_reason'] in ('stop', 'length')
                and type(usage.get('prompt_tokens')) is int and usage['prompt_tokens'] > 0
                and type(usage.get('completion_tokens')) is int
                and 0 < usage['completion_tokens'] <= 256 and result['reasoning_chars'] == 0):
            raise RuntimeError('stream_or_usage_incomplete_or_thinking_enabled')
        result['cached_tokens'] = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        if case_id.endswith('_continuation') and result['cached_tokens'] <= 0:
            raise RuntimeError('continuation_without_reported_cache_hit')
        result['status'] = 'PASS'
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        duration = time.perf_counter() - start
        c.close()
        sampler.stop.set()
        sampler.thread.join(timeout=6)
        result['duration_seconds'] = duration
        result['first_content_seconds'] = None if first is None else first - start
        result['vram_samples'] = sampler.samples
        result['vram_sampling_errors'] = sampler.errors
        result['peak_vram_mib'] = max((x['used_mib'] for x in sampler.samples), default=None)
        n = (result['usage'] or {}).get('completion_tokens', 0)
        result['completion_tokens_per_total_second'] = n / duration if n else None
        result['estimated_post_first_content_tokens_per_second'] = (
            (n - 1) / (duration - (first - start)) if first and n > 1 and duration > first - start else None)
        try:
            jit = [x for x in jit_records()[len(jit_before):] if x.get('event') == 'triton_compile']
            result['triton_compilation_events'] = jit
            result['triton_compile_seconds_sum'] = sum(x['compiler_microseconds'] for x in jit
                                                       if not x['cache_hit']) / 1e6
            result['triton_compile_cache_load_seconds_sum'] = sum(x['compiler_microseconds'] for x in jit
                                                                  if x['cache_hit']) / 1e6
            result['jit_separately_measured'] = True
            result['jit_scope'] = 'Triton compiler listener only; raw TTFT is not adjusted or subtracted'
        except Exception as exc:
            result['status'] = 'FAIL'
            result['jit_error'] = str(exc)
    emit(result)
    if result['status'] != 'PASS':
        raise RuntimeError('terminal_case_failure_' + case_id)
    return result['response_content']


def main(skip_short=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)

    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True,
                   add_generation_prompt=True, enable_thinking=False, return_dict=False))

    def execute(name, messages):
        n = count(messages)
        if n + 256 > 32768:
            raise RuntimeError('context_budget_exceeded')
        return stream(name, messages, n)

    def fresh(target, label):
        prefix = label + ': Synthetic source fixture. Read the code and describe its intent in two short sentences.\n'
        suffix = '\nGive only the brief description, without reasoning.'
        def messages(n):
            code = ''.join(f'def {label}_{i}(items):\n    return [x + {i % 7} for x in items if x >= 0]\n' for i in range(n))
            return [{'role': 'user', 'content': prefix + code + suffix}]
        lo, hi = 0, target
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(messages(mid)) <= target:
                lo = mid
            else:
                hi = mid - 1
        result = messages(lo)
        if not target - 128 <= count(result) <= target:
            raise RuntimeError('fixture_token_target_not_met')
        return result

    emit({'event': 'before_matrix', 'stats': get('/v1/stats'), 'cache': get('/v1/cache/status'),
          'startup_jit_records': jit_records()})
    if not skip_short:
        execute('short_stream', [{'role': 'user', 'content': 'Reply with one short sentence explaining what a Python function is.'}])
    for target, label in ((8000, 'alpha'), (28000, 'gamma')):
        messages = fresh(target, label)
        answer = execute(str(target) + '_fresh', messages)
        continuation = messages + [{'role': 'assistant', 'content': answer},
                                  {'role': 'user', 'content': 'Now name one useful unit test for this code in one sentence.'}]
        execute(str(target) + '_continuation', continuation)
    emit({'event': 'after_matrix', 'stats': get('/v1/stats'), 'cache': get('/v1/cache/status')})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-short-stream', action='store_true')
    main(skip_short=parser.parse_args().skip_short_stream)
