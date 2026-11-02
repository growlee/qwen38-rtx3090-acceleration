"""Optional observation of Triton 3.6 compilation through its listener API.

Loaded in the candidate's Python processes only. Does not replace compiler,
kernel selection, cache policy, model code, or generated kernels.
"""
import importlib.metadata
import json
import os
import time

if os.environ.get('FT013_JIT_TRACE') == '/cache/jit-trace.jsonl':
    try:
        from triton import knobs
        if importlib.metadata.version('triton') != '3.6.0' or knobs.compilation.listener is not None:
            raise RuntimeError('unexpected_triton_or_existing_compilation_listener')
        _fd = os.open('/cache/jit-trace.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)

        def _write(row):
            row.update(pid=os.getpid(), monotonic=time.perf_counter())
            data = (json.dumps(row, sort_keys=True) + '\n').encode()
            if len(data) > 4096 or os.write(_fd, data) != len(data):
                raise RuntimeError('jit_trace_write_failed')

        def _compiled(*, src, metadata, metadata_group, times, cache_hit):
            _write({'event': 'triton_compile', 'kernel': str(metadata.get('name', ''))[:200],
                    'cache_hit': bool(cache_hit), 'compiler_microseconds': int(times.total)})

        knobs.compilation.listener = _compiled
        _write({'event': 'jit_listener_installed', 'version': '3.6.0'})
    except BaseException as exc:
        os.write(2, ('FT013_JIT_OBSERVER_FAILED: ' + type(exc).__name__ + '\n').encode())
        os._exit(70)
