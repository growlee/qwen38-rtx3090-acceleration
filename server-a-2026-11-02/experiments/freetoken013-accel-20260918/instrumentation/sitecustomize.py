"""Candidate-only, trigger-gated CUDA-graph replay profiler.

The parent experiment mounts the prior JIT observer at /original-instrumentation.
This overlay executes that observer first, then observes one graph replay only after
the candidate user creates /cache/profile-request.  It never changes model code,
kernel selection, graph capture, or request handling.
"""

import json
import os
import time


_CACHE = "/cache"
_TRIGGER = _CACHE + "/profile-request"
_SUMMARY = _CACHE + "/profile-summary.txt"
_TRACE = _CACHE + "/profile-trace.json"
_TRACE_TMP = _TRACE + ".tmp"
_TRACE_LIMIT = 128 << 20
_STATE = {
    "armed": False,
    "disabled": False,
    "attempted": False,
    "hook": "freetoken.engine.graph.GraphRunner.replay",
}


def _stderr(message):
    try:
        os.write(2, ("FTACCEL_PROFILE: " + message + "\n").encode("utf-8", "replace"))
    except BaseException:
        pass


def _write_summary(row, table=""):
    """Best-effort bounded diagnostic output; never affects inference."""
    try:
        row = dict(row)
        row.update(
            hook=_STATE["hook"],
            pid=os.getpid(),
            monotonic=time.monotonic(),
        )
        text = json.dumps(row, sort_keys=True) + "\n"
        if table:
            text += table[:60 << 10]
            if len(table) > 60 << 10:
                text += "\n[table truncated]\n"
        data = text.encode("utf-8", "replace")
        tmp = _SUMMARY + ".tmp"
        with open(tmp, "wb") as out:
            out.write(data[:64 << 10])
        os.replace(tmp, _SUMMARY)
    except BaseException:
        _stderr("summary_write_failed")


def _disable(stage, exc):
    _STATE["disabled"] = True
    _write_summary(
        {
            "status": "disabled",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
            "timing_available": False,
            "actual_kernel_event_count": None,
        }
    )
    _stderr("disabled at " + stage + ": " + type(exc).__name__)


def _claim_trigger():
    """Consume one request atomically.  A missing/racing trigger is harmless."""
    if _STATE["disabled"] or _STATE["attempted"] or not os.path.exists(_TRIGGER):
        return False
    claimed = _TRIGGER + ".claimed." + str(os.getpid())
    try:
        os.replace(_TRIGGER, claimed)
    except FileNotFoundError:
        return False
    except BaseException as exc:
        _disable("claim_trigger", exc)
        return False
    try:
        os.unlink(claimed)
    except BaseException:
        pass
    _STATE["attempted"] = True
    return True


def _trace_kernel_count(path):
    """Return actual Chrome-trace kernel events, or None when unavailable."""
    try:
        size = os.path.getsize(path)
        if size > _TRACE_LIMIT:
            os.unlink(path)
            return None, "trace_dropped_over_128MiB"
        with open(path, "r", encoding="utf-8") as source:
            events = json.load(source).get("traceEvents", [])
        return sum(1 for event in events if event.get("cat") == "kernel"), None
    except BaseException as exc:
        return None, type(exc).__name__


def _profile_replay(original, runner, args, kwargs):
    """Profile one already-captured graph replay, then always return its result."""
    replay_started = False
    replay_returned = False
    result = None
    try:
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            post_processing_timeout_s=5,
        ) as profiler:
            replay_started = True
            result = original(runner, *args, **kwargs)
            replay_returned = True

        table = profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=50
        )
        device_event_count = len(profiler.events())
        kernel_count = None
        trace_note = None
        try:
            profiler.export_chrome_trace(_TRACE_TMP)
            if os.path.getsize(_TRACE_TMP) <= _TRACE_LIMIT:
                os.replace(_TRACE_TMP, _TRACE)
                kernel_count, trace_note = _trace_kernel_count(_TRACE)
            else:
                os.unlink(_TRACE_TMP)
                trace_note = "trace_dropped_over_128MiB"
        except BaseException as exc:
            try:
                if os.path.exists(_TRACE_TMP):
                    os.unlink(_TRACE_TMP)
            except BaseException:
                pass
            trace_note = "trace_export_" + type(exc).__name__

        cuda_kernel_timing_available = kernel_count is not None and kernel_count > 0
        _write_summary(
            {
                "status": "captured",
                "capture_mode": "one_cuda_graph_replay",
                "decode_forward_limit": 1,
                "timing_available": cuda_kernel_timing_available,
                "profiler_event_count": device_event_count,
                "actual_kernel_event_count": kernel_count,
                "trace_note": trace_note,
                "trace_path": _TRACE if kernel_count is not None else None,
            },
            table if cuda_kernel_timing_available else "",
        )
        return result
    except BaseException as exc:
        # Do not retry a real replay failure: its original exception and semantics win.
        if replay_started and not replay_returned:
            _disable("profile_replay_during_replay", exc)
            raise
        # Profiling must not convert a completed (or not-yet-started) replay into a
        # request failure.  If collection could not start, run it once without profiling.
        _disable("profile_replay", exc)
        return result if replay_returned else original(runner, *args, **kwargs)


def _install_hook():
    try:
        import torch
        from freetoken.engine.graph import GraphRunner

        original = GraphRunner.replay
        if getattr(original, "_ftaccel_profile_hook", False):
            _write_summary(
                {
                    "status": "disabled",
                    "stage": "install_hook",
                    "error_type": "RuntimeError",
                    "error": "GraphRunner.replay_already_wrapped",
                    "timing_available": False,
                    "actual_kernel_event_count": None,
                }
            )
            return

        def replay(runner, *args, **kwargs):
            if _claim_trigger():
                return _profile_replay(original, runner, args, kwargs)
            return original(runner, *args, **kwargs)

        replay._ftaccel_profile_hook = True
        GraphRunner.replay = replay
        _STATE["armed"] = True
        _write_summary(
            {
                "status": "armed",
                "capture_mode": "one_cuda_graph_replay",
                "decode_forward_limit": 1,
                "trigger": _TRIGGER,
                "timing_available": False,
                "actual_kernel_event_count": None,
            }
        )
    except BaseException as exc:
        _disable("install_hook", exc)


def _run_original():
    path = "/original-instrumentation/sitecustomize.py"
    try:
        with open(path, "rb") as source:
            code = compile(source.read(), path, "exec")
        namespace = {"__file__": path, "__name__": "_ftaccel_original_sitecustomize__"}
        exec(code, namespace, namespace)
    except FileNotFoundError:
        _write_summary(
            {
                "status": "disabled",
                "stage": "delegate_original",
                "error_type": "FileNotFoundError",
                "error": "original_instrumentation_not_mounted",
                "timing_available": False,
                "actual_kernel_event_count": None,
            }
        )
        _stderr("original instrumentation missing; profiler not installed")
        return False
    except BaseException:
        # Preserve the original observer's own failure semantics, including os._exit.
        raise
    return True


if _run_original():
    _install_hook()
