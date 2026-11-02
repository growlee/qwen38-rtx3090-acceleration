#!/usr/bin/env python3
"""Run and sanitize one frozen ``ft bench bw`` hardware calibration.

The upstream profile contains host and timestamp fields.  They are validated
but deliberately omitted from the immutable report, which contains only the
reviewed identity, numeric bandwidth measurements, the backend verdict, and a
binding to the concurrent GPU/power safety receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence


RUNNER_VERSION = "ai-server-freetoken-bandwidth-runner/v1"
REPORT_SCHEMA = "ai-server-freetoken-bandwidth-report/v1"
SUITE_ID = "acceleration-freetoken-bandwidth-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_JSON_BYTES = 1_048_576


class BandwidthError(RuntimeError):
    """Stable content-free benchmark failure."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact(value: Any, keys: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise BandwidthError(code)
    return value


def _positive(value: Any, code: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise BandwidthError(code)
    return float(value)


def load_suite(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BandwidthError("bandwidth_suite_unreadable") from exc
    _exact(
        value,
        {
            "schema_version", "profile_id", "command", "upstream", "workload",
            "measurement_counts", "gpu_safety", "acceptance", "privacy",
        },
        "bandwidth_suite_schema_invalid",
    )
    if value["schema_version"] != 1 or value["profile_id"] != SUITE_ID:
        raise BandwidthError("bandwidth_suite_identity_invalid")
    command = _exact(value["command"], {"family", "argv", "output_profile_path", "timeout_seconds"}, "bandwidth_command_invalid")
    expected_argv = [
        "ft", "bench", "bw", "--model", "gpt-oss-20b",
        "--out", "/cache/.cache/freetoken/benchbw.json",
    ]
    if (
        command["family"] != ["ft", "bench", "bw"]
        or command["argv"] != expected_argv
        or command["output_profile_path"] != "/cache/.cache/freetoken/benchbw.json"
        or command["timeout_seconds"] != 1800
    ):
        raise BandwidthError("bandwidth_command_invalid")
    upstream = _exact(value["upstream"], {"repository", "revision", "release", "profile_version"}, "bandwidth_upstream_invalid")
    if upstream != {
        "repository": "https://github.com/FlashML-org/FreeToken",
        "revision": "9db1a39455a3fb107f3db83e381d10ceadfe5d99",
        "release": "v0.1.2",
        "profile_version": 4,
    }:
        raise BandwidthError("bandwidth_upstream_invalid")
    workload = _exact(
        value["workload"],
        {"model_repository", "model_revision", "model_name", "expert_format_cli", "expert_format_profile", "hidden", "intermediate", "experts", "top_k"},
        "bandwidth_workload_invalid",
    )
    if workload != {
        "model_repository": "openai/gpt-oss-20b",
        "model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "model_name": "gpt-oss-20b",
        "expert_format_cli": "mxfp4",
        "expert_format_profile": "mxfp4_triton",
        "hidden": 2880,
        "intermediate": 2880,
        "experts": 32,
        "top_k": 4,
    }:
        raise BandwidthError("bandwidth_workload_invalid")
    counts = _exact(value["measurement_counts"], {"runs", "cpu_stream_iterations", "pcie_linear_iterations", "kernel_cpu_iterations", "kernel_pcie_iterations"}, "bandwidth_counts_invalid")
    if counts != {"runs": 1, "cpu_stream_iterations": 8, "pcie_linear_iterations": 30, "kernel_cpu_iterations": 64, "kernel_pcie_iterations": 20}:
        raise BandwidthError("bandwidth_counts_invalid")
    safety = _exact(value["gpu_safety"], {"device_index", "expected_name", "power_cap_w", "maximum_temperature_c", "minimum_vram_free_mib", "minimum_ram_available_mib", "maximum_swap_io_delta", "minimum_safety_sample_count"}, "bandwidth_safety_invalid")
    if safety != {
        "device_index": 0, "expected_name": "NVIDIA GeForce RTX 3090", "power_cap_w": 300.0,
        "maximum_temperature_c": 85.0, "minimum_vram_free_mib": 512.0,
        "minimum_ram_available_mib": 4096.0, "maximum_swap_io_delta": 0,
        "minimum_safety_sample_count": 2,
    }:
        raise BandwidthError("bandwidth_safety_invalid")
    acceptance = _exact(value["acceptance"], {"status", "reason", "metric_minimum_exclusive_gbs", "decision_threshold_cpu_to_pcie", "ratio_absolute_tolerance", "decision_rule"}, "bandwidth_acceptance_invalid")
    if acceptance != {
        "status": "measurement_only",
        "reason": "upstream_defines_backend_selection_but_no_authoritative_rtx3090_performance_floor",
        "metric_minimum_exclusive_gbs": 0,
        "decision_threshold_cpu_to_pcie": 2.0,
        "ratio_absolute_tolerance": 0.002,
        "decision_rule": "hybrid_iff_cpu_moe_gbs_gt_threshold_times_pcie_gather_gbs",
    }:
        raise BandwidthError("bandwidth_acceptance_invalid")
    if value["privacy"] != {"prompts_recorded": False, "responses_recorded": False, "secrets_recorded": False}:
        raise BandwidthError("bandwidth_privacy_invalid")
    return dict(value), _digest(value)


def _kernel(value: Any, *, threshold: float, tolerance: float, code: str) -> dict[str, Any]:
    entry = _exact(
        value,
        {
            "expert_bytes", "synth_experts", "cpu_moe_gbs", "cpu_moe_isa", "isa_sweep",
            "pcie_gather_gbs", "cpu_moe_overlap_gbs", "pcie_gather_overlap_gbs",
            "ratio", "recommended", "note",
        },
        code,
    )
    if not isinstance(entry["expert_bytes"], int) or isinstance(entry["expert_bytes"], bool) or entry["expert_bytes"] <= 0:
        raise BandwidthError(code)
    if not isinstance(entry["synth_experts"], int) or isinstance(entry["synth_experts"], bool) or entry["synth_experts"] <= 0:
        raise BandwidthError(code)
    cpu = _positive(entry["cpu_moe_gbs"], code)
    pcie = _positive(entry["pcie_gather_gbs"], code)
    cpu_overlap = _positive(entry["cpu_moe_overlap_gbs"], code)
    pcie_overlap = _positive(entry["pcie_gather_overlap_gbs"], code)
    ratio = _positive(entry["ratio"], code)
    if abs(ratio - round(cpu / pcie, 3)) > tolerance:
        raise BandwidthError("bandwidth_ratio_inconsistent")
    expected = "hybrid" if cpu > threshold * pcie else "offload"
    if entry["recommended"] != expected:
        raise BandwidthError("bandwidth_decision_inconsistent")
    return {
        "expert_bytes": entry["expert_bytes"],
        "synth_experts": entry["synth_experts"],
        "cpu_moe_gbs": cpu,
        "pcie_gather_gbs": pcie,
        "cpu_moe_overlap_gbs": cpu_overlap,
        "pcie_gather_overlap_gbs": pcie_overlap,
        "ratio": ratio,
        "recommended": expected,
        "hybrid_fetch_fraction": round(pcie_overlap / (pcie_overlap + cpu_overlap), 6),
    }


def parse_profile(path: Path, suite: Mapping[str, Any]) -> dict[str, Any]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise BandwidthError("bandwidth_profile_invalid")
        raw = path.read_bytes()
        if not raw or len(raw) > MAX_JSON_BYTES:
            raise BandwidthError("bandwidth_profile_invalid")
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BandwidthError("bandwidth_profile_invalid") from exc
    profile = _exact(value, {"version", "timestamp", "epoch", "host", "gpu", "cpu", "threshold", "ceilings", "dtypes", "dtype_kernels", "workloads"}, "bandwidth_profile_schema_invalid")
    if profile["version"] != suite["upstream"]["profile_version"] or profile["threshold"] != suite["acceptance"]["decision_threshold_cpu_to_pcie"]:
        raise BandwidthError("bandwidth_profile_identity_invalid")
    if (
        not isinstance(profile["timestamp"], str)
        or not profile["timestamp"]
        or not isinstance(profile["epoch"], int)
        or isinstance(profile["epoch"], bool)
        or profile["epoch"] <= 0
        or not isinstance(profile["host"], str)
        or not profile["host"]
    ):
        raise BandwidthError("bandwidth_profile_identity_invalid")
    cpu = _exact(profile["cpu"], {"physical_cores", "threads_used"}, "bandwidth_cpu_invalid")
    if any(not isinstance(cpu[name], int) or isinstance(cpu[name], bool) or cpu[name] <= 0 for name in cpu):
        raise BandwidthError("bandwidth_cpu_invalid")
    gpu = _exact(profile["gpu"], {"index", "name"}, "bandwidth_gpu_invalid")
    if gpu["index"] != suite["gpu_safety"]["device_index"] or gpu["name"] != suite["gpu_safety"]["expected_name"]:
        raise BandwidthError("bandwidth_gpu_invalid")
    ceilings = _exact(profile["ceilings"], {"cpu_stream_read_gbs", "pcie_linear_h2d_gbs", "pcie_linear_d2h_gbs"}, "bandwidth_ceilings_invalid")
    metrics = {name: _positive(value, "bandwidth_ceilings_invalid") for name, value in ceilings.items()}
    fmt = suite["workload"]["expert_format_profile"]
    if profile["dtypes"] != {} or profile["dtype_kernels"] != {}:
        raise BandwidthError("bandwidth_dtype_identity_invalid")
    workloads = _exact(profile["workloads"], {suite["workload"]["model_name"]}, "bandwidth_workload_identity_invalid")
    workload = _exact(workloads[suite["workload"]["model_name"]], {"model", "kernels", "recommended_moe_backend"}, "bandwidth_workload_identity_invalid")
    if workload["model"] != {
        "name": suite["workload"]["model_name"], "hidden": suite["workload"]["hidden"],
        "inter": suite["workload"]["intermediate"], "experts": suite["workload"]["experts"],
        "top_k": suite["workload"]["top_k"],
    }:
        raise BandwidthError("bandwidth_workload_identity_invalid")
    kernels = _exact(workload["kernels"], {fmt}, "bandwidth_workload_identity_invalid")
    threshold = float(suite["acceptance"]["decision_threshold_cpu_to_pcie"])
    tolerance = float(suite["acceptance"]["ratio_absolute_tolerance"])
    model_result = _kernel(kernels[fmt], threshold=threshold, tolerance=tolerance, code="bandwidth_model_metrics_invalid")
    if workload["recommended_moe_backend"] != {fmt: model_result["recommended"]}:
        raise BandwidthError("bandwidth_decision_inconsistent")
    return {
        "runtime_profile_sha256": hashlib.sha256(raw).hexdigest(),
        "gpu_name": gpu["name"],
        "cpu": dict(cpu),
        "ceilings_gbs": metrics,
        "model_kernel": model_result,
    }


def build_report(
    suite: Mapping[str, Any],
    suite_sha256: str,
    measured: Mapping[str, Any],
    identity: Mapping[str, Any],
    safety_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    required_identity = {"candidate_id", "runtime_id", "manifest_id", "image_digest", "model_manifest_sha256"}
    if set(identity) != required_identity or identity.get("candidate_id") != "freetoken":
        raise BandwidthError("bandwidth_identity_invalid")
    if not all(isinstance(identity.get(name), str) for name in required_identity):
        raise BandwidthError("bandwidth_identity_invalid")
    if not OCI_DIGEST.fullmatch(str(identity["image_digest"])) or not SHA256.fullmatch(str(identity["model_manifest_sha256"])):
        raise BandwidthError("bandwidth_identity_invalid")
    safety_receipt_sha256 = validate_safety_receipt(safety_receipt, suite)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "runner_version": RUNNER_VERSION,
        "candidate_id": "freetoken",
        "runtime_id": identity["runtime_id"],
        "manifest_id": identity["manifest_id"],
        "image_digest": identity["image_digest"],
        "model": {
            "repository": suite["workload"]["model_repository"],
            "revision": suite["workload"]["model_revision"],
            "manifest_sha256": identity["model_manifest_sha256"],
        },
        "source_revision": suite["upstream"]["revision"],
        "bandwidth_profile_sha256": suite_sha256,
        "runtime_profile_sha256": measured["runtime_profile_sha256"],
        "command_argv_sha256": _digest(suite["command"]["argv"]),
        "sample_count": suite["measurement_counts"]["runs"],
        "measurement_counts": suite["measurement_counts"],
        "gpu": {"index": suite["gpu_safety"]["device_index"], "name": measured["gpu_name"]},
        "cpu": measured["cpu"],
        "safety": {
            "receipt_sha256": safety_receipt_sha256,
            "required_passed": True,
            "power_cap_w": suite["gpu_safety"]["power_cap_w"],
            "maximum_temperature_c": suite["gpu_safety"]["maximum_temperature_c"],
            "minimum_safety_sample_count": suite["gpu_safety"]["minimum_safety_sample_count"],
        },
        "metrics": {
            "ceilings_gbs": measured["ceilings_gbs"],
            "model_kernel": measured["model_kernel"],
        },
        "acceptance": {
            "status": "measurement_only",
            "reason": suite["acceptance"]["reason"],
            "performance_floor_applied": False,
            "structural_metric_minimum_exclusive_gbs": suite["acceptance"]["metric_minimum_exclusive_gbs"],
            "decision_consistency": "passed",
        },
        "content_recorded": False,
        "prompts_recorded": False,
        "responses_recorded": False,
        "secrets_recorded": False,
    }
    report["content_sha256"] = _digest(report)
    return report


def validate_report(
    report: Mapping[str, Any],
    suite: Mapping[str, Any],
    suite_sha256: str,
    identity: Mapping[str, Any],
    safety_receipt: Mapping[str, Any],
) -> str:
    """Rebuild one sanitized report and require byte-equivalent identities."""

    value = _exact(
        report,
        {
            "schema_version", "runner_version", "candidate_id", "runtime_id",
            "manifest_id", "image_digest", "model", "source_revision",
            "bandwidth_profile_sha256", "runtime_profile_sha256",
            "command_argv_sha256", "sample_count", "measurement_counts", "gpu", "cpu",
            "safety", "metrics", "acceptance", "content_recorded",
            "prompts_recorded", "responses_recorded", "secrets_recorded",
            "content_sha256",
        },
        "bandwidth_report_invalid",
    )
    metrics = _exact(value["metrics"], {"ceilings_gbs", "model_kernel"}, "bandwidth_report_invalid")
    gpu = _exact(value["gpu"], {"index", "name"}, "bandwidth_report_invalid")
    cpu_identity = _exact(value["cpu"], {"physical_cores", "threads_used"}, "bandwidth_report_invalid")
    if any(
        not isinstance(cpu_identity[name], int)
        or isinstance(cpu_identity[name], bool)
        or cpu_identity[name] <= 0
        for name in cpu_identity
    ):
        raise BandwidthError("bandwidth_report_invalid")
    profile_sha = value.get("runtime_profile_sha256")
    if not isinstance(profile_sha, str) or SHA256.fullmatch(profile_sha) is None:
        raise BandwidthError("bandwidth_report_invalid")
    ceilings = _exact(
        metrics["ceilings_gbs"],
        {"cpu_stream_read_gbs", "pcie_linear_h2d_gbs", "pcie_linear_d2h_gbs"},
        "bandwidth_report_invalid",
    )
    normalized_ceilings = {
        name: _positive(number, "bandwidth_report_invalid")
        for name, number in ceilings.items()
    }
    kernel = _exact(
        metrics["model_kernel"],
        {
            "expert_bytes", "synth_experts", "cpu_moe_gbs", "pcie_gather_gbs",
            "cpu_moe_overlap_gbs", "pcie_gather_overlap_gbs", "ratio",
            "recommended", "hybrid_fetch_fraction",
        },
        "bandwidth_report_invalid",
    )
    if any(
        not isinstance(kernel[name], int) or isinstance(kernel[name], bool) or kernel[name] <= 0
        for name in ("expert_bytes", "synth_experts")
    ):
        raise BandwidthError("bandwidth_report_invalid")
    cpu = _positive(kernel["cpu_moe_gbs"], "bandwidth_report_invalid")
    pcie = _positive(kernel["pcie_gather_gbs"], "bandwidth_report_invalid")
    cpu_overlap = _positive(kernel["cpu_moe_overlap_gbs"], "bandwidth_report_invalid")
    pcie_overlap = _positive(kernel["pcie_gather_overlap_gbs"], "bandwidth_report_invalid")
    ratio = _positive(kernel["ratio"], "bandwidth_report_invalid")
    threshold = float(suite["acceptance"]["decision_threshold_cpu_to_pcie"])
    tolerance = float(suite["acceptance"]["ratio_absolute_tolerance"])
    expected_backend = "hybrid" if cpu > threshold * pcie else "offload"
    expected_fraction = round(pcie_overlap / (pcie_overlap + cpu_overlap), 6)
    if (
        abs(ratio - round(cpu / pcie, 3)) > tolerance
        or kernel["recommended"] != expected_backend
        or not isinstance(kernel["hybrid_fetch_fraction"], (int, float))
        or isinstance(kernel["hybrid_fetch_fraction"], bool)
        or abs(float(kernel["hybrid_fetch_fraction"]) - expected_fraction) > 0.0000005
    ):
        raise BandwidthError("bandwidth_report_invalid")
    normalized_kernel = {
        "expert_bytes": kernel["expert_bytes"],
        "synth_experts": kernel["synth_experts"],
        "cpu_moe_gbs": cpu,
        "pcie_gather_gbs": pcie,
        "cpu_moe_overlap_gbs": cpu_overlap,
        "pcie_gather_overlap_gbs": pcie_overlap,
        "ratio": ratio,
        "recommended": expected_backend,
        "hybrid_fetch_fraction": expected_fraction,
    }
    measured = {
        "runtime_profile_sha256": profile_sha,
        "gpu_name": gpu["name"],
        "cpu": dict(cpu_identity),
        "ceilings_gbs": normalized_ceilings,
        "model_kernel": normalized_kernel,
    }
    try:
        expected = build_report(suite, suite_sha256, measured, identity, safety_receipt)
    except (KeyError, TypeError, BandwidthError) as exc:
        raise BandwidthError("bandwidth_report_invalid") from exc
    if dict(value) != expected:
        raise BandwidthError("bandwidth_report_invalid")
    return str(value["content_sha256"])


def validate_safety_receipt(receipt: Mapping[str, Any], suite: Mapping[str, Any]) -> str:
    value = _exact(receipt, {"schema", "candidate_id", "passed", "sample_count", "extrema", "fault_counts"}, "bandwidth_safety_receipt_invalid")
    if (
        value["schema"] != "ai-server.acceleration-safety.v1"
        or value["candidate_id"] != "freetoken-freetoken"
        or value["passed"] is not True
        or not isinstance(value["sample_count"], int)
        or isinstance(value["sample_count"], bool)
        or value["sample_count"] < suite["gpu_safety"]["minimum_safety_sample_count"]
    ):
        raise BandwidthError("bandwidth_safety_receipt_invalid")
    extrema = _exact(value["extrema"], {"maximum_temperature_c", "maximum_power_draw_w", "minimum_vram_free_mib", "minimum_ram_available_mib", "maximum_swap_in_delta", "maximum_swap_out_delta"}, "bandwidth_safety_receipt_invalid")
    faults = _exact(value["fault_counts"], {"xid", "oom", "restart_delta", "backend_unavailable", "incomplete_stream"}, "bandwidth_safety_receipt_invalid")
    for field in ("maximum_temperature_c", "maximum_power_draw_w", "minimum_vram_free_mib", "minimum_ram_available_mib"):
        if not isinstance(extrema[field], (int, float)) or isinstance(extrema[field], bool) or not math.isfinite(float(extrema[field])):
            raise BandwidthError("bandwidth_safety_receipt_invalid")
    if (
        float(extrema["maximum_temperature_c"]) > suite["gpu_safety"]["maximum_temperature_c"]
        or float(extrema["maximum_power_draw_w"]) > suite["gpu_safety"]["power_cap_w"]
        or float(extrema["minimum_vram_free_mib"]) < suite["gpu_safety"]["minimum_vram_free_mib"]
        or float(extrema["minimum_ram_available_mib"]) < suite["gpu_safety"]["minimum_ram_available_mib"]
        or extrema["maximum_swap_in_delta"] != 0
        or extrema["maximum_swap_out_delta"] != 0
        or any(not isinstance(item, int) or isinstance(item, bool) or item != 0 for item in faults.values())
    ):
        raise BandwidthError("bandwidth_safety_receipt_failed")
    return hashlib.sha256(_canonical(value) + b"\n").hexdigest()


def run_fixed_command(
    suite: Mapping[str, Any],
    *,
    command_runner: Callable[[Sequence[str], float, Mapping[str, str]], int] | None = None,
) -> None:
    argv = tuple(str(part) for part in suite["command"]["argv"])
    timeout = float(suite["command"]["timeout_seconds"])
    env = dict(os.environ)
    env["FREETOKEN_BENCH_PROGRESS"] = "1"
    if command_runner is None:
        try:
            result = subprocess.run(argv, check=False, capture_output=True, timeout=timeout, env=env, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BandwidthError("bandwidth_command_failed") from exc
        returncode = result.returncode
    else:
        returncode = command_runner(argv, timeout, env)
    if returncode != 0:
        raise BandwidthError("bandwidth_command_failed")


def execute(
    suite_path: Path,
    profile_path: Path,
    identity: Mapping[str, Any],
    safety_receipt: Mapping[str, Any],
    *,
    command_runner: Callable[[Sequence[str], float, Mapping[str, str]], int] | None = None,
) -> dict[str, Any]:
    suite, suite_sha256 = load_suite(suite_path)
    if profile_path.exists() or profile_path.is_symlink():
        raise BandwidthError("bandwidth_profile_preexists")
    run_fixed_command(suite, command_runner=command_runner)
    measured = parse_profile(profile_path, suite)
    return build_report(suite, suite_sha256, measured, identity, safety_receipt)
