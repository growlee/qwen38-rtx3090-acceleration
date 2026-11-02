#!/usr/bin/env python3
"""Sealed, single-arm orchestration for acceleration candidate measurements.

The live operation supplies only an allowlisted candidate and arm.  Runtime,
endpoint, model, suite, evaluator and mode are obtained from the reviewed
candidate contract; they are never caller overrides.  Component execution is
dependency injected so tests can prove ordering and failure behaviour without
touching a live host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Protocol

try:
    from .candidate_contract import validate_candidate_contract
except ImportError:  # pragma: no cover - direct execution from experiments/
    from candidate_contract import validate_candidate_contract  # type: ignore[no-redef]

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
import candidate_benchmark_runner as benchmark_runner  # noqa: E402
import candidate_openai_smoke as openai_smoke  # noqa: E402
import freetoken_bandwidth_runner as bandwidth_runner  # noqa: E402
import candidate_quality_evaluator as quality_evaluator  # noqa: E402
import safety_sampler  # noqa: E402


SCHEMA = "ai-server-acceleration-arm-result/v1"
EVIDENCE_SCHEMA = "ai-server-acceleration-arm-evidence/v1"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_COMPONENT_SECONDS = 6_600.0
REQUEST_TIMEOUT_SECONDS = 300.0
ROOT = Path(__file__).resolve().parent
AI_SERVER_ROOT = ROOT.parent
BENCHMARK_SUITE = AI_SERVER_ROOT / "benchmarks" / "suites" / "acceleration-benchmark-v1.json"
QUALITY_SUITE = AI_SERVER_ROOT / "benchmarks" / "suites" / "acceleration-quality-v1.json"
BANDWIDTH_SUITE = AI_SERVER_ROOT / "benchmarks" / "suites" / "acceleration-freetoken-bandwidth-v1.json"
OBSERVATION_DEADLINE_SECONDS = 1.0


class HarnessError(RuntimeError):
    """Stable content-free harness failure."""


def _reject(code: str) -> NoReturn:
    raise HarnessError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise HarnessError("suite_unreadable") from exc


@dataclass(frozen=True)
class ArmSpec:
    candidate_id: str
    arm: str
    mode: str
    component_type: str
    component_arm: str
    quality_arm: str


SPECS: dict[tuple[str, str], ArmSpec] = {
    ("qwen38-vllm", "mtp"): ArmSpec("qwen38-vllm", "mtp", "mtp", "syv", "mtp", "mtp"),
    ("qwen38-vllm", "dflash2"): ArmSpec("qwen38-vllm", "dflash2", "dflash2", "syv", "dflash2", "dflash2"),
    ("freetoken", "freetoken"): ArmSpec("freetoken", "freetoken", "serve", "freetoken", "freetoken", "plain"),
    ("llama-dflash2-official", "plain"): ArmSpec("llama-dflash2-official", "plain", "plain", "syv", "mtp", "plain"),
    ("llama-dflash2-official", "mtp"): ArmSpec("llama-dflash2-official", "mtp", "mtp", "syv", "mtp", "mtp"),
    ("llama-dflash2-official", "dflash2-n4"): ArmSpec("llama-dflash2-official", "dflash2-n4", "dflash2-n4", "syv", "dflash2", "dflash2-n4"),
    ("llama-dflash2-official", "dflash2-n7"): ArmSpec("llama-dflash2-official", "dflash2-n7", "dflash2-n7", "syv", "dflash2", "dflash2-n7"),
}


class ComponentDependencies(Protocol):
    """Narrow boundary implemented by the sealed package adapter."""

    def smoke(self, context: "ArmContext", timeout: float) -> Mapping[str, Any]: ...
    def safety(self, context: "ArmContext", timeout: float) -> Mapping[str, Any]: ...
    def benchmark(
        self, context: "ArmContext", safety_receipt: Mapping[str, Any], timeout: float
    ) -> Mapping[str, Any]: ...
    def quality(
        self, context: "ArmContext", benchmark_report: Mapping[str, Any], timeout: float
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ArmContext:
    operation_id: str
    spec: ArmSpec
    contract: Mapping[str, Any]
    runtime_id: str
    manifest_id: str
    benchmark_suite_sha256: str
    quality_suite_sha256: str
    bandwidth_profile_sha256: str | None = None
    bandwidth_report_sha256: str | None = None


def _runtime_id(contract: Mapping[str, Any]) -> str:
    image = contract.get("image")
    if not isinstance(image, Mapping):
        _reject("runtime_pin_invalid")
    reference = image.get("reference")
    digest = image.get("digest")
    if not isinstance(reference, str) or "@sha256:" not in reference:
        _reject("runtime_pin_invalid")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        _reject("runtime_pin_invalid")
    if not reference.endswith("@" + digest):
        _reject("runtime_pin_mismatch")
    # Component config identifiers deliberately exclude registry paths while
    # retaining the complete immutable runtime coordinate.
    return "runtime@" + digest


def build_context(
    candidate_id: str,
    arm: str,
    operation_id: str,
    *,
    contract_loader: Callable[..., Mapping[str, Any]] = validate_candidate_contract,
) -> ArmContext:
    if SAFE_ID.fullmatch(operation_id) is None:
        _reject("operation_id_invalid")
    spec = SPECS.get((candidate_id, arm))
    if spec is None:
        _reject("candidate_arm_not_allowlisted")
    try:
        contract = contract_loader(candidate_id, spec.mode, root=ROOT, require_live=True)
    except Exception as exc:
        raise HarnessError("candidate_contract_invalid") from exc
    if not isinstance(contract, Mapping) or contract.get("candidate_id") != candidate_id:
        _reject("candidate_contract_invalid")
    modes = contract.get("modes")
    benchmark = contract.get("benchmark")
    if not isinstance(modes, Mapping) or spec.mode not in modes or not isinstance(benchmark, Mapping):
        _reject("candidate_contract_invalid")
    benchmark_suite_sha256 = _file_sha256(BENCHMARK_SUITE)
    quality_value = json.loads(QUALITY_SUITE.read_text(encoding="utf-8"))
    quality_suite_sha256 = _digest(quality_value)
    if benchmark.get("suite") != "acceleration-quality-v1" or benchmark.get("suite_sha256") != quality_suite_sha256:
        _reject("quality_suite_pin_mismatch")
    bandwidth_profile_sha256: str | None = None
    if candidate_id == "freetoken":
        try:
            _bandwidth_suite, bandwidth_profile_sha256 = bandwidth_runner.load_suite(BANDWIDTH_SUITE)
        except bandwidth_runner.BandwidthError as exc:
            raise HarnessError("bandwidth_profile_invalid") from exc
        if benchmark.get("bandwidth_profile_sha256") != bandwidth_profile_sha256:
            _reject("bandwidth_profile_pin_mismatch")
    return ArmContext(
        operation_id=operation_id,
        spec=spec,
        contract=contract,
        runtime_id=_runtime_id(contract),
        manifest_id=_digest(contract),
        benchmark_suite_sha256=benchmark_suite_sha256,
        quality_suite_sha256=quality_suite_sha256,
        bandwidth_profile_sha256=bandwidth_profile_sha256,
    )


def _content_hash(receipt: Mapping[str, Any], code: str) -> str:
    value = receipt.get("content_sha256")
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        _reject(code)
    core = dict(receipt)
    core.pop("content_sha256")
    if _digest(core) != value:
        _reject(code)
    return value


def _validate_smoke(value: Mapping[str, Any], context: ArmContext) -> str:
    if not isinstance(value, Mapping):
        _reject("smoke_receipt_invalid")
    if (
        value.get("candidate_id") != context.spec.candidate_id
        or value.get("candidate_type") != context.spec.component_type
        or value.get("runtime_id") != context.runtime_id
        or value.get("manifest_id") != context.manifest_id
        or value.get("status") != "pass"
        or value.get("suite_sha256") != openai_smoke._SUITE_SHA256
        or value.get("content_recorded") is not False
        or value.get("secrets_recorded") is not False
        or value.get("token_counts_recorded") is not False
    ):
        _reject("smoke_receipt_invalid")
    return _content_hash(value, "smoke_receipt_invalid")


def _validate_safety(value: Mapping[str, Any], context: ArmContext) -> str:
    if not isinstance(value, Mapping):
        _reject("safety_receipt_invalid")
    if value.get("schema") != "ai-server.acceleration-safety.v1" or value.get("passed") is not True:
        _reject("safety_receipt_invalid")
    if value.get("candidate_id") != f"{context.spec.candidate_id}-{context.spec.arm}":
        _reject("safety_receipt_invalid")
    if set(value) != {"schema", "candidate_id", "passed", "sample_count", "extrema", "fault_counts", "identity"}:
        _reject("safety_receipt_invalid")
    return hashlib.sha256(_canonical(value) + b"\n").hexdigest()


def _validate_benchmark(value: Mapping[str, Any], context: ArmContext, safety_sha256: str) -> str:
    if not isinstance(value, Mapping):
        _reject("benchmark_report_invalid")
    if (
        value.get("candidate_id") != context.spec.candidate_id
        or value.get("candidate_type") != context.spec.component_type
        or value.get("arm") != context.spec.component_arm
        or value.get("runtime_id") != context.runtime_id
        or value.get("manifest_id") != context.manifest_id
        or value.get("suite_sha256") != context.benchmark_suite_sha256
        or value.get("safety_receipt_sha256") != safety_sha256
        or value.get("status") != "pass"
        or value.get("content_recorded") is not False
        or value.get("secrets_recorded") is not False
        or value.get("token_counts_recorded") is not False
        or value.get("endpoint_recorded") is not False
    ):
        _reject("benchmark_report_invalid")
    if context.spec.candidate_id == "freetoken":
        if value.get("bandwidth_report_sha256") != context.bandwidth_report_sha256:
            _reject("benchmark_report_invalid")
    elif "bandwidth_report_sha256" in value:
        _reject("benchmark_report_invalid")
    return _content_hash(value, "benchmark_report_invalid")


def _load_prestart_bandwidth_report(context: ArmContext, output_dir: Path) -> str:
    if context.spec.candidate_id != "freetoken":
        return ""
    operation_dir = output_dir.parent.parent
    report_path = operation_dir / "freetoken-bandwidth-report.json"
    safety_path = operation_dir / "freetoken-bandwidth-safety.json"
    try:
        for path in (report_path, safety_path):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                _reject("bandwidth_report_invalid")
            if os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o400:
                _reject("bandwidth_report_invalid")
        report_raw = report_path.read_bytes()
        safety_raw = safety_path.read_bytes()
        if not report_raw or len(report_raw) > 1_048_576 or not safety_raw or len(safety_raw) > 1_048_576:
            _reject("bandwidth_report_invalid")
        report = json.loads(report_raw.decode("utf-8"))
        safety_receipt = json.loads(safety_raw.decode("utf-8"))
        suite, suite_sha = bandwidth_runner.load_suite(BANDWIDTH_SUITE)
        image = context.contract["image"]
        target = next(model for model in context.contract["models"] if model["role"] == "target")
        identity = {
            "candidate_id": "freetoken",
            "runtime_id": context.runtime_id,
            "manifest_id": context.manifest_id,
            "image_digest": image["digest"],
            "model_manifest_sha256": target["manifest_sha256"],
        }
        bandwidth_runner.validate_report(report, suite, suite_sha, identity, safety_receipt)
        return hashlib.sha256(report_raw).hexdigest()
    except HarnessError:
        raise
    except (OSError, UnicodeError, KeyError, StopIteration, TypeError, json.JSONDecodeError, bandwidth_runner.BandwidthError) as exc:
        raise HarnessError("bandwidth_report_invalid") from exc


def _validate_quality(value: Mapping[str, Any], context: ArmContext, report_sha256: str) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        _reject("quality_receipt_invalid")
    status = value.get("quality_status")
    if (
        value.get("arm") != context.spec.quality_arm
        or value.get("suite_canonical_sha256") != context.quality_suite_sha256
        or value.get("benchmark_report_sha256") != report_sha256
        or status not in {"pass", "fail", "not_evaluated"}
    ):
        _reject("quality_receipt_invalid")
    privacy = value.get("privacy")
    if not isinstance(privacy, Mapping) or privacy.get("prompts_recorded") is not False or privacy.get("responses_recorded") is not False:
        _reject("quality_receipt_invalid")
    return _content_hash(value, "quality_receipt_invalid"), str(status)


def _write_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        _reject("evidence_exists")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        _reject("output_dir_invalid")
    payload = _canonical(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("evidence_write_failed")
            view = view[written:]
        os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, 0o400)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def execute(
    candidate_id: str,
    arm: str,
    operation_id: str,
    output_dir: Path,
    dependencies: ComponentDependencies,
    *,
    contract_loader: Callable[..., Mapping[str, Any]] = validate_candidate_contract,
) -> dict[str, Any]:
    """Run exactly one arm once.  A component failure stops the sequence."""

    context = build_context(candidate_id, arm, operation_id, contract_loader=contract_loader)
    if output_dir.is_symlink() or not output_dir.is_dir():
        _reject("output_dir_invalid")
    if candidate_id == "freetoken":
        context = replace(
            context,
            bandwidth_report_sha256=_load_prestart_bandwidth_report(context, output_dir),
        )
    smoke_receipt = dependencies.smoke(context, MAX_COMPONENT_SECONDS)
    smoke_sha256 = _validate_smoke(smoke_receipt, context)
    # Production dependencies start their concurrent monitor here and fill the
    # aggregate mapping when benchmark() stops it. Test dependencies may return
    # an already complete mapping.
    safety_receipt = dependencies.safety(context, MAX_COMPONENT_SECONDS)
    benchmark_report = dependencies.benchmark(context, safety_receipt, MAX_COMPONENT_SECONDS)
    quality_receipt = dependencies.quality(context, benchmark_report, MAX_COMPONENT_SECONDS)
    safety_sha256 = _validate_safety(safety_receipt, context)
    benchmark_sha256 = _validate_benchmark(benchmark_report, context, safety_sha256)
    quality_sha256, quality_status = _validate_quality(quality_receipt, context, benchmark_sha256)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "operation_id": operation_id,
        "candidate_id": candidate_id,
        "arm": arm,
        "runtime_id_sha256": hashlib.sha256(context.runtime_id.encode("utf-8")).hexdigest(),
        "manifest_id": context.manifest_id,
        "benchmark_suite_sha256": context.benchmark_suite_sha256,
        "quality_suite_sha256": context.quality_suite_sha256,
        "bandwidth_profile_sha256": context.bandwidth_profile_sha256,
        "bandwidth_report_sha256": context.bandwidth_report_sha256,
        "smoke_receipt_sha256": smoke_sha256,
        "safety_receipt_sha256": safety_sha256,
        "benchmark_report_sha256": benchmark_sha256,
        "quality_receipt_sha256": quality_sha256,
        "quality_status": quality_status,
        "content_recorded": False,
        "secrets_recorded": False,
        "token_counts_recorded": False,
    }
    evidence["content_sha256"] = _digest(evidence)
    _write_new(output_dir / "smoke-receipt.json", smoke_receipt)
    _write_new(output_dir / "safety-receipt.json", safety_receipt)
    _write_new(output_dir / "benchmark-report.json", benchmark_report)
    _write_new(output_dir / "quality-receipt.json", quality_receipt)
    _write_new(output_dir / "arm-evidence.json", evidence)
    return {
        "schema_version": SCHEMA,
        "candidate_id": candidate_id,
        "arm": arm,
        "passed": quality_status == "pass",
        "report_sha256": evidence["content_sha256"],
        "quality_status": {"pass": "passed", "fail": "failed", "not_evaluated": "not_evaluated"}[quality_status],
    }


class _HostProbe:
    """Fixed-command, bounded Server A observation for one allowlisted container."""

    def __init__(self, container: str, *, pinned_container_id: str | None = None) -> None:
        self.container = container
        if pinned_container_id is None:
            pinned_container_id = self._run((
                "/usr/bin/docker", "inspect", container, "--format", "{{.Id}}",
            ), timeout=2.0).strip()
        if re.fullmatch(r"[0-9a-f]{64}", pinned_container_id) is None:
            raise safety_sampler.SafetyError("safety_container_id_invalid")
        self.container_id = pinned_container_id
        self._require_compute_pid = False
        self._swap: tuple[int, int] | None = None
        self._started = datetime.now(timezone.utc).isoformat()

    def require_compute_pid(self) -> None:
        """Tighten identity after a successful readiness/smoke request."""

        self._require_compute_pid = True

    @staticmethod
    def _boot_id() -> str:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip().lower()
        except OSError as exc:
            raise safety_sampler.SafetyError("boot_id_unavailable") from exc

    @staticmethod
    def _gpu_compute_pids(raw: str) -> tuple[int, ...]:
        pids: list[int] = []
        for row in raw.splitlines():
            value = row.strip()
            if not value:
                continue
            if not value.isascii() or not value.isdigit():
                raise safety_sampler.SafetyError("gpu_compute_pid_shape_invalid")
            pid = int(value)
            if pid <= 0 or pid in pids:
                raise safety_sampler.SafetyError("gpu_compute_pid_shape_invalid")
            pids.append(pid)
        return tuple(pids)

    @staticmethod
    def _pid_owned_by_container(pid: int, container_id: str) -> bool:
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
        except OSError as exc:
            raise safety_sampler.SafetyError("gpu_compute_pid_cgroup_unavailable") from exc
        return re.search(rf"(?<![0-9a-f]){re.escape(container_id)}(?![0-9a-f])", cgroup.lower()) is not None

    def terminate_candidate(self) -> None:
        self._run(("/usr/bin/docker", "stop", "--time", "10", self.container_id), timeout=15.0)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise safety_sampler.SafetyError("safety_observation_timeout")
        return remaining

    @staticmethod
    def _run(
        argv: tuple[str, ...], timeout: float = 10.0, *, include_stderr: bool = False,
        deadline: float | None = None,
    ) -> str:
        effective_timeout = timeout
        if deadline is not None:
            effective_timeout = min(timeout, _HostProbe._remaining(deadline))
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, check=False,
                timeout=effective_timeout, shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            code = "safety_observation_timeout" if deadline is not None else "safety_probe_command_failed"
            raise safety_sampler.SafetyError(code) from exc
        except OSError as exc:
            raise safety_sampler.SafetyError("safety_probe_command_failed") from exc
        if deadline is not None:
            _HostProbe._remaining(deadline)
        output = result.stdout + result.stderr if include_stderr else result.stdout
        if result.returncode != 0 or len(output) > 1_048_576:
            raise safety_sampler.SafetyError("safety_probe_command_failed")
        return output

    @staticmethod
    def _proc_values() -> tuple[float, int, int, int]:
        try:
            mem = Path("/proc/meminfo").read_text(encoding="ascii")
            vm = Path("/proc/vmstat").read_text(encoding="ascii")
            available = int(re.search(r"(?m)^MemAvailable:\s+(\d+)\s+kB$", mem).group(1)) / 1024  # type: ignore[union-attr]
            pin = int(re.search(r"(?m)^pswpin\s+(\d+)$", vm).group(1))  # type: ignore[union-attr]
            pout = int(re.search(r"(?m)^pswpout\s+(\d+)$", vm).group(1))  # type: ignore[union-attr]
            major = int(re.search(r"(?m)^pgmajfault\s+(\d+)$", vm).group(1))  # type: ignore[union-attr]
        except (OSError, AttributeError, ValueError) as exc:
            raise safety_sampler.SafetyError("safety_proc_invalid") from exc
        return available, pin, pout, major

    def observe(self) -> safety_sampler.SafetyObservation:
        deadline = time.monotonic() + OBSERVATION_DEADLINE_SECONDS
        gpu = self._run((
            "/usr/bin/nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,power.limit,memory.free",
            "--format=csv,noheader,nounits",
        ), deadline=deadline)
        rows = [row for row in gpu.splitlines() if row.strip()]
        if len(rows) != 1:
            raise safety_sampler.SafetyError("safety_gpu_shape_invalid")
        try:
            temperature, power, cap, free = (float(item.strip()) for item in rows[0].split(","))
        except ValueError as exc:
            raise safety_sampler.SafetyError("safety_gpu_shape_invalid") from exc
        inspect = self._run((
            "/usr/bin/docker", "inspect", self.container,
            "--format", "{{json .Id}}|{{json .State}}|{{json .RestartCount}}",
        ), deadline=deadline)
        try:
            id_raw, state_raw, restart_raw = inspect.strip().split("|", 2)
            container_id = json.loads(id_raw)
            state = json.loads(state_raw)
            restart = json.loads(restart_raw)
            if not isinstance(container_id, str) or re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
                raise ValueError
            if container_id != self.container_id:
                raise safety_sampler.SafetyError("safety_container_identity_changed")
            if not isinstance(state, dict) or isinstance(restart, bool) or not isinstance(restart, int):
                raise ValueError
            unavailable = int(state.get("Running") is not True or state.get("Health", {}).get("Status") != "healthy")
        except safety_sampler.SafetyError:
            raise
        except (ValueError, TypeError, json.JSONDecodeError, AttributeError) as exc:
            raise safety_sampler.SafetyError("safety_container_shape_invalid") from exc
        compute_raw = self._run((
            "/usr/bin/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ), deadline=deadline)
        compute_pids = self._gpu_compute_pids(compute_raw)
        compute_owner_valid = (
            (bool(compute_pids) or not self._require_compute_pid)
            and all(self._pid_owned_by_container(pid, self.container_id) for pid in compute_pids)
        )
        self._remaining(deadline)
        available, pin, pout, major = self._proc_values()
        self._remaining(deadline)
        previous = self._swap
        previous_major = getattr(self, "_major_faults", None)
        self._swap = (pin, pout)
        self._major_faults = major
        swap_in = 0 if previous is None else pin - previous[0]
        swap_out = 0 if previous is None else pout - previous[1]
        if swap_in < 0 or swap_out < 0:
            raise safety_sampler.SafetyError("safety_swap_counter_invalid")
        major_delta = 0 if previous_major is None else major - previous_major
        if major_delta < 0:
            raise safety_sampler.SafetyError("safety_major_fault_counter_invalid")
        kernel = self._run((
            "/usr/bin/journalctl", "-k", "--since", self._started,
            "--no-pager", "--output=cat",
        ), timeout=15.0, deadline=deadline)
        container_logs = self._run((
            "/usr/bin/docker", "logs", "--since", self._started, self.container_id,
        ), timeout=15.0, include_stderr=True, deadline=deadline)
        lower = kernel.lower()
        workload_lower = container_logs.lower()
        boot_id = self._boot_id()
        self._remaining(deadline)
        return safety_sampler.SafetyObservation(
            monotonic_ns=time.monotonic_ns(),
            gpu_temperature_c=temperature,
            gpu_power_draw_w=power,
            power_cap_w=cap,
            vram_free_mib=free,
            ram_available_mib=available,
            swap_in_delta=swap_in,
            swap_out_delta=swap_out,
            nvidia_xid="nvrm: xid" in lower,
            host_oom="out of memory" in lower or "oom-killer" in lower,
            container_oom_killed=state.get("OOMKilled") is True,
            restart_count=restart,
            backend_unavailable_count=unavailable,
            incomplete_stream_count=0,
            boot_id=boot_id,
            gpu_compute_pid_count=len(compute_pids),
            gpu_compute_pid_owner_valid=compute_owner_valid,
            major_fault_delta=major_delta,
            cuda_oom=(
                "cuda out of memory" in lower
                or "cuda_error_out_of_memory" in lower
                or "cuda out of memory" in workload_lower
                or "cuda_error_out_of_memory" in workload_lower
            ),
        )


class NativeDependencies:
    """Self-contained live adapter using only reviewed local contracts."""

    _MODELS = {
        "qwen38-vllm": "dbirks/Qwen3.8-27B-W4A16-AutoRound",
        "freetoken": "gpt-oss-20b",
        "llama-dflash2-official": "Qwen3.8-27B-Q4_K_M.gguf",
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._monitor: safety_sampler.ConcurrentSafetyMonitor | None = None
        self._probe: _HostProbe | None = None
        self._safety_receipt: dict[str, Any] | None = None

    @staticmethod
    def _mode(context: ArmContext) -> Mapping[str, Any]:
        return context.contract["modes"][context.spec.mode]

    def _smoke_config(self, context: ArmContext) -> dict[str, Any]:
        mode = self._mode(context)
        return {
            "schema_version": 1,
            "candidate_id": context.spec.candidate_id,
            "candidate_type": context.spec.component_type,
            "runtime_id": context.runtime_id,
            "manifest_id": context.manifest_id,
            "base_url": mode["endpoint"],
            "model_id": self._MODELS[context.spec.candidate_id],
            "auth_mode": mode["auth_mode"],
            "token_file": mode["auth_file"],
            "connect_timeout_seconds": 5,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "cancel_after_deltas": 2,
            "capabilities": {
                "responses_api": False,
                "constrained_response_format": False,
                "tools": "tools" in mode["capabilities"],
                "vision": "vision" in mode["capabilities"],
            },
        }

    @staticmethod
    def _temporary_json(value: Mapping[str, Any]):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
        try:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            return Path(handle.name)
        finally:
            handle.close()

    def smoke(self, context: ArmContext, timeout: float) -> Mapping[str, Any]:
        self._start_monitor(context)
        path = self._temporary_json(self._smoke_config(context))
        try:
            def workload() -> Mapping[str, Any]:
                receipt = openai_smoke.execute(path)
                if receipt.get("status") != "pass":
                    raise safety_sampler.SafetyError("backend_unavailable_detected")
                return receipt
            receipt = self._guard(workload, timeout)
            probe = self._probe
            if probe is None:
                _reject("safety_monitor_missing")
            probe.require_compute_pid()
            return receipt
        except BaseException:
            self._stop_failed_monitor()
            raise
        finally:
            path.unlink(missing_ok=True)

    def _limits(self, context: ArmContext) -> safety_sampler.SafetyLimits:
        freetoken = context.spec.candidate_id == "freetoken"
        return safety_sampler.SafetyLimits(
            candidate_id=f"{context.spec.candidate_id}-{context.spec.arm}",
            maximum_temperature_c=85.0,
            maximum_power_draw_w=300.0,
            expected_power_cap_w=300.0,
            minimum_vram_free_mib=1536.0,
            minimum_ram_available_mib=4096.0 if freetoken else 2048.0,
            maximum_swap_io_delta=0,
            minimum_sample_count=2,
            # FreeToken's unspecified "sustained thrash" is deterministic at
            # three consecutive 1-second samples. Other arms retain the plan's
            # explicit "more than 9 seconds" threshold.
            maximum_consecutive_swap_samples=2 if freetoken else 9,
            require_runtime_identity=True,
        )

    def _start_monitor(self, context: ArmContext) -> None:
        if self._monitor is not None or self._probe is not None:
            _reject("safety_monitor_state_invalid")
        mode = self._mode(context)
        probe = _HostProbe(mode["container_name"])
        monitor = safety_sampler.ConcurrentSafetyMonitor(self._limits(context), probe, interval_seconds=1.0)
        monitor.start()
        self._probe = probe
        self._monitor = monitor

    def _guard(self, workload: Callable[[], Any], timeout: float) -> Any:
        monitor = self._monitor
        probe = self._probe
        if monitor is None or probe is None:
            _reject("safety_monitor_missing")
        return monitor.run_guarded(workload, probe.terminate_candidate, timeout_seconds=timeout)

    def _stop_failed_monitor(self) -> None:
        monitor = self._monitor
        self._monitor = None
        self._probe = None
        if monitor is not None:
            try:
                monitor.stop()
            except safety_sampler.SafetyError:
                pass

    def safety(self, context: ArmContext, timeout: float) -> Mapping[str, Any]:
        if self._monitor is None or self._safety_receipt is not None:
            _reject("safety_monitor_missing")
        self._safety_receipt = {}
        return self._safety_receipt

    def benchmark(self, context: ArmContext, safety_receipt: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        monitor = self._monitor
        if monitor is None or not isinstance(safety_receipt, dict):
            _reject("safety_monitor_missing")
        mode = self._mode(context)
        config = benchmark_runner.Config(
            candidate_id=context.spec.candidate_id,
            candidate_type=context.spec.component_type,
            arm=context.spec.component_arm,
            runtime_id=context.runtime_id,
            manifest_id=context.manifest_id,
            base_url=mode["endpoint"],
            model_id=self._MODELS[context.spec.candidate_id],
            auth_mode=mode["auth_mode"],
            token_file=None if mode["auth_file"] is None else Path(mode["auth_file"]),
            connect_timeout_seconds=5.0,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            stability_seconds=1800,
            cold_load_seconds=0.0,
            safety_receipt_sha256="0" * 64,
        )
        suite, suite_sha = benchmark_runner.load_suite(BENCHMARK_SUITE)
        token = openai_smoke._read_secret(config.token_file) if config.token_file is not None else None
        started = datetime.now(timezone.utc)
        try:
            def workload():
                value = benchmark_runner.run_benchmark(config, suite, token)
                _metrics, _error_code, _http_status = value
                if _error_code is not None or (_http_status is not None and _http_status >= 500):
                    code = (
                        "incomplete_stream_detected"
                        if "stream" in str(_error_code)
                        else "backend_unavailable_detected"
                    )
                    raise safety_sampler.SafetyError(code)
                return value
            metrics, error_code, http_status = self._guard(workload, timeout)
        except BaseException:
            self._stop_failed_monitor()
            raise
        safety_sha = "0" * 64
        config_identity = {
            "candidate_id": config.candidate_id, "candidate_type": config.candidate_type,
            "arm": config.arm, "runtime_id": config.runtime_id, "manifest_id": config.manifest_id,
            "suite_sha256": suite_sha, "safety_receipt_sha256": safety_sha,
        }
        report: dict[str, Any] = {
            "schema_version": benchmark_runner.REPORT_VERSION,
            "runner_version": benchmark_runner.RUNNER_VERSION,
            "candidate_id": config.candidate_id,
            "candidate_type": config.candidate_type,
            "arm": config.arm,
            "runtime_id": config.runtime_id,
            "manifest_id": config.manifest_id,
            "config_sha256": _digest(config_identity),
            "suite_sha256": suite_sha,
            "safety_receipt_sha256": safety_sha,
            "started_at": started.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if error_code is None else "fail",
            "error_code": error_code,
            "http_status": http_status,
            "metrics": metrics,
            "content_recorded": False,
            "secrets_recorded": False,
            "token_counts_recorded": False,
            "endpoint_recorded": False,
        }
        if context.bandwidth_report_sha256 is not None:
            report["bandwidth_report_sha256"] = context.bandwidth_report_sha256
        return report

    def _finish_monitor(
        self,
        safety_receipt: dict[str, Any],
        benchmark_report: Mapping[str, Any],
        quality_receipt: dict[str, Any],
    ) -> str:
        monitor = self._monitor
        probe = self._probe
        if (
            monitor is None
            or probe is None
            or not isinstance(benchmark_report, dict)
            or not isinstance(quality_receipt, dict)
        ):
            _reject("safety_monitor_missing")
        try:
            result = monitor.stop().as_dict()
        except safety_sampler.SafetyError:
            try:
                probe.terminate_candidate()
            finally:
                self._monitor = None
                self._probe = None
            raise
        self._monitor = None
        self._probe = None
        safety_receipt.clear()
        safety_receipt.update(result)
        safety_sha = hashlib.sha256(_canonical(safety_receipt) + b"\n").hexdigest()
        benchmark_report["safety_receipt_sha256"] = safety_sha
        config_identity = {
            key: benchmark_report[key]
            for key in ("candidate_id", "candidate_type", "arm", "runtime_id", "manifest_id", "suite_sha256", "safety_receipt_sha256")
        }
        benchmark_report["config_sha256"] = _digest(config_identity)
        benchmark_report.pop("content_sha256", None)
        benchmark_report["content_sha256"] = _digest(benchmark_report)
        quality_receipt["benchmark_report_sha256"] = benchmark_report["content_sha256"]
        quality_receipt.pop("content_sha256", None)
        quality_receipt["content_sha256"] = _digest(quality_receipt)
        return benchmark_report["content_sha256"]

    @staticmethod
    def _completion(value: Mapping[str, Any], tool: bool) -> Any:
        try:
            message = value["choices"][0]["message"]
            return {"tool_calls": message["tool_calls"]} if tool else message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise HarnessError("quality_response_shape_invalid") from exc

    def _baseline(self, context: ArmContext) -> Mapping[str, Any] | None:
        if not context.spec.quality_arm.startswith("dflash2"):
            return None
        path = self.output_dir.parent / "mtp" / "quality-receipt.json"
        if path.is_symlink() or not path.is_file():
            _reject("quality_baseline_missing")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError("quality_baseline_invalid") from exc
        if not isinstance(value, Mapping):
            _reject("quality_baseline_invalid")
        return value

    def quality(self, context: ArmContext, benchmark_report: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        suite, suite_sha = quality_evaluator.load_suite(QUALITY_SUITE)
        smoke_value = self._smoke_config(context)
        config = openai_smoke.Config(
            candidate_id=smoke_value["candidate_id"], candidate_type=smoke_value["candidate_type"],
            runtime_id=smoke_value["runtime_id"], manifest_id=smoke_value["manifest_id"],
            base_url=smoke_value["base_url"], model_id=smoke_value["model_id"],
            auth_mode=smoke_value["auth_mode"],
            token_file=None if smoke_value["token_file"] is None else Path(smoke_value["token_file"]),
            connect_timeout_seconds=5.0, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            cancel_after_deltas=2, capabilities=smoke_value["capabilities"],
        )
        token = openai_smoke._read_secret(config.token_file) if config.token_file is not None else None
        client = openai_smoke.Client(config, token)
        try:
            def quality_workload() -> dict[str, Any]:
                responses: dict[str, Any] = {}
                try:
                    for case in suite["cases"]:
                        request = case["request"]
                        generation = request["generation"]
                        body = {
                            "model": config.model_id,
                            "messages": request["messages"],
                            "temperature": generation["temperature"],
                            "seed": generation["seed"],
                            "max_tokens": generation["max_tokens"],
                        }
                        if "tools" in request:
                            body["tools"] = request["tools"]
                            body["tool_choice"] = request["tool_choice"]
                        value, _elapsed = client.json_request(
                            "POST", "/v1/chat/completions", body
                        )
                        responses[case["case_id"]] = self._completion(
                            value, "tools" in request
                        )
                    # The final benchmark hash depends on the monitor's closing
                    # observation. Evaluate with a valid provisional binding,
                    # then rebind and reseal the content-free receipt after the
                    # monitor stops.
                    return quality_evaluator.evaluate(
                        suite, suite_sha, "0" * 64, context.spec.quality_arm,
                        responses, baseline_receipt=self._baseline(context),
                    )
                except safety_sampler.SafetyError:
                    raise
                except Exception as exc:
                    raise safety_sampler.SafetyError("quality_evaluation_failed") from exc

            quality_receipt = self._guard(quality_workload, timeout)
            if not isinstance(quality_receipt, dict):
                _reject("quality_receipt_invalid")
            safety_receipt = self._safety_receipt
            if safety_receipt is None:
                _reject("safety_monitor_missing")
            self._finish_monitor(safety_receipt, benchmark_report, quality_receipt)
        except BaseException:
            self._stop_failed_monitor()
            raise
        return quality_receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(
            args.candidate_id, args.arm, args.operation_id, args.output_dir,
            NativeDependencies(args.output_dir),
        )
    except (HarnessError, OSError, ValueError, openai_smoke.SmokeError, safety_sampler.SafetyError) as exc:
        code = str(exc)
        if re.fullmatch(r"[a-z0-9_]+", code) is None:
            code = "candidate_arm_failed"
        print(code, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
