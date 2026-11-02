#!/usr/bin/env python3
"""Content-free safety aggregation for one acceleration candidate arm."""

from __future__ import annotations

import math
import re
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Callable, Final, NoReturn, Protocol


SCHEMA: Final = "ai-server.acceleration-safety.v1"
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class SafetyError(RuntimeError):
    """A stable content-free safety failure."""


def _reject(code: str) -> NoReturn:
    raise SafetyError(code)


@dataclass(frozen=True)
class SafetyLimits:
    candidate_id: str
    maximum_temperature_c: float
    maximum_power_draw_w: float
    expected_power_cap_w: float
    minimum_vram_free_mib: float
    minimum_ram_available_mib: float
    maximum_swap_io_delta: int
    minimum_sample_count: int = 2
    maximum_consecutive_swap_samples: int = 0
    require_runtime_identity: bool = False


@dataclass(frozen=True)
class SafetyObservation:
    monotonic_ns: int
    gpu_temperature_c: float
    gpu_power_draw_w: float
    power_cap_w: float
    vram_free_mib: float
    ram_available_mib: float
    swap_in_delta: int
    swap_out_delta: int
    nvidia_xid: bool
    host_oom: bool
    container_oom_killed: bool
    restart_count: int
    backend_unavailable_count: int
    incomplete_stream_count: int
    boot_id: str = ""
    gpu_compute_pid_count: int = 0
    gpu_compute_pid_owner_valid: bool = True
    major_fault_delta: int = 0
    cuda_oom: bool = False


@dataclass(frozen=True)
class SafetyResult:
    schema: str
    candidate_id: str
    passed: bool
    sample_count: int
    maximum_temperature_c: float
    maximum_power_draw_w: float
    minimum_vram_free_mib: float
    minimum_ram_available_mib: float
    maximum_swap_in_delta: int
    maximum_swap_out_delta: int
    xid_count: int
    oom_count: int
    restart_delta: int
    backend_unavailable_count: int
    incomplete_stream_count: int
    boot_change_count: int
    gpu_owner_violation_count: int
    maximum_major_fault_delta: int
    maximum_sample_interval_seconds: float
    boot_id_sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "extrema": {
                "maximum_temperature_c": self.maximum_temperature_c,
                "maximum_power_draw_w": self.maximum_power_draw_w,
                "minimum_vram_free_mib": self.minimum_vram_free_mib,
                "minimum_ram_available_mib": self.minimum_ram_available_mib,
                "maximum_swap_in_delta": self.maximum_swap_in_delta,
                "maximum_swap_out_delta": self.maximum_swap_out_delta,
                "maximum_major_fault_delta": self.maximum_major_fault_delta,
                "maximum_sample_interval_seconds": self.maximum_sample_interval_seconds,
            },
            "fault_counts": {
                "xid": self.xid_count,
                "oom": self.oom_count,
                "restart_delta": self.restart_delta,
                "backend_unavailable": self.backend_unavailable_count,
                "incomplete_stream": self.incomplete_stream_count,
                "boot_change": self.boot_change_count,
                "gpu_owner_violation": self.gpu_owner_violation_count,
            },
            "identity": {
                "boot_id_sha256": self.boot_id_sha256,
                "boot_changed": self.boot_change_count != 0,
                "configured_sample_interval_seconds": 1.0,
                "sample_interval_seconds": self.maximum_sample_interval_seconds,
            },
        }


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def validate_limits(limits: SafetyLimits) -> SafetyLimits:
    if not isinstance(limits, SafetyLimits):
        _reject("safety_limits_type_invalid")
    if not isinstance(limits.candidate_id, str) or not _SAFE_ID_RE.fullmatch(limits.candidate_id):
        _reject("candidate_id_invalid")
    for value in (
        limits.maximum_temperature_c,
        limits.maximum_power_draw_w,
        limits.expected_power_cap_w,
        limits.minimum_vram_free_mib,
        limits.minimum_ram_available_mib,
    ):
        if not _finite_nonnegative(value):
            _reject("safety_limit_invalid")
    if (
        isinstance(limits.maximum_swap_io_delta, bool)
        or not isinstance(limits.maximum_swap_io_delta, int)
        or limits.maximum_swap_io_delta < 0
        or isinstance(limits.minimum_sample_count, bool)
        or not isinstance(limits.minimum_sample_count, int)
        or limits.minimum_sample_count < 1
        or isinstance(limits.maximum_consecutive_swap_samples, bool)
        or not isinstance(limits.maximum_consecutive_swap_samples, int)
        or limits.maximum_consecutive_swap_samples < 0
        or not isinstance(limits.require_runtime_identity, bool)
    ):
        _reject("safety_limit_invalid")
    return limits


def validate_observation(observation: SafetyObservation) -> SafetyObservation:
    if not isinstance(observation, SafetyObservation):
        _reject("safety_observation_type_invalid")
    for value in (
        observation.gpu_temperature_c,
        observation.gpu_power_draw_w,
        observation.power_cap_w,
        observation.vram_free_mib,
        observation.ram_available_mib,
    ):
        if not _finite_nonnegative(value):
            _reject("safety_observation_number_invalid")
    for value in (
        observation.monotonic_ns,
        observation.swap_in_delta,
        observation.swap_out_delta,
        observation.restart_count,
        observation.backend_unavailable_count,
        observation.incomplete_stream_count,
        observation.gpu_compute_pid_count,
        observation.major_fault_delta,
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _reject("safety_observation_integer_invalid")
    for value in (
        observation.nvidia_xid,
        observation.host_oom,
        observation.container_oom_killed,
        observation.gpu_compute_pid_owner_valid,
        observation.cuda_oom,
    ):
        if not isinstance(value, bool):
            _reject("safety_observation_boolean_invalid")
    return observation


class SafetySampler:
    """Collect observations and emit aggregate extrema/fault counts only."""

    def __init__(self, limits: SafetyLimits) -> None:
        self._limits = validate_limits(limits)
        self._samples: list[SafetyObservation] = []
        self._boot_id: str | None = None
        self._consecutive_swap_samples = 0

    def _check(self, value: SafetyObservation) -> None:
        if value.gpu_temperature_c > self._limits.maximum_temperature_c:
            _reject("temperature_limit_exceeded")
        if value.gpu_power_draw_w > self._limits.maximum_power_draw_w:
            _reject("power_draw_limit_exceeded")
        if abs(value.power_cap_w - self._limits.expected_power_cap_w) > 0.5:
            _reject("power_cap_drift")
        if value.vram_free_mib < self._limits.minimum_vram_free_mib:
            _reject("vram_floor_failed")
        if value.ram_available_mib < self._limits.minimum_ram_available_mib:
            _reject("ram_floor_failed")
        swap_active = (
            value.swap_in_delta > self._limits.maximum_swap_io_delta
            or value.swap_out_delta > self._limits.maximum_swap_io_delta
        )
        self._consecutive_swap_samples = self._consecutive_swap_samples + 1 if swap_active else 0
        if self._consecutive_swap_samples > self._limits.maximum_consecutive_swap_samples:
            _reject("swap_activity_limit_exceeded")
        if value.nvidia_xid:
            _reject("nvidia_xid_detected")
        if value.host_oom or value.container_oom_killed or value.cuda_oom:
            _reject("oom_detected")
        if value.restart_count != 0:
            _reject("container_restart_detected")
        if value.backend_unavailable_count:
            _reject("backend_unavailable_detected")
        if value.incomplete_stream_count:
            _reject("incomplete_stream_detected")
        if self._limits.require_runtime_identity:
            if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value.boot_id):
                _reject("boot_id_invalid")
            if self._boot_id is None:
                self._boot_id = value.boot_id
            elif value.boot_id != self._boot_id:
                _reject("boot_id_changed")
            if not value.gpu_compute_pid_owner_valid:
                _reject("gpu_compute_pid_owner_invalid")

    def add(self, observation: SafetyObservation) -> None:
        value = validate_observation(observation)
        if self._samples and value.monotonic_ns <= self._samples[-1].monotonic_ns:
            _reject("safety_sample_order_invalid")
        self._check(value)
        self._samples.append(value)

    def evaluate(self) -> SafetyResult:
        if len(self._samples) < self._limits.minimum_sample_count:
            _reject("safety_samples_insufficient")
        first_restart = self._samples[0].restart_count
        maximum_restart = max(sample.restart_count for sample in self._samples)
        if any(sample.restart_count < first_restart for sample in self._samples):
            _reject("restart_counter_invalid")

        result = SafetyResult(
            schema=SCHEMA,
            candidate_id=self._limits.candidate_id,
            passed=True,
            sample_count=len(self._samples),
            maximum_temperature_c=max(sample.gpu_temperature_c for sample in self._samples),
            maximum_power_draw_w=max(sample.gpu_power_draw_w for sample in self._samples),
            minimum_vram_free_mib=min(sample.vram_free_mib for sample in self._samples),
            minimum_ram_available_mib=min(sample.ram_available_mib for sample in self._samples),
            maximum_swap_in_delta=max(sample.swap_in_delta for sample in self._samples),
            maximum_swap_out_delta=max(sample.swap_out_delta for sample in self._samples),
            xid_count=sum(1 for sample in self._samples if sample.nvidia_xid),
            oom_count=sum(
                1 for sample in self._samples if sample.host_oom or sample.container_oom_killed
            ),
            restart_delta=maximum_restart - first_restart,
            backend_unavailable_count=sum(sample.backend_unavailable_count for sample in self._samples),
            incomplete_stream_count=sum(sample.incomplete_stream_count for sample in self._samples),
            boot_change_count=0,
            gpu_owner_violation_count=0,
            maximum_major_fault_delta=max(sample.major_fault_delta for sample in self._samples),
            maximum_sample_interval_seconds=max(
                (right.monotonic_ns - left.monotonic_ns) / 1_000_000_000
                for left, right in zip(self._samples, self._samples[1:])
            ),
            boot_id_sha256=(
                hashlib.sha256(self._boot_id.encode("ascii")).hexdigest()
                if self._boot_id is not None else None
            ),
        )
        return result


class ObservationProbe(Protocol):
    """One bounded host observation; implementations must not retry."""

    def observe(self) -> SafetyObservation: ...


class ConcurrentSafetyMonitor:
    """Collect safety observations while a bounded workload is running."""

    def __init__(
        self,
        limits: SafetyLimits,
        probe: ObservationProbe,
        *,
        interval_seconds: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or not 0.1 <= float(interval_seconds) <= 60.0
        ):
            _reject("safety_interval_invalid")
        self._sampler = SafetySampler(limits)
        self._probe = probe
        self._interval = float(interval_seconds)
        if limits.require_runtime_identity and abs(self._interval - 1.0) > 1e-9:
            _reject("safety_interval_invalid")
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._failure_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            _reject("safety_monitor_state_invalid")
        # A synchronous first sample makes startup failure deterministic.
        self._sampler.add(self._probe.observe())
        self._thread = threading.Thread(target=self._run, name="candidate-safety", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        deadline = self._monotonic() + self._interval
        try:
            while not self._stop.wait(max(0.0, deadline - self._monotonic())):
                self._sampler.add(self._probe.observe())
                deadline += self._interval
        except BaseException as exc:  # surfaced by stop(); never silently ignored
            self._failure = exc
            self._failure_event.set()
            self._stop.set()

    def raise_if_failed(self) -> None:
        if self._failure_event.is_set():
            failure = self._failure
            if isinstance(failure, SafetyError):
                raise failure
            raise SafetyError("safety_probe_failed") from failure

    def run_guarded(
        self,
        workload: Callable[[], object],
        cancel: Callable[[], None],
        *,
        timeout_seconds: float,
        cancellation_timeout_seconds: float = 15.0,
    ) -> object:
        """Run one external workload and cancel it on the first monitor failure."""

        if self._thread is None:
            _reject("safety_monitor_state_invalid")
        if not _finite_nonnegative(timeout_seconds) or timeout_seconds == 0:
            _reject("safety_workload_timeout_invalid")
        result: list[object] = []
        failure: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(workload())
            except BaseException as exc:
                failure.append(exc)

        worker = threading.Thread(target=invoke, name="candidate-workload", daemon=True)
        worker.start()
        deadline = self._monotonic() + float(timeout_seconds)
        monitor_failure: BaseException | None = None
        while worker.is_alive():
            if self._failure_event.wait(0.05):
                monitor_failure = self._failure or SafetyError("safety_probe_failed")
                break
            if self._monotonic() >= deadline:
                monitor_failure = SafetyError("safety_workload_timeout")
                break
        if monitor_failure is not None:
            try:
                cancel()
            except BaseException as exc:
                raise SafetyError("safety_workload_cancel_failed") from exc
            worker.join(timeout=cancellation_timeout_seconds)
            if worker.is_alive():
                _reject("safety_workload_cancel_timeout")
            if isinstance(monitor_failure, SafetyError):
                raise monitor_failure
            raise SafetyError("safety_probe_failed") from monitor_failure
        worker.join()
        self.raise_if_failed()
        if failure:
            if isinstance(failure[0], SafetyError):
                try:
                    cancel()
                except BaseException as exc:
                    raise SafetyError("safety_workload_cancel_failed") from exc
            raise failure[0]
        if len(result) != 1:
            _reject("safety_workload_result_invalid")
        return result[0]

    def stop(self) -> SafetyResult:
        thread = self._thread
        if thread is None:
            _reject("safety_monitor_state_invalid")
        self._stop.set()
        thread.join(timeout=self._interval + 5.0)
        if thread.is_alive():
            _reject("safety_monitor_stop_timeout")
        self._thread = None
        if self._failure is not None:
            self.raise_if_failed()
        # Always take a final observation, guaranteeing at least two samples.
        self._sampler.add(self._probe.observe())
        return self._sampler.evaluate()
