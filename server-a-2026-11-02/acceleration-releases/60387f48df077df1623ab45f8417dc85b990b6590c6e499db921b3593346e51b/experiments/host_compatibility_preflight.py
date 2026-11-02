#!/usr/bin/env python3
"""Fail-closed evaluator for the acceleration host-compatibility preflight.

The module deliberately has no subprocess, Docker, systemd, filesystem-write,
network, or lifecycle code.  A platform adapter supplies one already-collected
read-only :class:`PreflightSnapshot`; this evaluator returns only bounded,
content-free metadata.  Collection failures must be represented by a failed
``collector_ok`` field and can never be treated as an empty/healthy result.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Final, Mapping, NoReturn


SCHEMA: Final = "ai-server.acceleration-host-preflight.v1"
REQUIRED_POWER_LIMIT_W: Final = 300.0
MIN_DRIVER_MAJOR: Final = 580
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_TERMINAL_CONTAINER_STATES = frozenset({"exited", "dead", "not_found"})
_QWEN_LIFECYCLE_BASELINES = frozenset({"running_owner", "idle_unloaded"})


class PreflightError(RuntimeError):
    """A stable failure code containing no observed value or user content."""


def _reject(code: str) -> NoReturn:
    raise PreflightError(code)


@dataclass(frozen=True)
class DockerContract:
    """Result of a fixture/read-only Docker CLI contract probe.

    stdout is intentionally absent: success is based on the return code and an
    independent inspect/compute-process observation.
    """

    command_returncode: int
    inspect_state: str
    gpu_compute_pid_count: int
    timed_out: bool = False


@dataclass(frozen=True)
class ComposeRender:
    returncode: int
    timed_out: bool
    exact_environment: bool
    restart_policy_no: bool
    loopback_only: bool


@dataclass(frozen=True)
class SystemdState:
    active_state: str
    sub_state: str
    main_pid: int
    parse_ok: bool = True


@dataclass(frozen=True)
class SocketState:
    path_exact: bool
    is_socket: bool
    is_symlink: bool
    owner_exact: bool
    group_exact: bool
    mode_exact: bool


@dataclass(frozen=True)
class ResourceState:
    power_limit_w: float
    gpu_compute_pid_count: int
    mem_available_mib: int
    required_mem_available_mib: int
    swap_in_delta: int
    swap_out_delta: int
    disk_available_bytes: int
    calculated_candidate_bytes: int
    restore_reserve_bytes: int
    temperature_c: float
    vram_free_mib: int


@dataclass(frozen=True)
class RuntimeSafetyState:
    unexpected_xid: bool
    host_oom: bool
    container_oom_killed: bool
    unexpected_restart_count: int


@dataclass(frozen=True)
class IdentityState:
    current_cas_manifest_sha256: str
    expected_current_cas_manifest_sha256: str
    rollback_cas_manifest_sha256: str
    expected_rollback_cas_manifest_sha256: str
    caddy_bind_source_match: bool
    caddy_bytes_sha256: str
    expected_caddy_bytes_sha256: str
    caddy_auth_identity_sha256: str
    expected_caddy_auth_identity_sha256: str
    private_routes_match: bool
    qwen_runtime_identity_match: bool
    qwen_lifecycle_baseline: str


@dataclass(frozen=True)
class AdmissionState:
    phase: str
    capacity: str
    unfinished_operation_count: int
    reconciliation_pending: bool
    breaker_open: bool
    fifo_active_count: int
    fifo_owner_count: int
    clean_drain_possible: bool


@dataclass(frozen=True)
class StaleState:
    candidate_container_count: int
    candidate_mount_count: int
    candidate_gpu_pid_count: int
    host_lock_present: bool
    unfinished_journal_present: bool


@dataclass(frozen=True)
class AcceptanceEvidence:
    """Content-free results for the active A-01 through A-05 acceptance gates."""

    a01_services_healthy: bool
    a01_heavy_owner_valid: bool
    a02_qwen_answer_complete: bool
    a02_ui_responsive: bool
    a03_wai_png_valid: bool
    a03_pony_v7_png_valid: bool
    a03_fifo_terminal_single_resident: bool
    a03_no_raw_backend: bool
    a04_qwen_return_complete: bool
    a04_no_interrupt_or_replay: bool
    a05_unauthenticated_401: bool
    a05_saved_credential_editor: bool
    a05_coordinator_gate: bool
    a05_no_raw_listener: bool
    a05_fifo_bypass_rejected: bool


@dataclass(frozen=True)
class AcceptanceReceipt:
    gate_id: str
    passed: bool
    evidence_complete: bool
    receipt_sha256: str


@dataclass(frozen=True)
class PreflightSnapshot:
    candidate_id: str
    collector_ok: bool
    boot_id: str
    journal_cursor: str
    driver_version: str
    cdi_gpu_all: bool
    cdi_gpu_zero: bool
    cuda13_compatible: bool
    toolkit_available: bool
    docker_version_supported: bool
    compose_version_supported: bool
    docker_contract: DockerContract
    compose_render: ComposeRender
    systemd_states: Mapping[str, SystemdState]
    socket_states: Mapping[str, SocketState]
    resources: ResourceState
    runtime_safety: RuntimeSafetyState
    identities: IdentityState
    admission: AdmissionState
    stale: StaleState
    acceptance: AcceptanceEvidence
    acceptance_receipts: Mapping[str, AcceptanceReceipt]
    writable_roots_exact: bool
    production_cas_read_only: bool


@dataclass(frozen=True)
class PreflightResult:
    schema: str
    passed: bool
    candidate_id: str
    boot_id_sha256: str
    journal_cursor_sha256: str
    identity_sha256: str
    gates: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "candidate_id": self.candidate_id,
            "boot_id_sha256": self.boot_id_sha256,
            "journal_cursor_sha256": self.journal_cursor_sha256,
            "identity_sha256": self.identity_sha256,
            "gates": list(self.gates),
        }


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _driver_major(value: str) -> int:
    if not isinstance(value, str) or len(value) > 64:
        _reject("driver_version_invalid")
    match = re.fullmatch(r"([0-9]{1,4})(?:\.[0-9]{1,5}){1,3}", value)
    if match is None:
        _reject("driver_version_invalid")
    return int(match.group(1))


def _validate_shape(snapshot: PreflightSnapshot) -> None:
    if not isinstance(snapshot, PreflightSnapshot):
        _reject("snapshot_type_invalid")
    nested_types = (
        (snapshot.docker_contract, DockerContract),
        (snapshot.compose_render, ComposeRender),
        (snapshot.resources, ResourceState),
        (snapshot.runtime_safety, RuntimeSafetyState),
        (snapshot.identities, IdentityState),
        (snapshot.admission, AdmissionState),
        (snapshot.stale, StaleState),
        (snapshot.acceptance, AcceptanceEvidence),
    )
    if any(not isinstance(value, expected) for value, expected in nested_types):
        _reject("snapshot_shape_invalid")
    if not isinstance(snapshot.candidate_id, str) or not _SAFE_ID_RE.fullmatch(snapshot.candidate_id):
        _reject("candidate_id_invalid")
    if not isinstance(snapshot.collector_ok, bool):
        _reject("collector_status_invalid")
    if not isinstance(snapshot.boot_id, str) or not _BOOT_ID_RE.fullmatch(snapshot.boot_id):
        _reject("boot_id_invalid")
    # The cursor is deliberately opaque: bounded/non-empty is the entire syntax contract.
    if (
        not isinstance(snapshot.journal_cursor, str)
        or not snapshot.journal_cursor
        or len(snapshot.journal_cursor.encode("utf-8")) > 4096
        or "\x00" in snapshot.journal_cursor
    ):
        _reject("journal_cursor_invalid")
    for value in (snapshot.systemd_states, snapshot.socket_states):
        if not isinstance(value, Mapping) or not value:
            _reject("structured_state_missing")
    for name in (*snapshot.systemd_states.keys(), *snapshot.socket_states.keys()):
        if not isinstance(name, str) or not _SAFE_ID_RE.fullmatch(name):
            _reject("structured_state_name_invalid")
    boolean_values = (
        snapshot.collector_ok,
        snapshot.cdi_gpu_all,
        snapshot.cdi_gpu_zero,
        snapshot.cuda13_compatible,
        snapshot.toolkit_available,
        snapshot.docker_version_supported,
        snapshot.compose_version_supported,
        snapshot.writable_roots_exact,
        snapshot.production_cas_read_only,
        *(getattr(snapshot.acceptance, field) for field in AcceptanceEvidence.__dataclass_fields__),
    )
    if any(not isinstance(value, bool) for value in boolean_values):
        _reject("boolean_value_invalid")
    nested_booleans = (
        snapshot.docker_contract.timed_out,
        snapshot.compose_render.timed_out,
        snapshot.compose_render.exact_environment,
        snapshot.compose_render.restart_policy_no,
        snapshot.compose_render.loopback_only,
        snapshot.runtime_safety.unexpected_xid,
        snapshot.runtime_safety.host_oom,
        snapshot.runtime_safety.container_oom_killed,
        snapshot.identities.caddy_bind_source_match,
        snapshot.identities.private_routes_match,
        snapshot.identities.qwen_runtime_identity_match,
        snapshot.admission.reconciliation_pending,
        snapshot.admission.breaker_open,
        snapshot.admission.clean_drain_possible,
        snapshot.stale.host_lock_present,
        snapshot.stale.unfinished_journal_present,
    )
    if any(not isinstance(value, bool) for value in nested_booleans):
        _reject("boolean_value_invalid")
    if (
        not isinstance(snapshot.identities.qwen_lifecycle_baseline, str)
        or snapshot.identities.qwen_lifecycle_baseline not in _QWEN_LIFECYCLE_BASELINES
    ):
        _reject("qwen_baseline_invalid")
    nonnegative_integers = (
        snapshot.docker_contract.gpu_compute_pid_count,
        snapshot.resources.gpu_compute_pid_count,
        snapshot.runtime_safety.unexpected_restart_count,
        snapshot.admission.unfinished_operation_count,
        snapshot.admission.fifo_active_count,
        snapshot.admission.fifo_owner_count,
        snapshot.stale.candidate_container_count,
        snapshot.stale.candidate_mount_count,
        snapshot.stale.candidate_gpu_pid_count,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in nonnegative_integers):
        _reject("integer_value_invalid")
    for field in (
        snapshot.resources.power_limit_w,
        snapshot.resources.mem_available_mib,
        snapshot.resources.required_mem_available_mib,
        snapshot.resources.swap_in_delta,
        snapshot.resources.swap_out_delta,
        snapshot.resources.disk_available_bytes,
        snapshot.resources.calculated_candidate_bytes,
        snapshot.resources.restore_reserve_bytes,
        snapshot.resources.temperature_c,
        snapshot.resources.vram_free_mib,
    ):
        if not _finite_nonnegative(field):
            _reject("resource_value_invalid")
    if snapshot.resources.gpu_compute_pid_count < 0:
        _reject("resource_value_invalid")
    identity = snapshot.identities
    for value in (
        identity.current_cas_manifest_sha256,
        identity.expected_current_cas_manifest_sha256,
        identity.rollback_cas_manifest_sha256,
        identity.expected_rollback_cas_manifest_sha256,
        identity.caddy_bytes_sha256,
        identity.expected_caddy_bytes_sha256,
        identity.caddy_auth_identity_sha256,
        identity.expected_caddy_auth_identity_sha256,
    ):
        if not _valid_sha256(value):
            _reject("identity_digest_invalid")
    expected_gates = {"A-01", "A-02", "A-03", "A-04", "A-05"}
    if not isinstance(snapshot.acceptance_receipts, Mapping) or set(snapshot.acceptance_receipts) != expected_gates:
        _reject("acceptance_receipts_invalid")
    for gate_id, receipt in snapshot.acceptance_receipts.items():
        if (
            not isinstance(receipt, AcceptanceReceipt)
            or receipt.gate_id != gate_id
            or not isinstance(receipt.passed, bool)
            or not isinstance(receipt.evidence_complete, bool)
            or not _valid_sha256(receipt.receipt_sha256)
        ):
            _reject("acceptance_receipt_invalid")


def evaluate_snapshot(snapshot: PreflightSnapshot) -> PreflightResult:
    """Evaluate one snapshot, stopping on the first stable content-free blocker."""

    _validate_shape(snapshot)
    if not snapshot.collector_ok:
        _reject("collector_failed")
    if _driver_major(snapshot.driver_version) < MIN_DRIVER_MAJOR:
        _reject("driver_too_old")
    if not all(
        (
            snapshot.cdi_gpu_all,
            snapshot.cdi_gpu_zero,
            snapshot.cuda13_compatible,
            snapshot.toolkit_available,
            snapshot.docker_version_supported,
            snapshot.compose_version_supported,
        )
    ):
        _reject("gpu_container_stack_incompatible")

    docker = snapshot.docker_contract
    if not isinstance(docker.command_returncode, int) or isinstance(docker.command_returncode, bool):
        _reject("docker_contract_invalid")
    if (
        docker.timed_out
        or docker.command_returncode != 0
        or docker.inspect_state not in _TERMINAL_CONTAINER_STATES
        or docker.gpu_compute_pid_count != 0
    ):
        _reject("docker_contract_failed")

    compose = snapshot.compose_render
    if (
        compose.timed_out
        or compose.returncode != 0
        or not compose.exact_environment
        or not compose.restart_policy_no
        or not compose.loopback_only
    ):
        _reject("compose_render_failed")

    for state in snapshot.systemd_states.values():
        if (
            not isinstance(state, SystemdState)
            or not isinstance(state.parse_ok, bool)
            or not state.parse_ok
            or state.active_state not in {"active", "inactive"}
            or not state.sub_state
            or isinstance(state.main_pid, bool)
            or not isinstance(state.main_pid, int)
            or state.main_pid < 0
        ):
            _reject("systemd_state_invalid")
    for state in snapshot.socket_states.values():
        if (
            not isinstance(state, SocketState)
            or any(
                not isinstance(value, bool)
                for value in (
                    state.path_exact,
                    state.is_socket,
                    state.is_symlink,
                    state.owner_exact,
                    state.group_exact,
                    state.mode_exact,
                )
            )
            or not state.path_exact
            or not state.is_socket
            or state.is_symlink
            or not state.owner_exact
            or not state.group_exact
            or not state.mode_exact
        ):
            _reject("socket_state_invalid")

    resource = snapshot.resources
    if abs(resource.power_limit_w - REQUIRED_POWER_LIMIT_W) > 0.5:
        _reject("power_limit_not_300w")
    identity = snapshot.identities
    if not identity.qwen_runtime_identity_match:
        _reject("qwen_identity_mismatch")
    # Preserve the existing generic ownership blockers before correlating the
    # accepted Qwen lifecycle baseline with the exact remaining cardinality.
    if resource.gpu_compute_pid_count > 1:
        _reject("unexpected_gpu_owners")
    if resource.gpu_compute_pid_count == 1 and not snapshot.acceptance.a01_heavy_owner_valid:
        _reject("unexpected_gpu_owner")
    if identity.qwen_lifecycle_baseline == "running_owner":
        if resource.gpu_compute_pid_count != 1 or not snapshot.acceptance.a01_heavy_owner_valid:
            _reject("qwen_baseline_mismatch")
    elif resource.gpu_compute_pid_count != 0:
        _reject("qwen_baseline_mismatch")
    # The no-cutover preflight runs while accepted production Qwen may still
    # own the GPU.  One process is allowed only when the independently supplied
    # A-01 evidence proves that heavy owner; candidate/post-stop idle checks
    # remain separate and require strict zero.
    if resource.mem_available_mib < resource.required_mem_available_mib:
        _reject("ram_floor_failed")
    if resource.swap_in_delta != 0 or resource.swap_out_delta != 0:
        _reject("swap_activity_present")
    required_disk = resource.calculated_candidate_bytes + resource.restore_reserve_bytes
    if resource.disk_available_bytes < required_disk:
        _reject("calculated_disk_budget_failed")

    safety = snapshot.runtime_safety
    if (
        safety.unexpected_xid
        or safety.host_oom
        or safety.container_oom_killed
        or safety.unexpected_restart_count != 0
    ):
        _reject("baseline_runtime_fault")

    if (
        identity.current_cas_manifest_sha256 != identity.expected_current_cas_manifest_sha256
        or identity.rollback_cas_manifest_sha256 != identity.expected_rollback_cas_manifest_sha256
    ):
        _reject("cas_identity_mismatch")
    if (
        not identity.caddy_bind_source_match
        or identity.caddy_bytes_sha256 != identity.expected_caddy_bytes_sha256
        or identity.caddy_auth_identity_sha256 != identity.expected_caddy_auth_identity_sha256
        or not identity.private_routes_match
    ):
        _reject("caddy_auth_identity_mismatch")
    admission = snapshot.admission
    if (
        admission.phase != "public"
        or admission.capacity != "serialized-99"
        or admission.unfinished_operation_count != 0
        or admission.reconciliation_pending
        or admission.breaker_open
        or admission.fifo_active_count != 0
        or admission.fifo_owner_count != 0
        or not admission.clean_drain_possible
    ):
        _reject("admission_capacity_fifo_invalid")

    stale = snapshot.stale
    if (
        stale.candidate_container_count != 0
        or stale.candidate_mount_count != 0
        or stale.candidate_gpu_pid_count != 0
        or stale.host_lock_present
        or stale.unfinished_journal_present
    ):
        _reject("stale_experiment_state_present")
    if not snapshot.writable_roots_exact or not snapshot.production_cas_read_only:
        _reject("storage_boundary_invalid")

    acceptance = snapshot.acceptance
    for gate_id in ("A-01", "A-02", "A-03", "A-04", "A-05"):
        receipt = snapshot.acceptance_receipts[gate_id]
        if not receipt.passed or not receipt.evidence_complete:
            _reject(gate_id.replace("-", "_") + "_RECEIPT_FAILED")
    a01 = (
        acceptance.a01_services_healthy
        and acceptance.a01_heavy_owner_valid
        and safety.unexpected_restart_count == 0
        and not safety.container_oom_killed
        and resource.gpu_compute_pid_count <= 1
    )
    if not a01:
        _reject("A_01_FAILED")
    if not (acceptance.a02_qwen_answer_complete and acceptance.a02_ui_responsive):
        _reject("A_02_FAILED")
    if not (
        acceptance.a03_wai_png_valid
        and acceptance.a03_pony_v7_png_valid
        and acceptance.a03_fifo_terminal_single_resident
        and acceptance.a03_no_raw_backend
    ):
        _reject("A_03_FAILED")
    if not (acceptance.a04_qwen_return_complete and acceptance.a04_no_interrupt_or_replay):
        _reject("A_04_FAILED")
    if not (
        acceptance.a05_unauthenticated_401
        and acceptance.a05_saved_credential_editor
        and acceptance.a05_coordinator_gate
        and acceptance.a05_no_raw_listener
        and acceptance.a05_fifo_bypass_rejected
    ):
        _reject("A_05_FAILED")

    identity_digest = _digest(
        "|".join(
            (
                identity.current_cas_manifest_sha256,
                identity.rollback_cas_manifest_sha256,
                identity.caddy_bytes_sha256,
                identity.caddy_auth_identity_sha256,
                identity.qwen_lifecycle_baseline,
                *(snapshot.acceptance_receipts[gate].receipt_sha256 for gate in sorted(snapshot.acceptance_receipts)),
            )
        )
    )
    return PreflightResult(
        schema=SCHEMA,
        passed=True,
        candidate_id=snapshot.candidate_id,
        boot_id_sha256=_digest(snapshot.boot_id),
        journal_cursor_sha256=_digest(snapshot.journal_cursor),
        identity_sha256=identity_digest,
        gates=("A-01", "A-02", "A-03", "A-04", "A-05", "HOST_COMPAT"),
    )


def run_preflight(collector: Callable[[], PreflightSnapshot]) -> PreflightResult:
    """Call one injected read-only collector and evaluate exactly one snapshot."""

    if not callable(collector):
        _reject("collector_invalid")
    try:
        snapshot = collector()
    except Exception as exc:
        raise PreflightError("collector_failed") from exc
    return evaluate_snapshot(snapshot)
