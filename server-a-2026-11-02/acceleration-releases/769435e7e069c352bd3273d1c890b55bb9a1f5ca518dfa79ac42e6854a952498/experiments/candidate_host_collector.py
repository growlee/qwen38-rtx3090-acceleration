#!/usr/bin/env python3
"""Shell-free adapter for the sealed Server A read-only host probe.

The host probe has one fixed installed argv and returns one bounded JSON object.
This adapter deliberately keeps expected identities outside the probe result so
an observed mismatch survives into :class:`PreflightSnapshot`.  Raw stdout and
stderr are never included in errors or receipts.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

try:  # The sealed probe runs only on Server A's POSIX host.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX development hosts
    fcntl = None  # type: ignore[assignment]

try:
    from . import host_compatibility_preflight as preflight
except ImportError:  # pragma: no cover - direct execution from experiments/
    import host_compatibility_preflight as preflight  # type: ignore[no-redef]


PYTHON = "/usr/bin/python3"
MAX_OUTPUT_BYTES = 1_048_576
MAX_TIMEOUT_SECONDS = 60.0
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_OPAQUE = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_CANDIDATE_COMPOSE_RELATIVE = {
    "qwen38-vllm": (
        "experiments/qwen38_vllm/compose.mtp.yaml",
        "experiments/qwen38_vllm/compose.dflash2.yaml",
    ),
    "freetoken": ("experiments/freetoken/compose.yaml",),
    "llama-dflash2-official": (
        "experiments/llama_dflash2_official/compose.plain.yaml",
        "experiments/llama_dflash2_official/compose.mtp.yaml",
        "experiments/llama_dflash2_official/compose.dflash2.yaml",
    ),
}

# Authoritative Server A read-only map.  These are deliberately not caller
# configurable and are retained here for the self-hosted probe implementation.
PROJECT_ROOT = "/opt/homelab/ai-server"
RELEASES_ROOT = "/srv/ai-server/artifacts/releases"
MODEL_STORE_ROOT = "/srv/ai-server/model-store"
CADDY_SOURCE = "/opt/homelab/ai-server/deploy/caddy/Caddyfile.telegram-miniapp.candidate"
CADDY_TARGET = "/etc/caddy/Caddyfile"
OPERATOR_SOCKET = "/run/ai-server/supervisor/operator/supervisor.sock"
CURRENT_RELEASE = "ai-cas-current-20260829"
ROLLBACK_RELEASE = "ai-cas-rollback-20260829"
EXPECTED_CAPACITY = "serialized-99"
EXPECTED_CADDY_SHA256 = "4aa02ad542b05a24e0ca4c48bd1d7a3ccabc5930cd53c5c8ad65b47fba45c8f7"
EXPECTED_AUTH_IDENTITY_SHA256 = "5bf9a7ffc684c5ea55262232e5b7a7f2103cf1818ab93da5f49f5ad9d4f8b5a0"

JOURNALCTL = "/usr/bin/journalctl"
JOURNAL_FAULT_PATTERN = (
    "NVRM: Xid|NVIDIA Xid|NV_ERR_NO_MEMORY|"
    "out of memory: killed process|oom-kill:"
)
NVIDIA_SMI = "/usr/bin/nvidia-smi"
NVIDIA_CTK = "/usr/bin/nvidia-ctk"
DOCKER = "/usr/bin/docker"
SYSTEMCTL = "/usr/bin/systemctl"
PRODUCTION_CONTAINERS = ("ai-server-qwen", "ai-server-gateway", "ai-server-open-webui", "ai-server-edge")
FIXTURE_CONTAINER = "ai-exp-host-preflight-fixture"
SOCKET_PATHS = {"admission": OPERATOR_SOCKET}
HOST_LOCK = "/srv/ai-server/state/acceleration-experiments/locks/host.lock"
_LOCAL_TIMEOUT = 20.0
_LOCAL_MAX = 262_144
_OPERATOR_MAX = 65_536
_ADMISSION_OPERATOR_REQUEST_SCHEMA = "ai-admission-operator-command/v1"
_ADMISSION_OPERATOR_RESPONSE_SCHEMA = "ai-admission-operator-response/v1"
_SUPERVISOR_STATUS_SCHEMA = "ai-supervisor-status/v2"
_ADMISSION_PHASES = frozenset(
    {"closed", "qualification", "smoke", "public", "abort_pending", "reconciling"}
)
_CAPACITY_MODES = frozenset({"shared-48", "shared-64", "serialized-99"})
_PROFILE_STATES = frozenset(
    {
        "stopped", "starting", "ready", "busy", "idle", "degraded",
        "reconciling", "unknown_outcome",
    }
)
_PROFILE_STATE_KEYS = frozenset(
    {
        "state", "breaker_open", "idle_timeout_seconds", "last_transition_at",
        "recovery_attempts", "last_diagnostics", "diagnostics",
    }
)
_ADMISSION_STATUS_KEYS = frozenset(
    {
        "schema", "ok", "request_id", "admission_epoch", "phase",
        "resume_phase", "capacity_mode", "boot_id", "coordinator_generation",
        "restart_pending", "reconciliation_required", "queue_depth",
        "active_ledger", "active_ledger_digest",
    }
)
_SUPERVISOR_STATUS_KEYS = frozenset(
    {
        "ok", "status_schema", "active_profile", "active_by_lane",
        "active_profiles", "last_operation", "global_breaker_open",
        "last_global_health", "shared_mode_enabled", "capacity_mode",
        "admission_epoch", "admission_phase", "boot_id", "profile", "status",
    }
)
_ACTIVE_LEDGER_KEYS = frozenset(
    {
        "admission_epoch", "boot_id", "coordinator_generation", "client_service",
        "client_generation", "job_id", "idempotency_digest", "lane", "profile",
        "turn_seq", "enqueue_monotonic", "confirm_deadline", "queue_deadline",
        "state", "dispatch_grant_digest", "grant_consumed", "operation_id",
        "cancel_origin", "terminal_outcome", "terminal_reason", "queue_position",
        "queue_depth",
    }
)


class CollectorError(RuntimeError):
    """Stable, content-free collector failure."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes = b""


class CommandRunner(Protocol):
    def run(
        self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """Run the single allowlisted probe without a shell or inherited secrets."""

    def __init__(self, package_root: Path | str) -> None:
        self.package_root = _package_root(package_root)
        self.probe_argv = _probe_argv(self.package_root)

    def run(
        self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None
    ) -> CommandResult:
        expected = _probe_argv(self.package_root)
        if tuple(argv) != expected or tuple(argv) != self.probe_argv:
            raise CollectorError("probe_argv_not_allowlisted")
        completed = subprocess.run(
            expected,
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class CollectorConfig:
    """Sealed expectations and non-observable evidence for one candidate."""

    candidate_id: str
    compose_files: tuple[str, ...]
    expected_systemd_units: tuple[str, ...]
    expected_socket_names: tuple[str, ...]
    required_mem_available_mib: int
    calculated_candidate_bytes: int
    restore_reserve_bytes: int
    expected_current_cas_manifest_sha256: str
    expected_rollback_cas_manifest_sha256: str
    expected_caddy_bytes_sha256: str
    expected_caddy_auth_identity_sha256: str
    acceptance: preflight.AcceptanceEvidence
    acceptance_receipts: Mapping[str, preflight.AcceptanceReceipt]
    timeout_seconds: float = 20.0
    package_root: str = ""


def _package_root(value: Path | str) -> Path:
    root = Path(value)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        _fail("package_root_invalid")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise CollectorError("package_root_invalid") from exc
    if resolved != root.absolute():
        _fail("package_root_invalid")
    return resolved


def _package_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        _fail("package_file_invalid")
    path = root / relative_path
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CollectorError("package_file_invalid") from exc
    if (
        resolved != path.absolute()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
    ):
        _fail("package_file_invalid")
    return resolved


def candidate_compose_files(package_root: Path | str, candidate_id: str) -> tuple[str, ...]:
    root = _package_root(package_root)
    relatives = _CANDIDATE_COMPOSE_RELATIVE.get(candidate_id)
    if relatives is None:
        _fail("candidate_id_invalid")
    return tuple(str(_package_file(root, relative)) for relative in relatives)


def _probe_argv(package_root: Path | str) -> tuple[str, ...]:
    root = _package_root(package_root)
    probe = _package_file(root, "experiments/candidate_host_collector.py")
    return (PYTHON, "-B", str(probe), "--snapshot-v1")


def _fail(code: str) -> None:
    raise CollectorError(code)


def _caddy_auth_identity(caddy: bytes) -> str:
    entries: list[bytes] = []
    for line in caddy.splitlines():
        fields = line.strip().split()
        if fields and fields[0] == b"growlee":
            if len(fields) != 2:
                _fail("caddy_auth_identity_invalid")
            entries.append(fields[1])
    if not entries or any(entry != entries[0] for entry in entries):
        _fail("caddy_auth_identity_invalid")
    return hashlib.sha256(entries[0]).hexdigest()


def _journal_runtime_faults(journal: bytes) -> tuple[bool, bool]:
    lower = journal.decode("utf-8", "replace").lower()
    unexpected_xid = "nvrm: xid" in lower or "nvidia xid" in lower
    host_oom = (
        "out of memory: killed process" in lower
        or "oom-kill:" in lower
        or "nv_err_no_memory" in lower
        or ("nvrm:" in lower and "out of memory" in lower)
    )
    return unexpected_xid, host_oom


def _journal_fault_argv() -> tuple[str, ...]:
    return (
        JOURNALCTL,
        "-b",
        "-k",
        "--no-pager",
        "-p",
        "warning..alert",
        "--grep",
        JOURNAL_FAULT_PATTERN,
        "--case-sensitive=no",
        "-o",
        "cat",
    )


def _exact(value: object, keys: set[str], code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(code)
    return value


def _boolean(value: object, code: str) -> bool:
    if not isinstance(value, bool):
        _fail(code)
    return value


def _integer(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(code)
    return value


def _number(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(code)
    return float(value)


def _text(value: object, code: str, *, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value.encode()) > limit:
        _fail(code)
    return value


def _digest(value: object, code: str) -> str:
    text = _text(value, code, limit=64)
    if _SHA256.fullmatch(text) is None:
        _fail(code)
    return text


def _validate_config(config: CollectorConfig) -> None:
    if not isinstance(config, CollectorConfig):
        _fail("config_invalid")
    if _SAFE_ID.fullmatch(config.candidate_id) is None:
        _fail("candidate_id_invalid")
    root = _package_root(config.package_root)
    if config.compose_files != candidate_compose_files(root, config.candidate_id):
        _fail("compose_paths_invalid")
    for names, code in (
        (config.expected_systemd_units, "systemd_units_invalid"),
        (config.expected_socket_names, "socket_names_invalid"),
    ):
        if not names or len(set(names)) != len(names) or any(
            not isinstance(name, str) or _SAFE_ID.fullmatch(name) is None for name in names
        ):
            _fail(code)
    for value in (
        config.required_mem_available_mib,
        config.calculated_candidate_bytes,
        config.restore_reserve_bytes,
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail("resource_expectation_invalid")
    for value in (
        config.expected_current_cas_manifest_sha256,
        config.expected_rollback_cas_manifest_sha256,
        config.expected_caddy_bytes_sha256,
        config.expected_caddy_auth_identity_sha256,
    ):
        _digest(value, "identity_expectation_invalid")
    if (
        config.expected_caddy_bytes_sha256 != EXPECTED_CADDY_SHA256
        or config.expected_caddy_auth_identity_sha256 != EXPECTED_AUTH_IDENTITY_SHA256
    ):
        _fail("caddy_identity_expectation_invalid")
    if (
        isinstance(config.timeout_seconds, bool)
        or not isinstance(config.timeout_seconds, (int, float))
        or not math.isfinite(config.timeout_seconds)
        or not 0 < config.timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        _fail("timeout_invalid")
    if not isinstance(config.acceptance, preflight.AcceptanceEvidence):
        _fail("acceptance_invalid")
    if not isinstance(config.acceptance_receipts, Mapping):
        _fail("acceptance_receipts_invalid")


def _request(config: CollectorConfig) -> bytes:
    # Only public paths/names and numeric resource expectations cross stdin.
    value = {
        "schema": "ai-server.acceleration-host-probe-request.v1",
        "candidate_id": config.candidate_id,
        "compose_files": list(config.compose_files),
        "systemd_units": list(config.expected_systemd_units),
        "socket_names": list(config.expected_socket_names),
        "required_mem_available_mib": config.required_mem_available_mib,
        "calculated_candidate_bytes": config.calculated_candidate_bytes,
        "restore_reserve_bytes": config.restore_reserve_bytes,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _parse_snapshot(config: CollectorConfig, payload: object) -> preflight.PreflightSnapshot:
    root_keys = {
        "schema", "boot_id", "journal_cursor", "driver_version", "cdi_gpu_all",
        "cdi_gpu_zero", "cuda13_compatible", "toolkit_available",
        "docker_version_supported", "compose_version_supported", "docker_contract",
        "compose_render", "systemd_states", "socket_states", "resources",
        "runtime_safety", "identities", "admission", "stale", "writable_roots_exact",
        "production_cas_read_only",
    }
    root = _exact(payload, root_keys, "probe_output_shape_invalid")
    if root["schema"] != "ai-server.acceleration-host-probe.v1":
        _fail("probe_schema_invalid")

    docker = _exact(root["docker_contract"], {"command_returncode", "inspect_state", "gpu_compute_pid_count", "timed_out"}, "docker_probe_invalid")
    compose = _exact(root["compose_render"], {"returncode", "timed_out", "exact_environment", "restart_policy_no", "loopback_only"}, "compose_probe_invalid")
    resources = _exact(root["resources"], {"power_limit_w", "gpu_compute_pid_count", "mem_available_mib", "swap_in_delta", "swap_out_delta", "disk_available_bytes", "temperature_c", "vram_free_mib"}, "resource_probe_invalid")
    safety = _exact(root["runtime_safety"], {"unexpected_xid", "host_oom", "container_oom_killed", "unexpected_restart_count"}, "safety_probe_invalid")
    identities = _exact(root["identities"], {"current_cas_manifest_sha256", "rollback_cas_manifest_sha256", "caddy_bind_source_match", "caddy_bytes_sha256", "caddy_auth_identity_sha256", "private_routes_match", "qwen_runtime_identity_match"}, "identity_probe_invalid")
    admission = _exact(root["admission"], {"phase", "capacity", "unfinished_operation_count", "reconciliation_pending", "breaker_open", "fifo_active_count", "fifo_owner_count", "clean_drain_possible"}, "admission_probe_invalid")
    stale = _exact(root["stale"], {"candidate_container_count", "candidate_mount_count", "candidate_gpu_pid_count", "host_lock_present", "unfinished_journal_present"}, "stale_probe_invalid")

    systemd_raw = root["systemd_states"]
    if not isinstance(systemd_raw, dict) or set(systemd_raw) != set(config.expected_systemd_units):
        _fail("systemd_probe_invalid")
    systemd_states: dict[str, preflight.SystemdState] = {}
    for name, raw in systemd_raw.items():
        item = _exact(raw, {"active_state", "sub_state", "main_pid", "parse_ok"}, "systemd_probe_invalid")
        systemd_states[name] = preflight.SystemdState(
            _text(item["active_state"], "systemd_probe_invalid", limit=64),
            _text(item["sub_state"], "systemd_probe_invalid", limit=64),
            _integer(item["main_pid"], "systemd_probe_invalid"),
            _boolean(item["parse_ok"], "systemd_probe_invalid"),
        )

    sockets_raw = root["socket_states"]
    if not isinstance(sockets_raw, dict) or set(sockets_raw) != set(config.expected_socket_names):
        _fail("socket_probe_invalid")
    socket_states: dict[str, preflight.SocketState] = {}
    socket_fields = {"path_exact", "is_socket", "is_symlink", "owner_exact", "group_exact", "mode_exact"}
    for name, raw in sockets_raw.items():
        item = _exact(raw, socket_fields, "socket_probe_invalid")
        socket_states[name] = preflight.SocketState(*(_boolean(item[field], "socket_probe_invalid") for field in ("path_exact", "is_socket", "is_symlink", "owner_exact", "group_exact", "mode_exact")))

    return preflight.PreflightSnapshot(
        candidate_id=config.candidate_id,
        collector_ok=True,
        boot_id=_text(root["boot_id"], "boot_id_invalid", limit=64),
        journal_cursor=_text(root["journal_cursor"], "journal_cursor_invalid"),
        driver_version=_text(root["driver_version"], "driver_version_invalid", limit=64),
        cdi_gpu_all=_boolean(root["cdi_gpu_all"], "gpu_stack_probe_invalid"),
        cdi_gpu_zero=_boolean(root["cdi_gpu_zero"], "gpu_stack_probe_invalid"),
        cuda13_compatible=_boolean(root["cuda13_compatible"], "gpu_stack_probe_invalid"),
        toolkit_available=_boolean(root["toolkit_available"], "gpu_stack_probe_invalid"),
        docker_version_supported=_boolean(root["docker_version_supported"], "gpu_stack_probe_invalid"),
        compose_version_supported=_boolean(root["compose_version_supported"], "gpu_stack_probe_invalid"),
        docker_contract=preflight.DockerContract(_integer(docker["command_returncode"], "docker_probe_invalid"), _text(docker["inspect_state"], "docker_probe_invalid", limit=64), _integer(docker["gpu_compute_pid_count"], "docker_probe_invalid"), _boolean(docker["timed_out"], "docker_probe_invalid")),
        compose_render=preflight.ComposeRender(_integer(compose["returncode"], "compose_probe_invalid"), _boolean(compose["timed_out"], "compose_probe_invalid"), _boolean(compose["exact_environment"], "compose_probe_invalid"), _boolean(compose["restart_policy_no"], "compose_probe_invalid"), _boolean(compose["loopback_only"], "compose_probe_invalid")),
        systemd_states=systemd_states,
        socket_states=socket_states,
        resources=preflight.ResourceState(
            _number(resources["power_limit_w"], "resource_probe_invalid"),
            _integer(resources["gpu_compute_pid_count"], "resource_probe_invalid"),
            _number(resources["mem_available_mib"], "resource_probe_invalid"),
            config.required_mem_available_mib,
            _integer(resources["swap_in_delta"], "resource_probe_invalid"),
            _integer(resources["swap_out_delta"], "resource_probe_invalid"),
            _integer(resources["disk_available_bytes"], "resource_probe_invalid"),
            config.calculated_candidate_bytes,
            config.restore_reserve_bytes,
            _number(resources["temperature_c"], "resource_probe_invalid"),
            _number(resources["vram_free_mib"], "resource_probe_invalid"),
        ),
        runtime_safety=preflight.RuntimeSafetyState(_boolean(safety["unexpected_xid"], "safety_probe_invalid"), _boolean(safety["host_oom"], "safety_probe_invalid"), _boolean(safety["container_oom_killed"], "safety_probe_invalid"), _integer(safety["unexpected_restart_count"], "safety_probe_invalid")),
        identities=preflight.IdentityState(
            _digest(identities["current_cas_manifest_sha256"], "identity_probe_invalid"),
            config.expected_current_cas_manifest_sha256,
            _digest(identities["rollback_cas_manifest_sha256"], "identity_probe_invalid"),
            config.expected_rollback_cas_manifest_sha256,
            _boolean(identities["caddy_bind_source_match"], "identity_probe_invalid"),
            _digest(identities["caddy_bytes_sha256"], "identity_probe_invalid"),
            config.expected_caddy_bytes_sha256,
            _digest(identities["caddy_auth_identity_sha256"], "identity_probe_invalid"),
            config.expected_caddy_auth_identity_sha256,
            _boolean(identities["private_routes_match"], "identity_probe_invalid"),
            _boolean(identities["qwen_runtime_identity_match"], "identity_probe_invalid"),
        ),
        admission=preflight.AdmissionState(_text(admission["phase"], "admission_probe_invalid", limit=64), _text(admission["capacity"], "admission_probe_invalid", limit=64), _integer(admission["unfinished_operation_count"], "admission_probe_invalid"), _boolean(admission["reconciliation_pending"], "admission_probe_invalid"), _boolean(admission["breaker_open"], "admission_probe_invalid"), _integer(admission["fifo_active_count"], "admission_probe_invalid"), _integer(admission["fifo_owner_count"], "admission_probe_invalid"), _boolean(admission["clean_drain_possible"], "admission_probe_invalid")),
        stale=preflight.StaleState(_integer(stale["candidate_container_count"], "stale_probe_invalid"), _integer(stale["candidate_mount_count"], "stale_probe_invalid"), _integer(stale["candidate_gpu_pid_count"], "stale_probe_invalid"), _boolean(stale["host_lock_present"], "stale_probe_invalid"), _boolean(stale["unfinished_journal_present"], "stale_probe_invalid")),
        acceptance=config.acceptance,
        acceptance_receipts=dict(config.acceptance_receipts),
        writable_roots_exact=_boolean(root["writable_roots_exact"], "storage_probe_invalid"),
        production_cas_read_only=_boolean(root["production_cas_read_only"], "storage_probe_invalid"),
    )


def collect_snapshot(
    config: CollectorConfig, runner: CommandRunner | None = None
) -> preflight.PreflightSnapshot:
    """Collect exactly one bounded, read-only Server A snapshot."""

    _validate_config(config)
    probe_argv = _probe_argv(config.package_root)
    command_runner = runner or SubprocessCommandRunner(config.package_root)
    try:
        result = command_runner.run(
            probe_argv, timeout=float(config.timeout_seconds), stdin=_request(config)
        )
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        raise CollectorError("probe_timeout") from exc
    except CollectorError:
        raise
    except Exception as exc:
        raise CollectorError("probe_execution_failed") from exc
    if not isinstance(result, CommandResult):
        _fail("probe_result_invalid")
    if result.returncode != 0:
        _fail("probe_failed")
    if not isinstance(result.stdout, bytes) or len(result.stdout) > MAX_OUTPUT_BYTES:
        _fail("probe_output_invalid")
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectorError("probe_output_invalid") from exc
    return _parse_snapshot(config, payload)


def _local_command(argv: tuple[str, ...], *, allowed: tuple[str, ...] | None = None) -> bytes:
    if not argv or argv[0] not in {JOURNALCTL, NVIDIA_SMI, NVIDIA_CTK, DOCKER, SYSTEMCTL}:
        _fail("local_probe_argv_invalid")
    if allowed is not None and argv != allowed:
        _fail("local_probe_argv_invalid")
    try:
        done = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            shell=False, timeout=_LOCAL_TIMEOUT,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        _fail("local_probe_timeout")
    if (
        done.returncode == 1
        and argv == _journal_fault_argv()
        and done.stdout == b""
        and done.stderr == b""
    ):
        return b""
    if done.returncode != 0 or len(done.stdout) > _LOCAL_MAX:
        _fail("local_probe_failed")
    return done.stdout


def _read_bytes(path: str, limit: int = _LOCAL_MAX) -> bytes:
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            _fail("local_file_invalid")
        with open(path, "rb") as handle:
            value = handle.read(limit + 1)
    except OSError:
        _fail("local_file_unavailable")
    if len(value) > limit:
        _fail("local_file_too_large")
    return value


def _json_bytes(value: bytes, code: str) -> Any:
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)


def _operator_json(raw: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("operator_status_invalid")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Any:
        _fail("operator_status_invalid")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail("operator_status_invalid")
    if not isinstance(value, dict):
        _fail("operator_status_invalid")
    return value


def _operator_exchange(
    request: Mapping[str, Any], *, socket_factory: Any | None = None
) -> dict[str, Any]:
    try:
        frame = (
            json.dumps(
                dict(request), sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError):
        _fail("operator_status_invalid")
    client = None
    try:
        factory = socket.socket if socket_factory is None else socket_factory
        af_unix = getattr(socket, "AF_UNIX", 1 if socket_factory is not None else None)
        sock_stream = getattr(socket, "SOCK_STREAM", 1 if socket_factory is not None else None)
        if not isinstance(af_unix, int) or not isinstance(sock_stream, int):
            _fail("operator_status_unavailable")
        client = factory(af_unix, sock_stream)
        client.settimeout(10.0)
        client.connect(OPERATOR_SOCKET)
        client.sendall(frame)
        raw = bytearray()
        while True:
            chunk = client.recv(4096)
            if not isinstance(chunk, bytes) or not chunk:
                _fail("operator_status_incomplete")
            raw.extend(chunk)
            if len(raw) > _OPERATOR_MAX:
                _fail("operator_status_too_large")
            newline = raw.find(b"\n")
            if newline >= 0:
                if newline != len(raw) - 1 or raw.count(b"\n") != 1:
                    _fail("operator_status_invalid")
                break
    except CollectorError:
        raise
    except (OSError, TimeoutError):
        _fail("operator_status_unavailable")
    finally:
        if client is not None:
            client.close()
    return _operator_json(bytes(raw[:-1]))


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("operator_status_invalid")
    return value


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        _fail("operator_status_invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        _fail("operator_status_invalid")
    if parsed.int == 0 or str(parsed) != value:
        _fail("operator_status_invalid")
    return value


def _opaque(value: object) -> str:
    if not isinstance(value, str) or _OPAQUE.fullmatch(value) is None:
        _fail("operator_status_invalid")
    return value


def _validate_active_ledger(value: object, queue_depth: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 21:
        _fail("operator_status_invalid")
    records: list[dict[str, Any]] = []
    turns: set[int] = set()
    jobs: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != _ACTIVE_LEDGER_KEYS:
            _fail("operator_status_invalid")
        for key in (
            "boot_id", "coordinator_generation", "client_service",
            "client_generation", "job_id", "lane", "profile", "state",
        ):
            _opaque(item[key])
        _digest(item["idempotency_digest"], "operator_status_invalid")
        for key in ("admission_epoch", "turn_seq", "queue_depth"):
            if _nonnegative_integer(item[key]) < (1 if key != "queue_depth" else 0):
                _fail("operator_status_invalid")
        if item["queue_depth"] != queue_depth:
            _fail("operator_status_invalid")
        position = item["queue_position"]
        if position is not None and _nonnegative_integer(position) > queue_depth:
            _fail("operator_status_invalid")
        for key in ("enqueue_monotonic", "confirm_deadline", "queue_deadline"):
            number = item[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                _fail("operator_status_invalid")
        if type(item["grant_consumed"]) is not bool:
            _fail("operator_status_invalid")
        grant = item["dispatch_grant_digest"]
        if grant is not None:
            _digest(grant, "operator_status_invalid")
        for key in ("operation_id", "cancel_origin", "terminal_outcome", "terminal_reason"):
            if item[key] is not None:
                _opaque(item[key])
        turn = item["turn_seq"]
        job = item["job_id"]
        if turn in turns or job in jobs:
            _fail("operator_status_invalid")
        turns.add(turn)
        jobs.add(job)
        records.append(item)
    return records


def _admission_operator_status(*, socket_factory: Any | None = None) -> dict[str, Any]:
    request_id = f"host-probe-{os.getpid()}-{time.monotonic_ns()}"
    request = {
        "schema": _ADMISSION_OPERATOR_REQUEST_SCHEMA,
        "verb": "status",
        "request_id": request_id,
    }
    value = _operator_exchange(request, socket_factory=socket_factory)
    if set(value) != _ADMISSION_STATUS_KEYS:
        _fail("operator_status_invalid")
    if (
        value["schema"] != _ADMISSION_OPERATOR_RESPONSE_SCHEMA
        or value["ok"] is not True
        or value["request_id"] != request_id
    ):
        _fail("operator_status_invalid")
    _nonnegative_integer(value["admission_epoch"])
    if value["phase"] not in _ADMISSION_PHASES or value["resume_phase"] not in _ADMISSION_PHASES:
        _fail("operator_status_invalid")
    if value["capacity_mode"] not in _CAPACITY_MODES:
        _fail("operator_status_invalid")
    _canonical_uuid(value["boot_id"])
    _opaque(value["coordinator_generation"])
    if type(value["restart_pending"]) is not bool or type(value["reconciliation_required"]) is not bool:
        _fail("operator_status_invalid")
    queue_depth = _nonnegative_integer(value["queue_depth"])
    records = _validate_active_ledger(value["active_ledger"], queue_depth)
    if queue_depth > len(records):
        _fail("operator_status_invalid")
    digest_records = [
        {
            key: item[key]
            for key in sorted(_ACTIVE_LEDGER_KEYS - {"queue_position", "queue_depth"})
        }
        for item in sorted(records, key=lambda item: item["turn_seq"])
    ]
    expected_digest = hashlib.sha256(
        json.dumps(
            digest_records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if _digest(value["active_ledger_digest"], "operator_status_invalid") != expected_digest:
        _fail("operator_status_invalid")
    return value


def _supervisor_operator_status(*, socket_factory: Any | None = None) -> dict[str, Any]:
    value = _operator_exchange(
        {"profile": "qwen", "verb": "status"}, socket_factory=socket_factory
    )
    if set(value) != _SUPERVISOR_STATUS_KEYS:
        _fail("operator_status_invalid")
    if value["ok"] is not True or value["status_schema"] != _SUPERVISOR_STATUS_SCHEMA or value["profile"] != "qwen":
        _fail("operator_status_invalid")
    lanes = value["active_by_lane"]
    if not isinstance(lanes, dict) or set(lanes) != {"text", "image"}:
        _fail("operator_status_invalid")
    lane_profiles = {
        "text": {None, "qwen", "qwen-dflash2"},
        "image": {None, "pony-v6", "pony-v7"},
    }
    if any(lanes[lane] not in allowed for lane, allowed in lane_profiles.items()):
        _fail("operator_status_invalid")
    active_profiles = [profile for profile in lanes.values() if profile is not None]
    if (
        value["active_profiles"] != sorted(active_profiles)
        or value["active_profile"] != (active_profiles[0] if len(active_profiles) == 1 else None)
    ):
        _fail("operator_status_invalid")
    if type(value["shared_mode_enabled"]) is not bool or value["capacity_mode"] not in _CAPACITY_MODES:
        _fail("operator_status_invalid")
    if value["shared_mode_enabled"] != (value["capacity_mode"] in {"shared-48", "shared-64"}):
        _fail("operator_status_invalid")
    _nonnegative_integer(value["admission_epoch"])
    if value["admission_phase"] not in _ADMISSION_PHASES:
        _fail("operator_status_invalid")
    _canonical_uuid(value["boot_id"])
    if type(value["global_breaker_open"]) is not bool:
        _fail("operator_status_invalid")
    if value["last_operation"] is not None and not isinstance(value["last_operation"], dict):
        _fail("operator_status_invalid")
    if value["last_global_health"] is not None and not isinstance(value["last_global_health"], dict):
        _fail("operator_status_invalid")
    status = value["status"]
    if not isinstance(status, dict) or set(status) != _PROFILE_STATE_KEYS:
        _fail("operator_status_invalid")
    if status["state"] not in _PROFILE_STATES or type(status["breaker_open"]) is not bool:
        _fail("operator_status_invalid")
    if status["idle_timeout_seconds"] != 1800:
        _fail("operator_status_invalid")
    _nonnegative_integer(status["recovery_attempts"])
    transitioned = status["last_transition_at"]
    if transitioned is not None and (
        isinstance(transitioned, bool)
        or not isinstance(transitioned, (int, float))
        or not math.isfinite(float(transitioned))
    ):
        _fail("operator_status_invalid")
    if status["last_diagnostics"] is not None and not isinstance(status["last_diagnostics"], dict):
        _fail("operator_status_invalid")
    if not isinstance(status["diagnostics"], dict):
        _fail("operator_status_invalid")
    return value


def _operator_status() -> tuple[dict[str, Any], dict[str, Any]]:
    admission = _admission_operator_status()
    supervisor = _supervisor_operator_status()
    if (
        supervisor["admission_epoch"] != admission["admission_epoch"]
        or supervisor["admission_phase"] != admission["phase"]
        or supervisor["capacity_mode"] != admission["capacity_mode"]
        or supervisor["boot_id"] != admission["boot_id"]
    ):
        _fail("operator_status_inconsistent")
    return admission, supervisor


def _admission_observation(
    admission: Mapping[str, Any],
    supervisor: Mapping[str, Any],
    *,
    unfinished_operation_count: int,
) -> tuple[dict[str, Any], bool]:
    qwen_status = supervisor["status"]
    breaker_open = (
        supervisor["global_breaker_open"] or qwen_status["breaker_open"]
    )
    observation = {
        "phase": admission["phase"],
        "capacity": admission["capacity_mode"],
        "unfinished_operation_count": unfinished_operation_count,
        "reconciliation_pending": (
            admission["restart_pending"] or admission["reconciliation_required"]
        ),
        "breaker_open": breaker_open,
        "fifo_active_count": len(admission["active_ledger"]),
        "fifo_owner_count": admission["queue_depth"],
        "clean_drain_possible": admission["phase"] == "public",
    }
    qwen_active = (
        supervisor["active_by_lane"]["text"] == "qwen"
        and qwen_status["state"] in {"ready", "busy", "idle"}
    )
    return observation, qwen_active


def _request_value() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        _fail("probe_request_too_large")
    value = _json_bytes(raw, "probe_request_invalid")
    keys = {"schema", "candidate_id", "compose_files", "systemd_units", "socket_names", "required_mem_available_mib", "calculated_candidate_bytes", "restore_reserve_bytes"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != "ai-server.acceleration-host-probe-request.v1":
        _fail("probe_request_invalid")
    if not isinstance(value.get("candidate_id"), str) or _SAFE_ID.fullmatch(value["candidate_id"]) is None:
        _fail("probe_request_invalid")
    for key in ("compose_files", "systemd_units", "socket_names"):
        if not isinstance(value.get(key), list) or not value[key] or any(not isinstance(item, str) for item in value[key]):
            _fail("probe_request_invalid")
    package_root = _package_root(Path(__file__).resolve(strict=True).parent.parent)
    if tuple(value["compose_files"]) != candidate_compose_files(
        package_root, value["candidate_id"]
    ):
        _fail("probe_request_invalid")
    if any(_SAFE_ID.fullmatch(name) is None for name in value["systemd_units"]):
        _fail("probe_request_invalid")
    if any(name not in SOCKET_PATHS for name in value["socket_names"]):
        _fail("probe_request_invalid")
    return value


def _release_digest(name: str) -> str:
    root = Path(RELEASES_ROOT) / name
    ready = _json_bytes(_read_bytes(str(root / "RELEASE_READY.json"), 16384), "release_marker_invalid")
    if not isinstance(ready, dict) or ready.get("schema_version") != "ai-server-artifact-release/v1" or ready.get("manifest_id") != name:
        _fail("release_marker_invalid")
    digest = ready.get("manifest_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        _fail("release_marker_invalid")
    manifest = _read_bytes(str(root / "manifest.json"))
    if hashlib.sha256(manifest).hexdigest() != digest:
        _fail("release_manifest_mismatch")
    return digest


def _compose_observation(paths: list[str]) -> tuple[dict[str, Any], bool]:
    exact_environment = restart_no = loopback_only = True
    rc = 0
    for path in paths:
        argv = (DOCKER, "compose", "-f", path, "config", "--format", "json")
        raw = _local_command(argv)
        value = _json_bytes(raw, "compose_probe_invalid")
        services = value.get("services") if isinstance(value, dict) else None
        if not isinstance(services, dict) or not services:
            _fail("compose_probe_invalid")
        for service in services.values():
            if not isinstance(service, dict):
                _fail("compose_probe_invalid")
            restart_no = restart_no and service.get("restart", "no") == "no"
            for port in service.get("ports", []):
                if not isinstance(port, dict) or port.get("host_ip") not in {None, "127.0.0.1", "::1"}:
                    loopback_only = False
            for mount in service.get("volumes", []):
                if isinstance(mount, dict) and mount.get("source") == MODEL_STORE_ROOT and mount.get("read_only") is not True:
                    exact_environment = False
    return {"returncode": rc, "timed_out": False, "exact_environment": exact_environment, "restart_policy_no": restart_no, "loopback_only": loopback_only}, exact_environment


def _host_lock_present() -> bool:
    """Return whether the exact experiment host lock is held by another owner.

    This runs before :class:`CandidateLiveOperation` acquires the lock.  It must
    not create the lock file: an absent or unsafe lock boundary is not evidence
    that no operation owns Server A, so the read-only preflight fails closed.
    """

    if os.name != "posix" or fcntl is None:
        _fail("host_lock_probe_unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(HOST_LOCK, flags)
    except FileNotFoundError:
        _fail("host_lock_probe_missing")
    except OSError:
        _fail("host_lock_probe_unavailable")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            _fail("host_lock_probe_invalid")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            _fail("host_lock_probe_unavailable")
        try:
            return False
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _self_hosted_payload(request: dict[str, Any]) -> dict[str, Any]:
    boot_id = _read_bytes("/proc/sys/kernel/random/boot_id", 128).decode("ascii").strip()
    cursor_raw = _local_command((JOURNALCTL, "--show-cursor", "-n", "0", "--no-pager"))
    cursor_lines = [line.removeprefix("-- cursor: ") for line in cursor_raw.decode("utf-8", "strict").splitlines() if line.startswith("-- cursor: ")]
    if len(cursor_lines) != 1:
        _fail("journal_cursor_invalid")
    gpu_raw = _local_command((NVIDIA_SMI, "--query-gpu=driver_version,power.limit,temperature.gpu,memory.free", "--format=csv,noheader,nounits"))
    gpu_lines = [line for line in gpu_raw.decode("ascii", "strict").splitlines() if line.strip()]
    if len(gpu_lines) != 1 or len([part.strip() for part in gpu_lines[0].split(",")]) != 4:
        _fail("gpu_probe_invalid")
    driver, power, temperature, vram = [part.strip() for part in gpu_lines[0].split(",")]
    apps_raw = _local_command((NVIDIA_SMI, "--query-compute-apps=pid", "--format=csv,noheader,nounits"))
    compute_pids = [int(line.strip()) for line in apps_raw.decode("ascii", "strict").splitlines() if line.strip()]
    compute_count = len(compute_pids)
    cdi = _local_command((NVIDIA_CTK, "cdi", "list")) .decode("utf-8", "strict")
    docker_info = _json_bytes(_local_command((DOCKER, "info", "--format", "{{json .}}")), "docker_info_invalid")
    compose_version = _local_command((DOCKER, "compose", "version", "--short")).decode("ascii", "strict").strip()
    compose, storage_exact = _compose_observation(request["compose_files"])

    fixture = subprocess.run((DOCKER, "inspect", FIXTURE_CONTAINER), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, check=False, timeout=_LOCAL_TIMEOUT, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
    if len(fixture.stdout) > _LOCAL_MAX:
        _fail("docker_fixture_invalid")
    fixture_state = "not_found"
    fixture_rc = 0
    if fixture.returncode == 0:
        values = _json_bytes(fixture.stdout, "docker_fixture_invalid")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            _fail("docker_fixture_invalid")
        fixture_state = values[0].get("State", {}).get("Status")

    systemd_states: dict[str, Any] = {}
    for unit in request["systemd_units"]:
        raw = _local_command((SYSTEMCTL, "show", unit, "--property=ActiveState", "--property=SubState", "--property=MainPID"))
        fields = dict(line.split("=", 1) for line in raw.decode("ascii", "strict").splitlines() if "=" in line)
        if set(fields) != {"ActiveState", "SubState", "MainPID"}:
            _fail("systemd_probe_invalid")
        systemd_states[unit] = {"active_state": fields["ActiveState"], "sub_state": fields["SubState"], "main_pid": int(fields["MainPID"]), "parse_ok": True}

    socket_states: dict[str, Any] = {}
    for name in request["socket_names"]:
        path = SOCKET_PATHS[name]
        info = os.lstat(path)
        socket_states[name] = {"path_exact": path == OPERATOR_SOCKET, "is_socket": stat.S_ISSOCK(info.st_mode), "is_symlink": stat.S_ISLNK(info.st_mode), "owner_exact": info.st_uid == 0, "group_exact": info.st_gid == 0, "mode_exact": stat.S_IMODE(info.st_mode) == 0o600}

    meminfo = _read_bytes("/proc/meminfo", 65536).decode("ascii", "strict")
    match = re.search(r"^MemAvailable:\s+([0-9]+) kB$", meminfo, re.MULTILINE)
    if match is None:
        _fail("meminfo_invalid")
    vm_before = _read_bytes("/proc/vmstat", 65536).decode("ascii", "strict")
    time.sleep(0.1)
    vm_after = _read_bytes("/proc/vmstat", 65536).decode("ascii", "strict")
    def counter(text: str, name: str) -> int:
        found = re.search(rf"^{name} ([0-9]+)$", text, re.MULTILINE)
        if found is None: _fail("vmstat_invalid")
        return int(found.group(1))
    swap_in = counter(vm_after, "pswpin") - counter(vm_before, "pswpin")
    swap_out = counter(vm_after, "pswpout") - counter(vm_before, "pswpout")
    disk = os.statvfs("/srv/ai-server/staging")

    admission_status, supervisor_status = _operator_status()

    inspect = _json_bytes(_local_command((DOCKER, "inspect", *PRODUCTION_CONTAINERS)), "container_inspect_invalid")
    if not isinstance(inspect, list) or len(inspect) != len(PRODUCTION_CONTAINERS):
        _fail("container_inspect_invalid")
    edge = next((item for item in inspect if item.get("Name") == "/ai-server-edge"), None)
    qwen = next((item for item in inspect if item.get("Name") == "/ai-server-qwen"), None)
    if not isinstance(edge, dict) or not isinstance(qwen, dict):
        _fail("container_inspect_invalid")
    if len(compute_pids) == 1:
        container_id = qwen.get("Id")
        if not isinstance(container_id, str) or len(container_id) < 12:
            _fail("container_inspect_invalid")
        try:
            cgroup = _read_bytes(f"/proc/{compute_pids[0]}/cgroup", 65536)
        except CollectorError:
            compute_count = 2
        else:
            if container_id[:12].encode("ascii") not in cgroup and container_id.encode("ascii") not in cgroup:
                compute_count = 2
    mounts = edge.get("Mounts", [])
    caddy_bind = any(isinstance(m, dict) and m.get("Source") == CADDY_SOURCE and m.get("Destination") == CADDY_TARGET and m.get("RW") is False for m in mounts)
    caddy = _local_command((DOCKER, "exec", "ai-server-edge", "cat", CADDY_TARGET))
    auth_digest = _caddy_auth_identity(caddy)
    private_routes = all(listener not in caddy for listener in (b"0.0.0.0:1919", b"0.0.0.0:18020", b"0.0.0.0:18021"))
    restart_count = sum(int(item.get("RestartCount", 0)) for item in inspect if isinstance(item, dict))
    oom = any(item.get("State", {}).get("OOMKilled") is True for item in inspect if isinstance(item, dict))
    candidate_rows = _local_command((DOCKER, "ps", "-a", "--filter", "name=ai-exp-", "--format", "{{.ID}}"))
    candidate_count = len([line for line in candidate_rows.splitlines() if line.strip()])
    operation_root = Path("/srv/ai-server/state/acceleration-experiments/operations")
    unfinished = sum(1 for entry in operation_root.iterdir() if entry.is_dir() and not (entry / "receipt.json").is_file()) if operation_root.exists() else 0
    admission_observation, active = _admission_observation(
        admission_status,
        supervisor_status,
        unfinished_operation_count=unfinished,
    )
    journal_faults = _local_command(_journal_fault_argv())
    unexpected_xid, host_oom = _journal_runtime_faults(journal_faults)

    return {
        "schema": "ai-server.acceleration-host-probe.v1", "boot_id": boot_id,
        "journal_cursor": cursor_lines[0], "driver_version": driver,
        "cdi_gpu_all": "nvidia.com/gpu=all" in cdi, "cdi_gpu_zero": "nvidia.com/gpu=0" in cdi,
        "cuda13_compatible": int(driver.split(".", 1)[0]) >= 580,
        "toolkit_available": bool(cdi.strip()), "docker_version_supported": isinstance(docker_info, dict),
        "compose_version_supported": bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", compose_version)),
        "docker_contract": {"command_returncode": fixture_rc, "inspect_state": fixture_state, "gpu_compute_pid_count": 0, "timed_out": False},
        "compose_render": compose, "systemd_states": systemd_states, "socket_states": socket_states,
        "resources": {"power_limit_w": float(power), "gpu_compute_pid_count": compute_count, "mem_available_mib": int(match.group(1)) / 1024, "swap_in_delta": swap_in, "swap_out_delta": swap_out, "disk_available_bytes": disk.f_bavail * disk.f_frsize, "temperature_c": float(temperature), "vram_free_mib": float(vram)},
        "runtime_safety": {"unexpected_xid": unexpected_xid, "host_oom": host_oom, "container_oom_killed": oom, "unexpected_restart_count": restart_count},
        "identities": {"current_cas_manifest_sha256": _release_digest(CURRENT_RELEASE), "rollback_cas_manifest_sha256": _release_digest(ROLLBACK_RELEASE), "caddy_bind_source_match": caddy_bind, "caddy_bytes_sha256": hashlib.sha256(caddy).hexdigest(), "caddy_auth_identity_sha256": auth_digest, "private_routes_match": private_routes, "qwen_runtime_identity_match": active and qwen.get("State", {}).get("Running") is True},
        "admission": admission_observation,
        "stale": {"candidate_container_count": candidate_count, "candidate_mount_count": 0, "candidate_gpu_pid_count": 0, "host_lock_present": _host_lock_present(), "unfinished_journal_present": unfinished > 0},
        "writable_roots_exact": storage_exact, "production_cas_read_only": storage_exact,
    }


def _main(argv: Sequence[str]) -> int:
    if tuple(argv) != ("--snapshot-v1",):
        return 2
    try:
        payload = _self_hosted_payload(_request_value())
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        if len(encoded) > MAX_OUTPUT_BYTES:
            return 1
        sys.stdout.buffer.write(encoded)
        return 0
    except Exception:
        return 1


__all__ = [
    "CollectorConfig", "CollectorError", "CommandResult", "CommandRunner",
    "SubprocessCommandRunner", "candidate_compose_files", "collect_snapshot",
]


if __name__ == "__main__":  # pragma: no cover - exercised on installed Server A path
    raise SystemExit(_main(sys.argv[1:]))
