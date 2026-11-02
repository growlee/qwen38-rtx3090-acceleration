"""Concrete, fail-closed Server A backend for ``CandidateLiveOperation``.

This is deliberately not a command launcher.  Candidate identities, source
pins, Compose paths, project names, services, containers, sockets and
endpoints are constants.  The adapter never accepts caller-provided argv,
environment, URL or filesystem paths.

The benchmark/quality/safety phase is delegated to one sealed arm harness.  A
package which does not contain that reviewed harness fails before a candidate
is started with ``candidate_arm_harness_missing``; this module does not guess a
replacement command contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from .candidate_artifact_custody import (
        CustodyError,
        CustodyLockGuard,
        custody_locks,
        write_immutable_receipt,
    )
    from .candidate_artifact_prepare import PrepareError, prepare_candidate_artifacts
    from .candidate_image_loader import ImageLoadError, ensure_embedded_image_loaded
    from .candidate_contract import ContractError, validate_candidate_contract
    from .candidate_host_collector import (
        CADDY_SOURCE,
        CADDY_TARGET,
        _caddy_auth_identity,
        CollectorConfig,
        CollectorError,
        CommandRunner as HostCommandRunner,
        collect_snapshot,
    )
    from .candidate_live_operation import LiveOperationError
    from .candidate_route_evidence import RouteEvidenceError
    from .host_compatibility_preflight import PreflightError, evaluate_snapshot
    from . import safety_sampler
except ImportError:  # pragma: no cover - direct execution from experiments/
    from candidate_artifact_custody import (  # type: ignore[no-redef]
        CustodyError,
        CustodyLockGuard,
        custody_locks,
        write_immutable_receipt,
    )
    from candidate_artifact_prepare import PrepareError, prepare_candidate_artifacts  # type: ignore[no-redef]
    from candidate_image_loader import ImageLoadError, ensure_embedded_image_loaded  # type: ignore[no-redef]
    from candidate_contract import ContractError, validate_candidate_contract  # type: ignore[no-redef]
    from candidate_host_collector import (  # type: ignore[no-redef]
        CADDY_SOURCE,
        CADDY_TARGET,
        _caddy_auth_identity,
        CollectorConfig,
        CollectorError,
        CommandRunner as HostCommandRunner,
        collect_snapshot,
    )
    from candidate_live_operation import LiveOperationError  # type: ignore[no-redef]
    from candidate_route_evidence import RouteEvidenceError  # type: ignore[no-redef]
    from host_compatibility_preflight import PreflightError, evaluate_snapshot  # type: ignore[no-redef]
    import safety_sampler  # type: ignore[no-redef]

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
import freetoken_bandwidth_runner as bandwidth_runner  # noqa: E402


DOCKER = "/usr/bin/docker"
PYTHON = "/usr/bin/python3"
GATEWAY_PYTHON = "/usr/local/bin/python3"
NVIDIA_SMI = "/usr/bin/nvidia-smi"
EXPERIMENTS_ROOT = Path("/opt/homelab/ai-server/experiments")
STAGING_ROOT = Path("/srv/ai-server/staging/experiments")
STATE_ROOT = Path("/srv/ai-server/state/acceleration-experiments")
PRODUCTION_CAS_ROOT = Path("/srv/ai-server/model-store")
HOST_LOCK = STATE_ROOT / "locks" / "host.lock"
PREFLIGHT_RECEIPT_ROOT = STATE_ROOT / "preflight-receipts"
SUPERVISOR_SOCKET = Path("/run/ai-server/supervisor/operator/supervisor.sock")
# The reviewed root operator socket multiplexes supervisor and admission
# request schemas; there is no separate admission socket.
ADMISSION_SOCKET = SUPERVISOR_SOCKET
ARM_HARNESS = EXPERIMENTS_ROOT / "candidate_arm_harness.py"
BANDWIDTH_SUITE = EXPERIMENTS_ROOT.parent / "benchmarks" / "suites" / "acceleration-freetoken-bandwidth-v1.json"
BANDWIDTH_CONTAINER = "ai-exp-freetoken-bench"
BANDWIDTH_PROFILE_RELATIVE = Path("cache/.cache/freetoken/benchbw.json")

QWEN_CONTAINER = "ai-server-qwen"
GATEWAY_CONTAINER = "ai-server-gateway"
OPENWEBUI_CONTAINER = "ai-server-open-webui"
EDGE_CONTAINER = "ai-server-edge"
PONY_V6_CONTAINER = "ai-server-pony-v6"
PONY_V7_CONTAINER = "ai-server-pony-v7"
PONY_UI_CONTAINER = "ai-server-pony-ui"
PONY_ORCHESTRATOR_CONTAINER = "ai-server-pony-orchestrator"
IMAGE_FACADE_CONTAINER = "ai-server-image-facade"
STATUS_PAGE_CONTAINER = "ai-server-status-page"
COMFY_ADMIN_GATE_CONTAINER = "ai-server-comfy-admin-gate"
PRODUCTION_CONTAINERS = (
    QWEN_CONTAINER,
    GATEWAY_CONTAINER,
    OPENWEBUI_CONTAINER,
    EDGE_CONTAINER,
    PONY_V6_CONTAINER,
    PONY_V7_CONTAINER,
    PONY_UI_CONTAINER,
    PONY_ORCHESTRATOR_CONTAINER,
    IMAGE_FACADE_CONTAINER,
    STATUS_PAGE_CONTAINER,
    COMFY_ADMIN_GATE_CONTAINER,
)
ALWAYS_RUNNING_PRODUCTION_CONTAINERS = frozenset({
    QWEN_CONTAINER,
    GATEWAY_CONTAINER,
    OPENWEBUI_CONTAINER,
    EDGE_CONTAINER,
    PONY_UI_CONTAINER,
    IMAGE_FACADE_CONTAINER,
    STATUS_PAGE_CONTAINER,
    COMFY_ADMIN_GATE_CONTAINER,
})
QWEN_STOPPED_GPU_COMPANION_CONTAINERS = frozenset({
    PONY_V6_CONTAINER,
    PONY_V7_CONTAINER,
    PONY_ORCHESTRATOR_CONTAINER,
})
PRODUCTION_FILES = (
    Path("/opt/homelab/ai-server/deploy/compose.qwen.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.gateway.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.ui.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.pony-v6.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.pony-v6.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.pony-v7.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.pony-v7.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.pony-ui.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.pony-orchestrator.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.image-facade.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.status-page.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.comfy-admin-gate.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.comfy-admin-gate.serialized-full-gpu.candidate.yaml"),
    Path("/opt/homelab/ai-server/deploy/compose.edge.yaml"),
    Path(CADDY_SOURCE),
)
CADDY_DESTINATION = CADDY_TARGET
POWER_LIMIT_W = 300.0
EMPTY_LEDGER_SHA256 = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SYV_IMAGE_REFERENCE = (
    "ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@"
    "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
)
SYV_REPO_DIGEST = (
    "ghcr.io/syv-ai/qwen38-27b-rtx3090@"
    "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
)
SYV_MANIFEST_DIGEST = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
SYV_CONFIG_DIGEST = "sha256:b0516eab4a689391cd73629593d53111777691c5f3c453d5064e77fe55981101"
SYV_IMAGE_LABELS = {
    "org.opencontainers.image.created": "2026-08-27T08:52:29.474Z",
    "org.opencontainers.image.licenses": "Apache-2.0",
    "org.opencontainers.image.revision": "69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c",
    "org.opencontainers.image.source": "https://github.com/syv-ai/qwen38-27b-rtx3090",
    "org.opencontainers.image.title": "qwen38-27b-rtx3090",
}


@dataclass(frozen=True)
class ArmContract:
    directory: str
    compose: str
    project: str
    service: str
    container: str
    endpoint: str
    ready_timeout: int


_PINS = {
    "qwen38-vllm": "69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c",
    "freetoken": "9db1a39455a3fb107f3db83e381d10ceadfe5d99",
    "llama-dflash2-official": "b10f9ca58c89ccfc3653ac01e979dd085d582b76",
}
_MODE_NAMES = {
    ("qwen38-vllm", "mtp"): "mtp",
    ("qwen38-vllm", "dflash2"): "dflash2",
    ("freetoken", "freetoken"): "serve",
    ("llama-dflash2-official", "plain"): "plain",
    ("llama-dflash2-official", "mtp"): "mtp",
    ("llama-dflash2-official", "dflash2-n4"): "dflash2-n4",
    ("llama-dflash2-official", "dflash2-n7"): "dflash2-n7",
}
_ARMS = {
    ("qwen38-vllm", "mtp"): ArmContract("qwen38_vllm", "compose.mtp.yaml", "ai-exp-qwen38-vllm-mtp", "qwen38-vllm-mtp", "ai-exp-qwen38-vllm-mtp", "http://127.0.0.1:18020", 900),
    ("qwen38-vllm", "dflash2"): ArmContract("qwen38_vllm", "compose.dflash2.yaml", "ai-exp-qwen38-vllm-dflash2", "qwen38-vllm-dflash2", "ai-exp-qwen38-vllm-dflash2", "http://127.0.0.1:18020", 900),
    ("freetoken", "freetoken"): ArmContract("freetoken", "compose.yaml", "ai-exp-freetoken-serve", "freetoken-serve", "ai-exp-freetoken-serve", "http://127.0.0.1:1919", 900),
    ("llama-dflash2-official", "plain"): ArmContract("llama_dflash2_official", "compose.plain.yaml", "ai-exp-llama-dflash2-plain", "llama-dflash2-plain", "ai-exp-llama-dflash2-plain", "http://127.0.0.1:18021", 600),
    ("llama-dflash2-official", "mtp"): ArmContract("llama_dflash2_official", "compose.mtp.yaml", "ai-exp-llama-dflash2-mtp", "llama-dflash2-mtp", "ai-exp-llama-dflash2-mtp", "http://127.0.0.1:18021", 600),
    ("llama-dflash2-official", "dflash2-n4"): ArmContract("llama_dflash2_official", "compose.dflash2.yaml", "ai-exp-llama-dflash2-dflash2", "llama-dflash2-n4", "ai-exp-llama-dflash2-n4", "http://127.0.0.1:18021", 600),
    ("llama-dflash2-official", "dflash2-n7"): ArmContract("llama_dflash2_official", "compose.dflash2.yaml", "ai-exp-llama-dflash2-dflash2", "llama-dflash2-n7", "ai-exp-llama-dflash2-n7", "http://127.0.0.1:18021", 600),
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None) -> CommandResult: ...


class SubprocessRunner:
    """Shell-free runner preserving rc/stdout/stderr as separate values."""

    def run(self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None) -> CommandResult:
        completed = subprocess.run(
            tuple(argv), input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False, shell=False,
        )
        return CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )


class _BandwidthHostProbe:
    """Bounded safety input pinned to one pre-created bandwidth container."""

    def __init__(self, runner: CommandRunner, container_id: str) -> None:
        if SHA256.fullmatch(container_id) is None:
            raise safety_sampler.SafetyError("safety_container_id_invalid")
        self.runner = runner
        self.container_id = container_id
        self.phase = "created"
        self._owned_compute_pid_observed = False
        self._swap: tuple[int, int] | None = None
        self._major_faults: int | None = None
        self._since = datetime.now(timezone.utc).isoformat()

    def mark_running(self) -> None:
        if self.phase != "created":
            raise safety_sampler.SafetyError("safety_container_phase_invalid")
        self.phase = "running"

    def mark_completed(self) -> None:
        if self.phase != "running":
            raise safety_sampler.SafetyError("safety_container_phase_invalid")
        self.phase = "completed"

    def assert_owned_compute_pid_observed(self) -> None:
        if not self._owned_compute_pid_observed:
            raise safety_sampler.SafetyError("gpu_compute_pid_not_observed")

    def _run(self, argv: Sequence[str], deadline: float) -> CommandResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            raise safety_sampler.SafetyError("safety_probe_deadline_exceeded")
        try:
            result = self.runner.run(tuple(argv), timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise safety_sampler.SafetyError("safety_probe_command_failed") from exc
        if (
            not isinstance(result, CommandResult)
            or result.returncode != 0
            or len(result.stdout) + len(result.stderr) > 1_048_576
        ):
            raise safety_sampler.SafetyError("safety_probe_command_failed")
        return result

    @staticmethod
    def _boot_id() -> str:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip().lower()
        except OSError as exc:
            raise safety_sampler.SafetyError("boot_id_unavailable") from exc

    @staticmethod
    def _gpu_compute_pids(raw: str) -> tuple[int, ...]:
        values: list[int] = []
        for row in raw.splitlines():
            value = row.strip()
            if not value:
                continue
            if not value.isascii() or not value.isdigit():
                raise safety_sampler.SafetyError("gpu_compute_pid_shape_invalid")
            pid = int(value)
            if pid <= 0 or pid in values:
                raise safety_sampler.SafetyError("gpu_compute_pid_shape_invalid")
            values.append(pid)
        return tuple(values)

    def _pid_owned(self, pid: int) -> bool:
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
        except OSError as exc:
            raise safety_sampler.SafetyError("gpu_compute_pid_cgroup_unavailable") from exc
        return re.search(
            rf"(?<![0-9a-f]){re.escape(self.container_id)}(?![0-9a-f])", cgroup.lower()
        ) is not None

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
        deadline = time.monotonic() + 0.9
        gpu = self._run((
            NVIDIA_SMI,
            "--query-gpu=temperature.gpu,power.draw,power.limit,memory.free",
            "--format=csv,noheader,nounits",
        ), deadline).stdout
        rows = [row for row in gpu.splitlines() if row.strip()]
        if len(rows) != 1:
            raise safety_sampler.SafetyError("safety_gpu_shape_invalid")
        try:
            temperature, power, cap, free = (float(item.strip()) for item in rows[0].split(","))
        except ValueError as exc:
            raise safety_sampler.SafetyError("safety_gpu_shape_invalid") from exc
        raw = self._run((
            DOCKER, "inspect", self.container_id,
            "--format", "{{json .Id}}|{{json .State}}|{{json .RestartCount}}",
        ), deadline).stdout
        try:
            id_raw, state_raw, restart_raw = raw.strip().split("|", 2)
            observed_id = json.loads(id_raw)
            state = json.loads(state_raw)
            restart = json.loads(restart_raw)
            if (
                observed_id != self.container_id
                or not isinstance(state, Mapping)
                or isinstance(restart, bool)
                or not isinstance(restart, int)
            ):
                raise ValueError
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise safety_sampler.SafetyError("safety_container_shape_invalid") from exc
        phase_valid = {
            "created": state.get("Running") is False and state.get("Status") == "created",
            "running": state.get("Running") is True and state.get("Status") == "running",
            "completed": state.get("Running") is False and state.get("Status") == "exited",
        }.get(self.phase, False)
        compute_raw = self._run((
            NVIDIA_SMI, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ), deadline).stdout
        compute_pids = self._gpu_compute_pids(compute_raw)
        owner_valid = all(self._pid_owned(pid) for pid in compute_pids)
        if self.phase == "running" and compute_pids and owner_valid:
            self._owned_compute_pid_observed = True
        available, pin, pout, major = self._proc_values()
        previous = self._swap
        self._swap = (pin, pout)
        swap_in = 0 if previous is None else pin - previous[0]
        swap_out = 0 if previous is None else pout - previous[1]
        if swap_in < 0 or swap_out < 0:
            raise safety_sampler.SafetyError("safety_swap_counter_invalid")
        major_delta = 0 if self._major_faults is None else major - self._major_faults
        self._major_faults = major
        if major_delta < 0:
            raise safety_sampler.SafetyError("safety_major_fault_counter_invalid")
        kernel = self._run((
            "/usr/bin/journalctl", "-k", "--since", self._since,
            "--no-pager", "--output=cat",
        ), deadline).stdout.lower()
        logs_result = self._run((
            DOCKER, "logs", "--since", self._since, self.container_id,
        ), deadline)
        logs = (logs_result.stdout + logs_result.stderr).lower()
        return safety_sampler.SafetyObservation(
            monotonic_ns=time.monotonic_ns(),
            gpu_temperature_c=temperature,
            gpu_power_draw_w=power,
            power_cap_w=cap,
            vram_free_mib=free,
            ram_available_mib=available,
            swap_in_delta=swap_in,
            swap_out_delta=swap_out,
            nvidia_xid="nvrm: xid" in kernel,
            host_oom="out of memory" in kernel or "oom-killer" in kernel,
            container_oom_killed=state.get("OOMKilled") is True,
            restart_count=restart,
            backend_unavailable_count=int(not phase_valid),
            incomplete_stream_count=0,
            boot_id=self._boot_id(),
            gpu_compute_pid_count=len(compute_pids),
            gpu_compute_pid_owner_valid=owner_valid,
            major_fault_delta=major_delta,
            cuda_oom=(
                "cuda out of memory" in kernel
                or "cuda_error_out_of_memory" in kernel
                or "cuda out of memory" in logs
                or "cuda_error_out_of_memory" in logs
            ),
        )


class JsonSocketClient(Protocol):
    def call(self, request: Mapping[str, Any], *, timeout: float) -> dict[str, Any]: ...


class UnixJsonSocketClient:
    def __init__(self, path: Path, *, admission: bool) -> None:
        expected = ADMISSION_SOCKET if admission else SUPERVISOR_SOCKET
        if path != expected:
            raise ValueError("operator_socket_not_allowlisted")
        self.path = path

    def call(self, request: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        payload = json.dumps(dict(request), sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        if len(payload) > 4096:
            raise LiveOperationError("operator_request_too_large")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(str(self.path))
            client.sendall(payload)
            response = b""
            while b"\n" not in response:
                chunk = client.recv(65536)
                if not chunk:
                    raise LiveOperationError("operator_response_incomplete", ambiguous=True)
                response += chunk
                if len(response) > 1_048_576:
                    raise LiveOperationError("operator_response_too_large", ambiguous=True)
        except LiveOperationError:
            raise
        except (OSError, TimeoutError) as exc:
            raise LiveOperationError("operator_outcome_unknown", ambiguous=True) from exc
        finally:
            client.close()
        try:
            value = json.loads(response.split(b"\n", 1)[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiveOperationError("operator_response_invalid", ambiguous=True) from exc
        if not isinstance(value, dict):
            raise LiveOperationError("operator_response_invalid", ambiguous=True)
        return value


class ProductionProbe(Protocol):
    def completion(self, ordinal: int) -> None: ...


class AuthenticatedRouteProbe(Protocol):
    """Owner-supplied, content-free proof of the three authenticated UI paths."""

    def probe(self) -> Mapping[str, Any]: ...


class GatewayProductionProbe:
    _SCRIPT = b"""import json,sys,urllib.request
ordinal=int(sys.argv[1])
if ordinal not in (1,2): raise SystemExit(20)
marker='QWEN_RESTORE_OK_'+str(ordinal)
token=open('<redacted-secret-path>','r',encoding='ascii').read().strip()
body=json.dumps({'model':'qwen3.8-27b-uncensored','messages':[{'role':'user','content':'Give a complete final answer containing this exact verification token once: '+marker}],'temperature':0,'max_tokens':512,'stream':False},separators=(',',':')).encode()
request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=body,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','Accept':'application/json'},method='POST')
with urllib.request.urlopen(request,timeout=120) as response:
    raw=response.read(1048577)
if len(raw)>1048576: raise SystemExit(21)
value=json.loads(raw)
choices=value.get('choices')
if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict): raise SystemExit(22)
choice=choices[0]
message=choice.get('message')
content=message.get('content') if isinstance(message,dict) else None
if choice.get('finish_reason')!='stop' or not isinstance(content,str) or not content.strip(): raise SystemExit(22)
if content.count(marker)!=1: raise SystemExit(22)
print('QWEN_COMPLETION_OK')
"""

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    def completion(self, ordinal: int) -> None:
        if ordinal not in {1, 2}:
            raise LiveOperationError("completion_ordinal_invalid")
        try:
            result = self.runner.run(
                (DOCKER, "exec", "-i", GATEWAY_CONTAINER, GATEWAY_PYTHON, "-B", "-", str(ordinal)),
                timeout=150,
                stdin=self._SCRIPT,
            )
        except subprocess.TimeoutExpired as exc:
            raise LiveOperationError("production_completion_transport_failed") from exc
        if not isinstance(result, CommandResult) or result.returncode != 0:
            raise LiveOperationError("production_completion_failed")
        if result.stdout.strip() != "QWEN_COMPLETION_OK":
            raise LiveOperationError("production_completion_output_invalid")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _file_identity(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise LiveOperationError("production_file_invalid")
    return {
        "path": str(path),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "size": info.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _owner_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o077:
        raise LiveOperationError("evidence_directory_invalid")


def _write_new_fsynced(path: Path, payload: bytes, mode: int) -> None:
    _owner_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("evidence_short_write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _preflight_receipt_directory() -> Path:
    root = PREFLIGHT_RECEIPT_ROOT
    if root.exists() or root.is_symlink():
        _owner_directory(root)
        return root
    _owner_directory(root.parent)
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        _owner_directory(root)
        return root
    os.chmod(root, 0o700)
    directory = os.open(root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return root


class CandidateLiveBackend:
    """Allowlisted Server A implementation of the live-operation protocol."""

    def __init__(
        self,
        operation_id: str,
        candidate_id: str,
        candidate_pin: str,
        *,
        code_root: Path,
        runner: CommandRunner | None = None,
        supervisor: JsonSocketClient | None = None,
        admission: JsonSocketClient | None = None,
        production_probe: ProductionProbe | None = None,
        authenticated_route_probe: AuthenticatedRouteProbe | None = None,
        preflight_config: CollectorConfig | None = None,
        preflight_runner: HostCommandRunner | None = None,
        lock_factory: Callable[..., AbstractContextManager[Any]] = custody_locks,
        artifact_preparer: Callable[..., Mapping[str, object]] = prepare_candidate_artifacts,
        image_loader: Callable[..., Mapping[str, object]] = ensure_embedded_image_loaded,
        contract_loader: Callable[..., Mapping[str, Any]] = validate_candidate_contract,
        bandwidth_monitor_factory: Callable[[Any, Any], Any] = safety_sampler.ConcurrentSafetyMonitor,
        local_payload_inventory: Path | None = None,
    ) -> None:
        if SAFE_ID.fullmatch(operation_id) is None:
            raise ValueError("operation_id_invalid")
        if candidate_id not in _PINS or candidate_pin != _PINS[candidate_id]:
            raise ValueError("candidate_identity_not_allowlisted")
        if local_payload_inventory is not None and candidate_id != "qwen38-vllm":
            raise ValueError("local_payload_candidate_not_allowed")
        if (
            not code_root.is_absolute()
            or code_root.is_symlink()
            or not code_root.is_dir()
            or code_root.resolve(strict=True) != code_root.absolute()
        ):
            raise ValueError("code_root_invalid")
        experiments_root = code_root / "experiments"
        if experiments_root.is_symlink() or not experiments_root.is_dir():
            raise ValueError("code_root_invalid")
        self.operation_id = operation_id
        self.candidate_id = candidate_id
        self.candidate_pin = candidate_pin
        self.experiments_root = experiments_root
        self.arm_harness = experiments_root / "candidate_arm_harness.py"
        self.runner = runner or SubprocessRunner()
        self.supervisor = supervisor or UnixJsonSocketClient(SUPERVISOR_SOCKET, admission=False)
        self.admission = admission or UnixJsonSocketClient(ADMISSION_SOCKET, admission=True)
        self.production_probe = production_probe or GatewayProductionProbe(self.runner)
        self.authenticated_route_probe = authenticated_route_probe
        self.preflight_config = preflight_config
        self.preflight_runner = preflight_runner
        self.lock_factory = lock_factory
        self.artifact_preparer = artifact_preparer
        self.image_loader = image_loader
        self.contract_loader = contract_loader
        self.bandwidth_monitor_factory = bandwidth_monitor_factory
        self.local_payload_inventory = local_payload_inventory
        self._locks: AbstractContextManager[Any] | None = None
        self._lock_entered = False
        self._candidate_lock_entered = False
        self._custody_guard: CustodyLockGuard | None = None
        self._prestate: Mapping[str, Any] | None = None
        self._prestate_kernel_cursor: str | None = None
        self._loaded_image_identity: Mapping[str, object] | None = None
        self._admission_request_sequence = 0
        # A FreeToken prestart creates a separate GPU container before an arm is
        # started.  Keep its exact identity until cleanup is independently
        # proven so operation-level recovery cannot mistake an unknown cleanup
        # outcome for an idle host.
        self._prestart_bandwidth_container_id: str | None = None
        self._prestart_bandwidth_create_uncertain = False
        self._prestart_bandwidth_identity: tuple[ArmContract, str, str] | None = None

    def _result(self, argv: Sequence[str], *, timeout: float, code: str, ambiguous: bool = False, stdin: bytes | None = None) -> CommandResult:
        try:
            result = self.runner.run(tuple(argv), timeout=timeout, stdin=stdin)
        except subprocess.TimeoutExpired as exc:
            raise LiveOperationError(code, ambiguous=True) from exc
        if not isinstance(result, CommandResult):
            raise LiveOperationError("command_result_invalid", ambiguous=True)
        if result.returncode != 0:
            raise LiveOperationError(code, ambiguous=ambiguous)
        return result

    @contextmanager
    def host_lock(self, operation_id: str):
        if operation_id != self.operation_id or self._lock_entered:
            raise LiveOperationError("host_lock_request_invalid")
        candidate_lock = STATE_ROOT / "locks" / f"{self.candidate_id}-{self.candidate_pin}.lock"
        manager = self.lock_factory(HOST_LOCK, candidate_lock, production_cas_root=PRODUCTION_CAS_ROOT)
        self._locks = manager
        with manager as guard:
            if not isinstance(guard, CustodyLockGuard):
                raise LiveOperationError("custody_lock_guard_invalid")
            self._lock_entered = True
            self._custody_guard = guard
            try:
                yield None
            finally:
                self._custody_guard = None
                self._lock_entered = False
                self._locks = None

    @contextmanager
    def candidate_lock(self, candidate_id: str, candidate_pin: str):
        if not self._lock_entered or (candidate_id, candidate_pin) != (self.candidate_id, self.candidate_pin):
            raise LiveOperationError("candidate_lock_request_invalid")
        # Both locks are already held by candidate_artifact_custody, in order.
        if self._candidate_lock_entered:
            raise LiveOperationError("candidate_lock_request_invalid")
        self._candidate_lock_entered = True
        try:
            yield None
        finally:
            self._candidate_lock_entered = False

    def _operation_dir(self) -> Path:
        return STATE_ROOT / "operations" / self.operation_id

    def assert_no_stale_state(self, operation_id: str) -> None:
        if operation_id != self.operation_id:
            raise LiveOperationError("operation_identity_mismatch")
        root = self._operation_dir()
        if root.exists() or root.is_symlink():
            raise LiveOperationError("stale_operation_state")
        _owner_directory(root.parent)

    def _docker_values(self, containers: Sequence[str]) -> list[Mapping[str, Any]]:
        if not containers or any(name not in PRODUCTION_CONTAINERS and name != BANDWIDTH_CONTAINER and name not in {arm.container for arm in _ARMS.values()} for name in containers):
            raise LiveOperationError("container_not_allowlisted")
        result = self._result((DOCKER, "inspect", *containers), timeout=30, code="container_inspect_failed")
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LiveOperationError("container_inspect_invalid") from exc
        if not isinstance(values, list) or len(values) != len(containers) or any(not isinstance(value, dict) for value in values):
            raise LiveOperationError("container_inspect_invalid")
        for expected, value in zip(containers, values, strict=True):
            if value.get("Name") != "/" + expected:
                raise LiveOperationError("container_identity_mismatch")
        return values

    def _optional_container_value(self, container: str) -> Mapping[str, Any] | None:
        if container not in PRODUCTION_CONTAINERS and container not in {
            arm.container for arm in _ARMS.values()
        }:
            raise LiveOperationError("container_not_allowlisted")
        try:
            result = self.runner.run((DOCKER, "inspect", container), timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise LiveOperationError("container_state_unknown", ambiguous=True) from exc
        if not isinstance(result, CommandResult):
            raise LiveOperationError("container_state_unknown", ambiguous=True)
        if result.returncode == 1 and not result.stdout.strip() and result.stderr.strip() in {
            f"Error: No such object: {container}",
            f"Error response from daemon: No such object: {container}",
        }:
            return None
        if result.returncode != 0:
            raise LiveOperationError("container_state_unknown", ambiguous=True)
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LiveOperationError("container_state_unknown", ambiguous=True) from exc
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], dict)
            or values[0].get("Name") != "/" + container
        ):
            raise LiveOperationError("container_state_unknown", ambiguous=True)
        return values[0]

    def _production_identities(self) -> dict[str, Any]:
        identities: dict[str, Any] = {}
        for container in PRODUCTION_CONTAINERS:
            value = self._optional_container_value(container)
            identities[container] = {
                "present": False,
            } if value is None else {
                "present": True,
                "identity": self._identity(value),
            }
        return identities

    @staticmethod
    def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
        config, host, state = value.get("Config"), value.get("HostConfig"), value.get("State")
        if not isinstance(config, dict) or not isinstance(host, dict) or not isinstance(state, dict):
            raise LiveOperationError("container_inspect_invalid")
        labels = config.get("Labels")
        health = state.get("Health")
        if labels is not None and not isinstance(labels, dict):
            raise LiveOperationError("container_inspect_invalid")
        if health is not None and not isinstance(health, dict):
            raise LiveOperationError("container_inspect_invalid")
        mounts = value.get("Mounts")
        if not isinstance(mounts, list):
            raise LiveOperationError("container_inspect_invalid")
        normalized_mounts = []
        for mount in mounts:
            if not isinstance(mount, dict):
                raise LiveOperationError("container_inspect_invalid")
            normalized_mounts.append({key: mount.get(key) for key in ("Type", "Source", "Destination", "RW", "Propagation")})
        normalized_mounts.sort(key=_canonical)
        return {
            "id": value.get("Id"),
            "image_id": value.get("Image"),
            "image_reference": config.get("Image"),
            "labels": dict(sorted((labels or {}).items())),
            "config_hash": (labels or {}).get("com.docker.compose.config-hash"),
            "project": (labels or {}).get("com.docker.compose.project"),
            "service": (labels or {}).get("com.docker.compose.service"),
            "restart_policy": host.get("RestartPolicy"),
            "oom_kill_disable": host.get("OomKillDisable"),
            "oom_killed": state.get("OOMKilled"),
            "running": state.get("Running"),
            "status": state.get("Status"),
            "health": None if health is None else health.get("Status"),
            "mounts": normalized_mounts,
        }

    def _authenticated_route_evidence(self) -> dict[str, Any]:
        if self.authenticated_route_probe is None:
            raise LiveOperationError("authenticated_route_probe_unavailable")
        try:
            value = self.authenticated_route_probe.probe()
        except LiveOperationError:
            raise
        except RouteEvidenceError as exc:
            raise LiveOperationError(exc.code) from exc
        except Exception as exc:
            raise LiveOperationError("authenticated_route_probe_failed") from exc
        expected = {"sequence", "ai", "pony", "comfy", "response_sha256"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise LiveOperationError("authenticated_route_probe_invalid")
        if (
            type(value.get("sequence")) is not int
            or value["sequence"] not in {1, 2}
            or value.get("ai") is not True
            or value.get("pony") is not True
            or value.get("comfy") is not True
            or not isinstance(value.get("response_sha256"), str)
            or SHA256.fullmatch(value["response_sha256"]) is None
        ):
            raise LiveOperationError("authenticated_route_probe_failed")
        return {key: value[key] for key in sorted(expected)}

    def _kernel_cursor(self) -> str:
        result = self._result(
            ("/usr/bin/journalctl", "-k", "--show-cursor", "-n", "0", "--no-pager"),
            timeout=30,
            code="kernel_cursor_unavailable",
        )
        cursors = [
            line.removeprefix("-- cursor: ")
            for line in result.stdout.splitlines()
            if line.startswith("-- cursor: ")
        ]
        if len(cursors) != 1 or not cursors[0] or len(cursors[0]) > 4096:
            raise LiveOperationError("kernel_cursor_invalid")
        return cursors[0]

    def _post_restore_fault_evidence(self) -> dict[str, bool]:
        cursor = self._prestate_kernel_cursor
        if cursor is None:
            raise LiveOperationError("kernel_cursor_missing", ambiguous=True)
        result = self._result(
            ("/usr/bin/journalctl", "-k", "--after-cursor", cursor, "--no-pager", "--output=cat"),
            timeout=30,
            code="kernel_fault_probe_failed",
            ambiguous=True,
        )
        lower = result.stdout.lower()
        evidence = {
            "host_oom": "out of memory" in lower or "oom-kill" in lower,
            "nvidia_xid": "nvrm: xid" in lower or "nvidia xid" in lower,
        }
        if any(evidence.values()):
            raise LiveOperationError("post_restore_kernel_fault", ambiguous=True)
        return evidence

    def _gpu_owner_evidence(self, expected_container: str) -> dict[str, Any]:
        if expected_container not in PRODUCTION_CONTAINERS:
            raise LiveOperationError("gpu_owner_container_invalid")
        current = self._optional_container_value(expected_container)
        if current is None:
            raise LiveOperationError("gpu_owner_container_missing", ambiguous=True)
        container_id = current.get("Id")
        if not isinstance(container_id, str) or len(container_id) < 12:
            raise LiveOperationError("gpu_owner_container_invalid", ambiguous=True)
        result = self._result(
            (NVIDIA_SMI, "--query-compute-apps=pid", "--format=csv,noheader,nounits"),
            timeout=15,
            code="gpu_compute_process_query_failed",
            ambiguous=True,
        )
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        if len(rows) != 1 or not rows[0].isdigit():
            raise LiveOperationError("gpu_owner_not_exact_qwen", ambiguous=True)
        try:
            cgroup = Path(f"/proc/{int(rows[0])}/cgroup").read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise LiveOperationError("gpu_owner_unavailable", ambiguous=True) from exc
        exact = container_id in cgroup or container_id[:12] in cgroup
        if not exact:
            raise LiveOperationError("gpu_owner_not_exact_qwen", ambiguous=True)
        return {"compute_process_count": 1, "owner_container": expected_container, "owner_exact": True}

    def _next_admission_request_id(self, action: str) -> str:
        self._admission_request_sequence += 1
        request_id = (
            f"accel-{self.operation_id}-{self._admission_request_sequence:04d}-{action}"
        )
        if len(request_id) > 128:
            raise LiveOperationError("admission_request_id_too_long")
        return request_id

    def _admission_call(self, verb: str) -> dict[str, Any]:
        request_id = self._next_admission_request_id(verb)
        request: dict[str, Any] = {"schema": "ai-admission-operator-command/v1", "verb": verb, "request_id": request_id}
        if verb == "advance-phase":
            raise AssertionError("phase_required")
        value = self.admission.call(request, timeout=30)
        if value.get("ok") is not True or value.get("schema") != "ai-admission-operator-response/v1" or value.get("request_id") != request_id:
            raise LiveOperationError("admission_operation_failed")
        return value

    def _admission_advance(self, phase: str) -> dict[str, Any]:
        if phase not in {"closed", "public"}:
            raise LiveOperationError("admission_phase_not_allowlisted")
        request_id = self._next_admission_request_id(phase)
        request = {"schema": "ai-admission-operator-command/v1", "verb": "advance-phase", "request_id": request_id, "phase": phase}
        value = self.admission.call(request, timeout=30)
        if value.get("ok") is not True or value.get("schema") != "ai-admission-operator-response/v1" or value.get("request_id") != request_id or value.get("phase") != phase:
            raise LiveOperationError("admission_operation_failed", ambiguous=True)
        return value

    @staticmethod
    def _assert_admission(value: Mapping[str, Any], phase: str) -> None:
        if (
            value.get("phase") != phase
            or value.get("capacity_mode") != "serialized-99"
            or value.get("restart_pending") is not False
            or value.get("reconciliation_required") is not False
            or value.get("active_ledger") != []
            or value.get("queue_depth") != 0
            or value.get("active_ledger_digest") != EMPTY_LEDGER_SHA256
        ):
            raise LiveOperationError("admission_capacity_fifo_invalid")

    def _supervisor_call(self, verb: str, suffix: str | None = None) -> dict[str, Any]:
        if verb not in {"status", "start", "stop"}:
            raise LiveOperationError("supervisor_verb_not_allowlisted")
        request: dict[str, Any] = {"verb": verb, "profile": "qwen"}
        if suffix is not None:
            request["idempotency_key"] = f"accel:{self.operation_id}:{suffix}"
        value = self.supervisor.call(request, timeout=180)
        if value.get("ok") is not True:
            raise LiveOperationError("supervisor_operation_failed", ambiguous=verb in {"start", "stop"})
        return value

    def _power_and_idle(self) -> None:
        power = self._result((NVIDIA_SMI, "--query-gpu=power.limit", "--format=csv,noheader,nounits"), timeout=15, code="power_limit_unavailable")
        try:
            value = float(power.stdout.strip())
        except ValueError as exc:
            raise LiveOperationError("power_limit_invalid") from exc
        if abs(value - POWER_LIMIT_W) > 0.05:
            raise LiveOperationError("power_limit_not_300w")
        apps = self._result((NVIDIA_SMI, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"), timeout=15, code="gpu_compute_process_query_failed")
        if apps.stdout.strip():
            raise LiveOperationError("gpu_compute_processes_not_empty")

    def _power_limit(self) -> float:
        result = self._result(
            (NVIDIA_SMI, "--query-gpu=power.limit", "--format=csv,noheader,nounits"),
            timeout=15,
            code="power_limit_unavailable",
        )
        try:
            value = float(result.stdout.strip())
        except ValueError as exc:
            raise LiveOperationError("power_limit_invalid") from exc
        if abs(value - POWER_LIMIT_W) > 0.05:
            raise LiveOperationError("power_limit_not_300w")
        return value

    def _active_caddy_identity(self) -> dict[str, str]:
        result = self._result(
            (DOCKER, "exec", EDGE_CONTAINER, "cat", CADDY_DESTINATION),
            timeout=30,
            code="active_caddy_unavailable",
        )
        if "\ufffd" in result.stdout:
            raise LiveOperationError("caddy_bytes_invalid")
        raw = result.stdout.encode("utf-8")
        try:
            auth_identity = _caddy_auth_identity(raw)
        except CollectorError as exc:
            raise LiveOperationError("caddy_auth_identity_invalid") from exc
        return {
            "active_sha256": hashlib.sha256(raw).hexdigest(),
            "auth_identity_sha256": auth_identity,
        }

    def run_preflight(self, candidate_id: str, candidate_pin: str) -> str:
        if (candidate_id, candidate_pin) != (self.candidate_id, self.candidate_pin):
            raise LiveOperationError("candidate_identity_mismatch")
        config = self.preflight_config
        if config is None or config.candidate_id != candidate_id:
            raise LiveOperationError("host_preflight_config_missing")
        try:
            snapshot = collect_snapshot(config, self.preflight_runner)
            result = evaluate_snapshot(snapshot)
        except (CollectorError, PreflightError) as exc:
            raise LiveOperationError("host_preflight_failed") from exc
        receipt = {
            "schema_version": "ai-server-acceleration-preflight-receipt/v1",
            "operation_id": self.operation_id,
            "candidate_id": candidate_id,
            "candidate_pin": candidate_pin,
            "result": result.as_dict(),
        }
        payload = _canonical(receipt) + b"\n"
        try:
            receipt_root = _preflight_receipt_directory()
            _write_new_fsynced(
                receipt_root / f"{self.operation_id}.json", payload, 0o400
            )
        except (FileExistsError, OSError, LiveOperationError) as exc:
            raise LiveOperationError("preflight_receipt_write_failed") from exc
        return hashlib.sha256(payload).hexdigest()

    def capture_prestate(self, operation_id: str) -> Mapping[str, Any]:
        if operation_id != self.operation_id:
            raise LiveOperationError("operation_identity_mismatch")
        files = [_file_identity(path) for path in PRODUCTION_FILES]
        containers = self._production_identities()
        for name in ALWAYS_RUNNING_PRODUCTION_CONTAINERS:
            item = containers[name]
            if (
                item.get("present") is not True
                or item["identity"]["running"] is not True
                or item["identity"]["oom_killed"] is not False
                or item["identity"]["health"] not in {None, "healthy"}
            ):
                raise LiveOperationError("production_prestate_unhealthy")
        for name in QWEN_STOPPED_GPU_COMPANION_CONTAINERS:
            item = containers[name]
            if item.get("present") is True and (
                item["identity"]["running"] is not False
                or item["identity"]["oom_killed"] is not False
            ):
                raise LiveOperationError("production_prestate_heavy_owner_invalid")
        edge_mounts = containers[EDGE_CONTAINER]["identity"]["mounts"]
        caddy = [mount for mount in edge_mounts if mount["Destination"] == CADDY_DESTINATION]
        if len(caddy) != 1 or caddy[0]["Type"] != "bind" or caddy[0]["Source"] != CADDY_SOURCE or caddy[0]["RW"] is not False:
            raise LiveOperationError("caddy_bind_identity_invalid")
        admission = self._admission_call("status")
        self._assert_admission(admission, "public")
        supervisor = self._supervisor_call("status")
        if supervisor.get("active_profile") != "qwen" or supervisor.get("profile") != "qwen" or not isinstance(supervisor.get("status"), dict) or supervisor["status"].get("state") != "ready":
            raise LiveOperationError("production_supervisor_invalid")
        # A running Qwen legitimately owns the GPU; only the cap is checked here.
        self._power_limit()
        gpu_owner = self._gpu_owner_evidence(QWEN_CONTAINER)
        caddy_identity = self._active_caddy_identity()
        routes = self._authenticated_route_evidence()
        if routes["sequence"] != 1:
            raise LiveOperationError("authenticated_route_sequence_invalid")
        cursor = self._kernel_cursor()
        epoch = admission.get("admission_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise LiveOperationError("admission_epoch_invalid")
        value = {
            "schema_version": "ai-server-acceleration-prestate/v2",
            "files": files,
            "containers": containers,
            "caddy_identity": caddy_identity,
            "admission": {key: admission.get(key) for key in (
                "admission_epoch", "phase", "capacity_mode", "restart_pending",
                "reconciliation_required", "queue_depth", "active_ledger_digest",
            )},
            "authenticated_routes": routes,
            "gpu_owner": gpu_owner,
            "kernel_cursor_sha256": hashlib.sha256(cursor.encode("utf-8")).hexdigest(),
            "power_limit_w": POWER_LIMIT_W,
        }
        self._prestate_kernel_cursor = cursor
        self._prestate = value
        return value

    def create_journal(self, operation_id: str, candidate_id: str, prestate: Mapping[str, Any]) -> None:
        if (operation_id, candidate_id) != (self.operation_id, self.candidate_id) or prestate is not self._prestate:
            raise LiveOperationError("journal_identity_invalid")
        root = self._operation_dir()
        if root.exists() or root.is_symlink():
            raise LiveOperationError("stale_operation_state")
        os.mkdir(root, 0o700)
        os.chmod(root, 0o700)
        directory = os.open(root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _write_new_fsynced(root / "prestate.json", _canonical(prestate) + b"\n", 0o400)
        first = {"schema_version": "ai-server-acceleration-journal/v1", "operation_id": self.operation_id, "candidate_id": self.candidate_id, "phase": "created"}
        _write_new_fsynced(root / "journal.jsonl", _canonical(first) + b"\n", 0o600)

    def record_phase(self, operation_id: str, phase: str) -> None:
        if operation_id != self.operation_id or re.fullmatch(r"[a-z0-9-]{3,96}", phase) is None:
            raise LiveOperationError("journal_phase_invalid")
        path = self._operation_dir() / "journal.jsonl"
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o600:
            raise LiveOperationError("journal_file_invalid")
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.write(descriptor, _canonical({"phase": phase}) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _verify_syv_image(value: Mapping[str, Any]) -> dict[str, object]:
        config = value.get("Config")
        repo_digests = value.get("RepoDigests")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        if (
            value.get("Id") != SYV_CONFIG_DIGEST
            or value.get("Architecture") != "amd64"
            or value.get("Os") != "linux"
            or not isinstance(config, Mapping)
            or not isinstance(labels, Mapping)
            or any(labels.get(key) != expected for key, expected in SYV_IMAGE_LABELS.items())
            or not isinstance(repo_digests, list)
            or repo_digests.count(SYV_REPO_DIGEST) != 1
        ):
            raise LiveOperationError("syv_image_runtime_identity_mismatch", ambiguous=True)
        return {
            "reference": SYV_IMAGE_REFERENCE,
            "image_digest": SYV_CONFIG_DIGEST,
            "manifest_digest": SYV_MANIFEST_DIGEST,
        }

    def _inspect_syv_image(self) -> Mapping[str, Any] | None:
        try:
            result = self.runner.run(
                (DOCKER, "image", "inspect", "--platform", "linux/amd64", SYV_IMAGE_REFERENCE),
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise LiveOperationError("syv_image_inspect_failed", ambiguous=True) from exc
        if not isinstance(result, CommandResult):
            raise LiveOperationError("syv_image_inspect_failed", ambiguous=True)
        if result.returncode == 0:
            try:
                values = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise LiveOperationError("syv_image_inspect_invalid", ambiguous=True) from exc
            if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
                raise LiveOperationError("syv_image_inspect_invalid", ambiguous=True)
            return values[0]
        stderr = result.stderr.strip().lower()
        if (
            result.returncode == 1
            and result.stdout.strip() in {"", "[]"}
            and ("no such image" in stderr or "no such object" in stderr)
        ):
            return None
        raise LiveOperationError("syv_image_state_unknown", ambiguous=True)

    def _ensure_syv_image(self, expected_image: Mapping[str, Any]) -> dict[str, object]:
        if not self._lock_entered or not self._candidate_lock_entered or self._custody_guard is None:
            raise LiveOperationError("syv_image_acquire_locks_not_held")
        if (
            expected_image.get("reference") != SYV_IMAGE_REFERENCE
            or expected_image.get("digest") != SYV_MANIFEST_DIGEST
            or expected_image.get("revision_label") != _PINS["qwen38-vllm"]
        ):
            raise LiveOperationError("syv_image_contract_mismatch")
        value = self._inspect_syv_image()
        if value is not None:
            return {**self._verify_syv_image(value), "status": "reused"}
        try:
            result = self.runner.run(
                (DOCKER, "image", "pull", "--platform", "linux/amd64", SYV_IMAGE_REFERENCE),
                timeout=1800,
            )
        except subprocess.TimeoutExpired as exc:
            raise LiveOperationError("syv_image_pull_failed", ambiguous=True) from exc
        if not isinstance(result, CommandResult) or result.returncode != 0:
            raise LiveOperationError("syv_image_pull_failed", ambiguous=True)
        value = self._inspect_syv_image()
        if value is None:
            raise LiveOperationError("syv_image_not_present_after_pull", ambiguous=True)
        return {**self._verify_syv_image(value), "status": "pulled"}

    def prepare_candidate(self, candidate_id: str, candidate_pin: str) -> str:
        if (candidate_id, candidate_pin) != (self.candidate_id, self.candidate_pin):
            raise LiveOperationError("candidate_identity_mismatch")
        if not self._lock_entered or not self._candidate_lock_entered or self._custody_guard is None:
            raise LiveOperationError("candidate_prepare_locks_not_held")
        try:
            modes = [mode for (candidate, mode) in _MODE_NAMES if candidate == candidate_id]
            candidate_contract: Mapping[str, Any] | None = None
            for arm in modes:
                validated = validate_candidate_contract(
                    candidate_id,
                    _MODE_NAMES[(candidate_id, arm)],
                    root=self.experiments_root,
                    require_live=True,
                )
                if candidate_contract is None:
                    candidate_contract = validated
                elif candidate_contract.get("image") != validated.get("image"):
                    raise LiveOperationError("candidate_mode_image_identity_mismatch")
            loaded_image: Mapping[str, object] | None = None
            if candidate_id == "qwen38-vllm":
                if candidate_contract is None or not isinstance(candidate_contract.get("image"), Mapping):
                    raise LiveOperationError("candidate_image_contract_missing")
                loaded_image = self._ensure_syv_image(candidate_contract["image"])
                if loaded_image.get("status") not in {"pulled", "reused"}:
                    raise LiveOperationError("candidate_image_load_result_invalid", ambiguous=True)
            elif candidate_id in {"freetoken", "llama-dflash2-official"}:
                if candidate_contract is None or not isinstance(candidate_contract.get("image"), Mapping):
                    raise LiveOperationError("candidate_image_contract_missing")
                loaded_image = self.image_loader(
                    self.experiments_root.parent,
                    candidate_id,
                    candidate_pin,
                    candidate_contract["image"],
                    self.runner,
                )
                if loaded_image.get("status") not in {"loaded", "reused"}:
                    raise LiveOperationError("candidate_image_load_result_invalid", ambiguous=True)
            candidate_lock = STATE_ROOT / "locks" / f"{candidate_id}-{candidate_pin}.lock"
            prepare_kwargs: dict[str, Any] = {
                "experiments_root": self.experiments_root,
                "staging_root": STAGING_ROOT,
                "production_cas_root": PRODUCTION_CAS_ROOT,
                "lock_guard": self._custody_guard,
                "expected_host_lock": HOST_LOCK,
                "expected_candidate_lock": candidate_lock,
                "sealed_image_identity": (
                    loaded_image if candidate_id in {"freetoken", "llama-dflash2-official"} else None
                ),
            }
            if self.local_payload_inventory is not None:
                prepare_kwargs["local_payload_inventory"] = self.local_payload_inventory
            prepared = self.artifact_preparer(
                candidate_id,
                candidate_pin,
                **prepare_kwargs,
            )
            digest = prepared.get("manifest_sha256")
            if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
                raise LiveOperationError("candidate_prepare_result_invalid")
            self._loaded_image_identity = loaded_image
        except ImageLoadError as exc:
            raise LiveOperationError("candidate_embedded_image_not_live_ready", ambiguous=exc.ambiguous) from exc
        except (ContractError, CustodyError, PrepareError, OSError) as exc:
            raise LiveOperationError("candidate_package_not_live_ready") from exc
        return digest

    def close_admission(self) -> None:
        value = self._admission_advance("closed")
        try:
            self._assert_admission(value, "closed")
        except LiveOperationError as exc:
            raise LiveOperationError(exc.code, ambiguous=True) from exc

    def ensure_admission_closed(self) -> None:
        """Prove closed/empty, applying at most one compensating transition."""

        try:
            observed = self._admission_call("status")
            try:
                self._assert_admission(observed, "closed")
            except LiveOperationError:
                if observed.get("phase") == "closed":
                    raise LiveOperationError("admission_closed_not_safe", ambiguous=True)
                transitioned = self._admission_advance("closed")
                self._assert_admission(transitioned, "closed")
            verified = self._admission_call("status")
            self._assert_admission(verified, "closed")
        except LiveOperationError as exc:
            raise LiveOperationError(exc.code, ambiguous=True) from exc

    def wait_fifo_drained(self) -> None:
        value = self._admission_call("status")
        self._assert_admission(value, "closed")

    def stop_production_qwen(self, operation_id: str) -> None:
        if operation_id != self.operation_id:
            raise LiveOperationError("operation_identity_mismatch")
        value = self._supervisor_call("stop", "production-stop")
        if value.get("profile") != "qwen" or value.get("state") != "stopped" or value.get("active_profile") is not None:
            raise LiveOperationError("production_stop_response_invalid", ambiguous=True)
        observed = self._identity(self._docker_values((QWEN_CONTAINER,))[0])
        if observed["running"] is not False or observed["status"] != "exited":
            raise LiveOperationError("production_stop_not_observed", ambiguous=True)

    def assert_gpu_idle(self) -> None:
        self._power_and_idle()

    def _arm(self, candidate_id: str, arm: str) -> ArmContract:
        if candidate_id != self.candidate_id or (candidate_id, arm) not in _ARMS:
            raise LiveOperationError("candidate_arm_not_allowlisted")
        return _ARMS[(candidate_id, arm)]

    def _compose_argv(self, contract: ArmContract, *tail: str) -> tuple[str, ...]:
        directory = str(self.experiments_root / contract.directory)
        compose = str(self.experiments_root / contract.directory / contract.compose)
        return (DOCKER, "compose", "--project-directory", directory, "-f", compose, "--project-name", contract.project, *tail)

    def _bandwidth_container_absent(self, identifier: str = BANDWIDTH_CONTAINER) -> bool:
        try:
            result = self.runner.run((DOCKER, "inspect", identifier), timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise LiveOperationError("bandwidth_container_state_unknown", ambiguous=True) from exc
        if not isinstance(result, CommandResult):
            raise LiveOperationError("command_result_invalid", ambiguous=True)
        if result.returncode == 0:
            return False
        expected = {
            f"Error: No such object: {identifier}",
            f"Error response from daemon: No such object: {identifier}",
        }
        if result.returncode == 1 and not result.stdout.strip() and result.stderr.strip() in expected:
            return True
        raise LiveOperationError("bandwidth_container_state_unknown", ambiguous=True)

    def _bandwidth_container_value(self, identifier: str) -> Mapping[str, Any]:
        result = self._result(
            (DOCKER, "inspect", identifier), timeout=30,
            code="bandwidth_container_inspect_failed", ambiguous=True,
        )
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LiveOperationError("bandwidth_container_inspect_invalid", ambiguous=True) from exc
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], dict)
            or values[0].get("Name") != "/" + BANDWIDTH_CONTAINER
        ):
            raise LiveOperationError("bandwidth_container_inspect_invalid", ambiguous=True)
        return values[0]

    def _cleanup_bandwidth_container(self, container_id: str) -> None:
        if SHA256.fullmatch(container_id) is None:
            raise LiveOperationError("bandwidth_container_id_invalid", ambiguous=True)
        name_absent = self._bandwidth_container_absent()
        id_absent = self._bandwidth_container_absent(container_id)
        if name_absent and id_absent:
            return
        if name_absent or id_absent:
            raise LiveOperationError("bandwidth_container_identity_drift", ambiguous=True)
        raw = self._bandwidth_container_value(BANDWIDTH_CONTAINER)
        if raw.get("Id") != container_id:
            raise LiveOperationError("bandwidth_container_identity_drift", ambiguous=True)
        state = raw.get("State")
        if (
            not isinstance(state, Mapping)
            or state.get("Running") is not False
            or state.get("Status") not in {"created", "exited"}
        ):
            raise LiveOperationError("bandwidth_container_not_terminal", ambiguous=True)
        self._result(
            (DOCKER, "rm", container_id),
            timeout=60,
            code="bandwidth_container_cleanup_failed",
            ambiguous=True,
        )
        if not self._bandwidth_container_absent() or not self._bandwidth_container_absent(container_id):
            raise LiveOperationError("bandwidth_container_cleanup_failed", ambiguous=True)

    def _cancel_bandwidth_container(self, container_id: str) -> None:
        if SHA256.fullmatch(container_id) is None:
            raise LiveOperationError("bandwidth_container_id_invalid", ambiguous=True)
        self._result(
            (DOCKER, "stop", "--time", "10", container_id),
            timeout=15,
            code="bandwidth_container_cancel_failed",
            ambiguous=True,
        )
        raw = self._bandwidth_container_value(container_id)
        state = raw.get("State")
        if (
            not isinstance(state, Mapping)
            or state.get("Running") is not False
            or state.get("Status") != "exited"
        ):
            raise LiveOperationError("bandwidth_container_cancel_failed", ambiguous=True)

    def _clear_prestart_bandwidth_tracking(self) -> None:
        self._prestart_bandwidth_container_id = None
        self._prestart_bandwidth_create_uncertain = False
        self._prestart_bandwidth_identity = None

    def _recover_prestart_bandwidth_identity(self) -> str | None:
        """Resolve only the fixed prestart name after a lost create outcome."""

        if not self._prestart_bandwidth_create_uncertain:
            return self._prestart_bandwidth_container_id
        if self._bandwidth_container_absent():
            self._clear_prestart_bandwidth_tracking()
            return None
        raw = self._bandwidth_container_value(BANDWIDTH_CONTAINER)
        container_id = raw.get("Id")
        if not isinstance(container_id, str) or SHA256.fullmatch(container_id) is None:
            raise LiveOperationError("bandwidth_container_id_invalid", ambiguous=True)
        identity = self._prestart_bandwidth_identity
        state = raw.get("State")
        if identity is None or not isinstance(state, Mapping):
            raise LiveOperationError("bandwidth_container_identity_drift", ambiguous=True)
        status = state.get("Status")
        if status not in {"created", "running", "exited"}:
            raise LiveOperationError("bandwidth_container_not_terminal", ambiguous=True)
        # Before an ID was pinned, require the complete immutable identity.  An
        # exited non-zero command cannot be safely attributed after an unknown
        # create acknowledgement, so it stays fenced rather than being removed.
        self._validate_bandwidth_identity(
            raw, container_id, identity[0], identity[1], identity[2],
            expected_status=status,
        )
        self._prestart_bandwidth_container_id = container_id
        self._prestart_bandwidth_create_uncertain = False
        return container_id

    def ensure_prestart_stopped(self, candidate_id: str, arm: str) -> None:
        """Fence a failed FreeToken prestart before any production restore.

        This is deliberately not cleanup retry logic.  It performs at most one
        compensating stop of the already pinned ID, then requires that exact ID
        to be terminal or absent.  A lost create/cleanup acknowledgement that
        cannot meet that proof leaves admission closed through the caller.
        """

        self._arm(candidate_id, arm)
        if (candidate_id, arm) != ("freetoken", "freetoken"):
            return
        container_id = self._recover_prestart_bandwidth_identity()
        if container_id is None:
            return
        name_absent = self._bandwidth_container_absent()
        id_absent = self._bandwidth_container_absent(container_id)
        if name_absent and id_absent:
            self._clear_prestart_bandwidth_tracking()
            return
        if name_absent or id_absent:
            raise LiveOperationError("bandwidth_container_identity_drift", ambiguous=True)
        raw = self._bandwidth_container_value(container_id)
        if raw.get("Id") != container_id:
            raise LiveOperationError("bandwidth_container_identity_drift", ambiguous=True)
        state = raw.get("State")
        if not isinstance(state, Mapping):
            raise LiveOperationError("bandwidth_container_not_terminal", ambiguous=True)
        if state.get("Running") is True and state.get("Status") == "running":
            identity = self._prestart_bandwidth_identity
            if identity is None:
                raise LiveOperationError("bandwidth_container_identity_drift", ambiguous=True)
            self._validate_bandwidth_identity(
                raw, container_id, identity[0], identity[1], identity[2],
                expected_status="running",
            )
            self._cancel_bandwidth_container(container_id)
            raw = self._bandwidth_container_value(container_id)
            state = raw.get("State")
        if (
            raw.get("Id") != container_id
            or not isinstance(state, Mapping)
            or state.get("Running") is not False
            or state.get("Status") not in {"created", "exited"}
        ):
            raise LiveOperationError("bandwidth_container_not_terminal", ambiguous=True)
        self._clear_prestart_bandwidth_tracking()

    @staticmethod
    def _validate_bandwidth_identity(
        raw: Mapping[str, Any], container_id: str, contract: ArmContract,
        image_reference: str, image_digest: str, *, expected_status: str,
    ) -> None:
        if expected_status not in {"created", "running", "exited"}:
            raise LiveOperationError("bandwidth_container_identity_invalid", ambiguous=True)
        config, host, state = raw.get("Config"), raw.get("HostConfig"), raw.get("State")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        environment = config.get("Env") if isinstance(config, Mapping) else None
        mounts = raw.get("Mounts")
        ports = host.get("PortBindings") if isinstance(host, Mapping) else None
        tmpfs = host.get("Tmpfs") if isinstance(host, Mapping) else None
        device_requests = host.get("DeviceRequests") if isinstance(host, Mapping) else None
        expected_root = STAGING_ROOT / "freetoken" / _PINS["freetoken"]
        expected_mounts = {
            (str(expected_root / "models" / "openai-gpt-oss-20b"), "/models/openai-gpt-oss-20b", False),
            (str(expected_root / "cache"), "/cache", True),
        }
        actual_mounts = set()
        if isinstance(mounts, list):
            for mount in mounts:
                if isinstance(mount, Mapping) and mount.get("Type") == "bind":
                    actual_mounts.add((mount.get("Source"), mount.get("Destination"), mount.get("RW")))
        device_request_valid = (
            isinstance(device_requests, list)
            and len(device_requests) == 1
            and isinstance(device_requests[0], Mapping)
            and device_requests[0].get("Driver") == "nvidia"
            and device_requests[0].get("Count") == 1
            and device_requests[0].get("Capabilities") == [["gpu"]]
        )
        tmpfs_options = tmpfs.get("/tmp") if isinstance(tmpfs, Mapping) else None
        tmpfs_valid = (
            isinstance(tmpfs_options, str)
            and {"rw", "noexec", "nosuid", "size=4g"}.issubset(set(tmpfs_options.split(",")))
        )
        if (
            raw.get("Name") != "/" + BANDWIDTH_CONTAINER
            or raw.get("Id") != container_id
            or raw.get("Image") != image_digest
            or not isinstance(config, Mapping)
            or not isinstance(host, Mapping)
            or not isinstance(state, Mapping)
            or not isinstance(labels, Mapping)
            or not isinstance(environment, list)
            or not {"HF_HUB_OFFLINE=1", "HOME=/cache"}.issubset(set(environment))
            or config.get("Image") != image_reference
            or config.get("User") != "65532:65532"
            or config.get("Entrypoint") != ["ft"]
            or config.get("Cmd") != ["bench", "bw", "--model", "gpt-oss-20b", "--out", "/cache/.cache/freetoken/benchbw.json"]
            or labels.get("com.docker.compose.project") != contract.project
            or labels.get("com.docker.compose.service") != "freetoken-bench"
            or labels.get("com.docker.compose.oneoff") != "False"
            or labels.get("com.docker.compose.container-number") != "1"
            or host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}
            or host.get("ReadonlyRootfs") is not True
            or host.get("AutoRemove") is not False
            or not device_request_valid
            or not tmpfs_valid
            or ports not in (None, {})
            or actual_mounts != expected_mounts
            or state.get("Running") is not (expected_status == "running")
            or state.get("Status") != expected_status
            or state.get("ExitCode") != 0
            or state.get("OOMKilled") is not False
        ):
            raise LiveOperationError("bandwidth_container_identity_invalid", ambiguous=True)

    def run_prestart(self, candidate_id: str, arm: str) -> str | None:
        contract = self._arm(candidate_id, arm)
        if (candidate_id, arm) != ("freetoken", "freetoken"):
            return None
        operation_dir = self._operation_dir()
        report_path = operation_dir / "freetoken-bandwidth-report.json"
        safety_path = operation_dir / "freetoken-bandwidth-safety.json"
        profile_path = STAGING_ROOT / candidate_id / self.candidate_pin / BANDWIDTH_PROFILE_RELATIVE
        if any(path.exists() or path.is_symlink() for path in (report_path, safety_path, profile_path)):
            raise LiveOperationError("bandwidth_evidence_preexists")
        try:
            candidate = self.contract_loader(
                candidate_id, "serve", root=self.experiments_root, require_live=True
            )
            suite, suite_sha = bandwidth_runner.load_suite(
                self.experiments_root.parent / "benchmarks" / "suites"
                / "acceleration-freetoken-bandwidth-v1.json"
            )
            image = candidate["image"]
            target = next(model for model in candidate["models"] if model["role"] == "target")
            identity = {
                "candidate_id": "freetoken",
                "runtime_id": "runtime@" + image["digest"],
                "manifest_id": hashlib.sha256(_canonical(candidate)).hexdigest(),
                "image_digest": image["digest"],
                "model_manifest_sha256": target["manifest_sha256"],
            }
            if suite_sha != candidate["benchmark"]["bandwidth_profile_sha256"]:
                raise LiveOperationError("bandwidth_profile_pin_mismatch")
        except LiveOperationError:
            raise
        except (ContractError, KeyError, StopIteration, TypeError, bandwidth_runner.BandwidthError) as exc:
            raise LiveOperationError("bandwidth_contract_invalid") from exc
        if not self._bandwidth_container_absent():
            raise LiveOperationError("bandwidth_container_stale", ambiguous=True)
        if (
            self._prestart_bandwidth_container_id is not None
            or self._prestart_bandwidth_create_uncertain
            or self._prestart_bandwidth_identity is not None
        ):
            raise LiveOperationError("bandwidth_prestart_reconciliation_required", ambiguous=True)
        limits = safety_sampler.SafetyLimits(
            candidate_id="freetoken-freetoken",
            maximum_temperature_c=85.0,
            maximum_power_draw_w=300.0,
            expected_power_cap_w=300.0,
            minimum_vram_free_mib=1536.0,
            minimum_ram_available_mib=4096.0,
            maximum_swap_io_delta=0,
            minimum_sample_count=2,
            maximum_consecutive_swap_samples=2,
            require_runtime_identity=True,
        )
        container_id: str | None = None
        monitor: Any | None = None
        monitor_started = False
        safety_receipt: Mapping[str, Any] | None = None
        measured: Mapping[str, Any] | None = None
        pending: BaseException | None = None
        try:
            self._prestart_bandwidth_identity = (
                contract, image["reference"], image["digest"],
            )
            self._prestart_bandwidth_create_uncertain = True
            self._result(
                self._compose_argv(
                    contract, "up", "--no-start", "--no-build", "--pull", "never",
                    "freetoken-bench",
                ),
                timeout=60,
                code="bandwidth_container_create_failed",
                ambiguous=True,
            )
            created = self._bandwidth_container_value(BANDWIDTH_CONTAINER)
            container_id = created.get("Id")
            if not isinstance(container_id, str) or SHA256.fullmatch(container_id) is None:
                raise LiveOperationError("bandwidth_container_id_invalid", ambiguous=True)
            self._prestart_bandwidth_container_id = container_id
            self._prestart_bandwidth_create_uncertain = False
            self._validate_bandwidth_identity(
                created, container_id, contract, image["reference"], image["digest"],
                expected_status="created",
            )
            probe = _BandwidthHostProbe(self.runner, container_id)
            monitor = self.bandwidth_monitor_factory(limits, probe)
            monitor.start()
            monitor_started = True

            def workload() -> Mapping[str, Any]:
                self._result(
                    (DOCKER, "start", container_id), timeout=30,
                    code="bandwidth_container_start_failed", ambiguous=True,
                )
                probe.mark_running()
                running = self._bandwidth_container_value(container_id)
                self._validate_bandwidth_identity(
                    running, container_id, contract, image["reference"], image["digest"],
                    expected_status="running",
                )
                self._result(
                    (DOCKER, "wait", container_id), timeout=1830,
                    code="bandwidth_container_wait_failed", ambiguous=True,
                )
                probe.mark_completed()
                raw = self._bandwidth_container_value(container_id)
                self._validate_bandwidth_identity(
                    raw, container_id, contract, image["reference"], image["digest"],
                    expected_status="exited",
                )
                try:
                    return bandwidth_runner.parse_profile(profile_path, suite)
                except bandwidth_runner.BandwidthError as exc:
                    raise LiveOperationError("bandwidth_profile_invalid") from exc

            measured = monitor.run_guarded(
                workload,
                lambda: self._cancel_bandwidth_container(container_id),
                timeout_seconds=1830,
                cancellation_timeout_seconds=15,
            )
            probe.assert_owned_compute_pid_observed()
        except BaseException as exc:
            pending = exc
        if monitor_started:
            try:
                safety_receipt = monitor.stop().as_dict()
            except BaseException as exc:
                if pending is None:
                    pending = exc
        if container_id is not None:
            try:
                self._cleanup_bandwidth_container(container_id)
                self._clear_prestart_bandwidth_tracking()
            except BaseException as exc:
                pending = exc
        if pending is not None:
            if isinstance(pending, LiveOperationError):
                raise pending
            raise LiveOperationError("bandwidth_prestart_failed") from pending
        if safety_receipt is None or measured is None:
            raise LiveOperationError("bandwidth_safety_missing")
        try:
            report = bandwidth_runner.build_report(suite, suite_sha, measured, identity, safety_receipt)
            safety_payload = _canonical(safety_receipt) + b"\n"
            report_payload = _canonical(report) + b"\n"
            _write_new_fsynced(safety_path, safety_payload, 0o400)
            _write_new_fsynced(report_path, report_payload, 0o400)
        except (OSError, LiveOperationError, bandwidth_runner.BandwidthError) as exc:
            raise LiveOperationError("bandwidth_report_write_failed") from exc
        return hashlib.sha256(report_payload).hexdigest()

    def start_arm(self, candidate_id: str, arm: str) -> None:
        contract = self._arm(candidate_id, arm)
        tail = ["up", "-d", "--no-build"]
        if self._loaded_image_identity is None:
            raise LiveOperationError("candidate_image_not_verified")
        tail.extend(("--pull", "never"))
        tail.extend(("--wait", "--wait-timeout", str(contract.ready_timeout), contract.service))
        argv = self._compose_argv(contract, *tail)
        self._result(argv, timeout=contract.ready_timeout + 30, code="candidate_start_failed", ambiguous=True)

    def wait_arm_ready(self, candidate_id: str, arm: str) -> None:
        contract = self._arm(candidate_id, arm)
        identity = self._identity(self._docker_values((contract.container,))[0])
        if identity["running"] is not True or identity["health"] != "healthy" or identity["project"] != contract.project or identity["service"] != contract.service or identity["restart_policy"] != {"Name": "no", "MaximumRetryCount": 0} or identity["oom_killed"] is not False:
            raise LiveOperationError("candidate_identity_or_readiness_invalid")
        if (
            self._loaded_image_identity is None
            or identity["image_id"] != self._loaded_image_identity.get("image_digest")
            or identity["image_reference"] != self._loaded_image_identity.get("reference")
        ):
            raise LiveOperationError("candidate_loaded_image_identity_drift", ambiguous=True)

    def run_arm(self, candidate_id: str, arm: str) -> Mapping[str, Any]:
        self._arm(candidate_id, arm)
        try:
            info = self.arm_harness.lstat()
        except FileNotFoundError as exc:
            raise LiveOperationError("candidate_arm_harness_missing") from exc
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
            raise LiveOperationError("candidate_arm_harness_invalid")
        output_dir = self._operation_dir() / "arms" / arm
        output_dir.mkdir(parents=True, mode=0o700)
        os.chmod(output_dir, 0o700)
        result = self._result((PYTHON, "-B", str(self.arm_harness), "--candidate-id", candidate_id, "--arm", arm, "--operation-id", self.operation_id, "--output-dir", str(output_dir)), timeout=7200, code="candidate_arm_harness_failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LiveOperationError("candidate_arm_result_invalid") from exc
        if not isinstance(value, dict):
            raise LiveOperationError("candidate_arm_result_invalid")
        return value

    def stop_arm(self, candidate_id: str, arm: str) -> None:
        contract = self._arm(candidate_id, arm)
        self._result(self._compose_argv(contract, "down", "--timeout", "60", "--remove-orphans"), timeout=90, code="candidate_stop_failed", ambiguous=True)

    def ensure_arm_stopped(self, candidate_id: str, arm: str) -> None:
        """Inspect first; perform at most one compensating stop; then prove idle."""

        contract = self._arm(candidate_id, arm)
        before = self._optional_container_value(contract.container)
        if before is not None:
            identity = self._identity(before)
            if identity["running"] is True and identity["status"] == "running":
                self.stop_arm(candidate_id, arm)
            elif identity["running"] is not False or identity["status"] not in {"exited", "dead"}:
                raise LiveOperationError("candidate_arm_state_unknown", ambiguous=True)
        after = self._optional_container_value(contract.container)
        if after is not None:
            identity = self._identity(after)
            if identity["running"] is not False or identity["status"] not in {"exited", "dead"}:
                raise LiveOperationError("candidate_arm_stop_not_proven", ambiguous=True)
        try:
            self._power_and_idle()
        except LiveOperationError as exc:
            raise LiveOperationError(exc.code, ambiguous=True) from exc

    def restore_production(self, prestate: Mapping[str, Any]) -> None:
        if prestate is not self._prestate:
            raise LiveOperationError("restore_prestate_mismatch", ambiguous=True)
        value = self._supervisor_call("start", "production-restore")
        if value.get("profile") != "qwen" or value.get("state") != "ready" or value.get("active_profile") != "qwen":
            raise LiveOperationError("production_restore_response_invalid", ambiguous=True)

    def _inspect_exact_restore(self, prestate: Mapping[str, Any]) -> dict[str, Any]:
        if prestate is not self._prestate:
            raise LiveOperationError("restore_prestate_mismatch", ambiguous=True)
        current_files = [_file_identity(path) for path in PRODUCTION_FILES]
        current = self._production_identities()
        expected = prestate.get("containers")
        if current_files != prestate.get("files") or current != expected:
            raise LiveOperationError("production_exact_restore_failed", ambiguous=True)
        supervisor = self._supervisor_call("status")
        if supervisor.get("active_profile") != "qwen" or not isinstance(supervisor.get("status"), dict) or supervisor["status"].get("state") != "ready":
            raise LiveOperationError("production_supervisor_not_restored", ambiguous=True)
        self._power_limit()
        caddy_identity = self._active_caddy_identity()
        if caddy_identity != prestate.get("caddy_identity"):
            raise LiveOperationError("caddy_auth_identity_not_restored", ambiguous=True)
        admission = self._admission_call("status")
        try:
            self._assert_admission(admission, "closed")
        except LiveOperationError as exc:
            raise LiveOperationError("admission_restore_not_reconciled", ambiguous=True) from exc
        before_admission = prestate.get("admission")
        before_epoch = before_admission.get("admission_epoch") if isinstance(before_admission, Mapping) else None
        current_epoch = admission.get("admission_epoch")
        if (
            isinstance(before_epoch, bool)
            or not isinstance(before_epoch, int)
            or isinstance(current_epoch, bool)
            or not isinstance(current_epoch, int)
            or current_epoch <= before_epoch
        ):
            raise LiveOperationError("admission_epoch_not_advanced", ambiguous=True)
        admission_evidence = {key: admission.get(key) for key in (
            "admission_epoch", "phase", "capacity_mode", "restart_pending",
            "reconciliation_required", "queue_depth", "active_ledger_digest",
        )}
        gpu_owner = self._gpu_owner_evidence(QWEN_CONTAINER)
        faults = self._post_restore_fault_evidence()
        return {
            "files": current_files,
            "containers": current,
            "caddy_identity": caddy_identity,
            "admission": admission_evidence,
            "gpu_owner": gpu_owner,
            "runtime_faults": faults,
        }

    def reconcile_exact_restore(self, prestate: Mapping[str, Any]) -> None:
        """Prove an ambiguous restore already committed without mutating runtime."""

        self._inspect_exact_restore(prestate)

    def verify_exact_restore(self, prestate: Mapping[str, Any]) -> str:
        evidence = self._inspect_exact_restore(prestate)
        try:
            routes = self._authenticated_route_evidence()
        except LiveOperationError as exc:
            raise LiveOperationError(exc.code, ambiguous=True) from exc
        if routes["sequence"] != 2:
            raise LiveOperationError("authenticated_route_sequence_invalid", ambiguous=True)
        receipt = {
            "schema_version": "ai-server-acceleration-restore/v2",
            "files": evidence["files"],
            "containers": evidence["containers"],
            "caddy_identity": evidence["caddy_identity"],
            "admission": evidence["admission"],
            "authenticated_routes": routes,
            "gpu_owner": evidence["gpu_owner"],
            "runtime_faults": evidence["runtime_faults"],
            "power_limit_w": POWER_LIMIT_W,
        }
        payload = _canonical(receipt) + b"\n"
        try:
            _write_new_fsynced(self._operation_dir() / "restore.json", payload, 0o400)
        except (FileExistsError, OSError, LiveOperationError) as exc:
            raise LiveOperationError("restore_receipt_write_failed", ambiguous=True) from exc
        return hashlib.sha256(payload).hexdigest()

    def production_qwen_completion(self, ordinal: int) -> None:
        self.production_probe.completion(ordinal)

    def open_admission(self) -> None:
        value = self._admission_advance("public")
        try:
            self._assert_admission(value, "public")
        except LiveOperationError as exc:
            raise LiveOperationError(exc.code, ambiguous=True) from exc

    def write_receipt(self, operation_id: str, receipt: Mapping[str, Any]) -> str:
        expected_keys = {
            "schema_version", "operation_id", "candidate_id", "candidate_pin",
            "preflight_receipt_sha256", "state", "passed",
            "original_error_code", "recovery_error_code",
            "candidate_inventory_sha256", "restored_receipt_sha256",
            "arm_results", "content_recorded",
        }
        if (
            operation_id != self.operation_id
            or set(receipt) != expected_keys
            or receipt.get("schema_version") != "ai-server-acceleration-operation/v1"
            or receipt.get("operation_id") != self.operation_id
            or receipt.get("candidate_id") != self.candidate_id
            or receipt.get("candidate_pin") != self.candidate_pin
            or receipt.get("content_recorded") is not False
            or receipt.get("state") not in {"passed_restored", "failed_preproduction", "failed_restored", "unknown_outcome"}
            or type(receipt.get("passed")) is not bool
            or not isinstance(receipt.get("arm_results"), list)
        ):
            raise LiveOperationError("operation_receipt_invalid")
        for key in ("preflight_receipt_sha256", "candidate_inventory_sha256", "restored_receipt_sha256"):
            value = receipt.get(key)
            if value is not None and (not isinstance(value, str) or SHA256.fullmatch(value) is None):
                raise LiveOperationError("operation_receipt_invalid")
        if receipt.get("preflight_receipt_sha256") is None:
            raise LiveOperationError("operation_receipt_invalid")
        if receipt.get("state") == "passed_restored" and (
            receipt.get("passed") is not True
            or receipt.get("candidate_inventory_sha256") is None
            or receipt.get("restored_receipt_sha256") is None
            or receipt.get("original_error_code") is not None
            or receipt.get("recovery_error_code") is not None
        ):
            raise LiveOperationError("operation_receipt_invalid")
        if receipt.get("state") != "passed_restored" and receipt.get("passed") is not False:
            raise LiveOperationError("operation_receipt_invalid")
        for key in ("original_error_code", "recovery_error_code"):
            value = receipt.get(key)
            if value is not None and (not isinstance(value, str) or re.fullmatch(r"[a-z0-9_]+", value) is None):
                raise LiveOperationError("operation_receipt_invalid")
        for arm_result in receipt["arm_results"]:
            if not isinstance(arm_result, Mapping) or set(arm_result) != {"schema_version", "candidate_id", "arm", "passed", "report_sha256", "quality_status"}:
                raise LiveOperationError("operation_receipt_invalid")
            if (
                arm_result.get("schema_version") != "ai-server-acceleration-arm-result/v1"
                or arm_result.get("candidate_id") != self.candidate_id
                or (self.candidate_id, arm_result.get("arm")) not in _ARMS
                or type(arm_result.get("passed")) is not bool
                or not isinstance(arm_result.get("report_sha256"), str)
                or SHA256.fullmatch(arm_result["report_sha256"]) is None
                or arm_result.get("quality_status") not in {"passed", "failed", "not_evaluated"}
            ):
                raise LiveOperationError("operation_receipt_invalid")
        path = self._operation_dir() / "receipt.json"
        try:
            result = write_immutable_receipt(path, receipt, production_cas_root=PRODUCTION_CAS_ROOT)
        except (CustodyError, OSError) as exc:
            raise LiveOperationError("operation_receipt_write_failed", ambiguous=True) from exc
        digest = result.get("receipt_sha256")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise LiveOperationError("operation_receipt_invalid", ambiguous=True)
        return digest
