#!/usr/bin/env python3
"""Run one bounded, maintenance-window Qwen quality capture on Server A.

This matrix reuses the expanded controller with just-in-time exact INT8 and
DFlash preparation. It never removes data or changes a persistent profile.
"""

from __future__ import annotations

import argparse
try:
    import fcntl
except ModuleNotFoundError:  # permits controller unit tests on the Windows authoring host
    fcntl = None  # type: ignore[assignment]
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import subprocess
import threading
import time
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


BUNDLE = Path(__file__).resolve().parent
SOURCES = BUNDLE / "sources"
HOST_LOCK = Path("/srv/ai-server/state/acceleration-experiments/locks/host.lock")
CAS_LOCK = Path("/run/lock/ai-server-cas-cutover.lock")
REVIEWED_HELPER = Path("/var/lib/ai-server-cutovers/higgs-queue-20260907/deploy_higgs_queue.py")
REVIEWED_HELPER_SHA256 = "80f26a9445da1610cbec836d28f23f46b9fd32077062f80d17ec0c64c3b767dd"
PROFILES = {"qwen", "qwen-dflash2", "pony-v6", "pony-v7", "higgs-tts3"}

GGUF_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:995fe9f237c622ac257905948c3cdde447ab343db22e0686322445b5ec6d3131"
GGUF_IMAGE_ID = "sha256:995fe9f237c622ac257905948c3cdde447ab343db22e0686322445b5ec6d3131"
SYV_IMAGE = "ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
SYV_IMAGE_ID = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
MODEL_ID = "qwen3.8-27b"
GGUF = Path("/srv/ai-server/model-store/sha256/3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec/Qwen3.8-27B-Uncensored-Q4_K_M.gguf")
GGUF_SIZE = 16_810_714_496
BASE_MODEL = Path("/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey/f958acb153eb8c7bbd25638149c8c75185559226/model")
BASE_RECEIPT = BASE_MODEL.parent / "COMPLETE.receipt"
BASE_RECEIPT_SHA256 = "a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a"
ADAPTIVE_MODEL = Path("/srv/ai-server/artifacts/extensions/qwen-syv-k4-4d4166b891489bfc/model")
ADAPTIVE_RECEIPT = ADAPTIVE_MODEL.parent / "COMPLETE.receipt"
ADAPTIVE_RECEIPT_SHA256 = "0031745a1a615256cc37b2a345f2dca2d1744787927d81f04a5752e498e822bb"
PRODUCTION_MODEL = ADAPTIVE_MODEL

RESERVE_BYTES = 150 * 1024**3
MAX_GROWTH_BYTES = 7 * 1024**3
MAX_JSONL_BYTES = 32 * 1024**2
WINDOW_SECONDS = 10800
RESTORE_RESERVE_SECONDS = 900
MODE_SECONDS = 3300
LOAD_SECONDS = 600
CLIENT_TIMEOUT_SECONDS = 180
TEMPERATURE_STOP_C = 79.0
RAM_STOP_BYTES = 512 * 1024**2
EXPECTED_POWER_W = 300.0

MODES = (
    {"id": "fast-mtp-int8-vocab40960", "kind": "syv", "model": Path('/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey-fast-mtp/f958acb153eb8c7bbd25638149c8c75185559226-syv69ba4d0-int8-vocab40960/model'), "image": SYV_IMAGE, "image_id": SYV_IMAGE_ID},
    {"id": "dflash2-fast-n7", "kind": "dflash2", "model": BASE_MODEL, "image": SYV_IMAGE, "image_id": SYV_IMAGE_ID},
    {"id": "dflash2-huge-n7", "kind": "dflash2", "model": BASE_MODEL, "image": SYV_IMAGE, "image_id": SYV_IMAGE_ID},
)
BLOCKED = {
    "historical-adaptive-k4": "historical_weights_and_exact_calibration_missing_helpers_recovered_current_calibration_differs",
}
FULL_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
# Only proven, contained candidate results may allow the next mode to launch.
# Every probe, monitoring, ownership, containment, or harness uncertainty aborts
# the matrix after exact-ID cleanup by default.
NONFATAL_MODE_CODES = {
    "candidate_exit_before_ready",
    "candidate_readiness_timeout",
    "mtp_runtime_not_proven",
    "dflash2_runtime_not_proven",
    "quality_client_failed",
}


class WindowError(RuntimeError):
    def __init__(self, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = details or {}


class WindowAbort(WindowError):
    def __init__(self, code: str, mode_result: dict[str, Any], details: dict[str, Any] | None = None) -> None:
        super().__init__(code, details)
        self.mode_result = mode_result


def sha256(path: Path, limit: int | None = None) -> str:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or (limit is not None and info.st_size > limit):
        raise WindowError("file_identity_invalid", {"path": str(path)})
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise WindowError("module_load_failed", {"file": path.name})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_bundle(manifest_digest: str) -> dict[str, str]:
    manifest_path = BUNDLE / "bundle-manifest-remaining.json"
    if sha256(manifest_path, 64 * 1024) != manifest_digest:
        raise WindowError("bundle_manifest_identity_invalid")
    try:
        value = json.loads(manifest_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WindowError("bundle_manifest_invalid") from exc
    expected = {"server_a_runner.py", "syv_acceleration_runner.py", "benchmark_client.py", "quality_client.py", "quality_suite.py", "quality_expanded_suite.py", "quality_expanded_v2.json"}
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, dict) or set(files) != expected:
        raise WindowError("bundle_manifest_invalid")
    for name, digest in files.items():
        if not isinstance(digest, str) or len(digest) != 64 or sha256(SOURCES / name, 1024 * 1024) != digest:
            raise WindowError("bundle_file_identity_invalid", {"file": name})
    for name,digest in value['other_files'].items():
        if sha256(BUNDLE/name,1024*1024)!=digest:
            raise WindowError('bundle_file_identity_invalid',{'file':name})
    if value.get("suite_id") != "qwen38-quality-expanded-200-v1" or value.get("suite_canonical_sha256") != "2ea25b195358d0d78a1f4e025c6c63e604693bc00ec4cbc5224e4772e77f187f":
        raise WindowError("expanded_manifest_suite_invalid")
    suite_module = load_module("quality_expanded_suite", SOURCES / "quality_expanded_suite.py")
    global EXPECTED_CASE_IDS
    EXPECTED_CASE_IDS = ["warmup"] + [c["case_id"] for c in suite_module.load_suite()["cases"]]
    return files


EXPECTED_CASE_IDS: list[str] = []


def load_runtime_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    import sys
    sys.path.insert(0, str(SOURCES))
    common = load_module("server_a_runner", SOURCES / "server_a_runner.py")
    sys.modules["server_a_runner"] = common
    syv = load_module("syv_acceleration_runner", SOURCES / "syv_acceleration_runner.py")
    sys.modules['syv_acceleration_runner'] = syv
    if sha256(REVIEWED_HELPER, 1024 * 1024) != REVIEWED_HELPER_SHA256:
        raise WindowError("reviewed_helper_identity_invalid")
    helper = load_module("reviewed_admission_helper", REVIEWED_HELPER)
    return common, syv, helper


def require_ro_tree(model: Path, receipt: Path, receipt_digest: str) -> dict[str, Any]:
    for directory in (model, model.parent):
        info = directory.lstat()
        if directory.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid != 0:
            raise WindowError("model_directory_identity_invalid", {"path": str(directory)})
    names = sorted(path.name for path in model.iterdir())
    if not names:
        raise WindowError("model_tree_empty", {"path": str(model)})
    logical = 0
    for child in model.iterdir():
        info = child.lstat()
        if child.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o222:
            raise WindowError("model_file_identity_invalid", {"path": str(child)})
        logical += info.st_size
    actual_receipt = sha256(receipt, 1024 * 1024)
    if actual_receipt != receipt_digest:
        raise WindowError("model_receipt_identity_invalid", {"path": str(receipt)})
    return {"path": str(model), "files": len(names), "logical_bytes": logical, "receipt_sha256": actual_receipt}


def verify_models() -> dict[str, Any]:
    return {
        "autoround-bf16-mtp": require_ro_tree(BASE_MODEL, BASE_RECEIPT, BASE_RECEIPT_SHA256),
        "current-adaptive-k4": require_ro_tree(ADAPTIVE_MODEL, ADAPTIVE_RECEIPT, ADAPTIVE_RECEIPT_SHA256),
    }


def inspect_image(common: ModuleType, reference: str, expected_id: str) -> None:
    result = common.run(["docker", "image", "inspect", reference, "--format", "{{.Id}}"], timeout=30)
    if result.stdout.decode("ascii").strip() != expected_id:
        raise WindowError("image_identity_invalid", {"image": reference})


def available_bytes(path: Path = Path("/srv/ai-server")) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise WindowError("ram_sample_invalid")


def exact_container(common: ModuleType, name: str) -> dict[str, Any] | None:
    value = common.run(["docker", "inspect", name], timeout=20, check=False)
    if value.returncode != 0:
        return None
    try:
        rows = json.loads(value.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WindowError("container_inspect_invalid", {"container": name}) from exc
    if not isinstance(rows, list) or len(rows) != 1:
        raise WindowError("container_inspect_invalid", {"container": name})
    return rows[0]


def production_identity(common: ModuleType) -> dict[str, Any]:
    value = exact_container(common, "ai-server-qwen")
    if value is None:
        raise WindowError("production_qwen_absent")
    mounts = value.get("Mounts") or []
    if value.get("Image") != SYV_IMAGE_ID or not any(
        item.get("Type") == "bind" and item.get("Source") == str(PRODUCTION_MODEL)
        and item.get("Destination") == "/models/qwen" and item.get("RW") is False
        for item in mounts if isinstance(item, dict)
    ):
        raise WindowError("production_qwen_identity_invalid")
    return {"id": value.get("Id"), "image": value.get("Image"), "model": str(PRODUCTION_MODEL), "running": value.get("State", {}).get("Running")}


def verify_preflight(common: ModuleType, helper: ModuleType) -> tuple[dict[str, Any], str, dict[str, Any]]:
    helper.empty("public")
    runtime, runtime_digest = helper.runtime(PROFILES)
    return runtime, runtime_digest, production_identity(common)


def candidate_name(mode_id: str) -> str:
    return "qwen-quality-remaining-20260908-" + mode_id


def client_name(mode_id: str) -> str:
    return candidate_name(mode_id) + "-client"


def ensure_names_absent(common: ModuleType) -> None:
    for mode in MODES:
        for name in (candidate_name(mode["id"]), client_name(mode["id"])):
            if exact_container(common, name) is not None:
                raise WindowError("stale_owned_container_present", {"container": name})


def created_id(value: subprocess.CompletedProcess[bytes], code: str) -> str:
    try:
        identity = value.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise WindowError(code) from exc
    if FULL_CONTAINER_ID.fullmatch(identity) is None:
        raise WindowError(code)
    return identity


def syv_args(syv: ModuleType, mode: dict[str, Any], name: str) -> list[str]:
    if mode['kind']=='dflash2':
        draft=load_draft(mode,name)
        args=draft.container_args()
        args[1]='create';args.remove('-d')
        pos=args.index('--gpus');args[pos:pos+2]=['--device','nvidia.com/gpu=0']
        args[args.index('VERIFY=1')]='VERIFY=0'
        pos=args.index(mode['image'])
        args[pos:pos]=['--env','NVIDIA_VISIBLE_DEVICES=void','--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
                      '--tmpfs','/app/models:rw,nosuid,size=16m','--log-opt','max-size=4m','--log-opt','max-file=1']
        return args
    syv.CONTAINER = name
    syv.TARGET_MODEL_DIR = mode["model"]
    syv.MODEL_MOUNT = "/app/models/qwen"
    syv.MODEL_ID = MODEL_ID
    syv.PORT = 8080
    syv.MAX_LEN = 57_344
    syv.BUNDLE = SOURCES
    args = syv.container_args()
    if args[:3] != ["docker", "run", "--pull"] or "-d" not in args:
        raise WindowError("syv_container_builder_contract_changed")
    args[1] = "create"
    args.remove("-d")
    pos = args.index("--gpus")
    del args[pos:pos + 2]
    args[pos:pos] = ["--device", "nvidia.com/gpu=0"]
    args[args.index("VERIFY=1")] = "VERIFY=0"
    image_position = args.index(mode["image"])
    args[image_position:image_position] = [
        "--env", "GPU_UTIL=0.93", "--env", "DRAFT_TOKENS=4", "--env", "PREFIX_CACHE=0",
        "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "--tmpfs", "/app/models:rw,nosuid,size=16m", "--log-opt", "max-size=4m", "--log-opt", "max-file=1",
    ]
    return args


def load_draft(mode=None,name=None):
    draft=load_module('matrix_dflash',SOURCES/'syv_leminkozey_dflash2_runner.py')
    draft.MODEL_MOUNT='/app/models/qwen';draft.PORT=8080;draft.BUNDLE=SOURCES
    draft.TARGET_MODEL_DIR=BASE_MODEL
    # Exact tree bound is unchanged; allow measured unrelated host writes
    # separately within the already-budgeted 2 GiB preparation phase.
    draft.DRAFT_HOST_ACTIVITY_ALLOWANCE_BYTES=2*1024**3-draft.DRAFT_TREE_ALLOCATION_BOUND_BYTES
    draft.DRAFT_PREPARATION_PEAK_BOUND_BYTES=2*1024**3
    if name: draft.CONTAINER=name
    if mode and mode['id']=='dflash2-huge-n7':
        draft.PROFILE.update(CTX='huge',DFLASH_MAX_LEN='245760',GPU_UTIL='0.97')
        draft.EXPECTED_FAST_PROCESS_ARGS.pop('attention_backend')
        draft.EXPECTED_FAST_PROCESS_ARGS.update(max_model_len='--max-model-len 245760',
            max_num_seqs='--max-num-seqs 2',kv_cache_dtype='--kv-cache-dtype kvarn_k4v2_g128',
            block_size='--block-size 128',mamba_cache='--mamba-ssm-cache-dtype float16',
            batching='--max-num-batched-tokens 2048')
    draft.EXPECTED_FAST_PROCESS_ARGS['gpu_util']='--gpu-memory-utilization '+draft.PROFILE['GPU_UTIL']
    return draft


def prepare_candidate(common,mode):
    prep=load_module('matrix_preparation',BUNDLE/'prepare_remaining.py')
    if mode['kind']=='syv':
        return prep.prepare_fast(sys.modules[__name__],common)
    source_evidence=prep.verify_source()
    draft=load_draft()
    prep.checked_directory(draft.DRAFT_LEAF)
    return {**draft.prepare_draft(),'target_payload_verification':source_evidence}


def profile_evidence(common,syv,mode,name,deadline):
    if mode['kind']=='dflash2':
        draft=load_draft(mode,name)
        evidence=draft.startup_evidence(deadline)
        value=exact_container(common,name)
        env=set(value['Config'].get('Env') or [])
        binds={(m.get('Source'),m.get('Destination'),m.get('RW')) for m in value['Mounts']}
        evidence['profile_env_match']={f'{k}={v}' for k,v in draft.PROFILE.items()}.issubset(env)
        evidence['drafter_read_only']=(str(draft.DRAFT_MODEL_DIR),draft.DRAFT_MOUNT,False) in binds
        if not all(evidence.get(k) is True for k in ('dflash2_runtime_proven','profile_env_match','drafter_read_only')):
            raise WindowError('dflash2_runtime_not_proven',evidence)
        return evidence
    evidence=syv.startup_evidence(deadline)
    evidence['draft_vocab_runtime_proven']=any('MTP drafter uses a 40960-token draft head' in line for line in evidence.get('matched_lines',[]))
    top=common.run(['docker','top',name,'-eo','pid,args'],timeout=30).stdout.decode('utf-8','replace')
    expected=['--max-model-len 57344','--gpu-memory-utilization 0.93','"method":"mtp"','"num_speculative_tokens":4',
              '--max-num-seqs 8','--attention-backend FLASH_ATTN','--kv-cache-dtype bfloat16','"draft_sample_method":"probabilistic"']
    evidence['process_contract']={arg:arg in top for arg in expected}
    evidence['process_contract']['prefix_cache_disabled']='--enable-prefix-caching' not in top
    if evidence.get('mtp_runtime_proven') is not True or not evidence['draft_vocab_runtime_proven'] or not all(evidence['process_contract'].values()):
        raise WindowError('mtp_runtime_not_proven',evidence)
    return evidence


def gguf_args(mode: dict[str, Any], name: str) -> list[str]:
    return [
        "docker", "create", "--pull", "never", "--name", name, "--restart", "no",
        "--user", "65532:65532", "--device", "nvidia.com/gpu=0", "--network", "none",
        "--ipc", "private", "--read-only", "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
        "--pids-limit", "1024", "--memory", "24g", "--memory-swap", "28g", "--shm-size", "256m",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "--log-opt", "max-size=4m", "--log-opt", "max-file=1",
        "--mount", f"type=bind,src={mode['model']},dst=/models/qwen.gguf,readonly", mode["image"],
        "--model", "/models/qwen.gguf", "--alias", MODEL_ID, "--ctx-size", "16384", "--parallel", "1",
        "--n-gpu-layers", "99", "--jinja", "--reasoning", "off", "--host", "0.0.0.0", "--port", "8080",
        "--timeout", "1800", "--threads-http", "4", "--no-webui",
    ]


def verify_isolation(common: ModuleType, identity: str, model: Path, kind: str, *, running: bool) -> dict[str, Any]:
    value = exact_container(common, identity)
    if value is None:
        raise WindowError("candidate_absent_after_create", {"container_id": identity})
    host, config = value.get("HostConfig"), value.get("Config")
    if not isinstance(host, dict) or not isinstance(config, dict):
        raise WindowError("candidate_contract_invalid", {"container_id": identity})
    device_requests = host.get("DeviceRequests") or []
    cdi = any(item.get("Driver") == "cdi" and item.get("DeviceIDs") == ["nvidia.com/gpu=0"] for item in device_requests if isinstance(item, dict))
    binds = {(item.get("Source"), item.get("Destination"), item.get("RW")) for item in value.get("Mounts", []) if isinstance(item, dict) and item.get("Type") == "bind"}
    expected_destination = "/models/qwen.gguf" if kind == "gguf" else "/app/models/qwen"
    if (host.get("NetworkMode") != "none" or host.get("ReadonlyRootfs") is not True
            or host.get("PortBindings") not in (None, {}) or not cdi
            or (str(model), expected_destination, False) not in binds
            or "ALL" not in (host.get("CapDrop") or [])
            or not any(str(item).startswith("no-new-privileges") for item in host.get("SecurityOpt") or [])):
        raise WindowError("candidate_contract_invalid", {"container_id": identity})
    state = value.get("State") or {}
    if state.get("Running") is not running:
        raise WindowError("candidate_state_invalid", {"container_id": identity})
    if running and (not isinstance(state.get("Pid"), int) or state["Pid"] <= 0):
        raise WindowError("candidate_not_running", {"container_id": identity})
    return {"id": value.get("Id"), "pid": state["Pid"], "network": "none", "ports": False, "model_read_only": True, "cdi_gpu": "nvidia.com/gpu=0"}


def create_candidate(common: ModuleType, args: list[str], model: Path, kind: str) -> str:
    try:
        created = common.run(args, timeout=60, diagnostic_output=True)
    except common.RunError as exc:
        raise WindowError("candidate_create_outcome_unknown", {"runner_error": exc.code}) from exc
    identity = created_id(created, "candidate_create_identity_unknown")
    verify_isolation(common, identity, model, kind, running=False)
    return identity


def start_candidate(common: ModuleType, identity: str, model: Path, kind: str) -> dict[str, Any]:
    try:
        common.run(["docker", "start", identity], timeout=60, diagnostic_output=True)
    except common.RunError as exc:
        raise WindowError("candidate_start_outcome_unknown", {"runner_error": exc.code}) from exc
    return verify_isolation(common, identity, model, kind, running=True)


def wait_gguf(common: ModuleType, name: str, deadline: float) -> float:
    started = time.monotonic()
    while time.monotonic() < deadline:
        value = exact_container(common, name)
        if value is None or value.get("State", {}).get("Running") is not True:
            raise WindowError("candidate_exit_before_ready", {"container": name})
        probe = common.run(["docker", "exec", name, "curl", "-fsS", "http://127.0.0.1:8080/health"], timeout=10, check=False)
        if probe.returncode == 0:
            return round(time.monotonic() - started, 3)
        time.sleep(3)
    raise WindowError("candidate_readiness_timeout", {"container": name})


def write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LoadGuard:
    """Stop one exact owned candidate once if a live safety bound is crossed."""

    def __init__(self, common: ModuleType, container_id: str, free_before: int) -> None:
        self.common = common
        self.container_id = container_id
        self.free_before = free_before
        self.stop = threading.Event()
        self.error: WindowError | None = None
        self.stop_attempted = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            while not self.stop.wait(1.0):
                value = exact_container(self.common, self.container_id)
                if value is None or value.get("Id") != self.container_id:
                    raise WindowError("candidate_identity_lost_during_load")
                state = value.get("State") or {}
                if state.get("Running") is not True:
                    return
                if not isinstance(state.get("Pid"), int) or state["Pid"] <= 0:
                    raise WindowError("candidate_pid_invalid_during_load")
                gpu = self.common.gpu_rows()
                free_now = available_bytes()
                if gpu[4] >= TEMPERATURE_STOP_C:
                    raise WindowError("gpu_temperature_stop", {"temperature_c": gpu[4]})
                if abs(gpu[3] - EXPECTED_POWER_W) > 0.05:
                    raise WindowError("power_limit_drift", {"power_limit_w": gpu[3]})
                if mem_available_bytes() < RAM_STOP_BYTES:
                    raise WindowError("ram_reserve_exhausted")
                if free_now < RESERVE_BYTES or self.free_before - free_now > MAX_GROWTH_BYTES:
                    raise WindowError("storage_budget_exhausted", {"available_bytes": free_now})
        except Exception as exc:
            if isinstance(exc, WindowError):
                self.error = exc
            else:
                self.error = WindowError(
                    "load_guard_probe_failed",
                    {"probe_error": getattr(exc, "code", type(exc).__name__)},
                )
            # This is the only emergency stop.  The exact created ID is used and
            # the stopped record remains for inspection.
            self.stop_attempted = True
            self.common.run(["docker", "stop", "--time", "30", self.container_id], timeout=45, check=False)

    def __enter__(self) -> "LoadGuard":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise WindowError("load_guard_join_failed")
        if self.error is not None:
            raise self.error


def stop_owned_once(common: ModuleType, name: str, expected_id: str, *, mutation_allowed: bool = True) -> dict[str, Any]:
    value = exact_container(common, expected_id)
    if value is None or value.get("Id") != expected_id:
        raise WindowError("owned_container_identity_unknown", {"container": name})
    attempted = False
    if value.get("State", {}).get("Running") is True:
        if not mutation_allowed:
            raise WindowError("owned_container_stop_outcome_unknown", {"container": name})
        attempted = True
        common.run(["docker", "stop", "--time", "30", expected_id], timeout=45, check=False)
    final = exact_container(common, expected_id)
    if final is None or final.get("Id") != expected_id or final.get("State", {}).get("Running") is True:
        raise WindowError("owned_container_stop_not_proven", {"container": name})
    return {"id": expected_id, "stop_attempted": attempted, "retained": True, "exit_code": final.get("State", {}).get("ExitCode")}


def verify_client_isolation(common: ModuleType, identity: str, candidate_id: str, *, running: bool) -> dict[str, Any]:
    value = exact_container(common, identity)
    if value is None or value.get("Id") != identity:
        raise WindowError("client_identity_unknown")
    host = value.get("HostConfig") or {}
    mounts = value.get("Mounts") or []
    tmpfs = host.get("Tmpfs") or {}
    log_config = host.get("LogConfig") or {}
    state = value.get("State") or {}
    if (host.get("NetworkMode") != "container:" + candidate_id or host.get("ReadonlyRootfs") is not True
            or host.get("PortBindings") not in (None, {}) or host.get("DeviceRequests") not in (None, [])
            or log_config.get("Type") != "none" or not {"/cache", "/app/models"}.issubset(tmpfs)
            or (str(SOURCES), "/runner", False) not in {
                (item.get("Source"), item.get("Destination"), item.get("RW"))
                for item in mounts if isinstance(item, dict) and item.get("Type") == "bind"
            } or state.get("Running") is not running):
        raise WindowError("client_contract_invalid")
    if running and (not isinstance(state.get("Pid"), int) or state["Pid"] <= 0):
        raise WindowError("client_pid_invalid")
    return {"id": identity, "pid": state.get("Pid")}


def launch_client(common: ModuleType, mode_id: str, candidate_id: str, output: Path, deadline: float, free_before: int, ownership: dict[str, str | None]) -> dict[str, Any]:
    name = client_name(mode_id)
    args = [
        "docker", "create", "--pull", "never", "--name", name, "--restart", "no", "--log-driver", "none",
        "--network", "container:" + candidate_id, "--read-only", "--user", "0:0", "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL", "--pids-limit", "128", "--memory", "512m", "--memory-swap", "512m",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m", "--tmpfs", "/cache:rw,noexec,nosuid,size=32m",
        "--tmpfs", "/app/models:rw,noexec,nosuid,size=16m", "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--mount", f"type=bind,src={SOURCES},dst=/runner,readonly", "--entrypoint", "/app/venv/bin/python",
        SYV_IMAGE, "-B", "/runner/quality_client.py", "--base-url", "http://127.0.0.1:8080",
        "--model", MODEL_ID, "--mode-id", mode_id, "--timeout", str(CLIENT_TIMEOUT_SECONDS),
        "--suite", "expanded", "--check-context", "llama" if mode_id == "original-gguf" else "vllm",
    ]
    try:
        created = common.run(args, timeout=60, diagnostic_output=True)
    except common.RunError as exc:
        raise WindowError("client_create_outcome_unknown", {"runner_error": exc.code}) from exc
    client_id = created_id(created, "client_create_identity_unknown")
    ownership["client_id"] = client_id
    verify_client_isolation(common, client_id, candidate_id, running=False)
    with output.open("xb", buffering=0) as sink:
        os.chmod(output, 0o600)
        process = subprocess.Popen(["docker", "start", "-a", client_id], stdout=sink, stderr=subprocess.PIPE)
        samples: list[list[float]] = []
        minimum_ram = mem_available_bytes()
        minimum_disk = available_bytes()
        last_sample = 0.0
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    raise WindowError("mode_deadline_exhausted")
                state = exact_container(common, client_id)
                if state is None or state.get("Id") != client_id:
                    raise WindowError("client_identity_changed")
                pid = state.get("State", {}).get("Pid")
                if state.get("State", {}).get("Running") is True and (not isinstance(pid, int) or pid <= 0):
                    raise WindowError("client_pid_invalid")
                candidate = exact_container(common, candidate_id)
                candidate_pid = None if candidate is None else candidate.get("State", {}).get("Pid")
                if (candidate is None or candidate.get("Id") != candidate_id
                        or candidate.get("State", {}).get("Running") is not True
                        or not isinstance(candidate_pid, int) or candidate_pid <= 0):
                    raise WindowError("candidate_not_running_during_client")
                if now - last_sample >= 1.0:
                    row = common.gpu_rows()
                    samples.append(row)
                    if row[4] >= TEMPERATURE_STOP_C:
                        raise WindowError("gpu_temperature_stop", {"temperature_c": row[4]})
                    if abs(row[3] - EXPECTED_POWER_W) > 0.05:
                        raise WindowError("power_limit_drift", {"power_limit_w": row[3]})
                    ram_now = mem_available_bytes()
                    minimum_ram = min(minimum_ram, ram_now)
                    if ram_now < RAM_STOP_BYTES:
                        raise WindowError("ram_reserve_exhausted")
                    free_now = available_bytes()
                    minimum_disk = min(minimum_disk, free_now)
                    if free_now < RESERVE_BYTES or free_before - free_now > MAX_GROWTH_BYTES:
                        raise WindowError("storage_budget_exhausted", {"available_bytes": free_now})
                    if output.stat().st_size > MAX_JSONL_BYTES:
                        raise WindowError("client_output_too_large")
                    last_sample = now
                time.sleep(0.25)
        except BaseException:
            # Leave the exact container mutation to run_mode's single cleanup
            # path.  Ending the CLI process does not remove its retained record.
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
        stderr = process.communicate(timeout=5)[1] or b""
    state = exact_container(common, client_id)
    if state is None or state.get("State", {}).get("Running") is True or state.get("Id") != client_id:
        raise WindowError("client_completion_not_proven")
    verify_client_isolation(common, client_id, candidate_id, running=False)
    if output.stat().st_size > MAX_JSONL_BYTES:
        raise WindowError("client_output_too_large")
    if process.returncode != 0:
        raise WindowError("quality_client_failed", {"returncode": process.returncode, "stderr": common.sanitized_tail(stderr)})
    rows = output.read_text(encoding="utf-8").splitlines()
    if len(rows) != 201:
        raise WindowError("quality_client_row_count_invalid", {"rows": len(rows)})
    parsed = [json.loads(line) for line in rows]
    case_ids = [row.get("case_id") for row in parsed]
    if len(EXPECTED_CASE_IDS) != 201 or case_ids != EXPECTED_CASE_IDS:
        raise WindowError("quality_client_case_set_invalid")
    if any(row.get("schema") != "qwen38-quality-jsonl/v1" or row.get("mode_id") != mode_id for row in parsed):
        raise WindowError("quality_client_protocol_invalid")
    return {
        "id": client_id, "pid_observed": True, "exit_code": process.returncode,
        "jsonl": str(output), "jsonl_bytes": output.stat().st_size,
        "rows": len(rows), "sha256": sha256(output, MAX_JSONL_BYTES),
        "resources": {
            "samples": len(samples), "max_vram_used_mib": max(row[0] for row in samples),
            "max_power_w": max(row[2] for row in samples), "max_temperature_c": max(row[4] for row in samples),
            "min_host_mem_available_bytes": minimum_ram, "min_storage_available_bytes": minimum_disk,
            "max_storage_growth_bytes": max(0, free_before - minimum_disk),
        },
    }


def run_mode(common: ModuleType, syv: ModuleType, mode: dict[str, Any], output_dir: Path, free_before: int, window_deadline: float) -> dict[str, Any]:
    mode_id, name = mode["id"], candidate_name(mode["id"])
    result: dict[str, Any] = {"status": "PENDING", "attempted": False, "model": str(mode["model"]), "image": mode["image"]}
    candidate_id = None
    ownership: dict[str, str | None] = {"client_id": None}
    load_guard: LoadGuard | None = None
    fatal: WindowError | None = None
    mode_deadline = min(window_deadline - RESTORE_RESERVE_SECONDS, time.monotonic() + MODE_SECONDS)
    try:
        result["attempted"] = True
        args = gguf_args(mode, name) if mode["kind"] == "gguf" else syv_args(syv, mode, name)
        candidate_id = create_candidate(common, args, mode["model"], mode["kind"])
        contract = start_candidate(common, candidate_id, mode["model"], mode["kind"])
        result["runtime_contract"] = contract
        ready_deadline = min(mode_deadline, time.monotonic() + LOAD_SECONDS)
        with LoadGuard(common, candidate_id, free_before) as active_guard:
            load_guard = active_guard
            if mode["kind"] == "gguf":
                result["load_seconds"] = wait_gguf(common, name, ready_deadline)
                result["startup_evidence"] = {"argv": gguf_args(mode, name)[-25:], "reasoning": "off"}
            else:
                syv.CONTAINER = name
                syv.PORT = 8080
                result["load_seconds"] = syv.wait_ready(ready_deadline)
                result["startup_evidence"] = profile_evidence(common,syv,mode,name,ready_deadline)
        client = launch_client(common, mode_id, candidate_id, output_dir / f"{mode_id}.jsonl", mode_deadline, free_before, ownership)
        result["client"] = client
        result["status"] = "CAPTURED"
    except (WindowError, getattr(common, "RunError")) as exc:
        result["status"] = "FAIL"
        result["reason"] = getattr(exc, "code", type(exc).__name__)
        details = getattr(exc, "details", None)
        if details:
            result["details"] = details
        if result["reason"] not in NONFATAL_MODE_CODES:
            fatal = WindowError(result["reason"], details or {})
    finally:
        cleanup: dict[str, Any] = {}
        if ownership["client_id"] is not None:
            cleanup["client"] = stop_owned_once(common, client_name(mode_id), ownership["client_id"])
        if candidate_id is not None:
            cleanup["candidate"] = stop_owned_once(
                common, name, candidate_id,
                mutation_allowed=load_guard is None or not load_guard.stop_attempted,
            )
        common.wait_gpu_idle(90)
        result["cleanup"] = cleanup
    if fatal is not None:
        raise WindowAbort(fatal.code, result, fatal.details)
    return result


def restore(common: ModuleType, helper: ModuleType, operator: Any, production_before: dict[str, Any], close_attempted: bool) -> dict[str, Any]:
    status = helper.operator()
    phase = status.get("phase")
    if close_attempted and phase == "closed":
        operator.qwen("start")
        if common.qwen_state(operator.qwen_status()) != "ready":
            raise WindowError("qwen_restore_not_ready")
        current = production_identity(common)
        if current["id"] != production_before["id"]:
            raise WindowError("production_qwen_container_changed")
        helper.operator("advance-phase", phase="smoke")
        helper.empty("smoke")
        script = common.RESTORE_SCRIPT
        needle = b"'temperature':0,'max_tokens':64,'stream':False}"
        replacement = b"'temperature':0,'max_tokens':64,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}"
        if not isinstance(script, bytes) or script.count(needle) != 1 or b"chat_template_kwargs" in script:
            raise WindowError("restore_script_contract_changed")
        common.RESTORE_SCRIPT = script.replace(needle, replacement)
        common.restore_completion(1)
        common.restore_completion(2)
        helper.operator("advance-phase", phase="public")
        helper.empty("public")
        if common.qwen_state(operator.qwen_status()) != "ready":
            raise WindowError("qwen_not_ready_after_public")
        final = production_identity(common)
        if final["id"] != production_before["id"] or final["image"] != production_before["image"] or final["model"] != production_before["model"]:
            raise WindowError("production_qwen_identity_changed_after_public")
        return {"qwen_ready": True, "container": final, "gateway_completions": 2, "admission_public": True,
                "thinking_disabled_in_markers": True}
    if not close_attempted and phase == "public":
        helper.empty("public")
        return {"qwen_ready": False, "gateway_completions": 0, "admission_public": True}
    if close_attempted:
        raise WindowError("restore_phase_unknown", {"phase": phase})
    raise WindowError("restore_phase_unknown", {"phase": phase})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("root_required")
    if fcntl is None:
        raise SystemExit("linux_required")
    if args.output_dir.exists() or args.output_dir.is_symlink() or not args.output_dir.parent.is_dir():
        raise SystemExit("output_dir_invalid")
    os.mkdir(args.output_dir, 0o700)
    os.chmod(args.output_dir, 0o700)
    report: dict[str, Any] = {
        "schema": "qwen38-quality-window/v1", "suite_id": "qwen38-quality-expanded-200-v1", "started_epoch": int(time.time()),
        "source_commit": "ee209715492a4e4b004e9377103941ba821727ea",
        "content_recorded": True, "secrets_recorded": False,
        "limits": {"window_seconds": WINDOW_SECONDS, "restore_reserve_seconds": RESTORE_RESERVE_SECONDS,
                   "mode_seconds": MODE_SECONDS, "case_timeout_seconds": CLIENT_TIMEOUT_SECONDS,
                   "storage_reserve_bytes": RESERVE_BYTES, "max_growth_bytes": MAX_GROWTH_BYTES},
        "modes": {key: {"status": "BLOCKED", "attempted": False, "reason": reason} for key, reason in BLOCKED.items()},
        "outcome": "running", "error_code": None, "restore_error_code": None,
    }
    common = syv = helper = None
    close_attempted = False
    production_before: dict[str, Any] | None = None
    locks: list[int] = []
    window_deadline = time.monotonic() + WINDOW_SECONDS
    interrupted = {"signal": None}

    def on_signal(number: int, _frame: Any) -> None:
        interrupted["signal"] = number
        raise WindowError("signal_received", {"signal": number})

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, on_signal)
    try:
        verify_bundle(args.bundle_manifest_sha256)
        common, syv, helper = load_runtime_modules()
        for lock_path in (CAS_LOCK, HOST_LOCK):
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(descriptor)
                raise WindowError("experiment_lock_busy", {"path": str(lock_path)}) from exc
            locks.append(descriptor)
        report["runtime_prestate"], report["runtime_prestate_sha256"], production_before = verify_preflight(common, helper)
        report["production_prestate"] = production_before
        report["models"] = verify_models()
        inspect_image(common, SYV_IMAGE, SYV_IMAGE_ID)
        ensure_names_absent(common)
        free_before = available_bytes()
        report["storage_available_before_bytes"] = free_before
        if free_before < RESERVE_BYTES + MAX_GROWTH_BYTES:
            raise WindowError("storage_preflight_insufficient", {"available_bytes": free_before})
        if mem_available_bytes() < RAM_STOP_BYTES:
            raise WindowError("ram_preflight_insufficient")
        gpu = common.gpu_rows()
        if abs(gpu[3] - EXPECTED_POWER_W) > 0.05 or gpu[4] >= TEMPERATURE_STOP_C:
            raise WindowError("gpu_preflight_invalid", {"power_limit_w": gpu[3], "temperature_c": gpu[4]})
        close_attempted = True
        helper.operator("advance-phase", phase="closed")
        helper.empty("closed")
        for mode in MODES:
            report["modes"][mode["id"]] = {"status": "PENDING", "attempted": False}
        for mode in MODES:
            if time.monotonic() >= window_deadline - RESTORE_RESERVE_SECONDS:
                report["modes"][mode["id"]] = {"status": "BLOCKED", "attempted": False, "reason": "window_restore_reserve_reached"}
                continue
            try:
                print(json.dumps({'mode':mode['id'],'phase':'preparation'}),flush=True)
                try:
                    preparation=prepare_candidate(common,mode)
                except RuntimeError as exc:
                    if str(exc).startswith('recovered_bytes_differ_from_history:'):
                        report['modes'][mode['id']]={'status':'BLOCKED','attempted':False,'reason':str(exc),'artifacts_retained':True}
                        continue
                    raise
                report['models'][mode['id']]=preparation
                if available_bytes()<RESERVE_BYTES or free_before-available_bytes()>MAX_GROWTH_BYTES:
                    raise WindowError('preparation_storage_bound_exceeded')
                print(json.dumps({'mode':mode['id'],'phase':'capture'}),flush=True)
                report["modes"][mode["id"]] = run_mode(common, syv, mode, args.output_dir, free_before, window_deadline)
                print(json.dumps({'mode':mode['id'],'phase':'capture_finished','status':report['modes'][mode['id']]['status']}),flush=True)
            except WindowAbort as exc:
                report["modes"][mode["id"]] = exc.mode_result
                raise
        report["outcome"] = "captured_restoration_pending"
    except BaseException as exc:
        report["error_code"] = getattr(exc, "code", str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__)
        details = getattr(exc, "details", None)
        if details:
            report["error_details"] = details
        for mode in MODES:
            row = report["modes"].get(mode["id"])
            if not isinstance(row, dict) or row.get("status") == "PENDING":
                report["modes"][mode["id"]] = {
                    "status": "BLOCKED", "attempted": False,
                    "reason": "window_aborted_before_launch", "abort_code": report["error_code"],
                }
    finally:
        if common is not None and helper is not None and production_before is not None:
            try:
                # Refuse to overlap production with any candidate whose stopped state is unproven.
                for mode in MODES:
                    for name in (candidate_name(mode["id"]), client_name(mode["id"])):
                        value = exact_container(common, name)
                        if value is not None and value.get("State", {}).get("Running") is True:
                            raise WindowError("owned_container_still_running", {"container": name})
                common.wait_gpu_idle(90)
                report["restore"] = restore(common, helper, common.Operator(), production_before, close_attempted)
            except BaseException as exc:
                report["restore_error_code"] = getattr(exc, "code", type(exc).__name__)
                details = getattr(exc, "details", None)
                if details:
                    report["restore_error_details"] = details
        for descriptor in reversed(locks):
            os.close(descriptor)
        report["storage_available_after_bytes"] = available_bytes()
        report["ended_epoch"] = int(time.time())
        if report.get("restore_error_code"):
            report["outcome"] = "restore_unknown"
        elif report.get("error_code"):
            report["outcome"] = "failed_restored"
        elif report.get("restore", {}).get("admission_public") is True:
            report["outcome"] = "captured_restored"
        write_exclusive(args.output_dir / "window-report.json", json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    print(json.dumps({"outcome": report["outcome"], "report": str(args.output_dir / "window-report.json")}, sort_keys=True))
    return 0 if report["outcome"] == "captured_restored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
