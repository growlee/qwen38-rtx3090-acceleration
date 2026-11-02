#!/usr/bin/env python3
"""One bounded SYV DFlash2 run for the sealed Leminkozey target."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import urllib.request
from pathlib import Path
from typing import Any

import server_a_runner as common
import syv_acceleration_runner as base


TARGET_REPOSITORY = base.TARGET_REPOSITORY
TARGET_REVISION = base.TARGET_REVISION
TARGET_MODEL_DIR = base.TARGET_MODEL_DIR
TARGET_RECEIPT_SHA256 = base.TARGET_RECEIPT_SHA256

IMAGE = base.IMAGE
IMAGE_REVISION = base.IMAGE_LABELS["org.opencontainers.image.revision"]
CURRENT_SOURCE_REVISION = "bb739e4e5d6bc4fbdcea9e583bfcf190e245b032"
COMPATIBILITY_SOURCE = "syv-ai/qwen38-27b-rtx3090#45"
CONTAINER = "ai-exp-syv-leminkozey-dflash2"
MODEL_ID = base.MODEL_ID
PORT = base.PORT
MODEL_MOUNT = base.MODEL_MOUNT
DRAFT_MOUNT = "/app/models/Qwen3.8-27B-DFlash2-W4A16"

DRAFT_REPOSITORY = "syvai/Qwen3.8-27B-DFlash2-W4A16"
DRAFT_REVISION = "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279"
DRAFT_LEAF = (
    Path("/srv/ai-server/staging/experiments/qwen38-syv-leminkozey-dflash2")
    / TARGET_REVISION
    / "drafter"
    / DRAFT_REVISION
)
DRAFT_MODEL_DIR = DRAFT_LEAF / "model"
DRAFT_RECEIPT = DRAFT_LEAF / "COMPLETE.receipt"
DRAFT_FILES = {
    "config.json": (2_264, "61d6276fe8d76295232cb02d26cbb0d29c25565911f50441e779c88c9220c556"),
    "model.safetensors": (
        1_280_633_960,
        "ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7",
    ),
}
DRAFT_LOGICAL_BYTES = sum(size for size, _digest in DRAFT_FILES.values())
DRAFT_RECEIPT_BYTES = (
    "complete=true\n"
    f"repo={DRAFT_REPOSITORY}\n"
    f"revision={DRAFT_REVISION}\n"
    f"logical_payload_bytes={DRAFT_LOGICAL_BYTES}\n"
    + "".join(
        f"{digest}  {size}  {name}\n"
        for name, (size, digest) in sorted(DRAFT_FILES.items())
    )
).encode("ascii")
DRAFT_RECEIPT_SHA256 = hashlib.sha256(DRAFT_RECEIPT_BYTES).hexdigest()
FILESYSTEM_BLOCK_BYTES = 4096
DRAFT_FILE_ALLOCATION_SLACK_BLOCKS = 1


def _block_allocation_bound(size: int) -> int:
    return ((size + FILESYSTEM_BLOCK_BYTES - 1) // FILESYSTEM_BLOCK_BYTES) * FILESYSTEM_BLOCK_BYTES


DRAFT_TREE_ALLOCATION_BOUND_BYTES = (
    sum(
        _block_allocation_bound(size)
        + DRAFT_FILE_ALLOCATION_SLACK_BLOCKS * FILESYSTEM_BLOCK_BYTES
        for size, _digest in DRAFT_FILES.values()
    )
    + _block_allocation_bound(len(DRAFT_RECEIPT_BYTES))
    + 2 * FILESYSTEM_BLOCK_BYTES
)
# Root-free snapshots include bounded writes from the still-running Qwen/gateway
# services. Keep that measurement allowance separate from the exact draft tree.
DRAFT_HOST_ACTIVITY_ALLOWANCE_BYTES = 256 * 1024
DRAFT_PREPARATION_PEAK_BOUND_BYTES = (
    DRAFT_TREE_ALLOCATION_BOUND_BYTES + DRAFT_HOST_ACTIVITY_ALLOWANCE_BYTES
)

PROFILE = {
    "SPEC": "dflash2",
    "CTX": "fast",
    "PREFIX_CACHE": "0",
    "DFLASH_MAX_LEN": "65536",
    "DFLASH_TOKENS": "7",
    "GPU_UTIL": "0.93",
    "VISION": "0",
}
MODE_ID = "leminkozey-dflash2-fast-n7"
EXPECTED_FAST_PROCESS_ARGS = {
    "max_model_len": "--max-model-len 65536",
    "max_num_seqs": "--max-num-seqs 8",
    "attention_backend": "--attention-backend FLASH_ATTN",
    "kv_cache_dtype": "--kv-cache-dtype bfloat16",
    "spec_method": '"method":"dflash"',
    "draft_model": f'"model":"{DRAFT_MOUNT}"',
    "draft_tokens": '"num_speculative_tokens":7',
}
MIN_SPEEDUP_PERCENT = 5.0
RESERVE_BYTES = base.RESERVE_BYTES
LIVE_BUDGET_SECONDS = base.LIVE_BUDGET_SECONDS
STOP_RESERVE_SECONDS = base.STOP_RESERVE_SECONDS
BUNDLE = Path(__file__).resolve().parent
DFLASH_RUNTIME_EVIDENCE = re.compile(
    r"(?i)(?=.*(?:dflash2|dflash[_ -]?2))(?=.*(?:speculative|spec_decode|drafter|draft[_ -]?model|method))"
)
EXPECTED_CASE_IDS = {"operations-guide-ru", "architecture-analysis-en", "python-service-code"}
BASELINE_REPORT = Path(
    "/srv/ai-server/benchmarks/experiments/"
    "syv-leminkozey-int4-fast-mtp-adaptive-live09r5-990f93f2.json"
)
BASELINE_REPORT_SHA256 = "454ecd10b26edbfe26cd5e3cc73f5b09ae80b790c604986c858dbfe4892c64f6"
BASELINE_RESULT_KEYS = ("benchmark", "result")
BASELINE_DECODE_TPS = 103.615801
BASELINE_END_TO_END_TPS = 102.115791
BASELINE_TTFT_SECONDS = 0.134937
BASELINE_ACCEPTANCE = 0.339793


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise common.RunError("draft_file_read_failed", {"file": path.name}) from exc
    return digest.hexdigest()


def _exact_draft_file(path: Path, size: int, digest: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise common.RunError("draft_file_missing", {"file": path.name}) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_size != size
        or _sha256_file(path) != digest
    ):
        raise common.RunError("draft_file_identity_invalid", {"file": path.name})
    return info


def verify_draft() -> str:
    try:
        leaf_info = DRAFT_LEAF.lstat()
        model_info = DRAFT_MODEL_DIR.lstat()
    except OSError as exc:
        raise common.RunError("draft_candidate_missing") from exc
    for path, info in ((DRAFT_LEAF, leaf_info), (DRAFT_MODEL_DIR, model_info)):
        if (
            path.is_symlink()
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != 0o555
        ):
            raise common.RunError("draft_directory_identity_invalid")
    if {item.name for item in DRAFT_LEAF.iterdir()} != {"model", "COMPLETE.receipt"}:
        raise common.RunError("draft_file_set_invalid")
    if {item.name for item in DRAFT_MODEL_DIR.iterdir()} != set(DRAFT_FILES):
        raise common.RunError("draft_file_set_invalid")
    receipt_info = _exact_draft_file(DRAFT_RECEIPT, len(DRAFT_RECEIPT_BYTES), DRAFT_RECEIPT_SHA256)
    if DRAFT_RECEIPT.read_bytes() != DRAFT_RECEIPT_BYTES:
        raise common.RunError("draft_receipt_identity_invalid")
    allocated = leaf_info.st_blocks * 512 + model_info.st_blocks * 512 + receipt_info.st_blocks * 512
    for name, (size, digest) in DRAFT_FILES.items():
        allocated += _exact_draft_file(DRAFT_MODEL_DIR / name, size, digest).st_blocks * 512
    if allocated > DRAFT_TREE_ALLOCATION_BOUND_BYTES:
        raise common.RunError(
            "draft_tree_allocation_bound_exceeded",
            {
                "allocated_bytes": allocated,
                "allocation_bound_bytes": DRAFT_TREE_ALLOCATION_BOUND_BYTES,
            },
        )
    return DRAFT_RECEIPT_SHA256


def finalize_downloaded_draft() -> str:
    """Seal exact completed downloads retained by a pre-seal harness failure."""
    try:
        leaf_info = DRAFT_LEAF.lstat()
        model_info = DRAFT_MODEL_DIR.lstat()
    except OSError as exc:
        raise common.RunError("draft_resume_candidate_missing") from exc
    if (
        DRAFT_LEAF.is_symlink()
        or not stat.S_ISDIR(leaf_info.st_mode)
        or leaf_info.st_uid != 0
        or leaf_info.st_gid != 0
        or stat.S_IMODE(leaf_info.st_mode) != 0o755
        or DRAFT_MODEL_DIR.is_symlink()
        or not stat.S_ISDIR(model_info.st_mode)
        or model_info.st_uid != 0
        or model_info.st_gid != 0
        or stat.S_IMODE(model_info.st_mode) != 0o700
        or DRAFT_RECEIPT.exists()
        or DRAFT_RECEIPT.is_symlink()
    ):
        raise common.RunError("draft_resume_identity_invalid")
    if {item.name for item in DRAFT_LEAF.iterdir()} != {"model"}:
        raise common.RunError("draft_resume_file_set_invalid")
    if {item.name for item in DRAFT_MODEL_DIR.iterdir()} != set(DRAFT_FILES):
        raise common.RunError("draft_resume_file_set_invalid")
    allocated = leaf_info.st_blocks * 512 + model_info.st_blocks * 512
    for name, (size, digest) in DRAFT_FILES.items():
        allocated += _exact_draft_file(DRAFT_MODEL_DIR / name, size, digest).st_blocks * 512
    receipt_allocation = _block_allocation_bound(len(DRAFT_RECEIPT_BYTES))
    if allocated + receipt_allocation > DRAFT_TREE_ALLOCATION_BOUND_BYTES:
        raise common.RunError(
            "draft_tree_allocation_bound_exceeded",
            {
                "allocated_bytes": allocated,
                "allocation_bound_bytes": DRAFT_TREE_ALLOCATION_BOUND_BYTES,
            },
        )
    available = _available_bytes()
    if available < RESERVE_BYTES + receipt_allocation:
        raise common.RunError(
            "draft_resume_storage_budget_not_met",
            {
                "available_bytes": available,
                "required_bytes": RESERVE_BYTES + receipt_allocation,
            },
        )
    try:
        with DRAFT_RECEIPT.open("xb") as handle:
            handle.write(DRAFT_RECEIPT_BYTES)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(DRAFT_RECEIPT, 0o444)
        os.chmod(DRAFT_MODEL_DIR, 0o555)
        os.chmod(DRAFT_LEAF, 0o555)
        _fsync_directory(DRAFT_LEAF)
        _fsync_directory(DRAFT_LEAF.parent)
    except OSError as exc:
        raise common.RunError("draft_resume_publish_failed") from exc
    return verify_draft()


def _available_bytes() -> int:
    return base.available_bytes()


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _retained_details(path: Path) -> dict[str, Any]:
    try:
        retained_bytes = path.lstat().st_size
    except OSError:
        retained_bytes = 0
    return {"retained_path": str(path), "retained_bytes": retained_bytes}


def _retained_download_path(partial: Path, final: Path) -> Path:
    if partial.exists() or partial.is_symlink():
        return partial
    if final.exists() or final.is_symlink():
        return final
    return partial


def rename_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic rename that refuses an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise common.RunError("rename_noreplace_unavailable", _retained_details(source))
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        raise common.RunError(
            "rename_noreplace_failed",
            {"errno": ctypes.get_errno(), "file": destination.name, **_retained_details(source)},
        )


def _download_one(name: str, size: int, digest: str) -> None:
    final = DRAFT_MODEL_DIR / name
    partial = final.with_name(final.name + ".partial")
    if final.exists() or final.is_symlink() or partial.exists() or partial.is_symlink():
        retained = partial if partial.exists() or partial.is_symlink() else final
        raise common.RunError(
            "draft_staging_path_not_empty", {"file": name, **_retained_details(retained)}
        )
    url = f"https://huggingface.co/{DRAFT_REPOSITORY}/resolve/{DRAFT_REVISION}/{name}?download=true"
    calculated = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("xb") as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                calculated.update(chunk)
                written += len(chunk)
                if written > size:
                    raise common.RunError("draft_download_identity_invalid", {"file": name})
            handle.flush()
            os.fsync(handle.fileno())
        if written != size or calculated.hexdigest() != digest:
            raise common.RunError("draft_download_identity_invalid", {"file": name})
        os.chmod(partial, 0o444)
        rename_noreplace(partial, final)
        _fsync_directory(DRAFT_MODEL_DIR)
    except common.RunError as exc:
        retained = _retained_download_path(partial, final)
        raise common.RunError(exc.code, {**exc.details, **_retained_details(retained)}) from exc
    except (OSError, ValueError) as exc:
        retained = _retained_download_path(partial, final)
        raise common.RunError(
            "draft_download_failed", {"file": name, **_retained_details(retained)}
        ) from exc


def prepare_draft() -> dict[str, Any]:
    if DRAFT_LEAF.exists() or DRAFT_LEAF.is_symlink():
        if DRAFT_RECEIPT.exists() or DRAFT_RECEIPT.is_symlink():
            return {
                "reused": True,
                "resumed": False,
                "receipt_sha256": verify_draft(),
                "downloaded_files": 0,
            }
        return {
            "reused": True,
            "resumed": True,
            "receipt_sha256": finalize_downloaded_draft(),
            "downloaded_files": 0,
        }
    available_before = _available_bytes()
    if available_before < RESERVE_BYTES + DRAFT_PREPARATION_PEAK_BOUND_BYTES:
        raise common.RunError(
            "draft_storage_budget_not_met",
            {
                "available_bytes": available_before,
                "required_bytes": RESERVE_BYTES + DRAFT_PREPARATION_PEAK_BOUND_BYTES,
            },
        )
    try:
        DRAFT_MODEL_DIR.mkdir(parents=True, mode=0o700)
    except OSError as exc:
        raise common.RunError("draft_staging_create_failed") from exc
    phase_start = _available_bytes()
    for name, (size, digest) in sorted(DRAFT_FILES.items()):
        if _available_bytes() < RESERVE_BYTES + size:
            raise common.RunError("draft_storage_reserve_would_be_crossed", {"file": name})
        _download_one(name, size, digest)
        phase_growth = max(0, phase_start - _available_bytes())
        if phase_growth > DRAFT_PREPARATION_PEAK_BOUND_BYTES or _available_bytes() < RESERVE_BYTES:
            raise common.RunError("draft_phase_growth_exceeded", {"observed_growth_bytes": phase_growth})
    try:
        with DRAFT_RECEIPT.open("xb") as handle:
            handle.write(DRAFT_RECEIPT_BYTES)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(DRAFT_RECEIPT, 0o444)
        os.chmod(DRAFT_MODEL_DIR, 0o555)
        os.chmod(DRAFT_LEAF, 0o555)
        _fsync_directory(DRAFT_LEAF)
        _fsync_directory(DRAFT_LEAF.parent)
    except OSError as exc:
        raise common.RunError("draft_receipt_publish_failed") from exc
    receipt_sha256 = verify_draft()
    available_after = _available_bytes()
    observed_growth = max(0, available_before - available_after)
    if observed_growth > DRAFT_PREPARATION_PEAK_BOUND_BYTES or available_after < RESERVE_BYTES:
        raise common.RunError("draft_phase_growth_exceeded", {"observed_growth_bytes": observed_growth})
    return {
        "reused": False,
        "resumed": False,
        "receipt_sha256": receipt_sha256,
        "downloaded_files": 2,
        "available_before_bytes": available_before,
        "available_after_bytes": available_after,
        "observed_growth_bytes": observed_growth,
        "preparation_peak_bound_bytes": DRAFT_PREPARATION_PEAK_BOUND_BYTES,
    }


def container_args() -> list[str]:
    env = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "DO_NOT_TRACK": "1", "HOME": "/cache", "XDG_CACHE_HOME": "/cache",
        "MODELS_DIR": "/app/models", "MODEL": MODEL_MOUNT, "DRAFT": DRAFT_MOUNT, "PORT": str(PORT),
        "PREPARE": "0", "VERIFY": "1", "VLLM_OFFLOAD_KEEP_SHM": "1", **PROFILE,
    }
    argv = [
        "docker", "run", "--pull", "never", "-d", "--name", CONTAINER, "--restart", "no",
        "--user", "0:0", "--gpus", "1", "--network", "none", "--ipc", "private", "--read-only",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--pids-limit", "2048",
        "--shm-size", "16g", "--memory", "30g", "--memory-swap", "32g",
        "--tmpfs", "/tmp:rw,nosuid,size=4g", "--tmpfs", "/cache:rw,exec,nosuid,size=4g",
    ]
    for key, value in env.items():
        argv.extend(("--env", f"{key}={value}"))
    argv.extend((
        "--mount", f"type=bind,src={TARGET_MODEL_DIR},dst={MODEL_MOUNT},readonly",
        "--mount", f"type=bind,src={DRAFT_MODEL_DIR},dst={DRAFT_MOUNT},readonly",
        "--mount", f"type=bind,src={BUNDLE},dst=/runner,readonly", IMAGE, "single",
    ))
    return argv


def verify_image() -> None:
    base.verify_image()
    present = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"], timeout=15
    ).stdout.decode("ascii").strip()
    if present:
        raise common.RunError("stale_candidate_container_present")


def verify_baseline() -> None:
    try:
        info = BASELINE_REPORT.lstat()
    except OSError as exc:
        raise common.RunError("baseline_report_identity_invalid") from exc
    if (
        BASELINE_REPORT.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size > 1024 * 1024
        or _sha256_file(BASELINE_REPORT) != BASELINE_REPORT_SHA256
    ):
        raise common.RunError("baseline_report_identity_invalid")
    try:
        value: Any = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
        for key in BASELINE_RESULT_KEYS:
            value = value[key]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise common.RunError("baseline_report_invalid") from exc
    if (
        value.get("metrics", {}).get("decode_tokens_per_second_median") != BASELINE_DECODE_TPS
        or value.get("metrics", {}).get("end_to_end_tokens_per_second_median") != BASELINE_END_TO_END_TPS
        or value.get("metrics", {}).get("ttft_seconds_median") != BASELINE_TTFT_SECONDS
        or value.get("spec_decode_summary", {}).get("acceptance_ratio") != BASELINE_ACCEPTANCE
    ):
        raise common.RunError("baseline_report_invalid")


def verify_prestate(operator: common.Operator) -> dict[str, Any]:
    status = operator.status()
    if not common.admission_safe(status, "public"):
        raise common.RunError("admission_preflight_not_public_empty")
    qwen = common.qwen_state(operator.qwen_status())
    if qwen not in {"ready", "stopped"}:
        raise common.RunError("qwen_preflight_state_unsupported", {"state": qwen})
    common.verify_production_qwen_identity()
    if qwen == "stopped":
        common.wait_gpu_idle()
    gateway_healthy = base.service_healthy("ai-server-gateway")
    open_webui_healthy = base.service_healthy("ai-server-open-webui")
    if not gateway_healthy or not open_webui_healthy:
        raise common.RunError("user_service_preflight_not_healthy")
    power = common.gpu_rows()[3]
    if abs(power - 300.0) > 0.05:
        raise common.RunError("power_limit_not_300w")
    return {
        "admission_public": True,
        "qwen_state": qwen,
        "gateway_healthy": gateway_healthy,
        "open_webui_healthy": open_webui_healthy,
        "power_limit_w": power,
    }


def verify_container_contract(deadline: float) -> dict[str, Any]:
    raw = common.run(["docker", "inspect", CONTAINER], timeout=base.bounded_timeout(deadline, 30)).stdout
    try:
        values = json.loads(raw.decode("utf-8"))
        value = values[0]
        host, config, mounts = value["HostConfig"], value["Config"], value["Mounts"]
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise common.RunError("candidate_container_contract_invalid") from exc
    env = set(config.get("Env") or [])
    expected_env = {f"{key}={value}" for key, value in PROFILE.items()} | {
        f"MODEL={MODEL_MOUNT}", f"DRAFT={DRAFT_MOUNT}", "MODELS_DIR=/app/models", "PREPARE=0", "VERIFY=1"
    }
    expected_mounts = {
        (str(TARGET_MODEL_DIR), MODEL_MOUNT, False),
        (str(DRAFT_MODEL_DIR), DRAFT_MOUNT, False),
        (str(BUNDLE), "/runner", False),
    }
    actual_mounts = {
        (item.get("Source"), item.get("Destination"), item.get("RW"))
        for item in mounts if isinstance(item, dict) and item.get("Type") == "bind"
    }
    cache_options = str((host.get("Tmpfs") or {}).get("/cache", "")).split(",")
    if (
        host.get("NetworkMode") != "none" or host.get("ReadonlyRootfs") is not True
        or host.get("IpcMode") != "private" or host.get("PidsLimit") != 2048
        or "ALL" not in (host.get("CapDrop") or [])
        or not any(str(item).startswith("no-new-privileges") for item in (host.get("SecurityOpt") or []))
        or "exec" not in cache_options or "noexec" in cache_options
        or host.get("PortBindings") not in (None, {}) or config.get("Cmd") != ["single"]
        or not expected_env.issubset(env) or not expected_mounts.issubset(actual_mounts)
    ):
        raise common.RunError("candidate_container_contract_invalid")
    probe = (
        "import mmap,os; rows=[x.split() for x in open('/proc/mounts')]; "
        "row=next((x for x in rows if len(x)>3 and x[1]=='/cache'),None); "
        "assert row and 'noexec' not in row[3].split(','); p='/cache/codex-exec-map-probe'; "
        "f=open(p,'wb'); f.write(b'\\0'*4096); f.close(); fd=os.open(p,os.O_RDONLY); "
        "m=mmap.mmap(fd,4096,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ|mmap.PROT_EXEC); "
        "m.close(); os.close(fd); os.unlink(p)"
    )
    common.run(
        ["docker", "exec", CONTAINER, "/app/venv/bin/python", "-B", "-c", probe],
        timeout=base.bounded_timeout(deadline, 15, "live_budget_exhausted_during_contract_check"),
    )
    return {
        "network_mode": "none", "cache_exec": True, "target_read_only": True,
        "draft_read_only": True, "draft": DRAFT_MOUNT, **PROFILE,
    }


def startup_evidence(deadline: float) -> dict[str, Any]:
    logs = common.run(["docker", "logs", "--tail", "600", CONTAINER], timeout=base.bounded_timeout(deadline, 30), check=False)
    top = common.run(["docker", "top", CONTAINER, "-eo", "pid,args"], timeout=base.bounded_timeout(deadline, 30), check=False)
    log_text = (logs.stdout + logs.stderr).decode("utf-8", "replace")
    process_text = (top.stdout + top.stderr).decode("utf-8", "replace")
    matches = [
        f"{source}:{common.sanitized_tail(line)[:1500]}"
        for source, raw in (("log", logs.stdout + logs.stderr), ("process", top.stdout + top.stderr))
        for line in raw.decode("utf-8", "replace").splitlines()
        if DFLASH_RUNTIME_EVIDENCE.search(line)
    ]
    process_contract = {
        name: argument in process_text
        for name, argument in EXPECTED_FAST_PROCESS_ARGS.items()
    }
    process_contract["prefix_cache_disabled"] = "--enable-prefix-caching" not in process_text
    architecture_proven = "Resolved architecture: DFlash2DraftModel" in log_text
    payload = "\n".join(matches[:30]).encode("utf-8")
    return {
        "matched_line_count": len(matches),
        "matched_lines_sha256": hashlib.sha256(payload).hexdigest(),
        "matched_lines": matches[:30],
        "architecture_proven": architecture_proven,
        "process_contract": process_contract,
        "dflash2_runtime_proven": bool(matches) and architecture_proven and all(process_contract.values()),
    }


def run_benchmark(deadline: float) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining < 90:
        raise common.RunError("live_budget_insufficient_for_benchmark")
    timeout = min(600.0, max(60.0, remaining - 30.0))
    result = common.run([
        "docker", "exec", "-i", CONTAINER, "/app/venv/bin/python", "-B",
        "/runner/syv_acceleration_benchmark_client.py", "--base-url", f"http://127.0.0.1:{PORT}",
        "--model", MODEL_ID, "--mode-id", MODE_ID, "--timeout", str(timeout),
    ], timeout=max(60.0, remaining), check=False)
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise common.RunError("benchmark_diagnostic_invalid", common.command_diagnostic(result)) from exc
    if result.returncode != 0:
        raise common.RunError("benchmark_client_failed", {
            "client_returncode": result.returncode,
            "benchmark_diagnostic": base.validate_benchmark_diagnostic(value),
        })
    samples = value.get("samples") if isinstance(value, dict) else None
    metrics = value.get("metrics") if isinstance(value, dict) else None
    spec = value.get("spec_decode_summary") if isinstance(value, dict) else None
    numeric_metrics = (
        "decode_tokens_per_second_median", "end_to_end_tokens_per_second_median",
        "end_to_end_seconds_median", "ttft_seconds_median",
    )
    metrics_valid = all(
        isinstance(metrics.get(key), (int, float))
        and not isinstance(metrics.get(key), bool)
        and metrics[key] > 0
        for key in numeric_metrics
    ) if isinstance(metrics, dict) else False
    drafted = spec.get("draft_tokens") if isinstance(spec, dict) else None
    accepted = spec.get("accepted_tokens") if isinstance(spec, dict) else None
    acceptance = spec.get("acceptance_ratio") if isinstance(spec, dict) else None
    spec_valid = (
        isinstance(drafted, (int, float)) and not isinstance(drafted, bool) and drafted > 0
        and isinstance(accepted, (int, float)) and not isinstance(accepted, bool) and 0 <= accepted <= drafted
        and isinstance(acceptance, (int, float)) and not isinstance(acceptance, bool) and 0 <= acceptance <= 1
        and abs(acceptance - accepted / drafted) <= 0.000001
    )
    if (
        value.get("schema") != "qwen38-syv-acceleration-benchmark/v1"
        or value.get("warmup_count") != 1 or value.get("measured_request_count") != 3
        or value.get("max_output_tokens_per_request") != 1024
        or not isinstance(samples, list) or len(samples) != 3
        or {item.get("case_id") for item in samples if isinstance(item, dict)} != EXPECTED_CASE_IDS
        or any(
            not isinstance(item, dict)
            or item.get("completion_tokens") != 1024
            or item.get("finish_reason") != "length"
            for item in samples
        )
        or not metrics_valid or not spec_valid
        or value.get("request_text_recorded") is not False
        or value.get("response_text_recorded") is not False
        or value.get("secrets_recorded") is not False
    ):
        raise common.RunError("benchmark_result_invalid")
    return value


def exact_marker_probe(deadline: float) -> dict[str, Any]:
    marker = "LEMINKOZEY_DFLASH2_OK_" + DRAFT_RECEIPT_SHA256[:12]
    probe = b"""import json,sys,urllib.request
marker=sys.argv[1]
body=json.dumps({'model':'qwen3.8-27b','messages':[{'role':'user','content':'Return exactly: '+marker}],'temperature':0,'max_tokens':64,'stream':False,'chat_template_kwargs':{'enable_thinking':False}},separators=(',',':')).encode()
request=urllib.request.Request('http://127.0.0.1:18020/v1/chat/completions',data=body,headers={'Content-Type':'application/json'},method='POST')
with urllib.request.urlopen(request,timeout=180) as response: value=json.load(response)
choice=value['choices'][0]; content=choice['message']['content']
if choice.get('finish_reason')!='stop' or not isinstance(content,str) or content.strip()!=marker: raise SystemExit(22)
print('EXACT_MARKER_OK')
"""
    result = common.run(
        ["docker", "exec", "-i", CONTAINER, "/app/venv/bin/python", "-B", "-", marker],
        timeout=base.bounded_timeout(deadline, 210, "live_budget_exhausted_during_correctness_probe"),
        input_bytes=probe,
        diagnostic_output=True,
    )
    if result.stdout.decode("ascii").strip() != "EXACT_MARKER_OK":
        raise common.RunError("exact_marker_probe_invalid")
    return {
        "exact_marker": True,
        "marker_sha256": hashlib.sha256(marker.encode("ascii")).hexdigest(),
        "response_text_recorded": False,
    }


def add_baseline_comparison(value: dict[str, Any]) -> None:
    metrics = value["metrics"]
    summary = value["spec_decode_summary"]
    decode = metrics["decode_tokens_per_second_median"]
    end_to_end = metrics["end_to_end_tokens_per_second_median"]
    ttft = metrics["ttft_seconds_median"]
    acceptance = summary["acceptance_ratio"]
    decode_change = round((decode / BASELINE_DECODE_TPS - 1) * 100, 3)
    end_to_end_change = round((end_to_end / BASELINE_END_TO_END_TPS - 1) * 100, 3)
    value["comparison"] = {
        "baseline_report": str(BASELINE_REPORT),
        "baseline_report_sha256": BASELINE_REPORT_SHA256,
        "baseline_decode_tokens_per_second_median": BASELINE_DECODE_TPS,
        "candidate_decode_tokens_per_second_median": decode,
        "decode_change_percent": decode_change,
        "baseline_end_to_end_tokens_per_second_median": BASELINE_END_TO_END_TPS,
        "candidate_end_to_end_tokens_per_second_median": end_to_end,
        "end_to_end_change_percent": end_to_end_change,
        "baseline_ttft_seconds_median": BASELINE_TTFT_SECONDS,
        "candidate_ttft_seconds_median": ttft,
        "ttft_change_percent": round((ttft / BASELINE_TTFT_SECONDS - 1) * 100, 3),
        "baseline_acceptance_ratio": BASELINE_ACCEPTANCE,
        "candidate_acceptance_ratio": acceptance,
        "acceptance_change_percentage_points": round((acceptance - BASELINE_ACCEPTANCE) * 100, 3),
        "minimum_speedup_percent": MIN_SPEEDUP_PERCENT,
        "speed_goal_met": (
            decode_change >= MIN_SPEEDUP_PERCENT
            and end_to_end_change >= MIN_SPEEDUP_PERCENT
        ),
    }


def _with_candidate_container(function: Any, *args: Any, **kwargs: Any) -> Any:
    original = base.CONTAINER
    base.CONTAINER = CONTAINER
    try:
        return function(*args, **kwargs)
    finally:
        base.CONTAINER = original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("root_required")
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise SystemExit("output_path_invalid")

    report: dict[str, Any] = {
        "schema": "qwen38-syv-leminkozey-dflash2-result/v1",
        "started_epoch": int(time.time()), "content_recorded": False, "secrets_recorded": False,
        "coordinates": {
            "image": IMAGE, "source_revision": IMAGE_REVISION,
            "current_source_revision": CURRENT_SOURCE_REVISION,
            "compatibility_source": COMPATIBILITY_SOURCE,
            "target_repository": TARGET_REPOSITORY, "target_revision": TARGET_REVISION,
            "target_receipt_sha256": TARGET_RECEIPT_SHA256,
            "draft_repository": DRAFT_REPOSITORY, "draft_revision": DRAFT_REVISION,
            "draft_receipt_sha256": DRAFT_RECEIPT_SHA256, "profile": PROFILE,
        },
        "storage": {
            "required_reserve_bytes": RESERVE_BYTES,
            "draft_peak_bound_bytes": DRAFT_PREPARATION_PEAK_BOUND_BYTES,
        },
        "mode": {"status": "PENDING", "attempted": False},
        "restore": {
            "candidate_absent": False, "qwen_ready": False, "gateway_completion_markers": [],
            "admission_public": False, "gateway_healthy": False, "open_webui_healthy": False,
            "power_limit_w": None,
        },
        "outcome": "running", "error_code": None, "cleanup_error_code": None,
    }
    operator = common.Operator()
    close_attempted = stop_attempted = candidate_attempted = request_started = False
    prestate_valid = False
    prestate_qwen_state: str | None = None
    error: common.RunError | None = None
    cleanup_error: common.RunError | None = None
    hard_deadline: float | None = None
    since_epoch = int(time.time())
    sampler: common.Sampler | None = None
    descriptor = os.open(common.LOCK, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise common.RunError("host_lock_busy") from exc
        try:
            report["prestate"] = verify_prestate(operator)
            prestate_valid = True
            prestate_qwen_state = report["prestate"]["qwen_state"]
            base.verify_dependencies()
            base.verify_target()
            verify_image()
            verify_baseline()
            report["draft"] = prepare_draft()
            close_attempted = True
            operator.advance("closed")
            if not common.admission_safe(operator.status(), "closed"):
                raise common.RunError("admission_close_not_proven")
            if prestate_qwen_state == "ready":
                stop_attempted = True
                operator.qwen("stop")
            if common.qwen_state(operator.qwen_status()) != "stopped":
                raise common.RunError("qwen_stop_not_proven")
            common.wait_gpu_idle()
            started = time.monotonic()
            hard_deadline = started + LIVE_BUDGET_SECONDS
            work_deadline = hard_deadline - STOP_RESERVE_SECONDS
            with common.Sampler() as active_sampler:
                sampler = active_sampler
                candidate_attempted = True
                report["mode"]["attempted"] = True
                common.run(container_args(), timeout=base.bounded_timeout(work_deadline, 60), diagnostic_output=True)
                contract = verify_container_contract(work_deadline)
                load_seconds = _with_candidate_container(base.wait_ready, work_deadline)
                proof = startup_evidence(work_deadline)
                if proof["dflash2_runtime_proven"] is not True:
                    raise common.RunError("dflash2_runtime_not_proven")
                request_started = True
                benchmark = run_benchmark(work_deadline)
                correctness = exact_marker_probe(work_deadline)
            add_baseline_comparison(benchmark)
            benchmark.update({
                "runtime_contract": contract, "load_seconds": load_seconds, "startup_evidence": proof,
                "correctness": correctness,
                "resources": sampler.report(), "target_receipt_sha256": TARGET_RECEIPT_SHA256,
                "draft_receipt_sha256": DRAFT_RECEIPT_SHA256,
            })
            report["mode"] = {"status": "PASS", "attempted": True, "result": benchmark}
        except common.RunError as exc:
            error = exc
            report["error_code"] = exc.code
            if report["mode"]["status"] == "PENDING":
                report["mode"] = {
                    "status": "FAIL", "attempted": report["mode"]["attempted"], "reason": exc.code,
                    **exc.details,
                    "diagnostics": _with_candidate_container(base.diagnostics, include_logs=not request_started)
                    if candidate_attempted else None,
                    **base.sampler_evidence(sampler),
                }
        except Exception as exc:
            error = common.RunError("unexpected_runner_failure", {"exception_type": type(exc).__name__})
            report["error_code"] = error.code
            report["mode"] = {"status": "FAIL", "attempted": report["mode"]["attempted"], "reason": error.code}
        finally:
            try:
                if candidate_attempted:
                    _with_candidate_container(base.stop_candidate, hard_deadline)
                remaining = common.run([
                    "docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"
                ], timeout=15).stdout.decode("ascii").strip()
                if remaining:
                    raise common.RunError("candidate_cleanup_not_proven")
                report["restore"]["candidate_absent"] = True
                if not prestate_valid:
                    raise common.RunError("restore_not_authorized_without_valid_prestate")
                state = common.qwen_state(operator.qwen_status())
                if state == "stopped":
                    if not stop_attempted and prestate_qwen_state != "stopped":
                        raise common.RunError("unexpected_qwen_prestate_drift")
                    common.wait_gpu_idle()
                    operator.qwen("start")
                if common.qwen_state(operator.qwen_status()) != "ready":
                    raise common.RunError("qwen_restore_not_proven")
                common.verify_production_qwen_identity()
                report["restore"]["qwen_ready"] = True
                completions = base.restore_admission(operator, close_attempted=close_attempted)
                if completions:
                    report["restore"]["gateway_completion_markers"] = ["QWEN_RESTORE_OK_1", "QWEN_RESTORE_OK_2"]
                report["restore"]["admission_public"] = True
                report["restore"]["gateway_healthy"] = base.service_healthy("ai-server-gateway")
                report["restore"]["open_webui_healthy"] = base.service_healthy("ai-server-open-webui")
                if not report["restore"]["gateway_healthy"] or not report["restore"]["open_webui_healthy"]:
                    raise common.RunError("user_service_health_not_proven")
                power = common.gpu_rows()[3]
                report["restore"]["power_limit_w"] = power
                if abs(power - 300.0) > 0.05:
                    raise common.RunError("power_limit_restore_not_proven")
                report["storage"]["available_after_bytes"] = _available_bytes()
                if report["storage"]["available_after_bytes"] < RESERVE_BYTES:
                    raise common.RunError("storage_reserve_not_met_after_run")
            except common.RunError as exc:
                cleanup_error = exc
                report["cleanup_error_code"] = exc.code
    finally:
        os.close(descriptor)
    report["kernel_fault_free"] = common.kernel_fault_free(since_epoch)
    report["ended_epoch"] = int(time.time())
    if cleanup_error is not None:
        report["outcome"] = "restore_unknown"
    elif error is not None or report["kernel_fault_free"] is not True:
        report["outcome"] = "failed_restored"
        if error is None:
            report["error_code"] = "kernel_fault_detected"
    elif report["mode"]["status"] == "PASS":
        report["outcome"] = "passed_restored"
    else:
        report["outcome"] = "failed_restored"
    common.write_report(args.output, report)
    print(json.dumps({"outcome": report["outcome"], "report": str(args.output)}, sort_keys=True))
    return 0 if report["outcome"] == "passed_restored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
