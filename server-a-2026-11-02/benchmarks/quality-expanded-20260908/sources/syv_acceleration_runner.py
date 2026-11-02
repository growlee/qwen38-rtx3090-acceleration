#!/usr/bin/env python3
"""One bounded official syv MTP run for the sealed Leminkozey target."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

import server_a_runner as common


IMAGE = "ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
MANIFEST_DIGEST = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
CONFIG_DIGEST = "sha256:b0516eab4a689391cd73629593d53111777691c5f3c453d5064e77fe55981101"
REPO_DIGEST = "ghcr.io/syv-ai/qwen38-27b-rtx3090@" + MANIFEST_DIGEST
IMAGE_LABELS = {
    "org.opencontainers.image.created": "2026-08-27T08:52:29.474Z",
    "org.opencontainers.image.licenses": "Apache-2.0",
    "org.opencontainers.image.revision": "69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c",
    "org.opencontainers.image.source": "https://github.com/syv-ai/qwen38-27b-rtx3090",
    "org.opencontainers.image.title": "qwen38-27b-rtx3090",
}

TARGET_REPOSITORY = "leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound"
TARGET_REVISION = "f958acb153eb8c7bbd25638149c8c75185559226"
TARGET_LEAF = Path("/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey") / TARGET_REVISION
TARGET_MODEL_DIR = TARGET_LEAF / "model"
TARGET_RECEIPT = TARGET_LEAF / "COMPLETE.receipt"
TARGET_RECEIPT_SHA256 = "a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a"
TARGET_LOGICAL_BYTES = 16_969_942_520
TARGET_ALLOCATED_BYTES = 16_970_018_816
TARGET_PHASE_BOUND_BYTES = 16_970_047_488
TARGET_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPECTED_MTP_WEIGHTS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}

CONTAINER = "ai-exp-syv-leminkozey-mtp"
MODEL_MOUNT = "/app/models/Qwen3.8-27B-W4A16-AutoRound"
MODEL_ID = "qwen3.8-27b"
PORT = 18020
MAX_LEN = 57_344
RESERVE_BYTES = 150 * 1024**3
LIVE_BUDGET_SECONDS = 30 * 60
STOP_RESERVE_SECONDS = 75
RECEIPT_ROW = re.compile(r"^([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9._-]+)$")
STARTUP_EVIDENCE = re.compile(r"(?i)(speculative|spec_decode|\bmtp\b|draft[_ -]tokens|flash.?attention)")
MTP_RUNTIME_EVIDENCE = re.compile(r"(?i)(speculative|spec_decode|\bmtp\b|draft[_ -]tokens)")
DIAGNOSTIC_ERROR_CODE = re.compile(
    r"^(?:health_failed|models_failed|models_invalid|model_missing|content_type_invalid|"
    r"response_too_large|sse_invalid|draft_timings_incomplete|draft_timings_invalid|"
    r"transport_error|stream_incomplete|usage_missing|empty_output|degenerate_output|http_[1-5][0-9]{2})$"
)
DIAGNOSTIC_CASE_IDS = {"warmup", "operations-guide-ru", "architecture-analysis-en", "python-service-code"}
MAX_DIAGNOSTIC_COUNTER = 16 * 1024 * 1024

BASELINE_PLAIN_TPS = 50.451
BASELINE_CURRENT_DFLASH2_TPS = 53.213
DRAFT_REPOSITORY = "syvai/Qwen3.8-27B-DFlash2-W4A16"
DRAFT_REVISION = "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279"
DRAFT_PAIRED_TARGET_REPOSITORY = "dbirks/Qwen3.8-27B-W4A16-AutoRound"
DRAFT_PAIRED_TARGET_REVISION = "1f05c441c4e64ae0549de44fa9ea5a6d43610314"
DRAFT_PAIRED_TARGET_MANIFEST_SHA256 = "1b19a6fb7a4cf7043f024dda30201af763a11db7b02b6cdda5c9d06c418a14cf"

BUNDLE = Path(__file__).resolve().parent
COMMON = BUNDLE / "server_a_runner.py"
BASE_CLIENT = BUNDLE / "benchmark_client.py"
ACCELERATION_CLIENT = BUNDLE / "syv_acceleration_benchmark_client.py"
DEPENDENCY_SHA256 = {
    COMMON.name: "91150887cd6bd054a7357e301dc1df96e98724315cf4307d105850427cb271c4",
    BASE_CLIENT.name: "98bcc2a476b10dfe2ab019ae7158000aab8aac1ea6f97b18a5bf76debdb6b134",
    ACCELERATION_CLIENT.name: "8a5f606d7c5d9894d8a4487850e2ce701b92f80e852e3bc3eb2b41839b397b0e",
}


def exact_ro_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise common.RunError("candidate_file_missing") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        raise common.RunError("candidate_file_identity_invalid")
    return info


def verify_dependencies() -> None:
    for name, expected in DEPENDENCY_SHA256.items():
        path = BUNDLE / name
        info = exact_ro_file(path)
        if info.st_size > 1024 * 1024 or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise common.RunError("runner_dependency_identity_invalid", {"file": name})


def verify_target() -> str:
    try:
        leaf_info = TARGET_LEAF.lstat()
        model_info = TARGET_MODEL_DIR.lstat()
    except OSError as exc:
        raise common.RunError("candidate_directory_missing") from exc
    for path, info in ((TARGET_LEAF, leaf_info), (TARGET_MODEL_DIR, model_info)):
        if (
            path.is_symlink()
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != 0o555
        ):
            raise common.RunError("candidate_directory_identity_invalid")
    if {path.name for path in TARGET_LEAF.iterdir()} != {"model", "COMPLETE.receipt"}:
        raise common.RunError("candidate_file_set_invalid")

    receipt_info = exact_ro_file(TARGET_RECEIPT)
    if receipt_info.st_size > 1024 * 1024:
        raise common.RunError("candidate_receipt_invalid")
    raw = TARGET_RECEIPT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TARGET_RECEIPT_SHA256:
        raise common.RunError("candidate_receipt_identity_invalid")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise common.RunError("candidate_receipt_invalid") from exc
    expected_header = [
        "complete=true",
        f"repo={TARGET_REPOSITORY}",
        f"revision={TARGET_REVISION}",
        f"logical_payload_bytes={TARGET_LOGICAL_BYTES}",
        f"allocated_model_bytes={TARGET_ALLOCATED_BYTES}",
        f"phase_physical_bound_bytes={TARGET_PHASE_BOUND_BYTES}",
    ]
    if len(lines) < 7 or lines[:6] != expected_header:
        raise common.RunError("candidate_receipt_invalid")

    rows: dict[str, tuple[int, str]] = {}
    ordered_names: list[str] = []
    for line in lines[6:]:
        match = RECEIPT_ROW.fullmatch(line)
        if match is None or match.group(3) in rows:
            raise common.RunError("candidate_receipt_invalid")
        digest, size, name = match.groups()
        rows[name] = (int(size), digest)
        ordered_names.append(name)
    if not rows or ordered_names != sorted(ordered_names):
        raise common.RunError("candidate_receipt_invalid")
    if sorted(path.name for path in TARGET_MODEL_DIR.iterdir()) != ordered_names:
        raise common.RunError("candidate_file_set_invalid")

    logical = 0
    allocated = model_info.st_blocks * 512
    for name in ordered_names:
        info = exact_ro_file(TARGET_MODEL_DIR / name)
        expected_size, _digest = rows[name]
        if info.st_size != expected_size:
            raise common.RunError("candidate_file_identity_invalid", {"file": name})
        logical += info.st_size
        allocated += info.st_blocks * 512
    if logical != TARGET_LOGICAL_BYTES or allocated != TARGET_ALLOCATED_BYTES:
        raise common.RunError("candidate_receipt_invalid")
    if rows.get("tokenizer.json", (None, None))[1] != TARGET_TOKENIZER_SHA256:
        raise common.RunError("candidate_tokenizer_identity_invalid")

    try:
        index = json.loads((TARGET_MODEL_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise common.RunError("candidate_index_invalid") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or {key for key in weight_map if key.startswith("mtp.")} != EXPECTED_MTP_WEIGHTS:
        raise common.RunError("candidate_mtp_weights_invalid")
    if "mtp.draft_lm_head.weight_packed" in weight_map or (TARGET_MODEL_DIR / "mtp_draft_vocab_ids.pt").exists():
        raise common.RunError("unexpected_fast_variant_overlay")
    return TARGET_RECEIPT_SHA256


def verify_image() -> None:
    raw = common.run(["docker", "image", "inspect", IMAGE, "--format", "{{json .}}"], timeout=30).stdout
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise common.RunError("syv_image_identity_invalid") from exc
    config = value.get("Config") if isinstance(value, dict) else None
    labels = config.get("Labels") if isinstance(config, dict) else None
    if (
        value.get("Id") not in {CONFIG_DIGEST, MANIFEST_DIGEST}
        or value.get("Architecture") != "amd64"
        or value.get("Os") != "linux"
        or value.get("RepoDigests") != [REPO_DIGEST]
        or not isinstance(labels, dict)
        or any(labels.get(key) != expected for key, expected in IMAGE_LABELS.items())
    ):
        raise common.RunError("syv_image_identity_invalid")
    present = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"], timeout=15
    ).stdout.decode("ascii").strip()
    if present:
        raise common.RunError("stale_candidate_container_present")


def available_bytes(path: Path = Path("/srv/ai-server")) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def bounded_timeout(deadline_monotonic: float, cap_seconds: float, code: str = "live_budget_exhausted") -> float:
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise common.RunError(code)
    return max(0.05, min(cap_seconds, remaining))


def container_args() -> list[str]:
    return [
        "docker", "run", "--pull", "never", "-d", "--name", CONTAINER, "--restart", "no",
        "--user", "0:0", "--gpus", "1", "--network", "none", "--ipc", "private", "--read-only",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--pids-limit", "2048",
        "--shm-size", "16g", "--memory", "30g", "--memory-swap", "32g",
        "--tmpfs", "/tmp:rw,nosuid,size=4g",
        "--tmpfs", "/cache:rw,exec,nosuid,size=4g",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "HF_DATASETS_OFFLINE=1", "--env", "DO_NOT_TRACK=1",
        "--env", "HOME=/cache", "--env", "XDG_CACHE_HOME=/cache",
        "--env", f"MODEL={MODEL_MOUNT}", "--env", "CTX=fast", "--env", "SPEC=mtp",
        "--env", f"MAX_LEN={MAX_LEN}",
        "--env", "PREPARE=0", "--env", "VERIFY=1", "--env", "VLLM_OFFLOAD_KEEP_SHM=1",
        "--env", f"PORT={PORT}",
        "--mount", f"type=bind,src={TARGET_MODEL_DIR},dst={MODEL_MOUNT},readonly",
        "--mount", f"type=bind,src={BUNDLE},dst=/runner,readonly",
        IMAGE, "single",
    ]


def start_candidate(deadline_monotonic: float) -> None:
    common.run(
        container_args(),
        timeout=bounded_timeout(deadline_monotonic, 60, "live_budget_exhausted_during_start"),
        diagnostic_output=True,
    )


def verify_container_contract(
    deadline_monotonic: float, *, expected_verify: str = "1"
) -> dict[str, Any]:
    if expected_verify not in {"0", "1"}:
        raise common.RunError("candidate_container_contract_invalid")
    raw = common.run(
        ["docker", "inspect", CONTAINER],
        timeout=bounded_timeout(deadline_monotonic, 30, "live_budget_exhausted_during_contract_check"),
    ).stdout
    try:
        values = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise common.RunError("candidate_container_contract_invalid") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise common.RunError("candidate_container_contract_invalid")
    value = values[0]
    host = value.get("HostConfig")
    config = value.get("Config")
    mounts = value.get("Mounts")
    if not isinstance(host, dict) or not isinstance(config, dict) or not isinstance(mounts, list):
        raise common.RunError("candidate_container_contract_invalid")
    cache_options = str((host.get("Tmpfs") or {}).get("/cache", ""))
    if (
        host.get("NetworkMode") != "none"
        or host.get("ReadonlyRootfs") is not True
        or host.get("IpcMode") != "private"
        or host.get("PidsLimit") != 2048
        or not any(str(item).startswith("no-new-privileges") for item in (host.get("SecurityOpt") or []))
        or "ALL" not in (host.get("CapDrop") or [])
        or "exec" not in cache_options.split(",")
        or "noexec" in cache_options.split(",")
        or host.get("PortBindings") not in (None, {})
        or config.get("Cmd") != ["single"]
    ):
        raise common.RunError("candidate_container_contract_invalid")
    env = set(config.get("Env") or [])
    for expected in (
        f"MODEL={MODEL_MOUNT}", "CTX=fast", "SPEC=mtp", f"MAX_LEN={MAX_LEN}",
        "PREPARE=0", f"VERIFY={expected_verify}"
    ):
        if expected not in env:
            raise common.RunError("candidate_container_contract_invalid")
    expected_mounts = {
        (str(TARGET_MODEL_DIR), MODEL_MOUNT, False),
        (str(BUNDLE), "/runner", False),
    }
    actual_mounts = {
        (mount.get("Source"), mount.get("Destination"), mount.get("RW"))
        for mount in mounts if isinstance(mount, dict) and mount.get("Type") == "bind"
    }
    if not expected_mounts.issubset(actual_mounts):
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
        timeout=bounded_timeout(deadline_monotonic, 15, "live_budget_exhausted_during_contract_check"),
    )
    return {
        "network_mode": "none",
        "read_only_rootfs": True,
        "cache_exec": True,
        "target_mount_read_only": True,
        "runner_mount_read_only": True,
        "spec": "mtp",
        "ctx": "fast",
        "max_model_len": MAX_LEN,
        "entrypoint_verify": expected_verify == "1",
    }


def wait_ready(deadline_monotonic: float) -> float:
    started = time.monotonic()
    probe = f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{PORT}/health',timeout=5).read(1)"
    while True:
        state_raw = common.run(
            ["docker", "inspect", CONTAINER, "--format", "{{json .State}}"],
            timeout=bounded_timeout(deadline_monotonic, 15, "live_budget_exhausted_during_readiness"),
        ).stdout
        try:
            state = json.loads(state_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise common.RunError("candidate_state_invalid") from exc
        if not state.get("Running"):
            raise common.RunError(
                "candidate_exit_before_ready",
                {"exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled")},
            )
        result = common.run(
            ["docker", "exec", CONTAINER, "/app/venv/bin/python", "-B", "-c", probe],
            timeout=bounded_timeout(deadline_monotonic, 10, "live_budget_exhausted_during_readiness"),
            check=False,
        )
        if result.returncode == 0:
            return round(time.monotonic() - started, 3)
        time.sleep(bounded_timeout(deadline_monotonic, 5, "live_budget_exhausted_during_readiness"))


def startup_evidence(deadline_monotonic: float) -> dict[str, Any]:
    logs = common.run(
        ["docker", "logs", "--tail", "600", CONTAINER],
        timeout=bounded_timeout(deadline_monotonic, 30, "live_budget_exhausted_during_startup_evidence"),
        check=False,
    )
    top = common.run(
        ["docker", "top", CONTAINER, "-eo", "pid,args"],
        timeout=bounded_timeout(deadline_monotonic, 30, "live_budget_exhausted_during_startup_evidence"),
        check=False,
    )
    matches: list[str] = []
    for source, raw in (("log", logs.stdout + logs.stderr), ("process", top.stdout + top.stderr)):
        for line in raw.decode("utf-8", "replace").splitlines():
            if STARTUP_EVIDENCE.search(line):
                matches.append(f"{source}:{common.sanitized_tail(line)[:1500]}")
    payload = "\n".join(matches[:30])
    mtp_matches = [line for line in matches if MTP_RUNTIME_EVIDENCE.search(line)]
    return {
        "matched_line_count": len(matches),
        "matched_lines_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "matched_lines": matches[:30],
        "mtp_runtime_proven": bool(mtp_matches),
    }


def validate_benchmark_diagnostic(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema", "status", "error_code", "case_id", "protocol",
        "request_text_recorded", "response_text_recorded", "secrets_recorded",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema") != "qwen38-syv-acceleration-diagnostic/v1"
        or value.get("status") != "FAIL"
        or value.get("request_text_recorded") is not False
        or value.get("response_text_recorded") is not False
        or value.get("secrets_recorded") is not False
        or not isinstance(value.get("error_code"), str)
        or DIAGNOSTIC_ERROR_CODE.fullmatch(value["error_code"]) is None
    ):
        raise common.RunError("benchmark_diagnostic_invalid")
    case_id = value.get("case_id")
    protocol = value.get("protocol")
    if protocol is None:
        if case_id is not None:
            raise common.RunError("benchmark_diagnostic_invalid")
        return value
    expected_protocol_keys = {
        "saw_first_delta", "saw_finish_reason", "finish_reason", "saw_done",
        "saw_usage", "usage_complete", "termination", "response_bytes", "delta_count",
    }
    if (
        case_id not in DIAGNOSTIC_CASE_IDS
        or not isinstance(protocol, dict)
        or set(protocol) != expected_protocol_keys
        or any(
            not isinstance(protocol.get(key), bool)
            for key in (
                "saw_first_delta", "saw_finish_reason", "saw_done", "saw_usage", "usage_complete"
            )
        )
        or protocol.get("finish_reason") not in {None, "stop", "length", "other"}
        or protocol.get("termination") not in {"pending", "clean_eof", "done_sentinel", "transport_error"}
        or not isinstance(protocol.get("response_bytes"), int)
        or isinstance(protocol.get("response_bytes"), bool)
        or not 0 <= protocol["response_bytes"] <= MAX_DIAGNOSTIC_COUNTER
        or not isinstance(protocol.get("delta_count"), int)
        or isinstance(protocol.get("delta_count"), bool)
        or not 0 <= protocol["delta_count"] <= MAX_DIAGNOSTIC_COUNTER
    ):
        raise common.RunError("benchmark_diagnostic_invalid")
    return value


def run_benchmark(deadline_monotonic: float) -> dict[str, Any]:
    remaining = deadline_monotonic - time.monotonic()
    if remaining < 90:
        raise common.RunError("live_budget_insufficient_for_benchmark")
    client_timeout = min(600.0, max(60.0, remaining - 30.0))
    result = common.run(
        [
            "docker", "exec", "-i", CONTAINER, "/app/venv/bin/python", "-B",
            "/runner/syv_acceleration_benchmark_client.py",
            "--base-url", f"http://127.0.0.1:{PORT}",
            "--model", MODEL_ID,
            "--mode-id", "leminkozey-mtp-fast-maxlen57344",
            "--timeout", str(client_timeout),
        ],
        timeout=max(60.0, remaining),
        check=False,
    )
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        details = common.command_diagnostic(result)
        details.pop("stdout_tail", None)
        details.pop("stderr_tail", None)
        raise common.RunError("benchmark_diagnostic_invalid", details) from exc
    if result.returncode != 0:
        diagnostic = validate_benchmark_diagnostic(value)
        raise common.RunError(
            "benchmark_client_failed",
            {"client_returncode": result.returncode, "benchmark_diagnostic": diagnostic},
        )
    if (
        not isinstance(value, dict)
        or value.get("schema") != "qwen38-syv-acceleration-benchmark/v1"
        or value.get("warmup_count") != 1
        or value.get("measured_request_count") != 3
        or value.get("max_output_tokens_per_request") != 1024
        or value.get("request_text_recorded") is not False
        or value.get("response_text_recorded") is not False
        or value.get("secrets_recorded") is not False
    ):
        raise common.RunError("benchmark_result_invalid")
    return value


def enrich_result(value: dict[str, Any]) -> None:
    metrics = value.get("metrics")
    decode = metrics.get("decode_tokens_per_second_median") if isinstance(metrics, dict) else None
    if isinstance(decode, (int, float)) and decode > 0:
        value["comparison"] = {
            "plain_baseline_tps": BASELINE_PLAIN_TPS,
            "change_percent_vs_plain": round((decode / BASELINE_PLAIN_TPS - 1) * 100, 3),
            "current_dflash2_baseline_tps": BASELINE_CURRENT_DFLASH2_TPS,
            "change_percent_vs_current_dflash2": round((decode / BASELINE_CURRENT_DFLASH2_TPS - 1) * 100, 3),
            "baseline_provenance": "goal-confirmed prior measurements",
        }


def diagnostics(*, include_logs: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    state = common.run(
        ["docker", "inspect", CONTAINER, "--format", "{{json .State}}"], timeout=15, check=False
    )
    if state.returncode == 0:
        try:
            value = json.loads(state.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            result["state_error"] = "candidate_state_invalid"
        else:
            result["state"] = {
                "error": common.sanitized_tail(str(value.get("Error") or "")),
                "exit_code": value.get("ExitCode"),
                "oom_killed": value.get("OOMKilled"),
            }
    else:
        result["state_error"] = "state_query_failed"
    if include_logs:
        logs = common.run(["docker", "logs", "--tail", "200", CONTAINER], timeout=30, check=False)
        result["logs"] = common.command_diagnostic(logs)
    else:
        result["logs_omitted"] = "request_execution_started"
    return result


def stop_candidate(hard_deadline_monotonic: float | None = None) -> None:
    def cleanup_timeout(cap: float) -> float:
        if hard_deadline_monotonic is None:
            return cap
        remaining_seconds = hard_deadline_monotonic - time.monotonic()
        if remaining_seconds <= 0:
            raise common.RunError("candidate_cleanup_deadline_exhausted")
        return min(cap, remaining_seconds)

    remaining = None if hard_deadline_monotonic is None else hard_deadline_monotonic - time.monotonic()
    if remaining is not None and remaining <= STOP_RESERVE_SECONDS:
        common.run(
            ["docker", "rm", "-f", CONTAINER],
            timeout=cleanup_timeout(30),
            check=False,
        )
    else:
        present = common.run(
            ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"],
            timeout=cleanup_timeout(15),
        ).stdout.decode("ascii").strip()
        if present:
            running = common.run(
                ["docker", "inspect", CONTAINER, "--format", "{{.State.Running}}"],
                timeout=cleanup_timeout(15),
            ).stdout.decode("ascii").strip()
            if running == "true":
                remaining = None if hard_deadline_monotonic is None else hard_deadline_monotonic - time.monotonic()
                stop_seconds = 60 if remaining is None else max(1, min(60, int(max(1, remaining - 30))))
                stopped = common.run(
                    ["docker", "stop", "--time", str(stop_seconds), CONTAINER],
                    timeout=cleanup_timeout(stop_seconds + 15),
                    check=False,
                )
                if stopped.returncode != 0:
                    common.run(
                        ["docker", "rm", "-f", CONTAINER],
                        timeout=cleanup_timeout(30),
                        check=False,
                    )
                else:
                    common.run(
                        ["docker", "rm", CONTAINER],
                        timeout=cleanup_timeout(30),
                        check=False,
                    )
            else:
                common.run(
                    ["docker", "rm", CONTAINER],
                    timeout=cleanup_timeout(30),
                    check=False,
                )
    final = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"],
        timeout=cleanup_timeout(15),
        check=False,
    )
    if final.returncode != 0 or final.stdout.decode("ascii").strip():
        raise common.RunError("candidate_cleanup_not_proven")


def service_healthy(name: str) -> bool:
    raw = common.run(["docker", "inspect", name, "--format", "{{json .State}}"], timeout=30).stdout
    try:
        state = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise common.RunError("service_state_invalid", {"service": name}) from exc
    return state.get("Running") is True and (state.get("Health") or {}).get("Status") == "healthy"


def verify_prestate(operator: common.Operator) -> dict[str, Any]:
    if not common.admission_safe(operator.status(), "public"):
        raise common.RunError("admission_preflight_not_public_empty")
    if common.qwen_state(operator.qwen_status()) != "ready":
        raise common.RunError("qwen_preflight_not_ready")
    common.verify_production_qwen_identity()
    gateway_healthy = service_healthy("ai-server-gateway")
    open_webui_healthy = service_healthy("ai-server-open-webui")
    if not gateway_healthy or not open_webui_healthy:
        raise common.RunError("user_service_preflight_not_healthy")
    power_limit = common.gpu_rows()[3]
    if abs(power_limit - 300.0) > 0.05:
        raise common.RunError("power_limit_not_300w")
    return {
        "admission_public": True,
        "qwen_ready": True,
        "gateway_healthy": gateway_healthy,
        "open_webui_healthy": open_webui_healthy,
        "power_limit_w": power_limit,
    }


def restore_admission(operator: common.Operator, *, close_attempted: bool) -> int:
    status = operator.status()
    phase = status.get("phase")
    if not close_attempted:
        if not common.admission_safe(status, "public"):
            raise common.RunError("admission_prestate_drift")
        return 0
    if phase == "closed":
        operator.advance("smoke")
        status = operator.status()
        phase = status.get("phase")
    if phase == "smoke":
        if not common.admission_safe(status, "smoke"):
            raise common.RunError("admission_smoke_not_proven")
    elif phase != "public":
        raise common.RunError("admission_restore_phase_unknown", {"phase": phase})
    common.restore_completion(1)
    common.restore_completion(2)
    if phase != "public":
        operator.advance("public")
    if not common.admission_safe(operator.status(), "public"):
        raise common.RunError("admission_public_not_proven")
    return 2


def dflash2_blocker() -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "attempted": False,
        "reason": "draft_target_pairing_receipt_missing",
        "draft_repository": DRAFT_REPOSITORY,
        "draft_revision": DRAFT_REVISION,
        "draft_paired_target_repository": DRAFT_PAIRED_TARGET_REPOSITORY,
        "draft_paired_target_revision": DRAFT_PAIRED_TARGET_REVISION,
        "draft_paired_target_manifest_sha256": DRAFT_PAIRED_TARGET_MANIFEST_SHA256,
        "tested_target_repository": TARGET_REPOSITORY,
        "tested_target_revision": TARGET_REVISION,
        "tested_target_receipt_sha256": TARGET_RECEIPT_SHA256,
        "missing_invariant": (
            "provider target-draft compatibility receipt binding tokenizer, shared embedding/lm_head, "
            "and target computations for hidden-state taps 5,19,33,47,61"
        ),
    }


def sampler_evidence(sampler: common.Sampler | None) -> dict[str, Any]:
    if sampler is None:
        return {}
    try:
        return {"resources": sampler.report()}
    except common.RunError as exc:
        return {"resource_sampling_error": exc.code}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("root_required")
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise SystemExit("output_path_invalid")

    report: dict[str, Any] = {
        "schema": "qwen38-syv-leminkozey-acceleration-result/v1",
        "started_epoch": int(time.time()),
        "content_recorded": False,
        "secrets_recorded": False,
        "coordinates": {
            "image": IMAGE,
            "image_manifest_digest": MANIFEST_DIGEST,
            "source_revision": IMAGE_LABELS["org.opencontainers.image.revision"],
            "target_repository": TARGET_REPOSITORY,
            "target_revision": TARGET_REVISION,
            "target_receipt_sha256": TARGET_RECEIPT_SHA256,
            "max_model_len": MAX_LEN,
        },
        "storage": {"required_reserve_bytes": RESERVE_BYTES},
        "live_budget_seconds": LIVE_BUDGET_SECONDS,
        "modes": {
            "mtp_fast": {"status": "PENDING", "attempted": False},
            "dflash2_fast": dflash2_blocker(),
        },
        "restore": {
            "candidate_absent": False,
            "qwen_ready": False,
            "gateway_completions": 0,
            "admission_public": False,
            "gateway_healthy": False,
            "open_webui_healthy": False,
            "power_limit_w": None,
        },
        "prestate": None,
        "outcome": "running",
        "error_code": None,
        "cleanup_error_code": None,
    }
    operator = common.Operator()
    admission_close_attempted = False
    qwen_stop_attempted = False
    candidate_mutation_attempted = False
    request_started = False
    error: common.RunError | None = None
    cleanup_error: common.RunError | None = None
    live_started_monotonic: float | None = None
    live_hard_deadline: float | None = None
    sampler: common.Sampler | None = None
    runtime_evidence: dict[str, Any] = {}
    since_epoch = int(time.time())

    descriptor = os.open(common.LOCK, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise common.RunError("host_lock_busy") from exc
        try:
            report["prestate"] = verify_prestate(operator)
            verify_dependencies()
            receipt_sha = verify_target()
            verify_image()
            free_before = available_bytes()
            report["storage"]["available_before_bytes"] = free_before
            if free_before < RESERVE_BYTES:
                raise common.RunError("storage_reserve_not_met", {"available_bytes": free_before})
            admission_close_attempted = True
            operator.advance("closed")
            if not common.admission_safe(operator.status(), "closed"):
                raise common.RunError("admission_close_not_proven")
            qwen_stop_attempted = True
            operator.qwen("stop")
            if common.qwen_state(operator.qwen_status()) != "stopped":
                raise common.RunError("qwen_stop_not_proven")
            common.wait_gpu_idle()

            with common.Sampler() as active_sampler:
                sampler = active_sampler
                candidate_mutation_attempted = True
                report["modes"]["mtp_fast"]["attempted"] = True
                live_started_monotonic = time.monotonic()
                live_hard_deadline = live_started_monotonic + LIVE_BUDGET_SECONDS
                live_work_deadline = live_hard_deadline - STOP_RESERVE_SECONDS
                start_candidate(live_work_deadline)
                contract = verify_container_contract(live_work_deadline)
                runtime_evidence["runtime_contract"] = contract
                load_seconds = wait_ready(live_work_deadline)
                runtime_evidence["load_seconds"] = load_seconds
                proof = startup_evidence(live_work_deadline)
                runtime_evidence["startup_evidence"] = proof
                if proof["mtp_runtime_proven"] is not True:
                    raise common.RunError("mtp_runtime_not_proven")
                request_started = True
                benchmark = run_benchmark(live_work_deadline)
            benchmark.update(runtime_evidence)
            benchmark.update(sampler_evidence(sampler))
            benchmark["target_receipt_sha256"] = receipt_sha
            enrich_result(benchmark)
            report["modes"]["mtp_fast"] = {"status": "PASS", "attempted": True, "result": benchmark}
        except common.RunError as exc:
            error = exc
            report["error_code"] = exc.code
            if report["modes"]["mtp_fast"]["status"] == "PENDING":
                mode_failure = {
                    "status": "FAIL",
                    "attempted": report["modes"]["mtp_fast"].get("attempted", False),
                    "reason": exc.code,
                    **exc.details,
                    "diagnostics": diagnostics(include_logs=not request_started) if candidate_mutation_attempted else None,
                }
                mode_failure.update(runtime_evidence)
                mode_failure.update(sampler_evidence(sampler))
                report["modes"]["mtp_fast"] = mode_failure
        except Exception as exc:
            error = common.RunError("unexpected_runner_failure", {"exception_type": type(exc).__name__})
            report["error_code"] = error.code
            if report["modes"]["mtp_fast"]["status"] == "PENDING":
                mode_failure = {
                    "status": "FAIL",
                    "attempted": report["modes"]["mtp_fast"].get("attempted", False),
                    "reason": error.code,
                    **error.details,
                    "diagnostics": diagnostics(include_logs=not request_started) if candidate_mutation_attempted else None,
                }
                mode_failure.update(runtime_evidence)
                mode_failure.update(sampler_evidence(sampler))
                report["modes"]["mtp_fast"] = mode_failure
        finally:
            try:
                if candidate_mutation_attempted:
                    stop_candidate(live_hard_deadline)
                    if live_started_monotonic is not None:
                        report["live_candidate_elapsed_seconds"] = round(time.monotonic() - live_started_monotonic, 3)
                remaining = common.run(
                    ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"], timeout=15
                ).stdout.decode("ascii").strip()
                if remaining:
                    raise common.RunError("candidate_cleanup_not_proven")
                report["restore"]["candidate_absent"] = True
                state = common.qwen_state(operator.qwen_status())
                if state == "stopped":
                    if not qwen_stop_attempted:
                        raise common.RunError("unexpected_qwen_prestate_drift")
                    common.wait_gpu_idle()
                    operator.qwen("start")
                if common.qwen_state(operator.qwen_status()) != "ready":
                    raise common.RunError("qwen_restore_not_proven")
                common.verify_production_qwen_identity()
                report["restore"]["qwen_ready"] = True
                report["restore"]["gateway_completions"] = restore_admission(
                    operator, close_attempted=admission_close_attempted
                )
                report["restore"]["admission_public"] = True
                report["restore"]["gateway_healthy"] = service_healthy("ai-server-gateway")
                report["restore"]["open_webui_healthy"] = service_healthy("ai-server-open-webui")
                if not report["restore"]["gateway_healthy"] or not report["restore"]["open_webui_healthy"]:
                    raise common.RunError("user_service_health_not_proven")
                power_limit = common.gpu_rows()[3]
                report["restore"]["power_limit_w"] = power_limit
                if abs(power_limit - 300.0) > 0.05:
                    raise common.RunError("power_limit_restore_not_proven")
                free_after = available_bytes()
                report["storage"]["available_after_bytes"] = free_after
                if free_after < RESERVE_BYTES:
                    raise common.RunError("storage_reserve_not_met_after_run", {"available_bytes": free_after})
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
    elif report["modes"]["mtp_fast"].get("status") == "PASS" and report["modes"]["dflash2_fast"].get("status") == "BLOCKED":
        report["outcome"] = "passed_with_dflash2_blocked_restored"
    else:
        report["outcome"] = "failed_restored"
    common.write_report(args.output, report)
    print(json.dumps({"outcome": report["outcome"], "report": str(args.output)}, sort_keys=True))
    return 0 if report["outcome"] == "passed_with_dflash2_blocked_restored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
