#!/usr/bin/env python3
"""One-shot, manifest-bound cleanup for the 2026-09-24 FreeToken residue."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "ai-server-freetoken-cleanup/2026-09-24/v1"
LOCKS = (
    "/srv/ai-server/state/acceleration-experiments/locks/host.lock",
    "/run/ai-server-experiment.lock",
    "/run/ai-server-backup-export.lock",
)
WHEEL_ROOT = Path("/srv/ai-server/artifacts/experiments/freetoken013-accel-20260918")
WHEELHOUSE = WHEEL_ROOT / "wheelhouse"
WHEEL_MANIFEST = WHEEL_ROOT / "artifact-manifest.json"
WHEEL_MANIFEST_SHA256 = "115458c260f6d0d700949b1cfe6a5a7ffbd5f4a890e3cc4990b64106985dfd59"
COMPOSE = Path("/opt/homelab/ai-server/deploy/compose.qwen.yaml")
COMPOSE_SHA256 = "88c7de590dd7dea693dba225242d81e52387415a1e4e0f5796b568dfc4468c3c"

ACCEL_TAG = "local/ai-server-freetoken:0.1.3-cac247a-accel-20260918"
ACCEL_IMAGE = "sha256:e64e250ab64f1c06f2246f7ac16027d59d998155d7d61a6301865e4e263bed03"
BASELINE_IMAGE = "sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134"
SYV_IMAGE = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
AGENT3_MANIFEST = "537439b6f0e8efaaba764303a386b83b6e12cf011dc05c2f3a58d1ce2df4b172"
SCORE_RECEIPTS = {
    "qwen-quality-20260908-code-dflash2-fast-n7-prefix32k-coding120-20260919-": (
        Path("/srv/ai-server/artifacts/experiments/dflash-standard120-20260919/run-20260919T005212Z/scored.jsonl"),
        "81b3126f24c1533af030e1ca6764119dd9864c616bb45ce77472a1dd83ae8230",
    ),
    "qwen-quality-20260908-code-freetoken013-nvfp4-coding120-": (
        Path("/srv/ai-server/artifacts/experiments/freetoken013-benchmark120-20260918/run-20260918T095259758850Z/scores.jsonl"),
        "ec9371b296cbf2d0947bc52da45e80607add00c6aad2bf120f3e85835ac37b56",
    ),
}

CPU_PREFIX_COUNTS = {
    "qwen-quality-20260908-code-dflash2-fast-n7-prefix32k-coding120-20260919-": 120,
    "qwen-quality-20260908-code-freetoken013-nvfp4-coding120-": 120,
    "ai-exp-dflash-agent3-20260919-cpu-": 14,
}

FT_NAMES = (
    "ai-server-freetoken-q38-013-speed2-fi",
    "ai-server-freetoken-q38-013-speed2-native",
    "ai-server-freetoken-q38-013-speed2-baseline",
    "ai-server-freetoken-q38-013-speed-baseline",
    "ai-server-freetoken013-tokenizer-probe",
    "ai-server-freetoken-q38-013-nvfp4-tokenizer-count",
    "ai-server-freetoken013-loopback-probe-venv",
    "ai-server-freetoken013-loopback-probe",
    "ai-server-freetoken-q38-013-nvfp4-internal-net",
    "ai-server-freetoken013-dynamo-cache-fix-probe",
    "ai-server-freetoken013-dynamo-probe",
    "ai-server-freetoken-q38-013-nvfp4-dynamo-uid",
    "ai-server-freetoken013-cache-probe-exec",
    "ai-server-freetoken013-cache-probe-original",
    "ai-server-freetoken-q38-013-nvfp4-cache-noexec",
    "ai-server-freetoken013-version-check",
)
CREATED_FT = "ai-server-freetoken013-loopback-probe"
MAIN_FT = "ai-server-freetoken-q38-013-nvfp4"

EXPECTED_WHEELS = {
    "cuda_tile-1.6.0-cp312-cp312-manylinux2014_x86_64.whl": (382170, "7e38f97bfedff359c926710387a40fbfc8f36efb7fe9ad2cb759d5a098eedde3"),
    "flashinfer_python-0.6.18.post1-py3-none-any.whl": (18348022, "adb6d2952b47131138e108bd2999b1fe18b71f1a4c0b2684f168753db9563991"),
    "nccl4py-0.5.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl": (3572033, "51156cbc35eb9122b870aa7ba1244d0c4781266442036c2bb3e1abb7e88d9376"),
    "ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": (183365, "65a24341b5ac09fcadcc37082660be40a94174e51a937fabf6e2cae26225fa2c"),
    "nvidia_cudnn_frontend-1.29.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": (7292886, "596c9b7591457b4b49227fa2e22bd5960957a198e4336ce9ada8b5d2a1e4a5a1"),
    "nvidia_cutlass_dsl_libs_cu13-4.7.1-cp312-cp312-manylinux_2_28_x86_64.whl": (88260436, "20c14e16125cb8e36bc9947a9f0a57a660f44651365337b911c84abde4967f5f"),
    "nvidia_ml_py-13.610.43-py3-none-any.whl": (53163, "f13c72698edef492f985cc225f14faafe68ae065a2e407f45bdf6f4b9b43fde8"),
    "sglang_kernel-0.4.5-cp310-abi3-manylinux2014_x86_64.whl": (382990222, "5927a14dfff92ced8b1f7c7d5844144bc574ec58e604d4fd53e6f53c709003a8"),
    "tabulate-0.10.0-py3-none-any.whl": (39814, "f0b0622e567335c8fabaaa659f1b33bcb6ddfe2e496071b743aa113f8774f2d3"),
}


class CleanupError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    value = hashlib.file_digest(os.fdopen(os.dup(fd), "rb", closefd=True), "sha256").hexdigest()
    os.lseek(fd, 0, os.SEEK_SET)
    return value


def run(command: list[str], timeout: int = 25) -> bytes:
    try:
        return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        raise CleanupError("command_failed:" + command[1] + ":" + stderr.decode("utf-8", "replace")[-300:]) from exc


def docker_json(command: list[str]) -> Any:
    try:
        return json.loads(run(["docker", *command]))
    except json.JSONDecodeError as exc:
        raise CleanupError("docker_json_invalid:" + command[0]) from exc


def read_safe_file(path: Path, expected_sha: str | None = None) -> tuple[os.stat_result, str, bytes]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode) or value.st_uid != 0 or value.st_nlink != 1 or value.st_mode & 0o022:
            raise CleanupError("unsafe_file:" + str(path))
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        actual_sha = digest_bytes(raw)
        if expected_sha is not None and actual_sha != expected_sha:
            raise CleanupError("file_sha256_drift:" + str(path))
        return value, actual_sha, raw
    finally:
        os.close(fd)


def safe_file(path: Path, expected_sha: str | None = None) -> tuple[os.stat_result, str]:
    value, actual_sha, _ = read_safe_file(path, expected_sha)
    return value, actual_sha


def safe_directory(path: Path) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink() or value.st_uid != 0 or value.st_mode & 0o022:
        raise CleanupError("unsafe_directory:" + str(path))
    return value


def stat_identity(path: str) -> dict[str, Any]:
    candidate = Path(path)
    link_value = candidate.lstat()
    result = {
        "path": path,
        "device": link_value.st_dev,
        "inode": link_value.st_ino,
        "mode": stat.S_IFMT(link_value.st_mode),
        "uid": link_value.st_uid,
        "gid": link_value.st_gid,
    }
    if stat.S_ISLNK(link_value.st_mode):
        resolved = candidate.stat()
        result["symlink_target"] = os.readlink(candidate)
        result["resolved"] = {
            "device": resolved.st_dev,
            "inode": resolved.st_ino,
            "mode": stat.S_IFMT(resolved.st_mode),
            "uid": resolved.st_uid,
            "gid": resolved.st_gid,
        }
    return result


def acquire_locks() -> list[int]:
    descriptors: list[int] = []
    try:
        for path in LOCKS:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
            value = os.fstat(fd)
            if not stat.S_ISREG(value.st_mode) or value.st_uid != 0 or value.st_nlink != 1 or value.st_mode & 0o022:
                os.close(fd)
                raise CleanupError("unsafe_lock:" + path)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors.append(fd)
        return descriptors
    except Exception:
        for fd in reversed(descriptors):
            os.close(fd)
        raise


def release_locks(descriptors: list[int]) -> None:
    for fd in reversed(descriptors):
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def container_state(value: dict[str, Any]) -> dict[str, Any]:
    state = value.get("State") or {}
    return {
        "status": state.get("Status"),
        "running": state.get("Running"),
        "paused": state.get("Paused"),
        "restarting": state.get("Restarting"),
        "dead": state.get("Dead"),
        "exit_code": state.get("ExitCode"),
    }


def config_sha(value: dict[str, Any]) -> str:
    # Config includes Env but only this digest is emitted.
    stable = {key: value.get(key) for key in ("Id", "Image", "Name", "Config", "HostConfig", "Mounts")}
    return digest_bytes(canonical(stable))


def sanitized_container(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(value.get("Name", "")).removeprefix("/"),
        "id": value.get("Id"),
        "image": value.get("Image"),
        "state": container_state(value),
        "config_sha256": config_sha(value),
    }


def all_containers() -> list[dict[str, Any]]:
    ids = run(["docker", "container", "ls", "-aq", "--no-trunc"]).decode().split()
    if not ids:
        raise CleanupError("container_inventory_empty")
    values = docker_json(["container", "inspect", *ids])
    if not isinstance(values, list) or len(values) != len(ids):
        raise CleanupError("container_inventory_incomplete")
    names = [str(item.get("Name", "")).removeprefix("/") for item in values]
    if len(names) != len(set(names)):
        raise CleanupError("duplicate_container_name")
    return values


def all_image_ids() -> set[str]:
    values = run(["docker", "image", "ls", "-aq", "--no-trunc"]).decode().split()
    if any(not re.fullmatch(r"sha256:[a-f0-9]{64}", value) for value in values):
        raise CleanupError("image_inventory_invalid")
    return set(values)


def paths_overlap(first: str, second: str) -> bool:
    first = os.path.normpath(first)
    second = os.path.normpath(second)
    try:
        common = os.path.commonpath((first, second))
    except ValueError:
        return False
    return common in (first, second)


def restart_is_no(value: dict[str, Any]) -> bool:
    return ((value.get("HostConfig") or {}).get("RestartPolicy") or {}).get("Name") == "no"


def validate_target(value: dict[str, Any], kind: str) -> None:
    name = str(value.get("Name", "")).removeprefix("/")
    state = container_state(value)
    if state["running"] is not False or state["paused"] is not False or state["restarting"] is not False or state["dead"] is not False:
        raise CleanupError("target_not_stopped:" + name)
    if not restart_is_no(value):
        raise CleanupError("target_restart_policy:" + name)
    if kind == "cpu":
        if value.get("Image") != SYV_IMAGE or value.get("Mounts") != [] or state["status"] not in ("exited", "created"):
            raise CleanupError("cpu_target_identity:" + name)
        if name.startswith("ai-exp-dflash-agent3-20260919-cpu-"):
            labels = (value.get("Config") or {}).get("Labels") or {}
            if labels.get("ai.server.experiment.role") != "agent3-cpu" or labels.get("ai.server.experiment.manifest") != AGENT3_MANIFEST:
                raise CleanupError("agent3_target_labels:" + name)
    else:
        expected_status = "created" if name == CREATED_FT else "exited"
        if state["status"] != expected_status or value.get("Image") not in (BASELINE_IMAGE, ACCEL_IMAGE):
            raise CleanupError("freetoken_target_identity:" + name)


def image_identity(reference: str, expected_id: str, exact_tags: set[str] | None = None) -> dict[str, Any]:
    values = docker_json(["image", "inspect", reference])
    if not isinstance(values, list) or len(values) != 1 or values[0].get("Id") != expected_id:
        raise CleanupError("image_identity:" + reference)
    if exact_tags is not None and set(values[0].get("RepoTags") or []) != exact_tags:
        raise CleanupError("image_tags:" + reference)
    return {"reference": reference, "id": expected_id}


def ownership_receipt(prefix: str, targets: list[str], by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path, expected_sha = SCORE_RECEIPTS[prefix]
    value, actual_sha, raw = read_safe_file(path, expected_sha)
    rows = []
    try:
        for line in raw.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise CleanupError("score_receipt_json:" + str(path)) from exc
    if len(rows) != 120:
        raise CleanupError("score_receipt_count:" + str(path))
    recorded = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("container_name"), str) or not isinstance(row.get("container_id"), str):
            raise CleanupError("score_receipt_row:" + str(path))
        name, container_id = row["container_name"], row["container_id"]
        if name in recorded or not name.startswith(prefix) or not re.fullmatch(r"[a-f0-9]{64}", container_id):
            raise CleanupError("score_receipt_identity:" + str(path))
        recorded[name] = container_id
    expected = {name: by_name[name]["Id"] for name in targets}
    if recorded != expected:
        raise CleanupError("score_receipt_target_mismatch:" + str(path))
    return {"path": str(path), "sha256": actual_sha, "device": value.st_dev, "inode": value.st_ino, "rows": len(rows)}


def wheel_binding() -> dict[str, Any]:
    safe_directory(WHEEL_ROOT)
    root_stat = safe_directory(WHEELHOUSE)
    manifest_stat, manifest_sha, raw = read_safe_file(WHEEL_MANIFEST, WHEEL_MANIFEST_SHA256)
    try:
        rows = json.loads(raw).get("artifacts")
    except (AttributeError, json.JSONDecodeError) as exc:
        raise CleanupError("wheel_manifest_invalid") from exc
    observed = {}
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_WHEELS):
        raise CleanupError("wheel_manifest_count")
    for row in rows:
        if not isinstance(row, dict) or set(("filename", "size", "sha256")) - set(row):
            raise CleanupError("wheel_manifest_row")
        name = row["filename"]
        if name not in EXPECTED_WHEELS or (row["size"], row["sha256"]) != EXPECTED_WHEELS[name]:
            raise CleanupError("wheel_manifest_identity:" + str(name))
        observed[name] = (row["size"], row["sha256"])
    if set(observed) != set(EXPECTED_WHEELS):
        raise CleanupError("wheel_manifest_names")
    entries = []
    actual_names = {entry.name for entry in os.scandir(WHEELHOUSE)}
    if actual_names != set(EXPECTED_WHEELS):
        raise CleanupError("wheelhouse_entry_set")
    for name in sorted(EXPECTED_WHEELS):
        expected_size, expected_sha = EXPECTED_WHEELS[name]
        value, actual_sha = safe_file(WHEELHOUSE / name, expected_sha)
        if value.st_size != expected_size:
            raise CleanupError("wheel_size:" + name)
        entries.append({"filename": name, "size": expected_size, "sha256": actual_sha, "device": value.st_dev, "inode": value.st_ino})
    return {
        "path": str(WHEELHOUSE),
        "device": root_stat.st_dev,
        "inode": root_stat.st_ino,
        "manifest": {"path": str(WHEEL_MANIFEST), "sha256": manifest_sha, "device": manifest_stat.st_dev, "inode": manifest_stat.st_ino},
        "files": entries,
    }


def collect_binding() -> tuple[dict[str, Any], dict[str, Any]]:
    safe_file(COMPOSE, COMPOSE_SHA256)
    values = all_containers()
    for value in values:
        for mount in value.get("Mounts") or []:
            source = mount.get("Source")
            if isinstance(source, str) and source.startswith("/") and paths_overlap(source, str(WHEELHOUSE)):
                name = str(value.get("Name", "")).removeprefix("/")
                raise CleanupError("wheelhouse_mount_overlap:" + name)
    by_name = {str(item["Name"]).removeprefix("/"): item for item in values}
    if MAIN_FT not in by_name:
        raise CleanupError("retained_main_freetoken_missing")
    if any(name.startswith("ai-server-freetoken") and name != MAIN_FT and name not in FT_NAMES for name in by_name):
        raise CleanupError("unexpected_freetoken_name")
    if set(FT_NAMES) - set(by_name):
        raise CleanupError("freetoken_target_missing")

    cpu_names: list[str] = []
    cpu_groups: dict[str, list[str]] = {}
    for prefix, count in CPU_PREFIX_COUNTS.items():
        selected = sorted(name for name in by_name if name.startswith(prefix))
        if len(selected) != count:
            raise CleanupError("cpu_prefix_count:" + prefix)
        cpu_groups[prefix] = selected
        cpu_names.extend(selected)
    if len(cpu_names) != len(set(cpu_names)):
        raise CleanupError("cpu_prefix_overlap")

    target_names = set(cpu_names) | set(FT_NAMES)
    targets = []
    for name in sorted(target_names):
        kind = "freetoken" if name in FT_NAMES else "cpu"
        validate_target(by_name[name], kind)
        row = sanitized_container(by_name[name])
        row["kind"] = kind
        targets.append(row)

    # Every reference to the removable image must itself be in the reviewed set.
    accel_users = {name for name, value in by_name.items() if value.get("Image") == ACCEL_IMAGE}
    if not accel_users or not accel_users <= target_names:
        raise CleanupError("accel_image_foreign_reference")

    non_targets = [sanitized_container(value) for name, value in by_name.items() if name not in target_names]
    non_targets.sort(key=lambda row: row["name"])
    mount_stats = []
    seen_sources = set()
    for name, value in sorted(by_name.items()):
        if name in target_names:
            continue
        for mount in value.get("Mounts") or []:
            if mount.get("Type") == "bind" and isinstance(mount.get("Source"), str) and mount["Source"] not in seen_sources:
                seen_sources.add(mount["Source"])
                mount_stats.append(stat_identity(mount["Source"]))
    mount_stats.sort(key=lambda row: row["path"])

    images = {
        "accel": image_identity(ACCEL_TAG, ACCEL_IMAGE, {ACCEL_TAG}),
        "baseline": image_identity(BASELINE_IMAGE, BASELINE_IMAGE),
        "syv": image_identity(SYV_IMAGE, SYV_IMAGE),
    }
    ownership = [ownership_receipt(prefix, cpu_groups[prefix], by_name) for prefix in sorted(SCORE_RECEIPTS)]
    wheelhouse = wheel_binding()
    binding = {
        "schema": SCHEMA,
        "compose": {"path": str(COMPOSE), "sha256": COMPOSE_SHA256},
        "images": images,
        "cpu_ownership_receipts": ownership,
        "targets": targets,
        "non_targets": non_targets,
        "retained_mount_stats": mount_stats,
        "wheelhouse": wheelhouse,
    }
    usage = os.statvfs("/srv/ai-server")
    observed = {
        "available_bytes": usage.f_bavail * usage.f_frsize,
        "container_count": len(values),
        "target_count": len(targets),
        "cpu_target_count": len(cpu_names),
        "freetoken_target_count": len(FT_NAMES),
        "wheel_bytes": sum(item[0] for item in EXPECTED_WHEELS.values()),
        "mode_checks": {
            "locks_held": list(LOCKS),
            "all_targets_stopped": True,
            "target_restart_policy_none": True,
            "cpu_targets_mountless": True,
            "anonymous_volumes_out_of_scope": True,
        },
    }
    return binding, observed


def emit(event: str, **details: Any) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True, separators=(",", ":")), flush=True)


def exact_target(value: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = sanitized_container(value)
    expected_actual = {key: expected[key] for key in ("name", "id", "image", "state", "config_sha256")}
    if actual != expected_actual:
        raise CleanupError("target_drift:" + expected["name"])
    validate_target(value, expected["kind"])


def remove_containers(binding: dict[str, Any]) -> None:
    for expected in binding["targets"]:
        values = docker_json(["container", "inspect", expected["id"]])
        if not isinstance(values, list) or len(values) != 1:
            raise CleanupError("target_missing_before_action:" + expected["name"])
        exact_target(values[0], expected)
        run(["docker", "container", "rm", expected["id"]])
        emit("container_removed", name=expected["name"], id=expected["id"])


def remove_wheels(binding: dict[str, Any]) -> None:
    expected = binding["wheelhouse"]
    current = wheel_binding()
    if current != expected:
        raise CleanupError("wheelhouse_drift_before_action")
    directory_fd = os.open(WHEELHOUSE, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        for row in expected["files"]:
            fd = os.open(row["filename"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                value = os.fstat(fd)
                if (not stat.S_ISREG(value.st_mode) or value.st_uid != 0 or value.st_nlink != 1 or value.st_mode & 0o022
                        or value.st_dev != row["device"] or value.st_ino != row["inode"] or value.st_size != row["size"]
                        or digest_fd(fd) != row["sha256"]):
                    raise CleanupError("wheel_drift_before_unlink:" + row["filename"])
            finally:
                os.close(fd)
            os.unlink(row["filename"], dir_fd=directory_fd)
            emit("wheel_unlinked", filename=row["filename"], bytes=row["size"])
    finally:
        os.close(directory_fd)
    os.rmdir(WHEELHOUSE)
    emit("wheelhouse_removed", path=str(WHEELHOUSE))


def remove_accel_image(binding: dict[str, Any]) -> None:
    image_identity(ACCEL_TAG, ACCEL_IMAGE, {ACCEL_TAG})
    values = all_containers()
    if any(value.get("Image") == ACCEL_IMAGE for value in values):
        raise CleanupError("accel_image_still_referenced")
    run(["docker", "image", "rm", ACCEL_TAG])
    emit("image_tag_removed", reference=ACCEL_TAG, id=ACCEL_IMAGE)
    if ACCEL_IMAGE in all_image_ids():
        raise CleanupError("accel_image_id_present_after_tag_rm")
    emit("image_id_absent", id=ACCEL_IMAGE)


def verify_after(binding: dict[str, Any], available_before: int) -> dict[str, Any]:
    safe_file(COMPOSE, COMPOSE_SHA256)
    values = all_containers()
    rows = sorted((sanitized_container(value) for value in values), key=lambda row: row["name"])
    if rows != binding["non_targets"]:
        raise CleanupError("non_target_container_drift_after")
    if any(row["name"] in {target["name"] for target in binding["targets"]} for row in rows):
        raise CleanupError("target_present_after")
    image_identity(BASELINE_IMAGE, BASELINE_IMAGE)
    image_identity(SYV_IMAGE, SYV_IMAGE)
    for mount in binding["retained_mount_stats"]:
        if stat_identity(mount["path"]) != mount:
            raise CleanupError("retained_mount_drift_after:" + mount["path"])
    if WHEELHOUSE.exists() or WHEELHOUSE.is_symlink():
        raise CleanupError("wheelhouse_present_after")
    if ACCEL_IMAGE in all_image_ids():
        raise CleanupError("accel_image_present_after")
    usage = os.statvfs("/srv/ai-server")
    available_after = usage.f_bavail * usage.f_frsize
    return {
        "targets_absent": True,
        "non_targets_unchanged": True,
        "compose_unchanged": True,
        "retained_mounts_unchanged": True,
        "baseline_image_present": True,
        "syv_image_present": True,
        "accel_image_absent": True,
        "wheelhouse_absent": True,
        "available_before_bytes": available_before,
        "available_after_bytes": available_after,
        "available_delta_bytes": available_after - available_before,
    }


def load_manifest(path: Path, expected_sha: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha):
        raise CleanupError("manifest_sha256_format")
    _, actual_sha, raw = read_safe_file(path, expected_sha)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CleanupError("manifest_json_invalid") from exc
    if not isinstance(value, dict) or set(value) != {"schema", "binding", "observed"} or value.get("schema") != SCHEMA:
        raise CleanupError("manifest_schema")
    # The exact outer file hash is the authority; this event aids the journal.
    emit("manifest_loaded", path=str(path), sha256=actual_sha)
    return value


def inventory() -> None:
    locks = acquire_locks()
    try:
        binding, observed = collect_binding()
        result = {"schema": SCHEMA, "binding": binding, "observed": observed}
        print(canonical(result).decode(), end="", flush=True)
    finally:
        release_locks(locks)


def apply(manifest_path: Path, expected_sha: str) -> None:
    locks = acquire_locks()
    mutated = False
    try:
        manifest = load_manifest(manifest_path, expected_sha)
        current_binding, current_observed = collect_binding()
        if current_binding != manifest["binding"]:
            raise CleanupError("manifest_preflight_drift")
        available_before = current_observed["available_bytes"]
        emit("preflight_pass", binding_sha256=digest_bytes(canonical(current_binding)), available_before_bytes=available_before)
        mutated = True
        remove_containers(current_binding)
        remove_wheels(current_binding)
        remove_accel_image(current_binding)
        emit("verification_pass", **verify_after(current_binding, available_before))
    except Exception as exc:
        emit("failed", error=type(exc).__name__, reason=str(exc), mutation_started=mutated)
        raise
    finally:
        release_locks(locks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inventory", action="store_true", help="emit the exact read-only JSON manifest to stdout")
    group.add_argument("--apply", nargs=2, metavar=("MANIFEST", "SHA256"), help="apply one exact reviewed inventory manifest")
    args = parser.parse_args()
    if os.name != "posix" or os.geteuid() != 0:
        raise CleanupError("root_posix_required")
    if args.inventory:
        inventory()
    else:
        apply(Path(args.apply[0]), args.apply[1])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"event": "fatal", "error": type(exc).__name__, "reason": str(exc)}, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(1)
