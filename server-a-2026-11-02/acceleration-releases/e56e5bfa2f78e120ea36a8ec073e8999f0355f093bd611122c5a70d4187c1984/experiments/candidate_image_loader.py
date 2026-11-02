#!/usr/bin/env python3
"""Fail-closed loading of package-embedded candidate OCI images.

The package verifier establishes package custody.  This module repeats the
image-specific bindings at the runtime boundary and then verifies the exact
identity Docker made available under the sealed local tag.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Protocol, Sequence

try:
    from . import candidate_oci_build
except ImportError:  # pragma: no cover - direct execution
    import candidate_oci_build  # type: ignore[no-redef]


DOCKER = "/usr/bin/docker"
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CUSTOM_IMAGES = {
    "freetoken": "freetoken",
    "llama-dflash2-official": "llama-dflash2-official",
}


class ImageLoadError(RuntimeError):
    """The embedded image cannot be proven safe for candidate execution."""

    def __init__(self, code: str, *, ambiguous: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.ambiguous = ambiguous


class Result(Protocol):
    returncode: int
    stdout: str
    stderr: str


class Runner(Protocol):
    def run(
        self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None
    ) -> Result: ...


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, root: Path, code: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ImageLoadError(code)
    try:
        resolved = path.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise ImageLoadError(code) from exc
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root_resolved)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise ImageLoadError(code)
    return resolved


def _canonical_json(path: Path, root: Path, code: str) -> tuple[dict[str, Any], bytes]:
    path = _regular_file(path, root, code)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImageLoadError(code) from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ImageLoadError(code)
    return value, raw


def validate_embedded_image(
    package_root: Path,
    candidate_id: str,
    candidate_pin: str,
    expected_image: Mapping[str, object],
) -> dict[str, object]:
    """Validate archive/lock/receipt/candidate bindings without invoking Docker."""

    if candidate_id not in CUSTOM_IMAGES or not re.fullmatch(r"[0-9a-f]{40}", candidate_pin):
        raise ImageLoadError("embedded_image_candidate_invalid")
    if (
        not package_root.is_absolute()
        or package_root.is_symlink()
        or not package_root.is_dir()
        or package_root.resolve(strict=True) != package_root.absolute()
    ):
        raise ImageLoadError("embedded_image_package_root_invalid")
    stem = CUSTOM_IMAGES[candidate_id]
    docker_archive = package_root / "artifacts" / "images" / f"{stem}.docker.tar"
    oci_archive = package_root / "artifacts" / "images" / f"{stem}.oci.tar"
    lock_path = package_root / "experiments" / "immutable_metadata" / "images" / f"{stem}.oci-lock.json"
    receipt_path = package_root / "experiments" / "immutable_metadata" / "images" / f"{stem}.oci-build-receipt.json"
    docker_archive = _regular_file(
        docker_archive, package_root, "embedded_docker_archive_invalid"
    )
    oci_archive = _regular_file(
        oci_archive, package_root, "embedded_oci_archive_invalid"
    )
    lock, lock_raw = _canonical_json(lock_path, package_root, "embedded_image_lock_invalid")
    receipt, receipt_raw = _canonical_json(receipt_path, package_root, "embedded_image_receipt_invalid")
    try:
        verified_lock, runtime, labels = candidate_oci_build._load_transport_lock(
            docker_archive, oci_archive, lock_path
        )
    except (candidate_oci_build.OciBuildError, OSError, ValueError, TypeError) as exc:
        raise ImageLoadError("embedded_image_transport_invalid") from exc
    if verified_lock != lock:
        raise ImageLoadError("embedded_image_lock_changed_during_validation")

    image = lock.get("image")
    archives = lock.get("archives")
    receipt_lock = lock.get("receipt")
    oci = lock.get("oci")
    probe = lock.get("docker_load_probe")
    source = lock.get("source")
    expected_keys = {"reference", "digest", "lock_sha256", "revision_label"}
    if set(expected_image) != expected_keys:
        raise ImageLoadError("embedded_image_candidate_binding_invalid")
    if (
        lock.get("format") != "ai-server-candidate-image-lock-v2"
        or lock.get("candidate_id") != candidate_id
        or lock.get("revision") != candidate_pin
        or not isinstance(image, dict)
        or image.get("digest_kind") != "docker-image-config"
        or image.get("digest") != image.get("docker_image_id")
        or image.get("reference") != expected_image.get("reference")
        or image.get("digest") != expected_image.get("digest")
        or expected_image.get("revision_label") != candidate_pin
        or expected_image.get("lock_sha256") != hashlib.sha256(lock_raw).hexdigest()
        or not isinstance(image.get("reference"), str)
        or "@" in str(image.get("reference"))
        or SHA256.fullmatch(str(image.get("digest"))) is None
        or SHA256.fullmatch(str(image.get("manifest_digest"))) is None
    ):
        raise ImageLoadError("embedded_image_candidate_binding_invalid")
    if (
        not isinstance(archives, dict)
        or archives != {
            "docker": {
                "filename": docker_archive.name,
                "format": "docker-image-save-tar",
                "sha256": _sha256(docker_archive),
                "size": docker_archive.stat().st_size,
            },
            "oci": {
                "filename": oci_archive.name,
                "format": "oci-image-layout-tar",
                "sha256": _sha256(oci_archive),
                "size": oci_archive.stat().st_size,
            },
        }
        or not isinstance(receipt_lock, dict)
        or receipt_lock != {
            "filename": receipt_path.name,
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "size": len(receipt_raw),
        }
        or receipt.get("candidate_id") != candidate_id
        or receipt.get("output") != oci
    ):
        raise ImageLoadError("embedded_image_evidence_binding_invalid")
    locked_runtime = probe.get("expected_runtime_identity") if isinstance(probe, dict) else None
    if (
        not isinstance(locked_runtime, dict)
        or locked_runtime != runtime
        or set(runtime) != {"architecture", "os", "config_digest", "rootfs_diff_ids", "manifest_digest"}
        or runtime.get("architecture") != "amd64"
        or runtime.get("os") != "linux"
        or runtime.get("config_digest") != image.get("digest")
        or runtime.get("manifest_digest") != image.get("manifest_digest")
        or not isinstance(runtime.get("rootfs_diff_ids"), list)
        or not runtime["rootfs_diff_ids"]
        or any(SHA256.fullmatch(str(item)) is None for item in runtime["rootfs_diff_ids"])
        or not isinstance(labels, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items())
        or probe.get("argv") != ["docker", "image", "load", "--input", docker_archive.name]
        or probe.get("network_required") is not False
    ):
        raise ImageLoadError("embedded_image_runtime_identity_invalid")
    build_input = source.get("sealed_input_manifest_sha256") if isinstance(source, dict) else None
    if not isinstance(build_input, str) or HEX64.fullmatch(build_input) is None:
        raise ImageLoadError("embedded_image_build_input_identity_invalid")
    return {
        "docker_archive": str(docker_archive),
        "oci_archive": str(oci_archive),
        "build_input_manifest_sha256": build_input,
        "image_digest": image["digest"],
        "labels": dict(labels),
        "manifest_digest": image["manifest_digest"],
        "reference": image["reference"],
        "rootfs_diff_ids": list(runtime["rootfs_diff_ids"]),
        "platform": "linux/amd64",
    }


def _inspect(runner: Runner, reference: str) -> tuple[str, Mapping[str, Any] | None]:
    result = runner.run(
        (DOCKER, "image", "inspect", "--platform", "linux/amd64", reference),
        timeout=30,
    )
    if result.returncode == 0:
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ImageLoadError("docker_image_inspect_invalid", ambiguous=True) from exc
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise ImageLoadError("docker_image_inspect_invalid", ambiguous=True)
        return "present", values[0]
    expected = {
        f"Error: No such image: {reference}",
        f"Error: No such object: {reference}",
        f"Error response from daemon: No such image: {reference}",
        f"Error response from daemon: No such object: {reference}",
    }
    if result.returncode == 1 and not result.stdout.strip() and result.stderr.strip() in expected:
        return "absent", None
    raise ImageLoadError("docker_image_state_unknown", ambiguous=True)


def _verify_runtime(value: Mapping[str, Any], identity: Mapping[str, object]) -> None:
    config = value.get("Config")
    rootfs = value.get("RootFS")
    repo_tags = value.get("RepoTags")
    if (
        value.get("Id") != identity["image_digest"]
        or value.get("Architecture") != "amd64"
        or value.get("Os") != "linux"
        or not isinstance(config, dict)
        or config.get("Labels") != identity["labels"]
        or not isinstance(rootfs, dict)
        or rootfs.get("Type") != "layers"
        or rootfs.get("Layers") != identity["rootfs_diff_ids"]
        or not isinstance(repo_tags, list)
        or repo_tags.count(identity["reference"]) != 1
    ):
        raise ImageLoadError("loaded_image_runtime_identity_mismatch", ambiguous=True)


def ensure_embedded_image_loaded(
    package_root: Path,
    candidate_id: str,
    candidate_pin: str,
    expected_image: Mapping[str, object],
    runner: Runner,
) -> dict[str, object]:
    """Load a missing sealed tag, or safely reuse an already exact tag."""

    identity = validate_embedded_image(package_root, candidate_id, candidate_pin, expected_image)
    state, value = _inspect(runner, str(identity["reference"]))
    if state == "present":
        assert value is not None
        _verify_runtime(value, identity)
        return {**identity, "status": "reused"}
    result = runner.run(
        (DOCKER, "image", "load", "--input", str(identity["docker_archive"])), timeout=1800
    )
    if result.returncode != 0:
        raise ImageLoadError("stock_docker_image_load_failed", ambiguous=True)
    state, value = _inspect(runner, str(identity["reference"]))
    if state != "present" or value is None:
        raise ImageLoadError("loaded_image_not_present", ambiguous=True)
    _verify_runtime(value, identity)
    return {**identity, "status": "loaded"}
