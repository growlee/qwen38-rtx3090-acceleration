#!/usr/bin/env python3
"""Deterministically seal and offline-verify the acceleration experiment package."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
from typing import BinaryIO, Iterable, Mapping

try:
    from . import candidate_artifact_custody as custody
except ImportError:  # pragma: no cover - direct script execution
    import candidate_artifact_custody as custody  # type: ignore[no-redef]


POLICY_VERSION = "ai-server-acceleration-package-inventory/v3"
MANIFEST_SCHEMA = "ai-server-acceleration-package-manifest/v2"
RECEIPT_SCHEMA = "ai-server-acceleration-package-receipt/v2"
ARCHIVE_NAME = "candidate-package.tar"
RECEIPT_NAME = "candidate-package.receipt.json"
MANIFEST_MEMBER = "MANIFEST.json"
PAYLOAD_PREFIX = "payload"

CORE_FILES = (
    "experiments/candidate_contract.py",
    "experiments/candidate_build_contract.py",
    "experiments/candidate_artifact_custody.py",
    "experiments/candidate_host_collector.py",
    "experiments/candidate_image_loader.py",
    "experiments/host_compatibility_preflight.py",
    "experiments/safety_sampler.py",
    "experiments/candidate_arm_harness.py",
    "experiments/candidate_live_operation.py",
    "experiments/candidate_live_backend.py",
    "experiments/candidate_route_evidence.py",
    "experiments/candidate_live_cli.py",
    "experiments/candidate_package.py",
    "experiments/candidate_storage_budget.py",
    "experiments/candidate_payload_quarantine.py",
    "experiments/candidate_quality_evaluator.py",
    "experiments/candidate_artifact_prepare.py",
    "experiments/candidate_build_inputs.py",
    "experiments/candidate_oci_build.py",
    "experiments/llama_analysis_build.py",
)

CANDIDATE_FILES: Mapping[str, tuple[str, ...]] = {
    "qwen38_vllm": (
        "candidate.json",
        "compose.dflash2.yaml",
        "compose.mtp.yaml",
    ),
    "freetoken": (
        "artifact-manifest.json",
        "artifact-supplements.json",
        "candidate.json",
        "compose.yaml",
        "Dockerfile",
        "oci-build-receipt.schema.json",
        "pylock.freetoken.toml",
        "requirements.in",
        "requirements.lock",
        "requirements.offline.lock",
        "requirements.resolve.in",
        "requirements.resolved.lock",
        "resolver-receipt.json",
        "seal_resolution.py",
        "size-overrides.json",
    ),
    "llama_dflash2_official": (
        "candidate.json",
        "compose.dflash2.yaml",
        "compose.mtp.yaml",
        "compose.plain.yaml",
        "Dockerfile",
        "healthcheck.c",
        "resolve_ubuntu_snapshot.py",
        "runtime-elf-evidence.schema.json",
        "runtime-package-resolution.schema.json",
        "seal-elf-runtime.sh",
        "ubuntu-snapshot-evidence.schema.json",
    ),
    "higgs_tts3": (
        "candidate.json",
        "compose.yaml",
        "higgs-3090.yaml",
        "install_model.py",
        "model-manifest.json",
        "oci-layer-receipt.json",
    ),
}

CANDIDATE_SOURCE_DIRECTORIES: Mapping[str, str] = {
    "qwen38-vllm": "qwen38_vllm",
    "freetoken": "freetoken",
    "llama-dflash2-official": "llama_dflash2_official",
    "higgs-tts3": "higgs_tts3",
}

BENCHMARK_FILES = (
    "benchmarks/candidate_benchmark_runner.py",
    "benchmarks/candidate_openai_smoke.py",
    "benchmarks/freetoken_bandwidth_runner.py",
    "benchmarks/suites/acceleration-benchmark-v1.json",
    "benchmarks/suites/acceleration-freetoken-bandwidth-v1.json",
    "benchmarks/suites/acceleration-quality-v1.json",
)

IMMUTABLE_METADATA_FILES = (
    "experiments/immutable_metadata/images/qwen38-vllm.oci-lock.json",
    "experiments/immutable_metadata/images/freetoken.oci-build-receipt.json",
    "experiments/immutable_metadata/images/freetoken.oci-lock.json",
    "experiments/immutable_metadata/images/llama-dflash2-official.oci-build-receipt.json",
    "experiments/immutable_metadata/images/llama-dflash2-official.oci-lock.json",
    "experiments/immutable_metadata/models/freetoken.target.json",
    "experiments/immutable_metadata/models/higgs-tts3.target.json",
    "experiments/immutable_metadata/models/llama-dflash2-official.drafter.json",
    "experiments/immutable_metadata/models/llama-dflash2-official.target.json",
    "experiments/immutable_metadata/models/qwen38-vllm.drafter.json",
    "experiments/immutable_metadata/models/qwen38-vllm.fast-variant.json",
    "experiments/immutable_metadata/models/qwen38-vllm.target.json",
    "experiments/immutable_metadata/suites/higgs-tts3-russian-smoke.json",
)

EXTERNAL_IMAGE_ARTIFACTS: Mapping[str, Mapping[str, str]] = {
    "freetoken": {
        "candidate_path": "experiments/freetoken/candidate.json",
        "lock_path": "experiments/immutable_metadata/images/freetoken.oci-lock.json",
        "receipt_path": "experiments/immutable_metadata/images/freetoken.oci-build-receipt.json",
        "stem": "freetoken",
    },
    "llama-dflash2-official": {
        "candidate_path": "experiments/llama_dflash2_official/candidate.json",
        "lock_path": "experiments/immutable_metadata/images/llama-dflash2-official.oci-lock.json",
        "receipt_path": "experiments/immutable_metadata/images/llama-dflash2-official.oci-build-receipt.json",
        "stem": "llama-dflash2-official",
    },
}

CANDIDATE_SCOPE_IDS = tuple(CANDIDATE_SOURCE_DIRECTORIES)
_SCOPED_METADATA_FILES: Mapping[str, tuple[str, ...]] = {
    candidate_id: tuple(
        path for path in IMMUTABLE_METADATA_FILES
        if f"/{candidate_id}." in path
    )
    for candidate_id in CANDIDATE_SCOPE_IDS
}


def _scoped_core_files(candidate_id: str) -> tuple[str, ...]:
    excluded = {
        "qwen38-vllm": {
            "experiments/candidate_build_inputs.py",
            "experiments/llama_analysis_build.py",
        },
        "freetoken": {"experiments/llama_analysis_build.py"},
        "llama-dflash2-official": set(),
        "higgs-tts3": {
            "experiments/candidate_build_inputs.py",
            "experiments/llama_analysis_build.py",
        },
    }.get(candidate_id)
    if excluded is None:
        raise PackageError("candidate package scope is not allowlisted")
    return tuple(path for path in CORE_FILES if path not in excluded)


def _scope_artifact_candidate_ids(candidate_id: str | None) -> tuple[str, ...]:
    if candidate_id is None:
        return tuple(sorted(EXTERNAL_IMAGE_ARTIFACTS))
    if candidate_id not in CANDIDATE_SOURCE_DIRECTORIES:
        raise PackageError("candidate package scope is not allowlisted")
    return (candidate_id,) if candidate_id in EXTERNAL_IMAGE_ARTIFACTS else ()


def _scope_archive_keys(candidate_id: str | None) -> tuple[str, ...]:
    return tuple(
        f"{artifact_candidate_id}.{transport}"
        for artifact_candidate_id in _scope_artifact_candidate_ids(candidate_id)
        for transport in ("docker", "oci")
    )


EXTERNAL_ARCHIVE_KEYS = _scope_archive_keys(None)

EXCLUDED_RUNTIME_DIRECTORIES = frozenset({
    "__pycache__", "download", "downloads", "evidence", "payload", "payloads",
    "receipts", "secrets", "tests",
})


class PackageError(RuntimeError):
    """Raised when the package identity or source inventory is unsafe."""


def _canonical(value: object) -> bytes:
    return custody._canonical_json(value)


def _policy_paths(candidate_id: str | None = None) -> tuple[str, ...]:
    if candidate_id is None:
        candidate_paths = (
            f"experiments/{candidate}/{name}"
            for candidate, names in CANDIDATE_FILES.items()
            for name in names
        )
        return tuple(sorted((*CORE_FILES, *candidate_paths, *BENCHMARK_FILES, *IMMUTABLE_METADATA_FILES)))
    candidate_directory = CANDIDATE_SOURCE_DIRECTORIES.get(candidate_id)
    if candidate_directory is None:
        raise PackageError("candidate package scope is not allowlisted")
    candidate_paths = (
        f"experiments/{candidate_directory}/{name}"
        for name in CANDIDATE_FILES[candidate_directory]
    )
    return tuple(sorted((
        *_scoped_core_files(candidate_id),
        *candidate_paths,
        *BENCHMARK_FILES,
        *_SCOPED_METADATA_FILES[candidate_id],
    )))


POLICY_PATHS = _policy_paths()


def _real_root(path: Path) -> Path:
    try:
        return custody._real_directory(path.absolute(), "repository root")
    except custody.CustodyError as exc:
        raise PackageError(str(exc)) from exc


def _safe_path(value: str, label: str = "package path") -> str:
    try:
        relative = custody._safe_relative(value, label)
    except custody.CustodyError as exc:
        raise PackageError(str(exc)) from exc
    canonical = relative.as_posix()
    if canonical != value or "\\" in value or ":" in value or "\x00" in value:
        raise PackageError(f"{label} must be a canonical POSIX relative path")
    return canonical


def _source_file(root: Path, relative: str) -> tuple[Path, os.stat_result]:
    canonical = _safe_path(relative, "source path")
    path = root.joinpath(*PurePosixPath(canonical).parts)
    try:
        parent = custody._real_directory(path.parent, "source parent")
        metadata = custody._regular_file(path, canonical)
    except custody.CustodyError as exc:
        raise PackageError(str(exc)) from exc
    if not parent.is_relative_to(root) or path.resolve(strict=True) != path.absolute():
        raise PackageError(f"source path escapes repository root: {canonical}")
    return path, metadata


def _candidate_inventory(root: Path, candidate_id: str | None = None) -> None:
    """Reject anything not explicitly classified in a candidate source subtree."""

    if candidate_id is None:
        candidates = CANDIDATE_FILES.items()
    else:
        candidate_directory = CANDIDATE_SOURCE_DIRECTORIES.get(candidate_id)
        if candidate_directory is None:
            raise PackageError("candidate package scope is not allowlisted")
        candidates = ((candidate_directory, CANDIDATE_FILES[candidate_directory]),)
    for candidate, names in candidates:
        directory = root / "experiments" / candidate
        expected = set(names)
        try:
            custody._real_directory(directory, f"{candidate} source directory")
        except custody.CustodyError as exc:
            raise PackageError(str(exc)) from exc
        found: set[str] = set()
        for current, directories, files in os.walk(directory, topdown=True, followlinks=False):
            current_path = Path(current)
            for name in directories:
                child = current_path / name
                if child.is_symlink():
                    raise PackageError(f"symlink is forbidden in candidate subtree: {child.relative_to(root).as_posix()}")
            directories[:] = [name for name in directories if name not in EXCLUDED_RUNTIME_DIRECTORIES]
            for name in files:
                child = current_path / name
                relative_candidate = child.relative_to(directory).as_posix()
                if child.is_symlink():
                    raise PackageError(f"symlink is forbidden in candidate subtree: {child.relative_to(root).as_posix()}")
                if (
                    name.endswith((".partial", ".pyc"))
                    or "secret" in name.lower()
                    or ("receipt" in name.lower() and relative_candidate not in expected)
                ):
                    continue
                found.add(relative_candidate)
        unexpected = sorted(found - expected)
        missing = sorted(expected - found)
        if unexpected:
            raise PackageError(f"unexpected candidate runtime file: experiments/{candidate}/{unexpected[0]}")
        if missing:
            raise PackageError(f"required candidate source is missing: experiments/{candidate}/{missing[0]}")


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        metadata = custody._regular_file(path, label)
        raw = path.read_bytes()
        value = json.loads(raw)
    except (custody.CustodyError, OSError, ValueError) as exc:
        raise PackageError(f"{label} is invalid") from exc
    if metadata.st_size != len(raw) or not isinstance(value, dict) or raw != _canonical(value) + b"\n":
        raise PackageError(f"{label} is not canonical")
    return value, raw


def _sha_descriptor(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"media_type", "digest", "size"}:
        raise PackageError(f"{label} descriptor is invalid")
    digest = value.get("digest")
    size = value.get("size")
    try:
        custody._validate_sha256(str(digest)[7:] if str(digest).startswith("sha256:") else "")
    except custody.CustodyError as exc:
        raise PackageError(f"{label} descriptor is invalid") from exc
    if not isinstance(value.get("media_type"), str) or type(size) is not int or size <= 0:
        raise PackageError(f"{label} descriptor is invalid")
    return dict(value)


def _oci_member_bytes(archive: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo], name: str) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile():
        raise PackageError(f"OCI archive member is missing: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise PackageError(f"OCI archive member is unreadable: {name}")
    return stream.read()


def _oci_blob_identity(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    descriptor: Mapping[str, object],
    label: str,
) -> None:
    digest = str(descriptor["digest"])
    name = "blobs/sha256/" + digest[7:]
    member = members.get(name)
    if member is None or not member.isfile() or member.size != descriptor["size"]:
        raise PackageError(f"OCI {label} blob identity mismatch")
    stream = archive.extractfile(member)
    if stream is None or _hash_member(stream) != (digest[7:], descriptor["size"]):
        raise PackageError(f"OCI {label} blob identity mismatch")


def _inspect_oci_archive(path: Path | BinaryIO) -> dict[str, object]:
    file_path = path if isinstance(path, Path) else None
    try:
        if file_path is not None:
            metadata = custody._regular_file(file_path, "external OCI archive")
            if file_path.resolve(strict=True) != file_path.absolute():
                raise PackageError("external OCI archive path must be canonical")
            opened = tarfile.open(file_path, mode="r:*")
        else:
            metadata = None
            opened = tarfile.open(fileobj=path, mode="r:*")
        with opened as archive:
            members: dict[str, tarfile.TarInfo] = {}
            for member in archive.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute() or not pure.parts
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or member.name in members
                    or not (member.isfile() or member.isdir())
                ):
                    raise PackageError("external OCI archive contains an unsafe member")
                members[member.name] = member
            layout = json.loads(_oci_member_bytes(archive, members, "oci-layout"))
            if layout != {"imageLayoutVersion": "1.0.0"}:
                raise PackageError("external OCI archive layout is invalid")
            index_raw = _oci_member_bytes(archive, members, "index.json")
            index = json.loads(index_raw)
            manifests = index.get("manifests") if isinstance(index, dict) else None
            if not isinstance(index, dict) or index.get("schemaVersion") != 2 or not isinstance(manifests, list) or len(manifests) != 1:
                raise PackageError("external OCI archive must contain exactly one image")
            selected = manifests[0]
            if not isinstance(selected, dict):
                raise PackageError("OCI manifest descriptor is invalid")
            manifest = _sha_descriptor({
                "media_type": selected.get("mediaType"), "digest": selected.get("digest"), "size": selected.get("size"),
            }, "OCI manifest")
            if manifest["media_type"] != "application/vnd.oci.image.manifest.v1+json":
                raise PackageError("OCI manifest media type is invalid")
            annotations = selected.get("annotations")
            tag = annotations.get("org.opencontainers.image.ref.name") if isinstance(annotations, dict) else None
            if not isinstance(tag, str) or not tag or "@" in tag:
                raise PackageError("OCI index tag annotation is invalid")
            _oci_blob_identity(archive, members, manifest, "manifest")
            manifest_raw = _oci_member_bytes(archive, members, "blobs/sha256/" + str(manifest["digest"])[7:])
            manifest_json = json.loads(manifest_raw)
            if not isinstance(manifest_json, dict) or manifest_json.get("schemaVersion") != 2:
                raise PackageError("OCI manifest is invalid")
            config_source = manifest_json.get("config")
            if not isinstance(config_source, dict):
                raise PackageError("OCI config descriptor is invalid")
            config = _sha_descriptor({
                "media_type": config_source.get("mediaType"), "digest": config_source.get("digest"), "size": config_source.get("size"),
            }, "OCI config")
            if config["media_type"] != "application/vnd.oci.image.config.v1+json":
                raise PackageError("OCI config media type is invalid")
            layers_source = manifest_json.get("layers")
            if not isinstance(layers_source, list) or not layers_source:
                raise PackageError("OCI layers are missing")
            layers = []
            for item in layers_source:
                if not isinstance(item, dict):
                    raise PackageError("OCI layer descriptor is invalid")
                layer = _sha_descriptor({
                    "media_type": item.get("mediaType"), "digest": item.get("digest"), "size": item.get("size"),
                }, "OCI layer")
                if layer["media_type"] not in {
                    "application/vnd.oci.image.layer.v1.tar+gzip",
                    "application/vnd.oci.image.layer.v1.tar+zstd",
                }:
                    raise PackageError("OCI layer media type is invalid")
                layers.append(layer)
            if len({str(item["digest"]) for item in layers}) != len(layers):
                raise PackageError("OCI layer digest is duplicated")
            _oci_blob_identity(archive, members, config, "config")
            for layer in layers:
                _oci_blob_identity(archive, members, layer, "layer")
            expected_files = {
                "oci-layout", "index.json",
                "blobs/sha256/" + str(manifest["digest"])[7:],
                "blobs/sha256/" + str(config["digest"])[7:],
                *("blobs/sha256/" + str(layer["digest"])[7:] for layer in layers),
            }
            if set(members) != expected_files or any(not member.isfile() for member in members.values()):
                raise PackageError("OCI archive contains a hybrid or unexpected layout")
            config_raw = _oci_member_bytes(archive, members, "blobs/sha256/" + str(config["digest"])[7:])
            config_json = json.loads(config_raw)
            rootfs = config_json.get("rootfs") if isinstance(config_json, dict) else None
            diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, dict) else None
            if (
                not isinstance(config_json, dict)
                or config_json.get("architecture") != "amd64"
                or config_json.get("os") != "linux"
                or not isinstance(rootfs, dict) or rootfs.get("type") != "layers"
                or not isinstance(diff_ids, list) or len(diff_ids) != len(layers)
                or any(not isinstance(item, str) or not item.startswith("sha256:") for item in diff_ids)
            ):
                raise PackageError("OCI runtime config identity is invalid")
    except (custody.CustodyError, tarfile.TarError, OSError, ValueError) as exc:
        if isinstance(exc, PackageError):
            raise
        raise PackageError("external OCI archive is invalid") from exc
    result = {
        "index": {
            "media_type": "application/vnd.oci.image.index.v1+json",
            "digest": "sha256:" + hashlib.sha256(index_raw).hexdigest(),
            "size": len(index_raw),
        },
        "manifest": manifest,
        "config": config,
        "layers": layers,
        "runtime": {
            "architecture": "amd64", "os": "linux", "config_digest": config["digest"],
            "manifest_digest": manifest["digest"], "rootfs_diff_ids": diff_ids,
        },
        "tag": tag,
    }
    if file_path is not None and metadata is not None:
        result["archive_sha256"] = custody._sha256_file(file_path)
        result["archive_size"] = metadata.st_size
    return result


def _inspect_docker_archive(
    path: Path | BinaryIO,
    expected_reference: str,
    expected_runtime: Mapping[str, object],
) -> dict[str, object]:
    file_path = path if isinstance(path, Path) else None
    try:
        if file_path is not None:
            metadata = custody._regular_file(file_path, "external Docker archive")
            if file_path.resolve(strict=True) != file_path.absolute():
                raise PackageError("external Docker archive path must be canonical")
            opened = tarfile.open(file_path, mode="r:*")
        else:
            metadata = None
            opened = tarfile.open(fileobj=path, mode="r:*")
        with opened as archive:
            members: dict[str, tarfile.TarInfo] = {}
            for member in archive.getmembers():
                name = _safe_path(member.name, "Docker archive member")
                if name in members or not member.isfile():
                    raise PackageError("Docker archive contains duplicate or non-file members")
                members[name] = member
            manifest_raw = _oci_member_bytes(archive, members, "manifest.json")
            manifests = json.loads(manifest_raw)
            if not isinstance(manifests, list) or len(manifests) != 1 or not isinstance(manifests[0], dict):
                raise PackageError("Docker archive must contain exactly one image")
            entry = manifests[0]
            if set(entry) != {"Config", "RepoTags", "Layers"} or entry.get("RepoTags") != [expected_reference]:
                raise PackageError("Docker archive manifest identity is invalid")
            config_name = entry.get("Config")
            layer_names = entry.get("Layers")
            names = [config_name, *(layer_names if isinstance(layer_names, list) else [])]
            if (
                not isinstance(config_name, str) or not isinstance(layer_names, list) or not layer_names
                or any(not isinstance(name, str) or not name.startswith("blobs/sha256/") or len(name) != 77 for name in names)
            ):
                raise PackageError("Docker archive blob path is invalid")
            required = {"manifest.json", "repositories", config_name, *layer_names}
            if set(members) != required:
                raise PackageError("Docker archive contains a hybrid or unexpected layout")
            config_raw = _oci_member_bytes(archive, members, config_name)
            config_digest = "sha256:" + hashlib.sha256(config_raw).hexdigest()
            if config_name != "blobs/sha256/" + config_digest[7:]:
                raise PackageError("Docker archive config identity mismatch")
            config = json.loads(config_raw)
            rootfs = config.get("rootfs") if isinstance(config, dict) else None
            diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, dict) else None
            if (
                not isinstance(config, dict) or config.get("architecture") != "amd64" or config.get("os") != "linux"
                or not isinstance(rootfs, dict) or rootfs.get("type") != "layers"
                or not isinstance(diff_ids, list) or len(diff_ids) != len(layer_names)
            ):
                raise PackageError("Docker archive runtime identity is invalid")
            layers: list[dict[str, object]] = []
            for name, diff_id in zip(layer_names, diff_ids, strict=True):
                member = members[name]
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageError("Docker archive layer is unreadable")
                digest, size = _hash_member(stream)
                if diff_id != "sha256:" + digest or name != "blobs/sha256/" + digest:
                    raise PackageError("Docker archive layer order or diffID identity mismatch")
                layers.append({"path": name, "digest": diff_id, "size": size})
            repositories = json.loads(_oci_member_bytes(archive, members, "repositories"))
            repository, tag = expected_reference.rsplit(":", 1)
            if repositories != {repository: {tag: layer_names[-1].rsplit("/", 1)[1]}}:
                raise PackageError("Docker archive repository identity is invalid")
            if (
                expected_runtime.get("config_digest") != config_digest
                or expected_runtime.get("rootfs_diff_ids") != diff_ids
                or expected_runtime.get("architecture") != "amd64"
                or expected_runtime.get("os") != "linux"
            ):
                raise PackageError("Docker and OCI runtime identities disagree")
    except (custody.CustodyError, tarfile.TarError, OSError, ValueError) as exc:
        if isinstance(exc, PackageError):
            raise
        raise PackageError("external Docker archive is invalid") from exc
    result: dict[str, object] = {
        "config": {"digest": config_digest, "size": len(config_raw)},
        "layers": layers,
        "reference": expected_reference,
        "rootfs_diff_ids": list(diff_ids),
    }
    if file_path is not None and metadata is not None:
        result["archive_sha256"] = custody._sha256_file(file_path)
        result["archive_size"] = metadata.st_size
    return result


def _artifact_records(
    root: Path,
    image_archives: Mapping[str, Path] | None,
    candidate_id: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    supplied = dict(image_archives or {})
    artifact_candidate_ids = _scope_artifact_candidate_ids(candidate_id)
    expected_keys = set(_scope_archive_keys(candidate_id))
    if set(supplied) != expected_keys:
        missing = sorted(expected_keys - set(supplied))
        extra = sorted(set(supplied) - expected_keys)
        detail = f"missing={','.join(missing)} extra={','.join(extra)}"
        raise PackageError(f"external image archive set is not exact: {detail}")
    if not artifact_candidate_ids:
        return [], {}
    records: list[dict[str, object]] = []
    sources: dict[str, Path] = {}
    for artifact_candidate_id in artifact_candidate_ids:
        policy = EXTERNAL_IMAGE_ARTIFACTS[artifact_candidate_id]
        docker_archive = supplied[f"{artifact_candidate_id}.docker"]
        oci_archive = supplied[f"{artifact_candidate_id}.oci"]
        stem = policy["stem"]
        if (
            not docker_archive.is_absolute() or not oci_archive.is_absolute()
            or docker_archive == oci_archive
            or docker_archive.name != f"{stem}.docker.tar"
            or oci_archive.name != f"{stem}.oci.tar"
        ):
            raise PackageError(f"{artifact_candidate_id} transport names or paths are not canonical")
        oci_details = _inspect_oci_archive(oci_archive)
        candidate, _ = _load_canonical_json(root.joinpath(*PurePosixPath(policy["candidate_path"]).parts), f"{artifact_candidate_id} candidate")
        lock_path = root.joinpath(*PurePosixPath(policy["lock_path"]).parts)
        receipt_path = root.joinpath(*PurePosixPath(policy["receipt_path"]).parts)
        lock, lock_raw = _load_canonical_json(lock_path, f"{artifact_candidate_id} image lock")
        build_receipt, receipt_raw = _load_canonical_json(receipt_path, f"{artifact_candidate_id} build receipt")
        image = candidate.get("image")
        source_identity = candidate.get("source")
        locked_image = lock.get("image")
        locked_oci = lock.get("oci")
        reference = image.get("reference") if isinstance(image, dict) else None
        revision = source_identity.get("revision") if isinstance(source_identity, dict) else None
        expected_repository = f"local/ai-server-{artifact_candidate_id}"
        expected_reference = f"{expected_repository}:{revision}"
        docker_details = _inspect_docker_archive(docker_archive, str(reference), oci_details["runtime"])
        if (
            candidate.get("candidate_id") != artifact_candidate_id
            or not isinstance(source_identity, dict)
            or not isinstance(source_identity.get("revision"), str)
            or len(source_identity["revision"]) != 40
            or any(character not in "0123456789abcdef" for character in source_identity["revision"])
            or not isinstance(image, dict)
            or not all(isinstance(image.get(key), str) and image[key] for key in ("reference", "digest", "lock_sha256"))
            or image.get("revision_label") != source_identity.get("revision")
            or image.get("reference") != expected_reference
            or not isinstance(locked_image, dict)
            or lock.get("format") != "ai-server-candidate-image-lock-v2"
            or lock.get("candidate_id") != artifact_candidate_id
            or lock.get("revision") != image.get("revision_label")
            or locked_image.get("digest_kind") != "docker-image-config"
            or image.get("reference") != locked_image.get("reference")
            or image.get("digest") != locked_image.get("digest")
            or image.get("digest") != locked_image.get("docker_image_id")
            or locked_image.get("manifest_digest") != oci_details["manifest"]["digest"]  # type: ignore[index]
            or image.get("lock_sha256") != hashlib.sha256(lock_raw).hexdigest()
            or image.get("digest") != oci_details["config"]["digest"]  # type: ignore[index]
            or image.get("reference") != oci_details["tag"]
            or not isinstance(locked_oci, dict)
            or locked_oci.get("digest") != oci_details["index"]["digest"]  # type: ignore[index]
            or locked_oci.get("reference") != f"{expected_repository}@{oci_details['index']['digest']}"  # type: ignore[index]
            or locked_oci.get("index") != oci_details["index"]
            or locked_oci.get("manifest") != oci_details["manifest"]
            or locked_oci.get("config") != oci_details["config"]
            or locked_oci.get("layers") != oci_details["layers"]
        ):
            raise PackageError(f"{artifact_candidate_id} candidate image lock mismatch or unresolved identity")
        archive_locks = lock.get("archives")
        receipt_lock = lock.get("receipt")
        load_probe = lock.get("docker_load_probe")
        if (
            not isinstance(archive_locks, dict)
            or archive_locks != {
                "docker": {"filename": docker_archive.name, "format": "docker-image-save-tar", "sha256": docker_details["archive_sha256"], "size": docker_details["archive_size"]},
                "oci": {"filename": oci_archive.name, "format": "oci-image-layout-tar", "sha256": oci_details["archive_sha256"], "size": oci_details["archive_size"]},
            }
            or lock.get("docker") != {
                "config": docker_details["config"], "layers": docker_details["layers"],
                "rootfs_diff_ids": docker_details["rootfs_diff_ids"],
            }
            or not isinstance(load_probe, dict)
            or load_probe.get("argv") != ["docker", "image", "load", "--input", docker_archive.name]
            or load_probe.get("network_required") is not False
            or load_probe.get("expected_runtime_identity") != oci_details["runtime"]
            or not isinstance(receipt_lock, dict)
            or receipt_lock.get("filename") != receipt_path.name
            or receipt_lock.get("sha256") != hashlib.sha256(receipt_raw).hexdigest()
            or receipt_lock.get("size") != len(receipt_raw)
            or build_receipt.get("candidate_id") != artifact_candidate_id
            or not isinstance(build_receipt.get("format"), str)
            or not str(build_receipt["format"]).startswith("ai-server-")
            or not str(build_receipt["format"]).endswith("-oci-build-receipt-v1")
            or build_receipt.get("output") != locked_oci
        ):
            raise PackageError(f"{artifact_candidate_id} archive or build receipt lock mismatch")
        common = {
            "candidate_digest": image["digest"], "candidate_digest_kind": "docker-image-config",
            "candidate_id": artifact_candidate_id, "lock_path": policy["lock_path"],
            "lock_sha256": hashlib.sha256(lock_raw).hexdigest(), "receipt_path": policy["receipt_path"],
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        }
        for transport, archive, details in (
            ("docker", docker_archive, docker_details), ("oci", oci_archive, oci_details),
        ):
            package_path = f"artifacts/images/{stem}.{transport}.tar"
            identity = (
                {key: details[key] for key in ("config", "layers", "reference", "rootfs_diff_ids")}
                if transport == "docker"
                else {key: details[key] for key in ("config", "index", "layers", "manifest", "runtime")}
            )
            records.append({
                **common, "identity": identity, "path": package_path,
                "sha256": details["archive_sha256"], "size": details["archive_size"], "transport": transport,
            })
            sources[package_path] = archive
    return records, sources


def _validate_artifact_source_bindings(
    entries: Iterable[Mapping[str, object]], artifacts: Iterable[Mapping[str, object]],
) -> None:
    by_path = {str(entry["path"]): entry for entry in entries}
    for artifact in artifacts:
        lock_entry = by_path.get(str(artifact["lock_path"]))
        receipt_entry = by_path.get(str(artifact["receipt_path"]))
        payload_entry = by_path.get(str(artifact["path"]))
        if (
            lock_entry is None or lock_entry.get("sha256") != artifact.get("lock_sha256")
            or receipt_entry is None or receipt_entry.get("sha256") != artifact.get("receipt_sha256")
            or payload_entry is None
            or (payload_entry.get("sha256"), payload_entry.get("size")) != (artifact.get("sha256"), artifact.get("size"))
        ):
            raise PackageError("package artifact source binding mismatch")


def build_manifest(
    repo_root: Path,
    image_archives: Mapping[str, Path] | None = None,
    *,
    candidate_id: str | None = None,
) -> tuple[dict[str, object], dict[str, Path]]:
    root = _real_root(repo_root)
    paths = _policy_paths(candidate_id)
    _candidate_inventory(root, candidate_id)
    entries: list[dict[str, object]] = []
    sources: dict[str, Path] = {}
    for relative in paths:
        path, metadata = _source_file(root, relative)
        digest = custody._sha256_file(path)
        entries.append({"path": relative, "sha256": digest, "size": metadata.st_size})
        sources[relative] = path
    artifacts, artifact_sources = _artifact_records(root, image_archives, candidate_id)
    for artifact in artifacts:
        entries.append({"path": artifact["path"], "sha256": artifact["sha256"], "size": artifact["size"]})
    entries.sort(key=lambda item: str(item["path"]))
    sources.update(artifact_sources)
    _validate_artifact_source_bindings(entries, artifacts)
    manifest: dict[str, object] = {
        "artifacts": artifacts,
        "entries": entries,
        "inventory_policy": POLICY_VERSION,
        "schema_version": MANIFEST_SCHEMA,
    }
    if candidate_id is not None:
        manifest.update({"candidate_id": candidate_id, "package_scope": "candidate"})
    return manifest, sources


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PackageError("short write")
        view = view[written:]


def _exclusive_descriptor(path: Path, mode: int) -> int:
    if not path.is_absolute():
        raise PackageError("output path must be absolute")
    try:
        parent = custody._real_directory(path.parent, "output directory")
    except custody.CustodyError as exc:
        raise PackageError(str(exc)) from exc
    if path.exists() or path.is_symlink():
        raise PackageError(f"output already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent / path.name, flags, mode)
    except FileExistsError as exc:
        raise PackageError(f"output already exists: {path}") from exc
    os.fchmod(descriptor, mode)
    return descriptor


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


class _IdentityReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        self.size += len(chunk)
        return chunk


def _archive_source(archive: tarfile.TarFile, member_name: str, source: Path, entry: Mapping[str, object]) -> None:
    with source.open("rb", buffering=0) as payload:
        before = os.fstat(payload.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != entry["size"]:
            raise PackageError(f"source changed before streaming: {entry['path']}")
        reader = _IdentityReader(payload)
        archive.addfile(_tar_info(member_name, int(entry["size"])), reader)
        after = os.fstat(payload.fileno())
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if (
            identity_before != identity_after
            or reader.size != entry["size"]
            or reader.digest.hexdigest() != entry["sha256"]
        ):
            raise PackageError(f"source changed during streaming: {entry['path']}")


def _write_archive(path: Path, manifest: Mapping[str, object], sources: Mapping[str, Path]) -> None:
    descriptor = _exclusive_descriptor(path, 0o400)
    stream = os.fdopen(descriptor, "wb", buffering=0)
    try:
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.GNU_FORMAT) as archive:
            manifest_payload = _canonical(dict(manifest)) + b"\n"
            archive.addfile(_tar_info(MANIFEST_MEMBER, len(manifest_payload)), io.BytesIO(manifest_payload))
            for entry in manifest["entries"]:  # type: ignore[index]
                assert isinstance(entry, dict)
                relative = str(entry["path"])
                source = sources[relative]
                _archive_source(archive, f"{PAYLOAD_PREFIX}/{relative}", source, entry)
        stream.flush()
        os.fsync(stream.fileno())
    finally:
        stream.close()
    custody._fsync_directory(path.parent)


def _write_receipt(path: Path, content: Mapping[str, object]) -> None:
    canonical_content = _canonical(dict(content))
    envelope = {
        "canonical_content_sha256": hashlib.sha256(canonical_content).hexdigest(),
        "content": dict(content),
        "schema_version": custody.RECEIPT_SCHEMA,
    }
    descriptor = _exclusive_descriptor(path, 0o400)
    try:
        _write_all(descriptor, _canonical(envelope) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    custody._fsync_directory(path.parent)


def _embedded_scope(
    value: Mapping[str, object], base_keys: set[str], label: str,
) -> str | None:
    keys = set(value)
    if keys == base_keys:
        return None
    if keys != base_keys | {"candidate_id", "package_scope"}:
        raise PackageError(f"{label} is invalid")
    candidate_id = value.get("candidate_id")
    if value.get("package_scope") != "candidate" or candidate_id not in CANDIDATE_SCOPE_IDS:
        raise PackageError(f"{label} scope is invalid")
    return str(candidate_id)


def seal(
    repo_root: Path,
    output_dir: Path,
    image_archives: Mapping[str, Path] | None = None,
    *,
    candidate_id: str | None = None,
) -> tuple[Path, Path]:
    root = _real_root(repo_root)
    output = _real_root(output_dir)
    if output == root or output.is_relative_to(root):
        raise PackageError("output directory must be outside the repository root")
    archive_path = output / ARCHIVE_NAME
    receipt_path = output / RECEIPT_NAME
    manifest, sources = build_manifest(root, image_archives, candidate_id=candidate_id)
    _write_archive(archive_path, manifest, sources)
    archive_metadata = archive_path.stat()
    manifest_payload = _canonical(manifest) + b"\n"
    content = {
        "archive_sha256": custody._sha256_file(archive_path),
        "archive_size": archive_metadata.st_size,
        "content_recorded": False,
        "artifact_count": len(manifest["artifacts"]),
        "artifacts": manifest["artifacts"],
        "file_count": len(manifest["entries"]),
        "inventory_policy": POLICY_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "schema_version": RECEIPT_SCHEMA,
    }
    if candidate_id is not None:
        content.update({"candidate_id": candidate_id, "package_scope": "candidate"})
    try:
        _write_receipt(receipt_path, content)
    except BaseException:
        # The package is intentionally immutable.  Leave it for explicit operator cleanup.
        raise
    return archive_path, receipt_path


def _load_receipt(path: Path) -> dict[str, object]:
    try:
        metadata = custody._regular_file(path, "verification receipt")
        raw = path.read_bytes()
        value = json.loads(raw)
    except (custody.CustodyError, OSError, ValueError) as exc:
        raise PackageError("verification receipt is invalid") from exc
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o400:
        raise PackageError("verification receipt mode is not 0400")
    if not isinstance(value, dict) or set(value) != {"canonical_content_sha256", "content", "schema_version"}:
        raise PackageError("verification receipt envelope is invalid")
    content = value.get("content")
    if value.get("schema_version") != custody.RECEIPT_SCHEMA or not isinstance(content, dict):
        raise PackageError("verification receipt envelope is invalid")
    if value.get("canonical_content_sha256") != hashlib.sha256(_canonical(content)).hexdigest():
        raise PackageError("verification receipt canonical hash mismatch")
    if raw != _canonical(value) + b"\n":
        raise PackageError("verification receipt is not canonical")
    expected_keys = {
        "archive_sha256", "archive_size", "artifact_count", "artifacts", "content_recorded", "file_count",
        "inventory_policy", "manifest_sha256", "schema_version",
    }
    candidate_id = _embedded_scope(content, expected_keys, "verification receipt content")
    if content.get("schema_version") != RECEIPT_SCHEMA:
        raise PackageError("verification receipt content is invalid")
    if content.get("content_recorded") is not False or content.get("inventory_policy") != POLICY_VERSION:
        raise PackageError("verification receipt policy mismatch")
    try:
        custody._validate_sha256(content["archive_sha256"])
        custody._validate_sha256(content["manifest_sha256"])
    except custody.CustodyError as exc:
        raise PackageError("verification receipt digest is invalid") from exc
    if type(content.get("archive_size")) is not int or int(content["archive_size"]) < 0:
        raise PackageError("verification receipt archive size is invalid")
    expected_artifact_count = len(_scope_archive_keys(candidate_id))
    expected_file_count = len(_policy_paths(candidate_id)) + expected_artifact_count
    if type(content.get("artifact_count")) is not int or content["artifact_count"] != expected_artifact_count:
        raise PackageError("verification receipt artifact count is invalid")
    if type(content.get("file_count")) is not int or content["file_count"] != expected_file_count:
        raise PackageError("verification receipt file count is invalid")
    _artifact_manifest_records(content.get("artifacts"), candidate_id)
    return content


def _load_manifest(archive: tarfile.TarFile, members: list[tarfile.TarInfo]) -> dict[str, object]:
    if not members or members[0].name != MANIFEST_MEMBER or not members[0].isfile():
        raise PackageError("package manifest member is missing")
    stream = archive.extractfile(members[0])
    if stream is None or members[0].size > 1024 * 1024:
        raise PackageError("package manifest is invalid")
    try:
        raw = stream.read()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise PackageError("package manifest is invalid") from exc
    base_keys = {"artifacts", "entries", "inventory_policy", "schema_version"}
    if not isinstance(value, dict):
        raise PackageError("package manifest is invalid")
    _embedded_scope(value, base_keys, "package manifest")
    if value.get("schema_version") != MANIFEST_SCHEMA or value.get("inventory_policy") != POLICY_VERSION:
        raise PackageError("package manifest policy mismatch")
    if _canonical(value) + b"\n" != raw:
        raise PackageError("package manifest is not canonical")
    return value


def _hash_member(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(custody.CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _manifest_entries(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    candidate_id = _embedded_scope(
        manifest,
        {"artifacts", "entries", "inventory_policy", "schema_version"},
        "package manifest",
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise PackageError("package manifest entries are invalid")
    expected_paths: list[str] = []
    checked: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise PackageError("package manifest entry is invalid")
        relative = _safe_path(str(entry["path"]), "manifest path")
        digest = entry["sha256"]
        size = entry["size"]
        try:
            custody._validate_sha256(digest)
        except custody.CustodyError as exc:
            raise PackageError(str(exc)) from exc
        if type(size) is not int or size < 0:
            raise PackageError("package manifest size is invalid")
        expected_paths.append(relative)
        checked.append({"path": relative, "sha256": digest, "size": size})
    expected_inventory = sorted((
        *_policy_paths(candidate_id),
        *(
            f"artifacts/images/{policy['stem']}.{transport}.tar"
            for policy in (
                EXTERNAL_IMAGE_ARTIFACTS[artifact_candidate_id]
                for artifact_candidate_id in _scope_artifact_candidate_ids(candidate_id)
            )
            for transport in ("docker", "oci")
        ),
    ))
    if expected_paths != expected_inventory:
        raise PackageError("package inventory does not match the built-in policy")
    return checked


def _artifact_manifest_records(
    value: object, candidate_id: str | None = None,
) -> list[dict[str, object]]:
    expected_ids = _scope_archive_keys(candidate_id)
    if not isinstance(value, list) or len(value) != len(expected_ids):
        raise PackageError("package artifact records are invalid")
    expected_keys = {
        "candidate_digest", "candidate_digest_kind", "candidate_id", "identity", "lock_path", "lock_sha256",
        "path", "receipt_path", "receipt_sha256", "sha256", "size", "transport",
    }
    checked: list[dict[str, object]] = []
    ids: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise PackageError("package artifact record is invalid")
        artifact_candidate_id = str(item.get("candidate_id"))
        policy = EXTERNAL_IMAGE_ARTIFACTS.get(artifact_candidate_id)
        if policy is None:
            raise PackageError("package artifact candidate is not allowlisted")
        transport = item.get("transport")
        if (
            transport not in {"docker", "oci"}
            or item.get("path") != f"artifacts/images/{policy['stem']}.{transport}.tar"
            or item.get("lock_path") != policy["lock_path"]
            or item.get("receipt_path") != policy["receipt_path"]
            or type(item.get("size")) is not int or int(item["size"]) <= 0
        ):
            raise PackageError("package artifact path or size is invalid")
        for key in ("sha256", "lock_sha256", "receipt_sha256"):
            try:
                custody._validate_sha256(item[key])
            except (custody.CustodyError, KeyError) as exc:
                raise PackageError("package artifact digest is invalid") from exc
        digest = item.get("candidate_digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise PackageError("package artifact candidate digest is invalid")
        try:
            custody._validate_sha256(digest[7:])
        except custody.CustodyError as exc:
            raise PackageError("package artifact candidate digest is invalid") from exc
        identity = item.get("identity")
        if item.get("candidate_digest_kind") != "docker-image-config" or not isinstance(identity, dict):
            raise PackageError("package artifact transport identity is invalid")
        if transport == "oci":
            if set(identity) != {"config", "index", "layers", "manifest", "runtime"}:
                raise PackageError("package OCI artifact identity is invalid")
            config = _sha_descriptor(identity.get("config"), "artifact config")
            _sha_descriptor(identity.get("manifest"), "artifact manifest")
            _sha_descriptor(identity.get("index"), "artifact index")
            layers_value = identity.get("layers")
            runtime = identity.get("runtime")
            if not isinstance(layers_value, list) or not layers_value or not isinstance(runtime, dict):
                raise PackageError("package OCI artifact layers or runtime are invalid")
            for layer in layers_value:
                _sha_descriptor(layer, "artifact layer")
            if (
                set(runtime) != {"architecture", "os", "config_digest", "manifest_digest", "rootfs_diff_ids"}
                or runtime.get("architecture") != "amd64" or runtime.get("os") != "linux"
                or digest != config["digest"] or runtime.get("config_digest") != digest
                or not isinstance(runtime.get("rootfs_diff_ids"), list)
            ):
                raise PackageError("package artifact candidate digest does not match OCI config")
        else:
            if set(identity) != {"config", "layers", "reference", "rootfs_diff_ids"}:
                raise PackageError("package Docker artifact identity is invalid")
            config = identity.get("config")
            layers = identity.get("layers")
            diff_ids = identity.get("rootfs_diff_ids")
            valid_layers = (
                isinstance(layers, list) and bool(layers)
                and all(
                    isinstance(layer, dict) and set(layer) == {"path", "digest", "size"}
                    and isinstance(layer.get("path"), str) and str(layer["path"]).startswith("blobs/sha256/")
                    and isinstance(layer.get("digest"), str) and str(layer["digest"]).startswith("sha256:")
                    and type(layer.get("size")) is int and layer["size"] > 0
                    for layer in layers
                )
            )
            if (
                not isinstance(config, dict) or set(config) != {"digest", "size"}
                or config.get("digest") != digest or type(config.get("size")) is not int or config["size"] <= 0
                or not valid_layers or not isinstance(diff_ids, list)
                or [layer.get("digest") for layer in layers if isinstance(layer, dict)] != diff_ids
            ):
                raise PackageError("package Docker artifact identity is invalid")
        checked.append(dict(item))
        ids.append(f"{artifact_candidate_id}.{transport}")
    if ids != list(expected_ids):
        raise PackageError("package artifact inventory is not canonical")
    return checked


def _archive_inventory(package_path: Path, *, verify_payloads: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        with tarfile.open(package_path, mode="r:") as archive:
            members = archive.getmembers()
            names: set[str] = set()
            for member in members:
                name = _safe_path(member.name, "archive member")
                if name in names or not member.isfile():
                    raise PackageError("archive contains duplicate or non-file members")
                names.add(name)
            manifest = _load_manifest(archive, members)
            candidate_id = _embedded_scope(
                manifest,
                {"artifacts", "entries", "inventory_policy", "schema_version"},
                "package manifest",
            )
            entries = _manifest_entries(manifest)
            artifacts = _artifact_manifest_records(manifest.get("artifacts"), candidate_id)
            expected_members = {MANIFEST_MEMBER, *(f"{PAYLOAD_PREFIX}/{entry['path']}" for entry in entries)}
            if names != expected_members:
                raise PackageError("archive contains unexpected or missing members")
            if verify_payloads:
                for entry in entries:
                    relative = str(entry["path"])
                    member = archive.getmember(f"{PAYLOAD_PREFIX}/{relative}")
                    stream = archive.extractfile(member)
                    if stream is None or _hash_member(stream) != (entry["sha256"], entry["size"]):
                        raise PackageError(f"package payload identity mismatch: {relative}")
                inspected_oci: dict[str, dict[str, object]] = {}
                for artifact in (item for item in artifacts if item["transport"] == "oci"):
                    member = archive.getmember(f"{PAYLOAD_PREFIX}/{artifact['path']}")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise PackageError(f"package artifact is missing: {artifact['candidate_id']}")
                    inspected = _inspect_oci_archive(stream)
                    if {key: inspected[key] for key in ("config", "index", "layers", "manifest", "runtime")} != artifact["identity"]:
                            raise PackageError(f"package artifact OCI identity mismatch: {artifact['candidate_id']}")
                    inspected_oci[str(artifact["candidate_id"])] = inspected
                for artifact in (item for item in artifacts if item["transport"] == "docker"):
                    member = archive.getmember(f"{PAYLOAD_PREFIX}/{artifact['path']}")
                    stream = archive.extractfile(member)
                    oci = inspected_oci.get(str(artifact["candidate_id"]))
                    if stream is None or oci is None:
                        raise PackageError(f"package artifact is missing: {artifact['candidate_id']}")
                    inspected = _inspect_docker_archive(
                        stream, str(artifact["identity"]["reference"]), oci["runtime"],  # type: ignore[index]
                    )
                    if {key: inspected[key] for key in ("config", "layers", "reference", "rootfs_diff_ids")} != artifact["identity"]:
                        raise PackageError(f"package artifact Docker identity mismatch: {artifact['candidate_id']}")
            _validate_artifact_source_bindings(entries, artifacts)
            return manifest, entries
    except (tarfile.TarError, KeyError, OSError) as exc:
        raise PackageError("candidate package archive is invalid") from exc


def verify(
    package_path: Path,
    receipt_path: Path,
    *,
    candidate_id: str | None = None,
) -> dict[str, object]:
    try:
        package_metadata = custody._regular_file(package_path, "candidate package")
    except custody.CustodyError as exc:
        raise PackageError(str(exc)) from exc
    receipt = _load_receipt(receipt_path)
    if receipt.get("archive_size") != package_metadata.st_size or receipt.get("archive_sha256") != custody._sha256_file(package_path):
        raise PackageError("candidate package identity mismatch")
    manifest, _entries = _archive_inventory(package_path, verify_payloads=True)
    manifest_scope = _embedded_scope(
        manifest,
        {"artifacts", "entries", "inventory_policy", "schema_version"},
        "package manifest",
    )
    receipt_scope = _embedded_scope(
        receipt,
        {
            "archive_sha256", "archive_size", "artifact_count", "artifacts",
            "content_recorded", "file_count", "inventory_policy", "manifest_sha256",
            "schema_version",
        },
        "verification receipt content",
    )
    if manifest_scope != receipt_scope or (candidate_id is not None and manifest_scope != candidate_id):
        raise PackageError("candidate package scope mismatch")
    if receipt.get("archive_size") != package_path.stat().st_size or receipt.get("archive_sha256") != custody._sha256_file(package_path):
        raise PackageError("candidate package changed during verification")
    manifest_hash = hashlib.sha256(_canonical(manifest) + b"\n").hexdigest()
    if (
        receipt.get("manifest_sha256") != manifest_hash
        or receipt.get("file_count") != len(manifest["entries"])
        or receipt.get("artifacts") != manifest.get("artifacts")
    ):
        raise PackageError("verification receipt manifest binding mismatch")
    result = {
        "archive_sha256": receipt["archive_sha256"],
        "artifact_count": len(manifest["artifacts"]),
        "file_count": len(manifest["entries"]),
        "inventory_policy": POLICY_VERSION,
        "status": "verified",
    }
    if manifest_scope is not None:
        result.update({"candidate_id": manifest_scope, "package_scope": "candidate"})
    return result


def _new_materialization_root(destination: Path) -> Path:
    if not destination.is_absolute():
        raise PackageError("materialization destination must be absolute")
    try:
        parent = custody._real_directory(destination.parent, "materialization parent")
    except custody.CustodyError as exc:
        raise PackageError(str(exc)) from exc
    target = parent / destination.name
    if target.exists() or target.is_symlink():
        raise PackageError("materialization destination must be new")
    try:
        os.mkdir(target, 0o700)
    except FileExistsError as exc:
        raise PackageError("materialization destination must be new") from exc
    custody._fsync_directory(parent)
    return target


def _materialization_parent(root: Path, relative: str) -> Path:
    current = root
    for component in PurePosixPath(relative).parts[:-1]:
        child = current / component
        if not child.exists() and not child.is_symlink():
            try:
                os.mkdir(child, 0o700)
            except FileExistsError as exc:
                raise PackageError(f"materialization path appeared: {relative}") from exc
            custody._fsync_directory(current)
        try:
            metadata = child.lstat()
        except OSError as exc:
            raise PackageError(f"materialization directory is invalid: {relative}") from exc
        if not stat.S_ISDIR(metadata.st_mode) or child.is_symlink():
            raise PackageError(f"materialization directory is invalid: {relative}")
        current = child
    return current


def _materialize_file(root: Path, relative: str, source: BinaryIO, expected_sha256: str, expected_size: int) -> None:
    parent = _materialization_parent(root, relative)
    destination = root.joinpath(*PurePosixPath(relative).parts)
    if destination.exists() or destination.is_symlink():
        raise PackageError(f"materialization path already exists: {relative}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o400)
    except FileExistsError as exc:
        raise PackageError(f"materialization path already exists: {relative}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := source.read(custody.CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
            _write_all(descriptor, chunk)
        if digest.hexdigest() != expected_sha256 or size != expected_size:
            raise PackageError(f"materialized payload identity mismatch: {relative}")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    custody._fsync_directory(parent)


def _freeze_directories(root: Path) -> None:
    directories: list[Path] = []
    for current, names, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        if files or names or current_path == root:
            directories.append(current_path)
    for directory in directories:
        os.chmod(directory, 0o555)
        custody._fsync_directory(directory)
    custody._fsync_directory(root.parent)


def materialize(
    package_path: Path,
    receipt_path: Path,
    destination: Path,
    *,
    candidate_id: str | None = None,
) -> dict[str, object]:
    """Materialize the reviewed payload once; failures leave an explicit partial tree."""

    verified = verify(package_path, receipt_path, candidate_id=candidate_id)
    root = _new_materialization_root(destination)
    try:
        with tarfile.open(package_path, mode="r:") as archive:
            members = archive.getmembers()
            names: set[str] = set()
            for member in members:
                name = _safe_path(member.name, "archive member")
                if name in names or not member.isfile():
                    raise PackageError("archive contains duplicate or non-file members")
                names.add(name)
            manifest = _load_manifest(archive, members)
            entries = _manifest_entries(manifest)
            embedded_candidate = _embedded_scope(
                manifest,
                {"artifacts", "entries", "inventory_policy", "schema_version"},
                "package manifest",
            )
            _artifact_manifest_records(manifest.get("artifacts"), embedded_candidate)
            expected_members = {MANIFEST_MEMBER, *(f"{PAYLOAD_PREFIX}/{entry['path']}" for entry in entries)}
            if names != expected_members:
                raise PackageError("archive contains unexpected or missing members")
            for entry in entries:
                relative = str(entry["path"])
                stream = archive.extractfile(archive.getmember(f"{PAYLOAD_PREFIX}/{relative}"))
                if stream is None:
                    raise PackageError(f"package payload is missing: {relative}")
                _materialize_file(root, relative, stream, str(entry["sha256"]), int(entry["size"]))
    except (tarfile.TarError, KeyError, OSError) as exc:
        raise PackageError("candidate package archive is invalid") from exc
    _freeze_directories(root)
    if custody._sha256_file(package_path) != verified["archive_sha256"]:
        raise PackageError("candidate package changed during materialization")
    result = verify_materialized_tree(
        package_path, receipt_path, root, candidate_id=candidate_id,
    )
    return {**result, "destination": str(root), "status": "materialized"}


def verify_materialized_tree(
    package_path: Path,
    receipt_path: Path,
    root: Path,
    *,
    candidate_id: str | None = None,
) -> dict[str, object]:
    """Verify an exact, frozen materialized tree against the reviewed archive."""

    verified = verify(package_path, receipt_path, candidate_id=candidate_id)
    try:
        materialized = custody._real_directory(root, "materialized root")
    except custody.CustodyError as exc:
        raise PackageError(str(exc)) from exc
    _manifest, entries = _archive_inventory(package_path, verify_payloads=False)
    expected_files = {str(entry["path"]): entry for entry in entries}
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    found_files: set[str] = set()
    found_directories = {"."}
    for current, directories, files in os.walk(materialized, topdown=True, followlinks=False):
        current_path = Path(current)
        if os.name == "posix" and stat.S_IMODE(current_path.lstat().st_mode) != 0o555:
            raise PackageError(f"materialized directory mode mismatch: {current_path.relative_to(materialized).as_posix() or '.'}")
        for name in directories:
            child = current_path / name
            metadata = child.lstat()
            relative = child.relative_to(materialized).as_posix()
            if child.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise PackageError(f"materialized tree contains symlink or special entry: {relative}")
            found_directories.add(relative)
        for name in files:
            child = current_path / name
            relative = child.relative_to(materialized).as_posix()
            try:
                metadata = custody._regular_file(child, relative)
            except custody.CustodyError as exc:
                raise PackageError(str(exc)) from exc
            if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o444:
                raise PackageError(f"materialized file mode mismatch: {relative}")
            found_files.add(relative)
            expected = expected_files.get(relative)
            if expected is None or metadata.st_size != expected["size"] or custody._sha256_file(child) != expected["sha256"]:
                raise PackageError(f"materialized file identity mismatch: {relative}")
    if found_files != set(expected_files) or found_directories != expected_directories:
        raise PackageError("materialized tree contains unexpected or missing entries")
    result = {
        "archive_sha256": verified["archive_sha256"],
        "file_count": len(expected_files),
        "inventory_policy": POLICY_VERSION,
        "root": str(materialized),
        "status": "verified",
    }
    if verified.get("package_scope") == "candidate":
        result.update({
            "candidate_id": verified["candidate_id"],
            "package_scope": "candidate",
        })
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--repo-root", type=Path, required=True)
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    seal_parser.add_argument("--candidate-id", choices=CANDIDATE_SCOPE_IDS)
    seal_parser.add_argument(
        "--image-archive", action="append", default=[], metavar="CANDIDATE=ABSOLUTE_PATH",
        help="exact external transport; required as CANDIDATE.docker and CANDIDATE.oci",
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--package", type=Path, required=True)
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--candidate-id", choices=CANDIDATE_SCOPE_IDS)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--package", type=Path, required=True)
    materialize_parser.add_argument("--receipt", type=Path, required=True)
    materialize_parser.add_argument("--destination", type=Path, required=True)
    materialize_parser.add_argument("--candidate-id", choices=CANDIDATE_SCOPE_IDS)
    return parser


def _parse_image_archives(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        candidate_id, separator, raw_path = value.partition("=")
        if not separator or candidate_id not in EXTERNAL_ARCHIVE_KEYS or candidate_id in result or not raw_path:
            raise PackageError("image archive argument is invalid or duplicated")
        result[candidate_id] = Path(raw_path)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            package, receipt = seal(
                args.repo_root,
                args.output_dir,
                _parse_image_archives(args.image_archive),
                candidate_id=args.candidate_id,
            )
            result = {"package": str(package), "receipt": str(receipt), "status": "sealed"}
        elif args.command == "verify":
            result = verify(args.package, args.receipt, candidate_id=args.candidate_id)
            if result.get("package_scope") == "candidate" and args.candidate_id is None:
                raise PackageError("candidate package scope must be explicit")
        else:
            checked = verify(args.package, args.receipt, candidate_id=args.candidate_id)
            if checked.get("package_scope") == "candidate" and args.candidate_id is None:
                raise PackageError("candidate package scope must be explicit")
            result = materialize(
                args.package,
                args.receipt,
                args.destination,
                candidate_id=args.candidate_id,
            )
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
