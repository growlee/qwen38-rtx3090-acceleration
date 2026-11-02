#!/usr/bin/env python3
"""Materialize one sealed acceleration candidate below its exact staging root.

This module deliberately does not build or pull OCI images.  Custom image
toolchains are already sealed into the package OCI archive; live preparation
accepts the verified image identity and downloads only candidate model data.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Any, BinaryIO, Callable, ContextManager, Iterable, Mapping
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

try:
    from .candidate_artifact_custody import (
        CustodyError,
        CustodyLockGuard,
        ExternalArtifact,
        download_artifact,
        locate_reusable_cas_blob,
    )
    from .candidate_contract import ContractError, huggingface_resolve_url, validate_candidate_contract
except ImportError:  # pragma: no cover - direct execution from experiments/
    from candidate_artifact_custody import (  # type: ignore[no-redef]
        CustodyError,
        CustodyLockGuard,
        ExternalArtifact,
        download_artifact,
        locate_reusable_cas_blob,
    )
    from candidate_contract import ContractError, huggingface_resolve_url, validate_candidate_contract  # type: ignore[no-redef]


SCHEMA = "ai-server-candidate-prepare-manifest/v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PIN = re.compile(r"^[0-9a-f]{40}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,254}$")
ALLOWED_CANDIDATES = {
    "qwen38-vllm": "qwen38_vllm",
    "freetoken": "freetoken",
    "llama-dflash2-official": "llama_dflash2_official",
}
ALLOWED_HTTPS_HOSTS = {"github.com", "huggingface.co", "files.pythonhosted.org"}
ALLOWED_REDIRECT_HOSTS = ALLOWED_HTTPS_HOSTS | {
    "codeload.github.com",
    "release-assets.githubusercontent.com",
    "cdn-lfs.hf.co",
    "cas-bridge.xethub.hf.co",
}


class PrepareError(RuntimeError):
    """The sealed candidate could not be materialized exactly once."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _real_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise PrepareError(f"{label}_invalid")
    if path.resolve(strict=True) != path.absolute():
        raise PrepareError(f"{label}_invalid")
    return path


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or SAFE_COMPONENT.fullmatch(value) is None or value in {".", ".."}:
        raise PrepareError(f"{label}_invalid")
    return value


def _canonical_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or any(ord(character) < 32 for character in value):
        raise PrepareError(f"{label}_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise PrepareError(f"{label}_invalid")
    return path


def _mkdir_exact(parent: Path, name: str, mode: int) -> Path:
    _real_directory(parent, "staging_parent")
    component = _canonical_relative(name, "directory_name")
    if len(component.parts) != 1:
        raise PrepareError("directory_name_invalid")
    child = parent / component.as_posix()
    try:
        os.mkdir(child, mode)
    except FileExistsError:
        return _real_directory(child, "staging_directory")
    os.chmod(child, mode)
    if os.name == "posix":
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return _real_directory(child, "staging_directory")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_candidate(candidate_id: str, pin: str, experiments_root: Path) -> tuple[dict[str, Any], str]:
    directory = ALLOWED_CANDIDATES.get(candidate_id)
    if directory is None or PIN.fullmatch(pin) is None:
        raise PrepareError("candidate_identity_not_allowlisted")
    manifest_path = experiments_root / directory / "candidate.json"
    try:
        if manifest_path.is_symlink():
            raise PrepareError("candidate_metadata_not_live_ready")
        before = manifest_path.read_bytes()
        unverified = json.loads(before)
        mode = next(iter(unverified["modes"]))
        validate_candidate_contract(candidate_id, mode, root=experiments_root, require_live=True)
        raw = manifest_path.read_bytes()
        if raw != before:
            raise PrepareError("candidate_metadata_changed_during_validation")
        value = json.loads(raw)
    except (ContractError, OSError, UnicodeError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        raise PrepareError("candidate_metadata_not_live_ready") from exc
    if value.get("candidate_id") != candidate_id or value.get("source", {}).get("revision") != pin:
        raise PrepareError("candidate_identity_mismatch")
    unresolved = value.get("build_time_required")
    if unresolved != []:
        raise PrepareError("candidate_build_unresolved")
    image = value.get("image")
    if not isinstance(image, dict) or not isinstance(image.get("reference"), str) or not isinstance(image.get("digest"), str):
        raise PrepareError("candidate_image_unsealed")
    return value, hashlib.sha256(raw).hexdigest()


def _checked_url(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in ALLOWED_HTTPS_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.port is not None
        or bool(parts.query)
        or bool(parts.fragment)
        or not parts.path.startswith("/")
        or "//" in parts.path
        or any(piece in {"", ".", ".."} for piece in PurePosixPath(unquote(parts.path)).parts[1:])
    ):
        raise PrepareError("artifact_url_not_allowlisted")
    return url


def _default_fetcher(url: str) -> ContextManager[BinaryIO]:
    request = Request(_checked_url(url), headers={"User-Agent": "ai-server-sealed-candidate-prepare/1"})
    response = urlopen(request, timeout=60)  # nosec B310 - exact HTTPS allowlist above
    final = urlsplit(response.geturl())
    if final.scheme != "https" or final.hostname not in ALLOWED_REDIRECT_HOSTS or final.username is not None or final.password is not None or final.port is not None:
        response.close()
        raise PrepareError("artifact_redirect_not_allowlisted")
    return closing(response)


def _source_artifact(data: Mapping[str, Any]) -> dict[str, object]:
    source = data["source"]
    repository = str(source["repository"])
    revision = str(source["revision"])
    name = repository.rstrip("/").rsplit("/", 1)[-1]
    path = f"sources/{name}-{revision}.tar.gz"
    if data["candidate_id"] in {"freetoken", "llama-dflash2-official"}:
        path = f"build-inputs/{name}-{revision}.tar.gz"
    return {
        "path": path,
        "url": f"{repository}/archive/{revision}.tar.gz",
        "size": source["archive_size"],
        "sha256": source["archive_sha256"],
    }


def _model_path(candidate_id: str, repository: str, filename: str) -> str:
    if candidate_id == "llama-dflash2-official":
        return f"models/{filename}"
    directory = repository.rsplit("/", 1)[-1]
    if candidate_id == "freetoken":
        directory = repository.replace("/", "-")
    return f"models/{directory}/{filename}"


def _load_model_manifest(candidate_id: str, model: Mapping[str, Any], experiments_root: Path) -> dict[str, Any]:
    role = _safe_component(model.get("role"), "model_role")
    path = experiments_root / "immutable_metadata" / "models" / f"{candidate_id}.{role}.json"
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrepareError("model_manifest_missing_or_invalid") from exc
    canonical_metadata = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if SHA256.fullmatch(str(model.get("manifest_sha256", ""))) is None or hashlib.sha256(canonical_metadata).hexdigest() != model["manifest_sha256"]:
        raise PrepareError("model_manifest_digest_mismatch")
    if not isinstance(value, dict) or value.get("kind") != "huggingface-model-manifest" or value.get("repository") != model.get("repository") or value.get("revision") != model.get("revision"):
        raise PrepareError("model_manifest_identity_mismatch")
    if value.get("overlay") != model.get("overlay"):
        raise PrepareError("model_overlay_contract_mismatch")
    blobs = value.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise PrepareError("model_manifest_blobs_invalid")
    return value


def _artifact_specs(data: Mapping[str, Any], experiments_root: Path) -> list[dict[str, object]]:
    candidate_id = str(data["candidate_id"])
    specs = [_source_artifact(data)] if candidate_id == "qwen38-vllm" else []
    models_by_role = {str(item.get("role")): item for item in data["models"]}
    if len(models_by_role) != len(data["models"]) or "target" not in models_by_role:
        raise PrepareError("model_role_set_invalid")
    target_repository = str(models_by_role["target"]["repository"])
    ordered_models = sorted(data["models"], key=lambda item: (item.get("role") == "fast-variant", item.get("role") != "target"))
    for model in ordered_models:
        sealed_model = _load_model_manifest(candidate_id, model, experiments_root)
        role = str(model["role"])
        repository = str(sealed_model["repository"])
        revision = str(sealed_model["revision"])
        overlay = sealed_model.get("overlay")
        replaces: set[str] = set()
        adds: set[str] = set()
        if role == "fast-variant":
            if not isinstance(overlay, dict) or set(overlay) != {"base_manifest_sha256", "replaces", "adds"}:
                raise PrepareError("model_overlay_contract_invalid")
            if overlay.get("base_manifest_sha256") != models_by_role["target"].get("manifest_sha256"):
                raise PrepareError("model_overlay_base_mismatch")
            replace_list = overlay.get("replaces")
            add_list = overlay.get("adds")
            if not isinstance(replace_list, list) or not isinstance(add_list, list) or replace_list != sorted(replace_list) or add_list != sorted(add_list):
                raise PrepareError("model_overlay_paths_invalid")
            replaces = set(replace_list)
            adds = set(add_list)
            blob_paths = {str(item.get("filename")) for item in sealed_model["blobs"]}
            if replaces & adds or replaces | adds != blob_paths or len(blob_paths) != len(sealed_model["blobs"]):
                raise PrepareError("model_overlay_paths_invalid")
        elif overlay is not None:
            raise PrepareError("unexpected_model_overlay")
        for blob in sealed_model["blobs"]:
            filename = str(blob["filename"])
            destination_repository = target_repository if role == "fast-variant" else repository
            spec = {
                "path": _model_path(candidate_id, destination_repository, filename),
                "url": huggingface_resolve_url(repository, revision, filename),
                "size": blob["size"],
                "sha256": blob["sha256"],
                "cas_filename": PurePosixPath(filename).name,
            }
            if role == "fast-variant":
                spec["download_path"] = f"overlay-payload/{role}/{filename}"
                spec["overlay_action"] = "replace" if filename in replaces else "add"
            specs.append(spec)

    seen: set[str] = set()
    for spec in specs:
        path = str(spec["path"])
        storage_path = str(spec.get("download_path", path))
        relative = _canonical_relative(path, "artifact_path")
        storage_relative = _canonical_relative(storage_path, "artifact_path")
        if (
            storage_path in seen
        ):
            raise PrepareError("artifact_path_or_duplicate_invalid")
        seen.add(storage_path)
        if not isinstance(spec["size"], int) or spec["size"] < 0 or SHA256.fullmatch(str(spec["sha256"])) is None:
            raise PrepareError("artifact_identity_invalid")
        if "url" in spec:
            _checked_url(str(spec["url"]))
    return specs


def _load_local_payload_inventory(
    inventory_path: Path,
    *,
    candidate_id: str,
    candidate_pin: str,
    metadata_sha256: str,
    specs: list[dict[str, object]],
) -> tuple[Path, dict[str, Path]]:
    """Validate the exact qwen38 payload inventory before staging any bytes."""

    if candidate_id != "qwen38-vllm":
        raise PrepareError("local_payload_candidate_not_allowed")
    if not inventory_path.is_absolute() or inventory_path.is_symlink():
        raise PrepareError("local_payload_inventory_invalid")
    parent = _real_directory(inventory_path.parent, "local_payload_root")
    try:
        descriptor = os.open(
            inventory_path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PrepareError("local_payload_inventory_invalid")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read()
        finally:
            os.close(descriptor)
        value = json.loads(raw)
    except PrepareError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrepareError("local_payload_inventory_invalid") from exc
    if not isinstance(value, dict) or set(value) != {
        "candidate_id", "candidate_pin", "candidate_metadata_sha256", "artifacts",
    }:
        raise PrepareError("local_payload_inventory_schema_invalid")
    try:
        canonical = json.dumps(
            value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PrepareError("local_payload_inventory_schema_invalid") from exc
    if raw != canonical:
        raise PrepareError("local_payload_inventory_not_canonical")
    if (
        value.get("candidate_id") != candidate_id
        or value.get("candidate_pin") != candidate_pin
        or value.get("candidate_metadata_sha256") != metadata_sha256
    ):
        raise PrepareError("local_payload_inventory_identity_mismatch")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise PrepareError("local_payload_inventory_schema_invalid")

    expected: list[tuple[str, str, object, int, str]] = []
    for spec in specs:
        expected.append((
            str(spec["path"]),
            str(spec.get("download_path", spec["path"])),
            spec.get("overlay_action"),
            int(spec["size"]),
            str(spec["sha256"]),
        ))
    observed: list[tuple[str, str, object, int, str]] = []
    blobs: dict[str, Path] = {}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {
            "logical_path", "download_path", "overlay_action", "blob_path", "size", "sha256",
        }:
            raise PrepareError("local_payload_inventory_schema_invalid")
        logical = _canonical_relative(item.get("logical_path"), "local_payload_logical_path").as_posix()
        download = _canonical_relative(item.get("download_path"), "local_payload_download_path").as_posix()
        action = item.get("overlay_action")
        if action not in {None, "replace", "add"}:
            raise PrepareError("local_payload_inventory_schema_invalid")
        size = item.get("size")
        digest = item.get("sha256")
        blob_value = item.get("blob_path")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or SHA256.fullmatch(digest) is None or not isinstance(blob_value, str):
            raise PrepareError("local_payload_inventory_schema_invalid")
        blob = Path(blob_value)
        if not blob.is_absolute() or blob.is_symlink():
            raise PrepareError("local_payload_blob_invalid")
        try:
            resolved = blob.resolve(strict=True)
        except OSError as exc:
            raise PrepareError("local_payload_blob_invalid") from exc
        if resolved == parent or not resolved.is_relative_to(parent):
            raise PrepareError("local_payload_blob_outside_root")
        if download in blobs:
            raise PrepareError("local_payload_inventory_duplicate")
        blobs[download] = resolved
        observed.append((logical, download, action, size, digest))
    if observed != expected:
        raise PrepareError("local_payload_inventory_spec_mismatch")
    return parent, blobs


def _stream_verified_local_payload(
    path: Path,
    *,
    payload_root: Path,
    expected_size: int,
    expected_sha256: str,
    consumer: Callable[[BinaryIO], object],
) -> None:
    """Hash and stream one unchanged regular payload through the custody writer."""

    if path.is_symlink():
        raise PrepareError("local_payload_blob_invalid")
    try:
        resolved = path.resolve(strict=True)
        if resolved == payload_root or not resolved.is_relative_to(payload_root):
            raise PrepareError("local_payload_blob_outside_root")
        before = resolved.stat(follow_symlinks=False)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != expected_size
                or (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino)
                or resolved.resolve(strict=True) != resolved
            ):
                raise PrepareError("local_payload_blob_identity_mismatch")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 8 * 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise PrepareError("local_payload_blob_identity_mismatch")
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                consumer(source)
        finally:
            os.close(descriptor)
    except PrepareError:
        raise
    except OSError as exc:
        raise PrepareError("local_payload_blob_invalid") from exc


def _ensure_parent(root: Path, relative: str) -> None:
    current = root
    for component in PurePosixPath(relative).parts[:-1]:
        current = _mkdir_exact(current, component, 0o700)


def _write_secret(path: Path, secret_factory: Callable[[], str]) -> None:
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or (os.name == "posix" and (stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid())):
            raise PrepareError("candidate_secret_invalid")
        return
    value = secret_factory()
    if not isinstance(value, str) or len(value) < 32 or any(character.isspace() for character in value):
        raise PrepareError("candidate_secret_factory_invalid")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (value + "\n").encode("ascii")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PrepareError("candidate_secret_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _runtime_identity(data: Mapping[str, object]) -> tuple[int, int]:
    identity = data.get("runtime_identity")
    if not isinstance(identity, dict) or set(identity) != {"uid", "gid"}:
        raise PrepareError("candidate_runtime_identity_invalid")
    uid = identity.get("uid")
    gid = identity.get("gid")
    if type(uid) is not int or type(gid) is not int or not 0 <= uid <= 2**31 - 1 or not 0 <= gid <= 2**31 - 1:
        raise PrepareError("candidate_runtime_identity_invalid")
    return uid, gid


def _set_access(path: Path, *, uid: int, gid: int, mode: int) -> None:
    if os.name == "posix":
        os.chown(path, uid, gid, follow_symlinks=False)
    os.chmod(path, mode, follow_symlinks=False)


def _apply_runtime_layout(root: Path, data: Mapping[str, object], secret_paths: set[str]) -> dict[str, tuple[int, int, int]]:
    runtime_uid, runtime_gid = _runtime_identity(data)
    models = _mkdir_exact(root, "models", 0o550)
    for directory, names, files in os.walk(models, topdown=True, followlinks=False):
        base = Path(directory)
        _real_directory(base, "candidate_model_directory")
        _set_access(base, uid=0, gid=runtime_gid, mode=0o550)
        for name in names:
            _real_directory(base / name, "candidate_model_directory")
        for name in files:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PrepareError("candidate_model_entry_invalid")
            _set_access(path, uid=0, gid=runtime_gid, mode=0o440)
    cache = _mkdir_exact(root, "cache", 0o700)
    _set_access(cache, uid=runtime_uid, gid=runtime_gid, mode=0o700)
    secret_mode = 0o600 if runtime_uid == 0 and runtime_gid == 0 else 0o440
    secret_policy: dict[str, tuple[int, int, int]] = {}
    for relative in sorted(secret_paths):
        path = root.joinpath(*PurePosixPath(relative).parts)
        _set_access(path, uid=0, gid=runtime_gid, mode=secret_mode)
        secret_policy[relative] = (secret_mode, 0, runtime_gid)
    return secret_policy


def _verify_runtime_layout(root: Path, runtime_identity: Mapping[str, object], secret_policy: Mapping[str, tuple[int, int, int]]) -> None:
    runtime_uid, runtime_gid = _runtime_identity({"runtime_identity": runtime_identity})
    models = _real_directory(root / "models", "candidate_models")
    for directory, names, files in os.walk(models, topdown=True, followlinks=False):
        base = Path(directory)
        info = _real_directory(base, "candidate_model_directory").stat()
        if os.name == "posix" and (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0, runtime_gid, 0o550):
            raise PrepareError("candidate_runtime_model_access_drift")
        for name in names:
            _real_directory(base / name, "candidate_model_directory")
        for name in files:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PrepareError("candidate_model_entry_invalid")
            if os.name == "posix" and (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0, runtime_gid, 0o440):
                raise PrepareError("candidate_runtime_model_access_drift")
    cache = _real_directory(root / "cache", "candidate_cache")
    cache_info = cache.stat()
    if os.name == "posix" and (cache_info.st_uid, cache_info.st_gid, stat.S_IMODE(cache_info.st_mode)) != (runtime_uid, runtime_gid, 0o700):
        raise PrepareError("candidate_runtime_cache_access_drift")
    for relative, (mode, uid, gid) in secret_policy.items():
        info = root.joinpath(*PurePosixPath(relative).parts).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PrepareError("candidate_secret_invalid")
        if os.name == "posix" and (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (uid, gid, mode):
            raise PrepareError("candidate_secret_invalid")


def _inventory(root: Path, *, secret_policy: Mapping[str, tuple[int, int, int]], manifest_path: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    local: list[dict[str, object]] = []
    secrets_metadata: list[dict[str, object]] = []
    directories: list[str] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in names:
            child = _real_directory(base / name, "candidate_subdirectory")
            directories.append(child.relative_to(root).as_posix())
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PrepareError("candidate_inventory_entry_invalid")
            if relative == manifest_path:
                continue
            if relative in secret_policy:
                mode, uid, gid = secret_policy[relative]
                if os.name == "posix" and (stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid) != (mode, uid, gid):
                    raise PrepareError("candidate_secret_invalid")
                secrets_metadata.append({"path": relative, "mode": mode, "owner_uid": uid, "owner_gid": gid})
            else:
                local.append({"path": relative, "size": info.st_size, "sha256": _sha256_file(path)})
    return (
        sorted(local, key=lambda item: str(item["path"])),
        sorted(secrets_metadata, key=lambda item: str(item["path"])),
        sorted(directories),
    )


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> str:
    payload = _canonical(manifest) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o400)
    try:
        os.fchmod(descriptor, 0o400)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PrepareError("candidate_manifest_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(payload).hexdigest()


def _verify_manifest(root: Path, manifest: Mapping[str, object], manifest_path: Path, production_cas_root: Path) -> str:
    expected_keys = {
        "build_input_manifest_sha256", "candidate_id", "candidate_pin", "candidate_root",
        "cas_origins", "directories", "image", "local_artifacts",
        "metadata_sha256", "runtime_identity", "schema_version", "secret_files",
    }
    if set(manifest) != expected_keys or manifest.get("schema_version") != SCHEMA or manifest.get("candidate_root") != str(root):
        raise PrepareError("candidate_prepare_manifest_invalid")
    candidate_id = manifest.get("candidate_id")
    expected_runtime_identity = {"uid": 0, "gid": 0} if candidate_id == "qwen38-vllm" else {"uid": 65532, "gid": 65532}
    runtime_identity = manifest.get("runtime_identity")
    if candidate_id not in ALLOWED_CANDIDATES or runtime_identity != expected_runtime_identity:
        raise PrepareError("candidate_prepare_manifest_invalid")
    runtime_uid, runtime_gid = _runtime_identity({"runtime_identity": runtime_identity})
    expected_secret_mode = 0o600 if runtime_uid == 0 and runtime_gid == 0 else 0o440
    secret_policy: dict[str, tuple[int, int, int]] = {}
    secret_items = manifest.get("secret_files")
    if not isinstance(secret_items, list):
        raise PrepareError("candidate_prepare_manifest_invalid")
    for item in secret_items:
        if not isinstance(item, dict) or set(item) != {"path", "mode", "owner_uid", "owner_gid"}:
            raise PrepareError("candidate_prepare_manifest_invalid")
        relative = _canonical_relative(item["path"], "candidate_secret_path").as_posix()
        values = (item["mode"], item["owner_uid"], item["owner_gid"])
        if any(type(value) is not int for value in values) or relative in secret_policy:
            raise PrepareError("candidate_prepare_manifest_invalid")
        if values != (expected_secret_mode, 0, runtime_gid):
            raise PrepareError("candidate_prepare_manifest_invalid")
        secret_policy[relative] = values
    _verify_runtime_layout(root, runtime_identity, secret_policy)
    local, secret_metadata, directories = _inventory(root, secret_policy=secret_policy, manifest_path=manifest_path.relative_to(root).as_posix())
    if local != manifest.get("local_artifacts") or secret_metadata != manifest.get("secret_files") or directories != manifest.get("directories"):
        raise PrepareError("candidate_prepare_inventory_drift")
    origins = manifest.get("cas_origins")
    if not isinstance(origins, list):
        raise PrepareError("candidate_prepare_manifest_invalid")
    origin_logical: set[str] = set()
    local_by_path = {str(item["path"]): item for item in local}
    for item in origins:
        if not isinstance(item, dict):
            raise PrepareError("candidate_prepare_manifest_invalid")
        try:
            source_path = Path(str(item["source_path"]))
            expected_source = production_cas_root / "sha256" / str(item["sha256"]) / source_path.name
            if source_path != expected_source:
                raise PrepareError("candidate_external_artifact_outside_production_cas")
            verified = locate_reusable_cas_blob(
                production_cas_root, digest=str(item["sha256"]),
                filename=source_path.name, expected_size=int(item["size"]), logical_path=str(item["logical_path"]),
            )
        except (KeyError, ValueError, CustodyError) as exc:
            raise PrepareError("candidate_external_artifact_drift") from exc
        if verified.as_dict() != item:
            raise PrepareError("candidate_external_artifact_drift")
        logical_path = str(item["logical_path"])
        staged = local_by_path.get(logical_path)
        if logical_path in origin_logical or staged is None:
            raise PrepareError("candidate_external_artifact_alias")
        if staged.get("sha256") != item.get("sha256") or staged.get("size") != item.get("size"):
            raise PrepareError("candidate_external_artifact_drift")
        origin_logical.add(logical_path)
    raw = manifest_path.read_bytes()
    if raw != _canonical(dict(manifest)) + b"\n" or (os.name == "posix" and stat.S_IMODE(manifest_path.lstat().st_mode) != 0o400):
        raise PrepareError("candidate_prepare_manifest_drift")
    return hashlib.sha256(raw).hexdigest()


def prepare_candidate_artifacts(
    candidate_id: str,
    candidate_pin: str,
    *,
    experiments_root: Path,
    staging_root: Path,
    production_cas_root: Path,
    lock_guard: CustodyLockGuard,
    expected_host_lock: Path,
    expected_candidate_lock: Path,
    fetcher: Callable[[str], ContextManager[BinaryIO]] = _default_fetcher,
    artifact_downloader: Callable[..., dict[str, object]] = download_artifact,
    sealed_image_identity: Mapping[str, object] | None = None,
    local_payload_inventory: Path | None = None,
    secret_factory: Callable[[], str] = lambda: secrets.token_urlsafe(48),
) -> dict[str, object]:
    """Prepare one candidate exactly once while both custody locks are held."""

    if not isinstance(lock_guard, CustodyLockGuard) or lock_guard.host_lock != str(expected_host_lock) or lock_guard.candidate_lock != str(expected_candidate_lock):
        raise PrepareError("candidate_prepare_locks_not_held")
    experiments = _real_directory(experiments_root, "experiments_root")
    staging = _real_directory(staging_root, "staging_root")
    cas = _real_directory(production_cas_root, "production_cas_root")
    data, metadata_sha256 = _load_candidate(candidate_id, candidate_pin, experiments)
    if data.get("storage_root") != str(staging / candidate_id / candidate_pin):
        raise PrepareError("candidate_storage_root_mismatch")

    local_payload_root: Path | None = None
    local_payload_blobs: dict[str, Path] | None = None
    local_payload_specs: list[dict[str, object]] | None = None
    if local_payload_inventory is not None:
        local_payload_specs = _artifact_specs(data, experiments)
        local_payload_root, local_payload_blobs = _load_local_payload_inventory(
            local_payload_inventory,
            candidate_id=candidate_id,
            candidate_pin=candidate_pin,
            metadata_sha256=metadata_sha256,
            specs=local_payload_specs,
        )

    build_input_manifest_sha256: str | None = None
    if candidate_id in {"freetoken", "llama-dflash2-official"}:
        image = data["image"]
        if (
            not isinstance(sealed_image_identity, Mapping)
            or sealed_image_identity.get("reference") != image.get("reference")
            or sealed_image_identity.get("image_digest") != image.get("digest")
            or sealed_image_identity.get("status") not in {"loaded", "reused"}
        ):
            raise PrepareError("candidate_loaded_image_identity_missing")
        build_digest = sealed_image_identity.get("build_input_manifest_sha256")
        if not isinstance(build_digest, str) or SHA256.fullmatch(build_digest) is None:
            raise PrepareError("candidate_build_input_result_invalid")
        build_input_manifest_sha256 = build_digest
    elif sealed_image_identity is not None:
        raise PrepareError("candidate_unexpected_loaded_image_identity")

    candidate_parent = _mkdir_exact(staging, candidate_id, 0o700)
    root = candidate_parent / candidate_pin
    manifest_relative = ".candidate-prepare-manifest.json"
    manifest_path = root / manifest_relative
    if root.exists() or root.is_symlink():
        _real_directory(root, "candidate_root")
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PrepareError("incomplete_prepare_without_manifest")
        try:
            existing = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrepareError("candidate_prepare_manifest_invalid") from exc
        if not isinstance(existing, dict):
            raise PrepareError("candidate_prepare_manifest_invalid")
        existing_directories = existing.get("directories")
        if (
            existing.get("candidate_id") != candidate_id
            or existing.get("candidate_pin") != candidate_pin
            or existing.get("metadata_sha256") != metadata_sha256
            or existing.get("runtime_identity") != data.get("runtime_identity")
            or existing.get("build_input_manifest_sha256") != build_input_manifest_sha256
            or not isinstance(existing_directories, list)
            or any(
                item == "build-inputs" or str(item).startswith("build-inputs/")
                for item in (existing_directories if isinstance(existing_directories, list) else [])
            )
        ):
            raise PrepareError("candidate_prepare_manifest_identity_mismatch")
        return {"candidate_root": str(root), "manifest_sha256": _verify_manifest(root, existing, manifest_path, cas), "status": "reused"}

    os.mkdir(root, 0o700)
    os.chmod(root, 0o700)
    _fsync_directory(root.parent)
    root = _real_directory(root, "candidate_root")
    cas_origins: list[ExternalArtifact] = []
    try:
        specs = local_payload_specs if local_payload_specs is not None else _artifact_specs(data, experiments)
        for spec in specs:
            relative = str(spec["path"])
            download_relative = str(spec.get("download_path", relative))
            _ensure_parent(root, download_relative)
            if local_payload_blobs is not None:
                blob = local_payload_blobs[download_relative]
                assert local_payload_root is not None
                _stream_verified_local_payload(
                    blob,
                    payload_root=local_payload_root,
                    expected_size=int(spec["size"]),
                    expected_sha256=str(spec["sha256"]),
                    consumer=lambda source: artifact_downloader(
                        root, download_relative, source,
                        expected_size=int(spec["size"]),
                        expected_sha256=str(spec["sha256"]),
                        production_cas_root=cas,
                    ),
                )
            else:
                cas_filename = spec.get("cas_filename")
                cas_path = cas / "sha256" / str(spec["sha256"]) / str(cas_filename)
                if spec.get("overlay_action") is None and cas_filename is not None and (cas_path.exists() or cas_path.is_symlink()):
                    origin = locate_reusable_cas_blob(cas, digest=str(spec["sha256"]), filename=str(cas_filename), expected_size=int(spec["size"]), logical_path=relative)
                    with Path(origin.source_path).open("rb", buffering=0) as source:
                        artifact_downloader(root, download_relative, source, expected_size=int(spec["size"]), expected_sha256=str(spec["sha256"]), production_cas_root=cas)
                    cas_origins.append(origin)
                    continue
                if "local" in spec:
                    with Path(str(spec["local"])).open("rb", buffering=0) as source:
                        artifact_downloader(root, download_relative, source, expected_size=int(spec["size"]), expected_sha256=str(spec["sha256"]), production_cas_root=cas)
                else:
                    with fetcher(str(spec["url"])) as source:
                        artifact_downloader(root, download_relative, source, expected_size=int(spec["size"]), expected_sha256=str(spec["sha256"]), production_cas_root=cas)
            action = spec.get("overlay_action")
            if action is not None:
                source_path = root.joinpath(*PurePosixPath(download_relative).parts)
                destination = root.joinpath(*PurePosixPath(relative).parts)
                _ensure_parent(root, relative)
                if action == "replace":
                    if not destination.is_file() or destination.is_symlink():
                        raise PrepareError("model_overlay_replace_target_missing")
                elif action == "add":
                    if destination.exists() or destination.is_symlink():
                        raise PrepareError("model_overlay_add_target_exists")
                else:
                    raise PrepareError("model_overlay_action_invalid")
                os.replace(source_path, destination)
                _fsync_directory(source_path.parent)
                _fsync_directory(destination.parent)
                overlay_root = root / "overlay-payload"
                current = source_path.parent
                while current != overlay_root.parent:
                    try:
                        current.rmdir()
                    except OSError:
                        break
                    if current == overlay_root:
                        break
                    current = current.parent

        secret_paths: set[str] = set()
        for mode in data["modes"].values():
            if mode.get("auth_mode") == "bearer":
                auth_path = Path(str(mode["auth_file"]))
                try:
                    relative = auth_path.relative_to(root).as_posix()
                except ValueError as exc:
                    raise PrepareError("candidate_secret_path_invalid") from exc
                secret_paths.add(relative)
        for relative in sorted(secret_paths):
            _ensure_parent(root, relative)
            _write_secret(root / Path(*PurePosixPath(relative).parts), secret_factory)
        secret_policy = _apply_runtime_layout(root, data, secret_paths)
        _verify_runtime_layout(root, data["runtime_identity"], secret_policy)
        local, secret_metadata, directories = _inventory(root, secret_policy=secret_policy, manifest_path=manifest_relative)
        manifest = {
            "candidate_id": candidate_id,
            "candidate_pin": candidate_pin,
            "candidate_root": str(root),
            "build_input_manifest_sha256": build_input_manifest_sha256,
            "directories": directories,
            "cas_origins": [item.as_dict() for item in sorted(cas_origins, key=lambda item: item.logical_path)],
            "image": {key: data["image"][key] for key in ("reference", "digest", "lock_sha256", "revision_label")},
            "local_artifacts": local,
            "metadata_sha256": metadata_sha256,
            "runtime_identity": dict(data["runtime_identity"]),
            "schema_version": SCHEMA,
            "secret_files": secret_metadata,
        }
        _write_manifest(manifest_path, manifest)
        return {"candidate_root": str(root), "manifest_sha256": _verify_manifest(root, manifest, manifest_path, cas), "status": "published"}
    except (CustodyError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PrepareError("candidate_artifact_prepare_failed") from exc
