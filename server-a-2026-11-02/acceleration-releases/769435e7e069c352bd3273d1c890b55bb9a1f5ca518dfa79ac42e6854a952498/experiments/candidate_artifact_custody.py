#!/usr/bin/env python3
"""Fail-closed custody helpers for isolated AI acceleration candidates.

The production model store is deliberately read-only to this module.  A
matching CAS blob can be returned as an external reference, but publication or
garbage collection belongs to the existing model-store tooling.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, BinaryIO, Iterable, Iterator, Mapping

try:  # POSIX-only locking and cleanup; import must still work on Windows.
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows importability
    fcntl = None  # type: ignore[assignment]


CHUNK_SIZE = 8 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MANIFEST_SCHEMA = "ai-server-candidate-artifact-manifest/v1"
RECEIPT_SCHEMA = "ai-server-candidate-artifact-receipt/v1"


class CustodyError(RuntimeError):
    """Raised when custody cannot prove an operation safe."""


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ExternalArtifact:
    logical_path: str
    source_path: str
    size: int
    sha256: str
    storage: str = "production-cas-read-only"

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_path": self.logical_path,
            "sha256": self.sha256,
            "size": self.size,
            "source_path": self.source_path,
            "storage": self.storage,
        }


@dataclass(frozen=True)
class CustodyLockGuard:
    host_lock: str
    candidate_lock: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _validate_sha256(value: str) -> None:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise CustodyError("expected sha256 must be lowercase hexadecimal")


def _real_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise CustodyError(f"{label} must be an absolute real directory")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise CustodyError(f"{label} must be canonical")
    return resolved


def _regular_file(path: Path, label: str, *, single_link: bool = True) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise CustodyError(f"{label} is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CustodyError(f"{label} must be a regular file")
    if single_link and metadata.st_nlink != 1:
        raise CustodyError(f"{label} must have exactly one hard link")
    return metadata


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str | Path, label: str) -> Path:
    raw = str(value)
    posix = PurePosixPath(raw.replace("\\", "/"))
    if (
        not raw
        or posix.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise CustodyError(f"{label} must be a normalized relative path")
    return Path(*posix.parts)


def _assert_disjoint(candidate_root: Path, production_cas_root: Path | None) -> None:
    if production_cas_root is None:
        return
    cas = _real_directory(production_cas_root, "production CAS root")
    if candidate_root == cas or candidate_root.is_relative_to(cas) or cas.is_relative_to(candidate_root):
        raise CustodyError("candidate root must be disjoint from production CAS")


def _assert_write_outside_production(path: Path, production_cas_root: Path, label: str) -> None:
    cas = _real_directory(production_cas_root, "production CAS root")
    if not path.is_absolute():
        raise CustodyError(f"{label} path must be absolute")
    parent = _real_directory(path.parent, f"{label} parent")
    target = parent / path.name
    if target == cas or target.is_relative_to(cas):
        raise CustodyError(f"{label} must be outside production CAS")


def _open_lock(path: Path) -> int:
    parent = _real_directory(path.parent, "lock parent")
    if path.parent.resolve(strict=True) != parent:
        raise CustodyError("lock parent changed during validation")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CustodyError(f"cannot open lock: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CustodyError(f"unsafe lock file: {path}")
        os.fchmod(descriptor, 0o600)
        assert fcntl is not None
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, CustodyError) as exc:
        os.close(descriptor)
        if isinstance(exc, CustodyError):
            raise
        raise CustodyError(f"lock is already held: {path}") from exc
    return descriptor


@contextmanager
def custody_locks(
    host_lock: Path,
    candidate_lock: Path,
    *,
    production_cas_root: Path,
) -> Iterator[CustodyLockGuard]:
    """Acquire the host lock first, then the nested candidate/pin lock."""

    if os.name != "posix" or fcntl is None:
        raise CustodyError("custody locks require POSIX flock")
    if not host_lock.is_absolute() or not candidate_lock.is_absolute():
        raise CustodyError("lock paths must be absolute")
    if host_lock == candidate_lock:
        raise CustodyError("host and candidate locks must be distinct")
    _assert_write_outside_production(host_lock, production_cas_root, "host lock")
    _assert_write_outside_production(candidate_lock, production_cas_root, "candidate lock")
    host_fd = _open_lock(host_lock)
    try:
        candidate_fd = _open_lock(candidate_lock)
        try:
            yield CustodyLockGuard(str(host_lock), str(candidate_lock))
        finally:
            fcntl.flock(candidate_fd, fcntl.LOCK_UN)
            os.close(candidate_fd)
    finally:
        fcntl.flock(host_fd, fcntl.LOCK_UN)
        os.close(host_fd)


def _verified_identity(path: Path, logical_path: str) -> ArtifactIdentity:
    metadata = _regular_file(path, logical_path)
    return ArtifactIdentity(logical_path, metadata.st_size, _sha256_file(path))


def download_artifact(
    candidate_root: Path,
    relative_path: str | Path,
    source: BinaryIO | Iterable[bytes],
    *,
    expected_size: int,
    expected_sha256: str,
    production_cas_root: Path,
) -> dict[str, object]:
    """Write a verified artifact through a new ``.partial`` and atomic rename."""

    root = _real_directory(candidate_root, "candidate root")
    _assert_disjoint(root, production_cas_root)
    relative = _safe_relative(relative_path, "artifact path")
    if type(expected_size) is not int or expected_size < 0:
        raise CustodyError("expected size must be a non-negative integer")
    _validate_sha256(expected_sha256)
    destination = root / relative
    parent = _real_directory(destination.parent, "artifact parent")
    if not parent.is_relative_to(root):
        raise CustodyError("artifact parent escapes candidate root")

    if destination.exists() or destination.is_symlink():
        identity = _verified_identity(destination, relative.as_posix())
        if identity.size != expected_size or identity.sha256 != expected_sha256:
            raise CustodyError("existing artifact identity mismatch")
        return {"path": relative.as_posix(), "status": "reused", "size": identity.size, "sha256": identity.sha256}

    partial = destination.with_name(destination.name + ".partial")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(partial, flags, 0o600)
    except FileExistsError as exc:
        raise CustodyError(f"partial already exists: {partial}") from exc

    digest = hashlib.sha256()
    size = 0
    try:
        iterator: Iterable[bytes]
        if hasattr(source, "read"):
            def read_chunks() -> Iterator[bytes]:
                while chunk := source.read(CHUNK_SIZE):  # type: ignore[union-attr]
                    yield chunk
            iterator = read_chunks()
        else:
            iterator = source
        for chunk in iterator:
            if not isinstance(chunk, bytes):
                raise CustodyError("download source must yield bytes")
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise CustodyError("short write while downloading artifact")
                view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        if partial.exists() and not partial.is_symlink():
            partial.unlink()
            _fsync_directory(parent)
        raise
    else:
        os.close(descriptor)

    if size != expected_size or digest.hexdigest() != expected_sha256:
        partial.unlink()
        _fsync_directory(parent)
        raise CustodyError("downloaded artifact identity mismatch")
    _regular_file(partial, "partial artifact")
    if destination.exists() or destination.is_symlink():
        partial.unlink()
        _fsync_directory(parent)
        raise CustodyError("artifact destination appeared during download")
    os.rename(partial, destination)
    _fsync_directory(parent)
    identity = _verified_identity(destination, relative.as_posix())
    if identity.size != expected_size or identity.sha256 != expected_sha256:
        raise CustodyError("published artifact identity mismatch")
    return {"path": relative.as_posix(), "status": "published", "size": identity.size, "sha256": identity.sha256}


def locate_reusable_cas_blob(
    production_cas_root: Path,
    *,
    digest: str,
    filename: str,
    expected_size: int,
    logical_path: str,
) -> ExternalArtifact:
    """Return a verified external CAS reference without modifying the CAS."""

    root = _real_directory(production_cas_root, "production CAS root")
    _validate_sha256(digest)
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise CustodyError("CAS filename must be one path component")
    logical = _safe_relative(logical_path, "logical artifact path").as_posix()
    sha_root = _real_directory(root / "sha256", "production CAS sha256 root")
    digest_root = _real_directory(sha_root / digest, "production CAS digest root")
    blob = digest_root / filename
    metadata = _regular_file(blob, "production CAS blob", single_link=False)
    if metadata.st_size != expected_size or _sha256_file(blob) != digest:
        raise CustodyError("production CAS blob identity mismatch")
    return ExternalArtifact(logical, str(blob), expected_size, digest)


def candidate_inventory(candidate_root: Path) -> list[dict[str, object]]:
    """Hash every regular file below candidate root, rejecting special entries."""

    root = _real_directory(candidate_root, "candidate root")
    inventory: list[dict[str, object]] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            _real_directory(directory_path / name, "candidate subdirectory")
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            identity = _verified_identity(path, relative)
            inventory.append({"path": identity.path, "sha256": identity.sha256, "size": identity.size})
    return sorted(inventory, key=lambda item: str(item["path"]))


def build_manifest(
    candidate_root: Path,
    *,
    external_artifacts: Iterable[ExternalArtifact] = (),
) -> dict[str, object]:
    root = _real_directory(candidate_root, "candidate root")
    external = sorted((item.as_dict() for item in external_artifacts), key=lambda item: str(item["logical_path"]))
    logical_paths = [str(item["logical_path"]) for item in external]
    if len(set(logical_paths)) != len(logical_paths):
        raise CustodyError("duplicate external logical artifact path")
    local = candidate_inventory(root)
    if set(logical_paths) & {str(item["path"]) for item in local}:
        raise CustodyError("external logical artifact aliases local inventory")
    _verify_external_inventory(external)
    return {
        "candidate_root": str(root),
        "external_artifacts": external,
        "local_artifacts": local,
        "schema_version": MANIFEST_SCHEMA,
    }


def verify_manifest(candidate_root: Path, manifest: Mapping[str, object]) -> None:
    root = _real_directory(candidate_root, "candidate root")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise CustodyError("candidate manifest schema mismatch")
    if manifest.get("candidate_root") != str(root):
        raise CustodyError("candidate manifest root mismatch")
    expected = manifest.get("local_artifacts")
    if not isinstance(expected, list) or expected != candidate_inventory(root):
        raise CustodyError("candidate inventory drift")
    external = manifest.get("external_artifacts")
    if not isinstance(external, list):
        raise CustodyError("external artifact inventory is malformed")
    _verify_external_inventory(external)


def _verify_external_inventory(entries: Iterable[object]) -> None:
    logical_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"logical_path", "sha256", "size", "source_path", "storage"}:
            raise CustodyError("external artifact inventory is malformed")
        logical = _safe_relative(str(entry["logical_path"]), "external logical artifact path").as_posix()
        if logical in logical_paths:
            raise CustodyError("duplicate external logical artifact path")
        logical_paths.add(logical)
        if entry["storage"] != "production-cas-read-only":
            raise CustodyError("external artifact storage is not read-only production CAS")
        digest = entry["sha256"]
        size = entry["size"]
        source = Path(str(entry["source_path"]))
        _validate_sha256(digest)
        if type(size) is not int or size < 0 or not source.is_absolute():
            raise CustodyError("external artifact identity is malformed")
        metadata = _regular_file(source, "external artifact", single_link=False)
        if metadata.st_size != size or _sha256_file(source) != digest:
            raise CustodyError("external artifact identity mismatch")


def write_immutable_receipt(
    path: Path,
    content: Mapping[str, Any],
    *,
    production_cas_root: Path,
) -> dict[str, str]:
    """Create a canonical, content-addressed receipt exactly once."""

    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise CustodyError("receipt path must be a new absolute path")
    _assert_write_outside_production(path, production_cas_root, "receipt")
    parent = _real_directory(path.parent, "receipt parent")
    canonical_content = _canonical_json(dict(content))
    content_sha256 = hashlib.sha256(canonical_content).hexdigest()
    envelope = {
        "canonical_content_sha256": content_sha256,
        "content": dict(content),
        "schema_version": RECEIPT_SCHEMA,
    }
    payload = _canonical_json(envelope) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        os.fchmod(descriptor, 0o400)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CustodyError("short write while creating receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(parent)
    return {"canonical_content_sha256": content_sha256, "receipt_sha256": hashlib.sha256(payload).hexdigest()}


def _decode_mount_path(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def _mounts_below(root: Path, mountinfo_path: Path) -> list[str]:
    if not mountinfo_path.is_file():
        raise CustodyError("mount inventory is unavailable")
    found: list[str] = []
    for line in mountinfo_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise CustodyError("mount inventory is malformed")
        mount = Path(_decode_mount_path(fields[4]))
        try:
            resolved = mount.resolve(strict=False)
        except OSError as exc:
            raise CustodyError("cannot resolve mount inventory entry") from exc
        if resolved == root or resolved.is_relative_to(root):
            found.append(str(mount))
    return found


def _pids_referencing(root: Path, proc_root: Path) -> list[int]:
    if not proc_root.is_dir():
        raise CustodyError("PID inventory is unavailable")
    found: set[int] = set()
    for process in proc_root.iterdir():
        if not process.name.isdigit() or not process.is_dir():
            continue
        links = [process / "cwd", process / "root", process / "exe"]
        fd_root = process / "fd"
        try:
            if fd_root.is_dir():
                links.extend(fd_root.iterdir())
        except (FileNotFoundError, PermissionError):
            pass
        for link in links:
            try:
                target = link.resolve(strict=False)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target == root or target.is_relative_to(root):
                found.add(int(process.name))
                break
    return sorted(found)


def cleanup_candidate(
    candidate_root: Path,
    *,
    staging_experiments_root: Path,
    manifest: Mapping[str, object],
    production_cas_root: Path,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
    proc_root: Path = Path("/proc"),
) -> None:
    """Delete exactly one ``candidate/pin`` root after complete safety proof."""

    if os.name != "posix":
        raise CustodyError("candidate cleanup requires POSIX runtime evidence")
    staging = _real_directory(staging_experiments_root, "staging experiments root")
    root = _real_directory(candidate_root, "candidate root")
    _assert_disjoint(root, production_cas_root)
    try:
        relative = root.relative_to(staging)
    except ValueError as exc:
        raise CustodyError("candidate root escapes staging experiments root") from exc
    if len(relative.parts) != 2 or any(part in {"", ".", ".."} for part in relative.parts):
        raise CustodyError("cleanup target must be exact candidate/pin root")
    verify_manifest(root, manifest)
    mounts = _mounts_below(root, mountinfo_path)
    if mounts:
        raise CustodyError(f"candidate root is mounted: {mounts[0]}")
    pids = _pids_referencing(root, proc_root)
    if pids:
        raise CustodyError(f"candidate root is referenced by PID: {pids[0]}")

    # No glob or broad-root deletion: remove only the already-verified tree.
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        directory_path = Path(directory)
        for name in files:
            path = directory_path / name
            _regular_file(path, "cleanup artifact")
            path.unlink()
        for name in names:
            child = directory_path / name
            _real_directory(child, "cleanup directory")
            child.rmdir()
    root.rmdir()
    _fsync_directory(root.parent)
