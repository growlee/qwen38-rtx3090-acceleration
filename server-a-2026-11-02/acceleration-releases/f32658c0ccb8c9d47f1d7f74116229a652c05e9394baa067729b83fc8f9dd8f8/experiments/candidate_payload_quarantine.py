#!/usr/bin/env python3
"""Seal and verify prepositioned Qwen payloads in an immutable quarantine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import sys
from typing import Iterable, Mapping


CANDIDATE_ID = "qwen38-vllm"
SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")
PIN = __import__("re").compile(r"^[0-9a-f]{40}$")
MANIFEST_NAME = "payload-inventory.json"
RECEIPT_NAME = "quarantine.receipt.json"
RECEIPT_SCHEMA = "ai-server-qwen-payload-quarantine-receipt/v1"
CHUNK_SIZE = 8 * 1024 * 1024


class QuarantineError(RuntimeError):
    """Raised when ingress cannot be proven immutable and content-addressed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")


def _manifest_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True,
    ).encode("ascii") + b"\n"


def _relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise QuarantineError(f"{label}_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or path.as_posix() != value:
        raise QuarantineError(f"{label}_invalid")
    return value


def _regular_descriptor(path: Path, label: str) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or path.is_symlink():
        raise QuarantineError(f"{label}_invalid")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        observed = path.lstat()
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise QuarantineError(f"{label}_invalid") from exc
    if (
        not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
        or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
        or (info.st_dev, info.st_ino) != (observed.st_dev, observed.st_ino)
    ):
        os.close(descriptor)
        raise QuarantineError(f"{label}_invalid")
    return descriptor, info


def _read_json_file(path: Path, *, manifest: bool, label: str) -> tuple[dict[str, object], bytes]:
    descriptor, _info = _regular_descriptor(path, label)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read()
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QuarantineError(f"{label}_invalid") from exc
    if not isinstance(value, dict) or raw != (_manifest_bytes(value) if manifest else _canonical(value) + b"\n"):
        raise QuarantineError(f"{label}_not_canonical")
    return value, raw


def _inventory_identity(value: Mapping[str, object]) -> tuple[str, list[dict[str, object]]]:
    if set(value) != {"candidate_id", "candidate_pin", "candidate_metadata_sha256", "artifacts"}:
        raise QuarantineError("payload_inventory_schema_invalid")
    if value.get("candidate_id") != CANDIDATE_ID or not isinstance(value.get("candidate_pin"), str) or PIN.fullmatch(str(value["candidate_pin"])) is None:
        raise QuarantineError("payload_inventory_identity_invalid")
    metadata = value.get("candidate_metadata_sha256")
    if not isinstance(metadata, str) or SHA256.fullmatch(metadata) is None:
        raise QuarantineError("payload_inventory_identity_invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise QuarantineError("payload_inventory_schema_invalid")
    normalized: list[dict[str, object]] = []
    downloads: set[str] = set()
    digest_sizes: dict[str, int] = {}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"logical_path", "download_path", "overlay_action", "blob_path", "size", "sha256"}:
            raise QuarantineError("payload_inventory_schema_invalid")
        logical = _relative(item.get("logical_path"), "logical_path")
        download = _relative(item.get("download_path"), "download_path")
        action = item.get("overlay_action")
        size = item.get("size")
        digest = item.get("sha256")
        if download in downloads or action not in {None, "replace", "add"} or type(size) is not int or size < 0 or not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise QuarantineError("payload_inventory_schema_invalid")
        if digest in digest_sizes and digest_sizes[digest] != size:
            raise QuarantineError("payload_inventory_schema_invalid")
        downloads.add(download)
        digest_sizes[digest] = size
        normalized.append({
            "logical_path": logical, "download_path": download,
            "overlay_action": action, "size": size, "sha256": digest,
        })
    identity = {
        "candidate_id": value["candidate_id"],
        "candidate_pin": value["candidate_pin"],
        "candidate_metadata_sha256": metadata,
        "artifacts": normalized,
    }
    return hashlib.sha256(_canonical(identity)).hexdigest(), normalized


def _owner_directory(path: Path, mode: int, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path.absolute():
        raise QuarantineError(f"{label}_invalid")
    info = path.stat()
    if os.name == "posix" and (os.geteuid() != 0 or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != mode):
        raise QuarantineError(f"{label}_ownership_invalid")
    return path


def _mkdir(parent: Path, name: str, mode: int = 0o700) -> Path:
    path = parent / name
    try:
        os.mkdir(path, mode)
    except FileExistsError:
        return _owner_directory(path, mode, "quarantine_directory")
    os.chmod(path, mode)
    return path


def _copy_verified(source: Path, destination: Path, size: int, digest: str) -> None:
    source_fd, source_info = _regular_descriptor(source, "source_blob")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    destination_fd = os.open(destination, flags, 0o400)
    observed = hashlib.sha256()
    written = 0
    try:
        while chunk := os.read(source_fd, CHUNK_SIZE):
            observed.update(chunk)
            written += len(chunk)
            view = memoryview(chunk)
            while view:
                count = os.write(destination_fd, view)
                if count <= 0:
                    raise QuarantineError("quarantine_short_write")
                view = view[count:]
        after = os.fstat(source_fd)
        if (
            written != size or source_info.st_size != size or observed.hexdigest() != digest
            or (source_info.st_dev, source_info.st_ino, source_info.st_mtime_ns, source_info.st_size)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
        ):
            raise QuarantineError("source_blob_identity_mismatch")
        os.fchmod(destination_fd, 0o400)
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise QuarantineError("quarantine_short_write")
            view = view[count:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_incoming(path: Path) -> None:
    """Remove only the unique unfinalized tree, including read-only files on Windows."""

    def make_writable_then_retry(function: object, target: str, _error: object) -> None:
        os.chmod(target, stat.S_IWRITE)
        function(target)  # type: ignore[operator]

    shutil.rmtree(path, onexc=make_writable_then_retry)


def seal(source_inventory: Path, quarantine_root: Path) -> dict[str, object]:
    """Copy a mutable prepositioned payload into one atomically finalized tree."""

    source, _raw = _read_json_file(source_inventory, manifest=True, label="source_inventory")
    inventory_sha256, normalized = _inventory_identity(source)
    candidate_pin = str(source["candidate_pin"])
    base = _owner_directory(quarantine_root, 0o700, "quarantine_root")
    candidate_root = _mkdir(base, CANDIDATE_ID)
    pin_root = _mkdir(candidate_root, candidate_pin)
    final_root = pin_root / inventory_sha256
    if final_root.exists() or final_root.is_symlink():
        return verify(final_root / MANIFEST_NAME, final_root / RECEIPT_NAME, candidate_id=CANDIDATE_ID, candidate_pin=candidate_pin)
    temporary = pin_root / (".incoming-" + secrets.token_hex(16))
    os.mkdir(temporary, 0o700)
    try:
        blobs_root = _mkdir(temporary, "blobs")
        sha_root = _mkdir(blobs_root, "sha256")
        source_items = source["artifacts"]
        assert isinstance(source_items, list)
        final_artifacts: list[dict[str, object]] = []
        copied: set[str] = set()
        for original, identity in zip(source_items, normalized, strict=True):
            assert isinstance(original, dict)
            digest = str(identity["sha256"])
            destination = sha_root / digest
            if digest not in copied:
                _copy_verified(Path(str(original["blob_path"])), destination, int(identity["size"]), digest)
                copied.add(digest)
            final_artifacts.append({**identity, "blob_path": str(final_root / "blobs" / "sha256" / digest)})
        manifest = {
            "candidate_id": CANDIDATE_ID,
            "candidate_pin": candidate_pin,
            "candidate_metadata_sha256": source["candidate_metadata_sha256"],
            "artifacts": final_artifacts,
        }
        manifest_payload = _manifest_bytes(manifest)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        files = sorted({
            str(item["sha256"]): {"path": f"blobs/sha256/{item['sha256']}", "sha256": item["sha256"], "size": item["size"]}
            for item in normalized
        }.values(), key=lambda item: str(item["path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "candidate_pin": candidate_pin,
            "candidate_metadata_sha256": source["candidate_metadata_sha256"],
            "inventory_sha256": inventory_sha256,
            "manifest_sha256": manifest_sha256,
            "files": files,
        }
        _write_new(temporary / MANIFEST_NAME, manifest_payload)
        _write_new(temporary / RECEIPT_NAME, _canonical(receipt) + b"\n")
        for directory in (sha_root, blobs_root, temporary):
            os.chmod(directory, 0o500)
            _fsync_directory(directory)
        os.rename(temporary, final_root)
        _fsync_directory(pin_root)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            _remove_incoming(temporary)
        raise
    return verify(final_root / MANIFEST_NAME, final_root / RECEIPT_NAME, candidate_id=CANDIDATE_ID, candidate_pin=candidate_pin)


def verify(
    manifest_path: Path,
    receipt_path: Path,
    *,
    candidate_id: str,
    candidate_pin: str,
) -> dict[str, object]:
    if candidate_id != CANDIDATE_ID or PIN.fullmatch(candidate_pin) is None:
        raise QuarantineError("quarantine_expected_identity_invalid")
    if manifest_path.name != MANIFEST_NAME or receipt_path.name != RECEIPT_NAME or manifest_path.parent != receipt_path.parent:
        raise QuarantineError("quarantine_path_invalid")
    root = _owner_directory(manifest_path.parent, 0o500, "quarantine_tree")
    if root.parent.name != candidate_pin or root.parent.parent.name != candidate_id:
        raise QuarantineError("quarantine_path_invalid")
    _owner_directory(root.parent, 0o700, "quarantine_pin_root")
    _owner_directory(root.parent.parent, 0o700, "quarantine_candidate_root")
    _owner_directory(root.parent.parent.parent, 0o700, "quarantine_root")
    manifest, manifest_raw = _read_json_file(manifest_path, manifest=True, label="quarantine_manifest")
    receipt, receipt_raw = _read_json_file(receipt_path, manifest=False, label="quarantine_receipt")
    inventory_sha256, normalized = _inventory_identity(manifest)
    if manifest.get("candidate_id") != candidate_id or manifest.get("candidate_pin") != candidate_pin or root.name != inventory_sha256:
        raise QuarantineError("quarantine_identity_mismatch")
    expected_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_pin": candidate_pin,
        "candidate_metadata_sha256": manifest["candidate_metadata_sha256"],
        "inventory_sha256": inventory_sha256,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "files": sorted({
            str(item["sha256"]): {"path": f"blobs/sha256/{item['sha256']}", "sha256": item["sha256"], "size": item["size"]}
            for item in normalized
        }.values(), key=lambda item: str(item["path"])),
    }
    if receipt != expected_receipt:
        raise QuarantineError("quarantine_receipt_mismatch")
    manifest_artifacts = manifest["artifacts"]
    assert isinstance(manifest_artifacts, list)
    for artifact in manifest_artifacts:
        assert isinstance(artifact, dict)
        expected_blob = (root / "blobs" / "sha256" / str(artifact["sha256"])).absolute()
        blob_value = artifact.get("blob_path")
        if not isinstance(blob_value, str) or Path(blob_value) != expected_blob:
            raise QuarantineError("quarantine_blob_path_mismatch")
    expected_files = {MANIFEST_NAME, RECEIPT_NAME, *(str(item["path"]) for item in expected_receipt["files"])}
    found_files: set[str] = set()
    found_dirs = {".", "blobs", "blobs/sha256"}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root).as_posix() or "."
        info = current_path.lstat()
        if relative_dir not in found_dirs or current_path.is_symlink() or not stat.S_ISDIR(info.st_mode) or (os.name == "posix" and (info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o500)):
            raise QuarantineError("quarantine_directory_invalid")
        for name in directories:
            child = current_path / name
            if child.is_symlink():
                raise QuarantineError("quarantine_directory_invalid")
        for name in files:
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            descriptor, file_info = _regular_descriptor(child, "quarantine_file")
            try:
                if os.name == "posix" and (file_info.st_uid != 0 or stat.S_IMODE(file_info.st_mode) != 0o400):
                    raise QuarantineError("quarantine_file_mode_invalid")
            finally:
                os.close(descriptor)
            found_files.add(relative)
    if found_files != expected_files:
        raise QuarantineError("quarantine_inventory_mismatch")
    for item in expected_receipt["files"]:
        path = root.joinpath(*PurePosixPath(str(item["path"])).parts)
        descriptor, info = _regular_descriptor(path, "quarantine_blob")
        digest = hashlib.sha256()
        try:
            while chunk := os.read(descriptor, CHUNK_SIZE):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        if info.st_size != item["size"] or digest.hexdigest() != item["sha256"]:
            raise QuarantineError("quarantine_blob_identity_mismatch")
    return {
        "candidate_id": candidate_id,
        "candidate_pin": candidate_pin,
        "inventory_sha256": inventory_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "status": "verified",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal_command = commands.add_parser("seal")
    seal_command.add_argument("--source-inventory", type=Path, required=True)
    seal_command.add_argument("--quarantine-root", type=Path, required=True)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--manifest", type=Path, required=True)
    verify_command.add_argument("--receipt", type=Path, required=True)
    verify_command.add_argument("--candidate-id", choices=(CANDIDATE_ID,), required=True)
    verify_command.add_argument("--candidate-pin", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            result = seal(args.source_inventory, args.quarantine_root)
        else:
            result = verify(args.manifest, args.receipt, candidate_id=args.candidate_id, candidate_pin=args.candidate_pin)
    except QuarantineError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
