#!/usr/bin/env python3
"""Materialize the pinned Higgs TTS 3 snapshot once without an HF cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


REVISION = "7556c17e05201fccd9c8cc120bc216dcc7b5d561"
REPOSITORY = "bosonai/higgs-tts-3-4b"
DEFAULT_ROOT = Path(
    "/srv/ai-server/staging/experiments/higgs-tts3" "/7556c17e05201fccd9c8cc120bc216dcc7b5d561"
)
CHUNK_BYTES = 8 * 1024 * 1024


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_once(path: Path, value: object) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"receipt path is not fresh: {path}")
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    os.chmod(path, 0o644)
    _fsync_directory(path.parent)


def _hash_file(path: Path, digest=None) -> tuple[int, str]:
    digest = hashlib.sha256() if digest is None else digest
    observed_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            observed_size += len(chunk)
    return observed_size, digest.hexdigest()


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: install_model.py MANIFEST [CANDIDATE_ROOT]")
    manifest_path = Path(sys.argv[1]).resolve(strict=True)
    candidate_root = Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_ROOT
    candidate_root = candidate_root.resolve(strict=True)
    model_root = candidate_root / "model"
    receipt_root = candidate_root / "receipts"
    os.chmod(model_root, 0o755)
    if any(receipt_root.iterdir()):
        raise RuntimeError("candidate receipt directory is not empty")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("schema") != "ai-server-huggingface-model-manifest/v1"
        or manifest.get("repository") != REPOSITORY
        or manifest.get("revision") != REVISION
    ):
        raise RuntimeError("model manifest identity mismatch")

    allowed_paths: set[Path] = set()
    for item in manifest["files"]:
        destination = model_root / Path(item["path"])
        allowed_paths.add(destination)
        allowed_paths.add(destination.with_name(destination.name + ".partial"))
    for existing in model_root.rglob("*"):
        if existing.is_symlink():
            raise RuntimeError(f"symlink is not allowed in candidate model: {existing}")
        if existing.is_file() and existing not in allowed_paths:
            raise RuntimeError(f"unknown candidate model artifact: {existing}")

    installed: list[dict[str, object]] = []
    observed_total = 0
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe model path: {relative}")
        destination = model_root / relative
        partial = destination.with_name(destination.name + ".partial")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o755)
        if destination.exists():
            if partial.exists():
                raise RuntimeError(f"final and partial artifacts coexist: {relative}")
            observed_size, observed_digest = _hash_file(destination)
            if observed_size != item["size"] or observed_digest != item["sha256"]:
                raise RuntimeError(f"existing artifact identity mismatch: {relative}")
            os.chmod(destination, 0o644)
            observed_total += observed_size
            installed.append(
                {"path": relative.as_posix(), "size": observed_size, "sha256": observed_digest}
            )
            print(f"existing_verified {relative} {observed_size} {observed_digest}", flush=True)
            continue

        resume_size = partial.stat().st_size if partial.exists() else 0
        if resume_size >= item["size"]:
            raise RuntimeError(f"partial artifact has invalid size: {relative} {resume_size}")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{relative.as_posix()}"
        headers = {"User-Agent": "home-lab-higgs-tts3-installer/1"}
        if resume_size:
            headers["Range"] = f"bytes={resume_size}-"
            headers["Accept-Encoding"] = "identity"
        request = urllib.request.Request(
            url, headers=headers
        )
        digest = hashlib.sha256()
        observed_size = 0
        if resume_size:
            observed_size, prefix_digest = _hash_file(partial, digest)
            if observed_size != resume_size:
                raise RuntimeError(f"partial artifact changed while hashing: {relative}")
            print(
                f"download_resume {relative} {resume_size} prefix_sha256={prefix_digest}",
                flush=True,
            )
        else:
            print(f"download_start {relative}", flush=True)
        mode = "ab" if resume_size else "xb"
        with urllib.request.urlopen(request, timeout=60) as response, partial.open(mode) as stream:
            if resume_size:
                expected_range = f"bytes {resume_size}-{item['size'] - 1}/{item['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(
                        f"range response mismatch: {relative} "
                        f"status={response.status} content_range={response.headers.get('Content-Range')}"
                    )
            elif response.status != 200:
                raise RuntimeError(f"download response mismatch: {relative} status={response.status}")
            while chunk := response.read(CHUNK_BYTES):
                stream.write(chunk)
                digest.update(chunk)
                observed_size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        observed_digest = digest.hexdigest()
        if observed_size != item["size"] or observed_digest != item["sha256"]:
            raise RuntimeError(
                f"artifact identity mismatch: {relative} "
                f"size={observed_size} sha256={observed_digest}"
            )
        os.replace(partial, destination)
        os.chmod(destination, 0o644)
        _fsync_directory(destination.parent)
        observed_total += observed_size
        installed.append(
            {"path": relative.as_posix(), "size": observed_size, "sha256": observed_digest}
        )
        print(f"download_verified {relative} {observed_size} {observed_digest}", flush=True)

    if observed_total != manifest["total_bytes"]:
        raise RuntimeError("installed model total does not match manifest")
    receipt = {
        "schema": "ai-server-higgs-tts3-install-receipt/v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "total_bytes": observed_total,
        "files": installed,
    }
    _write_json_once(receipt_root / "model-install.json", receipt)
    print("install_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
