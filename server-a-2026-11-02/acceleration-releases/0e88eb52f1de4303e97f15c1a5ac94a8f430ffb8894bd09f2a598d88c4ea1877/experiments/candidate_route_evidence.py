"""Bounded owner/browser evidence exchange for authenticated production routes.

No credential, URL, prompt, response, cookie, or token is accepted or stored.
The live runner writes one immutable challenge and waits for one immutable,
content-free owner response for each of the two exact restore boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any, Callable, Mapping


ROUTE_EVIDENCE_ROOT = Path(
    "/srv/ai-server/state/acceleration-experiments/authenticated-route-evidence"
)
ROUTE_EVIDENCE_TIMEOUT_SECONDS = 900.0
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,63}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NONCE = re.compile(r"[0-9a-f]{32}\Z")
MAX_FILE_BYTES = 4096
CHALLENGE_SCHEMA = "ai-server-acceleration-route-challenge/v1"
RESPONSE_SCHEMA = "ai-server-acceleration-route-response/v1"


class RouteEvidenceError(RuntimeError):
    def __init__(self, code: str) -> None:
        if re.fullmatch(r"[a-z0-9_]+", code) is None:
            raise ValueError("route_evidence_error_invalid")
        self.code = code
        super().__init__(code)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def _secure_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RouteEvidenceError("route_evidence_directory_invalid") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or (
            os.name == "posix"
            and (stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != 0)
        )
    ):
        raise RouteEvidenceError("route_evidence_directory_invalid")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_root(root: Path) -> None:
    if root.exists() or root.is_symlink():
        _secure_directory(root)
        return
    _secure_directory(root.parent)
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        _secure_directory(root)
        return
    os.chmod(root, 0o700)
    _fsync_directory(root.parent)
    _secure_directory(root)


def _write_new(path: Path, payload: bytes) -> None:
    if not payload or len(payload) > MAX_FILE_BYTES:
        raise RouteEvidenceError("route_evidence_payload_invalid")
    _secure_directory(path.parent)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        raise RouteEvidenceError("route_evidence_write_failed") from exc
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise RouteEvidenceError("route_evidence_write_failed") from exc
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_secure_file(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RouteEvidenceError("route_evidence_file_invalid") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_FILE_BYTES
        or (
            os.name == "posix"
            and (stat.S_IMODE(before.st_mode) != 0o400 or before.st_uid != 0)
        )
    ):
        raise RouteEvidenceError("route_evidence_file_invalid")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_BINARY", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RouteEvidenceError("route_evidence_file_invalid")
            raw = os.read(descriptor, MAX_FILE_BYTES + 1)
            if os.read(descriptor, 1):
                raise RouteEvidenceError("route_evidence_file_invalid")
        finally:
            os.close(descriptor)
    except RouteEvidenceError:
        raise
    except OSError as exc:
        raise RouteEvidenceError("route_evidence_file_invalid") from exc
    if len(raw) != before.st_size:
        raise RouteEvidenceError("route_evidence_file_invalid")
    return raw


def _json(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteEvidenceError(code) from exc
    if not isinstance(value, dict):
        raise RouteEvidenceError(code)
    return value


def _operation_directory(root: Path, operation_id: str, *, create: bool) -> Path:
    if SAFE_ID.fullmatch(operation_id) is None:
        raise RouteEvidenceError("route_evidence_operation_invalid")
    _ensure_root(root)
    directory = root / operation_id
    if create:
        if directory.exists() or directory.is_symlink():
            raise RouteEvidenceError("route_evidence_stale")
        try:
            os.mkdir(directory, 0o700)
        except OSError as exc:
            raise RouteEvidenceError("route_evidence_directory_invalid") from exc
        os.chmod(directory, 0o700)
        _fsync_directory(root)
    _secure_directory(directory)
    return directory


def submit_route_evidence(
    operation_id: str,
    sequence: int,
    *,
    root: Path = ROUTE_EVIDENCE_ROOT,
) -> dict[str, Any]:
    """Write the one allowed all-true response for an existing challenge."""

    if type(sequence) is not int or sequence not in {1, 2}:
        raise RouteEvidenceError("route_evidence_sequence_invalid")
    directory = _operation_directory(root, operation_id, create=False)
    challenge_path = directory / f"challenge-{sequence}.json"
    response_path = directory / f"response-{sequence}.json"
    if response_path.exists() or response_path.is_symlink():
        raise RouteEvidenceError("route_evidence_response_exists")
    challenge_raw = _read_secure_file(challenge_path)
    challenge = _json(challenge_raw, "route_evidence_challenge_invalid")
    if set(challenge) != {"schema_version", "operation_id", "sequence", "nonce"} or (
        challenge.get("schema_version") != CHALLENGE_SCHEMA
        or challenge.get("operation_id") != operation_id
        or challenge.get("sequence") != sequence
        or not isinstance(challenge.get("nonce"), str)
        or NONCE.fullmatch(challenge["nonce"]) is None
    ):
        raise RouteEvidenceError("route_evidence_challenge_invalid")
    challenge_sha = hashlib.sha256(challenge_raw).hexdigest()
    response = {
        "schema_version": RESPONSE_SCHEMA,
        "operation_id": operation_id,
        "sequence": sequence,
        "nonce": challenge["nonce"],
        "challenge_sha256": challenge_sha,
        "ai": True,
        "pony": True,
        "comfy": True,
    }
    payload = _canonical(response)
    _write_new(response_path, payload)
    return {
        "operation_id": operation_id,
        "sequence": sequence,
        "challenge_sha256": challenge_sha,
        "response_sha256": hashlib.sha256(payload).hexdigest(),
    }


class BrowserRouteEvidenceProbe:
    """Sequential two-boundary challenge producer and response verifier."""

    def __init__(
        self,
        operation_id: str,
        *,
        root: Path = ROUTE_EVIDENCE_ROOT,
        timeout_seconds: float = ROUTE_EVIDENCE_TIMEOUT_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if SAFE_ID.fullmatch(operation_id) is None:
            raise ValueError("route_evidence_operation_invalid")
        if timeout_seconds <= 0 or timeout_seconds > ROUTE_EVIDENCE_TIMEOUT_SECONDS:
            raise ValueError("route_evidence_timeout_invalid")
        self.operation_id = operation_id
        self.root = root
        self.timeout_seconds = timeout_seconds
        self.monotonic = monotonic
        self.sleep = sleep
        self._next_sequence = 1
        self._stage_deadline: float | None = None

    def set_deadline(self, deadline: float | None) -> None:
        if deadline is not None and deadline <= self.monotonic():
            raise RouteEvidenceError("route_evidence_timeout")
        self._stage_deadline = deadline

    def probe(self) -> Mapping[str, Any]:
        sequence = self._next_sequence
        if sequence not in {1, 2}:
            raise RouteEvidenceError("route_evidence_sequence_exhausted")
        directory = _operation_directory(
            self.root, self.operation_id, create=sequence == 1
        )
        challenge_path = directory / f"challenge-{sequence}.json"
        response_path = directory / f"response-{sequence}.json"
        if (
            challenge_path.exists()
            or challenge_path.is_symlink()
            or response_path.exists()
            or response_path.is_symlink()
        ):
            raise RouteEvidenceError("route_evidence_stale")
        challenge = {
            "schema_version": CHALLENGE_SCHEMA,
            "operation_id": self.operation_id,
            "sequence": sequence,
            "nonce": secrets.token_hex(16),
        }
        challenge_raw = _canonical(challenge)
        challenge_sha = hashlib.sha256(challenge_raw).hexdigest()
        _write_new(challenge_path, challenge_raw)
        deadline = self.monotonic() + self.timeout_seconds
        if self._stage_deadline is not None:
            deadline = min(deadline, self._stage_deadline)
        while True:
            if response_path.exists() or response_path.is_symlink():
                break
            if self.monotonic() >= deadline:
                raise RouteEvidenceError("route_evidence_timeout")
            self.sleep(min(1.0, max(0.0, deadline - self.monotonic())))
        response_raw = _read_secure_file(response_path)
        response = _json(response_raw, "route_evidence_response_invalid")
        expected_keys = {
            "schema_version", "operation_id", "sequence", "nonce",
            "challenge_sha256", "ai", "pony", "comfy",
        }
        if set(response) != expected_keys or (
            response.get("schema_version") != RESPONSE_SCHEMA
            or response.get("operation_id") != self.operation_id
            or response.get("sequence") != sequence
            or response.get("nonce") != challenge["nonce"]
            or response.get("challenge_sha256") != challenge_sha
            or response.get("ai") is not True
            or response.get("pony") is not True
            or response.get("comfy") is not True
        ):
            raise RouteEvidenceError("route_evidence_response_invalid")
        self._next_sequence += 1
        return {
            "sequence": sequence,
            "ai": True,
            "pony": True,
            "comfy": True,
            "response_sha256": hashlib.sha256(response_raw).hexdigest(),
        }


__all__ = [
    "BrowserRouteEvidenceProbe", "RouteEvidenceError", "ROUTE_EVIDENCE_ROOT",
    "ROUTE_EVIDENCE_TIMEOUT_SECONDS", "submit_route_evidence",
]
