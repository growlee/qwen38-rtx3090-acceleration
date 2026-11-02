#!/usr/bin/env python3
"""Fail-closed entry point for one reviewed candidate live operation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

try:  # pragma: no cover - Server A is POSIX; Windows only runs unit tests.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

sys.dont_write_bytecode = True

try:
    from . import candidate_package
    from . import candidate_payload_quarantine as payload_quarantine
    from . import candidate_host_collector as collector
    from . import host_compatibility_preflight as preflight
    from .candidate_live_backend import (
        EXPERIMENTS_ROOT,
        HOST_LOCK,
        STATE_ROOT,
        CandidateLiveBackend,
    )
    from .candidate_live_operation import CandidateLiveOperation, LiveOperationError
    from .candidate_route_evidence import (
        BrowserRouteEvidenceProbe,
        submit_route_evidence,
    )
except ImportError:  # pragma: no cover - direct execution from experiments/
    import candidate_package  # type: ignore[no-redef]
    import candidate_payload_quarantine as payload_quarantine  # type: ignore[no-redef]
    import candidate_host_collector as collector  # type: ignore[no-redef]
    import host_compatibility_preflight as preflight  # type: ignore[no-redef]
    from candidate_live_backend import (  # type: ignore[no-redef]
        EXPERIMENTS_ROOT,
        HOST_LOCK,
        STATE_ROOT,
        CandidateLiveBackend,
    )
    from candidate_live_operation import (  # type: ignore[no-redef]
        CandidateLiveOperation,
        LiveOperationError,
    )
    from candidate_route_evidence import (  # type: ignore[no-redef]
        BrowserRouteEvidenceProbe,
        submit_route_evidence,
    )


SEALED_RELEASES_ROOT = Path("/opt/homelab/ai-server/releases/acceleration")
SEALED_PACKAGES_ROOT = STATE_ROOT / "packages"
PINS = {
    "qwen38-vllm": "69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c",
    "freetoken": "9db1a39455a3fb107f3db83e381d10ceadfe5d99",
    "llama-dflash2-official": "b10f9ca58c89ccfc3653ac01e979dd085d582b76",
}
CONFIG_SCHEMA = "ai-server-acceleration-preflight-config/v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,63}\Z")
RESULTS = {
    "passed_restored",
    "failed_preproduction",
    "failed_restored",
    "unknown_outcome",
    "failed_without_receipt",
    "offline_verified",
    "route_evidence_submitted",
}


class CliError(RuntimeError):
    """Content-free CLI validation failure."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliError("arguments_invalid")


class _ReceiptBackend(CandidateLiveBackend):
    last_receipt_sha256: str | None = None
    last_receipt_state: str | None = None

    def __init__(
        self,
        *args: Any,
        package_root: Path,
        package_archive: Path,
        package_receipt: Path,
        receipt_root: Path,
        reviewed_package_sha256: str,
        package_candidate_id: str,
        payload_quarantine_receipt: Path | None,
        payload_quarantine_inventory_sha256: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.package_root = package_root
        self.package_archive = package_archive
        self.package_receipt = package_receipt
        self.receipt_root = receipt_root
        self.reviewed_package_sha256 = reviewed_package_sha256
        self.package_candidate_id = package_candidate_id
        self.payload_quarantine_receipt = payload_quarantine_receipt
        self.payload_quarantine_inventory_sha256 = payload_quarantine_inventory_sha256

    @contextmanager
    def host_lock(self, operation_id: str):
        with super().host_lock(operation_id):
            # Repeat the package identity check while the host-wide lock is held
            # and before CandidateLiveOperation can preflight or prepare anything.
            try:
                observed = _package_identity(
                    self.package_archive,
                    self.package_receipt,
                    self.package_root,
                    self.receipt_root,
                    self.package_candidate_id,
                )
            except CliError as exc:
                raise LiveOperationError("reviewed_package_changed") from exc
            if observed != self.reviewed_package_sha256:
                raise LiveOperationError("reviewed_package_changed")
            yield None

    def provision_host_lock(self) -> None:
        try:
            _provision_lock_file(HOST_LOCK)
        except CliError as exc:
            raise LiveOperationError("host_lock_provision_failed") from exc

    def validate_candidate_ingress(self, candidate_id: str, candidate_pin: str) -> str | None:
        if candidate_id != "qwen38-vllm":
            if self.payload_quarantine_receipt is not None or self.payload_quarantine_inventory_sha256 is not None:
                raise LiveOperationError("candidate_ingress_unexpected")
            return None
        if self.local_payload_inventory is None or self.payload_quarantine_receipt is None:
            raise LiveOperationError("candidate_ingress_missing")
        try:
            verified = payload_quarantine.verify(
                self.local_payload_inventory,
                self.payload_quarantine_receipt,
                candidate_id=candidate_id,
                candidate_pin=candidate_pin,
            )
        except (payload_quarantine.QuarantineError, OSError, ValueError, TypeError) as exc:
            raise LiveOperationError("candidate_ingress_changed") from exc
        observed = verified.get("inventory_sha256")
        if observed != self.payload_quarantine_inventory_sha256 or not isinstance(observed, str) or SHA256.fullmatch(observed) is None:
            raise LiveOperationError("candidate_ingress_changed")
        return observed

    def write_receipt(self, operation_id: str, receipt: Mapping[str, Any]) -> str:
        digest = super().write_receipt(operation_id, receipt)
        self.last_receipt_sha256 = digest
        state = receipt.get("state")
        self.last_receipt_state = state if isinstance(state, str) else None
        return digest


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "run"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--candidate-id", choices=tuple(PINS), required=True)
        command.add_argument("--package-root", type=Path, required=True)
        command.add_argument("--package-archive", type=Path, required=True)
        command.add_argument("--package-receipt", type=Path, required=True)
        command.add_argument("--receipt-dir", type=Path, required=True)
        command.add_argument("--preflight-config", type=Path, required=True)
        if name == "run":
            command.add_argument("--operation-id", required=True)
            command.add_argument("--execute-reviewed-package-sha256", required=True)
            command.add_argument(
                "--acknowledge-production-drain", action="store_true", required=True
            )
            command.add_argument("--local-payload-inventory", type=Path)
            command.add_argument("--local-payload-receipt", type=Path)
    submit = commands.add_parser("submit-route-evidence", allow_abbrev=False)
    submit.add_argument("--operation-id", required=True)
    submit.add_argument("--sequence", type=int, choices=(1, 2), required=True)
    submit.add_argument("--confirm-ai", action="store_true", required=True)
    submit.add_argument("--confirm-pony", action="store_true", required=True)
    submit.add_argument("--confirm-comfy", action="store_true", required=True)
    return parser


def _canonical_directory(value: Path, expected: Path) -> Path:
    if not value.is_absolute() or value.is_symlink() or not value.is_dir():
        raise CliError("directory_invalid")
    try:
        resolved = value.resolve(strict=True)
        sealed = expected.resolve(strict=True)
    except OSError as exc:
        raise CliError("directory_invalid") from exc
    if resolved != value.absolute() or resolved != sealed:
        raise CliError("directory_not_sealed")
    return resolved


def _real_directory(value: Path) -> Path:
    if not value.is_absolute() or value.is_symlink() or not value.is_dir():
        raise CliError("directory_invalid")
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise CliError("directory_invalid") from exc
    if resolved != value.absolute():
        raise CliError("directory_invalid")
    return resolved


def _owner_only_directory(value: Path, expected: Path) -> Path:
    resolved = _canonical_directory(value, expected)
    if os.name == "posix":
        info = resolved.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise CliError("directory_not_owner_only")
    return resolved


def _local_payload_inventory(value: Path | None, candidate_id: str) -> Path:
    if candidate_id != "qwen38-vllm":
        if value is not None:
            raise CliError("local_payload_inventory_not_allowed")
        raise CliError("local_payload_inventory_candidate_invalid")
    if (
        value is None
        or not value.is_absolute()
        or any(part in {"", ".", ".."} for part in value.parts[1:])
    ):
        raise CliError("local_payload_inventory_invalid")
    # Do not stat, open, resolve, or otherwise trust ingress here. The caller
    # verifies the sealed quarantine before constructing the operation, and the
    # backend repeats that verification under both operation locks.
    return value


def _provision_lock_file(path: Path) -> None:
    """Create or validate one lock inode without treating it as acquired."""

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise CliError("host_lock_path_invalid")
    try:
        parent = path.parent.resolve(strict=True)
        if parent != path.parent.absolute() or not parent.is_dir() or path.parent.is_symlink():
            raise CliError("host_lock_parent_invalid")
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        locked = False
        try:
            opened = os.fstat(descriptor)
            observed = path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino)
                or (os.name == "posix" and opened.st_uid != os.geteuid())
            ):
                raise CliError("host_lock_file_invalid")
            if os.name == "posix":
                if fcntl is None:
                    raise CliError("host_lock_provision_failed")
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            if locked and fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if os.name == "posix":
            directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except CliError:
        raise
    except OSError as exc:
        raise CliError("host_lock_provision_failed") from exc


def _package_identity(
    archive: Path,
    receipt: Path,
    root: Path,
    receipt_root: Path,
    candidate_id: str,
) -> str:
    for path in (archive, receipt):
        if not path.is_absolute() or path.is_symlink():
            raise CliError("package_evidence_invalid")
        try:
            resolved = path.resolve(strict=True)
            info = path.lstat()
        except OSError as exc:
            raise CliError("package_evidence_invalid") from exc
        if (
            resolved.parent != receipt_root
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise CliError("package_evidence_invalid")
    verifier = getattr(candidate_package, "verify_materialized_tree", None)
    if not callable(verifier):
        raise CliError("materialized_tree_verifier_missing")
    try:
        verified = candidate_package.verify(archive, receipt)
        tree = verifier(archive, receipt, root)
    except (candidate_package.PackageError, OSError, ValueError, TypeError) as exc:
        raise CliError("package_verification_failed") from exc
    archive_sha = verified.get("archive_sha256") if isinstance(verified, Mapping) else None
    tree_sha = tree.get("archive_sha256") if isinstance(tree, Mapping) else None
    verified_scope = verified.get("package_scope")
    tree_scope = tree.get("package_scope")
    scoped_match = (
        verified_scope == "candidate"
        and tree_scope == "candidate"
        and verified.get("candidate_id") == candidate_id
        and tree.get("candidate_id") == candidate_id
    )
    full_match = (
        verified_scope is None and tree_scope is None
        and verified.get("candidate_id") is None and tree.get("candidate_id") is None
    )
    if (
        not isinstance(archive_sha, str)
        or SHA256.fullmatch(archive_sha) is None
        or tree_sha != archive_sha
        or not (scoped_match or full_match)
    ):
        raise CliError("package_verification_failed")
    return archive_sha


def _exact_dict(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CliError("preflight_config_invalid")
    return value


def _preflight_config(
    path: Path, receipt_root: Path, candidate_id: str, package_root: Path
) -> collector.CollectorConfig:
    if not path.is_absolute() or path.is_symlink():
        raise CliError("preflight_config_invalid")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise CliError("preflight_config_invalid") from exc
    if resolved.parent != receipt_root or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CliError("preflight_config_invalid")
    if os.name == "posix" and (
        info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise CliError("preflight_config_not_owner_only")
    try:
        if info.st_size <= 0 or info.st_size > 262_144:
            raise CliError("preflight_config_invalid")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError("preflight_config_invalid") from exc
    keys = {
        "schema_version", "candidate_id", "compose_files",
        "expected_systemd_units", "expected_socket_names",
        "required_mem_available_mib", "calculated_candidate_bytes",
        "restore_reserve_bytes", "expected_current_cas_manifest_sha256",
        "expected_rollback_cas_manifest_sha256", "expected_caddy_bytes_sha256",
        "expected_caddy_auth_identity_sha256", "acceptance",
        "acceptance_receipts", "timeout_seconds",
    }
    root = _exact_dict(value, keys)
    if root["schema_version"] != CONFIG_SCHEMA or root["candidate_id"] != candidate_id:
        raise CliError("preflight_config_invalid")
    acceptance_keys = set(preflight.AcceptanceEvidence.__dataclass_fields__)
    acceptance_value = _exact_dict(root["acceptance"], acceptance_keys)
    if any(type(item) is not bool for item in acceptance_value.values()):
        raise CliError("preflight_config_invalid")
    receipt_values = _exact_dict(
        root["acceptance_receipts"], {"A-01", "A-02", "A-03", "A-04", "A-05"}
    )
    receipts: dict[str, preflight.AcceptanceReceipt] = {}
    for gate, item in receipt_values.items():
        parsed = _exact_dict(
            item, {"gate_id", "passed", "evidence_complete", "receipt_sha256"}
        )
        if (
            parsed["gate_id"] != gate
            or type(parsed["passed"]) is not bool
            or type(parsed["evidence_complete"]) is not bool
            or not isinstance(parsed["receipt_sha256"], str)
            or SHA256.fullmatch(parsed["receipt_sha256"]) is None
        ):
            raise CliError("preflight_config_invalid")
        receipts[gate] = preflight.AcceptanceReceipt(**parsed)
    try:
        config = collector.CollectorConfig(
            candidate_id=candidate_id,
            compose_files=tuple(root["compose_files"]),
            expected_systemd_units=tuple(root["expected_systemd_units"]),
            expected_socket_names=tuple(root["expected_socket_names"]),
            required_mem_available_mib=root["required_mem_available_mib"],
            calculated_candidate_bytes=root["calculated_candidate_bytes"],
            restore_reserve_bytes=root["restore_reserve_bytes"],
            expected_current_cas_manifest_sha256=root["expected_current_cas_manifest_sha256"],
            expected_rollback_cas_manifest_sha256=root["expected_rollback_cas_manifest_sha256"],
            expected_caddy_bytes_sha256=root["expected_caddy_bytes_sha256"],
            expected_caddy_auth_identity_sha256=root["expected_caddy_auth_identity_sha256"],
            acceptance=preflight.AcceptanceEvidence(**acceptance_value),
            acceptance_receipts=receipts,
            timeout_seconds=root["timeout_seconds"],
            package_root=str(package_root),
        )
        collector._validate_config(config)
    except (KeyError, TypeError, ValueError, collector.CollectorError) as exc:
        raise CliError("preflight_config_invalid") from exc
    return config


def _emit(**values: object) -> None:
    if set(values) - {
        "operation_id", "package_sha256", "receipt_sha256", "result", "sequence",
        "challenge_sha256", "response_sha256",
    }:
        raise AssertionError("unsafe_output_field")
    if "operation_id" in values and (
        not isinstance(values["operation_id"], str)
        or SAFE_ID.fullmatch(values["operation_id"]) is None
    ):
        raise AssertionError("unsafe_operation_id")
    for key in (
        "package_sha256", "receipt_sha256", "challenge_sha256", "response_sha256",
    ):
        if key in values and values[key] is not None and (
            not isinstance(values[key], str) or SHA256.fullmatch(values[key]) is None
        ):
            raise AssertionError("unsafe_digest")
    if "sequence" in values and values["sequence"] not in {1, 2}:
        raise AssertionError("unsafe_sequence")
    if values.get("result") not in RESULTS:
        raise AssertionError("unsafe_result")
    print(json.dumps(values, sort_keys=True, separators=(",", ":")))


def _signal_failure(signum: int, frame: object) -> None:
    raise LiveOperationError("signal_received", ambiguous=True)


def _execute_with_signals(operation: CandidateLiveOperation) -> dict[str, Any]:
    watched = (signal.SIGINT, signal.SIGTERM)
    previous = {item: signal.getsignal(item) for item in watched}
    try:
        for item in watched:
            signal.signal(item, _signal_failure)
        return operation.execute()
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


BackendFactory = Callable[..., CandidateLiveBackend]
OperationFactory = Callable[..., CandidateLiveOperation]


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: BackendFactory = _ReceiptBackend,
    operation_factory: OperationFactory = CandidateLiveOperation,
) -> int:
    backend: CandidateLiveBackend | None = None
    package_sha: str | None = None
    operation_id: str | None = None
    try:
        args = _parser().parse_args(argv)
        if args.command == "submit-route-evidence":
            submitted = submit_route_evidence(args.operation_id, args.sequence)
            _emit(**submitted, result="route_evidence_submitted")
            return 0
        supplied_package_root = _real_directory(args.package_root)
        local_payload_inventory = (
            _local_payload_inventory(args.local_payload_inventory, args.candidate_id)
            if args.command == "run" and args.candidate_id == "qwen38-vllm"
            else None
        )
        local_payload_receipt = (
            _local_payload_inventory(args.local_payload_receipt, args.candidate_id)
            if args.command == "run" and args.candidate_id == "qwen38-vllm"
            else None
        )
        if (
            args.command == "run"
            and args.candidate_id != "qwen38-vllm"
            and (args.local_payload_inventory is not None or args.local_payload_receipt is not None)
        ):
            raise CliError("local_payload_inventory_not_allowed")
        supplied_receipt_root = _real_directory(args.receipt_dir)
        package_sha = _package_identity(
            args.package_archive,
            args.package_receipt,
            supplied_package_root,
            supplied_receipt_root,
            args.candidate_id,
        )
        package_root = _canonical_directory(
            args.package_root, SEALED_RELEASES_ROOT / package_sha
        )
        receipt_root = _owner_only_directory(
            args.receipt_dir, SEALED_PACKAGES_ROOT / package_sha
        )
        package_sha = _package_identity(
            args.package_archive,
            args.package_receipt,
            package_root,
            receipt_root,
            args.candidate_id,
        )
        config = _preflight_config(
            args.preflight_config, receipt_root, args.candidate_id, package_root
        )
        if args.command == "verify":
            _emit(package_sha256=package_sha, result="offline_verified")
            return 0
        quarantine_inventory_sha256: str | None = None
        if args.candidate_id == "qwen38-vllm":
            assert local_payload_inventory is not None and local_payload_receipt is not None
            try:
                quarantine_result = payload_quarantine.verify(
                    local_payload_inventory,
                    local_payload_receipt,
                    candidate_id=args.candidate_id,
                    candidate_pin=PINS[args.candidate_id],
                )
            except (payload_quarantine.QuarantineError, OSError, ValueError, TypeError) as exc:
                raise CliError("local_payload_quarantine_invalid") from exc
            quarantine_inventory_sha256 = quarantine_result.get("inventory_sha256")  # type: ignore[assignment]
            if not isinstance(quarantine_inventory_sha256, str) or SHA256.fullmatch(quarantine_inventory_sha256) is None:
                raise CliError("local_payload_quarantine_invalid")
        operation_id = args.operation_id
        if SAFE_ID.fullmatch(operation_id) is None:
            raise CliError("operation_id_invalid")
        reviewed_sha = args.execute_reviewed_package_sha256
        if SHA256.fullmatch(reviewed_sha) is None or reviewed_sha != package_sha:
            raise CliError("reviewed_package_mismatch")
        backend = backend_factory(
            operation_id,
            args.candidate_id,
            PINS[args.candidate_id],
            preflight_config=config,
            preflight_runner=collector.SubprocessCommandRunner(package_root),
            code_root=package_root,
            package_root=package_root,
            package_archive=args.package_archive,
            package_receipt=args.package_receipt,
            receipt_root=receipt_root,
            reviewed_package_sha256=reviewed_sha,
            package_candidate_id=args.candidate_id,
            local_payload_inventory=local_payload_inventory,
            payload_quarantine_receipt=local_payload_receipt,
            payload_quarantine_inventory_sha256=quarantine_inventory_sha256,
            authenticated_route_probe=BrowserRouteEvidenceProbe(operation_id),
        )
        operation = operation_factory(
            operation_id, args.candidate_id, PINS[args.candidate_id], backend
        )
        result = _execute_with_signals(operation)
        receipt_sha = getattr(backend, "last_receipt_sha256", None)
        if not isinstance(receipt_sha, str) or SHA256.fullmatch(receipt_sha) is None:
            raise CliError("receipt_identifier_missing")
        _emit(
            operation_id=operation_id,
            package_sha256=package_sha,
            receipt_sha256=receipt_sha,
            result=result["state"],
        )
        return 0
    except BaseException:
        if operation_id is not None and package_sha is not None:
            receipt_sha = getattr(backend, "last_receipt_sha256", None)
            result = getattr(backend, "last_receipt_state", None)
            if result not in RESULTS:
                result = "failed_without_receipt"
            _emit(
                operation_id=operation_id,
                package_sha256=package_sha,
                receipt_sha256=(receipt_sha if isinstance(receipt_sha, str) else None),
                result=result,
            )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
