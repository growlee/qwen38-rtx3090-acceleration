"""Pure, fail-closed physical storage budgeting for one experiment candidate."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIB = 1024**3
DEFAULT_ROOT_FREE_RESERVE_BYTES = 150 * GIB


class StorageBudgetError(ValueError):
    """Raised when sealed storage evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ReceiptBinding:
    """Exact binding between a sealed receipt and the observed component."""

    sealed_path: str
    observed_path: str
    sealed_sha256: str
    observed_sha256: str


@dataclass(frozen=True)
class PhysicalCopyReceipt:
    """A physical copy that coexists at candidate peak occupancy."""

    binding: ReceiptBinding
    physical_bytes: object = None


@dataclass(frozen=True)
class NewArtifactReceipt:
    """A new artifact whose partial is atomically renamed to its final path."""

    binding: ReceiptBinding
    partial_physical_bytes: object = None
    final_physical_bytes: object = None


@dataclass(frozen=True)
class DockerLayerReceipt:
    """A Docker layer budgeted for both content-store and unpacked occupancy."""

    binding: ReceiptBinding
    unpacked_physical_bytes: object = None
    compressed_oci_bytes: object = None
    compressed_binding: ReceiptBinding | None = None


@dataclass(frozen=True)
class CandidateStorageInputs:
    immutable_cas_blobs: tuple[PhysicalCopyReceipt, ...] = ()
    new_artifacts: tuple[NewArtifactReceipt, ...] = ()
    quarantine_copies: tuple[PhysicalCopyReceipt, ...] = ()
    writable_preparation: tuple[PhysicalCopyReceipt, ...] = ()
    required_upstream_backups: tuple[PhysicalCopyReceipt, ...] = ()
    docker_layers: tuple[DockerLayerReceipt, ...] = ()
    restoration_reserve_bytes: object = None


@dataclass(frozen=True)
class CandidateStorageBreakdown:
    immutable_cas_bytes: int
    new_artifact_stage_bytes: int
    quarantine_overlap_bytes: int
    writable_preparation_bytes: int
    required_upstream_backup_bytes: int
    image_content_layer_bytes: int
    image_unpacked_layer_bytes: int
    restoration_reserve_bytes: int
    required_bytes_before_mutation: int


def _size(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StorageBudgetError(f"{field} must be an integer byte count")
    if value < 0:
        raise StorageBudgetError(f"{field} must not be negative")
    return value


def _validate_binding(binding: object, field: str) -> ReceiptBinding:
    if not isinstance(binding, ReceiptBinding):
        raise StorageBudgetError(f"{field}.binding must be a ReceiptBinding")
    if (
        not isinstance(binding.sealed_path, str)
        or not isinstance(binding.observed_path, str)
        or not binding.sealed_path
        or not binding.observed_path
    ):
        raise StorageBudgetError(f"{field} paths must be non-empty strings")
    if binding.sealed_path != binding.observed_path:
        raise StorageBudgetError(f"{field} path does not match its sealed receipt")
    if not isinstance(binding.sealed_sha256, str) or not _SHA256_RE.fullmatch(
        binding.sealed_sha256
    ):
        raise StorageBudgetError(f"{field}.sealed_sha256 must be lowercase sha256")
    if not isinstance(binding.observed_sha256, str) or not _SHA256_RE.fullmatch(
        binding.observed_sha256
    ):
        raise StorageBudgetError(f"{field}.observed_sha256 must be lowercase sha256")
    if binding.sealed_sha256 != binding.observed_sha256:
        raise StorageBudgetError(f"{field} digest does not match its sealed receipt")
    return binding


def _sum_copies(receipts: Iterable[PhysicalCopyReceipt], field: str) -> int:
    total = 0
    for index, receipt in enumerate(receipts):
        item = f"{field}[{index}]"
        if not isinstance(receipt, PhysicalCopyReceipt):
            raise StorageBudgetError(f"{item} must be a PhysicalCopyReceipt")
        _validate_binding(receipt.binding, item)
        total += _size(receipt.physical_bytes, f"{item}.physical_bytes")
    return total


def _sum_deduplicated_cas(receipts: Iterable[PhysicalCopyReceipt]) -> int:
    by_physical_identity: dict[tuple[str, str], int] = {}
    for index, receipt in enumerate(receipts):
        item = f"immutable_cas_blobs[{index}]"
        if not isinstance(receipt, PhysicalCopyReceipt):
            raise StorageBudgetError(f"{item} must be a PhysicalCopyReceipt")
        binding = _validate_binding(receipt.binding, item)
        size = _size(receipt.physical_bytes, f"{item}.physical_bytes")
        key = (binding.sealed_path, binding.sealed_sha256)
        previous = by_physical_identity.setdefault(key, size)
        if previous != size:
            raise StorageBudgetError(
                "immutable CAS physical identity "
                f"{binding.sealed_path}:{binding.sealed_sha256} "
                "has conflicting physical sizes"
            )
    return sum(by_physical_identity.values())


def _sum_new_artifacts(receipts: Iterable[NewArtifactReceipt]) -> int:
    total = 0
    for index, receipt in enumerate(receipts):
        item = f"new_artifacts[{index}]"
        if not isinstance(receipt, NewArtifactReceipt):
            raise StorageBudgetError(f"{item} must be a NewArtifactReceipt")
        _validate_binding(receipt.binding, item)
        partial = _size(
            receipt.partial_physical_bytes, f"{item}.partial_physical_bytes"
        )
        final = _size(receipt.final_physical_bytes, f"{item}.final_physical_bytes")
        total += max(partial, final)
    return total


def _sum_docker_layers(receipts: Iterable[DockerLayerReceipt]) -> tuple[int, int]:
    unpacked_by_physical_identity: dict[tuple[str, str], int] = {}
    content_by_physical_identity: dict[tuple[str, str], int] = {}
    for index, receipt in enumerate(receipts):
        item = f"docker_layers[{index}]"
        if not isinstance(receipt, DockerLayerReceipt):
            raise StorageBudgetError(f"{item} must be a DockerLayerReceipt")
        binding = _validate_binding(receipt.binding, item)
        if receipt.compressed_oci_bytes is None or receipt.compressed_binding is None:
            raise StorageBudgetError(
                f"{item} requires sealed compressed content-store evidence"
            )
        content_binding = _validate_binding(
            receipt.compressed_binding, f"{item}.compressed_binding"
        )
        compressed = _size(
            receipt.compressed_oci_bytes, f"{item}.compressed_oci_bytes"
        )
        unpacked = _size(
            receipt.unpacked_physical_bytes, f"{item}.unpacked_physical_bytes"
        )
        unpacked_key = (binding.sealed_path, binding.sealed_sha256)
        previous = unpacked_by_physical_identity.setdefault(unpacked_key, unpacked)
        if previous != unpacked:
            raise StorageBudgetError(
                "Docker layer physical identity "
                f"{binding.sealed_path}:{binding.sealed_sha256} "
                "has conflicting physical sizes"
            )
        content_key = (
            content_binding.sealed_path,
            content_binding.sealed_sha256,
        )
        previous_content = content_by_physical_identity.setdefault(
            content_key, compressed
        )
        if previous_content != compressed:
            raise StorageBudgetError(
                "Docker content layer physical identity "
                f"{content_binding.sealed_path}:{content_binding.sealed_sha256} "
                "has conflicting physical sizes"
            )
    return (
        sum(unpacked_by_physical_identity.values()),
        sum(content_by_physical_identity.values()),
    )


def calculate_candidate_storage_budget(
    inputs: CandidateStorageInputs,
) -> CandidateStorageBreakdown:
    """Calculate peak physical bytes solely from validated component receipts."""

    if not isinstance(inputs, CandidateStorageInputs):
        raise StorageBudgetError("inputs must be CandidateStorageInputs")
    for field in (
        "immutable_cas_blobs",
        "new_artifacts",
        "quarantine_copies",
        "writable_preparation",
        "required_upstream_backups",
        "docker_layers",
    ):
        if not isinstance(getattr(inputs, field), tuple):
            raise StorageBudgetError(f"{field} must be a tuple of sealed receipts")

    immutable_cas = _sum_deduplicated_cas(inputs.immutable_cas_blobs)
    new_artifacts = _sum_new_artifacts(inputs.new_artifacts)
    quarantine = _sum_copies(inputs.quarantine_copies, "quarantine_copies")
    writable = _sum_copies(inputs.writable_preparation, "writable_preparation")
    backups = _sum_copies(
        inputs.required_upstream_backups, "required_upstream_backups"
    )
    unpacked_layers, content_layers = _sum_docker_layers(inputs.docker_layers)
    reserve = _size(inputs.restoration_reserve_bytes, "restoration_reserve_bytes")
    required = (
        immutable_cas
        + new_artifacts
        + quarantine
        + writable
        + backups
        + content_layers
        + unpacked_layers
        + reserve
    )
    return CandidateStorageBreakdown(
        immutable_cas_bytes=immutable_cas,
        new_artifact_stage_bytes=new_artifacts,
        quarantine_overlap_bytes=quarantine,
        writable_preparation_bytes=writable,
        required_upstream_backup_bytes=backups,
        image_content_layer_bytes=content_layers,
        image_unpacked_layer_bytes=unpacked_layers,
        restoration_reserve_bytes=reserve,
        required_bytes_before_mutation=required,
    )
