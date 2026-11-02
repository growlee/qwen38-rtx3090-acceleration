"""Fail-closed state machine for one reviewed acceleration candidate.

The module owns ordering and recovery policy only.  Host operations are supplied
through :class:`OperationBackend`; this keeps arbitrary commands out of the
state machine and makes the exact live adapter part of the reviewed package.
"""

from __future__ import annotations

import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PREPARE_TIMEOUT_SECONDS = 86_400
LIVE_TEST_TIMEOUT_SECONDS = 1_800
RECOVERY_RESERVE_SECONDS = 600
NORMALIZATION_RECONCILIATION_RESERVE_SECONDS = 30
NORMALIZATION_FENCE_RESERVE_SECONDS = 90


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    arms: tuple[str, ...]


CANDIDATE_PLANS = {
    "qwen38-vllm": CandidatePlan("qwen38-vllm", ("mtp", "dflash2")),
    "freetoken": CandidatePlan("freetoken", ("freetoken",)),
    "llama-dflash2-official": CandidatePlan(
        "llama-dflash2-official",
        ("plain", "mtp", "dflash2-n4", "dflash2-n7"),
    ),
    "higgs-tts3": CandidatePlan("higgs-tts3", ("tts",)),
}


class LiveOperationError(RuntimeError):
    """A bounded, content-free failure safe to include in evidence."""

    def __init__(self, code: str, *, ambiguous: bool = False) -> None:
        if re.fullmatch(r"[a-z0-9_]+", code) is None:
            raise ValueError("error_code_invalid")
        self.code = code
        self.ambiguous = ambiguous
        super().__init__(code)


class OperationBackend(Protocol):
    def provision_host_lock(self) -> None: ...

    def validate_candidate_ingress(
        self, candidate_id: str, candidate_pin: str,
    ) -> str | None: ...

    def host_lock(self, operation_id: str) -> AbstractContextManager[None]: ...

    def candidate_lock(
        self, candidate_id: str, candidate_pin: str
    ) -> AbstractContextManager[None]: ...

    def assert_no_stale_state(self, operation_id: str) -> None: ...

    def run_preflight(self, candidate_id: str, candidate_pin: str) -> str: ...

    def begin_stage(self, stage: str, timeout_seconds: int) -> None: ...

    def ensure_production_qwen_ready(self) -> None: ...

    def collect_route_evidence(self, expected_sequence: int) -> None: ...

    def capture_prestate(self, operation_id: str) -> Mapping[str, Any]: ...

    def create_journal(self, operation_id: str, candidate_id: str) -> None: ...

    def persist_prestate(
        self, operation_id: str, prestate: Mapping[str, Any]
    ) -> None: ...

    def prepare_candidate(self, candidate_id: str, candidate_pin: str) -> str: ...

    def record_phase(self, operation_id: str, phase: str) -> None: ...

    def close_admission(self) -> None: ...

    def wait_fifo_drained(self) -> None: ...

    def stop_production_qwen(self, operation_id: str) -> None: ...

    def assert_gpu_idle(self) -> None: ...

    def run_prestart(self, candidate_id: str, arm: str) -> str | None: ...

    def ensure_prestart_stopped(self, candidate_id: str, arm: str) -> None: ...

    def start_arm(self, candidate_id: str, arm: str) -> None: ...

    def wait_arm_ready(self, candidate_id: str, arm: str) -> None: ...

    def run_arm(self, candidate_id: str, arm: str) -> Mapping[str, Any]: ...

    def stop_arm(self, candidate_id: str, arm: str) -> None: ...

    def ensure_arm_stopped(self, candidate_id: str, arm: str) -> None: ...

    def restore_production(self, prestate: Mapping[str, Any]) -> None: ...

    def reconcile_exact_restore(self, prestate: Mapping[str, Any]) -> None: ...

    def verify_exact_restore(self, prestate: Mapping[str, Any]) -> str: ...

    def production_qwen_completion(self, ordinal: int) -> None: ...

    def smoke_admission(self) -> None: ...

    def open_admission(self) -> None: ...

    def ensure_admission_closed(
        self, safety_reserve_seconds: int | None = None
    ) -> None: ...

    def write_receipt(self, operation_id: str, receipt: Mapping[str, Any]) -> str: ...


def _validated_arm_result(
    value: Mapping[str, Any], candidate_id: str, arm: str
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "candidate_id",
        "arm",
        "passed",
        "report_sha256",
        "quality_status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LiveOperationError("arm_result_invalid")
    if (
        value["schema_version"] != "ai-server-acceleration-arm-result/v1"
        or value["candidate_id"] != candidate_id
        or value["arm"] != arm
        or type(value["passed"]) is not bool
        or not isinstance(value["report_sha256"], str)
        or SHA256.fullmatch(value["report_sha256"]) is None
        or value["quality_status"] not in {"passed", "failed", "not_evaluated"}
    ):
        raise LiveOperationError("arm_result_invalid")
    return dict(value)


def _failure_code(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    if isinstance(exc, LiveOperationError):
        return exc.code
    return "operation_error"


class CandidateLiveOperation:
    """Execute one candidate exactly once and restore the accepted Qwen."""

    def __init__(
        self,
        operation_id: str,
        candidate_id: str,
        candidate_pin: str,
        backend: OperationBackend,
    ) -> None:
        if SAFE_ID.fullmatch(operation_id) is None:
            raise ValueError("operation_id_invalid")
        if candidate_id not in CANDIDATE_PLANS:
            raise ValueError("candidate_id_invalid")
        if SAFE_ID.fullmatch(candidate_pin) is None:
            raise ValueError("candidate_pin_invalid")
        self.operation_id = operation_id
        self.plan = CANDIDATE_PLANS[candidate_id]
        self.candidate_pin = candidate_pin
        self.backend = backend

    def execute(self) -> dict[str, Any]:
        arm_results: list[dict[str, Any]] = []
        pending: BaseException | None = None
        recovery: BaseException | None = None
        restored_sha256: str | None = None
        inventory_sha256: str | None = None
        preflight_sha256: str | None = None
        prestate: Mapping[str, Any] | None = None
        production_mutation_started = False
        journal_created = False
        prestart_arm: str | None = None
        current_arm: str | None = None
        restore_attempted = False
        restore_verification_attempted = False
        restore_proven = False
        qwen_verified = False
        qwen_verification_attempted = False
        admission_closed_proven = False
        admission_fence_attempted = False
        admission_open_uncertain = False
        admission_public_attempted = False
        admission_public_proven = False
        preproduction_unknown = False

        # The host collector has to observe the lock while no operation owns it.
        # Provisioning creates or validates only the lock inode; it never treats
        # mutable candidate ingress as trusted and never acquires the lock.
        self.backend.provision_host_lock()
        # Its immutable receipt is then bound into this operation before either
        # lock, any candidate download, or any production mutation can occur.
        # There is deliberately no retry if this single preflight fails.
        preflight_sha256 = self.backend.run_preflight(
            self.plan.candidate_id, self.candidate_pin
        )
        if SHA256.fullmatch(preflight_sha256) is None:
            raise LiveOperationError("preflight_receipt_invalid")

        with self.backend.host_lock(self.operation_id):
            with self.backend.candidate_lock(
                self.plan.candidate_id, self.candidate_pin
            ):
                try:
                    ingress_sha256 = self.backend.validate_candidate_ingress(
                        self.plan.candidate_id, self.candidate_pin
                    )
                    if self.plan.candidate_id == "qwen38-vllm":
                        if not isinstance(ingress_sha256, str) or SHA256.fullmatch(ingress_sha256) is None:
                            raise LiveOperationError("candidate_ingress_invalid")
                    elif ingress_sha256 is not None:
                        raise LiveOperationError("candidate_ingress_invalid")
                    self.backend.assert_no_stale_state(self.operation_id)
                    self.backend.create_journal(
                        self.operation_id, self.plan.candidate_id
                    )
                    journal_created = True
                    self.backend.begin_stage("prepare", PREPARE_TIMEOUT_SECONDS)
                    inventory_sha256 = self.backend.prepare_candidate(
                        self.plan.candidate_id, self.candidate_pin
                    )
                    if SHA256.fullmatch(inventory_sha256) is None:
                        raise LiveOperationError("candidate_inventory_invalid")
                    self.backend.record_phase(self.operation_id, "candidate-prepared")

                    # Preparation may legitimately outlive the normal Qwen idle
                    # transition.  Normalize it through the reviewed supervisor
                    # path, prove the three public routes, and only then freeze
                    # the mutable restore target exactly once.
                    try:
                        self.backend.ensure_production_qwen_ready()
                    except LiveOperationError as exc:
                        if (
                            exc.code == "production_qwen_ready_unknown"
                            and exc.ambiguous
                        ):
                            preproduction_unknown = True
                        raise
                    self.backend.collect_route_evidence(1)
                    prestate = self.backend.capture_prestate(self.operation_id)
                    self.backend.persist_prestate(self.operation_id, prestate)
                    self.backend.record_phase(self.operation_id, "prestate-recorded")
                    self.backend.begin_stage("live", LIVE_TEST_TIMEOUT_SECONDS)

                    production_mutation_started = True
                    self.backend.close_admission()
                    admission_closed_proven = True
                    self.backend.record_phase(self.operation_id, "admission-closed")
                    self.backend.wait_fifo_drained()
                    self.backend.record_phase(self.operation_id, "fifo-drained")
                    self.backend.stop_production_qwen(self.operation_id)
                    self.backend.assert_gpu_idle()
                    self.backend.record_phase(self.operation_id, "production-stopped")

                    for arm in self.plan.arms:
                        prestart_arm = arm
                        prestart_sha256 = self.backend.run_prestart(
                            self.plan.candidate_id, arm
                        )
                        if self.plan.candidate_id == "freetoken":
                            if (
                                not isinstance(prestart_sha256, str)
                                or SHA256.fullmatch(prestart_sha256) is None
                            ):
                                raise LiveOperationError("candidate_prestart_result_invalid")
                        elif prestart_sha256 is not None:
                            raise LiveOperationError("candidate_prestart_result_invalid")
                        prestart_arm = None
                        current_arm = arm
                        self.backend.start_arm(self.plan.candidate_id, arm)
                        self.backend.record_phase(self.operation_id, f"arm-{arm}-started")
                        self.backend.wait_arm_ready(self.plan.candidate_id, arm)
                        result = _validated_arm_result(
                            self.backend.run_arm(self.plan.candidate_id, arm),
                            self.plan.candidate_id,
                            arm,
                        )
                        arm_results.append(result)
                        self.backend.stop_arm(self.plan.candidate_id, arm)
                        current_arm = None
                        self.backend.assert_gpu_idle()
                        self.backend.record_phase(self.operation_id, f"arm-{arm}-stopped")
                        if not result["passed"]:
                            raise LiveOperationError("arm_rejected")

                    restore_attempted = True
                    self.backend.restore_production(prestate)
                    restore_verification_attempted = True
                    restored_sha256 = self.backend.verify_exact_restore(prestate)
                    if SHA256.fullmatch(restored_sha256) is None:
                        raise LiveOperationError("restore_receipt_invalid", ambiguous=True)
                    restore_proven = True
                    self.backend.record_phase(self.operation_id, "production-restored")
                    qwen_verification_attempted = True
                    admission_open_uncertain = True
                    admission_closed_proven = False
                    self.backend.smoke_admission()
                    self.backend.record_phase(self.operation_id, "admission-smoke")
                    self.backend.production_qwen_completion(1)
                    self.backend.production_qwen_completion(2)
                    self.backend.record_phase(self.operation_id, "qwen-verified")
                    qwen_verified = True
                    admission_public_attempted = True
                    self.backend.open_admission()
                    self.backend.record_phase(self.operation_id, "admission-open")
                    admission_public_proven = True
                    admission_open_uncertain = False
                except BaseException as exc:
                    pending = exc

                if (
                    pending is not None
                    and production_mutation_started
                    and prestate is not None
                ):
                    try:
                        # Exactly one bounded recovery attempt.  There is no retry loop.
                        self.backend.begin_stage("recovery", RECOVERY_RESERVE_SECONDS)
                        if admission_open_uncertain:
                            admission_fence_attempted = True
                            self.backend.ensure_admission_closed(
                                NORMALIZATION_FENCE_RESERVE_SECONDS
                            )
                            admission_closed_proven = True
                            admission_open_uncertain = False
                        if prestart_arm is not None:
                            # A prestart can own a separate FreeToken GPU
                            # container before current_arm is assigned.  Its
                            # exact pinned identity must be terminal/absent
                            # before an idle check or production restore.
                            self.backend.ensure_prestart_stopped(
                                self.plan.candidate_id, prestart_arm
                            )
                            prestart_arm = None
                        if current_arm is not None:
                            # stop_arm may have completed before its result became
                            # unknowable.  The backend first inspects the exact arm
                            # identity and GPU state and only then conditionally
                            # performs one compensating stop.
                            self.backend.ensure_arm_stopped(
                                self.plan.candidate_id, current_arm
                            )
                            current_arm = None
                        if not restore_proven:
                            if restore_attempted:
                                # A restore command may have committed before its
                                # acknowledgement was lost.  Never repeat that
                                # production mutation.  Continue only if an
                                # independent read-only inspection proves the
                                # exact accepted production state already exists.
                                self.backend.reconcile_exact_restore(prestate)
                            else:
                                self.backend.assert_gpu_idle()
                                restore_attempted = True
                                try:
                                    self.backend.restore_production(prestate)
                                except LiveOperationError as exc:
                                    if not exc.ambiguous:
                                        raise
                                    # This is still the first and only restore
                                    # mutation.  Resolve a lost acknowledgement
                                    # by inspection; never issue another start.
                                    self.backend.reconcile_exact_restore(prestate)
                            if restore_verification_attempted:
                                # The exact-restore verifier writes immutable route
                                # evidence and a restore receipt.  If its outcome is
                                # already unknown, repeating it could overwrite the
                                # evidence boundary or demand a third browser proof.
                                raise LiveOperationError(
                                    "restore_verification_unknown", ambiguous=True
                                )
                            restore_verification_attempted = True
                            restored_sha256 = self.backend.verify_exact_restore(prestate)
                            if SHA256.fullmatch(restored_sha256) is None:
                                raise LiveOperationError(
                                    "restore_receipt_invalid", ambiguous=True
                                )
                            restore_proven = True
                            self.backend.record_phase(
                                self.operation_id, "production-restored-after-failure"
                            )
                        if not qwen_verified and not qwen_verification_attempted:
                            qwen_verification_attempted = True
                            admission_open_uncertain = True
                            admission_closed_proven = False
                            self.backend.smoke_admission()
                            self.backend.record_phase(
                                self.operation_id, "admission-smoke-after-failure"
                            )
                            self.backend.production_qwen_completion(1)
                            self.backend.production_qwen_completion(2)
                            self.backend.record_phase(
                                self.operation_id, "qwen-verified-after-failure"
                            )
                            qwen_verified = True
                        if not (
                            isinstance(pending, LiveOperationError)
                            and pending.ambiguous
                        ) and qwen_verified and not admission_public_attempted:
                            admission_open_uncertain = True
                            admission_closed_proven = False
                            admission_public_attempted = True
                            self.backend.open_admission()
                            self.backend.record_phase(
                                self.operation_id, "admission-open-after-failure"
                            )
                            admission_public_proven = True
                            admission_open_uncertain = False
                    except BaseException as exc:
                        recovery = exc

                unknown_outcome = (
                    recovery is not None
                    or admission_open_uncertain
                    or (
                        admission_public_attempted
                        and not admission_public_proven
                    )
                    or preproduction_unknown
                    or (
                        production_mutation_started
                        and isinstance(pending, LiveOperationError)
                        and pending.ambiguous
                    )
                )
                if (
                    unknown_outcome
                    and (production_mutation_started or preproduction_unknown)
                    and not admission_closed_proven
                    and not admission_fence_attempted
                ):
                    try:
                        admission_fence_attempted = True
                        self.backend.ensure_admission_closed(
                            NORMALIZATION_FENCE_RESERVE_SECONDS
                        )
                        admission_closed_proven = True
                    except BaseException as exc:
                        if recovery is None:
                            recovery = exc

                if pending is not None or recovery is not None:
                    if not journal_created:
                        if isinstance(pending, BaseException):
                            raise pending
                        if isinstance(recovery, BaseException):
                            raise recovery
                        raise AssertionError("failure_without_exception")
                    receipt = {
                        "schema_version": "ai-server-acceleration-operation/v1",
                        "operation_id": self.operation_id,
                        "candidate_id": self.plan.candidate_id,
                        "candidate_pin": self.candidate_pin,
                        "preflight_receipt_sha256": preflight_sha256,
                        "state": (
                            "unknown_outcome"
                            if unknown_outcome or recovery is not None
                            else (
                                "failed_restored"
                                if production_mutation_started
                                else "failed_preproduction"
                            )
                        ),
                        "passed": False,
                        "original_error_code": _failure_code(pending),
                        "recovery_error_code": _failure_code(recovery),
                        "candidate_inventory_sha256": inventory_sha256,
                        "restored_receipt_sha256": restored_sha256,
                        "arm_results": arm_results,
                        "content_recorded": False,
                    }
                    try:
                        receipt_sha256 = self.backend.write_receipt(
                            self.operation_id, receipt
                        )
                        if SHA256.fullmatch(receipt_sha256) is None:
                            raise LiveOperationError(
                                "operation_receipt_invalid", ambiguous=True
                            )
                    except BaseException as exc:
                        if (
                            not admission_closed_proven
                            and not preproduction_unknown
                            and not admission_fence_attempted
                        ):
                            admission_fence_attempted = True
                            self.backend.ensure_admission_closed(
                                NORMALIZATION_FENCE_RESERVE_SECONDS
                            )
                            admission_closed_proven = True
                        raise LiveOperationError(
                            "operation_receipt_unknown", ambiguous=True
                        ) from exc
                    if recovery is not None:
                        raise LiveOperationError(
                            "recovery_failed", ambiguous=True
                        ) from recovery
                    if isinstance(pending, BaseException):
                        raise pending
                    raise AssertionError("failure_without_exception")

                receipt = {
                    "schema_version": "ai-server-acceleration-operation/v1",
                    "operation_id": self.operation_id,
                    "candidate_id": self.plan.candidate_id,
                    "candidate_pin": self.candidate_pin,
                    "preflight_receipt_sha256": preflight_sha256,
                    "state": "passed_restored",
                    "passed": True,
                    "original_error_code": None,
                    "recovery_error_code": None,
                    "candidate_inventory_sha256": inventory_sha256,
                    "restored_receipt_sha256": restored_sha256,
                    "arm_results": arm_results,
                    "content_recorded": False,
                }
                try:
                    receipt_sha256 = self.backend.write_receipt(
                        self.operation_id, receipt
                    )
                    if SHA256.fullmatch(receipt_sha256) is None:
                        raise LiveOperationError(
                            "operation_receipt_invalid", ambiguous=True
                        )
                except BaseException as exc:
                    # The receipt may have reached stable storage before its
                    # acknowledgement was lost.  Do not rewrite it.  Re-fence
                    # admission and surface an unknown terminal outcome.
                    if not admission_closed_proven and not admission_fence_attempted:
                        admission_fence_attempted = True
                        self.backend.ensure_admission_closed(
                            NORMALIZATION_FENCE_RESERVE_SECONDS
                        )
                        admission_closed_proven = True
                    raise LiveOperationError(
                        "operation_receipt_unknown", ambiguous=True
                    ) from exc
                return receipt
