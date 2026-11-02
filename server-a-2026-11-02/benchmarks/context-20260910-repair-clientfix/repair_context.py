"""One-shot GGUF completion plus one distinct repaired DFlash profile.

This adapter imports the SHA-pinned r1/r2 controls.  It does not download,
prepare, copy, delete, promote, or retry a model or response.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
R1 = Path("/srv/ai-server/benchmarks/context-20260908-r1")
R2 = Path("/srv/ai-server/benchmarks/context-20260908-r2")
OLD = Path("/srv/ai-server/benchmarks/quality-remaining-20260908")
PREVIOUS = Path("/srv/ai-server/benchmarks/context-20260910-repair")
R1_MANIFEST_SHA = "c71c26c5cf3124347a021489d60148b102a5ad2c35ee4b1314e37ef8eafa36e6"
R1_RUN_SHA = "7cb2f97465a69992babc9887780477ab98d3eb548500e33052cba7eb70248816"
R1_CONTROLLER_SHA = "4ad216fc2ff90ea9a8971e10904681fd11396f86977bc3e865ae736cf7dcccc5"
R2_MANIFEST_SHA = "89e8667d4c2031b6ce91fe0f73b206592ea9a6f4066b2e9328b4d570e209299e"
R2_CONTINUATION_SHA = "6d2e89059388a5fef595cfc80f76a070d46f21e3a8865c38054b33691c6a5307"
R2_STORAGE_SHA = "2085436ed77ec5a9f64d61568bb612254cb58158027869fc036fa666ff914cfd"
R2_REMAINING_SHA = "d05b4125873eb7afdea27e64f47515c2ff2371c85771ec60fbaee5b1bc6e3979"
R2_COMBINED_SHA = "777d45c1eb2e495cc33cf1d9a031fa0eb5100dc69f6e2401aca87dc3de2a4af9"
R2_RESTORATION_SHA = "5c4db227957e37a03fb36cddd61d9d5eef4be0a1766ce9969a0a7fe4eb3328a0"
R2_GRADING_SUMMARY_SHA = "99f61e0b3b0746d2d69d57435e5281fd95bf4d3d0c80c6b46b44d28638488641"
PREVIOUS_ATTEMPT_SHA = "94fd433d011c4f655abf660ddf5ddfd14103dba3e0197f1c39ff6b81da10a04c"
PREVIOUS_REPORT_SHA = "64a362072a70b563cf88e89cf26c1a1c7eb3716d39de639e85243e7dffb912c8"
PREVIOUS_STDERR_SHA = "c6b626cee5a2f330fa5c039967543e8127fda0890480278d9039c1828359e89f"
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FIXTURE_SHA = "2df1be2cff66b53ce57b3eefe3eed8be2d16c59d1e37b783341f176995689d94"
PREFIX = "qwen-context-20260910-repair-clientfix"
REPAIRED = "dflash2-huge-n7-kv4gb-204800"
SOURCE_DFLASH = "dflash2-huge-n7"
FILES = {"repair_context.py", "quality_context_continue_client.py", "grade_repair.py",
         "selection.json", "runbook.md"}
HEX = re.compile(r"[0-9a-f]{64}")


def need(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def check_manifest(expected):
    need(HEX.fullmatch(expected or "") and sha(ROOT/"manifest.json") == expected,
         "repair_manifest_changed")
    manifest = json.loads((ROOT/"manifest.json").read_bytes())
    need(manifest.get("schema") == "qwen-context-repair-bundle/v1"
         and set(manifest.get("files", {})) == FILES, "repair_manifest_file_set_changed")
    for name, digest in manifest["files"].items():
        need(Path(name).name == name and sha(ROOT/name) == digest,
             "repair_source_changed:"+name)
    dependencies = {
        "r1/run_context.py": (R1/"run_context.py", R1_RUN_SHA),
        "r1/controller.py": (OLD/"run_quality_window_remaining.py", R1_CONTROLLER_SHA),
        "r1/frozen/frozen.json": (R1/"frozen/frozen.json", FIXTURE_SHA),
        "r2/manifest.json": (R2/"manifest.json", R2_MANIFEST_SHA),
        "r2/continue_context.py": (R2/"continue_context.py", R2_CONTINUATION_SHA),
        "r2/context_storage.py": (R2/"context_storage.py", R2_STORAGE_SHA),
        "r2/quality_context_remaining.py": (R2/"quality_context_remaining.py", R2_REMAINING_SHA),
        "r2/results/combined-scores.jsonl": (R2/"results/combined-scores.jsonl", R2_COMBINED_SHA),
        "r2/restoration-after-abort.json": (R2/"restoration-after-abort.json", R2_RESTORATION_SHA),
        "r2/results/grading-summary.json": (R2/"results/grading-summary.json", R2_GRADING_SUMMARY_SHA),
        "previous/attempt.json": (PREVIOUS/"attempt.json", PREVIOUS_ATTEMPT_SHA),
        "previous/results/context-window-report.json":
            (PREVIOUS/"results/context-window-report.json", PREVIOUS_REPORT_SHA),
        "previous/results/original-gguf.jsonl":
            (PREVIOUS/"results/original-gguf.jsonl", EMPTY_SHA),
        "previous/results/original-gguf.stderr":
            (PREVIOUS/"results/original-gguf.stderr", PREVIOUS_STDERR_SHA),
    }
    need(set(manifest.get("dependencies", {})) == set(dependencies),
         "repair_dependency_set_changed")
    for name, (path, digest) in dependencies.items():
        need(manifest["dependencies"][name] == digest and sha(path) == digest,
             "repair_dependency_changed:"+name)
    return manifest


def validate_previous_failure():
    attempt = json.loads((PREVIOUS/"attempt.json").read_bytes())
    report = json.loads((PREVIOUS/"results/context-window-report.json").read_bytes())
    original = report.get("modes", {}).get("original-gguf", {})
    repaired = report.get("modes", {}).get(REPAIRED, {})
    restore = report.get("restore", {})
    need(attempt.get("schema") == "qwen-context-repair-attempt/v1"
         and attempt.get("outcome") == "terminal_failure_graded"
         and attempt.get("manifest_sha256") ==
             "fc10b9dd1d5ced63b92104e16618f6aece18f851e262f0d8314b551b4e2af88b"
         and attempt.get("context_report_sha256") == PREVIOUS_REPORT_SHA,
         "previous_failure_attempt_changed")
    need(report.get("outcome") == "failed_restored"
         and report.get("error_code") == "context_client_failed"
         and original.get("attempted") is True and original.get("status") == "FAIL"
         and original.get("details", {}).get("returncode") == 1
         and repaired.get("attempted") is False
         and repaired.get("reason") == "window_aborted_before_launch"
         and restore.get("qwen_ready") is True
         and restore.get("admission_public") is True
         and restore.get("gateway_completions") == 2
         and restore.get("thinking_disabled_in_markers") is True,
         "previous_failure_report_changed")
    stderr = (PREVIOUS/"results/original-gguf.stderr").read_text("utf-8")
    need("ModuleNotFoundError: No module named 'quality_context_remaining'" in stderr,
         "previous_failure_cause_changed")
    return report


def validate_sources(selection):
    need(selection.get("schema") == "qwen-context-repair-selection/v1"
         and selection.get("source_combined_sha256") == R2_COMBINED_SHA
         and selection.get("fixture_sha256") == FIXTURE_SHA
         and list(selection.get("modes", {})) == ["original-gguf", REPAIRED],
         "repair_selection_identity_changed")
    fixture = json.loads((R1/"frozen/frozen.json").read_bytes())
    cases = fixture["cases"]
    gguf = [case["case_id"] for case in cases if case["target_tokens"] == 8192]
    all_ids = [case["case_id"] for case in cases]
    priority = "context-fact-start-t200000"
    repaired = [priority] + [case_id for case_id in all_ids if case_id != priority]
    specs = selection["modes"]
    need(specs["original-gguf"] == {
        "mode_id": "original-gguf", "max_len": 16384, "backend": "llama",
        "fixture_sha256": FIXTURE_SHA, "skip_recorded_warmup": False, "case_ids": gguf,
    }, "gguf_selection_changed")
    need(specs[REPAIRED] == {
        "mode_id": REPAIRED, "max_len": 204800, "backend": "vllm",
        "fixture_sha256": FIXTURE_SHA, "skip_recorded_warmup": False, "case_ids": repaired,
    }, "repaired_selection_changed")
    need(len(gguf) == 12 and len(repaired) == 84, "repair_selection_cardinality_changed")

    prior = [json.loads(line) for line in (R2/"results/combined-scores.jsonl").read_bytes().splitlines()]
    by_key = {(row["mode_id"], row["case_id"]): row for row in prior}
    need(len(prior) == len(by_key) == 612, "source_combined_coordinate_set_changed")
    gguf_rows = [by_key.get(("original-gguf", case_id)) for case_id in gguf]
    old_dflash = [row for row in prior if row["mode_id"] == SOURCE_DFLASH]
    need(all(row and row.get("outcome") == "BLOCKED" for row in gguf_rows)
         and len(old_dflash) == 84
         and all(row.get("outcome") == "BLOCKED" for row in old_dflash),
         "repair_would_replace_completed_answer")
    need(sum(row.get("outcome") in ("PASS", "FAIL") for row in prior) == 516,
         "source_completed_count_changed")
    return fixture


def validate_restoration_source():
    receipt = json.loads((R2/"restoration-after-abort.json").read_bytes())
    restored = receipt.get("restoration", {})
    expected = restored.get("container", {})
    need(receipt.get("outcome") == "failed_restored"
         and receipt.get("benchmark_dispatched") is False
         and restored.get("qwen_ready") is True
         and restored.get("admission_public") is True
         and restored.get("gateway_completions") == 2
         and restored.get("thinking_disabled_in_markers") is True
         and expected.get("running") is True
         and all(isinstance(expected.get(key), str) and expected[key]
                 for key in ("id", "image", "model")), "r2_restoration_not_terminal")
    return expected


def verify_old_ownership(w, common, previous):
    summary = json.loads((R2/"results/grading-summary.json").read_bytes())
    r2_owners = summary.get("storage_owned_containers", {})
    previous_owners = previous.get("storage_owned_containers", {})
    need(len(r2_owners) == 52 and len(previous_owners) == 2
         and not (set(r2_owners) & set(previous_owners)), "prior_ownership_set_changed")
    owners = {**r2_owners, **previous_owners}
    need(len(owners) == 54 and len(set(owners.values())) == 54, "prior_ownership_set_changed")
    for identity, name in owners.items():
        need(HEX.fullmatch(identity or ""), "r2_owned_id_invalid")
        row = w.exact_container(common, identity)
        state = (row or {}).get("State", {})
        need(row is not None and row.get("Id") == identity and row.get("Name") == "/"+name
             and state.get("Running") is False and state.get("Restarting") is False,
             "prior_owned_process_not_stopped")
    return owners


def install_unique_operator(common):
    """Give every instance a unique namespace; never reuse a lost mutation."""
    call = common.operator
    prefix = f"qwen-context-repair-{os.getpid()}-{secrets.token_hex(8)}"
    instance_sequence = iter(range(1, 1000))

    class RepairOperator:
        def __init__(self):
            self.instance = next(instance_sequence)
            self.sequence = 0

        def admission(self, verb, phase=None):
            self.sequence += 1
            request = {"schema": "ai-admission-operator-command/v1", "verb": verb,
                       "request_id": f"{prefix}-i{self.instance}-{self.sequence}"}
            if phase is not None:
                request["phase"] = phase
            value = call(request, 60.0)
            if (value.get("schema") != "ai-admission-operator-response/v1"
                    or value.get("request_id") != request["request_id"]):
                raise common.RunError("admission_response_invalid")
            return value

        def status(self):
            return self.admission("status")

        def advance(self, phase):
            value = self.admission("advance-phase", phase)
            if value.get("phase") != phase:
                raise common.RunError("admission_transition_not_proven")

        def qwen_status(self):
            return call({"verb": "status", "profile": "qwen"}, 60.0)

        def qwen(self, verb):
            self.sequence += 1
            value = call({"verb": verb, "profile": "qwen",
                          "idempotency_key": f"{prefix}-i{self.instance}-{verb}-{self.sequence}"})
            expected = "ready" if verb == "start" else "stopped"
            if value.get("profile") != "qwen" or value.get("state") != expected:
                raise common.RunError(f"qwen_{verb}_not_proven")

    common.Operator = RepairOperator
    return prefix


def install_repaired_profile(w):
    original = w.load_draft

    def load_draft(mode=None, name=None):
        repaired = mode is not None and mode.get("id") == REPAIRED
        mapped = {**mode, "id": SOURCE_DFLASH} if repaired else mode
        draft = original(mapped, name)
        if repaired:
            need(draft.PROFILE.get("DFLASH_MAX_LEN") == "245760"
                 and draft.PROFILE.get("GPU_UTIL") == "0.97"
                 and "KV_MEM" not in draft.PROFILE, "source_huge_profile_changed")
            draft.PROFILE.update(DFLASH_MAX_LEN="204800", KV_MEM="4000000000")
            expected = draft.EXPECTED_FAST_PROCESS_ARGS
            need(expected.get("max_num_seqs") == "--max-num-seqs 2"
                 and expected.get("draft_tokens") == '"num_speculative_tokens":7',
                 "source_huge_runtime_contract_changed")
            expected["max_model_len"] = "--max-model-len 204800"
            expected["kv_cache_memory"] = "--kv-cache-memory=4000000000"
        return draft

    w.load_draft = load_draft


def install_capacity_gate(w, common):
    inherited = w.profile_evidence
    pattern = re.compile(r"GPU KV cache size:\s*([0-9][0-9,]*)\s*tokens")

    def evidence(c, syv, mode, name, deadline):
        result = inherited(c, syv, mode, name, deadline)
        if mode["id"] != REPAIRED:
            return result
        captured = c.run(["docker", "logs", "--tail", "600", name], timeout=30, check=False)
        logs = (captured.stdout + captured.stderr).decode("utf-8", "replace")
        values = [int(match.replace(",", "")) for match in pattern.findall(logs)]
        result["gpu_kv_cache_tokens"] = max(values) if values else None
        result["required_kv_cache_tokens"] = 204800
        if not values or max(values) < 204800:
            raise w.WindowError("repaired_kv_capacity_not_proven", result)
        return result

    w.profile_evidence = evidence


def selected_modes(r1, w):
    original = next(mode for mode in r1.modes(w) if mode["id"] == "original-gguf")
    source = next(mode for mode in r1.modes(w) if mode["id"] == SOURCE_DFLASH)
    repaired = {**source, "id": REPAIRED, "max_len": 204800}
    return [original, repaired]


def run(manifest_sha):
    need(os.geteuid() == 0, "root_required")
    manifest = check_manifest(manifest_sha)  # No retained Python is imported before these pins.
    previous_failure = validate_previous_failure()
    selection = json.loads((ROOT/"selection.json").read_bytes())
    validate_sources(selection)
    expected_production = validate_restoration_source()

    sys.path.insert(0, str(R2))
    continuation = load("repair_pinned_continuation", R2/"continue_context.py")
    continuation.ROOT = ROOT
    r1 = load("repair_pinned_r1", R1/"run_context.py")
    r1.check_bundle(R1_MANIFEST_SHA)
    w = r1.load("repair_window", r1.OLD/"run_quality_window_remaining.py")
    w.verify_bundle(r1.OLD_SHA)
    common, syv, helper = w.load_runtime_modules()
    operator_namespace = install_unique_operator(common)
    modes = selected_modes(r1, w)

    r1.no_freetoken()
    helper.empty("public")
    production = w.production_identity(common)
    need(all(production.get(key) == expected_production[key] for key in ("id", "image", "model"))
         and type(production.get("running")) is bool, "original_production_identity_changed")
    qwen_state = common.qwen_state(common.Operator().qwen_status())
    need(qwen_state in ("ready", "stopped")
         and production["running"] is (qwen_state == "ready"), "public_qwen_state_unresolved")
    if qwen_state == "stopped":
        common.wait_gpu_idle(90)
    old_owners = verify_old_ownership(w, common, previous_failure)
    baseline = w.available_bytes()
    need(baseline >= w.RESERVE_BYTES + r1.GROWTH, "repair_storage_budget_insufficient")

    storage_module = sys.modules["context_storage"]
    storage = storage_module.OwnedStorage(ROOT)
    continuation.storage_check(storage)
    payloads = continuation.verify_selected_payloads(w, r1, common, modes)
    continuation.storage_check(storage)
    save(ROOT/"payload-preflight.json", payloads)

    install_repaired_profile(w)
    r1.PREFIX = PREFIX
    r1.configure(w, common, syv, helper, modes, FIXTURE_SHA)
    install_capacity_gate(w, common)
    budget = continuation.time_budget(selection)
    w.WINDOW_SECONDS = budget["window_seconds"]
    w.MODE_SECONDS = budget["capture_seconds"]
    continuation.configure_client(w, r1, selection, manifest["files"]["selection.json"], storage)
    continuation.configure_public_handoff(w, common, helper,
                                          {"production_prestate": expected_production})
    storage_module.install_controller(w, storage)

    settings = {
        "schema": "qwen-context-repair-execution/v1",
        "source_manifest_sha256": manifest_sha,
        "source_combined_sha256": R2_COMBINED_SHA,
        "source_restoration_sha256": R2_RESTORATION_SHA,
        "fixture_sha256": FIXTURE_SHA,
        "selection_sha256": manifest["files"]["selection.json"],
        "modes": [{**mode, "model": str(mode["model"])} for mode in modes],
        "repair": {"source_profile": SOURCE_DFLASH, "profile_id": REPAIRED,
                   "changed": {"KV_MEM": "4000000000", "DFLASH_MAX_LEN": "204800"},
                   "unchanged": {"GPU_UTIL": "0.97", "DFLASH_TOKENS": "7",
                                 "max_num_seqs": 2},
                   "first_case": "context-fact-start-t200000"},
        "coordinates": 96, "case_timeout_seconds": 1800,
        "storage_owned_limit_bytes": storage_module.OWNED_LIMIT,
        "storage_write_reserve_bytes": storage_module.WRITE_RESERVE,
        "root_free_floor_bytes": storage_module.ROOT_FLOOR,
        "planned_peak_bytes": 955252736,
        "operator_namespace": operator_namespace,
        **budget, "started_epoch": time.time(),
    }
    save(ROOT/"execution-settings.json", settings)

    attempt_path = ROOT/"attempt.json"
    with attempt_path.open("x", encoding="utf-8") as attempt:
        proof = {"schema": "qwen-context-repair-attempt/v1", "outcome": "capture_unknown",
                 "started_epoch": time.time(), "manifest_sha256": manifest_sha,
                 "selection_sha256": manifest["files"]["selection.json"],
                 "source_combined_sha256": R2_COMBINED_SHA, "old_owned_count": len(old_owners)}
        try:
            attempt.write(json.dumps(proof, sort_keys=True)); attempt.flush(); os.fsync(attempt.fileno())
            status = w.main(["--output-dir", str(ROOT/"results"),
                             "--bundle-manifest-sha256", r1.OLD_SHA])
            report = json.loads((ROOT/"results/window-report.json").read_bytes())
            post_free = w.available_bytes()
            storage_error = None
            try:
                continuation.storage_check(storage)
            except storage_module.StorageError as exc:
                storage_error = exc.code
            context_report = {**report, "schema": "qwen-context-repair-window/v1",
                "suite_id": "qwen38-context-repair-96-v1", "source_manifest_sha256": manifest_sha,
                "source_combined_sha256": R2_COMBINED_SHA,
                "selection_sha256": manifest["files"]["selection.json"],
                "fixture_sha256": FIXTURE_SHA,
                "legacy_controller_report_sha256": sha(ROOT/"results/window-report.json"),
                "storage_available_post_restore_bytes": post_free,
                "storage_owned_containers": dict(storage.containers),
                "storage_post_restore": storage.last, "storage_error": storage_error,
                "post_restore_storage_bound_ok": storage_error is None}
            save(ROOT/"results/context-window-report.json", context_report)
            continuation.storage_check(storage)
            grader = load("repair_grader", ROOT/"grade_repair.py")
            summary = grader.grade(root=ROOT, r1=r1, continuation=continuation,
                                   storage=storage, manifest_sha=manifest_sha)
            proof.update(outcome="tests_complete" if status == 0 else "terminal_failure_graded",
                         controller_exit_code=status, ended_epoch=time.time(),
                         context_report_sha256=sha(ROOT/"results/context-window-report.json"),
                         grading_summary_sha256=sha(ROOT/"results/grading-summary.json"),
                         grading_counts={"original": summary.get("original_counts"),
                                         "new_modes": summary.get("new_modes")})
            return 0 if status == 0 else 1
        except BaseException as exc:
            proof.update(outcome="terminal_failure", ended_epoch=time.time(),
                         error=type(exc).__name__+": "+str(exc))
            raise
        finally:
            attempt.seek(0); attempt.truncate()
            attempt.write(json.dumps(proof, indent=2, sort_keys=True)+"\n")
            attempt.flush(); os.fsync(attempt.fileno())


def main():
    need(len(sys.argv) == 2, "usage: repair_context.py MANIFEST_SHA")
    return run(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
