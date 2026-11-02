"""Continue the interrupted KV-4350 run without replaying 21 completed cases."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parent
R1 = Path("/srv/ai-server/benchmarks/context-20260908-r1")
R2 = Path("/srv/ai-server/benchmarks/context-20260908-r2")
OLD = Path("/srv/ai-server/benchmarks/quality-remaining-20260908")
FIRST = Path("/srv/ai-server/benchmarks/context-20260910-repair")
CLIENTFIX = Path("/srv/ai-server/benchmarks/context-20260910-repair-clientfix")
ABORTED = Path("/srv/ai-server/benchmarks/context-20260910-repair-kv4350")
R1_MANIFEST_SHA = "c71c26c5cf3124347a021489d60148b102a5ad2c35ee4b1314e37ef8eafa36e6"
R1_RUN_SHA = "7cb2f97465a69992babc9887780477ab98d3eb548500e33052cba7eb70248816"
R1_CONTROLLER_SHA = "4ad216fc2ff90ea9a8971e10904681fd11396f86977bc3e865ae736cf7dcccc5"
R2_CONTINUATION_SHA = "6d2e89059388a5fef595cfc80f76a070d46f21e3a8865c38054b33691c6a5307"
R2_STORAGE_SHA = "2085436ed77ec5a9f64d61568bb612254cb58158027869fc036fa666ff914cfd"
R2_REMAINING_SHA = "d05b4125873eb7afdea27e64f47515c2ff2371c85771ec60fbaee5b1bc6e3979"
R2_RESTORATION_SHA = "5c4db227957e37a03fb36cddd61d9d5eef4be0a1766ce9969a0a7fe4eb3328a0"
R2_GRADING_SHA = "99f61e0b3b0746d2d69d57435e5281fd95bf4d3d0c80c6b46b44d28638488641"
FIRST_REPORT_SHA = "64a362072a70b563cf88e89cf26c1a1c7eb3716d39de639e85243e7dffb912c8"
CLIENTFIX_REPORT_SHA = "d369f32e7a5a5729ee6054b8b1f9a56c352c808a99dde5085070cdadc4fb7561"
CLIENTFIX_GRADING_SHA = "22fbde807054aafd68183964664300f877ac97b60ca4b2d4d5b86b03a6cc2b1d"
CLIENTFIX_CONTROLLER_SHA = "7c7866bb0d8a0eeae3eb46019d0a051aef5b0da23971bd80a00fd25e282233c2"
COMBINED_SHA = "4f47916f1f0444478cc408d2a0610c2781777fdbfd1579f6bd83d589e320391e"
ABORTED_CONTROLLER_SHA = "c6641e164782f511ba64a220bb38e166600fdc8337bd8f6ddf19aefa63975c0b"
ABORTED_SELECTION_SHA = "a205678ff858691535f7c26b18b1bf5e38a2a75ad08ba54b344aff6923fcfc30"
ABORTED_ATTEMPT_SHA = "852ed3cbb3b498cef4913e3f03acdf1552985dee0a1b1e2fb9c4b11d00bcee0f"
ABORTED_REPORT_SHA = "a0fff7ea92c8f060a986e60ea2a2d9fa09393da82446a72c9f475eba4920294f"
ABORTED_RAW_SHA = "7ab958bb3a131122430226c4d2abd653f76df88206357c4ff91da5728237c1e9"
ABORTED_RESTORE_SHA = "bd837831136de91466b46e4bbf0ccf28b332640c42fff413b33bf241a433f308"
FIXTURE_SHA = "2df1be2cff66b53ce57b3eefe3eed8be2d16c59d1e37b783341f176995689d94"
PREFIX = "qwen-context-20260910-repair-kv4350-continue"
MODE = "dflash2-huge-n7-kv4350mb-204800"
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
        stream.flush(); os.fsync(stream.fileno())


def dependencies():
    return {
        "r1/run_context.py": (R1/"run_context.py", R1_RUN_SHA),
        "r1/controller.py": (OLD/"run_quality_window_remaining.py", R1_CONTROLLER_SHA),
        "r1/frozen/frozen.json": (R1/"frozen/frozen.json", FIXTURE_SHA),
        "r2/continue_context.py": (R2/"continue_context.py", R2_CONTINUATION_SHA),
        "r2/context_storage.py": (R2/"context_storage.py", R2_STORAGE_SHA),
        "r2/quality_context_remaining.py": (R2/"quality_context_remaining.py", R2_REMAINING_SHA),
        "r2/restoration-after-abort.json": (R2/"restoration-after-abort.json", R2_RESTORATION_SHA),
        "r2/results/grading-summary.json": (R2/"results/grading-summary.json", R2_GRADING_SHA),
        "first/results/context-window-report.json":
            (FIRST/"results/context-window-report.json", FIRST_REPORT_SHA),
        "clientfix/results/context-window-report.json":
            (CLIENTFIX/"results/context-window-report.json", CLIENTFIX_REPORT_SHA),
        "clientfix/repair_context.py":
            (CLIENTFIX/"repair_context.py", CLIENTFIX_CONTROLLER_SHA),
        "clientfix/results/grading-summary.json":
            (CLIENTFIX/"results/grading-summary.json", CLIENTFIX_GRADING_SHA),
        "clientfix/results/combined-original-scores.jsonl":
            (CLIENTFIX/"results/combined-original-scores.jsonl", COMBINED_SHA),
        "aborted/repair_context.py": (ABORTED/"repair_context.py", ABORTED_CONTROLLER_SHA),
        "aborted/selection.json": (ABORTED/"selection.json", ABORTED_SELECTION_SHA),
        "aborted/attempt.json": (ABORTED/"attempt.json", ABORTED_ATTEMPT_SHA),
        "aborted/results/context-window-report.json":
            (ABORTED/"results/context-window-report.json", ABORTED_REPORT_SHA),
        "aborted/results/dflash2-huge-n7-kv4350mb-204800.jsonl":
            (ABORTED/"results/dflash2-huge-n7-kv4350mb-204800.jsonl", ABORTED_RAW_SHA),
        "aborted/restoration-after-abort.json":
            (ABORTED/"restoration-after-abort.json", ABORTED_RESTORE_SHA),
    }


def check_manifest(expected):
    need(HEX.fullmatch(expected or "") and sha(ROOT/"manifest.json") == expected,
         "continuation_manifest_changed")
    manifest = json.loads((ROOT/"manifest.json").read_bytes())
    need(manifest.get("schema") == "qwen-context-repair-bundle/v1"
         and set(manifest.get("files", {})) == FILES, "continuation_manifest_file_set_changed")
    for name, digest in manifest["files"].items():
        need(Path(name).name == name and sha(ROOT/name) == digest,
             "continuation_source_changed:"+name)
    expected_dependencies = dependencies()
    need(set(manifest.get("dependencies", {})) == set(expected_dependencies),
         "continuation_dependency_set_changed")
    for name, (path, digest) in expected_dependencies.items():
        need(HEX.fullmatch(digest or "") and manifest["dependencies"][name] == digest
             and sha(path) == digest, "continuation_dependency_changed:"+name)
    return manifest


def validate_abort(selection):
    old_selection = json.loads((ABORTED/"selection.json").read_bytes())["modes"][MODE]["case_ids"]
    new_ids = selection["modes"][MODE]["case_ids"]
    raw = (ABORTED/"results"/(MODE+".jsonl")).read_bytes().splitlines()
    rows = [json.loads(line) for line in raw]
    terminal = [row for row in rows if row.get("kind") == "case"]
    starts = [row for row in rows if row.get("kind") == "case_start"]
    terminal_ids = [row.get("case_id") for row in terminal]
    start_ids = [row.get("case_id") for row in starts]
    need(len(old_selection) == 84 and terminal_ids == old_selection[:21]
         and start_ids == old_selection[:22] and new_ids == old_selection[21:]
         and len(new_ids) == len(set(new_ids)) == 63,
         "continuation_partition_changed")
    cases = json.loads((R1/"frozen/frozen.json").read_bytes())["cases"]
    by_id = {case["case_id"]: case for case in cases}
    need(all(row.get("mode_id") == MODE for row in terminal + starts),
         "completed_prefix_mode_changed")
    for row in terminal:
        case = by_id[row["case_id"]]
        need(row.get("request_sha256") == case["request_sha256"],
             "completed_prefix_request_changed")
        if case["check"]["type"] == "python_function_cases":
            need(row.get("outcome") == "pending_sandbox"
                 and row.get("error") == "python_sandbox_required",
                 "completed_code_not_pending_sandbox")
        elif row["case_id"] == "context-join-start-t32768":
            need(row.get("outcome") == "fail"
                 and row.get("error") == "finish_reason_length"
                 and row.get("finish_reason") == "length",
                 "completed_known_failure_changed")
        else:
            need(row.get("outcome") == "pass", "completed_noncode_not_pass")
    attempt = json.loads((ABORTED/"attempt.json").read_bytes())
    report = json.loads((ABORTED/"results/context-window-report.json").read_bytes())
    entry = report.get("modes", {}).get(MODE, {})
    need(attempt.get("outcome") == "terminal_failure"
         and report.get("error_code") == "root_storage_reserve_exhausted"
         and entry.get("status") == "BLOCKED" and entry.get("attempted") is False
         and entry.get("reason") == "window_aborted_before_launch"
         and entry.get("abort_code") == "root_storage_reserve_exhausted"
         and len(report.get("storage_owned_containers", {})) == 2,
         "aborted_window_identity_changed")
    receipt = json.loads((ABORTED/"restoration-after-abort.json").read_bytes())
    restored = receipt.get("restoration", {})
    need(receipt.get("outcome") == "failed_restored"
         and restored.get("qwen_ready") is True and restored.get("admission_public") is True
         and restored.get("gateway_completions") == 2
         and restored.get("thinking_disabled_in_markers") is True,
         "aborted_window_not_restored")
    return report


def validate_selection(selection):
    need(selection.get("schema") == "qwen-context-repair-selection/v1"
         and selection.get("source_combined_sha256") == COMBINED_SHA
         and selection.get("fixture_sha256") == FIXTURE_SHA
         and list(selection.get("modes", {})) == [MODE], "selection_identity_changed")
    ids = selection["modes"][MODE]["case_ids"]
    need(selection["modes"][MODE] == {
        "mode_id": MODE, "max_len": 204800, "backend": "vllm",
        "fixture_sha256": FIXTURE_SHA, "skip_recorded_warmup": False, "case_ids": ids,
    }, "selection_runtime_contract_changed")


def expected_production():
    receipt = json.loads((R2/"restoration-after-abort.json").read_bytes())
    expected = receipt["restoration"]["container"]
    need(all(isinstance(expected.get(k), str) and expected[k] for k in ("id", "image", "model")),
         "production_identity_missing")
    return expected


def verify_old_ownership(w, common, aborted_report):
    groups = [
        json.loads((R2/"results/grading-summary.json").read_bytes())["storage_owned_containers"],
        json.loads((FIRST/"results/context-window-report.json").read_bytes())["storage_owned_containers"],
        json.loads((CLIENTFIX/"results/grading-summary.json").read_bytes())["storage_owned_containers"],
        aborted_report["storage_owned_containers"],
    ]
    need([len(group) for group in groups] == [52, 2, 6, 2], "prior_owner_counts_changed")
    owners = {}
    for group in groups:
        need(not (set(owners) & set(group)), "prior_owner_id_overlap")
        owners.update(group)
    need(len(owners) == len(set(owners.values())) == 62, "prior_owner_name_overlap")
    for identity, name in owners.items():
        need(HEX.fullmatch(identity or ""), "prior_owned_id_invalid")
        row = w.exact_container(common, identity)
        state = (row or {}).get("State", {})
        need(row is not None and row.get("Id") == identity and row.get("Name") == "/"+name
             and state.get("Running") is False and state.get("Restarting") is False,
             "prior_owned_process_not_stopped")
    return owners


def run(manifest_sha):
    need(os.geteuid() == 0, "root_required")
    manifest = check_manifest(manifest_sha)
    selection = json.loads((ROOT/"selection.json").read_bytes())
    validate_selection(selection)
    aborted_report = validate_abort(selection)
    expected = expected_production()

    previous = load("continuation_pinned_kv4350", ABORTED/"repair_context.py")
    sys.path.insert(0, str(R2))
    continuation = load("continuation_pinned_r2", R2/"continue_context.py")
    continuation.ROOT = ROOT
    r1 = load("continuation_pinned_r1", R1/"run_context.py")
    r1.check_bundle(R1_MANIFEST_SHA)
    w = r1.load("continuation_window", r1.OLD/"run_quality_window_remaining.py")
    w.verify_bundle(r1.OLD_SHA)
    common, syv, helper = w.load_runtime_modules()
    base = previous.load("continuation_operator_source", CLIENTFIX/"repair_context.py")
    operator_namespace = base.install_unique_operator(common)
    source = next(mode for mode in r1.modes(w) if mode["id"] == SOURCE_DFLASH)
    modes = [{**source, "id": MODE, "max_len": 204800}]

    r1.no_freetoken(); helper.empty("public")
    production = w.production_identity(common)
    need(all(production.get(k) == expected[k] for k in ("id", "image", "model"))
         and type(production.get("running")) is bool, "production_identity_changed")
    qwen_state = common.qwen_state(common.Operator().qwen_status())
    need(qwen_state in ("ready", "stopped")
         and production["running"] is (qwen_state == "ready"), "public_qwen_state_unresolved")
    if qwen_state == "stopped":
        common.wait_gpu_idle(90)
    old_owners = verify_old_ownership(w, common, aborted_report)
    need(w.available_bytes() >= w.RESERVE_BYTES + r1.GROWTH, "continuation_storage_budget_insufficient")

    storage_module = sys.modules["context_storage"]
    storage = storage_module.OwnedStorage(ROOT)
    continuation.storage_check(storage)
    payloads = continuation.verify_selected_payloads(w, r1, common, modes)
    continuation.storage_check(storage)
    save(ROOT/"payload-preflight.json", payloads)

    previous.install_profile(previous, w)
    r1.PREFIX = PREFIX
    r1.configure(w, common, syv, helper, modes, FIXTURE_SHA)
    base.REPAIRED = MODE
    base.install_capacity_gate(w, common)
    budget = continuation.time_budget(selection)
    need(budget == {"capture_seconds": 354960, "restore_reserve_seconds": 2700,
                    "window_seconds": 357660, "unit_runtime_max_seconds": 364860},
         "continuation_time_budget_changed")
    w.WINDOW_SECONDS = budget["window_seconds"]
    w.MODE_SECONDS = budget["capture_seconds"]
    continuation.configure_client(w, r1, selection, manifest["files"]["selection.json"], storage)
    continuation.configure_public_handoff(w, common, helper, {"production_prestate": expected})
    storage_module.install_controller(w, storage)

    settings = {
        "schema": "qwen-context-repair-execution/v1", "source_manifest_sha256": manifest_sha,
        "source_combined_sha256": COMBINED_SHA, "source_abort_report_sha256": ABORTED_REPORT_SHA,
        "source_abort_raw_sha256": ABORTED_RAW_SHA,
        "source_abort_restoration_sha256": ABORTED_RESTORE_SHA, "fixture_sha256": FIXTURE_SHA,
        "selection_sha256": manifest["files"]["selection.json"],
        "modes": [{**mode, "model": str(mode["model"])} for mode in modes],
        "repair": {"source_profile": SOURCE_DFLASH, "profile_id": MODE,
                   "changed": {"KV_MEM": "4350000000", "DFLASH_MAX_LEN": "204800"},
                   "unchanged": {"GPU_UTIL": "0.97", "DFLASH_TOKENS": "7", "max_num_seqs": 2},
                   "completed_prefix": 21, "continued_coordinates": 63,
                   "first_case": "context-version-end-t32768"},
        "coordinates": 63, "final_profile_coordinates": 84, "case_timeout_seconds": 1800,
        "storage_owned_limit_bytes": storage_module.OWNED_LIMIT,
        "storage_write_reserve_bytes": storage_module.WRITE_RESERVE,
        "root_free_floor_bytes": storage_module.ROOT_FLOOR,
        "planned_peak_bytes": 955252736, "operator_namespace": operator_namespace,
        **budget, "started_epoch": time.time(),
    }
    save(ROOT/"execution-settings.json", settings)

    with (ROOT/"attempt.json").open("x", encoding="utf-8") as attempt:
        proof = {"schema": "qwen-context-repair-attempt/v1", "outcome": "capture_unknown",
                 "started_epoch": time.time(), "manifest_sha256": manifest_sha,
                 "selection_sha256": manifest["files"]["selection.json"],
                 "source_combined_sha256": COMBINED_SHA, "old_owned_count": len(old_owners)}
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
                "suite_id": "qwen38-context-repair-kv4350-continue-63-v1",
                "source_manifest_sha256": manifest_sha, "source_combined_sha256": COMBINED_SHA,
                "source_abort_raw_sha256": ABORTED_RAW_SHA,
                "source_abort_restoration_sha256": ABORTED_RESTORE_SHA,
                "selection_sha256": manifest["files"]["selection.json"], "fixture_sha256": FIXTURE_SHA,
                "legacy_controller_report_sha256": sha(ROOT/"results/window-report.json"),
                "storage_available_post_restore_bytes": post_free,
                "storage_owned_containers": dict(storage.containers),
                "storage_post_restore": storage.last, "storage_error": storage_error,
                "post_restore_storage_bound_ok": storage_error is None}
            save(ROOT/"results/context-window-report.json", context_report)
            continuation.storage_check(storage)
            entry = context_report.get("modes", {}).get(MODE, {})
            if status != 0 or entry.get("status") != "CAPTURED":
                proof.update(outcome="terminal_failure", controller_exit_code=status,
                             ended_epoch=time.time(), reason=entry.get("reason", "capture_incomplete"),
                             context_report_sha256=sha(ROOT/"results/context-window-report.json"))
                return 1
            grader = load("continuation_grader", ROOT/"grade_repair.py")
            summary = grader.grade(root=ROOT, r1=r1, continuation=continuation,
                                   storage=storage, manifest_sha=manifest_sha)
            proof.update(outcome="tests_complete", controller_exit_code=status, ended_epoch=time.time(),
                         context_report_sha256=sha(ROOT/"results/context-window-report.json"),
                         grading_summary_sha256=sha(ROOT/"results/grading-summary.json"),
                         grading_counts={"original": summary.get("original_counts"),
                                         "new_modes": summary.get("new_modes")})
            return 0
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
