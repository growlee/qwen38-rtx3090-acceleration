"""One-shot 84-case DFlash context repair with a 4.35 GB KV cache."""
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
PREVIOUS = Path("/srv/ai-server/benchmarks/context-20260910-repair-clientfix")
R1_MANIFEST_SHA = "c71c26c5cf3124347a021489d60148b102a5ad2c35ee4b1314e37ef8eafa36e6"
R1_RUN_SHA = "7cb2f97465a69992babc9887780477ab98d3eb548500e33052cba7eb70248816"
R1_CONTROLLER_SHA = "4ad216fc2ff90ea9a8971e10904681fd11396f86977bc3e865ae736cf7dcccc5"
R2_CONTINUATION_SHA = "6d2e89059388a5fef595cfc80f76a070d46f21e3a8865c38054b33691c6a5307"
R2_STORAGE_SHA = "2085436ed77ec5a9f64d61568bb612254cb58158027869fc036fa666ff914cfd"
R2_REMAINING_SHA = "d05b4125873eb7afdea27e64f47515c2ff2371c85771ec60fbaee5b1bc6e3979"
R2_RESTORATION_SHA = "5c4db227957e37a03fb36cddd61d9d5eef4be0a1766ce9969a0a7fe4eb3328a0"
R2_GRADING_SHA = "99f61e0b3b0746d2d69d57435e5281fd95bf4d3d0c80c6b46b44d28638488641"
FIRST_REPORT_SHA = "64a362072a70b563cf88e89cf26c1a1c7eb3716d39de639e85243e7dffb912c8"
PREVIOUS_ATTEMPT_SHA = "dce41e64417418c4886d5848db468d99fd22b5abd1eebc3cbd5b84b58e5971be"
PREVIOUS_REPORT_SHA = "d369f32e7a5a5729ee6054b8b1f9a56c352c808a99dde5085070cdadc4fb7561"
PREVIOUS_GRADING_SHA = "22fbde807054aafd68183964664300f877ac97b60ca4b2d4d5b86b03a6cc2b1d"
PREVIOUS_GGUF_SCORE_SHA = "5bf080a13c3aab07a92f4fa8e5c5ddee6d69282659d677473fd46313cc6430c3"
PREVIOUS_COMBINED_SHA = "4f47916f1f0444478cc408d2a0610c2781777fdbfd1579f6bd83d589e320391e"
PREVIOUS_KV4_SCORE_SHA = "530d7d2fa039aedb673eafaeeb5b32e2c859f0ca0db510341466451d3d126d2c"
PREVIOUS_RAW_SHA = "7b2d7c7c4b0950fdfd969b1c1c7ca6470c88048fb4c02357afd4829f911e1fcf"
BASE_SOURCE_SHA = "7c7866bb0d8a0eeae3eb46019d0a051aef5b0da23971bd80a00fd25e282233c2"
FIXTURE_SHA = "2df1be2cff66b53ce57b3eefe3eed8be2d16c59d1e37b783341f176995689d94"
PREFIX = "qwen-context-20260910-repair-kv4350"
REPAIRED = "dflash2-huge-n7-kv4350mb-204800"
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
        "previous/repair_context.py": (PREVIOUS/"repair_context.py", BASE_SOURCE_SHA),
        "previous/attempt.json": (PREVIOUS/"attempt.json", PREVIOUS_ATTEMPT_SHA),
        "previous/results/context-window-report.json":
            (PREVIOUS/"results/context-window-report.json", PREVIOUS_REPORT_SHA),
        "previous/results/grading-summary.json":
            (PREVIOUS/"results/grading-summary.json", PREVIOUS_GRADING_SHA),
        "previous/results/scores-original-gguf.jsonl":
            (PREVIOUS/"results/scores-original-gguf.jsonl", PREVIOUS_GGUF_SCORE_SHA),
        "previous/results/combined-original-scores.jsonl":
            (PREVIOUS/"results/combined-original-scores.jsonl", PREVIOUS_COMBINED_SHA),
        "previous/results/repaired-profile-scores.jsonl":
            (PREVIOUS/"results/repaired-profile-scores.jsonl", PREVIOUS_KV4_SCORE_SHA),
        "previous/results/original-gguf.jsonl":
            (PREVIOUS/"results/original-gguf.jsonl", PREVIOUS_RAW_SHA),
    }


def check_manifest(expected):
    need(HEX.fullmatch(expected or "") and sha(ROOT/"manifest.json") == expected,
         "repair_manifest_changed")
    manifest = json.loads((ROOT/"manifest.json").read_bytes())
    need(manifest.get("schema") == "qwen-context-repair-bundle/v1"
         and set(manifest.get("files", {})) == FILES, "repair_manifest_file_set_changed")
    for name, digest in manifest["files"].items():
        need(Path(name).name == name and sha(ROOT/name) == digest, "repair_source_changed:"+name)
    expected_dependencies = dependencies()
    need(set(manifest.get("dependencies", {})) == set(expected_dependencies),
         "repair_dependency_set_changed")
    for name, (path, digest) in expected_dependencies.items():
        need(manifest["dependencies"][name] == digest and sha(path) == digest,
             "repair_dependency_changed:"+name)
    return manifest


def validate_previous():
    attempt = json.loads((PREVIOUS/"attempt.json").read_bytes())
    report = json.loads((PREVIOUS/"results/context-window-report.json").read_bytes())
    summary = json.loads((PREVIOUS/"results/grading-summary.json").read_bytes())
    need(attempt.get("outcome") == "terminal_failure_graded"
         and attempt.get("context_report_sha256") == PREVIOUS_REPORT_SHA
         and attempt.get("grading_summary_sha256") == PREVIOUS_GRADING_SHA,
         "previous_attempt_changed")
    modes = report.get("modes", {})
    restore = report.get("restore", {})
    need(modes.get("original-gguf", {}).get("status") == "CAPTURED"
         and modes.get("dflash2-huge-n7-kv4gb-204800", {}).get("status") == "FAIL"
         and modes["dflash2-huge-n7-kv4gb-204800"].get("reason") == "candidate_exit_before_ready"
         and restore.get("qwen_ready") is True and restore.get("admission_public") is True
         and restore.get("gateway_completions") == 2
         and restore.get("thinking_disabled_in_markers") is True,
         "previous_window_not_terminal_restored")
    need(summary.get("original_counts") == {"BLOCKED": 84, "FAIL": 1, "PASS": 527}
         and summary.get("new_modes", {}).get("original-gguf") == {"PASS": 12}
         and summary.get("new_modes", {}).get("dflash2-huge-n7-kv4gb-204800") == {"BLOCKED": 84}
         and len(summary.get("storage_owned_containers", {})) == 6,
         "previous_grading_changed")
    need(not (PREVIOUS/"results/dflash2-huge-n7-kv4gb-204800.jsonl").exists(),
         "previous_dflash_was_dispatched")
    return report, summary


def validate_selection(selection):
    need(selection.get("schema") == "qwen-context-repair-selection/v1"
         and selection.get("source_combined_sha256") == PREVIOUS_COMBINED_SHA
         and selection.get("fixture_sha256") == FIXTURE_SHA
         and list(selection.get("modes", {})) == [REPAIRED], "selection_identity_changed")
    cases = json.loads((R1/"frozen/frozen.json").read_bytes())["cases"]
    priority = "context-fact-start-t200000"
    expected = [priority] + [c["case_id"] for c in cases if c["case_id"] != priority]
    need(selection["modes"][REPAIRED] == {
        "mode_id": REPAIRED, "max_len": 204800, "backend": "vllm",
        "fixture_sha256": FIXTURE_SHA, "skip_recorded_warmup": False,
        "case_ids": expected,
    } and len(expected) == 84, "selection_case_set_changed")


def expected_production():
    receipt = json.loads((R2/"restoration-after-abort.json").read_bytes())
    expected = receipt["restoration"]["container"]
    need(all(isinstance(expected.get(k), str) and expected[k] for k in ("id", "image", "model")),
         "production_identity_missing")
    return expected


def verify_old_ownership(base, w, common, first_report, previous_summary):
    r2 = json.loads((R2/"results/grading-summary.json").read_bytes())["storage_owned_containers"]
    first = first_report["storage_owned_containers"]
    previous = previous_summary["storage_owned_containers"]
    need((len(r2), len(first), len(previous)) == (52, 2, 6), "prior_owner_counts_changed")
    owners = {**r2, **first, **previous}
    need(len(owners) == len(set(owners.values())) == 60, "prior_ownership_overlap")
    for identity, name in owners.items():
        need(HEX.fullmatch(identity or ""), "prior_owned_id_invalid")
        row = w.exact_container(common, identity)
        state = (row or {}).get("State", {})
        need(row is not None and row.get("Id") == identity and row.get("Name") == "/"+name
             and state.get("Running") is False and state.get("Restarting") is False,
             "prior_owned_process_not_stopped")
    return owners


def install_profile(base, w):
    original = w.load_draft

    def load_draft(mode=None, name=None):
        repaired = mode is not None and mode.get("id") == REPAIRED
        mapped = {**mode, "id": SOURCE_DFLASH} if repaired else mode
        draft = original(mapped, name)
        if repaired:
            need(draft.PROFILE.get("DFLASH_MAX_LEN") == "245760"
                 and draft.PROFILE.get("GPU_UTIL") == "0.97"
                 and "KV_MEM" not in draft.PROFILE, "source_huge_profile_changed")
            draft.PROFILE.update(DFLASH_MAX_LEN="204800", KV_MEM="4350000000")
            expected = draft.EXPECTED_FAST_PROCESS_ARGS
            need(expected.get("max_num_seqs") == "--max-num-seqs 2"
                 and expected.get("draft_tokens") == '"num_speculative_tokens":7',
                 "source_huge_runtime_contract_changed")
            expected["max_model_len"] = "--max-model-len 204800"
            expected["kv_cache_memory"] = "--kv-cache-memory=4350000000"
        return draft

    w.load_draft = load_draft


def run(manifest_sha):
    need(os.geteuid() == 0, "root_required")
    manifest = check_manifest(manifest_sha)
    previous_report, previous_summary = validate_previous()
    selection = json.loads((ROOT/"selection.json").read_bytes())
    validate_selection(selection)
    expected = expected_production()

    base = load("kv4350_pinned_base", PREVIOUS/"repair_context.py")
    base.REPAIRED = REPAIRED
    sys.path.insert(0, str(R2))
    continuation = load("kv4350_pinned_continuation", R2/"continue_context.py")
    continuation.ROOT = ROOT
    r1 = load("kv4350_pinned_r1", R1/"run_context.py")
    r1.check_bundle(R1_MANIFEST_SHA)
    w = r1.load("kv4350_window", r1.OLD/"run_quality_window_remaining.py")
    w.verify_bundle(r1.OLD_SHA)
    common, syv, helper = w.load_runtime_modules()
    operator_namespace = base.install_unique_operator(common)
    source = next(mode for mode in r1.modes(w) if mode["id"] == SOURCE_DFLASH)
    modes = [{**source, "id": REPAIRED, "max_len": 204800}]

    r1.no_freetoken(); helper.empty("public")
    production = w.production_identity(common)
    need(all(production.get(k) == expected[k] for k in ("id", "image", "model"))
         and type(production.get("running")) is bool, "production_identity_changed")
    qwen_state = common.qwen_state(common.Operator().qwen_status())
    need(qwen_state in ("ready", "stopped")
         and production["running"] is (qwen_state == "ready"), "public_qwen_state_unresolved")
    if qwen_state == "stopped":
        common.wait_gpu_idle(90)
    first_report = json.loads((FIRST/"results/context-window-report.json").read_bytes())
    old_owners = verify_old_ownership(base, w, common, first_report, previous_summary)
    need(w.available_bytes() >= w.RESERVE_BYTES + r1.GROWTH, "repair_storage_budget_insufficient")

    storage_module = sys.modules["context_storage"]
    storage = storage_module.OwnedStorage(ROOT)
    continuation.storage_check(storage)
    payloads = continuation.verify_selected_payloads(w, r1, common, modes)
    continuation.storage_check(storage)
    save(ROOT/"payload-preflight.json", payloads)

    install_profile(base, w)
    r1.PREFIX = PREFIX
    r1.configure(w, common, syv, helper, modes, FIXTURE_SHA)
    base.install_capacity_gate(w, common)
    budget = continuation.time_budget(selection)
    w.WINDOW_SECONDS = budget["window_seconds"]
    w.MODE_SECONDS = budget["capture_seconds"]
    continuation.configure_client(w, r1, selection, manifest["files"]["selection.json"], storage)
    continuation.configure_public_handoff(w, common, helper, {"production_prestate": expected})
    storage_module.install_controller(w, storage)

    settings = {
        "schema": "qwen-context-repair-execution/v1", "source_manifest_sha256": manifest_sha,
        "source_combined_sha256": PREVIOUS_COMBINED_SHA,
        "source_restoration_sha256": R2_RESTORATION_SHA, "fixture_sha256": FIXTURE_SHA,
        "selection_sha256": manifest["files"]["selection.json"],
        "modes": [{**mode, "model": str(mode["model"])} for mode in modes],
        "repair": {"source_profile": SOURCE_DFLASH, "profile_id": REPAIRED,
                   "changed": {"KV_MEM": "4350000000", "DFLASH_MAX_LEN": "204800"},
                   "unchanged": {"GPU_UTIL": "0.97", "DFLASH_TOKENS": "7", "max_num_seqs": 2},
                   "first_case": "context-fact-start-t200000",
                   "prior_failed_profile": "dflash2-huge-n7-kv4gb-204800"},
        "coordinates": 84, "case_timeout_seconds": 1800,
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
                 "source_combined_sha256": PREVIOUS_COMBINED_SHA,
                 "old_owned_count": len(old_owners)}
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
                "suite_id": "qwen38-context-repair-kv4350-84-v1",
                "source_manifest_sha256": manifest_sha,
                "source_combined_sha256": PREVIOUS_COMBINED_SHA,
                "selection_sha256": manifest["files"]["selection.json"],
                "fixture_sha256": FIXTURE_SHA,
                "legacy_controller_report_sha256": sha(ROOT/"results/window-report.json"),
                "storage_available_post_restore_bytes": post_free,
                "storage_owned_containers": dict(storage.containers),
                "storage_post_restore": storage.last, "storage_error": storage_error,
                "post_restore_storage_bound_ok": storage_error is None}
            save(ROOT/"results/context-window-report.json", context_report)
            continuation.storage_check(storage)
            grader = load("kv4350_grader", ROOT/"grade_repair.py")
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
