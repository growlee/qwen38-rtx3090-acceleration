"""One-shot grading adapter for immutable r2 captures after later restoration.

This never launches inference or changes the failed capture report. It binds a
separate restoration receipt, reconstructs exact container ownership, then
calls the SHA-pinned grader once.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path('/srv/ai-server/benchmarks/context-20260908-r2')
SOURCE_MANIFEST_SHA = '89e8667d4c2031b6ce91fe0f73b206592ea9a6f4066b2e9328b4d570e209299e'
CONTEXT_REPORT_SHA = '19576e40e10a0977b3c62373a3614e7457ec5488dae037fe4e276bb4b9a8a80d'
LEGACY_REPORT_SHA = 'b16794d81771985c45d104eaeaab78dc3c00117903a1797a73a8ec653f89488a'
GRADER_SHA = 'e89b034035990aa87906a84a40ea03f16f71e0fe956ef98dfde57c3b48a4261b'
STORAGE_SHA = '2085436ed77ec5a9f64d61568bb612254cb58158027869fc036fa666ff914cfd'
CAPTURE_MODES = ('adaptive-mtp-huge', 'reconstructed-mtp-fast',
                 'reconstructed-mtp-huge', 'dflash2-fast-n7', 'dflash2-huge-n7')
HEX = re.compile(r'[0-9a-f]{64}')


def need(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expected_capture_files():
    return {f'results/{mode}{suffix}' for mode in CAPTURE_MODES
            for suffix in ('.jsonl', '-resources.json')}


def validate_restoration(window, receipt, report_sha):
    need(report_sha == CONTEXT_REPORT_SHA
         and window.get('outcome') == 'restore_unknown'
         and window.get('restore_error_code') == 'qwen_restore_not_ready'
         and window.get('source_manifest_sha256') == SOURCE_MANIFEST_SHA,
         'failed_capture_report_contract_changed')
    need(receipt.get('source_report_sha256') == report_sha
         and receipt.get('outcome') == 'failed_restored'
         and receipt.get('benchmark_dispatched') is False,
         'later_restoration_receipt_not_terminal')
    restored = receipt.get('restoration', {})
    need(restored.get('qwen_ready') is True
         and restored.get('admission_public') is True
         and restored.get('gateway_completions') == 2,
         'later_public_restoration_unproven')
    before, after = window.get('production_prestate', {}), restored.get('container', {})
    need(all(isinstance(before.get(key), str) and before[key]
             and after.get(key) == before[key] for key in ('id', 'image', 'model'))
         and after.get('running') is True, 'later_restoration_identity_changed')


def validate_capture(root, window, integrity, selection, suite, capture_rows):
    need(integrity.get('schema') == 'qwen-context-r2-capture-integrity/v1'
         and integrity.get('source_manifest_sha256') == SOURCE_MANIFEST_SHA
         and integrity.get('context_report_sha256') == CONTEXT_REPORT_SHA
         and integrity.get('legacy_report_sha256') == LEGACY_REPORT_SHA,
         'capture_integrity_identity_changed')
    need(set(integrity.get('files', {})) == expected_capture_files(),
         'capture_integrity_file_set_changed')
    for mode in selection['modes']:
        for suffix in ('.jsonl', '-resources.json'):
            name = f'results/{mode}{suffix}'
            if name not in integrity['files']:
                need(not (root/name).exists(), 'unbound_selected_capture_input:'+name)
    for name, identity in integrity['files'].items():
        path = root/name
        need(path.stat().st_size == identity.get('bytes')
             and sha(path) == identity.get('sha256'), 'capture_file_changed:'+name)
    need(sha(root/'results/window-report.json') == LEGACY_REPORT_SHA
         and window.get('legacy_controller_report_sha256') == LEGACY_REPORT_SHA,
         'legacy_capture_report_changed')
    owners = window.get('storage_owned_containers', {})
    need(len(owners) == 10 and owners == integrity.get('storage_owned_containers'),
         'capture_owned_container_set_changed')
    cleanup = {}
    for mode in CAPTURE_MODES:
        entry = window['modes'][mode]
        need(entry.get('attempted') is True, 'attempted_capture_mode_changed')
        for role in ('candidate', 'client'):
            item = entry.get('cleanup', {}).get(role, {})
            need(item.get('retained') is True and HEX.fullmatch(item.get('id', '')),
                 'capture_cleanup_identity_missing')
            cleanup[item['id']] = owners.get(item['id'])
    need(cleanup == owners and all(cleanup.values()), 'capture_cleanup_owner_mismatch')

    cases = {case['case_id']: case for case in suite['cases']}
    normal = errors = code = 0
    for mode in CAPTURE_MODES:
        raw = (root/'results'/(mode+'.jsonl')).read_bytes()
        rows, partial = capture_rows(raw)
        need(not partial, 'capture_partial_tail:'+mode)
        selected = set(selection['modes'][mode]['case_ids'])
        for row in rows:
            if row.get('kind') != 'case':
                continue
            need(row.get('mode_id') == mode and row.get('case_id') in selected,
                 'capture_coordinate_outside_selection')
            case = cases[row['case_id']]
            need(row.get('request_sha256') == case['request_sha256'],
                 'capture_request_identity_changed')
            if row.get('outcome') in ('error', 'blocked'):
                errors += 1
            else:
                normal += 1
                code += case['check']['type'] == 'python_function_cases'
    need((normal, errors, code) == (162, 1, 42), 'capture_grading_cardinality_changed')
    return owners, {'normal_captures': normal, 'error_captures': errors,
                    'captured_code_cases': code}


def inspect_and_rebind(storage, owners, inspect=None):
    inspect = inspect or (lambda cid: json.loads(subprocess.check_output(
        ['docker', 'inspect', cid], timeout=30))[0])
    for cid, name in sorted(owners.items()):
        row = inspect(cid)
        state = row.get('State', {})
        need(row.get('Id') == cid and row.get('Name') == '/'+name,
             'owned_container_live_identity_changed')
        need(state.get('Running') is False and state.get('Restarting') is False
             and state.get('OOMKilled') is False, 'owned_container_not_cleanly_stopped')
        storage.register_container(cid, name)


def run(manifest_sha, restoration_sha, integrity_sha, *, root=ROOT):
    need(os.geteuid() == 0, 'root_required')
    need(manifest_sha == SOURCE_MANIFEST_SHA
         and HEX.fullmatch(restoration_sha or '') and HEX.fullmatch(integrity_sha or ''),
         'grading_argument_identity_changed')
    need(sha(root/'grade_context_continuation.py') == GRADER_SHA
         and sha(root/'context_storage.py') == STORAGE_SHA,
         'frozen_grading_source_changed')
    report_path = root/'results/context-window-report.json'
    report_sha = sha(report_path)
    need(report_sha == CONTEXT_REPORT_SHA, 'failed_capture_report_changed')
    window = json.loads(report_path.read_bytes())
    restoration_path = root/'restoration-after-abort.json'
    integrity_path = root/'capture-integrity.json'
    need(sha(restoration_path) == restoration_sha, 'restoration_receipt_changed')
    need(sha(integrity_path) == integrity_sha, 'capture_integrity_receipt_changed')
    restoration = json.loads(restoration_path.read_bytes())
    integrity = json.loads(integrity_path.read_bytes())
    validate_restoration(window, restoration, report_sha)

    sys.path.insert(0, str(root))
    continuation = load('grading_continuation_contract', root/'continue_context.py')
    manifest, r1 = continuation.check_bundle(manifest_sha)
    settings, suite, previous_window, partition = continuation.check_source(manifest)
    selection = json.loads((root/'selection.json').read_bytes())
    continuation.validate_selection(selection, settings, suite, previous_window, partition)
    # check_source imported these exact ROOT modules under their canonical
    # names. Reuse them so StorageError identity remains the same inside the
    # frozen grader's exception fences.
    remaining = sys.modules['quality_context_remaining']
    owners, cardinality = validate_capture(root, window, integrity, selection, suite,
                                           remaining.capture_rows)
    selected = sum(len(value['case_ids']) for value in selection['modes'].values())
    need(selected == 258, 'grading_selection_cardinality_changed')
    results = root/'results'
    forbidden = [results/('scores-'+mode+'.jsonl') for mode in selection['modes']]
    forbidden += [results/'combined-scores.jsonl', results/'grading-summary.json']
    need(not any(path.exists() for path in forbidden), 'continuation_grading_already_attempted')

    storage_module = sys.modules['context_storage']
    need(Path(storage_module.__file__).resolve() == (root/'context_storage.py').resolve(),
         'grading_storage_module_origin_changed')
    storage = storage_module.OwnedStorage(root)
    inspect_and_rebind(storage, owners)
    storage.check(storage_module.WRITE_RESERVE)

    attempt_path = root/'grading-attempt.json'
    with attempt_path.open('x', encoding='utf-8') as attempt:
        proof = {'schema': 'qwen-context-r2-grading-attempt/v1',
                 'started_epoch': time.time(), 'outcome': 'grading_unknown',
                 'benchmark_dispatched': False, 'source_manifest_sha256': manifest_sha,
                 'context_report_sha256': report_sha,
                 'restoration_sha256': restoration_sha,
                 'capture_integrity_sha256': integrity_sha,
                 'storage_owned_containers': owners, **cardinality}
        try:
            attempt.write(json.dumps(proof, sort_keys=True)); attempt.flush(); os.fsync(attempt.fileno())
            grader = load('frozen_context_grader', root/'grade_context_continuation.py')

            def require_external_restore(observed):
                need(observed == window and sha(report_path) == report_sha,
                     'failed_capture_report_changed_during_grading')
                validate_restoration(observed, restoration, report_sha)

            grader.require_restored = require_external_restore
            original_argv = sys.argv
            try:
                sys.argv = [str(root/'grade_context_continuation.py'), manifest_sha]
                grader.main(storage=storage)
            finally:
                sys.argv = original_argv
            summary = json.loads((results/'grading-summary.json').read_bytes())
            need(summary.get('coordinates') == 612
                 and summary.get('newly_graded_coordinates') == 258,
                 'continuation_grading_summary_changed')
            proof.update(outcome='grading_complete', ended_epoch=time.time(),
                         combined_scores_sha256=sha(results/'combined-scores.jsonl'),
                         grading_summary_sha256=sha(results/'grading-summary.json'),
                         storage=storage.last)
            return proof
        except Exception as exc:
            proof.update(outcome='grading_failed', ended_epoch=time.time(),
                         error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            attempt.seek(0); attempt.truncate()
            attempt.write(json.dumps(proof, indent=2, sort_keys=True)+'\n')
            attempt.flush(); os.fsync(attempt.fileno())


def main():
    need(len(sys.argv) == 4, 'usage: grade_r2_after_restore.py MANIFEST_SHA RESTORATION_SHA INTEGRITY_SHA')
    result = run(*sys.argv[1:])
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
