"""Grade only 84 kv4350 captures once; reference the completed original matrix."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPAIRED = 'dflash2-huge-n7-kv4350mb-204800'
MODES = (REPAIRED,)
R2 = Path('/srv/ai-server/benchmarks/context-20260908-r2')
PRIOR_PATH = Path('/srv/ai-server/benchmarks/context-20260910-repair-clientfix/results/combined-original-scores.jsonl')
PRIOR_SHA = '4f47916f1f0444478cc408d2a0610c2781777fdbfd1579f6bd83d589e320391e'
FIXTURE_SHA = '2df1be2cff66b53ce57b3eefe3eed8be2d16c59d1e37b783341f176995689d94'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def write_rows(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def grade(*, root, r1, continuation, storage, manifest_sha):
    root = Path(root); results = root/'results'
    need(storage is not None, 'owning_storage_required')
    need(sha(root/'manifest.json') == manifest_sha, 'repair_manifest_changed')
    manifest = json.loads((root/'manifest.json').read_bytes())
    need(sha(root/'selection.json') == manifest['files']['selection.json'], 'repair_selection_changed')
    selection = json.loads((root/'selection.json').read_bytes())
    need(set(selection['modes']) == set(MODES), 'repair_mode_set')
    fixture_path = Path(r1.ROOT)/'frozen/frozen.json'
    need(sha(fixture_path) == FIXTURE_SHA, 'fixture_changed')
    cases = json.loads(fixture_path.read_bytes())['cases']
    by_case = {c['case_id']: c for c in cases}
    need(len(cases) == len(by_case) == 84, 'fixture_case_set')
    expected_ids = {
        REPAIRED: ['context-fact-start-t200000'] +
                  [c['case_id'] for c in cases if c['case_id'] != 'context-fact-start-t200000'],
    }
    for mode in MODES:
        need(selection['modes'][mode]['case_ids'] == expected_ids[mode], 'repair_case_order')
    prior_path = PRIOR_PATH
    need(sha(prior_path) == PRIOR_SHA, 'prior_scores_changed')
    previous = [json.loads(line) for line in prior_path.read_bytes().splitlines()]
    need(len(previous) == 612 and len({(r['mode_id'], r['case_id']) for r in previous}) == 612
         and Counter(r['outcome'] for r in previous) == {'PASS': 527, 'FAIL': 1, 'BLOCKED': 84},
         'prior_completed_matrix_changed')
    gguf = [r for r in previous if r['mode_id'] == 'original-gguf']
    need(len(gguf) == 12 and all(r['outcome'] == 'PASS' for r in gguf), 'completed_gguf_changed')
    # These imports reuse the exact source modules verified by the controller.
    sys.path.insert(0, str(r1.OLD/'sources'))
    sys.path.insert(0, str(R2))
    from quality_context_remaining import capture_rows, require_restored
    from context_storage import install_sandbox, WRITE_RESERVE
    from quality_client import evaluate_noncode
    import quality_sandbox
    window = json.loads((results/'context-window-report.json').read_bytes())
    require_restored(window)
    need(window.get('source_manifest_sha256') == manifest_sha
         and window.get('fixture_sha256') == FIXTURE_SHA
         and window.get('selection_sha256') == manifest['files']['selection.json']
         and set(window['modes']) == set(MODES), 'repair_report_identity')
    need(window.get('storage_owned_containers') == dict(storage.containers), 'capture_ownership_changed')
    for cid, name in storage.containers.items():
        row = json.loads(subprocess.check_output(['docker', 'inspect', cid], timeout=30))[0]
        need(row['Id'] == cid and row['Name'] == '/'+name
             and row['State']['Running'] is False and row['State']['Restarting'] is False,
             'owned_capture_not_stopped')
    storage.check(WRITE_RESERVE)
    outputs = [results/('scores-'+mode+'.jsonl') for mode in MODES]
    outputs += [results/name for name in ('repaired-profile-scores.jsonl', 'grading-summary.json')]
    need(not any(p.exists() for p in outputs), 'grading_already_attempted')
    captured, resources, fingerprints, started = {}, {}, {}, {}
    for mode in MODES:
        path = results/(mode+'.jsonl'); resource = results/(mode+'-resources.json')
        entry = window['modes'][mode]
        if path.exists() or resource.exists():
            need(entry.get('attempted') is True and path.exists() and resource.exists(),
                 'unbound_capture_files')
            client = entry.get('cleanup', {}).get('client', {})
            need(client.get('retained') is True and client.get('id') in storage.containers,
                 'capture_client_ownership_missing')
        raw = path.read_bytes() if path.exists() else b''
        envelopes, partial = capture_rows(raw)
        rows = [r for r in envelopes if r.get('kind') == 'case']
        ids = [r['case_id'] for r in rows]
        need(ids == expected_ids[mode][:len(ids)] and len(ids) == len(set(ids)), 'capture_prefix_changed')
        need(all(r['mode_id'] == mode and r['request_sha256'] == by_case[r['case_id']]['request_sha256']
                 for r in rows), 'capture_request_changed')
        if entry.get('status') == 'CAPTURED':
            need(not partial and ids == expected_ids[mode]
                 and sha(path) == entry['client']['sha256'], 'complete_capture_mismatch')
        captured[mode] = {r['case_id']: r for r in rows}
        starts = [r for r in envelopes if r.get('kind') == 'case_start']
        start_ids = [r['case_id'] for r in starts]
        started[mode] = set(start_ids)
        need(all(r.get('mode_id') == mode for r in starts)
             and start_ids == [cid for cid in expected_ids[mode] if cid in started[mode]],
             'unknown_duplicate_or_reordered_dispatch')
        resources[mode] = json.loads(resource.read_bytes()) if resource.exists() else []
        for p in (path, resource):
            if p.exists(): fingerprints[p.name] = sha(p)
    proof = {'started_epoch': time.time(), 'outcome': 'grading_unknown',
             'source_manifest_sha256': manifest_sha, 'source_scores_sha256': PRIOR_SHA,
             'capture_sha256': fingerprints, 'context_report_sha256': sha(results/'context-window-report.json')}
    # Claim the attempt before any code execution. Never restart partial grading.
    with (root/'grading-attempt.json').open('x', encoding='utf-8') as attempt:
        attempt.write(json.dumps(proof)); attempt.flush(); os.fsync(attempt.fileno())
        try:
            quality_sandbox.CONTAINER_PREFIX = 'qwen-context-20260910-repair-kv4350-code'
            install_sandbox(quality_sandbox, storage)
            scored = {}
            fenced = False
            for mode in MODES:
                scored[mode] = []
                with (results/('scores-'+mode+'.jsonl')).open('x', encoding='utf-8') as stream:
                    for cid in expected_ids[mode]:
                        case, row = by_case[cid], captured[mode].get(cid)
                        item = dict(mode_id=mode, case_id=cid, family_id=case['family_id'],
                                    target_tokens=case['target_tokens'], request_sha256=case['request_sha256'],
                                    result_window='repair-kv4350-20260910',
                                    dispatched=cid in started[mode] or
                                    (row is not None and row.get('error') != 'capacity_bound_exceeded'))
                        if row is None:
                            reason = window['modes'][mode].get('reason', 'not_dispatched')
                            verdict = dict(outcome='BLOCKED', error=(
                                'dispatched_without_terminal_envelope:'+str(reason)
                                if cid in started[mode] else reason))
                        elif row.get('outcome') in ('error', 'blocked'):
                            verdict = dict(outcome='BLOCKED', error=row.get('error'))
                        elif row.get('error') in ('finish_reason_length', 'empty_answer'):
                            verdict = dict(outcome='FAIL', error=row['error'])
                        elif case['check']['type'] == 'python_function_cases':
                            if fenced:
                                verdict = dict(outcome='BLOCKED', error='sandbox_safety_fence')
                            else:
                                storage.check(WRITE_RESERVE)
                                verdict = quality_sandbox.evaluate_code(case, row.get('content', ''), mode_id=mode)
                                storage.check(WRITE_RESERVE)
                                fenced = verdict['outcome'] == 'BLOCKED'
                        else:
                            passed = evaluate_noncode(case, row)
                            verdict = dict(outcome='PASS' if passed else 'FAIL', error=None if passed else 'quality_check_failed')
                        item.update(verdict)
                        if row:
                            for key in ('input_tokens', 'request_bytes', 'context_limit', 'usage', 'finish_reason',
                                        'elapsed_seconds', 'ttft_first_generated_seconds', 'ttft_first_content_or_tool_seconds',
                                        'decode_tokens_per_second', 'started_epoch', 'ended_epoch'):
                                item[key] = row.get(key)
                            item['content_sha256'] = hashlib.sha256(row.get('content', '').encode()).hexdigest()
                            taken = [s for s in resources[mode] if row.get('started_epoch', float('inf')) <= s[0] <= row.get('ended_epoch', 0)]
                            item['resource_samples'] = len(taken)
                            item['max_vram_mib'] = max((s[1] for s in taken), default=None)
                            item['min_host_available_bytes'] = min((s[7] for s in taken), default=None)
                        stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True)+'\n')
                        stream.flush(); os.fsync(stream.fileno())
                        scored[mode].append(item)
            write_rows(results/'repaired-profile-scores.jsonl', scored[REPAIRED])
            storage.check(WRITE_RESERVE)
            summary = {'original_coordinates': 612, 'repaired_coordinates': 84,
                       'newly_graded_coordinates': 84, 'sandbox_fenced': fenced,
                       'original_counts': dict(Counter(r['outcome'] for r in previous)),
                       'new_modes': {m: dict(Counter(r['outcome'] for r in scored[m])) for m in MODES},
                       'combined_original_sha256': PRIOR_SHA,
                       'combined_original_path': str(PRIOR_PATH),
                       'original_matrix_reused_without_grading': True,
                       'repaired_profile_sha256': sha(results/'repaired-profile-scores.jsonl'),
                       'storage_owned_containers': dict(storage.containers), 'storage': storage.last}
            save(results/'grading-summary.json', summary)
            proof.update(outcome='grading_complete', summary_sha256=sha(results/'grading-summary.json'))
            return summary
        except Exception as exc:
            proof.update(outcome='grading_failed', error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            proof['ended_epoch'] = time.time()
            attempt.seek(0); attempt.truncate()
            json.dump(proof, attempt, sort_keys=True, indent=2)
            attempt.write('\n'); attempt.flush(); os.fsync(attempt.fileno())
