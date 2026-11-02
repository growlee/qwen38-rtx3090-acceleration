"""Grade only r2 captures once, then merge with immutable r1 scores."""
from collections import Counter
import json
import os
import sys

from continue_context import (ROOT, R1, PREFIX, check_bundle, check_source,
                              require, save, sha, storage_check, validate_selection)
from quality_context_remaining import capture_rows, require_restored
from context_storage import StorageError, install_sandbox


def merge_scores(settings, suite, partition, selection, previous, current):
    """Pure merge: exactly 612 keys, no replacement of a captured r1 answer."""
    expected = {(m['id'], c['case_id']): c for m in settings['modes'] for c in suite['cases']
                if c['target_tokens'] in m['lengths']}
    require(len(expected) == 612, 'merged_matrix_changed')
    prior = {(r['mode_id'], r['case_id']): r for r in previous}
    new = {(r['mode_id'], r['case_id']): r for r in current}
    require(len(prior) == len(previous) and set(prior) == set(expected), 'prior_scores_incomplete_or_duplicate')
    selected = {(mode, case_id) for mode, value in selection['modes'].items() for case_id in value['case_ids']}
    require(len(new) == len(current) and set(new) == selected and selected <= set(expected),
            'continuation_scores_incomplete_or_duplicate')
    captured = {(mode, case_id) for mode, value in partition['modes'].items()
                for case_id in value['captured_case_ids']}
    require(not (captured & selected), 'merge_would_replace_captured_answer')
    for rows in (prior, new):
        for key, row in rows.items():
            require(row['request_sha256'] == expected[key]['request_sha256'], 'score_request_identity_changed')
            require(row['outcome'] in ('PASS', 'FAIL', 'BLOCKED'), 'score_not_terminal')
    # A previously scored PASS/FAIL cannot be replaced even if a malformed
    # partition falsely says the request was untouched.
    require(all(prior[key]['outcome'] == 'BLOCKED' for key in selected), 'merge_would_replace_valid_score')
    return [{**(new[key] if key in new else prior[key]), 'result_window': 'r2' if key in new else 'r1'}
            for key in expected]


def main(*, storage=None):
    require(storage is not None, 'grading_requires_owning_controller_storage')
    manifest, r1 = check_bundle(sys.argv[1])
    settings, suite, previous_window, partition = check_source(manifest)
    selection = json.loads((ROOT/'selection.json').read_bytes())
    selected_modes = validate_selection(selection, settings, suite, previous_window, partition)
    results = ROOT/'results'
    window = json.loads((results/'context-window-report.json').read_bytes())
    require_restored(window)
    require(window['source_manifest_sha256'] == sys.argv[1]
            and window['selection_sha256'] == manifest['files']['selection.json']
            and window['fixture_sha256'] == settings['fixture_sha256'], 'continuation_report_identity_changed')
    require(set(window['modes']) == set(selected_modes), 'continuation_report_mode_set_changed')
    # Existing partial score files are evidence of a prior grading attempt.
    # Never restart the grader blindly after an interrupted sandbox evaluation.
    require(not any((results/('scores-'+m+'.jsonl')).exists() for m in selected_modes),
            'continuation_grading_already_attempted')
    def check_disk():
        storage_check(storage)
    # Always checked, including empty/non-code-only selections, before grading.
    check_disk()
    sys.path.insert(0, str(r1.OLD/'sources'))
    from quality_client import evaluate_noncode
    import quality_sandbox
    quality_sandbox.CONTAINER_PREFIX = PREFIX+'-code'
    install_sandbox(quality_sandbox, storage)
    all_cases = {c['case_id']: c for c in suite['cases']}
    current_scores = []
    fenced = False
    for mode_id in selected_modes:
        source = results/(mode_id+'.jsonl')
        raw = source.read_bytes() if source.exists() else b''
        envelopes, partial_tail = capture_rows(raw)
        rows = [r for r in envelopes if r.get('kind') == 'case']
        by_id = {r['case_id']: r for r in rows}
        ids = selection['modes'][mode_id]['case_ids']
        require(len(by_id) == len(rows) and set(by_id) <= set(ids), 'continuation_capture_case_set_changed')
        require(all(r['mode_id'] == mode_id and r['request_sha256'] == all_cases[r['case_id']]['request_sha256']
                    for r in rows), 'continuation_capture_identity_changed')
        entry = window['modes'][mode_id]
        if entry.get('status') == 'CAPTURED':
            require(not partial_tail and list(by_id) == ids and sha(source) == entry['client']['sha256'],
                    'continuation_captured_profile_mismatch')
        resource_path = results/(mode_id+'-resources.json')
        samples = json.loads(resource_path.read_bytes()) if resource_path.exists() else []
        with (results/('scores-'+mode_id+'.jsonl')).open('x', encoding='utf-8') as stream:
            for case_id in ids:
                case = all_cases[case_id]
                row = by_id.get(case_id)
                item = dict(mode_id=mode_id, case_id=case_id, family_id=case['family_id'],
                            target_tokens=case['target_tokens'], request_sha256=case['request_sha256'],
                            cold_continuation=selection['modes'][mode_id]['skip_recorded_warmup'],
                            incomplete_final_envelope=partial_tail)
                if row is None:
                    verdict = dict(outcome='BLOCKED', error=entry.get('abort_code', entry.get('reason', 'not_dispatched')))
                elif row.get('outcome') in ('error', 'blocked'):
                    verdict = dict(outcome='BLOCKED', error=row.get('error'))
                elif row.get('error') in ('finish_reason_length', 'empty_answer'):
                    verdict = dict(outcome='FAIL', error=row['error'])
                elif case['check']['type'] == 'python_function_cases':
                    try:
                        check_disk()
                    except StorageError:
                        fenced = True
                    if fenced:
                        verdict = dict(outcome='BLOCKED', error='sandbox_safety_fence')
                    else:
                        try:
                            verdict = quality_sandbox.evaluate_code(case, row.get('content', ''), mode_id=mode_id)
                            check_disk()
                        except StorageError as exc:
                            verdict = dict(outcome='BLOCKED', error=exc.code)
                        if verdict['outcome'] == 'BLOCKED':
                            fenced = True
                else:
                    passed = evaluate_noncode(case, row)
                    verdict = dict(outcome='PASS' if passed else 'FAIL', error=None if passed else 'quality_check_failed')
                item.update(verdict)
                if row:
                    for key in ('input_tokens', 'request_bytes', 'context_limit', 'usage', 'finish_reason',
                                'elapsed_seconds', 'ttft_first_generated_seconds', 'ttft_first_content_or_tool_seconds',
                                'decode_tokens_per_second', 'started_epoch', 'ended_epoch'):
                        item[key] = row.get(key)
                    taken = [s for s in samples if row.get('started_epoch', float('inf')) <= s[0] <= row.get('ended_epoch', 0)]
                    item['resource_samples'] = len(taken)
                    item['max_vram_mib'] = max((s[1] for s in taken), default=None)
                    item['min_host_available_bytes'] = min((s[7] for s in taken), default=None)
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                current_scores.append(item)
    # Final r1 scoring was completed once on PC-A after the server-side
    # restoration blocker. check_source binds this exact consolidated file.
    previous_scores = [json.loads(line) for line in
        (ROOT/'final-r1/r1-final-scores.jsonl').read_bytes().splitlines()]
    merged = merge_scores(settings, suite, partition, selection, previous_scores, current_scores)
    with (results/'combined-scores.jsonl').open('x', encoding='utf-8') as stream:
        for row in merged:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    check_disk()  # A complete score set never hides post-grading disk exhaustion.
    summary = {'coordinates': 612, 'newly_graded_coordinates': len(current_scores), 'sandbox_fenced': fenced,
               'storage_owned_containers': dict(storage.containers), 'storage': storage.last,
               'combined_scores_sha256': sha(results/'combined-scores.jsonl'),
               'modes': {mode['id']: dict(Counter(r['outcome'] for r in merged if r['mode_id'] == mode['id']))
                         for mode in settings['modes']}}
    save(results/'grading-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
