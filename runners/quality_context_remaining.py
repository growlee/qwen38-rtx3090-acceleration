"""Read-only selection of untouched coordinates after a restored context window.

This does not launch inference, authorize retries, or grade retained answers.
The original frozen fixture and captured bytes remain the source of truth.
"""
from __future__ import annotations

import hashlib
import json


LENGTHS = (8192, 32768, 50000, 96000, 128000, 160000, 200000)
EXPECTED = {
    **{f'{name}-mtp-{ctx}': LENGTHS[:3] if ctx == 'fast' else LENGTHS
       for name in ('autoround', 'int8', 'adaptive', 'reconstructed')
       for ctx in ('fast', 'huge')},
    'dflash2-fast-n7': LENGTHS[:3],
    'dflash2-huge-n7': LENGTHS,
    'original-gguf': LENGTHS[:1],
}


def require(condition, error):
    if not condition:
        raise ValueError(error)


def capture_rows(raw):
    """Keep a valid final JSON envelope even when its trailing LF was lost."""
    lines = raw.splitlines()
    rows = []
    partial = False
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1 or raw.endswith(b'\n'):
                raise
            partial = True
    return rows, partial


def require_restored(window):
    restored = window.get('restore', {})
    require(window.get('outcome') in ('captured_restored', 'failed_restored'),
            'source_window_not_terminal_restored')
    require(restored.get('admission_public') is True and restored.get('qwen_ready') is True
            and restored.get('gateway_completions') == 2 and not window.get('restore_error_code'),
            'source_public_restore_unproven')
    before, after = window.get('production_prestate', {}), restored.get('container', {})
    require(all(isinstance(before.get(key), str) and before[key] and after.get(key) == before[key]
                for key in ('id', 'image', 'model')) and after.get('running') is True,
            'source_restore_identity_unproven')


def require_repaired_restoration(window, restoration):
    """Prove a later repair without rewriting the original failed report."""
    require(window.get('outcome') == 'restore_unknown'
            and window.get('error_code') == 'storage_budget_exhausted'
            and window.get('restore_error_code') == 'operator_command_failed'
            and window.get('restore') is None, 'source_storage_stop_identity_changed')
    require(restoration.get('public_restored') is True
            and restoration.get('qwen_ready') is True
            and restoration.get('gateway_completions') == 2
            and restoration.get('admission', {}).get('phase') == 'public',
            'repaired_public_restore_unproven')
    before = window.get('production_prestate', {})
    after = restoration.get('production_after', {})
    require(all(isinstance(before.get(key), str) and before[key]
                and after.get(key) == before[key] for key in ('id', 'image', 'model'))
            and after.get('running') is True, 'repaired_restore_identity_unproven')


def _select_remaining(settings, suite, window, captures, invalidated=None):
    """Partition the matrix, holding only an already-proven invalid envelope."""
    invalidated = invalidated or set()
    require(len(invalidated) <= 1, 'invalidated_dispatch_set_changed')
    modes = settings['modes']
    require([m['id'] for m in modes] == list(EXPECTED), 'source_mode_matrix_changed')
    require(settings.get('coordinates') == 612, 'source_coordinate_count_changed')
    require(set(captures).issubset(EXPECTED), 'unexpected_capture_mode')
    require(set(window['modes']) == set(EXPECTED), 'source_report_modes_changed')
    cases = suite['cases']
    require(suite.get('suite_id') == 'qwen38-context-12x7-v1', 'source_suite_changed')
    case_ids = [c['case_id'] for c in cases]
    require(len(case_ids) == len(set(case_ids)) == 84, 'source_fixture_case_set_changed')
    require([c['target_tokens'] for c in cases] == [t for t in LENGTHS for _ in range(12)],
            'source_fixture_length_order_changed')
    result = {'coordinates': 612, 'captured': 0, 'untouched': 0,
              'unresolved_dispatches': 0, 'modes': {}}
    seen_invalidated = set()
    for mode in modes:
        mode_id = mode['id']
        require(tuple(mode['lengths']) == EXPECTED[mode_id], 'source_mode_lengths_changed')
        expected = {c['case_id']: c for c in cases if c['target_tokens'] in mode['lengths']}
        raw = captures.get(mode_id, b'')
        require(isinstance(raw, bytes), 'capture_must_be_raw_bytes')
        rows, partial_tail = capture_rows(raw)
        complete, started = {}, set()
        warmups = []
        for row in rows:
            require(row.get('mode_id') == mode_id, 'capture_mode_identity_mismatch')
            kind = row.get('kind')
            require(kind in ('case', 'case_start', 'warmup'), 'capture_kind_unknown')
            if kind == 'warmup':
                warmups.append(row)
                continue
            case_id = row.get('case_id')
            require(case_id in expected, 'capture_case_outside_matrix')
            if kind == 'case_start':
                require(case_id not in started and case_id not in complete,
                        'duplicate_or_late_dispatch')
                started.add(case_id)
                continue
            require(case_id not in complete, 'duplicate_captured_answer')
            require(row.get('request_sha256') == expected[case_id]['request_sha256'],
                    'capture_request_identity_mismatch')
            require(case_id in started or row.get('error') == 'capacity_bound_exceeded',
                    'captured_answer_without_dispatch')
            key = (mode_id, case_id)
            if key in invalidated:
                require(key not in seen_invalidated, 'duplicate_invalidated_envelope')
                seen_invalidated.add(key)
            else:
                # Error envelopes normally remain completed and cannot be
                # replayed. Only the separately proven key above is held.
                complete[case_id] = row
        require(len(warmups) <= 1, 'duplicate_warmup')
        entry = window['modes'][mode_id]
        if entry.get('status') == 'CAPTURED':
            require(not partial_tail and set(complete) == set(expected),
                    'captured_profile_case_set_incomplete')
            require(hashlib.sha256(raw).hexdigest() == entry['client']['sha256'],
                    'captured_profile_sha_mismatch')
        held = [key for key in expected if key in started and key not in complete]
        untouched = [key for key in expected if key not in started and key not in complete]
        result['modes'][mode_id] = {
            'captured_case_ids': [key for key in expected if key in complete],
            'untouched_case_ids': untouched,
            'unresolved_dispatch_case_ids': held,
            'partial_final_envelope': partial_tail,
            'warmup_recorded': bool(warmups),
            'capture_bytes': len(raw),
            'capture_sha256': hashlib.sha256(raw).hexdigest(),
        }
        result['captured'] += len(complete)
        result['untouched'] += len(untouched)
        result['unresolved_dispatches'] += len(held)
    require(seen_invalidated == invalidated, 'invalidated_envelope_not_found')
    require(sum(result[key] for key in ('captured', 'untouched', 'unresolved_dispatches')) == 612,
            'continuation_partition_incomplete')
    return result


def select_remaining_after_repair(settings, suite, window, captures, restoration,
                                  final_scores, final_audit):
    """Adapt the one exact r1 storage interruption to a safe continuation."""
    require_repaired_restoration(window, restoration)
    require(final_audit.get('audit') == 'PASS' and final_audit.get('coordinates') == 612
            and final_audit.get('capture') == {'complete': 354, 'interrupted': 1,
                                                'not_dispatched': 257},
            'final_r1_audit_contract_changed')
    key = ('adaptive-mtp-huge', 'context-version-start-t200000')
    expected = {(m['id'], c['case_id']): c for m in settings['modes'] for c in suite['cases']
                if c['target_tokens'] in m['lengths']}
    scores = {(row['mode_id'], row['case_id']): row for row in final_scores}
    require(len(scores) == len(final_scores) == len(expected) == 612
            and set(scores) == set(expected), 'final_r1_scores_incomplete_or_duplicate')
    score = scores[key]
    require(score.get('outcome') == 'BLOCKED' and score.get('capture') == 'interrupted'
            and score.get('dispatched') is True
            and score.get('error') == 'stream_incomplete'
            and score.get('safety_cause') == 'storage_budget_exhausted'
            and score.get('request_sha256') == expected[key]['request_sha256']
            and score.get('ttft_first_generated_seconds') is None
            and score.get('ttft_first_content_or_tool_seconds') is None
            and score.get('usage') is None, 'final_r1_interruption_score_changed')
    raw_rows, partial = capture_rows(captures[key[0]])
    envelope = [row for row in raw_rows if row.get('kind') == 'case'
                and row.get('case_id') == key[1]]
    require(not partial and len(envelope) == 1, 'interrupted_raw_envelope_changed')
    row = envelope[0]
    require(row.get('outcome') == 'error' and row.get('error') == 'stream_incomplete'
            and row.get('content') == '' and row.get('reasoning') == ''
            and row.get('tool_calls') == [] and row.get('sse_bytes') == 0
            and row.get('finish_reason') is None and row.get('usage') is None
            and row.get('ttft_first_generated_seconds') is None
            and row.get('ttft_first_content_or_tool_seconds') is None
            and row.get('decode_tokens_per_second') is None,
            'interrupted_raw_not_zero_token_transport_abort')
    entry = window['modes'][key[0]]
    require(entry.get('status') == 'FAIL' and entry.get('reason') == 'storage_budget_exhausted',
            'interrupted_mode_storage_cause_changed')
    for role in ('client', 'candidate'):
        cleanup = entry.get('cleanup', {}).get(role, {})
        require(cleanup.get('retained') is True
                and isinstance(cleanup.get('id'), str) and len(cleanup['id']) == 64,
                'interrupted_process_cleanup_unproven')
    result = _select_remaining(settings, suite, window, captures, {key})
    require((result['captured'], result['untouched'], result['unresolved_dispatches']) == (354, 257, 1),
            'final_r1_partition_changed')
    result['invalidation_proof'] = {'mode_id': key[0], 'case_id': key[1],
        'reason': 'storage_abort_stream_incomplete_no_first_token',
        'request_sha256': expected[key]['request_sha256']}
    return result


def select_remaining(settings, suite, window, captures):
    """Partition all 612 coordinates; incomplete dispatches are held, not retried.

    Caller must bind settings/suite/window to their immutable source receipts
    before using this result to launch anything. Captures maps mode IDs to raw
    JSONL bytes; absent modes have no capture file. No IO or runtime mutation.
    """
    require_restored(window)
    require(window.get('error_code') in (None, 'window_restore_reserve_reached'),
            'source_non_time_safety_blocker')
    return _select_remaining(settings, suite, window, captures)
