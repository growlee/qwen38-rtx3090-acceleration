import copy
import hashlib
import json
import unittest

from quality_context_remaining import EXPECTED, LENGTHS, select_remaining


class RemainingTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'coordinates': 612, 'modes': [
            {'id': mode, 'lengths': list(lengths)} for mode, lengths in EXPECTED.items()]}
        self.suite = {'suite_id': 'qwen38-context-12x7-v1', 'cases': [
            {'case_id': f'case-{n}-t{target}', 'target_tokens': target,
             'request_sha256': hashlib.sha256(f'{n}:{target}'.encode()).hexdigest()}
            for target in LENGTHS for n in range(12)]}
        identity = {'id': 'a'*64, 'image': 'sha256:'+'b'*64, 'model': '/verified/model', 'running': True}
        self.window = {'outcome': 'failed_restored', 'production_prestate': identity,
                       'restore': {'admission_public': True, 'qwen_ready': True,
                                   'gateway_completions': 2, 'container': dict(identity)},
                       'restore_error_code': None, 'error_code': 'window_restore_reserve_reached',
                       'modes': {mode: {'status': 'BLOCKED'} for mode in EXPECTED}}
        self.mode = next(iter(EXPECTED))

    def raw(self, rows):
        return b''.join(json.dumps(r).encode() + b'\n' for r in rows)

    def row(self, index, kind='case', **extra):
        case = self.suite['cases'][index]
        return dict(kind=kind, mode_id=self.mode, case_id=case['case_id'],
                    request_sha256=case['request_sha256'], **extra)

    def test_empty_captures_select_exact_612_in_original_order(self):
        result = select_remaining(self.settings, self.suite, self.window, {})
        self.assertEqual((result['captured'], result['untouched'], result['unresolved_dispatches']), (0, 612, 0))
        self.assertEqual(list(result['modes']), list(EXPECTED))

    def test_never_replays_fail_pending_error_or_capacity_blocked(self):
        rows = []
        for i, outcome in enumerate(('pass', 'fail', 'pending_sandbox', 'error')):
            rows.extend((self.row(i, 'case_start'), self.row(i, outcome=outcome)))
        rows.append(self.row(4, outcome='blocked', error='capacity_bound_exceeded'))
        result = select_remaining(self.settings, self.suite, self.window, {self.mode: self.raw(rows)})
        self.assertEqual((result['captured'], result['untouched']), (5, 607))
        self.assertFalse(set(result['modes'][self.mode]['captured_case_ids']) &
                         set(result['modes'][self.mode]['untouched_case_ids']))

    def test_interrupted_request_is_held_not_automatically_retried(self):
        raw = self.raw([self.row(0, 'case_start')]) + b'{"kind":"case"'
        result = select_remaining(self.settings, self.suite, self.window, {self.mode: raw})
        self.assertEqual((result['captured'], result['untouched'], result['unresolved_dispatches']), (0, 611, 1))
        self.assertEqual(result['modes'][self.mode]['unresolved_dispatch_case_ids'], ['case-0-t8192'])
        self.assertTrue(result['modes'][self.mode]['partial_final_envelope'])

    def test_valid_final_answer_without_lf_is_captured_not_replayed(self):
        raw = self.raw([self.row(0, 'case_start'), self.row(0, outcome='pass')]).removesuffix(b'\n')
        result = select_remaining(self.settings, self.suite, self.window, {self.mode: raw})
        self.assertEqual((result['captured'], result['untouched'], result['unresolved_dispatches']), (1, 611, 0))
        self.assertFalse(result['modes'][self.mode]['partial_final_envelope'])

    def test_restore_requires_readiness_completions_and_exact_identity(self):
        for key, value in (('qwen_ready', False), ('gateway_completions', 1),
                           ('container', {**self.window['restore']['container'], 'image': 'different'})):
            window = copy.deepcopy(self.window)
            window['restore'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_remaining(self.settings, self.suite, window, {})

    def test_source_safety_or_missing_restoration_fences_selection(self):
        for change in ({'restore_error_code': 'restore_unknown'}, {'outcome': 'running'},
                       {'error_code': 'gpu_temperature_stop'}, {'restore': {'admission_public': False}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_remaining(self.settings, self.suite, {**self.window, **change}, {})

    def test_duplicate_changed_identity_and_interior_corruption_fence_selection(self):
        start, answer = self.row(0, 'case_start'), self.row(0, outcome='pass')
        variants = [self.raw([start, answer, answer]),
                    self.raw([start, {**answer, 'request_sha256': 'changed'}]),
                    self.raw([{**start, 'mode_id': 'other'}]), b'{broken}\n']
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises((ValueError, json.JSONDecodeError)):
                select_remaining(self.settings, self.suite, self.window, {self.mode: raw})

    def test_captured_profile_requires_complete_set_and_exact_sha(self):
        rows = []
        for index in range(36):
            rows.extend((self.row(index, 'case_start'), self.row(index, outcome='pass')))
        raw = self.raw(rows)
        window = copy.deepcopy(self.window)
        window['modes'][self.mode] = {'status': 'CAPTURED', 'client': {'sha256': hashlib.sha256(raw).hexdigest()}}
        result = select_remaining(self.settings, self.suite, window, {self.mode: raw})
        self.assertEqual((result['captured'], result['untouched']), (36, 576))
        window['modes'][self.mode]['client']['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'sha_mismatch'):
            select_remaining(self.settings, self.suite, window, {self.mode: raw})

    def test_changed_scope_cannot_silently_shrink_matrix(self):
        settings = copy.deepcopy(self.settings)
        settings['modes'][-1]['lengths'] = list(LENGTHS)
        with self.assertRaisesRegex(ValueError, 'mode_lengths_changed'):
            select_remaining(settings, self.suite, self.window, {})


if __name__ == '__main__':
    unittest.main()
