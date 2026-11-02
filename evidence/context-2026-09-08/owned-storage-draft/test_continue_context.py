import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'qwen38-rtx3090-acceleration' / 'runners'))
import quality_context_remaining as remaining
import continue_context as controller


class ContinueControllerTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'coordinates': 612, 'fixture_sha256': 'f'*64, 'modes': [
            {'id': mode, 'lengths': list(lengths), 'kind': 'gguf' if mode == 'original-gguf' else 'syv',
             'max_len': 16384 if mode == 'original-gguf' else 204800 if 'huge' in mode else 57344}
            for mode, lengths in remaining.EXPECTED.items()]}
        self.suite = {'suite_id': 'qwen38-context-12x7-v1', 'cases': [
            {'case_id': f'case-{n}-t{target}', 'target_tokens': target,
             'request_sha256': hashlib.sha256(f'{n}:{target}'.encode()).hexdigest()}
            for target in remaining.LENGTHS for n in range(12)]}
        identity = {'id': 'a'*64, 'image': 'sha256:'+'b'*64, 'model': '/verified/model', 'running': True}
        self.window = {'outcome': 'failed_restored', 'production_prestate': identity,
                       'restore': {'admission_public': True, 'qwen_ready': True,
                                   'gateway_completions': 2, 'container': dict(identity)},
                       'restore_error_code': None, 'error_code': 'window_restore_reserve_reached',
                       'modes': {mode: {'status': 'BLOCKED'} for mode in remaining.EXPECTED}}
        self.partition = remaining.select_remaining(self.settings, self.suite, self.window, {})
        self.selection = {'source_manifest_sha256': controller.R1_SHA, 'modes': {
            mode['id']: {'mode_id': mode['id'], 'case_ids': self.partition['modes'][mode['id']]['untouched_case_ids'][:],
                         'max_len': mode['max_len'], 'backend': 'llama' if mode['kind'] == 'gguf' else 'vllm',
                         'fixture_sha256': 'f'*64, 'skip_recorded_warmup': False}
            for mode in self.settings['modes']}}
        self.mode = next(iter(remaining.EXPECTED))
        self.case = self.selection['modes'][self.mode]['case_ids'][0]

    def validate(self):
        return controller.validate_selection(self.selection, self.settings, self.suite, self.window, self.partition)

    def test_selection_keeps_original_mode_order_even_if_json_keys_sorted(self):
        self.selection['modes'] = dict(sorted(self.selection['modes'].items()))
        self.assertEqual(self.validate(), list(remaining.EXPECTED))

    def test_completed_answer_cannot_be_added_to_continuation(self):
        part = self.partition['modes'][self.mode]
        part['untouched_case_ids'].remove(self.case)
        part['captured_case_ids'].append(self.case)
        with self.assertRaisesRegex(ValueError, 'selection_not_exact_remaining_case_set'):
            self.validate()

    def test_missing_and_reordered_case_selections_are_rejected(self):
        original = self.selection['modes'][self.mode]['case_ids'][:]
        for ids in (original[1:], original[::-1]):
            self.selection['modes'][self.mode]['case_ids'] = ids
            with self.assertRaisesRegex(ValueError, 'selection_not_exact_remaining_case_set'):
                self.validate()

    def test_runtime_and_warmup_contract_cannot_change(self):
        for field, value in (('max_len', 200000), ('backend', 'llama'),
                             ('fixture_sha256', 'changed'), ('skip_recorded_warmup', True)):
            original = copy.deepcopy(self.selection)
            self.selection['modes'][self.mode][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'profile_contract_changed'):
                self.validate()
            self.selection = original

    def held(self):
        part = self.partition['modes'][self.mode]
        part['untouched_case_ids'].remove(self.case)
        part['unresolved_dispatch_case_ids'].append(self.case)
        self.partition['untouched'] -= 1
        self.partition['unresolved_dispatches'] += 1

    def test_unresolved_dispatch_needs_exact_invalidation_and_cleanup_evidence(self):
        self.held()
        with self.assertRaisesRegex(ValueError, 'requires_classification'):
            self.validate()
        self.selection['invalidated_dispatches'] = [dict(mode_id=self.mode, case_id=self.case,
            reason='deadline_interruption_no_complete_capture', evidence_note='fixture: timeout interrupted capture')]
        entry = self.window['modes'][self.mode]
        entry['reason'] = 'window_restore_reserve_reached'
        with self.assertRaisesRegex(ValueError, 'cleanup_unproven'):
            self.validate()
        entry['cleanup'] = {role: {'retained': True, 'id': digit*64}
                            for role, digit in (('client', 'a'), ('candidate', 'b'))}
        self.assertEqual(self.validate(), list(remaining.EXPECTED))
        entry['reason'] = 'context_client_failed'
        with self.assertRaisesRegex(ValueError, 'not_proven_deadline_interruption'):
            self.validate()

    def test_invalidation_cannot_replay_a_completed_or_untouched_case(self):
        self.selection['invalidated_dispatches'] = [dict(mode_id=self.mode, case_id=self.case,
            reason='deadline_interruption_no_complete_capture', evidence_note='not evidence')]
        with self.assertRaisesRegex(ValueError, 'not_an_unresolved_dispatch'):
            self.validate()


class ContinuationSafetyTests(unittest.TestCase):
    def test_disk_accounting_is_unconditional_with_write_reserve(self):
        storage = mock.Mock()
        controller.storage_check(storage)
        storage.check.assert_called_once_with(512*1024**2)
        storage.check.side_effect = controller.StorageError('test_storage_budget_exhausted')
        with self.assertRaisesRegex(controller.StorageError, 'test_storage_budget_exhausted'):
            controller.storage_check(storage)

    def test_window_accounts_for_all_remaining_cases_not_another_12_hours(self):
        data = ContinueControllerTests()
        data.setUp()
        budget = controller.time_budget(data.selection)
        self.assertEqual(budget['capture_seconds'], 612*5520+11*4500+2700)
        self.assertEqual(budget['window_seconds']-budget['capture_seconds'], 2700)
        self.assertGreater(budget['window_seconds'], 12*3600)
        data.selection['modes'][data.mode]['case_ids'].pop()
        data.selection['modes'][data.mode]['skip_recorded_warmup'] = True
        self.assertEqual(budget['capture_seconds']-controller.time_budget(data.selection)['capture_seconds'], 7320)

    def test_all_selected_payloads_drafter_and_images_are_verified(self):
        gguf = mock.Mock()
        gguf.stat.return_value.st_size = 123
        gguf.parent.name = 'd'*64
        model = Path('/retained/model')
        w = SimpleNamespace(SYV_IMAGE='syv', SYV_IMAGE_ID='syv-id', GGUF=gguf, GGUF_SIZE=123,
                            inspect_image=mock.Mock(), load_draft=mock.Mock())
        r1 = SimpleNamespace(verify_payload=mock.Mock(return_value={'verified': True}))
        selected = [dict(kind='syv', model=model, image='syv', image_id='syv-id'),
                    dict(kind='dflash2', model=model, image='syv', image_id='syv-id'),
                    dict(kind='gguf', model=gguf, image='llama', image_id='llama-id')]
        with mock.patch.object(controller, 'sha', return_value='d'*64):
            controller.verify_selected_payloads(w, r1, 'common', selected)
        r1.verify_payload.assert_called_once_with(w, selected[0])
        w.load_draft.return_value.verify_draft.assert_called_once_with()
        self.assertEqual(w.inspect_image.call_count, 2)
        with mock.patch.object(controller, 'sha', return_value='wrong'), self.assertRaisesRegex(ValueError, 'gguf_payload_changed'):
            controller.verify_selected_payloads(w, r1, 'common', selected)

    def handoff(self, initial='ready', marker_fails=False):
        events, saved = [], []
        phase, state = ['public'], [initial]
        identity = dict(id='a'*64, image='b'*64, model='/verified/model', running=True)
        def qwen(verb):
            events.append(verb)
            if verb == 'stop':
                self.assertEqual(phase[0], 'closed')
            state[0] = 'ready' if verb == 'start' else 'stopped'
        op = SimpleNamespace(qwen_status=lambda: state[0], qwen=qwen)
        def empty(expected):
            self.assertEqual(phase[0], expected)
        def advance(verb='status', **kwargs):
            if verb == 'advance-phase':
                phase[0] = kwargs['phase']
                events.append(phase[0])
            return {'phase': phase[0]}
        def snapshot(_):
            return dict(capacity_mode='serialized-99', global_breaker_open=False,
                        active_profile='qwen' if state[0] == 'ready' else None,
                        active_profiles=['qwen'] if state[0] == 'ready' else [],
                        profiles={'qwen': {'state': state[0], 'breaker_open': False}})
        helper = SimpleNamespace(operator=advance, empty=empty, exchange=snapshot,
                                 runtime=mock.Mock(return_value=({'profiles': {'qwen': 'stopped'}}, 'digest')))
        def marker(_):
            events.append('public-marker')
            self.assertEqual(phase[0], 'public')
            self.assertIn(b'chat_template_kwargs', common.RESTORE_SCRIPT)
            if marker_fails:
                raise ValueError('marker_failed')
        script = b"{'temperature':0,'max_tokens':64,'stream':False}"
        common = SimpleNamespace(Operator=lambda: op, qwen_state=lambda x: x,
            wait_gpu_idle=lambda _: events.append('idle'), RESTORE_SCRIPT=script, restore_completion=marker)
        w = SimpleNamespace(PROFILES={'qwen'}, restore=mock.Mock(),
                            production_identity=lambda _: {**identity, 'running': state[0] == 'ready'})
        controller.configure_public_handoff(w, common, helper, {'production_prestate': identity})
        with mock.patch.object(controller, 'save', side_effect=lambda p, v: saved.append(p.name)):
            if marker_fails:
                with self.assertRaisesRegex(ValueError, 'marker_failed'):
                    w.verify_preflight(common, helper)
            else:
                proof, digest, current = w.verify_preflight(common, helper)
                self.assertTrue(proof['qwen_ready'])
                self.assertEqual(len(digest), 64)
                self.assertEqual(current, identity)
                helper.operator('advance-phase', phase='closed')
                helper.operator('advance-phase', phase='smoke')
        self.assertEqual(common.RESTORE_SCRIPT, script)
        return events, saved, helper

    def test_public_proof_then_closed_stop_then_fresh_idle(self):
        events, saved, helper = self.handoff()
        self.assertEqual(events, ['public-marker', 'closed', 'stop', 'idle', 'smoke'])
        self.assertEqual(saved, ['public-preflight.json', 'closed-idle-preflight.json'])
        helper.runtime.assert_called_once_with({'qwen'})

    def test_idle_public_qwen_is_started_before_marker_without_stopping_other_profiles(self):
        events, _, _ = self.handoff(initial='stopped')
        self.assertEqual(events, ['idle', 'start', 'public-marker', 'closed', 'stop', 'idle', 'smoke'])

    def test_failed_public_marker_does_not_close_or_stop_and_restores_script(self):
        events, saved, helper = self.handoff(marker_fails=True)
        self.assertEqual(events, ['public-marker'])
        self.assertEqual(saved, ['public-preflight-failure.json'])
        helper.runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
