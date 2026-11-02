import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import quality_context_continue_client as continuation


class ContinueClientTests(unittest.TestCase):
    def setUp(self):
        self.mode = 'autoround-mtp-fast'
        self.cases = [dict(case_id=f'case-{i}', family_id=f'family-{i}', target_tokens=8192,
                           frozen_input_tokens=8192, request_bytes=30000,
                           request_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                           request={'generation': {'max_tokens': 256}}) for i in range(3)]
        self.raw = json.dumps({'cases': self.cases}).encode()
        self.sha = hashlib.sha256(self.raw).hexdigest()
        self.selection = dict(mode_id=self.mode, max_len=57344, backend='vllm',
                              fixture_sha256=self.sha, case_ids=['case-1', 'case-2'],
                              skip_recorded_warmup=True)

    def capture(self, selection=None, *, response=None, count=8192, max_len=57344):
        fake = mock.Mock()
        fake.get.return_value = (200, json.dumps({'data': [
            {'id': 'qwen3.8-27b', 'max_model_len': max_len}]}).encode(), {})
        def default_response(client, model, mode, case, **kwargs):
            if kwargs.get('kind') == 'warmup':
                return {'kind': 'warmup', 'outcome': 'pass'}, False
            return dict(kind='case', case_id=case['case_id'], outcome='pass',
                        request_sha256=case['request_sha256'], usage={'prompt_tokens': count}), False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            # Test fixture creation only; no models, runtime or generation.
            path.write_bytes(self.raw)
            with mock.patch.object(continuation, 'QualityClient', return_value=fake), \
                 mock.patch.object(continuation, 'tokenize', return_value=count) as tokenizer, \
                 mock.patch.object(continuation, 'run_case', side_effect=response or default_response) as run, \
                 mock.patch.object(continuation, '_emit') as emit:
                code = continuation.capture_selected(path, self.sha, self.mode, max_len, 'vllm',
                                                      selection or self.selection)
        return code, run, tokenizer, emit

    def test_only_selected_cases_are_tokenized_and_dispatched_without_second_warmup(self):
        code, run, tokenizer, emit = self.capture()
        self.assertEqual(code, 0)
        self.assertEqual([c.args[3]['case_id'] for c in run.call_args_list], ['case-1', 'case-2'])
        self.assertEqual([c.args[1]['case_id'] for c in tokenizer.call_args_list], ['case-1', 'case-2'])
        self.assertEqual([c.args[0]['kind'] for c in emit.call_args_list],
                         ['case_start', 'case', 'case_start', 'case'])

    def test_unstarted_profile_receives_one_warmup(self):
        code, run, _, _ = self.capture({**self.selection, 'skip_recorded_warmup': False})
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args_list[0].kwargs, {'kind': 'warmup'})

    def test_bad_identity_duplicate_reordered_and_unknown_cases_rejected_before_dispatch(self):
        for change in ({'fixture_sha256': 'changed'}, {'case_ids': ['case-1', 'case-1']},
                       {'case_ids': ['case-2', 'case-1']}, {'case_ids': ['not-in-suite']},
                       {'backend': 'llama'}, {'case_ids': []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.capture({**self.selection, **change})

    def test_capacity_guard_kept_without_dispatch_or_started_envelopes(self):
        selection = {**self.selection, 'max_len': 8447}
        code, run, _, emit = self.capture(selection, max_len=8447)
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertEqual([c.args[0]['error'] for c in emit.call_args_list],
                         ['capacity_bound_exceeded', 'capacity_bound_exceeded'])

    def test_tokenizer_drift_prevents_any_inference(self):
        with self.assertRaisesRegex(ValueError, 'runtime_tokenization_differs'):
            self.capture(count=8193)

    def test_completed_empty_answer_is_fail_but_does_not_skip_remaining_case(self):
        def empty_response(client, model, mode, case, **kwargs):
            return dict(kind='case', case_id=case['case_id'], outcome='error',
                        error='empty_output', finish_reason='stop',
                        request_sha256=case['request_sha256'], usage={'prompt_tokens': 8192}), True
        code, run, _, emit = self.capture(response=empty_response)
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        answers = [c.args[0] for c in emit.call_args_list if c.args[0]['kind'] == 'case']
        self.assertTrue(all(r['outcome'] == 'fail' and r['error'] == 'empty_answer' for r in answers))


if __name__ == '__main__':
    unittest.main()
