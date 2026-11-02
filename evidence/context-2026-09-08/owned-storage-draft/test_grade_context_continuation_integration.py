"""Offline file-level check; sandbox and server evidence are explicit doubles."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'qwen38-rtx3090-acceleration' / 'runners'))
import test_continue_context as fixtures
import grade_context_continuation as grader


class GraderIntegrationTests(unittest.TestCase):
    def test_one_new_sandbox_call_exact_612_merge_and_repeat_is_fenced(self):
        data = fixtures.ContinueControllerTests()
        data.setUp()
        mode_id, case_id = data.mode, data.case
        for case in data.suite['cases']:
            case.update(family_id=case['case_id'].split('-t')[0], check={'type': 'python_function_cases'})
        for part in data.partition['modes'].values():
            part['captured_case_ids'] = part['untouched_case_ids'][:]
            part['untouched_case_ids'] = []
        part = data.partition['modes'][mode_id]
        part['captured_case_ids'].remove(case_id)
        part['untouched_case_ids'] = [case_id]
        part['warmup_recorded'] = True
        data.partition.update(captured=611, untouched=1)
        data.selection['modes'] = {mode_id: {**data.selection['modes'][mode_id],
                                           'case_ids': [case_id], 'skip_recorded_warmup': True}}
        case = data.suite['cases'][0]
        row = dict(kind='case', mode_id=mode_id, case_id=case_id,
                   request_sha256=case['request_sha256'], content='def example(): return 1',
                   outcome='pending_sandbox', error=None, finish_reason='stop',
                   started_epoch=100, ended_epoch=101, input_tokens=8192, request_bytes=30000,
                   context_limit=57344, elapsed_seconds=1, ttft_first_generated_seconds=0.8,
                   usage={'prompt_tokens': 8192, 'completion_tokens': 10}, decode_tokens_per_second=50)
        raw = json.dumps(row).encode()  # Proven complete JSON, deliberately no LF.
        manifest_sha, selection_sha = 'a'*64, 'b'*64
        report = dict(outcome='captured_restored', restore=data.window['restore'],
                      production_prestate=data.window['production_prestate'], restore_error_code=None,
                      source_manifest_sha256=manifest_sha, selection_sha256=selection_sha,
                      fixture_sha256='f'*64, storage_available_before_bytes=200*1024**3,
                      modes={mode_id: {'status': 'CAPTURED', 'client': {'sha256': hashlib.sha256(raw).hexdigest()}}})
        sandbox = SimpleNamespace(evaluate_code=mock.Mock(return_value={'outcome': 'PASS', 'error': None}),
                                  _short_command=mock.Mock())
        storage = mock.Mock(containers={}, last={})
        noncode = SimpleNamespace(evaluate_noncode=mock.Mock(side_effect=AssertionError('no new noncode case')))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, current = root/'r1', root/'r2'
            (source/'results').mkdir(parents=True)
            (current/'results').mkdir(parents=True)
            (current/'selection.json').write_text(json.dumps(data.selection), encoding='utf-8')
            (current/'results'/'context-window-report.json').write_text(json.dumps(report), encoding='utf-8')
            (current/'results'/(mode_id+'.jsonl')).write_bytes(raw)
            source_hashes = {}
            for mode in data.settings['modes']:
                rows = [dict(mode_id=mode['id'], case_id=c['case_id'], request_sha256=c['request_sha256'],
                             outcome='BLOCKED' if (mode['id'], c['case_id']) == (mode_id, case_id) else 'PASS')
                        for c in data.suite['cases'] if c['target_tokens'] in mode['lengths']]
                path = source/'results'/('scores-'+mode['id']+'.jsonl')
                encoded = ''.join(json.dumps(r)+'\n' for r in rows).encode()
                path.write_bytes(encoded)
                source_hashes[path] = hashlib.sha256(encoded).hexdigest()
            r1_double = SimpleNamespace(OLD=root/'unused', GROWTH=4*1024**3)
            manifest = {'files': {'selection.json': selection_sha}}
            with mock.patch.object(grader, 'ROOT', current), mock.patch.object(grader, 'R1', source), \
                 mock.patch.object(grader, 'check_bundle', return_value=(manifest, r1_double)), \
                 mock.patch.object(grader, 'check_source', return_value=(data.settings, data.suite, data.window, data.partition)), \
                 mock.patch.object(grader.os, 'statvfs', create=True, return_value=SimpleNamespace(f_bavail=200*1024**3, f_frsize=1)), \
                 mock.patch.object(sys, 'argv', ['grade', manifest_sha]), \
                 mock.patch.dict(sys.modules, quality_sandbox=sandbox, quality_client=noncode), \
                 contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(storage, 'check', side_effect=grader.StorageError('root_storage_reserve_exhausted')):
                    with self.assertRaisesRegex(grader.StorageError, 'root_storage_reserve_exhausted'):
                        grader.main(storage=storage)
                sandbox.evaluate_code.assert_not_called()
                self.assertFalse(any((current/'results').glob('scores-*')))
                grader.main(storage=storage)
                sandbox.evaluate_code.assert_called_once_with(case, row['content'], mode_id=mode_id)
                noncode.evaluate_noncode.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'grading_already_attempted'):
                    grader.main(storage=storage)
                self.assertEqual(sandbox.evaluate_code.call_count, 1)
            merged = [json.loads(line) for line in (current/'results'/'combined-scores.jsonl').read_bytes().splitlines()]
            self.assertEqual(len(merged), 612)
            self.assertEqual(sum(r['result_window'] == 'r2' for r in merged), 1)
            self.assertTrue(all(r['outcome'] == 'PASS' for r in merged))
            summary = json.loads((current/'results'/'grading-summary.json').read_bytes())
            self.assertEqual(summary['newly_graded_coordinates'], 1)
            self.assertFalse(summary['sandbox_fenced'])
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == digest
                                for p, digest in source_hashes.items()))


if __name__ == '__main__':
    unittest.main()
