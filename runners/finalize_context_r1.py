"""Finish retained r1 evidence after safety stop; never issue model requests.

Reuse 30 prior PC-A verdicts, score only the other 57 captured code answers in
the unchanged sandbox, then account for all 612 frozen coordinates locally.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import grade_autoround_pc_a as previous
import quality_context_suite
import quality_sandbox

ROOT, EVIDENCE = previous.ROOT, previous.EVIDENCE
RAW = {**previous.RAW,
    'int8-mtp-fast': '74f6c31955a1a91848071ec91c8a5e227e1faa5a15016a8f3a5f65b1317801c6',
    'int8-mtp-huge': '309333b0ecdef9e961dcff62bd915565a694ff6a84f7c02a2e9058f5955801ac',
    'adaptive-mtp-fast': '16697d22bc223faf1573265b4cf6ba8e0e7acaee3a3597c3566f12f156966cdf',
    'adaptive-mtp-huge': 'ffd16cf0cc75134fcd3b1349ca0c26e0b01efbf462dfd47d3a34657ffe53d700'}
PRIOR_SHA = '6d894c29f25f6fd6902e0261a1e5059f85e1edb6a55f0295f8a2eb3c2aad2583'
RESTORE_SHA = '6d5e4b444d69b8caf52dd598b06de59ebc2b20ab6e204ca80f9d142518b7b9c8'
OUTPUT = EVIDENCE / 'pc-a-remaining-code.jsonl'
PREFIX = 'qwen-context-pca-20260909-remaining'
GIB = 1024**3
sha, canonical, emit = previous.digest, previous.canonical, previous.emit


def checked(name, expected):
    raw = (EVIDENCE / name).read_bytes()
    assert sha(raw) == expected, name
    return raw


def lines(raw):
    assert raw.endswith(b'\n'), 'unexpected_partial_tail'
    return [json.loads(line) for line in raw.splitlines()]


def inputs():
    for name, expected in previous.SOURCE.items():
        assert sha(Path(__file__).with_name(name).read_bytes()) == expected
    plan = json.loads((ROOT / 'evidence/context-frozen-plan-2026-09-08.json').read_bytes())
    cases = [dict(quality_context_suite.build_case(i['family_id'], i['padding_count']), **i)
             for i in plan['cases']]
    frozen = canonical({**plan['suite_fields'], 'cases': cases})
    assert len(frozen) == 29781351 and sha(frozen) == previous.FROZEN
    settings = json.loads(checked('r1-execution-settings.json', '505ac09f22bd5d71ec04f80281b47f65a94b221da5a15b4fbe59fc89da057f20'))
    window = json.loads(checked('r1-context-window-report.json', 'd077d42f2077809667df9b320554171613bbbfb513596b018078ff679f9ec319'))
    restore = json.loads(checked('storage-repair-restoration.json', RESTORE_SHA))
    assert restore['public_restored'] and restore['qwen_ready'] and restore['gateway_completions'] == 2
    prior = lines(checked('pc-a-autoround-code.jsonl', PRIOR_SHA))
    assert prior[0]['frozen_sha256'] == previous.FROZEN
    assert prior[0]['sandbox_sha256'] == previous.SOURCE['quality_sandbox.py']
    assert prior[0]['image_id'] == previous.CONFIG_IMAGE
    assert sum(r['kind'] == 'reference' and r['outcome'] == 'PASS' for r in prior) == 3
    reused = {(r['mode_id'], r['case_id']): r for r in prior if r['kind'] == 'evaluation'}
    assert len(reused) == 30 and prior[-1]['evaluated'] == 30
    by_id = {c['case_id']: c for c in cases}
    raw_rows, pending = {}, []
    for mode, expected in RAW.items():
        rows = [r for r in lines(checked(mode + '.jsonl', expected)) if r.get('kind') == 'case']
        assert len(rows) == (79 if mode == 'adaptive-mtp-huge' else 36 if mode.endswith('fast') else 84)
        for row in rows:
            key = (mode, row['case_id'])
            assert key not in raw_rows and row['mode_id'] == mode
            case = by_id[row['case_id']]
            assert row['request_sha256'] == case['request_sha256']
            assert row['input_tokens'] == case['frozen_input_tokens']
            raw_rows[key] = row
            if row['outcome'] == 'pending_sandbox':
                assert case['check']['type'] == 'python_function_cases'
                if key in reused:
                    assert reused[key]['request_sha256'] == row['request_sha256']
                    assert reused[key]['content_sha256'] == sha(row['content'].encode())
                else:
                    pending.append((mode, case, row))
    expected_keys = {(m['id'], c['case_id']) for m in settings['modes'] for c in cases if c['target_tokens'] in m['lengths']}
    assert len(expected_keys) == 612 and set(raw_rows) <= expected_keys
    assert len(raw_rows) == 355 and len(pending) == 57 and set(reused) <= set(raw_rows)
    return cases, settings, window, raw_rows, reused, pending


def free(path):
    v = os.statvfs(path)
    return v.f_bavail * v.f_frsize


def execute(pending):
    assert platform.system() == 'Linux' and platform.node().lower() == 'growleepc'
    assert not OUTPUT.exists()
    assert not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=' + PREFIX], timeout=10).strip()
    receipt_path = ROOT.parent / 'home_lab-artifacts/ai-server-acceleration-20260830/syv-relay-v7/source-receipt.json'
    receipt = json.loads(receipt_path.read_bytes())
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', previous.CONFIG_IMAGE], timeout=10))[0]
    assert receipt['image_reference'] == previous.IMAGE and receipt['config_digest'] == image['Id'] == previous.CONFIG_IMAGE
    assert image['RootFS']['Layers'] == receipt['rootfs_diff_ids'] and image['Architecture'] == 'amd64' and image['Os'] == 'linux'
    baseline = {p: free(p) for p in ('/', '/mnt/e')}
    assert all(v >= 50 * GIB + GIB for v in baseline.values())
    started, counts = time.time(), collections.Counter()
    quality_sandbox.CONTAINER_PREFIX = PREFIX
    with OUTPUT.open('x', encoding='utf-8') as out:
        emit(out, dict(kind='execution', started_epoch=started, host=platform.node(),
                       image_id=image['Id'], image=previous.IMAGE, raw_sha256=RAW,
                       frozen_sha256=previous.FROZEN, prior_verdicts_sha256=PRIOR_SHA,
                       sandbox_sha256=previous.SOURCE['quality_sandbox.py'],
                       runner_sha256=sha(Path(__file__).read_bytes()), baseline_free_bytes=baseline,
                       archive_source_receipt_sha256=sha(receipt_path.read_bytes()),
                       restoration_sha256=RESTORE_SHA, references_reused=3, prior_answers_reused=30,
                       candidate_answers=57, model_requests=0, growth_bound_bytes=GIB,
                       disk_budget='57 retained metadata allowances x 8 MiB + 10 MiB evidence + 512 MiB VHD allocation margin = 978 MiB; existing image reused; no downloads, copies or writable layers; tmpfs sequential',
                       physical_floor_bytes=50*GIB, wall_seconds=1200))
        for mode, case, row in pending:
            assert time.time() - started < 1200
            assert all(free(p) >= 50*GIB and baseline[p] - free(p) < GIB for p in baseline)
            emit(out, dict(kind='evaluation_start', mode_id=mode, case_id=case['case_id']))
            verdict = quality_sandbox.evaluate_code(case, row['content'], previous.CONFIG_IMAGE, mode_id=mode)
            emit(out, dict(verdict, kind='evaluation', request_sha256=row['request_sha256'],
                           content_sha256=sha(row['content'].encode()), target_tokens=case['target_tokens']))
            counts[verdict['outcome']] += 1
            if verdict['outcome'] == 'BLOCKED':
                return 2  # Retain the exact partial receipt; no automatic replay.
        assert not subprocess.check_output(['docker', 'ps', '-q', '--filter', 'name=' + PREFIX], timeout=10).strip()
        final_free = {p: free(p) for p in baseline}
        assert all(final_free[p] >= 50*GIB and baseline[p] - final_free[p] < GIB for p in baseline)
        emit(out, dict(kind='summary', evaluated=sum(counts.values()), counts=counts,
                       ended_epoch=time.time(), free_bytes=final_free, owned_containers_running=False))
    print(json.dumps(dict(counts=counts, output=str(OUTPUT))))
    return 0


def report(data):
    cases, settings, window, raw_rows, verdicts, pending = data
    code_raw = OUTPUT.read_bytes()
    code = lines(code_raw)
    assert code[0]['raw_sha256'] == RAW and code[0]['prior_verdicts_sha256'] == PRIOR_SHA
    assert code[0]['sandbox_sha256'] == previous.SOURCE['quality_sandbox.py']
    assert code[0]['frozen_sha256'] == previous.FROZEN and code[0]['image_id'] == previous.CONFIG_IMAGE
    assert code[-1]['kind'] == 'summary' and code[-1]['evaluated'] == 57
    new = [r for r in code if r['kind'] == 'evaluation']
    assert len(new) == len({(r['mode_id'], r['case_id']) for r in new}) == 57
    assert {(r['mode_id'], r['case_id']) for r in new} == {(m,c['case_id']) for m,c,r in pending}
    for r in new:
        key = (r['mode_id'], r['case_id'])
        assert key not in verdicts
        assert r['content_sha256'] == sha(raw_rows[key]['content'].encode())
        assert r['request_sha256'] == raw_rows[key]['request_sha256']
        verdicts[key] = r
    scores, modes, resource_hashes = [], {}, {}
    for mode in settings['modes']:
        mid = mode['id']
        resource_path = EVIDENCE / (mid + '-resources.json')
        samples = json.loads(resource_path.read_bytes()) if resource_path.exists() else []
        if samples:
            resource_hashes[resource_path.name] = sha(resource_path.read_bytes())
        own = []
        for case in cases:
            if case['target_tokens'] not in mode['lengths']:
                continue
            key = (mid, case['case_id'])
            row = raw_rows.get(key)
            item = dict(mode_id=mid, case_id=case['case_id'], family_id=case['family_id'],
                        target_tokens=case['target_tokens'], request_sha256=case['request_sha256'],
                        expected_input_tokens=case['frozen_input_tokens'], dispatched=row is not None)
            if row is None:
                item.update(outcome='BLOCKED', error='storage_budget_exhausted', capture='not_dispatched')
            elif row['outcome'] in ('error', 'blocked'):
                item.update(outcome='BLOCKED', error=row['error'], capture='interrupted', safety_cause='storage_budget_exhausted')
            elif row['outcome'] == 'pending_sandbox':
                item.update(verdicts[key], verdict_source='pc-a-autoround-code.jsonl' if mid.startswith('autoround') else OUTPUT.name, capture='complete')
            else:
                assert row['outcome'] in ('pass', 'fail')
                item.update(outcome=row['outcome'].upper(), error=row.get('error'), capture='complete', verdict_source='frozen_client_raw')
            if row:
                for field in ('input_tokens','request_bytes','context_limit','usage','finish_reason','elapsed_seconds',
                              'ttft_first_generated_seconds','ttft_first_content_or_tool_seconds','decode_tokens_per_second','started_epoch','ended_epoch'):
                    item[field] = row.get(field)
                taken = [s for s in samples if row['started_epoch'] <= s[0] <= row['ended_epoch']]
                item.update(resource_samples=len(taken), max_vram_mib=max((s[1] for s in taken),default=None),
                            min_host_available_bytes=min((s[7] for s in taken),default=None))
            own.append(item)
        scores.extend(own)
        groups = {}
        for target in mode['lengths']:
            group = [r for r in own if r['target_tokens'] == target]
            complete = [r for r in group if r['capture'] == 'complete']
            metrics = {}
            for field in ('ttft_first_generated_seconds','decode_tokens_per_second','elapsed_seconds'):
                values = [r[field] for r in complete if isinstance(r.get(field),(float,int))]
                metrics[field] = dict(n=len(values),median=statistics.median(values) if values else None,max=max(values) if values else None)
            groups[target] = dict(counts=dict(collections.Counter(r['outcome'] for r in group)), completed=len(complete), metrics=metrics)
        modes[mid] = dict(counts=dict(collections.Counter(r['outcome'] for r in own)), lengths=groups,
                         max_completed_input_tokens=max((r['input_tokens'] for r in own if r['capture']=='complete'),default=None),
                         max_full_pass_target=max((t for t,g in groups.items() if g['counts']=={'PASS':12}),default=None),
                         first_quality_failure=next((r['case_id'] for r in own if r['outcome']=='FAIL'),None))
    assert len(scores) == len({(r['mode_id'],r['case_id']) for r in scores}) == 612
    summary = dict(terminal_outcome='SAFETY_STOP_RESTORED', coordinates=612,
                   counts=dict(collections.Counter(r['outcome'] for r in scores)), modes=modes,
                   raw_sha256=RAW, resource_sha256=resource_hashes, frozen_sha256=previous.FROZEN,
                   previous_code_sha256=PRIOR_SHA, remaining_code_sha256=sha(code_raw), restoration_sha256=RESTORE_SHA,
                   runner_sha256=sha(Path(__file__).read_bytes()), model_requests_added=0,
                   rubric='Original frozen client text verdicts reused; same CPU sandbox code checks; missing and interrupted cases BLOCKED, not quality FAIL')
    with (EVIDENCE/'r1-final-scores.jsonl').open('x',encoding='utf-8') as out:
        for row in scores:
            emit(out,row)
    with (EVIDENCE/'r1-final-summary.json').open('x',encoding='utf-8') as out:
        emit(out,summary)
    print(json.dumps(dict(coordinates=612, counts=summary['counts'], modes={m:v['counts'] for m,v in modes.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute-code',action='store_true')
    action.add_argument('--report',action='store_true')
    args = parser.parse_args()
    data = inputs()
    if args.execute_code:
        raise SystemExit(execute(data[-1]))
    elif args.report:
        report(data)
    else:
        print(json.dumps(dict(inputs_verified=True, coordinates=612, retained=355,
                              prior_answers_reused=30, code_to_evaluate=len(data[-1]), model_requests=0)))
