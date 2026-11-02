"""Grade exactly the retained context coordinates, once, after public restoration."""
import json
import os
import sys
from pathlib import Path
from run_context import ROOT, OLD, GROWTH, PREFIX, check_bundle, sha, save


def main():
    check_bundle(sys.argv[1])
    results = ROOT/'results'
    window = json.loads((results/'context-window-report.json').read_bytes())
    assert window.get('restore',{}).get('admission_public') and not window.get('restore_error_code')
    settings = json.loads((ROOT/'execution-settings.json').read_bytes())
    assert sha(ROOT/'frozen/frozen.json') == settings['fixture_sha256']
    suite = json.loads((ROOT/'frozen/frozen.json').read_bytes())
    sys.path.insert(0,str(OLD/'sources'))
    from quality_client import evaluate_noncode
    import quality_sandbox
    quality_sandbox.CONTAINER_PREFIX = PREFIX+'-code'
    summary = {}; fenced = False; total = 0
    for mode in settings['modes']:
        mode_id = mode['id']; source = results/(mode_id+'.jsonl')
        entry = window['modes'].get(mode_id,{})
        raw = source.read_bytes() if source.exists() else b''
        lines = raw.splitlines()
        # A stopped process may leave an incomplete last envelope. Keep every
        # complete prior answer; never infer completion or replay that request.
        partial_tail = bool(raw and not raw.endswith(b'\n'))
        if partial_tail: lines = lines[:-1]
        rows = [json.loads(line) for line in lines]
        actual = [r for r in rows if r.get('kind')=='case']
        by_id = {r['case_id']:r for r in actual}
        cases = [c for c in suite['cases'] if c['target_tokens'] in mode['lengths']]
        assert len(by_id)==len(actual) and set(by_id).issubset({c['case_id'] for c in cases})
        assert all(r['mode_id']==mode_id for r in actual)
        if entry.get('status')=='CAPTURED': assert sha(source)==entry['client']['sha256']
        resources = results/(mode_id+'-resources.json')
        samples = json.loads(resources.read_bytes()) if resources.exists() else []
        counts = {'PASS':0,'FAIL':0,'BLOCKED':0,'incomplete_final_envelope':partial_tail}
        with (results/('scores-'+mode_id+'.jsonl')).open('x',encoding='utf-8') as out:
            for case in cases:
                row = by_id.get(case['case_id'])
                item = dict(mode_id=mode_id,case_id=case['case_id'],family_id=case['family_id'],
                            target_tokens=case['target_tokens'], request_sha256=case['request_sha256'])
                if row is None:
                    verdict = dict(outcome='BLOCKED',error=entry.get('abort_code',entry.get('reason','not_dispatched')))
                elif row.get('outcome') in ('error','blocked'):
                    verdict = dict(outcome='BLOCKED',error=row.get('error'))
                elif row.get('error') in ('finish_reason_length','empty_answer'):
                    verdict = dict(outcome='FAIL',error=row['error'])
                elif case['check']['type']=='python_function_cases':
                    fs = os.statvfs(ROOT); free = fs.f_bavail*fs.f_frsize
                    if free < 150*1024**3 or window['storage_available_before_bytes']-free > GROWTH: fenced=True
                    if fenced: verdict = dict(outcome='BLOCKED',error='sandbox_safety_fence')
                    else:
                        verdict = quality_sandbox.evaluate_code(case,row.get('content',''),mode_id=mode_id)
                        if verdict['outcome']=='BLOCKED': fenced=True
                else:
                    passed = evaluate_noncode(case,row)
                    verdict = dict(outcome='PASS' if passed else 'FAIL',error=None if passed else 'quality_check_failed')
                item.update(verdict)
                if row:
                    assert row['request_sha256']==case['request_sha256']
                    for key in ('input_tokens','request_bytes','context_limit','usage','finish_reason',
                                'elapsed_seconds','ttft_first_generated_seconds','ttft_first_content_or_tool_seconds',
                                'decode_tokens_per_second','started_epoch','ended_epoch'):
                        item[key] = row.get(key)
                    taken = [s for s in samples if row.get('started_epoch',float('inf')) <= s[0] <= row.get('ended_epoch',0)]
                    item['resource_samples'] = len(taken)
                    item['max_vram_mib'] = max(s[1] for s in taken) if taken else None
                    item['min_host_available_bytes'] = min(s[7] for s in taken) if taken else None
                out.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n'); out.flush(); os.fsync(out.fileno())
                counts[item['outcome']] += 1; total += 1
        summary[mode_id] = counts
    assert total==612
    save(results/'grading-summary.json',{'coordinates':total,'modes':summary,'sandbox_fenced':fenced})
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
