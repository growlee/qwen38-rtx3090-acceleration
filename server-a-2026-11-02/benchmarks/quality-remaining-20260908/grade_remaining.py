"""One frozen CPU grading pass, including explicit blocked coordinates."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    assert len(sys.argv)==2
    raw=(ROOT/'bundle-manifest-remaining.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sys.argv[1]
    manifest=json.loads(raw)
    for n,h in {**{'sources/'+n:h for n,h in manifest['files'].items()},**manifest['other_files']}.items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
    sys.path.insert(0,str(ROOT/'sources'))
    import quality_sandbox
    from quality_expanded_suite import load_suite
    from score_quality import score
    quality_sandbox.CONTAINER_PREFIX='qwen-quality-remaining-20260908-code'
    suite=load_suite();results=ROOT/'results'
    window=json.loads((results/'window-report.json').read_bytes())
    assert window.get('restore',{}).get('admission_public') is True and not window.get('restore_error_code')
    summary={};sandbox_fenced=False
    for mode in ('fast-mtp-int8-vocab40960','dflash2-fast-n7','dflash2-huge-n7','historical-adaptive-k4'):
        entry=window['modes'][mode];source=results/(mode+'.jsonl');dest=results/('scores-'+mode+'.jsonl')
        if entry['status']=='CAPTURED' and not sandbox_fenced:
            assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['client']['sha256']
            summary[mode]=score(source,mode,dest,window['storage_available_before_bytes'],suite=suite,max_growth_bytes=7*1024**3)
            sandbox_fenced=summary[mode]['BLOCKED']>0
        else:
            # Preserve any partial capture ungraded on an invalid window; no replay.
            reason='sandbox_fenced_after_prior_blocked' if sandbox_fenced else entry.get('abort_code',entry.get('reason','candidate_not_captured'))
            with dest.open('x',encoding='utf-8') as f:
                for case in suite['cases']:
                    f.write(json.dumps({'mode_id':mode,'case_id':case['case_id'],'category':case['category'],
                                        'critical':case['critical'],'outcome':'BLOCKED','error':reason})+'\n')
                f.flush();os.fsync(f.fileno())
            summary[mode]={'PASS':0,'FAIL':0,'BLOCKED':200,'cases':200,'reason':reason}
    with (results/'grading-summary.json').open('x') as f:
        json.dump(summary,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(summary,sort_keys=True),flush=True)
if __name__=='__main__':main()
