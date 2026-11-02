"""One CPU grading pass; no generation, downloads, deletion or profile changes."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def main():
    if len(sys.argv)!=3: raise SystemExit('usage: grade_expanded.py MANIFEST_SHA256 RESULTS')
    raw=(ROOT/'bundle-manifest-expanded.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sys.argv[1]
    manifest=json.loads(raw)
    for name,digest in manifest['other_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((ROOT/'sources'/name).read_bytes()).hexdigest()==digest
    sys.path.insert(0,str(ROOT/'sources'))
    import quality_sandbox
    from quality_expanded_suite import load_suite
    from score_quality import score
    quality_sandbox.CONTAINER_PREFIX='qwen-quality-expanded-20260908-code'
    suite=load_suite()
    results=Path(sys.argv[2])
    assert results==ROOT/'results'
    window=json.loads((results/'window-report.json').read_bytes())
    assert window['outcome']=='captured_restored'
    baseline=window['storage_available_before_bytes']
    summary={}
    for mode in ('original-gguf','autoround-bf16-mtp','current-adaptive-k4'):
        source=results/(mode+'.jsonl')
        expected=window['modes'][mode]['client']['sha256']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        summary[mode]=score(source,mode,results/('scores-'+mode+'.jsonl'),baseline,suite=suite,max_growth_bytes=1024**3)
        if summary[mode]['BLOCKED']:
            raise SystemExit('grading_infrastructure_blocked_no_retry')
    with (results/'grading-summary.json').open('x') as out:
        json.dump(summary,out,sort_keys=True,indent=2);out.flush();os.fsync(out.fileno())
    print(json.dumps(summary,sort_keys=True),flush=True)

if __name__=='__main__': main()
