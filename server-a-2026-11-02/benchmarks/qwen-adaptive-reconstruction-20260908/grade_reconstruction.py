"""Grade only the reconstructed candidate once, after proven restoration."""
import json
import os
from pathlib import Path
import sys
from run_reconstruction import OLD, OLD_MANIFEST, MODE, ROOT, TOTAL_BOUND, digest, verify_new

def main():
    verify_new(sys.argv[1])
    manifest = OLD / 'bundle-manifest-remaining.json'
    assert digest(manifest) == OLD_MANIFEST
    value = json.loads(manifest.read_bytes())
    for name, sha in {**{'sources/'+n:h for n,h in value['files'].items()},**value['other_files']}.items():
        assert digest(OLD/name) == sha
    sys.path.insert(0,str(OLD/'sources'))
    import quality_sandbox
    from quality_expanded_suite import load_suite
    from score_quality import score
    quality_sandbox.CONTAINER_PREFIX = 'qwen-adaptive-reconstruction-20260908-code'
    results = ROOT/'results'
    window = json.loads((results/'window-report.json').read_bytes())
    assert window.get('restore',{}).get('admission_public') is True and not window.get('restore_error_code')
    entry = window['modes'][MODE]
    assert entry['status'] == 'CAPTURED'
    source = results/(MODE+'.jsonl')
    assert digest(source) == entry['client']['sha256']
    result = score(source,MODE,results/('scores-'+MODE+'.jsonl'),window['storage_available_before_bytes'],
                   suite=load_suite(),max_growth_bytes=TOTAL_BOUND)
    with (results/'grading-summary.json').open('x') as f:
        json.dump({MODE:result},f,sort_keys=True); f.flush(); os.fsync(f.fileno())
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
