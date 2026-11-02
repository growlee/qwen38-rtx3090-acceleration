"""Read-only unit, final report and public service evidence; no model calls."""
import json
import subprocess
import sys
from remote_window import ROOT,OLD,check_manifest,load

manifest=check_manifest(sys.argv[1])
process=subprocess.run(['systemctl','show','qwen-multiturn-context-20260910.service',
    '--property=ActiveState','--property=SubState','--property=Result'],capture_output=True,timeout=20,check=True)
unit=dict(line.split('=',1) for line in process.stdout.decode().splitlines() if '=' in line)
result=dict(unit=unit,report=None,public=None)
path=ROOT/'results/window-report.json'
if path.exists():
    assert path.stat().st_size<=2*1024**2
    result['report']=json.loads(path.read_bytes())
try:
    w=load('qmt_status_window',OLD/'run_quality_window_remaining.py')
    w.verify_bundle('0a6ba209c1a9320743c9e455af3934afe47f9d856e7c01e35ff6665cb2ab35e3')
    common,syv,helper=w.load_runtime_modules()
    phase=helper.operator().get('phase')
    helper.empty(phase)
    current=w.production_identity(common)
    state=common.qwen_state(common.Operator().qwen_status())
    result['public']=dict(phase=phase,empty=True,qwen_state=state,production=current,
        identity_matches=all(current[k]==manifest['production'][k] for k in ('id','image','model')))
except Exception as exc:result['public_read_error']=str(exc)[:300]
print(json.dumps(result,separators=(',',':')))
