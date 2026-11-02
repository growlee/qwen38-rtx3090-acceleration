"""Once-only launch of the exact reviewed transient unit. No restart or retry."""
import json
import os
import subprocess
import sys
from remote_window import ROOT,check_manifest

expected=sys.argv[1];manifest=check_manifest(expected)
assert os.geteuid()==0 and sys.argv[2]=='launch-once'
assert not (ROOT/'results').exists()
assert not any((ROOT/'responses').iterdir()) and not any((ROOT/'grades').iterdir())
first=manifest['coordinates'][0]['case_id']
assert sorted(p.name for p in (ROOT/'requests').iterdir())==[first+'.json']
for name in [*manifest['remote_files'],'manifest.json']:
    path=ROOT/name
    assert not path.is_symlink() and path.stat().st_uid==0 and path.stat().st_mode&0o777==0o444
unit='qwen-multiturn-context-20260910.service'
before=subprocess.run(['systemctl','show',unit,'--property=LoadState','--value'],capture_output=True,timeout=20)
assert before.stdout.strip()==b'not-found','owned_unit_already_exists'
assert not (ROOT/'launch-receipt.json').exists()
command=['systemd-run','--unit='+unit,'--property=Restart=no','--property=RuntimeMaxSec=737699s',
    '--property=TimeoutStopSec=1s','--property=KillMode=mixed',
    '--property=StandardOutput=append:'+str(ROOT/'controller.stdout'),
    '--property=StandardError=append:'+str(ROOT/'controller.stderr'),
    '--','/usr/bin/python3','-B',str(ROOT/'remote_window.py'),expected]
with (ROOT/'launch-receipt.json').open('x') as out:
    json.dump(dict(command=command,manifest_sha256=expected,attempted=True),out);out.flush();os.fsync(out.fileno())
result=subprocess.run(command,capture_output=True,timeout=30)
print(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())))
raise SystemExit(result.returncode)
