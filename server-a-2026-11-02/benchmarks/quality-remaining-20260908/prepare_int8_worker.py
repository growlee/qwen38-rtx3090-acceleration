"""Run two exact upstream CPU scripts once, with bounded disk and no network."""
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys

MODEL=Path('/model')
EXPECTED={
    'quant_mtp.py':'01f4c298c1917f9a26dd36b4d29c98d35eae6a3a3958ab6b3f6def25b6332bd0',
    'build_draft_vocab.py':'bb99b0f4c91370346b3d826313c1a144794ecf248a64cd3ae9be7a4763545c6f',
    'draft_vocab_ids.json':'b64b6dfcf5441eb995ddf77d3d37b018e91b88c56ad1b4c5774ad8fbfac1c388',
}
for name,digest in EXPECTED.items():
    if hashlib.sha256((Path('/app/prepare')/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('pinned_upstream_script_mismatch')

def available():
    v=os.statvfs(MODEL);return v.f_bavail*v.f_frsize

before=available()
for name,args in [('quant_mtp.py',['/model','--bits','8']),
                  ('build_draft_vocab.py',['/model','--ids','/app/prepare/draft_vocab_ids.json'])]:
    if available()<150*1024**3+2*1024**3:
        raise RuntimeError('worker_preparation_reserve_insufficient')
    script='/app/prepare/'+name
    subprocess.run([sys.executable,'-B',script,*args],check=True,timeout=250)
    if available()<150*1024**3 or before-available()>3*1024**3:
        raise RuntimeError('worker_phase_storage_exceeded')
with Path('/app/prepare/draft_vocab_ids.json').open('rb') as src, (MODEL/'draft_vocab_ids.json').open('xb') as dst:
    shutil.copyfileobj(src,dst)
