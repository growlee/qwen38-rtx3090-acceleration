"""Two finite preparation phases; shares existing layers and never mutates services."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path('/srv/ai-server/artifacts/experiments/freetoken013-accel-20260918')
ORIGINAL = Path('/srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918')
BASE = 'sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134'
BASE_TAG = 'local/ai-server-freetoken:0.1.3-cac247a-nvfp4-20260918'
TAG = 'local/ai-server-freetoken:0.1.3-cac247a-accel-20260918'
ORIGINAL_BUNDLE = 'b0bbaf4fddce284d2987b8dcabcc47d295c2f27686a7e4ff1ca45d246102dc95'
sys.path.insert(0, '/opt/homelab/ai-server')
from storage_guard.storage_guard import measure, admit, StoragePolicy


def checked(command, timeout=30):
    return subprocess.check_output(command, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['download','build'])
    ap.add_argument('--experiment-sha256',required=True)
    args = ap.parse_args()
    assert os.geteuid() == 0 and Path(__file__).resolve().parent == ROOT
    assert re.fullmatch('[a-f0-9]{64}',args.experiment_sha256)
    os.umask(0o022)
    manifest = ROOT/'experiment-inputs.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == args.experiment_sha256
    for directory in (ROOT,*ROOT.parents):
        st=directory.lstat()
        assert not directory.is_symlink() and st.st_uid==0 and not st.st_mode&0o022
    for name,digest in json.loads(manifest.read_text()).items():
        p=ROOT/name
        st=p.lstat()
        assert p.resolve().is_relative_to(ROOT) and not p.is_symlink() and st.st_uid==0 and not st.st_mode&0o022
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((ORIGINAL/'candidate-inputs.json').read_bytes()).hexdigest()==ORIGINAL_BUNDLE
    descriptors=[]
    for p in ['/srv/ai-server/state/acceleration-experiments/locks/host.lock','/run/ai-server-experiment.lock']:
        fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC); descriptors.append(fd)
        st=os.fstat(fd); assert st.st_uid==0 and not st.st_mode&0o022
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=json.loads((ROOT/'disk-budget.json').read_text())
    raw=(ROOT/'artifact-manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==budget['manifest_sha256']
    artifacts=json.loads(raw)['artifacts']
    before=measure(Path('/srv/ai-server'))
    baseline=ROOT/'preparation-baseline.json'
    if args.phase=='download':
        with baseline.open('x') as f: json.dump(asdict(before),f)
    initial=json.loads(baseline.read_text())
    def guard(limit,future=0):
        snap=measure(Path('/srv/ai-server'))
        growth=max(0,initial['root_free_bytes']-snap.root_free_bytes,snap.ai_usage_bytes-initial['ai_usage_bytes'])
        decision=admit(snap,future,StoragePolicy(root_free_floor_bytes=32*2**30))
        print(json.dumps({'storage':asdict(snap),'growth':growth,'limit':limit,'decision':asdict(decision)}),flush=True)
        if growth>limit or not decision.allowed: raise RuntimeError('storage_bound_or_reserve_exceeded')
    guard(budget['admission_expected_growth_bytes'],budget['admission_expected_growth_bytes'])
    wheels=ROOT/'wheelhouse'
    if args.phase=='download':
        wheels.mkdir(mode=0o755)
        for item in artifacts:
            name=item['filename']; assert Path(name).name==name and name.endswith('.whl')
            final=wheels/name; partial=wheels/(name+'.partial')
            assert not final.exists() and not partial.exists()
            subprocess.run(['curl','--fail','--location','--proto','=https','--proto-redir','=https',
                '--silent','--show-error','--connect-timeout','20','--max-time','1800',
                '--max-filesize',str(item['size']),'--output',str(partial),item['url']],check=True,timeout=1810)
            assert partial.stat().st_size==item['size']
            with partial.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
            with zipfile.ZipFile(partial) as z:
                assert len(z.infolist())==item['entries']
                assert sum(i.file_size for i in z.infolist())==item['uncompressed_bytes']
            partial.rename(final)
            print(json.dumps({'verified_wheel':name,'bytes':item['size']}),flush=True)
            guard(budget['download_phase_bound_bytes'])
        (ROOT/'download-complete.json').write_text(json.dumps({'manifest_sha256':budget['manifest_sha256']}))
    else:
        assert json.loads((ROOT/'download-complete.json').read_text())['manifest_sha256']==budget['manifest_sha256']
        for item in artifacts:
            p=wheels/item['filename']; assert not p.is_symlink() and p.stat().st_size==item['size']
            with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
        base=json.loads(checked(['docker','image','inspect',BASE_TAG]))[0]
        assert base['Id']==BASE
        assert not checked(['docker','images','-q',TAG]).strip()
        with (ROOT/'build.log').open('xb') as log:
            subprocess.run(['docker','build','--pull=false','--network=none','--progress=plain','--tag',TAG,
                '--iidfile',str(ROOT/'image-id.txt'),str(ROOT)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        image=json.loads(checked(['docker','image','inspect',TAG]))[0]
        assert image['RootFS']['Layers'][:len(base['RootFS']['Layers'])]==base['RootFS']['Layers']
        versions=None
        for line in (ROOT/'build.log').read_text().splitlines():
            if '{"' in line and '"flashinfer-python":' in line and '"sglang-kernel":' in line:
                versions=json.loads(line[line.index('{'):])
        assert versions is not None
        canon=lambda s:re.sub('[-_.]+','-',s).lower()
        normalized={canon(k):v for k,v in versions.items()}
        for k,v in json.loads((ROOT/'baseline-packages.json').read_text())['packages'].items():
            assert normalized.get(canon(k))==v,(k,'base dependency changed')
        for row in artifacts: assert normalized.get(canon(row['name']))==row['version']
        receipt=dict(id=image['Id'],base_id=BASE,layers=image['RootFS']['Layers'],base_layers=base['RootFS']['Layers'],
            size=image['Size'],labels=image['Config']['Labels'],bundle_sha256=args.experiment_sha256,
            original_bundle_sha256=ORIGINAL_BUNDLE,packages_added={r['name']:r['version'] for r in artifacts},
            installed_versions=versions,wheel_manifest_sha256=budget['manifest_sha256'])
        (ROOT/'installed-image.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'installed_image':image['Id'],'size':image['Size']}),flush=True)
    guard(budget['download_phase_bound_bytes'] if args.phase=='download' else budget['admission_expected_growth_bytes'])
    print(json.dumps({'phase':args.phase,'status':'PASS'}),flush=True)


if __name__=='__main__': main()
