"""Finite wheel download or offline image build; no cleanup or retry."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path('/srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918')
BASE = 'sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0'
TAG = 'local/ai-server-freetoken:0.1.3-cac247a-nvfp4-20260918'
NETWORK = 'freetoken013-nvfp4-internal'
sys.path.insert(0, '/opt/homelab/ai-server')
from storage_guard.storage_guard import measure, admit, StoragePolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['download', 'build'])
    args = parser.parse_args()
    assert os.geteuid() == 0 and Path(__file__).resolve().parent == ROOT
    os.umask(0o022)
    locks = []
    for path in ['/srv/ai-server/state/acceleration-experiments/locks/host.lock', '/run/ai-server-experiment.lock']:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        locks.append(fd)
        st = os.fstat(fd)
        assert st.st_uid == 0 and not st.st_mode & 0o022
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget = json.loads((ROOT / 'disk-budget.json').read_text())
    manifest = ROOT / 'artifact-manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == budget['manifest_sha256']
    before = measure(Path('/srv/ai-server'))
    baseline = ROOT / 'preparation-baseline.json'
    if args.phase == 'download':
        with baseline.open('x') as f:
            json.dump(asdict(before), f)
    original = json.loads(baseline.read_text())

    def guard(phase_limit, future=0):
        snap = measure(Path('/srv/ai-server'))
        growth = max(0, original['root_free_bytes'] - snap.root_free_bytes,
                     snap.ai_usage_bytes - original['ai_usage_bytes'])
        decision = admit(snap, future, StoragePolicy(root_free_floor_bytes=32 * 2**30))
        print(json.dumps({'storage': asdict(snap), 'growth': growth, 'limit': phase_limit,
                          'decision': asdict(decision)}), flush=True)
        if growth > phase_limit or not decision.allowed:
            raise RuntimeError('phase_storage_bound_or_reserve_exceeded')
        return snap

    guard(budget['admission_expected_growth_bytes'], budget['admission_expected_growth_bytes'])
    wheels = ROOT / 'wheelhouse'
    if args.phase == 'download':
        wheels.mkdir(mode=0o755)
        metadata = {x['filename']: x for x in json.loads((ROOT / 'wheel-physical-metadata.json').read_text())['wheels']}
        for item in json.loads(manifest.read_text())['artifacts']:
            name = item['filename']
            assert Path(name).name == name and name.endswith('.whl')
            target = wheels / name
            partial = wheels / (name + '.partial')
            assert not target.exists() and not partial.exists()
            subprocess.run(['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
                            '--silent', '--show-error', '--connect-timeout', '20', '--max-time', '1800',
                            '--max-filesize', str(item['size']), '--output', str(partial), item['url']], check=True)
            assert partial.stat().st_size == item['size']
            with partial.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == item['sha256']
            with zipfile.ZipFile(partial) as z:
                infos = z.infolist()
                assert sum(i.file_size for i in infos) == metadata[name]['uncompressed_bytes']
                assert len(infos) == metadata[name]['entries']
            partial.rename(target)
            print(json.dumps({'verified_wheel': name, 'bytes': item['size'], 'sha256': item['sha256']}), flush=True)
            guard(budget['phase_growth_bounds']['download'])
        (ROOT / 'download-complete.json').write_text(json.dumps({'manifest_sha256': budget['manifest_sha256'], 'count': len(metadata)}))
    else:
        assert json.loads((ROOT / 'download-complete.json').read_text())['manifest_sha256'] == budget['manifest_sha256']
        for item in json.loads(manifest.read_text())['artifacts']:
            p = wheels / item['filename']
            assert p.stat().st_size == item['size'] and not p.is_symlink()
            with p.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == item['sha256']
        assert not subprocess.check_output(['docker', 'images', '-q', TAG]).strip()
        assert not subprocess.check_output(['docker', 'network', 'ls', '-q', '--filter', 'name=^' + NETWORK + '$']).strip()
        actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', BASE]))[0]
        assert actual['Id'] == BASE
        with (ROOT / 'build.log').open('xb') as log:
            subprocess.run(['docker', 'build', '--pull=false', '--network=none', '--progress=plain',
                            '--tag', TAG, '--iidfile', str(ROOT / 'image-id.txt'), str(ROOT)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', TAG]))[0]
        assert image['RootFS']['Layers'][:len(actual['RootFS']['Layers'])] == actual['RootFS']['Layers']
        # One candidate-owned internal bridge preserves the required loopback endpoint.
        # No host firewall edits, external routing or shared network changes.
        network_id = subprocess.check_output(['docker','network','create','--internal','--driver','bridge',
                '--label','ai.server.experiment=qwen38-nvfp4-20260918',NETWORK],text=True).strip()
        (ROOT / 'installed-image.json').write_text(json.dumps({'id': image['Id'], 'size': image['Size'],
                'layers': image['RootFS']['Layers'], 'labels': image['Config']['Labels'],
                'base_id': BASE, 'network_id': network_id,
                'bundle_sha256':hashlib.sha256((ROOT/'candidate-inputs.json').read_bytes()).hexdigest()}, indent=2))
    guard(budget['phase_growth_bounds']['download' if args.phase == 'download' else 'build_total'])
    print(json.dumps({'phase': args.phase, 'status': 'PASS'}), flush=True)


if __name__ == '__main__':
    main()
