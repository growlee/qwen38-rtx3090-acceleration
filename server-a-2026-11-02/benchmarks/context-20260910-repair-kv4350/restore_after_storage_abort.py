"""One-shot exact restoration after the observed external storage transient."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/srv/ai-server/benchmarks/context-20260910-repair-kv4350')
KEY = 'qwen-context-kv4350-restore-20260910-bf113c51'
REPORT_SHA = 'a0fff7ea92c8f060a986e60ea2a2d9fa09393da82446a72c9f475eba4920294f'


def main():
    with (ROOT/'restoration-after-abort.json').open('x') as out:
        proof = {'started_epoch': time.time(), 'idempotency_key': KEY,
                 'outcome': 'restore_unknown', 'benchmark_dispatched': False}
        try:
            source = Path('/srv/ai-server/benchmarks/quality-remaining-20260908/run_quality_window_remaining.py')
            assert hashlib.sha256(source.read_bytes()).hexdigest() == '4ad216fc2ff90ea9a8971e10904681fd11396f86977bc3e865ae736cf7dcccc5'
            spec = importlib.util.spec_from_file_location('restore_window', source)
            w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
            c, _, h = w.load_runtime_modules()
            assert hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest() == '91150887cd6bd054a7357e301dc1df96e98724315cf4307d105850427cb271c4'
            assert hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest() == '80f26a9445da1610cbec836d28f23f46b9fd32077062f80d17ec0c64c3b767dd'
            locks = []
            for path in (w.CAS_LOCK, w.HOST_LOCK):
                fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
                locks.append(fd); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unit = subprocess.check_output(['systemctl', 'show', 'qwen-context-20260910-repair-kv4350.service',
                                           '-p', 'ActiveState', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
            assert 'ActiveState=failed' in unit and 'MainPID=0' in unit
            assert 'InvocationID=bf113c51638a4151a9215db16b11e139' in unit
            raw = (ROOT/'results/context-window-report.json').read_bytes()
            assert hashlib.sha256(raw).hexdigest() == REPORT_SHA
            window = json.loads(raw)
            assert window['outcome'] == 'restore_unknown'
            assert window['error_code'] == 'root_storage_reserve_exhausted'
            assert window['restore_error_code'] == 'operator_command_failed'
            assert window['source_manifest_sha256'] == 'd422861a845fb35a200e70b9fc0c10bbf1a7d0df76e7b1ab6c383ebc5c783afd'
            owners = window['storage_owned_containers']
            assert set(owners) == {
                'e5aec0877b5d576bdf4395cedbab12c157c537c213d26f9bf54bfb2040857a97',
                '5708e673f43d5fc29ef5cc11ca0b29b9f1eaa19179ce65bead9faca200fb12fd'}
            for cid, name in owners.items():
                row = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
                assert row['Id'] == cid and row['Name'] == '/'+name
                assert not row['State']['Running'] and not row['State']['Restarting']
            expected = window['production_prestate']
            current = w.production_identity(c)
            assert all(current[k] == expected[k] for k in ('id', 'image', 'model'))
            assert not current['running'] and c.qwen_state(c.Operator().qwen_status()) == 'stopped'
            h.empty('closed'); h.runtime(w.PROFILES)
            free = w.available_bytes()
            assert free >= 150*1024**3 + 512*1024**2
            proof.update(preflight_epoch=time.time(), root_free_before_bytes=free,
                         stopped_owned_containers=owners, production_before=current,
                         source_report_sha256=REPORT_SHA)
            out.write(json.dumps(proof, sort_keys=True)); out.flush(); os.fsync(out.fileno())

            class RestoreOperator(c.Operator):
                def qwen(self, verb):
                    assert verb == 'start'
                    value = c.operator({'verb': 'start', 'profile': 'qwen', 'idempotency_key': KEY})
                    assert value.get('profile') == 'qwen' and value.get('state') == 'ready'

            proof['restoration'] = w.restore(c, h, RestoreOperator(), expected, True)
            proof['admission'] = h.empty('public')
            proof['root_free_after_bytes'] = w.available_bytes()
            proof['outcome'] = 'failed_restored'
        except Exception as exc:
            proof['error'] = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            proof['ended_epoch'] = time.time()
            out.seek(0); out.truncate(); out.write(json.dumps(proof, indent=2, sort_keys=True)+'\n')
            out.flush(); os.fsync(out.fileno())
            print(json.dumps(proof, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

