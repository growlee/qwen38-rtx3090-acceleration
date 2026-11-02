"""One-shot restoration only; never dispatches a benchmark or retries a mutation."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/srv/ai-server/benchmarks/context-20260908-r2')
KEY = 'qwen-context-r2-restore-20260909-8d076ef1'


def main():
    receipt = ROOT/'restoration-after-abort.json'
    with receipt.open('x') as out:
        proof = {'started_epoch': time.time(), 'idempotency_key': KEY,
                 'outcome': 'restore_unknown', 'benchmark_dispatched': False}
        try:
            source = Path('/srv/ai-server/benchmarks/quality-remaining-20260908/run_quality_window_remaining.py')
            assert hashlib.sha256(source.read_bytes()).hexdigest() == '4ad216fc2ff90ea9a8971e10904681fd11396f86977bc3e865ae736cf7dcccc5'
            spec = importlib.util.spec_from_file_location('restore_window', source)
            w = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(w)
            c, _, h = w.load_runtime_modules()
            assert hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest() == '91150887cd6bd054a7357e301dc1df96e98724315cf4307d105850427cb271c4'
            locks = []
            for path in (w.CAS_LOCK, w.HOST_LOCK):
                fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
                locks.append(fd)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unit = subprocess.check_output(['systemctl', 'show', 'qwen-context-20260908-r2.service', '-p', 'ActiveState', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
            assert 'ActiveState=failed' in unit and 'MainPID=0' in unit
            assert 'InvocationID=2828b0102ddc417c81c886af0e0fe554' in unit
            raw = (ROOT/'results/context-window-report.json').read_bytes()
            proof['source_report_sha256'] = hashlib.sha256(raw).hexdigest()
            window = json.loads(raw)
            assert window['outcome'] == 'restore_unknown' and window['restore_error_code'] == 'qwen_restore_not_ready'
            assert window['source_manifest_sha256'] == '89e8667d4c2031b6ce91fe0f73b206592ea9a6f4066b2e9328b4d570e209299e'
            owners = window['storage_owned_containers']
            assert len(owners) == 10
            for cid, name in owners.items():
                row = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
                assert row['Id'] == cid and row['Name'] == '/'+name
                assert not row['State']['Running'] and not row['State']['Restarting']
            expected = window['production_prestate']
            current = w.production_identity(c)
            assert all(current[k] == expected[k] for k in ('id', 'image', 'model'))
            assert not current['running'] and c.qwen_state(c.Operator().qwen_status()) == 'stopped'
            h.empty('closed')
            h.runtime(w.PROFILES)
            free = w.available_bytes()
            assert free >= 150*1024**3 + 512*1024**2
            proof.update(preflight_epoch=time.time(), root_free_before_bytes=free,
                         stopped_owned_containers=owners, production_before=current)
            out.write(json.dumps(proof, sort_keys=True)); out.flush(); os.fsync(out.fileno())

            class RestoreOperator(c.Operator):
                def qwen(self, verb):
                    assert verb == 'start'
                    value = c.operator({'verb': 'start', 'profile': 'qwen', 'idempotency_key': KEY})
                    assert value.get('profile') == 'qwen' and value.get('state') == 'ready'

            proof['restoration'] = w.restore(c, h, RestoreOperator(), expected, True)
            proof['admission'] = h.empty('public')
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
