"""One bounded, admission-guarded FreeToken smoke matrix on server-a.

Run only after independent review and isolated image preparation. Never builds,
downloads, removes containers/images, changes production config, or starts Qwen.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918')
NAME = 'ai-server-freetoken-q38-013-nvfp4'
NETWORK = 'freetoken013-nvfp4-loopback'
NETWORK_OPTIONS = {'com.docker.network.bridge.enable_ip_masquerade': 'false',
                   'com.docker.network.bridge.enable_icc': 'false'}
COMMIT = 'cac247a860e316e06580d05aeb05f2e647bde214'
QWEN_ID = '15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f'
QWEN_IMAGE = 'sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0'
COMPOSE = Path('/opt/homelab/ai-server/deploy/compose.qwen.yaml')
COMPOSE_SHA = '88c7de590dd7dea693dba225242d81e52387415a1e4e0f5796b568dfc4468c3c'
MODEL = '/srv/ai-server/staging/experiments/qwen38-uncensored-nvfp4-37b5130a/model'
WEIGHT = '/srv/ai-server/model-store/sha256/bc642c1159ab9b0fcd2fdc65be4fee72fa560cf0570604741db50061ea063277/model.safetensors'
TOKENIZER = '/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey/f958acb153eb8c7bbd25638149c8c75185559226/model/tokenizer.json'
SERVE = ['serve', '--model', '/model', '--served-model-name', 'qwen38-nvfp4-freetoken',
         '--host', '0.0.0.0', '--port', '1919', '--max-running-requests', '1',
         '--max-seq-len-override', '32768', '--num-tokens', '32768', '--max-output-tokens', '256',
         '--max-prefill-length', '8192', '--attention-backend', 'triton',
         '--quant-backend', 'linear.nvfp4=triton', '--cache-type', 'radix',
         '--enable-cache-report', '--text-model-only']


def run(args, timeout=20, **kwargs):
    return subprocess.run(args, timeout=timeout, check=True, capture_output=True, **kwargs)


def inspect(name):
    return json.loads(run(['docker', 'inspect', name]).stdout)[0]


def existing_candidate():
    ids = run(['docker', 'ps', '-aq', '--filter', 'name=^/' + NAME + '$']).stdout.split()
    if len(ids) > 1:
        raise RuntimeError('ambiguous_candidate_identity')
    return None if not ids else inspect(ids[0].decode())


def production_safe():
    qwen = inspect('ai-server-qwen')
    if not (qwen['Id'] == QWEN_ID and qwen['Image'] == QWEN_IMAGE
            and not qwen['State']['Running'] and qwen['State']['Status'] == 'exited'
            and qwen['HostConfig']['RestartPolicy']['Name'] == 'no'
            and hashlib.sha256(COMPOSE.read_bytes()).hexdigest() == COMPOSE_SHA):
        raise RuntimeError('production_qwen_prestate_changed')
    gateway = inspect('ai-server-gateway')
    if not gateway['State']['Running'] or gateway['State'].get('Health', {}).get('Status') != 'healthy':
        raise RuntimeError('gateway_not_healthy')
    if run(['systemctl', 'is-active', 'ai-server-supervisor.service']).stdout.strip() != b'active':
        raise RuntimeError('supervisor_not_active')
    return {'qwen': 'stopped', 'container_id': QWEN_ID, 'compose_sha256': COMPOSE_SHA,
            'gateway': 'healthy'}


def gpu_idle():
    raw = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).stdout.strip()
    if raw:
        raise RuntimeError('gpu_has_compute_work')


def memory_available():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    # Record actual availability; the 24 GiB cgroup limit is a ceiling, not
    # an upfront allocation requirement. Do not reject a small reserve delta.
    return {'mem_available_bytes': int(values['MemAvailable'].split()[0]) * 1024,
            'swap_free_bytes': int(values['SwapFree'].split()[0]) * 1024,
            'container_limit_bytes': 24 * 2**30}


def require_capacity(stats, cache):
    # /v1/stats.kv is populated by generation replies, not startup. The cache
    # geometry uses the backend's load-time allocation before the first reply.
    model, geometry = stats.get('model') or {}, cache.get('geometry') or {}
    if (model.get('id') != 'qwen38-nvfp4-freetoken' or model.get('ctx') != 32768
            or model.get('input_modalities') != ['text']
            or cache.get('state') != 'serving'
            or geometry.get('num_pages', 0) * geometry.get('page_size', 0) < 32768):
        raise RuntimeError('requested_model_context_or_kv_capacity_not_proven')


def request_json(path):
    conn = http.client.HTTPConnection('127.0.0.1', 19191, timeout=10)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        body = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(body) > 1024 * 1024:
            raise RuntimeError('http_not_ready_' + path)
        return json.loads(body)
    finally:
        conn.close()


def mounts():
    return {MODEL: '/model', WEIGHT: WEIGHT, TOKENIZER: TOKENIZER,
            str(ROOT / 'instrumentation'): '/instrumentation'}


def candidate_matches(value, image_id, bundle_sha):
    cfg, host = value['Config'], value['HostConfig']
    actual_mounts = {x['Source']: x['Destination'] for x in value['Mounts'] if x['Type'] == 'bind'}
    devices = host.get('DeviceRequests') or []
    gpu_zero_only = (len(devices) == 1 and devices[0].get('DeviceIDs') == ['0']
                     and devices[0].get('Count') == 0
                     and devices[0].get('Driver') in ('', 'nvidia')
                     and devices[0].get('Capabilities') == [['gpu']])
    env = dict(x.split('=', 1) for x in cfg.get('Env', []) if '=' in x)
    return (value['Name'] == '/' + NAME and value['Image'] == image_id
            and cfg.get('Labels', {}).get('ai.server.experiment.bundle') == bundle_sha
            and cfg.get('Cmd') == SERVE and cfg.get('User') == '65532:65532'
            and cfg.get('Entrypoint') == ['/opt/freetoken013/bin/ft']
            and host['RestartPolicy']['Name'] == 'no' and host['ReadonlyRootfs'] is True
            and host.get('NetworkMode') == NETWORK and host.get('Privileged') is False
            and host.get('Memory') == 24 * 2**30 and host.get('MemorySwap') == 24 * 2**30
            and host.get('NanoCpus') == 6 * 10**9 and host.get('PidsLimit') == 512
            and host.get('ShmSize') == 2**30 and gpu_zero_only
            and {x.upper() for x in host.get('CapDrop') or []} == {'ALL'}
            and not host.get('CapAdd') and not host.get('Devices')
            and host.get('SecurityOpt') in (['no-new-privileges'], ['no-new-privileges:true'])
            and host.get('Tmpfs') == {'/tmp': 'rw,uid=65532,gid=65532,size=4g',
                                     '/cache': 'rw,exec,nosuid,nodev,uid=65532,gid=65532,size=2g',
                                     '/app/models': 'ro,uid=65532,gid=65532,size=1m'}
            and env.get('PYTHONPATH') == '/instrumentation'
            and env.get('FT013_JIT_TRACE') == '/cache/jit-trace.jsonl'
            and env.get('TORCHINDUCTOR_CACHE_DIR') == '/cache/torchinductor'
            and host.get('PortBindings') == {'1919/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '19191'}]}
            and actual_mounts == mounts() and all(not x['RW'] for x in value['Mounts'] if x['Type'] == 'bind')
            and all(x['Type'] in ('bind', 'tmpfs') for x in value['Mounts']))


def create_args(image_id, bundle_sha):
    args = ['docker', 'create', '--name', NAME, '--restart=no', '--gpus=device=0',
            '--read-only', '--user=65532:65532', '--cap-drop=ALL',
            '--security-opt=no-new-privileges', '--network=' + NETWORK,
            '--publish=127.0.0.1:19191:1919', '--memory=24g', '--memory-swap=24g', '--shm-size=1g',
            '--cpus=6', '--pids-limit=512',
            '--tmpfs=/tmp:rw,uid=65532,gid=65532,size=4g',
            '--tmpfs=/cache:rw,exec,nosuid,nodev,uid=65532,gid=65532,size=2g',
            '--tmpfs=/app/models:ro,uid=65532,gid=65532,size=1m',
            '--env=PYTHONPATH=/instrumentation', '--env=FT013_JIT_TRACE=/cache/jit-trace.jsonl',
            '--env=TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor',
            '--label=ai.server.experiment.bundle=' + bundle_sha,
            '--log-opt=max-size=16m', '--log-opt=max-file=2']
    for source, destination in mounts().items():
        args += ['--mount', f'type=bind,source={source},target={destination},readonly']
    return args + [image_id] + SERVE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--bundle-sha256', required=True)
    args = parser.parse_args()
    if (os.geteuid() != 0 or Path(__file__).resolve().parent != ROOT
            or not re.fullmatch(r'sha256:[a-f0-9]{64}', args.image_id)
            or not re.fullmatch(r'[a-f0-9]{64}', args.bundle_sha256)):
        raise RuntimeError('execution_identity_or_path')
    manifest = ROOT / 'candidate-inputs.json'
    for directory in (ROOT, *ROOT.parents):
        info = directory.lstat()
        if directory.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError('bundle_directory_custody')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != args.bundle_sha256:
        raise RuntimeError('reviewed_bundle_hash_changed')
    for name, digest in json.loads(manifest.read_text()).items():
        path = ROOT / name
        if (not path.resolve().is_relative_to(ROOT) or path.is_symlink() or path.stat().st_uid != 0
                or path.stat().st_mode & 0o022 or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise RuntimeError('reviewed_source_changed_' + name)
    from admission_window import exclusive_window
    from smoke_client import Vram
    receipt_path = ROOT / 'installed-image.json'
    info = receipt_path.lstat()
    if receipt_path.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022:
        raise RuntimeError('image_receipt_custody')
    receipt = json.loads(receipt_path.read_text())
    if receipt['id'] != args.image_id or receipt['bundle_sha256'] != args.bundle_sha256:
        raise RuntimeError('image_not_bound_to_reviewed_build')
    network = json.loads(run(['docker','network','inspect',NETWORK]).stdout)[0]
    if (network['Id'] != receipt['network_id'] or network['Internal'] is not False
            or network['Driver'] != 'bridge' or network.get('Containers')
            or network.get('Options') != NETWORK_OPTIONS):
        # A stopped candidate is not an attached running endpoint.
        raise RuntimeError('candidate_network_not_isolated_or_idle')
    image = inspect(args.image_id)
    if image['RootFS']['Layers'] != receipt['layers']:
        raise RuntimeError('image_layers_changed')
    labels = image['Config'].get('Labels', {})
    if labels.get('org.opencontainers.image.revision') != COMMIT or labels.get('ai.server.freetoken.version') != '0.1.3':
        raise RuntimeError('unverified_runtime_image')
    # Large weights were fully hashed during preflight; reject changes to the
    # exact verified inode/size/mtime before mapping them again.
    verified = json.loads((ROOT / 'nv-preflight-data.json').read_text())['weights']
    current = Path(WEIGHT).stat()
    if (current.st_size, current.st_ino, current.st_mtime_ns) != (verified['size'], verified['inode'], verified['mtime_ns']):
        raise RuntimeError('verified_weight_file_changed')
    for row in json.loads((ROOT / 'nv-integrity.json').read_text())['files']:
        path = Path(MODEL) / row['file']
        if str(path.resolve()) != row['resolved'] or path.stat().st_size != row['size']:
            raise RuntimeError('model_reference_changed')
        if row['file'] != 'model.safetensors' and hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('small_model_file_changed')
    sys.path.insert(0, '/opt/homelab/ai-server')
    from storage_guard.storage_guard import StoragePolicy, admit, measure
    if not admit(measure(Path('/srv/ai-server')), 256 * 2**20,
                 StoragePolicy(root_free_floor_bytes=32 * 2**30)).allowed:
        raise RuntimeError('runtime_storage_admission_denied')
    output = ROOT / ('run-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(mode=0o700)
    def record(event, value):
        with (output / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'event':event, 'data':value}) + '\n')
    record('prestate', production_safe())
    outcome = {'status':'FAIL', 'restoration_verified':False, 'image_id':args.image_id}
    owned = {'id':None, 'create_attempted':False}
    value = existing_candidate()
    if value is not None:
        if (not candidate_matches(value, args.image_id, args.bundle_sha256)
                or value['State']['Running'] or value['State']['Status'] not in ('created', 'exited')):
            raise RuntimeError('existing_candidate_not_safe_to_reuse')
        owned['id'] = value['Id']
    def restore():
        if owned['id'] is None and not owned['create_attempted']:
            gpu_idle()
            record('production_restored', production_safe())
            return
        value = existing_candidate()
        if value is not None:
            if not candidate_matches(value, args.image_id, args.bundle_sha256):
                raise RuntimeError('foreign_or_changed_candidate_no_stop')
            if owned['id'] is not None and value['Id'] != owned['id']:
                raise RuntimeError('candidate_identity_changed_no_stop')
            owned['id'] = value['Id']
            if value['State']['Running']:
                try:
                    trace = run(['docker','exec',NAME,'cat','/cache/jit-trace.jsonl'], timeout=10).stdout
                    (output/'jit-trace.jsonl').write_bytes(trace)
                except Exception as exc:
                    record('trace_capture_error',type(exc).__name__)
                try:
                    run(['docker','stop','--time=30',value['Id']], timeout=45)
                except Exception:
                    # One bounded read-only reconciliation, never resend stop.
                    if inspect(value['Id'])['State']['Running']:
                        raise RuntimeError('candidate_stop_unknown')
            final = inspect(value['Id'])
            if final['State']['Running'] or final['State']['Status'] not in ('exited','created'):
                raise RuntimeError('candidate_not_stopped')
            record('candidate_final',{'id':final['Id'],'state':final['State'],
                                      'host_memory':memory_available()})
        elif owned['id'] is not None:
            raise RuntimeError('owned_candidate_disappeared')
        gpu_idle()
        record('production_restored', production_safe())
    def interrupted(signum, _frame):
        raise RuntimeError('interrupted_signal_' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        with exclusive_window(restore, record):
            production_safe()
            gpu_idle()
            record('host_memory_before_start', memory_available())
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',19191))
            value = existing_candidate()
            if value is None:
                owned['create_attempted'] = True
                run(create_args(args.image_id,args.bundle_sha256), timeout=60)
                value = existing_candidate()
            if value is None or not candidate_matches(value,args.image_id,args.bundle_sha256) or value['State']['Running']:
                raise RuntimeError('candidate_config_or_prestate_invalid')
            owned['id'] = value['Id']
            sample = Vram()
            sample.thread.start()
            started = time.perf_counter()
            try:
                run(['docker','start',owned['id']], timeout=60)
                deadline = time.monotonic() + 900
                while True:
                    state = inspect(owned['id'])['State']
                    if not state['Running']:
                        raise RuntimeError('candidate_exited_during_load')
                    try:
                        health = request_json('/health')
                        if health.get('status') == 'loading':
                            raise RuntimeError('model_still_loading')
                        stats = request_json('/v1/stats')
                        cache = request_json('/v1/cache/status')
                    except (OSError, http.client.HTTPException, RuntimeError, ValueError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('load_deadline_not_met')
                        time.sleep(5)
                        continue
                    if health.get('status') != 'ok' or health.get('maintenance') != 'serving':
                        raise RuntimeError('backend_not_healthy_' + str(health))
                    require_capacity(stats, cache)
                    effective_ports = inspect(owned['id'])['NetworkSettings']['Ports']
                    if effective_ports.get('1919/tcp') != [{'HostIp':'127.0.0.1','HostPort':'19191'}]:
                        raise RuntimeError('effective_loopback_binding_not_proven')
                    break
                record('ready',{'load_and_startup_jit_seconds':time.perf_counter()-started,'stats':stats,
                                'health':health,'cache':cache,'effective_ports':effective_ports,
                                'vram_samples':sample.samples,'sampling_errors':sample.errors,
                                'host_memory':memory_available()})
            finally:
                sample.stop.set(); sample.thread.join(timeout=6)
                record('startup_observation', {'seconds':time.perf_counter()-started,
                    'vram_samples':sample.samples, 'sampling_errors':sample.errors,
                    'host_memory':memory_available()})
            with (output/'matrix.jsonl').open('wb') as stdout, (output/'matrix.stderr').open('wb') as stderr:
                completed = subprocess.run(['docker','exec','-i',NAME,'/opt/freetoken013/bin/python','-'],
                    input=(ROOT/'smoke_client.py').read_bytes(), stdout=stdout, stderr=stderr, timeout=1800)
            rows = [json.loads(x) for x in (output/'matrix.jsonl').read_text().splitlines() if x]
            cases = [x for x in rows if 'case' in x]
            if completed.returncode or [x['case'] for x in cases] != [
                'short_stream','8000_fresh','8000_continuation','28000_fresh','28000_continuation'] or any(x['status']!='PASS' for x in cases):
                raise RuntimeError('finite_matrix_failed_or_incomplete')
            record('after_matrix', request_json('/v1/stats'))
        outcome.update(status='PASS', restoration_verified=True)
    except BaseException as exc:
        outcome['error'] = type(exc).__name__ + ': ' + str(exc)
        # Inspection only: no retry, queue reconciliation, lifecycle or reopening.
        try:
            production_safe(); gpu_idle()
            if owned['id'] is not None:
                final = inspect(owned['id'])
                if (not candidate_matches(final,args.image_id,args.bundle_sha256)
                        or final['State']['Running'] or final['State']['Status'] not in ('exited','created')):
                    raise RuntimeError('candidate_restoration_not_proven')
            elif owned['create_attempted'] and existing_candidate() is not None:
                raise RuntimeError('created_candidate_restoration_not_proven')
            from admission_window import Admission, checked_client, require_empty
            require_empty(Admission(checked_client(),record).request(),'public')
            outcome['restoration_verified'] = True
        except Exception as restore_error:
            outcome['restoration_error'] = str(restore_error)
    finally:
        if owned['id']:
            try:
                logs = run(['docker','logs','--tail=4000',owned['id']],timeout=15)
                (output/'candidate.stdout.log').write_bytes(logs.stdout)
                (output/'candidate.stderr.log').write_bytes(logs.stderr)
            except Exception as exc:
                outcome['log_error'] = type(exc).__name__
        (output/'result.json').write_text(json.dumps(outcome,indent=2)+'\n')
        print(json.dumps({'output':str(output),**outcome}),flush=True)
    return 0 if outcome['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
