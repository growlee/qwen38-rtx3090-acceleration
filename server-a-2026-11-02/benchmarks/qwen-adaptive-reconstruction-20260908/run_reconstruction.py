"""One newly calibrated k4 candidate; reuse the frozen capture/restoration code."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/srv/ai-server/benchmarks/quality-remaining-20260908')
OLD_MANIFEST = '0a6ba209c1a9320743c9e455af3934afe47f9d856e7c01e35ff6665cb2ab35e3'
MODE = 'adaptive-k4-reconstructed-20260908'
PREFIX = 'qwen-adaptive-reconstruction-20260908'
PREP_BOUND = 20_186_937_672
TOTAL_BOUND = PREP_BOUND + 1024**3
SCOPE = Path('/srv/ai-server/staging/experiments/qwen38-syv-uncensored')
WORK = SCOPE / (PREFIX + '-work')
TARGET = SCOPE / (PREFIX + '-candidate')
ACTIVE = {}
BUILDS = []


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_new(expected):
    path = ROOT / 'manifest.json'
    if digest(path) != expected:
        raise RuntimeError('reconstruction_manifest_mismatch')
    files = json.loads(path.read_bytes())['files']
    if set(files) != {'run_reconstruction.py', 'grade_reconstruction.py', 'syv_int4_fast_mtp_runner.py',
                      'syv_gptq_utils_adaptive.py', 'syv_requant_mtp_gptq_adaptive.py'}:
        raise RuntimeError('reconstruction_file_set_mismatch')
    for name, sha in files.items():
        p = ROOT / name
        if p.is_symlink() or digest(p) != sha:
            raise RuntimeError('reconstruction_source_mismatch:' + name)


def configure_builder(builder):
    builder.configure_draft_tokens(4, retain_transient=True, follow_on_peak_bound_bytes=1024**3)
    builder.WORK_ROOT = WORK
    builder.TARGET_LEAF = TARGET
    builder.TARGET_MODEL = TARGET / 'model'
    builder.TARGET_RECEIPT = TARGET / 'COMPLETE.receipt'
    builder.BUILD_CANDIDATE_MODEL = WORK / 'candidate'
    builder.CAL_MODEL = WORK / 'calibration-model'
    builder.CAL_DATA = WORK / 'calibration'
    builder.RUN_DIR = WORK / 'runs/eval'
    builder.LM4_DIR = WORK / 'lm4'
    builder.MTP_WORK_DIR = WORK / 'mtp-work'
    builder.INGRESS_DIR = WORK / 'ingress'
    builder.HESSIANS = builder.RUN_DIR / 'mtp_hessians.pt'
    builder.HESSIAN_DIAGNOSTICS = builder.RUN_DIR / 'gptq_hessian_diagnostics.jsonl'
    builder.CONTAINER = PREFIX + '-prepare'
    builder.VARIANT_ID = MODE
    if builder.PREPARATION_PEAK_BOUND_BYTES != PREP_BOUND:
        raise RuntimeError('preparation_budget_contract_changed')


def retained_builder_args(builder, step, mounts, gpu):
    argv = builder.builder_args(step, mounts=mounts, gpu=gpu)
    argv[1] = 'create'
    argv.remove('--rm')
    pos = argv.index(builder.IMAGE)
    argv[pos:pos] = ['--log-opt', 'max-size=8m', '--log-opt', 'max-file=1',
                     '--cpus', '8', '--tmpfs', '/app/models:rw,nosuid,size=16m']
    # Every inode shared with the immutable source is read-only inside each writable tree.
    source_ids = {(p.stat().st_dev, p.stat().st_ino) for p in builder.SOURCE_MODEL.iterdir()}
    for source, destination, readonly in mounts:
        if readonly or not source.is_dir():
            continue
        for parent, dirs, files in os.walk(source, followlinks=False):
            for name in dirs + files:
                if (Path(parent) / name).is_symlink():
                    raise RuntimeError('symlink_in_builder_tree')
            for name in files:
                p = Path(parent) / name
                info = p.stat()
                if (info.st_dev, info.st_ino) in source_ids:
                    pos = argv.index(builder.IMAGE)
                    target = str(PurePosixPath(destination) / p.relative_to(source).as_posix())
                    argv[pos:pos] = ['--mount', f'type=bind,src={p},dst={target},readonly']
    return argv


def run_builder(window, common, builder, storage, step, command, *, mounts, deadline, env=(), gpu=True):
    if step in ACTIVE or any(r['step'] == step for r in BUILDS):
        raise RuntimeError('builder_step_already_attempted:' + step)
    storage.check('before-' + step)
    name = f'{builder.CONTAINER}-build-{step}'
    if window.exact_container(common, name) is not None:
        raise RuntimeError('builder_name_occupied:' + step)
    # Upstream hardlink calls cannot cross per-file read-only bind mounts. Make
    # only their exact unchanged links on the host first; upstream skips them.
    if step in {'gptq-lm-head', 'requant-mtp-int4'}:
        source = builder.CAL_MODEL if step == 'gptq-lm-head' else builder.MTP_WORK_DIR
        destination = builder.LM4_DIR if step == 'gptq-lm-head' else builder.BUILD_CANDIDATE_MODEL
        destination.mkdir(mode=0o700)
        lm_shard = json.loads((source/'model.safetensors.index.json').read_bytes())['weight_map']['lm_head.weight_packed']
        names = [p.name for p in source.iterdir() if p.name.startswith('model-0000') and p.name.endswith('.safetensors')
                 and (step != 'gptq-lm-head' or p.name != lm_shard)]
        names += ['tokenizer.json']
        if step == 'gptq-lm-head':
            names += ['model_extra_tensors.safetensors','mtp_draft_vocab_ids.pt']
        for filename in names:
            os.link(source/filename,destination/filename)
    argv = retained_builder_args(builder, step, mounts, gpu)
    pos = argv.index(builder.IMAGE)
    for assignment in env:
        argv[pos:pos] = ['--env', assignment]
        pos += 2
    argv += command
    created = common.run(argv, timeout=60, diagnostic_output=True)
    identity = window.created_id(created, 'builder_create_identity_unknown')
    ACTIVE[step] = identity
    value = window.exact_container(common, identity)
    host = value['HostConfig']
    if host['NetworkMode'] != 'none' or not host['ReadonlyRootfs'] or host.get('AutoRemove') or host.get('PortBindings'):
        raise RuntimeError('builder_isolation_invalid')
    guard = None
    process = None
    started = time.monotonic()
    try:
        common.run(['docker', 'start', identity], timeout=60, diagnostic_output=True)
        with window.LoadGuard(common, identity, storage.available_before) as guard:
            process = subprocess.Popen(['docker', 'wait', identity], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            while process.poll() is None:
                if guard.error is not None:
                    raise guard.error
                if time.monotonic() >= deadline:
                    raise RuntimeError('builder_deadline_exhausted:' + step)
                time.sleep(1)
            out, err = process.communicate(timeout=5)
            if process.returncode or out.strip() != b'0':
                raise RuntimeError('builder_failed:' + step)
        final = window.exact_container(common, identity)
        if final['State'].get('OOMKilled') or final['State']['Running'] or final['State']['ExitCode']:
            raise RuntimeError('builder_exit_invalid:' + step)
        raw = common.run(['docker', 'logs', identity], timeout=30).stdout
        window.write_exclusive(ROOT / 'results' / ('prepare-' + step + '.log'), raw)
        BUILDS.append({'step':step, 'id':identity, 'seconds':round(time.monotonic()-started,3),
                       'exit_code':0, 'retained':True, 'log_sha256':hashlib.sha256(raw).hexdigest()})
        print(json.dumps({'phase':'preparation-step-finished','step':step,'seconds':BUILDS[-1]['seconds']}), flush=True)
        return raw
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        window.stop_owned_once(common, name, identity, mutation_allowed=guard is None or not guard.stop_attempted)
        ACTIVE.pop(step)
        storage.check('after-' + step)


def prepare(window, common, builder, mode):
    prep = load('remaining_prep_readonly', OLD / 'prepare_remaining.py')
    prep.verify_source()
    builder.verify_fresh_paths(time.monotonic() + 120)
    patched = builder.verify_patched_recipe_files()
    storage = builder.StorageGuard()
    # Initial discovery found neither pinned BF16 blob by exact size in the managed stores.
    # Recheck immutable CAS addresses and known names before any download.
    for blob in builder.BACKUP_BLOBS:
        for top in ('model-store','staging','artifacts','cache','models-extra'):
            root = Path('/srv/ai-server') / top
            for parent, dirs, files in os.walk(root, followlinks=False):
                for filename in files:
                    p = Path(parent) / filename
                    if not p.is_symlink() and p.stat().st_size == blob.size:
                        raise RuntimeError('existing_backup_candidate_requires_exact_reuse:' + str(p))
    original_download = builder.download_blob
    def bounded_download(blob, deadline):
        storage.check('before-download-' + blob.installed_name)
        original_download(blob, deadline)
        storage.check('after-download-' + blob.installed_name)
    builder.download_blob = bounded_download
    builder.run_builder = lambda step, command, **kw: run_builder(window, common, builder, storage, step, command, **kw)
    records = builder.calibration_records()
    if len(records) != 128:
        raise RuntimeError('calibration_record_count_changed')
    operator = common.Operator()
    state = common.qwen_state(operator.qwen_status())
    if state == 'ready':
        operator.qwen('stop')
    elif state != 'stopped':
        raise RuntimeError('qwen_preparation_state_invalid')
    if common.qwen_state(operator.qwen_status()) != 'stopped':
        raise RuntimeError('qwen_stop_not_proven')
    common.wait_gpu_idle(90)
    deadline = time.monotonic() + 6 * 3600
    try:
        script = ('import json;from transformers import AutoTokenizer;'
                  't=AutoTokenizer.from_pretrained("/model");r=' + repr(records) + ';'
                  'n=[len(t(t.apply_chat_template(x["messages"],tokenize=False,add_generation_prompt=True,'
                  'enable_thinking=x["think"]),add_special_tokens=False)["input_ids"]) for x in r];'
                  'print(max(n));assert len(n)==128 and max(n)<=512')
        builder.run_builder('preflight-tokenizer',['/app/venv/bin/python','-B','-c',script],
                            mounts=((builder.SOURCE_MODEL,'/model',True),),deadline=deadline,gpu=False)
        receipt = builder.prepare_candidate(storage, deadline, patched)
        receipt['reconstruction'] = True
        receipt['historical_identity_claimed'] = False
        receipt['retained_builds'] = BUILDS
        receipt['source_payload_verification'] = prep.verify_source()
        if builder.verify_sealed_candidate(deadline) != receipt['receipt_sha256']:
            raise RuntimeError('sealed_reconstruction_identity_changed')
        return receipt
    finally:
        window.write_exclusive(ROOT / 'results' / 'preparation-evidence.json',
            json.dumps({'reconstruction':True,'historical_identity_claimed':False,
                        'builds':BUILDS,'active_owned':ACTIVE,'storage_samples':storage.samples},sort_keys=True).encode()+b'\n')
        if ACTIVE:
            raise RuntimeError('builder_stop_not_proven')


def main():
    expected = sys.argv[1]
    verify_new(expected)
    old_manifest = OLD / 'bundle-manifest-remaining.json'
    if digest(old_manifest) != OLD_MANIFEST:
        raise RuntimeError('old_bundle_manifest_changed')
    manifest = json.loads(old_manifest.read_bytes())
    for name, sha in {**{'sources/'+n:h for n,h in manifest['files'].items()},**manifest['other_files']}.items():
        if digest(OLD/name) != sha:
            raise RuntimeError('frozen_dependency_changed:' + name)
    window = load('reconstruction_window', OLD / 'run_quality_window_remaining.py')
    original_verify = window.verify_bundle
    window.verify_bundle = lambda ignored: original_verify(OLD_MANIFEST)
    window.verify_bundle(OLD_MANIFEST)
    common, syv, helper = window.load_runtime_modules()
    builder = load('reconstruction_builder', ROOT / 'syv_int4_fast_mtp_runner.py')
    configure_builder(builder)
    # Keep a single module identity for runtime dependencies and the scoped builder hooks.
    window.load_runtime_modules = lambda: (common, syv, helper)
    window.MODES = ({'id':MODE,'kind':'syv','model':TARGET/'model','image':window.SYV_IMAGE,'image_id':window.SYV_IMAGE_ID},)
    window.BLOCKED = {}
    window.MAX_GROWTH_BYTES = TOTAL_BOUND
    window.WINDOW_SECONDS = 7 * 3600
    window.candidate_name = lambda mode: PREFIX + '-' + mode
    window.prepare_candidate = lambda c, mode: prepare(window, c, builder, mode)
    original_restore = window.restore
    def restore(c, h, operator, prestate, closed):
        if ACTIVE:
            raise RuntimeError('active_builder_prevents_restoration')
        return original_restore(c,h,operator,prestate,closed)
    window.restore = restore
    return window.main(['--output-dir',str(ROOT/'results'),'--bundle-manifest-sha256',OLD_MANIFEST])


if __name__ == '__main__':
    raise SystemExit(main())
