"""Bounded preparation for this matrix only; never delete or replace artifacts."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey/f958acb153eb8c7bbd25638149c8c75185559226/model')
FAST_LEAF = Path('/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey-fast-mtp/f958acb153eb8c7bbd25638149c8c75185559226-syv69ba4d0-int8-vocab40960')
FAST_RECEIPT_SHA = '33ef6b4ab8d6047892dc1c123d87bcbacd22c382e85e59426a191e437ee89521'
SOURCE_RECEIPT_SHA = 'a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a'
MUTATED = {'config.json', 'model.safetensors.index.json', 'model_extra_tensors.safetensors'}
BACKUPS = {'config.json.bak-mtp', 'model.safetensors.index.json.bak-mtp', 'model_extra_tensors.safetensors.bak-mtp', 'model_extra_tensors.safetensors.bak-draft'}
FAST_BOUND = 4 * 1024**3
RESERVE = 150 * 1024**3
WORKER_NAME = 'qwen-quality-remaining-20260908-prepare-int8'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def receipt():
    path = ROOT / 'fast-mtp-historical.receipt'
    if sha(path) != FAST_RECEIPT_SHA:
        raise RuntimeError('historical_receipt_identity_invalid')
    return json.loads(path.read_bytes())


def free(path=SOURCE):
    fs = os.statvfs(path)
    return fs.f_bavail * fs.f_frsize


def budget(before, bound):
    now = free()
    if now < RESERVE or before - now > bound:
        raise RuntimeError('preparation_storage_bound_exceeded')
    return now


def checked_directory(path):
    # Reject symlink ancestors, including absent target ancestors before mkdir.
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise RuntimeError('symlink_in_preparation_path')
        if ancestor.exists() and not ancestor.is_dir():
            raise RuntimeError('non_directory_preparation_path')


def verify_fast(*, require_receipt=True):
    expected = receipt()
    model = FAST_LEAF / 'model'
    checked_directory(model)
    if require_receipt and sha(FAST_LEAF / 'COMPLETE.receipt') != FAST_RECEIPT_SHA:
        raise RuntimeError('recovered_receipt_identity_invalid')
    if {p.name for p in model.iterdir()} != {x['name'] for x in expected['files']}:
        raise RuntimeError('recovered_file_set_invalid')
    records = []
    for item in expected['files']:
        p = model / item['name']
        info = p.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o222:
            raise RuntimeError('recovered_file_custody_invalid')
        if info.st_size != item['size'] or sha(p) != item['sha256']:
            raise RuntimeError('recovered_bytes_differ_from_history:' + p.name)
        if p.name in expected['layout']['hardlinked_unchanged_source_files']:
            original = (SOURCE / p.name).stat()
            if (info.st_dev, info.st_ino) != (original.st_dev, original.st_ino):
                raise RuntimeError('recovered_source_reference_changed')
        records.append({'name':p.name,'sha256':item['sha256'],'size':info.st_size,'allocated_bytes_now':info.st_blocks*512})
    # Historical allocated_bytes/storage are provenance, never today's budget.
    return {'receipt_sha256':FAST_RECEIPT_SHA,'historical_bytes_match':True,'files':records}


def verify_source():
    checked_directory(SOURCE)
    if sha(SOURCE.parent/'COMPLETE.receipt')!=SOURCE_RECEIPT_SHA:
        raise RuntimeError('source_receipt_identity_invalid')
    rows=[]
    for line in (SOURCE.parent/'COMPLETE.receipt').read_text(encoding='ascii').splitlines()[6:]:
        digest,size,name=line.split('  ')
        if Path(name).name!=name:raise RuntimeError('source_receipt_name_invalid')
        rows.append((name,int(size),digest))
    if len(rows)!=16 or {n for n,_,_ in rows}!={p.name for p in SOURCE.iterdir()}:
        raise RuntimeError('source_file_set_changed')
    for name,size,digest in rows:
        p=SOURCE/name;info=p.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_mode&0o222:
            raise RuntimeError('source_custody_invalid')
        if info.st_size!=size or sha(p)!=digest:
            raise RuntimeError('source_bytes_differ_from_receipt:'+name)
    return {'files_hashed':16,'receipt_sha256':SOURCE_RECEIPT_SHA,'all_payload_sha256_match':True}


def worker_args(window, model, unchanged):
    args = ['docker','create','--pull','never','--name',WORKER_NAME,'--restart','no',
            '--network','none','--read-only','--user','0:0','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--cpus','2','--pids-limit','128',
            '--memory','12g','--memory-swap','12g','--log-driver','none',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--tmpfs','/cache:rw,noexec,nosuid,size=64m',
            '--tmpfs','/app/models:rw,noexec,nosuid,size=16m',
            '--env','NVIDIA_VISIBLE_DEVICES=void','--env','HF_HUB_OFFLINE=1',
            '--env','TRANSFORMERS_OFFLINE=1','--env','HF_HOME=/cache',
            '--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
            '--mount',f'type=bind,src={model},dst=/model',
            '--mount',f'type=bind,src={ROOT / "prepare_int8_worker.py"},dst=/prepare.py,readonly']
    # Even root in the container cannot mutate hardlinked source files.
    for name in sorted(unchanged):
        args += ['--mount',f'type=bind,src={SOURCE / name},dst=/model/{name},readonly']
    return args + ['--entrypoint','/app/venv/bin/python',window.SYV_IMAGE,'-B','/prepare.py']


def prepare_fast(window, common):
    verify_source()
    checked_directory(FAST_LEAF)
    if FAST_LEAF.exists():
        # Existing incomplete preparation is evidence, not an automatic retry.
        return {'reused':True, **verify_fast()}
    if window.exact_container(common, WORKER_NAME) is not None:
        raise RuntimeError('preparation_container_already_exists')
    if sha(SOURCE.parent / 'COMPLETE.receipt') != SOURCE_RECEIPT_SHA:
        raise RuntimeError('source_receipt_identity_invalid')
    expected = receipt()
    unchanged = set(expected['layout']['hardlinked_unchanged_source_files'])
    if {p.name for p in SOURCE.iterdir()} != unchanged | MUTATED:
        raise RuntimeError('source_file_set_changed')
    for p in SOURCE.iterdir():
        info=p.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o222:
            raise RuntimeError('source_custody_invalid')
    before=free()
    if before < RESERVE + FAST_BOUND:
        raise RuntimeError('fast_preparation_reserve_insufficient')
    FAST_LEAF.mkdir(parents=True,mode=0o700)
    model=FAST_LEAF/'model';model.mkdir(mode=0o700)
    backups=FAST_LEAF/'retained-upstream-backups';backups.mkdir(mode=0o700)
    for p in SOURCE.iterdir():
        if p.name in unchanged:
            os.link(p,model/p.name)
        else:
            # Minimum writable set required by upstream; no full model copy.
            with p.open('rb') as src, (model/p.name).open('xb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2)
            os.chmod(model/p.name,0o644)
    after_seed=budget(before,FAST_BOUND)
    identity=None
    try:
        result=common.run(worker_args(window,model,unchanged),timeout=60,diagnostic_output=True)
        identity=window.created_id(result,'int8_prepare_create_identity_unknown')
        value=window.exact_container(common,identity)
        host=value['HostConfig']
        if host['NetworkMode']!='none' or not host['ReadonlyRootfs'] or host.get('DeviceRequests') or host.get('PortBindings'):
            raise RuntimeError('int8_prepare_isolation_invalid')
        result=common.run(['docker','start','-a',identity],timeout=600,diagnostic_output=True)
        window.write_exclusive(FAST_LEAF/'preparation-output.txt',result.stdout)
        final=window.exact_container(common,identity)
        if final['State']['Running'] or final['State']['ExitCode']!=0 or final['State'].get('OOMKilled'):
            raise RuntimeError('int8_preparation_failed')
    finally:
        if identity is not None:
            window.stop_owned_once(common,WORKER_NAME,identity)
    after_worker=budget(before,FAST_BOUND)
    expected_names={x['name'] for x in expected['files']}
    if {p.name for p in model.iterdir()} != expected_names | BACKUPS:
        raise RuntimeError('upstream_output_set_changed')
    for name in sorted(BACKUPS):
        src=model/name;dst=backups/name
        if src.is_symlink() or not src.is_file() or dst.exists():
            raise RuntimeError('backup_retention_path_invalid')
        src.rename(dst)  # Retain exact new backup in same leaf; never delete.
    for p in model.iterdir():
        if p.name not in unchanged:os.chmod(p,0o444)
    os.chmod(model,0o555)
    evidence=verify_fast(require_receipt=False)
    window.write_exclusive(FAST_LEAF/'COMPLETE.receipt',(ROOT/'fast-mtp-historical.receipt').read_bytes())
    os.chmod(FAST_LEAF/'COMPLETE.receipt',0o444)
    evidence.update({'reused':False,'worker_container_id':identity,'free_before':before,'free_after_seed':after_seed,
                     'free_after_worker':after_worker,'free_after':budget(before,FAST_BOUND),
                     'peak_bound_bytes':FAST_BOUND,'historical_receipt_restored_as_provenance':True,
                     'backups_retained':sorted(BACKUPS)})
    window.write_exclusive(FAST_LEAF/'recovery-evidence.json',json.dumps(evidence,sort_keys=True).encode()+b'\n')
    os.chmod(FAST_LEAF,0o555)
    return evidence
