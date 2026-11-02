"""Bounded DFlash prefix experiment; reuse the installed exclusive_window guard."""
from __future__ import annotations
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

ROOT = Path('/srv/ai-server/artifacts/experiments/dflash-prefix-20260919')
BASE = Path('/srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918')
SPEED = Path('/srv/ai-server/artifacts/experiments/freetoken013-accel-20260918')
IMAGE = 'sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0'
REVISION = '69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c'
NAME = 'ai-exp-dflash-prefix-20260919'
NETWORK = 'freetoken013-nvfp4-loopback'
NETWORK_ID = '45324081b82e4f6f3a792747375ac79c0a51f895e9f1e81f3c3696b81dfb01c9'
TARGET = Path('/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey/f958acb153eb8c7bbd25638149c8c75185559226')
DRAFT = Path('/srv/ai-server/staging/experiments/qwen38-syv-leminkozey-dflash2/f958acb153eb8c7bbd25638149c8c75185559226/drafter/4d30ec736ffc6b8688dc2ae2b502d9b48bdec279')
MODEL_MOUNT = '/app/models/Qwen3.8-27B-W4A16-AutoRound'
DRAFT_MOUNT = '/app/models/Qwen3.8-27B-DFlash2-W4A16'
BOUND = 512 * 2**20
RESERVE = 32 * 2**30
MEMORY_LIMIT = 24 * 2**30
HOST_MEMORY_RESERVE = 2 * 2**30
HELPERS = {
    BASE/'admission_window.py': '412324fe3a8023406ba4d8ae77da6b5422891c837f21a2fae364f3832779f5c0',
    BASE/'smoke_client.py': '2e7c429177966929e82412befc493501d469cb019ebf9f2d769190392804391a',
    SPEED/'run_speed.py': '5e77af8c3a85ca78b23a00a04a7cc73cec038a34a9ebb093ccd29aa9534b1886',
}
ENV = {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
       'DO_NOT_TRACK':'1','HOME':'/cache','XDG_CACHE_HOME':'/cache',
       'PYTHONDONTWRITEBYTECODE':'1','MODELS_DIR':'/app/models',
       'MODEL':MODEL_MOUNT,'DRAFT':DRAFT_MOUNT,'PORT':'18020','PREPARE':'0','VERIFY':'1',
       'VLLM_OFFLOAD_KEEP_SHM':'1','SPEC':'dflash2','CTX':'fast','PREFIX_CACHE':'1',
       'DFLASH_MAX_LEN':'32768','DFLASH_TOKENS':'7','MAX_SEQS':'1',
       'GPU_UTIL':'0.93','VISION':'0','TOOLS':'0','VLLM_DFLASH2_CHAIN':'0'}

def run(args, timeout=30):
    return subprocess.run(args, check=True, capture_output=True, timeout=timeout)

def inspect(name):
    return json.loads(run(['docker','inspect',name]).stdout)[0]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def custody(path):
    for q in [path, *path.parents]:
        st=q.lstat()
        if q.is_symlink() or st.st_uid!=0 or st.st_mode & 0o022:
            raise RuntimeError('unsafe_source_custody:'+str(q))

def free():
    st=os.statvfs(ROOT)
    return st.f_bavail*st.f_frsize

def verify_weights(leaf, expected_receipt):
    custody(leaf)
    receipt=leaf/'COMPLETE.receipt'
    if sha(receipt)!=expected_receipt: raise RuntimeError('receipt_changed')
    files=[]
    for line in receipt.read_text().splitlines():
        match=re.fullmatch(r'([a-f0-9]{64})  ([0-9]+)  ([A-Za-z0-9._-]+)',line)
        if not match: continue
        digest,size,name=match.groups(); p=leaf/'model'/name; custody(p); st=p.stat()
        if st.st_size!=int(size) or sha(p)!=digest: raise RuntimeError('weight_identity:'+name)
        files.append({'file':name,'sha256':digest,'size':st.st_size,'inode':st.st_ino,
                      'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,'allocated':st.st_blocks*512})
    if not files or sorted(p.name for p in (leaf/'model').iterdir())!=sorted(x['file'] for x in files):
        raise RuntimeError('model_files_changed')
    return {'path':str(leaf),'receipt_sha256':expected_receipt,'files':files}

def same_weights(proof):
    for row in proof['files']:
        p=Path(proof['path'])/'model'/row['file'];custody(p);st=p.stat()
        if (st.st_size,st.st_ino,st.st_mtime_ns,st.st_ctime_ns)!=(row['size'],row['inode'],row['mtime_ns'],row['ctime_ns']):
            raise RuntimeError('verified_weights_drifted')

def create_args(manifest_sha):
    args=['docker','create','--pull=never','--name',NAME,'--restart=no','--gpus=device=0',
          '--network='+NETWORK,'--publish=127.0.0.1:19192:18020','--ipc=private','--read-only',
          '--user=0:0','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=2048',
          '--cpus=6','--memory=24g','--memory-swap=24g','--shm-size=16g',
          '--tmpfs=/tmp:rw,nosuid,nodev,size=4g','--tmpfs=/cache:rw,exec,nosuid,nodev,size=4g',
          '--tmpfs=/app/models:ro,nosuid,nodev,size=1m',
          '--log-opt=max-size=16m','--log-opt=max-file=2',
          '--label=ai.server.experiment.manifest='+manifest_sha]
    for k,v in ENV.items():args+=['--env',k+'='+v]
    for source,dest in [(str(TARGET/'model'),MODEL_MOUNT),(str(DRAFT/'model'),DRAFT_MOUNT),
                        (str(ROOT/'smoke_client.py'),'/smoke_client.py')]:
        args+=['--mount',f'type=bind,source={source},target={dest},readonly']
    return args+[IMAGE,'single']

def matches(value, manifest_sha):
    c,h=value['Config'],value['HostConfig']
    mounts={(x['Source'],x['Destination'],x['RW']) for x in value['Mounts'] if x['Type']=='bind'}
    expected={(str(TARGET/'model'),MODEL_MOUNT,False),(str(DRAFT/'model'),DRAFT_MOUNT,False),
              (str(ROOT/'smoke_client.py'),'/smoke_client.py',False)}
    env=dict(x.split('=',1) for x in c['Env'])
    return (value['Name']=='/'+NAME and value['Image']==IMAGE and c['Cmd']==['single']
        and c['Entrypoint']==['bash','docker/entrypoint.sh'] and c['User']=='0:0'
        and c.get('Labels',{}).get('ai.server.experiment.manifest')==manifest_sha
        and h['RestartPolicy']['Name']=='no' and h['ReadonlyRootfs'] and not h['Privileged']
        and h['NetworkMode']==NETWORK and h['PortBindings']=={'18020/tcp':[{'HostIp':'127.0.0.1','HostPort':'19192'}]}
        and h['Memory']==MEMORY_LIMIT and h['MemorySwap']==MEMORY_LIMIT and h['NanoCpus']==6*10**9
        and h['PidsLimit']==2048 and h['ShmSize']==16*2**30 and h['IpcMode']=='private'
        and h['CapDrop']==['ALL'] and not h['CapAdd'] and not h['Devices']
        and h['SecurityOpt'] in (['no-new-privileges'],['no-new-privileges:true'])
        and len(h['DeviceRequests'])==1 and h['DeviceRequests'][0]['DeviceIDs']==['0']
        and h['Tmpfs']=={'/tmp':'rw,nosuid,nodev,size=4g','/cache':'rw,exec,nosuid,nodev,size=4g',
                           '/app/models':'ro,nosuid,nodev,size=1m'}
        and mounts==expected and all(x['Type'] in ('bind','tmpfs') for x in value['Mounts'])
        and all(env.get(k)==v for k,v in ENV.items()))

def request(path):
    c=http.client.HTTPConnection('127.0.0.1',19192,timeout=5)
    try:
        c.request('GET',path);r=c.getresponse();body=r.read(2**20)
        if r.status!=200:raise RuntimeError('not_ready')
        return body
    finally:c.close()

def fingerprint():
    q=inspect('ai-server-qwen')
    fields={k:q[k] for k in ['Id','Image','Name','Config','HostConfig','Mounts','State']}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()

def main():
    if os.geteuid()!=0 or Path(__file__).resolve().parent!=ROOT:raise RuntimeError('execution_identity')
    custody(ROOT)
    manifest_path=ROOT/'inputs.json'; custody(manifest_path); manifest_sha=sha(manifest_path)
    if len(sys.argv)!=2 or sys.argv[1]!=manifest_sha:raise RuntimeError('manifest_identity')
    manifest=json.loads(manifest_path.read_text())
    if set(manifest)!={'run_smoke.py','smoke_client.py','disk-budget.json'}:raise RuntimeError('manifest_file_set')
    for name,digest in manifest.items():
        p=ROOT/name; custody(p)
        if sha(p)!=digest:raise RuntimeError('source_changed:'+name)
    for p,digest in HELPERS.items():
        custody(p)
        if sha(p)!=digest:raise RuntimeError('helper_changed')
    sys.path[:0]=[str(SPEED),str(BASE)]
    from run_speed import production_safe,gpu_idle,memory_available
    from admission_window import exclusive_window,Admission,checked_client,require_empty
    from smoke_client import Vram
    production_safe();gpu_idle()
    if free()<RESERVE+BOUND:raise RuntimeError('insufficient_physical_disk')
    image=inspect(IMAGE)
    if image['Config']['Labels']['org.opencontainers.image.revision']!=REVISION:raise RuntimeError('image_revision')
    net=json.loads(run(['docker','network','inspect',NETWORK]).stdout)[0]
    if (net['Id']!=NETWORK_ID or net['Driver']!='bridge' or net['Internal'] or net['Containers']
        or net['Options']!={'com.docker.network.bridge.enable_ip_masquerade':'false','com.docker.network.bridge.enable_icc':'false'}):
        raise RuntimeError('network_contract')
    proofs=[verify_weights(TARGET,'a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a'),
            verify_weights(DRAFT,'4908dc8560ffa3c236c57adc579c3a3fdec3ed3df9a2c7b7ca8861356e788238')]
    out=ROOT/('run-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(mode=0o700)
    def record(event,data):
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,'data':data})+'\n')
    initial_free=free();original=fingerprint();owned={'id':None,'attempted':False}
    record('prestate',{'qwen':production_safe(),'fingerprint':original,'memory':memory_available(),'free_bytes':initial_free,'weights':proofs})
    (out/'launch-args.json').write_text(json.dumps(create_args(manifest_sha),indent=2)+'\n')
    result={'status':'FAIL','phase':'preflight','restoration_verified':False,'output':str(out)}
    def candidate():
        ids=run(['docker','ps','-aq','--filter','name=^/'+NAME+'$']).stdout.split()
        return inspect(ids[0].decode()) if len(ids)==1 else None
    def restore():
        value=candidate()
        if value is not None:
            if (not matches(value,manifest_sha) or not owned['attempted']
                or (owned['id'] is not None and value['Id']!=owned['id'])):raise RuntimeError('foreign_candidate_no_stop')
            owned['id']=value['Id']
            if value['State']['Running']:
                try:run(['docker','stop','--time=30',owned['id']],timeout=45)
                except Exception:
                    if inspect(owned['id'])['State']['Running']:raise RuntimeError('candidate_stop_unknown')
            value=inspect(owned['id']);record('candidate_final',value['State'])
            if value['State']['Running']:raise RuntimeError('candidate_still_running')
        elif owned['id']:raise RuntimeError('owned_candidate_missing')
        gpu_idle();production_safe()
        if fingerprint()!=original:raise RuntimeError('original_qwen_changed')
        record('production_restored',{'fingerprint':fingerprint(),'memory':memory_available()})
    def interrupted(sig,_frame):raise RuntimeError('interrupted:'+str(sig))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    sampler=Vram()
    try:
        if candidate() is not None:raise RuntimeError('candidate_name_exists_use_new_reviewed_run')
        with exclusive_window(restore,record):
            production_safe();gpu_idle()
            if fingerprint()!=original:raise RuntimeError('prestate_drift')
            for proof in proofs:same_weights(proof)
            with socket.socket() as sock:sock.bind(('127.0.0.1',19192))
            if free()<RESERVE+BOUND:raise RuntimeError('storage_drift')
            memory=memory_available()
            if memory['mem_available_bytes']<MEMORY_LIMIT+HOST_MEMORY_RESERVE:
                raise RuntimeError('host_memory_reserve_changed')
            record('memory_admission',dict(memory,limit_bytes=MEMORY_LIMIT,host_reserve_bytes=HOST_MEMORY_RESERVE))
            sampler.thread.start();start=time.monotonic();result['phase']='startup';owned['attempted']=True
            run(create_args(manifest_sha),timeout=60)
            value=candidate()
            if value is None or not matches(value,manifest_sha):raise RuntimeError('candidate_contract')
            owned['id']=value['Id'];run(['docker','start',owned['id']],timeout=60)
            while True:
                if not inspect(owned['id'])['State']['Running']:raise RuntimeError('startup_container_exited')
                try:
                    request('/health');models=json.loads(request('/v1/models'))
                    if not any(x.get('id')=='qwen3.8-27b' and x.get('max_model_len')==32768 for x in models['data']):raise ValueError('model_context_mismatch')
                    break
                except (OSError,http.client.HTTPException,RuntimeError):
                    if time.monotonic()-start>900:raise RuntimeError('startup_timeout')
                    time.sleep(5)
            top=run(['docker','top',owned['id'],'-eo','pid,args']).stdout.decode()
            required=['--max-model-len 32768','--max-num-seqs 1','--enable-prefix-caching','--mamba-cache-mode align',
                      '"method":"dflash"','"num_speculative_tokens":7','--language-model-only','--kv-cache-dtype bfloat16']
            if not all(x in top for x in required):raise RuntimeError('effective_runtime_arguments')
            logs=run(['docker','logs',owned['id']]);both=logs.stdout+logs.stderr
            if b'Resolved architecture: DFlash2DraftModel' not in both:raise RuntimeError('draft_architecture_unproven')
            record('ready',{'seconds':time.monotonic()-start,'models':models,'process':top,
                            'ports':inspect(owned['id'])['NetworkSettings']['Ports']})
            (out/'startup.log').write_bytes(both)
            result['phase']='requests'
            with (out/'matrix.jsonl').open('wb') as stdout,(out/'matrix.stderr').open('wb') as stderr:
                completed=subprocess.run(['docker','exec',owned['id'],'/app/venv/bin/python','-B','/smoke_client.py'],
                                         stdout=stdout,stderr=stderr,timeout=900)
            rows=[json.loads(line) for line in (out/'matrix.jsonl').read_text().splitlines() if line]
            record('client_exit',{'returncode':completed.returncode})
            cases=[x for x in rows if x.get('case')]
            if completed.returncode or len(cases)!=3 or any(x.get('status')!='PASS' for x in cases):raise RuntimeError('smoke_pair_failed')
            result['status']='PASS';result['phase']='complete'
        require_empty(Admission(checked_client(),record).request(),'public')
        result['restoration_verified']=True
    except BaseException as exc:
        result.update(status='FAIL',error=type(exc).__name__+': '+str(exc))
        try:
            production_safe();gpu_idle()
            if fingerprint()!=original:raise RuntimeError('original_qwen_changed')
            if owned['id'] and inspect(owned['id'])['State']['Running']:raise RuntimeError('candidate_running')
            require_empty(Admission(checked_client(),record).request(),'public')
            result['restoration_verified']=True
        except Exception as e:result['restoration_error']=str(e)
    finally:
        sampler.stop.set()
        if sampler.thread.ident:sampler.thread.join(timeout=6)
        record('resources',{'vram_samples':sampler.samples,'errors':sampler.errors,'memory':memory_available(),
                            'initial_free_bytes':initial_free,'final_free_bytes':free(),'disk_growth_bytes':initial_free-free()})
        if free()<RESERVE or initial_free-free()>BOUND:result.update(status='FAIL',storage_bound_exceeded=True)
        if owned['id']:
            logs=run(['docker','logs',owned['id']]);(out/'candidate.stdout.log').write_bytes(logs.stdout);(out/'candidate.stderr.log').write_bytes(logs.stderr)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
