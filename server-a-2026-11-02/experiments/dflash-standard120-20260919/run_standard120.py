"""One standard120 run using the retained DFlash profile and existing guards."""
from __future__ import annotations
import contextlib
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import time

ROOT = Path('/srv/ai-server/artifacts/experiments/dflash-standard120-20260919')
RUNTIME = Path('/srv/ai-server/artifacts/experiments/dflash-agent3-20260919')
BENCH = Path('/srv/ai-server/artifacts/experiments/freetoken013-benchmark120-20260918')
RUNTIME_MANIFEST = '537439b6f0e8efaaba764303a386b83b6e12cf011dc05c2f3a58d1ce2df4b172'
BENCH_MANIFEST = 'a0ea87d1a3bba59d07b8b7d6f9528314c058255af72803373433bf555fca027a'
CANDIDATE_ID = '649d73c700620d1a378c9475178fa40d53f887a95ffa7371a9c8204fd1be0fd9'
MODE = 'dflash2-fast-n7-prefix32k-coding120-20260919'
RESERVE = 32 * 2**30
PHASE_BOUND = 256 * 2**20

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def custody(p):
    for q in [p,*p.parents]:
        s=q.lstat()
        if q.is_symlink() or s.st_uid!=0 or s.st_mode & 0o022:raise RuntimeError('source_custody:'+str(q))

def manifest(root,name,expected):
    p=root/name;custody(p)
    if digest(p)!=expected:raise RuntimeError('manifest_changed:'+str(p))
    value=json.loads(p.read_text())
    for name,sha in value.items():
        p=root/name
        if not p.resolve().is_relative_to(root):raise RuntimeError('manifest_path')
        custody(p)
        if digest(p)!=sha:raise RuntimeError('source_changed:'+str(p))
    return value

def free():
    s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize

def main():
    if os.geteuid()!=0 or Path(__file__).resolve().parent!=ROOT or len(sys.argv)!=2:raise RuntimeError('execution_identity')
    own=manifest(ROOT,'inputs.json',sys.argv[1])
    if set(own)!={'run_standard120.py','benchmark_adapter.py','disk-budget.json'}:raise RuntimeError('manifest_set')
    manifest(RUNTIME,'inputs.json',RUNTIME_MANIFEST)
    manifest(BENCH,'benchmark-inputs.json',BENCH_MANIFEST)
    sys.path.insert(0,str(RUNTIME))
    import run_agent3 as runtime
    for p,sha in runtime.HELPERS.items():
        custody(p)
        if digest(p)!=sha:raise RuntimeError('helper_changed:'+str(p))
    sys.path[:0]=[str(ROOT),str(runtime.PREFIX),str(BENCH),str(BENCH/'harness'),str(runtime.SPEED),str(runtime.BASE)]
    from run_speed import production_safe,gpu_idle,memory_available
    from admission_window import exclusive_window,Admission,checked_client,require_empty
    from smoke_client import Vram,get_metrics
    import smoke_client
    import benchmark_adapter
    import quality_sandbox
    from quality_expanded_suite import load_suite
    smoke_client.HOST='127.0.0.1';smoke_client.PORT=19193
    cases=[x for x in load_suite()['cases'] if x['category']=='coding']
    expected=[x['case_id'] for x in cases]
    if len(expected)!=120 or len(set(expected))!=120:raise RuntimeError('suite_count')
    cpu_names=[quality_sandbox._container_name(MODE,x) for x in expected]
    production_safe();gpu_idle();require_empty(Admission(checked_client(),lambda *_:None).request(),'public')
    candidate=runtime.inspect(CANDIDATE_ID)
    if not runtime.matches(candidate,RUNTIME_MANIFEST) or candidate['State']['Running']:raise RuntimeError('candidate_drift')
    network=json.loads(runtime.run(['docker','network','inspect',runtime.NETWORK]).stdout)[0]
    if (network['Id']!=runtime.NETWORK_ID or network['Containers'] or network['Driver']!='bridge'
        or network['Internal'] or network['Options']!={'com.docker.network.bridge.enable_ip_masquerade':'false','com.docker.network.bridge.enable_icc':'false'}):raise RuntimeError('network_drift')
    image=runtime.inspect(runtime.IMAGE)
    if image['Config']['Labels']['org.opencontainers.image.revision']!=runtime.REVISION:raise RuntimeError('image_revision')
    if free()<RESERVE+580*2**20:raise RuntimeError('disk_budget')
    # No names from this exact new mode may predate the scoring phase.
    existing=runtime.run(['docker','ps','-a','--format','{{.Names}}']).stdout.decode().splitlines()
    if set(cpu_names)&set(existing):raise RuntimeError('score_container_collision')
    proofs=[runtime.verify_weights(runtime.TARGET,'a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a'),
            runtime.verify_weights(runtime.DRAFT,'4908dc8560ffa3c236c57adc579c3a3fdec3ed3df9a2c7b7ca8861356e788238')]
    out=ROOT/('run-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(mode=0o700)
    def record(event,data):
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,'data':data})+'\n')
    original=runtime.fingerprint();initial_free=free();attempted=False
    log_since=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result={'status':'BLOCKED','phase':'preflight','restoration_verified':False,'output':str(out),'mode_id':MODE}
    record('prestate',{'production':production_safe(),'fingerprint':original,'candidate':candidate['State'],'candidate_id':CANDIDATE_ID,'memory':memory_available(),'free_bytes':initial_free,'weights':proofs})
    def check_restored():
        v=runtime.inspect(CANDIDATE_ID)
        if not runtime.matches(v,RUNTIME_MANIFEST) or v['State']['Running']:raise RuntimeError('candidate_restoration_not_proven')
        gpu_idle();production_safe()
        if runtime.fingerprint()!=original:raise RuntimeError('original_qwen_changed')
    def restore():
        v=runtime.inspect(CANDIDATE_ID)
        if not runtime.matches(v,RUNTIME_MANIFEST):raise RuntimeError('changed_candidate_no_stop')
        if v['State']['Running']:
            if not attempted:raise RuntimeError('unowned_running_candidate')
            try:runtime.run(['docker','stop','--time=30',CANDIDATE_ID],timeout=45)
            except Exception:
                if runtime.inspect(CANDIDATE_ID)['State']['Running']:raise RuntimeError('stop_unknown')
        check_restored();record('production_restored',{'fingerprint':runtime.fingerprint(),'candidate_state':runtime.inspect(CANDIDATE_ID)['State']})
    def stop_score_containers():
        found=[]
        names=runtime.run(['docker','ps','-a','--format','{{.Names}}']).stdout.decode().splitlines()
        for name in cpu_names:
            if name not in names:continue
            v=runtime.inspect(name);h=v['HostConfig']
            if (v['Image']!=runtime.IMAGE or h['NetworkMode']!='none' or not h['ReadonlyRootfs']
                or h['Privileged'] or h.get('DeviceRequests') or h.get('Devices') or h.get('CapAdd')
                or h.get('PortBindings') or v['Config']['User']!='65532:65532'
                or any(m['Type']!='tmpfs' for m in v['Mounts'])):raise RuntimeError('score_container_identity')
            if v['State']['Running']:
                try:runtime.run(['docker','stop','--time=2',v['Id']],timeout=10)
                except Exception:
                    if runtime.inspect(v['Id'])['State']['Running']:raise RuntimeError('score_stop_unknown')
            state=runtime.inspect(v['Id'])['State']
            if state['Running']:raise RuntimeError('score_container_running')
            found.append({'id':v['Id'],'name':name,'state':state})
        record('score_containers_stopped',found)
    def interrupt(sig,_):raise RuntimeError('interrupted:'+str(sig))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
    sample=Vram()
    try:
        with exclusive_window(restore,record):
            production_safe();gpu_idle()
            if runtime.fingerprint()!=original:raise RuntimeError('prestate_drift')
            for proof in proofs:runtime.same_weights(proof)
            if not runtime.matches(runtime.inspect(CANDIDATE_ID),RUNTIME_MANIFEST):raise RuntimeError('candidate_prestart_drift')
            if free()<RESERVE+580*2**20:raise RuntimeError('disk_drift')
            if memory_available()['mem_available_bytes']<runtime.MEMORY_LIMIT+runtime.HOST_MEMORY_RESERVE:raise RuntimeError('memory_drift')
            with socket.socket() as sock:sock.bind(('127.0.0.1',19193))
            sample.thread.start();started=time.monotonic();result['phase']='startup';attempted=True
            runtime.run(['docker','start',CANDIDATE_ID],timeout=60)
            while True:
                if not runtime.inspect(CANDIDATE_ID)['State']['Running']:raise RuntimeError('candidate_exited')
                try:
                    runtime.request('/health');models=json.loads(runtime.request('/v1/models'));break
                except (OSError,RuntimeError,http.client.HTTPException):
                    if time.monotonic()-started>900:raise RuntimeError('startup_timeout')
                    time.sleep(5)
            if not any(x.get('id')=='qwen3.8-27b' and x.get('max_model_len')==32768 for x in models['data']):raise RuntimeError('model_identity')
            top=runtime.run(['docker','top',CANDIDATE_ID,'-eo','pid,args']).stdout.decode()
            if not all(x in top for x in ['--max-model-len 32768','--max-num-seqs 1','--enable-prefix-caching','"method":"dflash"','"num_speculative_tokens":7','--tool-call-parser qwen3_coder']):raise RuntimeError('effective_arguments')
            record('ready',{'seconds':time.monotonic()-started,'models':models,'process':top,'ports':runtime.inspect(CANDIDATE_ID)['NetworkSettings']['Ports']})
            result['phase']='generation';before=get_metrics();record('metrics_before',before)
            with (out/'matrix.stdout').open('w') as stdout,(out/'matrix.stderr').open('w') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                code=benchmark_adapter.main(['--generate',str(out)])
            record('metrics_after',get_metrics())
            rows=[json.loads(x) for x in (out/'matrix.jsonl').read_text().splitlines()]
            if code or [x['case_id'] for x in rows if x.get('kind')=='case']!=expected:raise RuntimeError('generation_incomplete')
            growth=initial_free-free();record('generation_storage',{'growth_bytes':growth,'free_bytes':free()})
            if growth>PHASE_BOUND or free()<RESERVE:raise RuntimeError('generation_storage_bound')
        require_empty(Admission(checked_client(),record).request(),'public');result['restoration_verified']=True
        sample.stop.set();sample.thread.join(timeout=6)
        record('vram',{'samples':sample.samples,'errors':sample.errors})
        # Same frozen evaluator, after GPU and admission restoration.
        result['phase']='scoring';score_free=free()
        if memory_available()['mem_available_bytes']<2*2**30+256*2**20:raise RuntimeError('scoring_memory_drift')
        with (out/'score.stdout').open('w') as stdout,(out/'score.stderr').open('w') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            code=benchmark_adapter.main(['--score',str(out)])
        scored=[json.loads(x) for x in (out/'scored.jsonl').read_text().splitlines()]
        if code or [x['case_id'] for x in scored]!=expected:raise RuntimeError('score_incomplete')
        record('scoring_storage',{'growth_bytes':score_free-free(),'free_bytes':free()})
        if score_free-free()>PHASE_BOUND or free()<RESERVE:raise RuntimeError('scoring_storage_bound')
        result.update(status='COMPLETE' if all(x['outcome'] in ('PASS','FAIL') for x in scored) else 'BLOCKED',phase='complete',cases=len(scored),counts={k:sum(x['outcome']==k for x in scored) for k in ('PASS','FAIL','BLOCKED')})
    except BaseException as exc:
        result.update(status='BLOCKED',error=type(exc).__name__+': '+str(exc))
    finally:
        sample.stop.set()
        if sample.thread.ident:sample.thread.join(timeout=6)
        try:
            stop_score_containers();check_restored();require_empty(Admission(checked_client(),record).request(),'public');result['restoration_verified']=True
        except Exception as exc:result.update(status='BLOCKED',restoration_verified=False,restoration_error=str(exc))
        logs=runtime.run(['docker','logs','--since',log_since,CANDIDATE_ID]);(out/'candidate.stdout.log').write_bytes(logs.stdout);(out/'candidate.stderr.log').write_bytes(logs.stderr)
        record('final_resources',{'initial_free_bytes':initial_free,'free_bytes':free(),'growth_bytes':initial_free-free(),'memory':memory_available()})
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0 if result['status']=='COMPLETE' and result['restoration_verified'] else 1

if __name__=='__main__':raise SystemExit(main())
