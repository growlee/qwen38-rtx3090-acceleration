"""Finite matrix callback on the retained Server A controller and restoration.

Request files are exact per-coordinate transport artifacts. There is no job
discovery, queue service, retry, recovery loop, promotion or model preparation.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

ROOT=Path('/srv/ai-server/benchmarks/qwen-multiturn-context-20260910')
R1=Path('/srv/ai-server/benchmarks/context-20260908-r1')
R2=Path('/srv/ai-server/benchmarks/context-20260908-r2')
OLD=Path('/srv/ai-server/benchmarks/quality-remaining-20260908')
BASE=Path('/srv/ai-server/benchmarks/context-20260910-repair-clientfix/repair_context.py')
KV=Path('/srv/ai-server/benchmarks/context-20260910-repair-kv4350/repair_context.py')
PINS={str(R1/'run_context.py'):'7cb2f97465a69992babc9887780477ab98d3eb548500e33052cba7eb70248816',
      str(OLD/'run_quality_window_remaining.py'):'4ad216fc2ff90ea9a8971e10904681fd11396f86977bc3e865ae736cf7dcccc5',
      str(R2/'continue_context.py'):'6d2e89059388a5fef595cfc80f76a070d46f21e3a8865c38054b33691c6a5307',
      str(R2/'context_storage.py'):'2085436ed77ec5a9f64d61568bb612254cb58158027869fc036fa666ff914cfd',
      str(BASE):'7c7866bb0d8a0eeae3eb46019d0a051aef5b0da23971bd80a00fd25e282233c2',
      str(KV):'c6641e164782f511ba64a220bb38e166600fdc8337bd8f6ddf19aefa63975c0b'}
PREFIX='qwen-multiturn-20260910'
WINDOW=737700

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def wire(value):return json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf-8')
def save(path,value):
    raw=wire(value)
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def check_manifest(expected):
    assert sha(ROOT/'manifest.json')==expected,'manifest_changed'
    manifest=json.loads((ROOT/'manifest.json').read_bytes())
    assert manifest['authorized_window_seconds']==WINDOW and len(manifest['coordinates'])==600
    assert len({r['case_id'] for r in manifest['coordinates']})==600
    for name,digest in manifest['remote_files'].items():
        assert Path(name).name==name and sha(ROOT/name)==digest,'new_source_changed:'+name
    for name,digest in PINS.items():assert sha(name)==digest,'retained_source_changed:'+name
    return manifest

def install_capture(w,common,r1,storage,manifest):
    def launch(c,mode_id,candidate,output,deadline,free_before,ownership):
        name=w.client_name(mode_id)
        argv=['docker','create','-i','--pull','never','--name',name,'--restart','no','--log-driver','none',
              '--network','container:'+candidate,'--read-only','--user','0:0','--security-opt','no-new-privileges',
              '--cap-drop','ALL','--pids-limit','128','--memory','1g','--memory-swap','1g',
              '--tmpfs','/tmp:rw,noexec,nosuid,size=32m','--tmpfs','/cache:rw,noexec,nosuid,size=32m',
              '--tmpfs','/app/models:rw,noexec,nosuid,size=16m','--env','NVIDIA_VISIBLE_DEVICES=void',
              '--env','PYTHONPATH=/runner','--mount',f'type=bind,src={w.SOURCES},dst=/runner,readonly',
              '--mount',f'type=bind,src={ROOT}/remote_client.py,dst=/multiturn_client.py,readonly',
              '--entrypoint','/app/venv/bin/python',w.SYV_IMAGE,'-B','/multiturn_client.py']
        storage.check(512*1024**2)
        identity=w.created_id(c.run(argv,timeout=60),'multiturn_client_create_unknown')
        ownership['client_id']=identity;storage.register_container(identity,name)
        w.verify_client_isolation(c,identity,candidate,running=False)
        process=subprocess.Popen(['docker','start','-ai',identity],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        errors=bytearray();error_total=[0]
        def drain():
            while chunk:=process.stderr.read(65536):
                error_total[0]+=len(chunk)
                errors.extend(chunk[:max(0,1024**2-len(errors))])
        thread=threading.Thread(target=drain,daemon=True);thread.start()
        buffer=bytearray();index=[];last_sample=0;resource_bytes=0
        resources=(ROOT/'responses'/(mode_id+'-resources.jsonl')).open('xb')
        def observe():
            nonlocal last_sample,resource_bytes
            now=time.monotonic()
            if now>=deadline:raise w.WindowError('window_restore_reserve_reached')
            if error_total[0]>1024**2:raise w.WindowError('client_stderr_bound')
            if process.poll() is not None:raise w.WindowError('multiturn_client_exited',{'returncode':process.returncode})
            if now-last_sample>=1:
                r1.no_freetoken();measured=storage.check(512*1024**2)
                state=w.exact_container(c,candidate)['State']
                if not state['Running'] or state['OOMKilled']:raise w.WindowError('candidate_failed_during_capture')
                sample=wire([time.time(),now,*c.gpu_rows(),w.mem_available_bytes(),measured['owned_allocated_bytes'],measured['root_free_bytes']])+b'\n'
                if len(sample)>512 or resource_bytes+len(sample)>80*1024**2:raise w.WindowError('resource_capture_bound')
                resources.write(sample);resources.flush();resource_bytes+=len(sample);last_sample=now
        def receive(until):
            while b'\n' not in buffer:
                if time.monotonic()>=until:raise w.WindowError('client_response_deadline')
                # Read a final result before interpreting the client's exit code.
                if select.select([process.stdout],[],[],.25)[0]:
                    chunk=os.read(process.stdout.fileno(),65536)
                    if chunk:buffer.extend(chunk)
                    elif process.poll() is not None:raise w.WindowError('client_result_missing')
                if len(buffer)>2*1024**2:raise w.WindowError('client_response_bound')
                if b'\n' not in buffer:observe()
            line,_,rest=buffer.partition(b'\n');buffer[:]=rest
            return json.loads(line)
        def send(envelope):
            raw=wire(envelope)
            if len(raw)>2*1024**2+4096:raise w.WindowError('request_bound')
            process.stdin.write(raw+b'\n');process.stdin.flush()
            return receive(min(deadline,time.monotonic()+965))
        try:
            with w.LoadGuard(c,candidate,free_before):
                warmup={'case_id':mode_id+'--warmup','body':{'model':'qwen3.8-27b','messages':[{'role':'user','content':'Reply only: ready'}],
                    'temperature':0,'seed':8000,'max_tokens':32,'stream':True,'stream_options':{'include_usage':True},
                    'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':True}}}
                result=send(warmup);save(ROOT/'responses'/(mode_id+'--warmup.json'),result)
                if result.get('error') or result.get('pre_generation_error') or not result.get('stream_complete') or result.get('finish_reason')!='stop' or result.get('empty_output'):
                    raise w.WindowError('warmup_failed')
                selected=[r for r in manifest['coordinates'] if r['profile']==mode_id]
                assert len(selected)==120
                for coordinate in selected:
                    case_id=coordinate['case_id'];path=ROOT/'requests'/(case_id+'.json')
                    wait_until=min(deadline,time.monotonic()+240)
                    while not path.exists():
                        observe()
                        if time.monotonic()>=wait_until:raise w.WindowError('pc_a_request_deadline')
                        time.sleep(.5)
                    raw=path.read_bytes()
                    if path.is_symlink() or len(raw)>2*1024**2+4096:raise w.WindowError('request_file_contract')
                    envelope=json.loads(raw)
                    if envelope['case_id']!=case_id or envelope['input_target']!=coordinate['input_target']:raise w.WindowError('request_coordinate_contract')
                    body=envelope['body']
                    if body['max_tokens']!=4096 or body['temperature']!=0 or body['seed']!=20260910 or body['chat_template_kwargs']!={'enable_thinking':False,'preserve_thinking':True}:
                        raise w.WindowError('generation_contract')
                    save(ROOT/'responses'/(case_id+'--dispatch.json'),{'case_id':case_id,'envelope_sha256':hashlib.sha256(raw).hexdigest(),'started_epoch':time.time()})
                    result=send(envelope)
                    result.update(profile=mode_id,cold=False,warmup_count=1,candidate_id=candidate,client_id=identity)
                    digest=save(ROOT/'responses'/(case_id+'.json'),result)
                    index.append({'case_id':case_id,'sha256':digest})
                    if result.get('error') or result.get('pre_generation_error') or not result.get('stream_complete'):
                        raise w.WindowError('model_transport_or_capacity_stop',{'case_id':case_id,'reason':result.get('error') or result.get('pre_generation_error')})
                    receipt=ROOT/'grades'/(case_id+'.json');wait_until=min(deadline,time.monotonic()+240)
                    while not receipt.exists():
                        observe()
                        if time.monotonic()>=wait_until:raise w.WindowError('pc_a_grade_deadline')
                        time.sleep(.5)
                    if receipt.is_symlink() or receipt.stat().st_size>4096:raise w.WindowError('pc_a_grade_contract')
                    graded=json.loads(receipt.read_bytes())
                    if graded.get('case_id')!=case_id or graded.get('response_sha256')!=digest or graded.get('status') not in ('PASS','FAIL'):
                        raise w.WindowError('pc_a_grade_or_sandbox_blocked',graded)
            process.stdin.close();process.wait(timeout=15)
            if process.returncode!=0:raise w.WindowError('client_exit_nonzero')
            return {'id':identity,'rows':len(index),'warmups':1}
        finally:
            resources.close()
            if process.poll() is None:
                # Existing run_mode finally owns exact-ID stop, including timeout.
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            thread.join(timeout=2)
            save(output,{'responses':index,'warmups':1})
            with (ROOT/'responses'/(mode_id+'-client.stderr')).open('xb') as sink:sink.write(errors)
    w.launch_client=launch

def main(expected):
    assert os.geteuid()==0
    manifest=check_manifest(expected)
    assert not (ROOT/'results').exists(),'existing_attempt_no_retry'
    assert not any((ROOT/'responses').iterdir()) and not any((ROOT/'grades').iterdir()),'prior_capture_or_grade_exists'
    sys.path.insert(0,str(R2))
    r1=load('qmt_r1',R1/'run_context.py');r1.check_bundle('c71c26c5cf3124347a021489d60148b102a5ad2c35ee4b1314e37ef8eafa36e6')
    w=load('qmt_retained_window',OLD/'run_quality_window_remaining.py');w.verify_bundle(r1.OLD_SHA)
    common,syv,helper=w.load_runtime_modules()
    base=load('qmt_unique_operator',BASE);base.install_unique_operator(common)
    kv=load('qmt_exact_dflash',KV);kv.install_profile(kv,w)
    continuation=load('qmt_public_handoff',R2/'continue_context.py');continuation.ROOT=ROOT
    modes={m['id']:m for m in r1.modes(w)}
    modes['dflash2-huge-n7-kv4350mb-204800']={**modes['dflash2-huge-n7'],'id':'dflash2-huge-n7-kv4350mb-204800','max_len':204800}
    selected=[modes[name] for name in manifest['profiles']]
    r1.PREFIX=PREFIX;r1.configure(w,common,syv,helper,selected,expected)
    base.REPAIRED='dflash2-huge-n7-kv4350mb-204800';base.install_capacity_gate(w,common)
    storage_module=sys.modules['context_storage'];storage=storage_module.OwnedStorage(ROOT)
    continuation.configure_public_handoff(w,common,helper,{'production_prestate':manifest['production']})
    w.WINDOW_SECONDS=WINDOW;w.RESTORE_RESERVE_SECONDS=2700;w.MODE_SECONDS=WINDOW-2700
    w.LOAD_SECONDS=1800;w.CLIENT_TIMEOUT_SECONDS=900
    def verify_models():
        payloads=continuation.verify_selected_payloads(w,r1,common,selected)
        save(ROOT/'payload-preflight.json',payloads);return payloads
    w.verify_models=verify_models
    storage_module.install_controller(w,storage)
    install_capture(w,common,r1,storage,manifest)
    status=w.main(['--output-dir',str(ROOT/'results'),'--bundle-manifest-sha256',r1.OLD_SHA])
    save(ROOT/'storage-after.json',storage.check(512*1024**2))
    return status

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
