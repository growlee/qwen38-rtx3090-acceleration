"""One 612-coordinate window, adapting the retained reviewed controller.

No downloads, preparation of weights, removal, retries, or persistent promotion.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/srv/ai-server/benchmarks/quality-remaining-20260908')
OLD_SHA = '0a6ba209c1a9320743c9e455af3934afe47f9d856e7c01e35ff6665cb2ab35e3'
PREFIX = 'qwen-context-20260908'
GROWTH = 4 * 1024**3
LENGTHS = (8192, 32768, 50000, 96000, 128000, 160000, 200000)
RECON = Path('/srv/ai-server/staging/experiments/qwen38-syv-uncensored/qwen-adaptive-reconstruction-20260908-candidate/model')
RECEIPTS = {
    'autoround': 'a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a',
    'int8': '33ef6b4ab8d6047892dc1c123d87bcbacd22c382e85e59426a191e437ee89521',
    'adaptive': '0031745a1a615256cc37b2a345f2dca2d1744787927d81f04a5752e498e822bb',
    'reconstructed': '6b098f064a894e3efccf1adba96aabcaee88025e52ab01bde13857f7cb72011c',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024**2): h.update(b)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True)
        f.flush(); os.fsync(f.fileno())


def check_bundle(expected):
    assert sha(ROOT/'manifest.json') == expected, 'context_manifest_changed'
    for name, digest in json.loads((ROOT/'manifest.json').read_bytes())['files'].items():
        assert Path(name).name == name and sha(ROOT/name) == digest, 'context_source_changed'
    assert sha(OLD/'bundle-manifest-remaining.json') == OLD_SHA, 'old_manifest_changed'
    manifest = json.loads((OLD/'bundle-manifest-remaining.json').read_bytes())
    for name, digest in {**{'sources/'+n:h for n,h in manifest['files'].items()}, **manifest['other_files']}.items():
        assert sha(OLD/name) == digest, 'old_source_changed:'+name


def modes(w):
    out = []
    for key, model in [('autoround', w.BASE_MODEL), ('int8', w.MODES[0]['model']),
                       ('adaptive', w.ADAPTIVE_MODEL), ('reconstructed', RECON)]:
        for ctx in ('fast', 'huge'):
            out.append(dict(id=f'{key}-mtp-{ctx}', kind='syv', model=model, ctx=ctx,
                            max_len=57344 if ctx=='fast' else 204800,
                            lengths=LENGTHS[:3] if ctx=='fast' else LENGTHS,
                            image=w.SYV_IMAGE, image_id=w.SYV_IMAGE_ID, receipt=RECEIPTS[key]))
    for ctx in ('fast', 'huge'):
        out.append(dict(id=f'dflash2-{ctx}-n7', kind='dflash2', model=w.BASE_MODEL, ctx=ctx,
                        max_len=65536 if ctx=='fast' else 245760,
                        lengths=LENGTHS[:3] if ctx=='fast' else LENGTHS,
                        image=w.SYV_IMAGE, image_id=w.SYV_IMAGE_ID, receipt=RECEIPTS['autoround']))
    out.append(dict(id='original-gguf', kind='gguf', model=w.GGUF, ctx='fast', max_len=16384,
                    lengths=LENGTHS[:1], image=w.GGUF_IMAGE, image_id=w.GGUF_IMAGE_ID))
    assert sum(len(m['lengths'])*12 for m in out) == 612
    return out


def verify_payload(w, mode):
    model = mode['model']
    receipt = model.parent/'COMPLETE.receipt'
    result = w.require_ro_tree(model, receipt, mode['receipt'])
    raw = receipt.read_text()
    if raw.startswith('{'):
        files = json.loads(raw)['files']
        entries = [(f['name'], f['size'], f['sha256']) for f in files]
    else:
        entries = [(m[3], int(m[2]), m[1]) for line in raw.splitlines()
                   if (m := re.fullmatch(r'([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9._-]+)', line))]
    assert entries, 'receipt_payload_empty'
    assert {name for name,_,_ in entries} == {p.name for p in model.iterdir()}, 'receipt_file_set_mismatch'
    for name, size, digest in entries:
        assert Path(name).name == name and (model/name).stat().st_size == size
        assert sha(model/name) == digest, 'payload_changed:'+name
    result['payload_sha256_verified'] = True
    result['physical_bytes'] = sum(f.stat().st_blocks*512 for f in model.iterdir())
    return result


def no_freetoken():
    conflicts = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try: command = (proc/'cmdline').read_bytes().lower()
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(x in command for x in (b'/opt/homelab/freetoken', b'/var/lib/freetoken', b'freetoken-prefetch')):
            conflicts.append(int(proc.name))
    assert not conflicts, 'competing_freetoken_processes:'+str(conflicts)


def prepare_fixtures(w, common, frozen):
    name = PREFIX+'-tokenizer'
    args = ['docker','create','--pull','never','--name',name,'--restart','no','--network','none',
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--log-driver','none',
            '--memory','2g','--memory-swap','2g','--cpus','2','--pids-limit','128',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=32m','--tmpfs','/cache:rw,noexec,nosuid,size=32m',
            '--tmpfs','/app/models:rw,noexec,nosuid,size=16m','--env','NVIDIA_VISIBLE_DEVICES=void',
            '--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','PYTHONPATH=/runner',
            '--mount',f'type=bind,src={OLD}/sources,dst=/runner,readonly',
            '--mount',f'type=bind,src={ROOT},dst=/context,readonly',
            '--mount',f'type=bind,src={frozen},dst=/output',
            '--mount',f'type=bind,src={w.BASE_MODEL},dst=/model,readonly',
            '--entrypoint','/app/venv/bin/python',w.SYV_IMAGE,'-B','/context/quality_context_client.py',
            '--freeze-model','/model','--output','/output/frozen.json']
    identity = w.created_id(common.run(args, timeout=60), 'tokenizer_create_unknown')
    try:
        result = common.run(['docker','start','-a',identity], timeout=1800, check=False)
        state = w.exact_container(common, identity)['State']
        save(ROOT/'tokenizer-run.json', {'exit_code':result.returncode,
             'stdout_tail':result.stdout.decode('utf-8','replace')[-8192:],
             'stderr_tail':result.stderr.decode('utf-8','replace')[-8192:]})
        assert not state['Running'] and not state['OOMKilled'] and result.returncode == 0, 'fixture_freeze_failed'
        return {'container_id':identity,'sha256':sha(frozen/'frozen.json'),
                'bytes':(frozen/'frozen.json').stat().st_size}
    finally:
        w.stop_owned_once(common,name,identity)


def configure(w, common, syv, helper, selected, frozen_sha):
    original_args, original_evidence = w.syv_args, w.profile_evidence
    w.MODES = selected
    w.BLOCKED = {}
    w.MAX_GROWTH_BYTES = GROWTH
    w.WINDOW_SECONDS = 12*3600
    w.RESTORE_RESERVE_SECONDS = 2700
    w.MODE_SECONDS = 12*3600-2700
    w.LOAD_SECONDS = 600
    w.CLIENT_TIMEOUT_SECONDS = 1800
    w.NONFATAL_MODE_CODES = set()  # Unknown outcomes and infrastructure errors fence the matrix.
    w.candidate_name = lambda mode: PREFIX+'-'+mode
    w.load_runtime_modules = lambda: (common,syv,helper)
    # All payload preparation was read-only and completed before public closure.
    def prepared(c,m):
        no_freetoken()
        helper.empty('closed')
        return {'existing_receipt_sha256':m.get('receipt'), 'no_model_writes':True}
    w.prepare_candidate = prepared
    def args(module, mode, name):
        argv = original_args(module, mode, name)
        if mode['kind'] != 'dflash2' and mode['ctx']=='huge':
            for old,new in [('CTX=fast','CTX=huge'),('MAX_LEN=57344','MAX_LEN=204800'),
                            ('DRAFT_TOKENS=4','DRAFT_TOKENS=3')]:
                assert argv.count(old)==1, 'profile_env_ambiguous'
                argv[argv.index(old)] = new
        return argv
    w.syv_args = args
    def evidence(c,s,mode,name,deadline):
        if mode['kind']=='dflash2':
            return original_evidence(c,s,mode,name,deadline)
        result = s.startup_evidence(deadline)
        top = c.run(['docker','top',name,'-eo','pid,args'],timeout=30).stdout.decode()
        huge = mode['ctx']=='huge'
        expected = [f"--max-model-len {mode['max_len']}",'"method":"mtp"',
                    '"num_speculative_tokens":'+('3' if huge else '4'),
                    '--gpu-memory-utilization 0.93','--max-num-seqs 8',
                    '"draft_sample_method":"probabilistic"']
        expected += (['--kv-cache-dtype kvarn_k4v2_g128','--block-size 128'] if huge
                     else ['--kv-cache-dtype bfloat16','--attention-backend FLASH_ATTN'])
        result['process_contract'] = {arg:arg in top for arg in expected}
        result['process_contract']['prefix_cache_off'] = '--enable-prefix-caching' not in top
        if not result['mtp_runtime_proven'] or not all(result['process_contract'].values()):
            raise w.WindowError('mtp_runtime_contract_mismatch',result)
        return result
    w.profile_evidence = evidence

    def launch(c,mode_id,candidate_id,output,deadline,free_before,ownership):
        mode = next(m for m in selected if m['id']==mode_id)
        name = w.client_name(mode_id)
        argv = ['docker','create','--pull','never','--name',name,'--restart','no','--log-driver','none',
                '--network','container:'+candidate_id,'--read-only','--user','0:0',
                '--security-opt','no-new-privileges','--cap-drop','ALL','--pids-limit','128',
                '--memory','1g','--memory-swap','1g','--tmpfs','/tmp:rw,noexec,nosuid,size=32m',
                '--tmpfs','/cache:rw,noexec,nosuid,size=32m','--tmpfs','/app/models:rw,noexec,nosuid,size=16m',
                '--env','NVIDIA_VISIBLE_DEVICES=void','--env','PYTHONPATH=/runner',
                '--mount',f'type=bind,src={w.SOURCES},dst=/runner,readonly',
                '--mount',f'type=bind,src={ROOT},dst=/context,readonly',
                '--entrypoint','/app/venv/bin/python',w.SYV_IMAGE,'-B','/context/quality_context_client.py',
                '--fixture','/context/frozen/frozen.json','--sha',frozen_sha,'--mode',mode_id,
                '--max-len',str(mode['max_len']),'--lengths',','.join(map(str,mode['lengths'])),
                '--backend','llama' if mode['kind']=='gguf' else 'vllm']
        client_id = w.created_id(c.run(argv, timeout=60), 'client_create_identity_unknown')
        ownership['client_id'] = client_id
        w.verify_client_isolation(c, client_id, candidate_id, running=False)
        samples = []
        with output.open('xb', buffering=0) as sink, (output.parent/(mode_id+'.stderr')).open('xb') as err:
            p = subprocess.Popen(['docker','start','-a',client_id],stdout=sink,stderr=err)
            try:
                with w.LoadGuard(c,candidate_id,free_before):
                    while p.poll() is None:
                        if time.monotonic() >= deadline: raise w.WindowError('window_restore_reserve_reached')
                        no_freetoken()
                        state = w.exact_container(c,candidate_id)['State']
                        if not state['Running'] or state['OOMKilled']: raise w.WindowError('candidate_failed_during_client')
                        if output.stat().st_size > 32*1024**2 or err.tell() > 1024**2:
                            raise w.WindowError('capture_output_bound_exceeded')
                        samples.append([time.time(), *c.gpu_rows(), w.mem_available_bytes()])
                        time.sleep(1)
            finally:
                if p.poll() is None:
                    p.terminate()
                    try: p.wait(5)
                    except subprocess.TimeoutExpired: p.kill(); p.wait(5)
                save(output.parent/(mode_id+'-resources.json'), samples)
        w.verify_client_isolation(c,client_id,candidate_id,running=False)
        if p.returncode != 0: raise w.WindowError('context_client_failed',{'returncode':p.returncode})
        rows = [json.loads(line) for line in output.read_bytes().splitlines()]
        cases = [r for r in rows if r.get('kind')=='case']
        frozen = json.loads((ROOT/'frozen/frozen.json').read_bytes())
        expected = [x['case_id'] for x in frozen['cases'] if x['target_tokens'] in mode['lengths']]
        if [r['case_id'] for r in cases] != expected: raise w.WindowError('context_case_set_mismatch')
        return {'id':client_id,'rows':len(cases),'sha256':sha(output),'jsonl':str(output)}
    w.launch_client = launch


def main():
    preparation_started = time.monotonic()
    check_bundle(sys.argv[1])
    assert os.geteuid()==0
    w = load('context_window', OLD/'run_quality_window_remaining.py')
    w.verify_bundle(OLD_SHA)
    common,syv,helper = w.load_runtime_modules()
    selected = modes(w)
    no_freetoken()
    w.verify_preflight(common,helper)
    baseline = w.available_bytes()
    assert baseline >= w.RESERVE_BYTES+GROWTH, 'storage_budget_insufficient'
    for image,identity in [(w.SYV_IMAGE,w.SYV_IMAGE_ID),(w.GGUF_IMAGE,w.GGUF_IMAGE_ID)]:
        w.inspect_image(common,image,identity)
    seen = {}
    for mode in selected:
        if mode['kind'] != 'gguf' and str(mode['model']) not in seen:
            seen[str(mode['model'])] = verify_payload(w,mode)
    assert w.GGUF.stat().st_size == w.GGUF_SIZE and sha(w.GGUF)==w.GGUF.parent.name
    draft = w.load_draft()
    draft.verify_draft()  # exact retained payload only; no preparation/download path
    for filename in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        assert len({sha(Path(model)/filename) for model in seen})==1, 'tokenizer_mismatch'
    save(ROOT/'preflight.json', {'available_bytes':baseline,'models':seen,'growth_bound_bytes':GROWTH,
                               'no_competing_freetoken':True,'verified_epoch':time.time()})
    frozen = ROOT/'frozen'; frozen.mkdir(mode=0o700)
    receipt = prepare_fixtures(w,common,frozen)
    save(ROOT/'fixture-receipt.json',receipt)
    sys.path.insert(0,str(OLD/'sources'))
    import quality_sandbox
    quality_sandbox.CONTAINER_PREFIX = PREFIX+'-reference'
    cases = json.loads((frozen/'frozen.json').read_bytes())['cases']
    references = []
    for case in cases[:12]:
        if case['check']['type']=='python_function_cases':
            result = quality_sandbox.evaluate_code(case,case['reference_source'],mode_id='reference')
            references.append(result)
            assert result['outcome']=='PASS', 'reference_sandbox_failed'
    save(ROOT/'reference-results.json',references)
    no_freetoken()
    assert baseline-w.available_bytes() < 512*1024**2, 'preparation_growth_exceeded'
    assert time.monotonic()-preparation_started < 3600, 'preparation_time_budget_exceeded'
    configure(w,common,syv,helper,selected,receipt['sha256'])
    # Immutable settings/cases/timeout receipts all exist before any inference.
    save(ROOT/'execution-settings.json',{'modes':[{**m,'model':str(m['model'])} for m in selected],
        'case_timeout_seconds':1800,'window_seconds':43200,'restore_reserve_seconds':2700,
        'max_warmups':11,'coordinates':612,'fixture_sha256':receipt['sha256'],
        'started_epoch':time.time(),'approval':'user approved up to 12 hours, at least last 15 minutes restoration; conservative reserve extended to 45 minutes within same total'})
    result = w.main(['--output-dir',str(ROOT/'results'),'--bundle-manifest-sha256',OLD_SHA])
    legacy = json.loads((ROOT/'results/window-report.json').read_bytes())
    save(ROOT/'results/context-window-report.json', {**legacy,
         'suite_id':'qwen38-context-12x7-v1','source_commit':None,
         'source_manifest_sha256':sys.argv[1], 'fixture_sha256':receipt['sha256'],
         'legacy_controller_report_sha256':sha(ROOT/'results/window-report.json')})
    # The grading entry point independently requires proven public restoration.
    subprocess.run([sys.executable,'-B',str(ROOT/'grade_context.py'),sys.argv[1]], check=True)
    return result


if __name__=='__main__':
    raise SystemExit(main())
