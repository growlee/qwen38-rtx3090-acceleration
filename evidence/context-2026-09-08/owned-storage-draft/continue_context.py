"""One exact continuation window; reuse r1's frozen source and host controls.

Not a scheduler or retry loop. Requires a final restored r1, bound source
evidence, and an independently reviewed manifest before any server mutation.
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

from context_storage import OwnedStorage, StorageError, WRITE_RESERVE, install_controller

ROOT = Path(__file__).resolve().parent
R1 = Path('/srv/ai-server/benchmarks/context-20260908-r1')
R1_SHA = 'c71c26c5cf3124347a021489d60148b102a5ad2c35ee4b1314e37ef8eafa36e6'
R1_RUN_SHA = '7cb2f97465a69992babc9887780477ab98d3eb548500e33052cba7eb70248816'
PREFIX = 'qwen-context-20260908-r2'
FILES = {'continue_context.py', 'quality_context_remaining.py',
         'quality_context_continue_client.py', 'grade_context_continuation.py',
         'continuation-execution.md', 'selection.json', 'context_storage.py'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024**2):
            h.update(data)
    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def storage_check(storage):
    return storage.check(WRITE_RESERVE)


def verify_selected_payloads(w, r1, common, selected):
    evidence = {}
    for image, identity in {(m['image'], m['image_id']) for m in selected} | {(w.SYV_IMAGE, w.SYV_IMAGE_ID)}:
        w.inspect_image(common, image, identity)
    for mode in selected:
        model = str(mode['model'])
        if mode['kind'] == 'gguf':
            require(w.GGUF.stat().st_size == w.GGUF_SIZE and sha(w.GGUF) == w.GGUF.parent.name,
                    'continuation_gguf_payload_changed')
            evidence[model] = {'size': w.GGUF_SIZE, 'sha256': w.GGUF.parent.name}
        elif model not in evidence:
            evidence[model] = r1.verify_payload(w, mode)
    if any(m['kind'] == 'dflash2' for m in selected):
        w.load_draft().verify_draft()
        evidence['dflash_drafter_payload_verified'] = True
    return evidence


def time_budget(selection):
    # Each case: up to two tokenizer HTTP calls (GGUF), one 1800s
    # generation, and 120s local/transport allowance. This is an upper
    # safety budget, not an ETA. No second arbitrary 12h cutoff.
    capture = sum(len(mode['case_ids'])*(3*1800+120)
                  + (0 if mode['skip_recorded_warmup'] else 1800)
                  + 1800+600+300 for mode in selection['modes'].values())
    capture += 2700  # public readiness/preflight and controller overhead
    require(capture*256 + 64*1024**2 <= 1024*1024**2, 'sample_disk_budget_insufficient')
    return {'capture_seconds': capture, 'restore_reserve_seconds': 2700,
            'window_seconds': capture+2700, 'unit_runtime_max_seconds': capture+2700+7200}


def configure_public_handoff(w, common, helper, previous):
    """Use existing locks/closed phase; never stop Qwen while admission is public."""
    original_operator = helper.operator
    original_idle, original_restore = common.wait_gpu_idle, w.restore
    expected = previous['production_prestate']
    operator = common.Operator()
    proof = {'gateway_completions': 0, 'public_marker_outcome_unknown': False,
             'start_outcome_unknown': False}

    def identity():
        current = w.production_identity(common)
        require(all(current[key] == expected[key] for key in ('id', 'image', 'model')),
                'original_production_identity_changed')
        return current

    def preflight(c, h):
        h.empty('public')
        identity()
        status = h.exchange({'verb': 'status'})
        require(status['capacity_mode'] == 'serialized-99' and not status['global_breaker_open']
                and set(status['profiles']) == w.PROFILES
                and all(not value['breaker_open'] and
                        value['state'] in (('ready', 'stopped') if name == 'qwen' else ('stopped',))
                        for name, value in status['profiles'].items())
                and set(status['active_profiles']) <= {'qwen'}
                and status['active_profile'] in (None, 'qwen'), 'public_runtime_not_quiescent')
        if common.qwen_state(operator.qwen_status()) == 'stopped':
            c.wait_gpu_idle(90)
            try:
                operator.qwen('start')
            except Exception:
                proof['start_outcome_unknown'] = True
                save(ROOT/'public-preflight-failure.json', proof)
                raise
        require(common.qwen_state(operator.qwen_status()) == 'ready', 'public_qwen_not_ready')
        # Temporarily use the same non-thinking authenticated marker as restore.
        original = c.RESTORE_SCRIPT
        needle = b"'temperature':0,'max_tokens':64,'stream':False}"
        require(original.count(needle) == 1 and b'chat_template_kwargs' not in original,
                'public_marker_contract_changed')
        c.RESTORE_SCRIPT = original.replace(needle,
            b"'temperature':0,'max_tokens':64,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}")
        try:
            c.restore_completion(0)
        except Exception:
            proof['public_marker_outcome_unknown'] = True
            save(ROOT/'public-preflight-failure.json', proof)
            raise
        else:
            proof['gateway_completions'] += 1
        finally:
            c.RESTORE_SCRIPT = original
        h.empty('public')
        require(common.qwen_state(operator.qwen_status()) == 'ready', 'public_qwen_readiness_lost')
        current = identity()
        require(current['running'] is True, 'public_qwen_container_not_running')
        proof.update(qwen_ready=True, container=current,
                     verified_epoch=time.time())
        save(ROOT/'public-preflight.json', proof)
        return proof, hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest(), current

    def advance(*args, **kwargs):
        closing = args == ('advance-phase',) and kwargs == {'phase': 'closed'}
        try:
            result = original_operator(*args, **kwargs)
        except Exception:
            # Resolve the exact observed outcome, never repeat advance-phase.
            if not closing or original_operator().get('phase') != 'closed':
                raise
            result = {'phase': 'closed', 'transport_error_resolved_by_status': True}
        if closing:
            helper.empty('closed')
            require(proof.get('qwen_ready') is True and proof['gateway_completions'] == 1,
                    'public_proof_missing_before_stop')
            identity()
            if common.qwen_state(operator.qwen_status()) == 'ready':
                try:
                    operator.qwen('stop')
                except Exception:
                    # A stopped terminal state resolves a lost response. Any
                    # other state goes to the controller's restoration path.
                    if common.qwen_state(operator.qwen_status()) != 'stopped':
                        raise
            require(common.qwen_state(operator.qwen_status()) == 'stopped', 'qwen_stop_not_proven')
            common.wait_gpu_idle(90)
            runtime, digest = helper.runtime(w.PROFILES)
            save(ROOT/'closed-idle-preflight.json', {'runtime': runtime, 'sha256': digest})
        return result

    def production_only_or_idle(timeout):
        current = identity()
        if not current['running']:
            return original_idle(timeout)
        # The inherited finally must not mistake the original ready Qwen for
        # a surviving test candidate. Prove every GPU PID belongs to that exact
        # production container; the controller already checks owned candidates.
        phase = original_operator().get('phase')
        require(phase in ('public', 'closed') and common.qwen_state(operator.qwen_status()) == 'ready',
                'production_gpu_state_unresolved')
        helper.empty(phase)
        top = common.run(['docker', 'top', current['id'], '-eo', 'pid'], timeout=30).stdout.decode().splitlines()
        pids = {int(line.strip()) for line in top[1:]}
        active = common.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            timeout=30).stdout.decode().splitlines()
        require(bool(active) and {int(line.strip()) for line in active} <= pids,
                'nonproduction_gpu_process_present')

    def restore(c, h, restore_operator, production_before, close_attempted):
        phase = original_operator().get('phase')
        if phase == 'public':
            h.empty('public')
            current = identity()
            require(current['running'] is True and common.qwen_state(operator.qwen_status()) == 'ready',
                    'public_preservation_not_ready')
            # No closure/stop happened. Preserve the still-live service, and
            # label the one preflight marker honestly instead of inventing two.
            return {'qwen_ready': True, 'container': current,
                    'gateway_completions': proof['gateway_completions'],
                    'public_marker_outcome_unknown': proof['public_marker_outcome_unknown'],
                    'start_outcome_unknown': proof['start_outcome_unknown'],
                    'admission_public': True, 'public_preserved_without_cutover': True}
        require(phase == 'closed', 'handoff_restore_phase_unresolved')
        class RestoreOperator:
            def qwen_status(self):
                return restore_operator.qwen_status()

            def qwen(self, verb):
                require(verb == 'start', 'unexpected_restore_verb')
                current = identity()
                state = common.qwen_state(self.qwen_status())
                if state == 'ready' and current['running'] is True:
                    return  # Lost stop did not stop Qwen; never blindly restart.
                require(state == 'stopped' and current['running'] is False,
                        'handoff_restore_state_unresolved')
                return restore_operator.qwen('start')
        return original_restore(c, h, RestoreOperator(), production_before, True)

    w.verify_preflight = preflight
    w.restore = restore
    helper.operator = advance
    common.wait_gpu_idle = production_only_or_idle


def check_bundle(expected_sha):
    require(sha(ROOT/'manifest.json') == expected_sha, 'continuation_manifest_changed')
    manifest = json.loads((ROOT/'manifest.json').read_bytes())
    require(set(manifest['files']) == FILES, 'continuation_file_set_changed')
    for name, digest in manifest['files'].items():
        require(sha(ROOT/name) == digest, 'continuation_source_changed:'+name)
    require(sha(R1/'run_context.py') == R1_RUN_SHA, 'r1_controller_changed')
    r1 = load('retained_context_r1', R1/'run_context.py')
    r1.check_bundle(R1_SHA)
    return manifest, r1


def check_source(manifest):
    """Bind immutable source settings, fixture, capture and once-only scores."""
    from quality_context_remaining import EXPECTED, select_remaining
    sources = manifest['source_evidence']
    required = {'execution-settings.json', 'frozen/frozen.json',
                'results/context-window-report.json', 'results/grading-summary.json'}
    required |= {'results/scores-'+mode+'.jsonl' for mode in EXPECTED}
    allowed = required | {'results/'+mode+'.jsonl' for mode in EXPECTED}
    require(required <= set(sources) <= allowed, 'source_evidence_set_invalid')
    for name, digest in sources.items():
        require(sha(R1/name) == digest, 'source_evidence_changed:'+name)
    settings = json.loads((R1/'execution-settings.json').read_bytes())
    suite = json.loads((R1/'frozen/frozen.json').read_bytes())
    require(settings['fixture_sha256'] == sources['frozen/frozen.json'], 'source_fixture_mismatch')
    window = json.loads((R1/'results/context-window-report.json').read_bytes())
    grading = json.loads((R1/'results/grading-summary.json').read_bytes())
    require(grading['coordinates'] == 612 and not grading['sandbox_fenced'], 'source_grading_incomplete')
    captures = {}
    for mode in EXPECTED:
        path = R1/'results'/(mode+'.jsonl')
        name = 'results/'+mode+'.jsonl'
        require(path.exists() == (name in sources), 'source_capture_files_changed')
        if path.exists():
            captures[mode] = path.read_bytes()
    partition = select_remaining(settings, suite, window, captures)
    return settings, suite, window, partition


def validate_selection(selection, settings, suite, window, partition):
    """Never let selection JSON authorize a replay of a completed answer."""
    modes = {m['id']: m for m in settings['modes']}
    require(selection['source_manifest_sha256'] == R1_SHA, 'selection_source_manifest_mismatch')
    invalidated = selection.get('invalidated_dispatches', [])
    retry_keys = {(x['mode_id'], x['case_id']) for x in invalidated}
    require(len(retry_keys) == len(invalidated), 'duplicate_invalidation')
    for item in invalidated:
        mode, case_id = item['mode_id'], item['case_id']
        require(mode in modes and case_id in partition['modes'][mode]['unresolved_dispatch_case_ids'],
                'invalidation_not_an_unresolved_dispatch')
        entry = window['modes'][mode]
        require(item['reason'] == 'deadline_interruption_no_complete_capture'
                and window['error_code'] == 'window_restore_reserve_reached'
                and entry.get('reason') == 'window_restore_reserve_reached',
                'invalidation_not_proven_deadline_interruption')
        require(bool(item.get('evidence_note')), 'invalidation_evidence_note_missing')
        for role in ('client', 'candidate'):
            cleanup = entry.get('cleanup', {}).get(role, {})
            require(cleanup.get('retained') is True
                    and re.fullmatch('[0-9a-f]{64}', cleanup.get('id', '')) is not None,
                    'interrupted_process_cleanup_unproven')
    require(len(retry_keys) == partition['unresolved_dispatches'], 'unresolved_dispatch_requires_classification')
    expected_modes = []
    for mode_id, source in modes.items():
        part = partition['modes'][mode_id]
        remaining = set(part['untouched_case_ids']) | {
            case_id for m, case_id in retry_keys if m == mode_id}
        if not remaining:
            require(mode_id not in selection['modes'], 'selection_replays_completed_profile')
            continue
        expected_modes.append(mode_id)
        target = selection['modes'][mode_id]
        ids = [c['case_id'] for c in suite['cases'] if c['case_id'] in remaining]
        require(target['case_ids'] == ids, 'selection_not_exact_remaining_case_set')
        expected_backend = 'llama' if source['kind'] == 'gguf' else 'vllm'
        require(target['mode_id'] == mode_id and target['max_len'] == source['max_len']
                and target['backend'] == expected_backend
                and target['fixture_sha256'] == settings['fixture_sha256']
                and target['skip_recorded_warmup'] is part['warmup_recorded'],
                'selection_profile_contract_changed')
    require(set(selection['modes']) == set(expected_modes) and bool(expected_modes),
            'selection_mode_order_or_empty')
    return expected_modes


def configure_client(w, r1, selection, selection_sha, storage):
    def launch(c, mode_id, candidate_id, output, deadline, free_before, ownership):
        name = w.client_name(mode_id)
        argv = ['docker', 'create', '--pull', 'never', '--name', name, '--restart', 'no', '--log-driver', 'none',
                '--network', 'container:'+candidate_id, '--read-only', '--user', '0:0',
                '--security-opt', 'no-new-privileges', '--cap-drop', 'ALL', '--pids-limit', '128',
                '--memory', '1g', '--memory-swap', '1g', '--tmpfs', '/tmp:rw,noexec,nosuid,size=32m',
                '--tmpfs', '/cache:rw,noexec,nosuid,size=32m', '--tmpfs', '/app/models:rw,noexec,nosuid,size=16m',
                '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'PYTHONPATH=/context:/runner',
                '--mount', f'type=bind,src={w.SOURCES},dst=/runner,readonly',
                '--mount', f'type=bind,src={R1},dst=/context,readonly',
                '--mount', f'type=bind,src={ROOT},dst=/continuation,readonly',
                '--entrypoint', '/app/venv/bin/python', w.SYV_IMAGE, '-B',
                '/continuation/quality_context_continue_client.py', '--fixture', '/context/frozen/frozen.json',
                '--sha', selection['modes'][mode_id]['fixture_sha256'],
                '--selection', '/continuation/selection.json', '--selection-sha', selection_sha, '--mode', mode_id]
        storage_check(storage)
        identity = w.created_id(c.run(argv, timeout=60), 'client_create_identity_unknown')
        ownership['client_id'] = identity
        storage.register_container(identity, name)
        storage_check(storage)
        w.verify_client_isolation(c, identity, candidate_id, running=False)
        samples = []
        with output.open('xb', buffering=0) as sink, (output.parent/(mode_id+'.stderr')).open('xb') as err:
            process = subprocess.Popen(['docker', 'start', '-a', identity], stdout=sink, stderr=err)
            try:
                with w.LoadGuard(c, candidate_id, free_before):
                    while process.poll() is None:
                        if time.monotonic() >= deadline:
                            raise w.WindowError('window_restore_reserve_reached')
                        r1.no_freetoken()
                        state = w.exact_container(c, candidate_id)['State']
                        if not state['Running'] or state['OOMKilled']:
                            raise w.WindowError('candidate_failed_during_client')
                        if output.stat().st_size > 32*1024**2 or err.tell() > 1024**2:
                            raise w.WindowError('capture_output_bound_exceeded')
                        measured = storage_check(storage)
                        sample = [time.time(), *c.gpu_rows(), w.mem_available_bytes(),
                                  measured['owned_allocated_bytes'], measured['root_free_bytes']]
                        require(len(json.dumps(sample).encode())+2 <= 256, 'resource_row_bound_exceeded')
                        samples.append(sample)
                        time.sleep(1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(5)
                save(output.parent/(mode_id+'-resources.json'), samples)
        w.verify_client_isolation(c, identity, candidate_id, running=False)
        if process.returncode != 0:
            raise w.WindowError('context_client_failed', {'returncode': process.returncode})
        cases = [r for line in output.read_bytes().splitlines()
                 if (r := json.loads(line)).get('kind') == 'case']
        if [r['case_id'] for r in cases] != selection['modes'][mode_id]['case_ids']:
            raise w.WindowError('context_case_set_mismatch')
        return {'id': identity, 'rows': len(cases), 'sha256': sha(output), 'jsonl': str(output),
                'cold_continuation': selection['modes'][mode_id]['skip_recorded_warmup']}
    w.launch_client = launch


def main():
    require(os.geteuid() == 0, 'root_required')
    manifest, r1 = check_bundle(sys.argv[1])
    status = subprocess.run(['systemctl', 'show', 'qwen-context-20260908-r1.service',
                             '-p', 'ActiveState', '-p', 'MainPID'], check=True,
                            capture_output=True, text=True, timeout=30).stdout.splitlines()
    status = dict(line.split('=', 1) for line in status if '=' in line)
    require(status.get('ActiveState') in ('inactive', 'failed') and status.get('MainPID') == '0',
            'source_window_still_running')
    settings, suite, previous, partition = check_source(manifest)
    selection = json.loads((ROOT/'selection.json').read_bytes())
    selected_ids = validate_selection(selection, settings, suite, previous, partition)
    w = r1.load('continuation_window', r1.OLD/'run_quality_window_remaining.py')
    w.verify_bundle(r1.OLD_SHA)
    common, syv, helper = w.load_runtime_modules()
    all_modes = r1.modes(w)
    selected = [m for m in all_modes if m['id'] in selected_ids]
    # Match actual launch settings, not only the selection's nominal context.
    for mode in all_modes:
        original = next(m for m in settings['modes'] if m['id'] == mode['id'])
        require({**mode, 'model': str(mode['model']), 'lengths': list(mode['lengths'])} == original,
                'retained_mode_contract_changed')
    r1.no_freetoken()
    helper.empty('public')
    production = w.production_identity(common)
    require(all(production[key] == previous['production_prestate'][key] for key in ('id', 'image', 'model')),
            'original_production_identity_changed')
    # Fresh exact-ID stop evidence is mandatory after the prior window.
    for entry in previous['modes'].values():
        for cleanup in entry.get('cleanup', {}).values():
            if isinstance(cleanup, dict) and cleanup.get('id'):
                current = w.exact_container(common, cleanup['id'])
                require(current is not None and current['State']['Running'] is False,
                        'prior_owned_process_not_stopped')
    # No second fixture copy, CPU tokenizer pass, weight preparation or download.
    baseline = w.available_bytes()
    require(baseline >= w.RESERVE_BYTES+r1.GROWTH, 'continuation_storage_budget_insufficient')
    storage = OwnedStorage(ROOT)
    storage_check(storage)
    payloads = verify_selected_payloads(w, r1, common, selected)
    storage_check(storage)
    save(ROOT/'payload-preflight.json', payloads)
    r1.PREFIX = PREFIX
    r1.configure(w, common, syv, helper, selected, settings['fixture_sha256'])
    budget = time_budget(selection)
    w.WINDOW_SECONDS = budget['window_seconds']
    w.MODE_SECONDS = budget['capture_seconds']
    configure_client(w, r1, selection, manifest['files']['selection.json'], storage)
    configure_public_handoff(w, common, helper, previous)
    install_controller(w, storage)
    save(ROOT/'execution-settings.json', {'modes': [{**m, 'model': str(m['model'])} for m in selected],
         'fixture_sha256': settings['fixture_sha256'], 'source_manifest_sha256': R1_SHA,
         'source_partition': partition, 'selection_sha256': manifest['files']['selection.json'],
         **budget,
         'case_timeout_seconds': 1800, 'coordinates': sum(len(s['case_ids']) for s in selection['modes'].values()),
         'storage_available_before_bytes': baseline, 'max_growth_bytes': r1.GROWTH,
         'storage_budget_scope': 'run_root_and_receipt_bound_container_allocations',
         'storage_write_reserve_bytes': WRITE_RESERVE,
         'approval': 'user extended time to complete the same finite 612-coordinate matrix',
         'started_epoch': time.time()})
    result = w.main(['--output-dir', str(ROOT/'results'), '--bundle-manifest-sha256', r1.OLD_SHA])
    report = json.loads((ROOT/'results/window-report.json').read_bytes())
    post_restore_free = w.available_bytes()
    storage_error = None
    try:
        storage_check(storage)
    except StorageError as exc:
        storage_error = exc.code
    save(ROOT/'results/context-window-report.json', {**report,
         'suite_id': 'qwen38-context-12x7-v1', 'source_commit': None,
         'source_manifest_sha256': sys.argv[1], 'fixture_sha256': settings['fixture_sha256'],
         'selection_sha256': manifest['files']['selection.json'],
         'legacy_controller_report_sha256': sha(ROOT/'results/window-report.json'),
         'storage_available_post_restore_bytes': post_restore_free,
         'storage_budget_scope': 'run_root_and_receipt_bound_container_allocations',
         'storage_owned_containers': dict(storage.containers),
         'storage_post_restore': storage.last, 'storage_error': storage_error,
         'post_restore_storage_bound_ok': storage_error is None})
    storage_check(storage)
    # Keep exact creation ownership in memory across capture and grading;
    # standalone grading must not silently forget retained container layers.
    grader = load('context_continuation_grader', ROOT/'grade_context_continuation.py')
    grader.main(storage=storage)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
