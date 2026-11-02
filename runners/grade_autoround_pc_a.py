"""One independent PC-A check of retained AutoRound code; no model requests."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import quality_context_suite
import quality_sandbox

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/context-2026-09-08'
IMAGE = 'ghcr.io/syv-ai/qwen38-27b-rtx3090@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0'
CONFIG_IMAGE = 'sha256:b0516eab4a689391cd73629593d53111777691c5f3c453d5064e77fe55981101'
RAW = {
    'autoround-mtp-fast': '88df9e73ff07f5a2a8270102af14c2ab9579384088233fa7638be49d38e3c3f0',
    'autoround-mtp-huge': 'd987d712cd164bf843252fe820875926e25bcc35a6b68a071e7eced4a60c1bab',
}
SOURCE = {
    'quality_context_suite.py': 'bcfbb9a7978c3f8d5a4f9b4484b314ad8d5e4b141de6765fc5aeb1b88d2d2ea8',
    'quality_sandbox.py': 'f9065cac6b88f6ba987deb91e6210ca91e00ea4a0a9a12a2c45797efeeb2006a',
}
FROZEN = '2df1be2cff66b53ce57b3eefe3eed8be2d16c59d1e37b783341f176995689d94'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def inputs():
    for name, expected in SOURCE.items():
        assert digest(Path(__file__).with_name(name).read_bytes()) == expected, name
    plan = json.loads((ROOT / 'evidence/context-frozen-plan-2026-09-08.json').read_bytes())
    cases = []
    for item in plan['cases']:
        case = quality_context_suite.build_case(item['family_id'], item['padding_count'])
        case.update(item)
        cases.append(case)
    frozen = canonical({**plan['suite_fields'], 'cases': cases})
    assert len(frozen) == 29781351 and digest(frozen) == FROZEN
    by_id = {c['case_id']: c for c in cases}
    selected = []
    for mode, expected in RAW.items():
        raw = (EVIDENCE / (mode + '.jsonl')).read_bytes()
        assert digest(raw) == expected, mode
        rows = [json.loads(line) for line in raw.splitlines()]
        complete = [r for r in rows if r.get('kind') == 'case']
        assert len(complete) == (36 if mode.endswith('fast') else 84)
        assert len({r['case_id'] for r in complete}) == len(complete)
        for row in complete:
            case = by_id[row['case_id']]
            assert row['mode_id'] == mode and row['request_sha256'] == case['request_sha256']
            assert row['input_tokens'] == case['frozen_input_tokens']
            if case['check']['type'] == 'python_function_cases':
                assert row['outcome'] == 'pending_sandbox' and row['error'] == 'python_sandbox_required'
                selected.append((mode, case, row))
    assert len(selected) == 30
    return selected


def emit(stream, value):
    stream.write(canonical(value).decode() + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    selected = inputs()
    if not args.execute:
        print(json.dumps({'inputs_verified': True, 'candidate_answers': len(selected),
                          'frozen_sha256': FROZEN, 'model_requests': 0}))
        return 0
    assert platform.system() == 'Linux' and platform.node().lower() == 'growleepc'
    assert os.statvfs('/mnt/e').f_bavail * os.statvfs('/mnt/e').f_frsize >= 50 * 1024**3
    # Classic overlay2 names a loaded image by config digest; the verified OCI
    # archive binds this config and its diff IDs to the same server manifest.
    source_receipt = ROOT.parent / 'home_lab-artifacts/ai-server-acceleration-20260830/syv-relay-v7/source-receipt.json'
    receipt = json.loads(source_receipt.read_bytes())
    inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', CONFIG_IMAGE], timeout=10))[0]
    assert receipt['image_reference'] == IMAGE and receipt['config_digest'] == CONFIG_IMAGE
    assert inspected['Id'] == CONFIG_IMAGE
    assert inspected['RootFS']['Layers'] == receipt['rootfs_diff_ids']
    assert inspected['Architecture'] == 'amd64' and inspected['Os'] == 'linux'
    prefix = 'qwen-context-pca-20260908-autoround'
    assert not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=' + prefix], timeout=10).strip()
    output = EVIDENCE / 'pc-a-autoround-code.jsonl'
    quality_sandbox.CONTAINER_PREFIX = prefix
    started = time.time()
    baseline = os.statvfs('/mnt/e').f_bavail * os.statvfs('/mnt/e').f_frsize
    counts = collections.Counter()
    with output.open('x', encoding='utf-8') as stream:
        emit(stream, {'kind': 'execution', 'host': platform.node(), 'image': IMAGE,
                      'image_id': inspected['Id'], 'host_python': platform.python_version(),
                      'archive_source_receipt_sha256': digest(source_receipt.read_bytes()),
                      'runner_sha256': digest(Path(__file__).read_bytes()),
                      'sandbox_sha256': SOURCE['quality_sandbox.py'], 'raw_sha256': RAW,
                      'frozen_sha256': FROZEN, 'started_epoch': started,
                      'scope': 'independent PC-A evaluation; canonical r1 unchanged',
                      'preflight_sha256': digest((EVIDENCE / 'pc-a-autoround-preflight.json').read_bytes())})
        references = {c['family_id']: c for _, c, _ in selected}
        for family, case in references.items():
            reference = dict(case, case_id=family + '-reference')
            verdict = quality_sandbox.evaluate_code(reference, case['reference_source'], CONFIG_IMAGE, mode_id='reference')
            emit(stream, {'kind': 'reference', **verdict})
            if verdict['outcome'] != 'PASS':
                print(json.dumps({'status': 'BLOCKED', 'reason': 'reference_failed', 'result': verdict}))
                return 2
        for mode, case, row in selected:
            free = os.statvfs('/mnt/e').f_bavail * os.statvfs('/mnt/e').f_frsize
            assert free >= 50 * 1024**3 and baseline - free < 1024**3
            assert time.time() - started < 1200
            emit(stream, {'kind': 'evaluation_start', 'mode_id': mode, 'case_id': case['case_id']})
            verdict = quality_sandbox.evaluate_code(case, row['content'], CONFIG_IMAGE, mode_id=mode)
            emit(stream, {'kind': 'evaluation', **verdict, 'request_sha256': row['request_sha256'],
                          'content_sha256': digest(row['content'].encode()), 'target_tokens': case['target_tokens']})
            counts[verdict['outcome']] += 1
            if verdict['outcome'] == 'BLOCKED':
                print(json.dumps({'status': 'BLOCKED', 'counts': counts}))
                return 2
        active = subprocess.check_output(['docker', 'ps', '-q', '--filter', 'name=' + prefix], timeout=10).strip()
        assert not active, 'owned_container_still_running'
        emit(stream, {'kind': 'summary', 'counts': counts, 'evaluated': sum(counts.values()),
                      'ended_epoch': time.time(), 'owned_containers_running': False,
                      'physical_free_bytes': os.statvfs('/mnt/e').f_bavail * os.statvfs('/mnt/e').f_frsize})
    print(json.dumps({'status': 'COMPLETE', 'counts': counts, 'result': str(output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
