"""Finite experiment window; reuses the deployed checked operator transport.

No standalone CLI and no Qwen/Pony lifecycle verbs. The caller must stop its
own candidate and prove runtime restoration before this window reopens admission.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import time

CLIENT = Path('/opt/homelab/ai-server/supervisor/operator_client.py')
CLIENT_SHA = '399c6280ec618d1ac825e7540abccbbd9cc4ae67e493f834738823e60784a522'
LOCKS = (
    Path('/srv/ai-server/state/acceleration-experiments/locks/host.lock'),
    Path('/run/ai-server-experiment.lock'),
)
EMPTY_DIGEST = '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'


def checked_client():
    info = CLIENT.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise RuntimeError('operator_source_custody')
    if hashlib.sha256(CLIENT.read_bytes()).hexdigest() != CLIENT_SHA:
        raise RuntimeError('operator_source_changed')
    spec = importlib.util.spec_from_file_location('freetoken013_checked_operator', CLIENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_empty(value, phase, identity=None, epoch=None):
    if not (value.get('ok') is True and value.get('phase') == phase
            and value.get('capacity_mode') == 'serialized-99'
            and value.get('restart_pending') is False
            and value.get('reconciliation_required') is False
            and value.get('queue_depth') == 0 and value.get('active_ledger') == []
            and value.get('active_ledger_digest') == EMPTY_DIGEST
            and type(value.get('admission_epoch')) is int):
        raise RuntimeError('admission_not_empty_safe_' + phase)
    if identity is not None and any(value.get(k) != identity.get(k) for k in
                                     ('boot_id', 'coordinator_generation', 'capacity_mode')):
        raise RuntimeError('admission_identity_changed')
    if epoch is not None and value['admission_epoch'] != epoch:
        raise RuntimeError('admission_epoch_changed')


class Admission:
    def __init__(self, client, record):
        self.client, self.record = client, record

    def qwen_stopped(self):
        value, code = self.client.operate('status', 'qwen')
        self.record('qwen_status', value)
        if code != 0 or value.get('state') != 'stopped' or value.get('active_profiles') != []:
            raise RuntimeError('initial_stopped_qwen_or_idle_lanes_changed')

    def request(self, phase=None):
        request = {'schema': 'ai-admission-operator-command/v1',
                   'verb': 'status' if phase is None else 'advance-phase',
                   'request_id': 'ft013-' + str(time.time_ns())}
        if phase is not None:
            if phase not in ('closed', 'public'):
                raise ValueError('phase_out_of_scope')
            request['phase'] = phase
        self.record('operator_dispatch', request)
        value = self.client._exchange(
            self.client.SOCKET_PATH, request, expected_uid=0,
            expected_gid=self.client._server_group_gid(), platform_name=sys.platform,
            os_name=os.name)
        self.record('operator_response', value)
        if (value.get('schema') != 'ai-admission-operator-response/v1'
                or value.get('request_id') != request['request_id'] or value.get('ok') is not True):
            raise RuntimeError('admission_response_invalid_or_rejected')
        return value

    def transition_once(self, before, phase):
        expected_epoch = before['admission_epoch'] + 1
        try:
            value = self.request(phase)
            require_empty(value, phase, before, expected_epoch)
        except Exception as exc:
            # Reconcile once, read-only. Never resend a mutation automatically.
            self.record('transition_reconciliation', {'phase': phase, 'error': type(exc).__name__})
            value = self.request()
            require_empty(value, phase, before, expected_epoch)
        return value


@contextlib.contextmanager
def exclusive_window(restore_candidate_and_verify, record):
    """Caller supplies exact-container stop + GPU/Docker/config verification.

    If restoration or identity proof fails, leave admission closed and raise.
    This deliberately cannot restart Qwen, reset breakers or reconcile queues.
    """
    import fcntl
    client = checked_client()
    descriptors = []
    try:
        for path in LOCKS:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
            descriptors.append(fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                raise RuntimeError('experiment_lock_custody')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        operator = Admission(client, record)
        initial = operator.request()
        require_empty(initial, 'public')
        operator.qwen_stopped()
        closed = operator.transition_once(initial, 'closed')
        try:
            operator.qwen_stopped()
            yield {'initial': initial, 'closed': closed}
        finally:
            restore_candidate_and_verify()
            operator.qwen_stopped()
            current = operator.request()
            require_empty(current, 'closed', closed, closed['admission_epoch'])
            operator.transition_once(current, 'public')
    finally:
        for fd in reversed(descriptors):
            os.close(fd)
