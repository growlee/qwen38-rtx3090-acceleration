"""Physical accounting adapter for the one context continuation, not a daemon.

Only receipt-bound owned paths/IDs are accepted. No create, mount, delete or
model operation is performed here. The caller must register every writer and
reserve its next bounded write before dispatch. Global free space is always
real statvfs data; it is never replaced with a synthetic available-byte value.
"""
from __future__ import annotations

import json
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import threading
import textwrap

GIB = 1024**3
OWNED_LIMIT = 4 * GIB
ROOT_FLOOR = 150 * GIB
# Retain the existing r2 runbook's restoration/log allowance. It is headroom,
# not a claim that shared daemon metadata is exactly attributable to this run.
WRITE_RESERVE = 512 * 1024**2
SNAPSHOTS = PurePosixPath('/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots')
CONTAINERS = Path('/var/lib/docker/containers')
ID = re.compile(r'[a-f0-9]{64}')


class StorageError(RuntimeError):
    def __init__(self, code, details=None):
        super().__init__(code)
        self.code, self.details = code, details or {}


def need(condition, code):
    if not condition:
        raise StorageError(code)


def command(argv):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                            check=True, timeout=15)
    need(len(result.stdout) <= 1024**2, 'storage_probe_output_bound')
    return result.stdout.decode('utf-8', 'strict')


def physical_bytes(roots):
    """Count allocated blocks once per inode; never traverse a symlink/mount.

    A missing/changing/unreadable tree is not interpreted as zero usage.
    Tree races fail closed where observed. This is a sample of occupancy, not
    cumulative bytes written and not an atomic snapshot of concurrent writers.
    """
    seen, total = set(), 0
    for root in roots:
        root = Path(root)
        need(root.is_absolute() and root.resolve(strict=True) == root,
             'storage_root_not_exact')
        initial = root.lstat()
        device = initial.st_dev
        pending = [root]
        while pending:
            path = pending.pop()
            info = path.lstat()
            need(info.st_dev == device, 'storage_nested_mount')
            need(hasattr(info, 'st_blocks'), 'physical_blocks_unavailable')
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            need(info.st_blocks >= 0, 'physical_blocks_invalid')
            total += info.st_blocks * 512
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(path) as entries:
                    pending.extend(Path(item.path) for item in entries)
                after = path.lstat()
                need((after.st_dev, after.st_ino) == key, 'storage_directory_replaced')
        final = root.lstat()
        need((initial.st_dev, initial.st_ino) == (final.st_dev, final.st_ino),
             'storage_root_replaced')
    return total


def snapshot_roots(container_id, run=command):
    """Ask ctr for the mount *description*, never execute it or use --temp.

    Docker 29/containerd 2 overlayfs has an active ID and a committed ID-init.
    The first lower directory must be that committed init snapshot; shared
    image lower layers are retained inputs and are not charged as new copies.
    Unsupported storage layout is a hard failure, not a SizeRw fallback.
    """
    need(isinstance(container_id, str) and ID.fullmatch(container_id), 'storage_bad_container_id')
    base = ['ctr', '--namespace', 'moby', 'snapshots']
    active = json.loads(run(base + ['info', container_id]))
    init = json.loads(run(base + ['info', container_id + '-init']))
    need(active.get('Name') == container_id and active.get('Kind') == 'Active'
         and active.get('Parent') == container_id + '-init'
         and init.get('Name') == container_id + '-init' and init.get('Kind') == 'Committed',
         'storage_snapshot_identity')
    args = shlex.split(run(base + ['mounts', '/__context_readonly_probe__', container_id]))
    need(len(args) == 7 and args[:5] == ['mount', '-t', 'overlay', 'overlay',
         '/__context_readonly_probe__'] and args[5] == '-o', 'storage_mount_description')
    return _roots_from_options(args[6])


def _roots_from_options(options):
    values = {}
    for option in options.split(','):
        key, sep, value = option.partition('=')
        need(key not in values and sep, 'storage_mount_options')
        values[key] = value
    upper = values.get('upperdir', '')
    lower = values.get('lowerdir', '').split(':')[0]
    pattern = re.escape(str(SNAPSHOTS)) + r'/([0-9]+)/fs'
    upper_match, lower_match = re.fullmatch(pattern, upper), re.fullmatch(pattern, lower)
    need(upper_match and lower_match and upper != lower, 'storage_snapshot_paths')
    root = PurePosixPath(upper).parent
    need(values.get('workdir') == str(root/'work'), 'storage_snapshot_workdir')
    return (Path(root), Path(PurePosixPath(lower).parent))


class OwnedStorage:
    """One run's in-memory ownership; no persistent lifecycle or cleanup."""
    def __init__(self, root, *, run=command, free=None):
        self.root = Path(root)
        self.run = run
        self.free = free or self._free
        self.lock = threading.RLock()
        self.paths = {}
        self.containers = {}
        self.last = None
        self.register_path(self.root)

    def _free(self):
        fs = os.statvfs(self.root)
        return fs.f_bavail * fs.f_frsize

    def register_path(self, path):
        path = Path(path)
        need(path.is_absolute() and path.resolve(strict=True) == path, 'storage_root_not_exact')
        info = path.lstat()
        need(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), 'storage_root_type')
        with self.lock:
            identity = (info.st_dev, info.st_ino)
            need(not self.paths or info.st_dev == self.paths[self.root][0],
                 'storage_root_on_other_filesystem')
            need(path not in self.paths or self.paths[path] == identity, 'storage_root_replaced')
            self.paths[path] = identity

    def register_container(self, container_id, expected_name):
        need(isinstance(container_id, str) and ID.fullmatch(container_id), 'storage_bad_container_id')
        with self.lock:
            data = json.loads(self.run(['docker', 'inspect', container_id]))
            need(len(data) == 1, 'storage_container_ambiguous')
            row = data[0]
            need(row.get('Id') == container_id and row.get('Name') == '/' + expected_name,
                 'storage_container_identity')
            need(row.get('Driver') == 'overlayfs' and row['HostConfig'].get('ReadonlyRootfs') is True,
                 'storage_container_writable_or_unsupported')
            need(all(m.get('Type') == 'bind' and m.get('RW') is False for m in row['Mounts']),
                 'storage_unregistered_container_write_mount')
            need(container_id not in self.containers or self.containers[container_id] == expected_name,
                 'storage_container_rebound')
            for path in (CONTAINERS/container_id, *snapshot_roots(container_id, self.run)):
                self.register_path(path)
            self.containers[container_id] = expected_name

    def check(self, reserve_bytes=0):
        need(type(reserve_bytes) is int and reserve_bytes >= 0, 'storage_bad_write_reserve')
        with self.lock:
            try:
                for path, identity in self.paths.items():
                    info = path.lstat()
                    need((info.st_dev, info.st_ino) == identity, 'storage_root_replaced')
                owned = physical_bytes(self.paths)
                free = self.free()
                need(type(free) is int and free >= 0, 'storage_bad_free_measurement')
                self.last = dict(owned_allocated_bytes=owned, root_free_bytes=free,
                                 owned_container_count=len(self.containers), reserved_write_bytes=reserve_bytes)
                need(owned + reserve_bytes <= OWNED_LIMIT, 'test_storage_budget_exhausted')
                need(free >= ROOT_FLOOR + reserve_bytes, 'root_storage_reserve_exhausted')
                return dict(self.last)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise StorageError('storage_measurement_unavailable') from exc


def install_controller(w, storage):
    """Adapt two predicates in the already SHA-verified inherited controller.

    No frozen source is edited. Exact-match guards reject upstream drift.
    All GPU/RAM/deadline/stop/finally behavior remains the original code.
    """
    def check():
        try:
            storage.check(WRITE_RESERVE)
        except StorageError as exc:
            raise w.WindowError(exc.code, exc.details) from exc
        return True

    patches = [
        (w.LoadGuard, '_run',
         'free_now < RESERVE_BYTES or self.free_before - free_now > MAX_GROWTH_BYTES'),
        (w, 'main',
         'available_bytes()<RESERVE_BYTES or free_before-available_bytes()>MAX_GROWTH_BYTES'),
    ]
    compiled = []
    for owner, name, predicate in patches:
        source = textwrap.dedent(inspect.getsource(getattr(owner, name)))
        need(source.count(predicate) == 1, 'storage_controller_contract_changed')
        source = source.replace(predicate, 'not context_storage_check()')
        namespace = {}
        exec(compile(source, '<context-owned-storage:'+name+'>', 'exec'), w.__dict__, namespace)
        compiled.append((owner, name, namespace[name]))
    w.context_storage_check = check
    for owner, name, function in compiled:
        setattr(owner, name, function)
    original_create = w.create_candidate

    def create(common, args, model, kind):
        check()
        identity = original_create(common, args, model, kind)
        # The original creates but does not start. A failed registration leaves
        # this exact stopped object retained; no unaccounted writer is started.
        storage.register_container(identity, args[args.index('--name')+1])
        check()
        return identity
    w.create_candidate = create


def install_sandbox(sandbox, storage):
    """Register the existing sandbox's create receipt before its first start."""
    original = sandbox._short_command

    def short(argv, *args, **kwargs):
        creating = argv[:2] == ['docker', 'create']
        if creating:
            storage.check(WRITE_RESERVE)
        result = original(argv, *args, **kwargs)
        if creating and not result.error and result.returncode == 0:
            identity = result.stdout.decode('ascii', 'strict').strip()
            storage.register_container(identity, argv[argv.index('--name')+1])
            storage.check(WRITE_RESERVE)
        return result
    sandbox._short_command = short
