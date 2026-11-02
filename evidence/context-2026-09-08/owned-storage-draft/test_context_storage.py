import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest import mock
import uuid


sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_storage as storage


GIB = 1024**3
CID = 'a' * 64
INHERITED = (Path(__file__).resolve().parents[1] / 'qwen-quality-remaining-20260908'
             / 'run_quality_window_remaining.py')


class StorageTestCase(unittest.TestCase):
    def assert_storage_error(self, code, call):
        with self.assertRaises(storage.StorageError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def owned(self, allocated, free_values):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        values = iter(free_values)
        owned = storage.OwnedStorage(root, free=lambda: next(values))
        blocks = mock.patch.object(storage, 'physical_bytes', return_value=allocated)
        blocks.start()
        self.addCleanup(blocks.stop)
        return owned

    def inherited(self):
        name = '_context_storage_inherited_' + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, INHERITED)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        if os.name == 'nt':
            fcntl = ModuleType('fcntl')
            with mock.patch.dict(sys.modules, {'fcntl': fcntl}):
                spec.loader.exec_module(module)
        else:
            spec.loader.exec_module(module)
        return module


class OwnedStorageLimitTests(StorageTestCase):
    def test_owned_four_gib_boundary_is_inclusive_and_independent(self):
        self.owned(4 * GIB, [200 * GIB]).check()
        owned = self.owned(4 * GIB + 1, [200 * GIB])
        self.assert_storage_error('test_storage_budget_exhausted', owned.check)

    def test_root_floor_is_inclusive_and_independent(self):
        self.owned(1, [150 * GIB]).check()
        owned = self.owned(1, [150 * GIB - 1])
        self.assert_storage_error('root_storage_reserve_exhausted', owned.check)

    def test_foreign_root_drop_over_four_gib_is_allowed_while_floor_holds(self):
        owned = self.owned(1 * GIB, [160 * GIB, 155 * GIB])
        owned.check()
        result = owned.check()
        self.assertEqual(result['root_free_bytes'], 155 * GIB)
        self.assertEqual(result['owned_allocated_bytes'], 1 * GIB)

    def test_write_reserve_is_charged_to_both_constraints(self):
        self.owned(3 * GIB, [151 * GIB]).check(reserve_bytes=1 * GIB)
        owned = self.owned(3 * GIB + 1, [200 * GIB])
        self.assert_storage_error(
            'test_storage_budget_exhausted', lambda: owned.check(reserve_bytes=1 * GIB))
        owned = self.owned(1, [151 * GIB - 1])
        self.assert_storage_error(
            'root_storage_reserve_exhausted', lambda: owned.check(reserve_bytes=1 * GIB))

    def test_missing_or_replaced_registered_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, 'owned')
            root.mkdir()
            owned = storage.OwnedStorage(root.resolve(), free=lambda: 200 * GIB)
            root.rmdir()
            self.assert_storage_error('storage_measurement_unavailable', owned.check)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, 'owned')
            displaced = Path(temporary, 'old')
            root.mkdir()
            owned = storage.OwnedStorage(root.resolve(), free=lambda: 200 * GIB)
            root.rename(displaced)
            root.mkdir()
            self.assert_storage_error('storage_root_replaced', owned.check)


class PhysicalAccountingTests(StorageTestCase):
    def test_allocated_blocks_are_deduplicated_and_symlinks_are_not_followed(self):
        class FakePath:
            def __init__(self, name, info):
                self.name, self.info = name, info

            def is_absolute(self):
                return True

            def resolve(self, strict=False):
                return self

            def lstat(self):
                return self.info

            def __hash__(self):
                return hash(self.name)

            def __eq__(self, other):
                return isinstance(other, FakePath) and self.name == other.name

        root = FakePath('/owned', SimpleNamespace(
            st_dev=1, st_ino=1, st_mode=stat.S_IFDIR | 0o700, st_blocks=8, st_size=4096))
        sparse = FakePath('/owned/sparse', SimpleNamespace(
            st_dev=1, st_ino=2, st_mode=stat.S_IFREG | 0o600, st_blocks=2, st_size=100 * GIB))
        hardlink = FakePath('/owned/hardlink', sparse.info)
        link = FakePath('/owned/link', SimpleNamespace(
            st_dev=1, st_ino=3, st_mode=stat.S_IFLNK | 0o777, st_blocks=1, st_size=4096))
        paths = {item.name: item for item in (root, sparse, hardlink, link)}

        class Entries:
            def __enter__(self):
                return [SimpleNamespace(path=name) for name in paths if name != '/owned']

            def __exit__(self, *_):
                return False

        with mock.patch.object(storage, 'Path', side_effect=lambda value: value if isinstance(value, FakePath) else paths[value]), \
                mock.patch.object(storage.os, 'scandir', return_value=Entries()):
            self.assertEqual(storage.physical_bytes([root]), (8 + 2 + 1) * 512)


class ContainerRegistrationTests(StorageTestCase):
    def inspect_row(self, **changes):
        row = {
            'Id': CID,
            'Name': '/candidate',
            'Driver': 'overlayfs',
            'HostConfig': {'ReadonlyRootfs': True},
            'Mounts': [{'Type': 'bind', 'RW': False}],
        }
        row.update(changes)
        return row

    def test_invalid_id_and_inspected_identity_mismatch_fail_closed(self):
        run = mock.Mock()
        owned = self.owned(1, [200 * GIB])
        owned.run = run
        self.assert_storage_error('storage_bad_container_id',
                                  lambda: owned.register_container('not-an-id', 'candidate'))
        run.assert_not_called()
        owned = self.owned(1, [200 * GIB])
        owned.run = lambda _: __import__('json').dumps([self.inspect_row(Id='b' * 64)])
        self.assert_storage_error('storage_container_identity',
                                  lambda: owned.register_container(CID, 'candidate'))

    def test_writable_or_unsupported_container_layout_fails_closed(self):
        cases = (
            ('storage_container_writable_or_unsupported',
             self.inspect_row(HostConfig={'ReadonlyRootfs': False})),
            ('storage_container_writable_or_unsupported', self.inspect_row(Driver='other')),
            ('storage_unregistered_container_write_mount',
             self.inspect_row(Mounts=[{'Type': 'bind', 'RW': True}])),
            ('storage_unregistered_container_write_mount',
             self.inspect_row(Mounts=[{'Type': 'volume', 'RW': False}])),
        )
        for code, row in cases:
            with self.subTest(code=code, row=row):
                owned = self.owned(1, [200 * GIB])
                owned.run = lambda _, value=row: __import__('json').dumps([value])
                self.assert_storage_error(code, lambda: owned.register_container(CID, 'candidate'))


class SnapshotParserTests(StorageTestCase):
    def snapshot_run(self, mount_description, active=None, init=None):
        active = active or {'Name': CID, 'Kind': 'Active', 'Parent': CID + '-init'}
        init = init or {'Name': CID + '-init', 'Kind': 'Committed'}

        def run(argv):
            if argv[-2:] == ['info', CID]:
                return __import__('json').dumps(active)
            if argv[-2:] == ['info', CID + '-init']:
                return __import__('json').dumps(init)
            self.assertEqual(argv[-3:], ['mounts', '/__context_readonly_probe__', CID])
            return mount_description

        return run

    def options(self, upper='41', lower='40', work=None, tail='/shared/image/fs'):
        base = str(storage.SNAPSHOTS)
        work = work or f'{base}/{upper}/work'
        return f'workdir={work},upperdir={base}/{upper}/fs,lowerdir={base}/{lower}/fs:{tail}'

    def test_exact_seven_argv_mount_description_uses_upper_and_first_lower(self):
        description = ('mount -t overlay overlay /__context_readonly_probe__ -o ' + self.options())
        roots = storage.snapshot_roots(CID, self.snapshot_run(description))
        self.assertEqual(roots, (Path(storage.SNAPSHOTS/'41'), Path(storage.SNAPSHOTS/'40')))

    def test_snapshot_identity_mismatch_fails_closed(self):
        description = ('mount -t overlay overlay /__context_readonly_probe__ -o ' + self.options())
        run = self.snapshot_run(description, active={
            'Name': CID, 'Kind': 'Committed', 'Parent': CID + '-init'})
        self.assert_storage_error('storage_snapshot_identity', lambda: storage.snapshot_roots(CID, run))

    def test_extra_mount_argv_is_rejected(self):
        description = ('mount -t overlay overlay /__context_readonly_probe__ --extra -o ' + self.options())
        self.assert_storage_error(
            'storage_mount_description', lambda: storage.snapshot_roots(CID, self.snapshot_run(description)))

    def test_paths_outside_snapshot_root_and_wrong_workdir_are_rejected(self):
        base = str(storage.SNAPSHOTS)
        variants = (
            ('storage_snapshot_paths',
             f'workdir=/tmp/41/work,upperdir=/tmp/41/fs,lowerdir={base}/40/fs'),
            ('storage_snapshot_paths',
             f'workdir={base}/41/work,upperdir={base}/41/fs,lowerdir=/tmp/40/fs'),
            ('storage_snapshot_workdir', self.options(work='/tmp/work')),
        )
        for code, options in variants:
            with self.subTest(code=code, options=options):
                self.assert_storage_error(code,
                                          lambda value=options: storage._roots_from_options(value))


class AdapterTests(StorageTestCase):
    def test_controller_rewrites_exactly_two_predicates_without_touching_source_or_available(self):
        inherited = self.inherited()
        before = hashlib.sha256(INHERITED.read_bytes()).hexdigest()
        available = inherited.available_bytes
        original_getsource = storage.inspect.getsource
        originals, compiled = [], []

        def source(obj):
            value = original_getsource(obj)
            originals.append(storage.textwrap.dedent(value))
            return value

        real_compile = compile

        def record(value, filename, *args, **kwargs):
            if filename.startswith('<context-owned-storage:'):
                compiled.append(value)
            return real_compile(value, filename, *args, **kwargs)

        accounting = mock.Mock()
        accounting.check.return_value = {}
        with mock.patch.object(storage.inspect, 'getsource', side_effect=source), \
                mock.patch('builtins.compile', side_effect=record):
            storage.install_controller(inherited, accounting)

        predicates = (
            'free_now < RESERVE_BYTES or self.free_before - free_now > MAX_GROWTH_BYTES',
            'available_bytes()<RESERVE_BYTES or free_before-available_bytes()>MAX_GROWTH_BYTES',
        )
        self.assertEqual(len(originals), 2)
        self.assertEqual(len(compiled), 2)
        for original, changed, predicate in zip(originals, compiled, predicates):
            self.assertEqual(changed, original.replace(predicate, 'not context_storage_check()'))
        self.assertIs(inherited.available_bytes, available)
        self.assertEqual(hashlib.sha256(INHERITED.read_bytes()).hexdigest(), before)

    def test_candidate_hook_reserves_then_registers_and_never_starts_on_registration_failure(self):
        def exercise(registration_fails):
            inherited = self.inherited()
            events = []

            class Accounting:
                def check(self, reserve):
                    events.append(('check', reserve))

                def register_container(self, identity, name):
                    events.append(('register', identity, name))
                    if registration_fails:
                        raise storage.StorageError('registration_failed')

            def create(common, args, model, kind):
                events.append(('create', tuple(args)))
                common.run(['docker', 'create'])
                return CID

            common = SimpleNamespace(run=lambda argv: events.append(('docker', tuple(argv))))
            inherited.create_candidate = create
            storage.install_controller(inherited, Accounting())
            call = lambda: inherited.create_candidate(
                common, ['--name', 'candidate'], '/model', 'syv')
            if registration_fails:
                with self.assertRaises(storage.StorageError):
                    call()
            else:
                self.assertEqual(call(), CID)
            return events

        success = exercise(False)
        self.assertEqual(success, [
            ('check', storage.WRITE_RESERVE),
            ('create', ('--name', 'candidate')),
            ('docker', ('docker', 'create')),
            ('register', CID, 'candidate'),
            ('check', storage.WRITE_RESERVE),
        ])
        failed = exercise(True)
        self.assertEqual(failed[:4], success[:4])
        self.assertEqual(len(failed), 4)
        self.assertFalse(any(event[0] == 'docker' and event[1][:2] == ('docker', 'start')
                             for event in failed))

    def test_sandbox_registers_successful_create_receipt_before_return(self):
        events = []
        result = SimpleNamespace(error=None, returncode=0, stdout=(CID + '\n').encode('ascii'))

        class Accounting:
            def check(self, reserve):
                events.append(('check', reserve))

            def register_container(self, identity, name):
                events.append(('register', identity, name))

        sandbox = SimpleNamespace(
            _short_command=lambda argv, *args, **kwargs: events.append(('create', tuple(argv))) or result)
        storage.install_sandbox(sandbox, Accounting())
        returned = sandbox._short_command(['docker', 'create', '--name', 'sandbox'])
        self.assertIs(returned, result)
        self.assertEqual(events, [
            ('check', storage.WRITE_RESERVE),
            ('create', ('docker', 'create', '--name', 'sandbox')),
            ('register', CID, 'sandbox'),
            ('check', storage.WRITE_RESERVE),
        ])


if __name__ == '__main__':
    unittest.main()
