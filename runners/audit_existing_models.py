"""Read-only SHA/size audit of the three retained Qwen model trees on Server A.

No imports of model code, GPU use, downloads, subprocesses, or filesystem writes.
Output is JSON lines; identical unchanged hardlinks are hashed only once.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time


BASE = Path('/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey/f958acb153eb8c7bbd25638149c8c75185559226')
ADAPTIVE = Path('/srv/ai-server/artifacts/extensions/qwen-syv-k4-4d4166b891489bfc')
GGUF = Path('/srv/ai-server/model-store/sha256/3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec/Qwen3.8-27B-Uncensored-Q4_K_M.gguf')
PINS = {
    BASE: 'a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a',
    ADAPTIVE: '0031745a1a615256cc37b2a345f2dca2d1744787927d81f04a5752e498e822bb',
}


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def main():
    started = time.time()
    cache = {}
    plans = []
    for root, expected in PINS.items():
        path = root / 'COMPLETE.receipt'
        assert not path.is_symlink() and path.is_file(), str(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == expected, f'receipt mismatch: {path}'
        if root == BASE:
            rows = []
            for line in raw.decode().splitlines():
                match = re.fullmatch(r'([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9_.-]+)', line)
                if match:
                    rows.append(dict(sha256=match[1], size=int(match[2]), name=match[3]))
            assert len(rows) == 16
        else:
            rows = json.loads(raw)['files']
            assert len(rows) == 18
        model = root / 'model'
        assert not model.is_symlink() and model.is_dir()
        assert {p.name for p in model.iterdir()} == {r['name'] for r in rows}, str(model)
        plans.append((str(root), [(model / row['name'], row) for row in rows]))
    plans.append(('original-gguf', [(GGUF, dict(name=GGUF.name, size=16810714496, sha256=GGUF.parent.name))]))
    passed = True
    physical = {}
    for model_id, entries in plans:
        for path, expected in entries:
            assert path.name == expected['name'] and path.parent.resolve() == path.parent
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                assert stat.S_ISREG(before.st_mode)
                key = fingerprint(before)
                assert before.st_size == expected['size'], f'size mismatch: {path}'
                reused = key in cache
                if not reused:
                    digest = hashlib.sha256()
                    while chunk := os.read(fd, 8 * 1024 * 1024):
                        digest.update(chunk)
                    cache[key] = digest.hexdigest()
                assert fingerprint(os.fstat(fd)) == key, f'file changed: {path}'
                assert fingerprint(path.stat()) == key, f'path changed: {path}'
            finally:
                os.close(fd)
            ok = cache[key] == expected['sha256']
            passed = passed and ok
            physical[(before.st_dev, before.st_ino)] = before.st_blocks * 512
            emit(dict(model=model_id, file=path.name, bytes=before.st_size,
                      allocated_bytes=before.st_blocks * 512, hardlinks=before.st_nlink,
                      sha256=cache[key], expected_sha256=expected['sha256'],
                      hash_reused_for_same_inode=reused, status='PASS' if ok else 'FAIL'))
    emit(dict(status='PASS' if passed else 'FAIL', files_checked=sum(len(rows) for _, rows in plans),
              distinct_files_hashed=len(cache), distinct_allocated_bytes=sum(physical.values()),
              elapsed_seconds=round(time.time() - started, 3), writes_performed=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
