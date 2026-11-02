"""Exact-file Remote SSH byte transfer for this finite benchmark only."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from remote_window import ROOT,check_manifest

def guard():
    sys.path.insert(0,'/srv/ai-server/benchmarks/context-20260908-r2')
    from context_storage import OwnedStorage
    # Include the planned ten container metadata allowance in this separate
    # writer's reserve; the active controller measures the exact owned IDs.
    return OwnedStorage(ROOT).check((512+40)*1024**2)

def main():
    manifest=check_manifest(sys.argv[1]);action,case_id=sys.argv[2:4]
    assert case_id in {r['case_id'] for r in manifest['coordinates']},'coordinate_not_in_manifest'
    guard();partial=ROOT/'ingress'/(case_id+'.gz.partial');packed=ROOT/'ingress'/(case_id+'.gz')
    final=ROOT/'requests'/(case_id+'.json')
    if action=='chunk':
        offset=int(sys.argv[4]);chunk=base64.b64decode(sys.argv[5],validate=True)
        assert len(chunk)<=18000 and offset>=0 and offset+len(chunk)<=512*1024
        assert not packed.exists() and not final.exists(),'request_already_finalized'
        assert (not partial.exists() and offset==0) or (partial.is_file() and not partial.is_symlink() and partial.stat().st_size==offset),'unexpected_ingress_offset'
        with partial.open('xb' if offset==0 else 'r+b') as output:
            output.seek(offset);output.write(chunk);output.flush();os.fsync(output.fileno())
        guard();print(json.dumps({'offset':partial.stat().st_size}))
    elif action=='finish':
        expected=sys.argv[4];assert len(expected)==64
        assert not packed.exists() and not final.exists() and not partial.is_symlink()
        assert partial.stat().st_size<=512*1024
        with gzip.open(partial,'rb') as source:raw=source.read(2*1024**2+4097)
        assert len(raw)<=2*1024**2+4096 and hashlib.sha256(raw).hexdigest()==expected,'request_identity'
        value=json.loads(raw);assert value['case_id']==case_id
        transient=final.with_suffix('.json.partial');assert not transient.exists()
        with transient.open('xb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
        # Exact newly created paths only; renames add no retained full copy.
        partial.rename(packed);transient.rename(final)
        guard();print(json.dumps({'case_id':case_id,'bytes':len(raw),'sha256':expected}))
    elif action=='grade':
        raw=base64.b64decode(sys.argv[4],validate=True);assert len(raw)<=4096
        value=json.loads(raw);assert value['case_id']==case_id and value['status'] in ('PASS','FAIL','BLOCKED')
        response=ROOT/'responses'/(case_id+'.json');assert hashlib.sha256(response.read_bytes()).hexdigest()==value['response_sha256']
        path=ROOT/'grades'/(case_id+'.json');assert not path.exists()
        transient=path.with_suffix('.json.partial')
        with transient.open('xb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
        transient.rename(path);guard();print(json.dumps({'case_id':case_id,'grade_saved':True}))
    elif action=='read':
        response=ROOT/'responses'/(case_id+'.json')
        if not response.exists():print(json.dumps({'exists':False}));return
        raw=response.read_bytes();assert len(raw)<=2*1024**2
        print(json.dumps({'exists':True,'sha256':hashlib.sha256(raw).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode('ascii')}))
    else:raise ValueError('operation_not_supported')

if __name__=='__main__':main()
