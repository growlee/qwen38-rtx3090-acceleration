"""One existing CPU sandbox checks all fixed references, without GPU access."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent

SOURCE='''def reference_check(cases):
    import copy
    def same(a,b):
        if type(a) is bool or type(b) is bool: return type(a) is type(b) and a==b
        if isinstance(a,(int,float)) and isinstance(b,(int,float)): return a==b
        if type(a) is not type(b): return False
        if isinstance(a,list): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        return a==b
    out=[]
    for case in cases:
        ns={}
        exec(case['reference_source'],ns,ns)
        function=ns[case['check']['function_name']]
        good=True
        for vector in case['check']['cases']:
            try:
                actual=function(*copy.deepcopy(vector['args']),**copy.deepcopy(vector['kwargs']))
                passed='expected' in vector and same(actual,vector['expected'])
            except Exception as exc:
                passed=vector.get('expected_exception')=='ValueError' and type(exc) is ValueError
            good=good and passed
        out.append(good)
    return out
'''

def main():
    assert len(sys.argv)==2
    raw=(ROOT/'bundle-manifest-expanded.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sys.argv[1]
    manifest=json.loads(raw)
    for name,digest in manifest['files'].items(): assert hashlib.sha256((ROOT/'sources'/name).read_bytes()).hexdigest()==digest
    for name,digest in manifest['other_files'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    sys.path.insert(0,str(ROOT/'sources'))
    import quality_sandbox
    from quality_expanded_suite import load_suite
    quality_sandbox.CONTAINER_PREFIX='qwen-quality-expanded-20260908-preflight'
    suite=load_suite()
    cases=[c for c in suite['cases'] if c['category']=='coding']
    case={'case_id':'trusted-references','check':{'type':'python_function_cases','function_name':'reference_check','cases':[{'args':[cases],'kwargs':{},'expected':[True]*120}]}}
    result=quality_sandbox.evaluate_code(case,SOURCE,mode_id='trusted')
    with (ROOT/'trusted-preflight.json').open('x') as out: json.dump(result,out,sort_keys=True)
    print(json.dumps(result,sort_keys=True))
    return result['outcome']!='PASS'

if __name__=='__main__': raise SystemExit(main())
