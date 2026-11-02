"""Trusted tokenizer-only preflight inside the existing FreeToken container."""
import hashlib
import json
import sys
from transformers import AutoTokenizer

raw = sys.stdin.buffer.read(600000)
canonical = raw[:-1] if raw.endswith(b'\n') else raw
assert hashlib.sha256(canonical).hexdigest() == '2ea25b195358d0d78a1f4e025c6c63e604693bc00ec4cbc5224e4772e77f187f'
suite = json.loads(raw)
cases = [c for c in suite['cases'] if c['category'] == 'coding']
assert len(cases) == 120
tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
counts = {}
for case in cases:
    request = case['request']
    ids = tokenizer.apply_chat_template(request['messages'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert isinstance(ids, list) and all(type(t) is int for t in ids)
    count = len(ids)
    assert 0 < count and count + request['generation']['max_tokens'] + 128 <= 16384
    counts[case['case_id']] = count
print(json.dumps({'cases': len(counts), 'within_shared_context_bound': True,
    'shared_context_limit': 16384, 'margin_tokens': 128, 'counts': counts}))
