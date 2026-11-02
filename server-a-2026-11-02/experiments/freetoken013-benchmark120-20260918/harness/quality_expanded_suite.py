"""Explicit, hash-bound selection of the 200-case comparison."""
import hashlib
import json
from pathlib import Path

SUITE_ID = 'qwen38-quality-expanded-200-v1'
SHA256 = '2ea25b195358d0d78a1f4e025c6c63e604693bc00ec4cbc5224e4772e77f187f'

def load_suite():
    raw = Path(__file__).with_name('quality_expanded_v2.json').read_bytes()
    if raw.endswith(b'\n'):
        raw = raw[:-1]
    if len(raw) != 520254 or hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('expanded_suite_identity_invalid')
    suite = json.loads(raw)
    if suite.get('suite_id') != SUITE_ID or len(suite.get('cases', [])) != 200:
        raise ValueError('expanded_suite_contract_invalid')
    if len({c['case_id'] for c in suite['cases']}) != 200:
        raise ValueError('expanded_case_ids_invalid')
    return suite
