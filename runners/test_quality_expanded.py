import collections
import io
import json
import unittest
from unittest import mock
from quality_client import QualityClient, context_preflight
from quality_expanded_suite import load_suite
from quality_suite import SUITE

class Response:
    status=200
    def __init__(self,value): self.raw=json.dumps(value).encode()
    def read(self,limit): return self.raw[:limit]

class Fake:
    def __init__(self,replies): self.replies=iter(replies);self.sent=[]
    def connection(self): return self
    def headers(self,_): return {}
    def request(self,method,path,body,headers): self.sent.append((method,path,json.loads(body)))
    def getresponse(self): return Response(next(self.replies))
    def close(self): pass

class ExpandedTests(unittest.TestCase):
    def test_explicit_new_suite_preserves_old(self):
        suite=load_suite()
        self.assertEqual(len(SUITE['cases']),60)
        self.assertEqual(collections.Counter(c['category'] for c in suite['cases']),{'coding':120,'reasoning_ru':40,'instruction':20,'retrieval':20})
        self.assertFalse(set(c['case_id'] for c in suite['cases']) & set(c['case_id'] for c in SUITE['cases']))
        client=QualityClient('http://127.0.0.1:8080',None,180)
        for case in suite['cases']:
            body=client._body('qwen3.8-27b',case)
            self.assertNotIn('reference_source',body)
            self.assertNotIn('broken_source',body)
            self.assertNotIn('check',body)
            self.assertEqual(body['chat_template_kwargs'],{'enable_thinking':False})
            self.assertEqual(body['temperature'],0)
    def test_llama_preflight_counts_real_rendered_template(self):
        case=load_suite()['cases'][0]
        client=Fake([{'prompt':'rendered'}, {'tokens':[1,2,3]}])
        result=context_preflight(client,'model',{'cases':[case]},'llama')
        self.assertEqual(result['counts'],{case['case_id']:3})
        self.assertEqual(client.sent[1][2],{'content':'rendered','add_special':True})
    def test_vllm_preflight_rejects_shared_context_overflow(self):
        case=load_suite()['cases'][0]
        client=Fake([{'count':16384}])
        with self.assertRaisesRegex(Exception,'shared_context_bound_exceeded'):
            context_preflight(client,'model',{'cases':[case]},'vllm')
    def test_vllm_tools_forwarded_and_boolean_count_rejected(self):
        case=next(c for c in load_suite()['cases'] if c['check']['type']=='tool_call')
        client=Fake([{'count':True}])
        with self.assertRaises(Exception):context_preflight(client,'model',{'cases':[case]},'vllm')
        self.assertEqual(client.sent[0][2]['tools'],case['request']['tools'])

if __name__=='__main__':unittest.main()
