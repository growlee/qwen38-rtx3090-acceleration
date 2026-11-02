"""Bounded stdin/stdout transport for one pinned runtime; never executes model code.

Reuses the retained benchmark Client's loopback validation, connection and headers,
and quality_client's numeric usage filter. SSE semantics follow quality_client;
this text-only adapter preserves exact monotonic anchors and completed empty output.
"""
import hashlib
import http.client
import json
import socket
import sys
import time

from benchmark_client import Client
from quality_client import _numeric_usage

MAX_REQUEST=2*1024**2
MODEL='qwen3.8-27b'

def wire(value):return json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf-8')

def post(body,path,timeout=60):
    client=Client('http://127.0.0.1:8080',None,timeout);connection=client.connection()
    try:
        connection.request('POST',path,body=wire(body),headers=client.headers('application/json'))
        response=connection.getresponse();raw=response.read(4*1024**2+1)
        if response.status!=200 or len(raw)>4*1024**2:raise ValueError('tokenizer_http_or_size')
        return json.loads(raw)
    finally:connection.close()

def stream(body,timeout=900):
    raw=wire(body)
    result=dict(stream_complete=False,request_sha256=hashlib.sha256(raw).hexdigest(),
                request_bytes=len(raw),anchors=dict(send=None,first_generated=None,first_content=None,stream_end=None),
                content='',reasoning='',tool_calls=[],usage=None,prompt_tokens=None,output_tokens=None,
                reasoning_tokens=None,finish_reason=None,error=None,elapsed_wait_seconds=None,
                path='isolated benchmark backend; client shares exact candidate network namespace',
                prefix_cache=False,clock='time.perf_counter; same transport process')
    client=Client('http://127.0.0.1:8080',None,timeout);connection=client.connection()
    anchors=result['anchors'];size=0;generated=0
    try:
        anchors['send']=time.perf_counter();deadline=anchors['send']+timeout
        connection.request('POST','/v1/chat/completions',body=raw,headers=client.headers('text/event-stream'))
        response=connection.getresponse()
        if response.status!=200:raise ValueError('generation_http_'+str(response.status))
        if 'text/event-stream' not in response.getheader('Content-Type','').lower():raise ValueError('sse_content_type')
        while True:
            remaining=deadline-time.perf_counter()
            if remaining<=0:raise TimeoutError()
            if connection.sock is not None:connection.sock.settimeout(remaining)
            line=response.readline(1024**2+1);now=time.perf_counter()
            if not line:raise ValueError('stream_incomplete')
            size+=len(line)
            if len(line)>1024**2 or size>4*1024**2:raise ValueError('sse_size_limit')
            if not line.startswith(b'data:'):continue
            payload=line[5:].strip()
            if payload==b'[DONE]':anchors['stream_end']=now;result['stream_complete']=True;break
            event=json.loads(payload)
            if not isinstance(event,dict):raise ValueError('invalid_sse_event')
            for choice in event.get('choices',[]):
                delta=choice.get('delta') or {}
                for field,value,visible in [('reasoning',delta.get('reasoning_content',delta.get('reasoning')),False),
                                            ('content',delta.get('content'),True),('tool_calls',delta.get('tool_calls'),True)]:
                    if not value:continue
                    if field=='tool_calls':
                        if not isinstance(value,list):raise ValueError('invalid_tool_delta')
                        encoded=wire(value);result[field].extend(value)
                    else:
                        if not isinstance(value,str):raise ValueError('invalid_text_delta')
                        encoded=value.encode('utf-8');result[field]+=value
                    generated+=len(encoded)
                    if generated>256*1024:raise ValueError('generated_content_limit')
                    if anchors['first_generated'] is None:anchors['first_generated']=now
                    if visible and anchors['first_content'] is None:anchors['first_content']=now
                if choice.get('finish_reason') is not None:result['finish_reason']=str(choice['finish_reason'])
            if isinstance(event.get('usage'),dict):result['usage']=_numeric_usage(event['usage'])
        usage=result['usage'] or {}
        for output,key in [('prompt_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
            value=usage.get(key);result[output]=value if type(value) is int and value>=0 else None
        value=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        result['reasoning_tokens']=value if type(value) is int and value>=0 else None
        if result['finish_reason'] is None:result['error']='finish_reason_missing'
        elif result['output_tokens'] is None:result['error']='usage_missing'
    except (socket.timeout,TimeoutError):result['error']='timeout'
    except (OSError,http.client.HTTPException,ValueError,TypeError,KeyError) as exc:
        result['error']=str(exc)[:160] or type(exc).__name__
    finally:
        result['elapsed_wait_seconds']=time.perf_counter()-anchors['send'] if anchors['send'] is not None else None
        connection.close()
    result['empty_output']=not result['content'] and not result['tool_calls']
    result['truncation']=result['finish_reason']=='length'
    text=result['content'];result['repetition']=len(text)>=1024 and text[-128:]*8 in text
    result['sse_bytes']=size
    return result

def capture(envelope):
    body=envelope['body'];target=envelope.get('input_target')
    if body.get('model')!=MODEL or body.get('stream') is not True:raise ValueError('request_contract')
    if len(wire(body))>MAX_REQUEST:raise ValueError('request_size')
    started=time.perf_counter()
    count=post(dict(model=MODEL,messages=body['messages'],add_generation_prompt=True,
                    chat_template_kwargs=body['chat_template_kwargs']),'/tokenize').get('count')
    tokenizer_seconds=time.perf_counter()-started
    if type(count) is not int or count<=0:raise ValueError('tokenizer_count_invalid')
    if target is not None and (count!=envelope['prompt_tokens_local'] or not target<=count<=target+256):
        raise ValueError('backend_tokenizer_parity_failed')
    if count+body['max_tokens']+128>204800:raise ValueError('capacity_BLOCKED')
    response=stream(body)
    response.update(case_id=envelope['case_id'],tokenizer_seconds=tokenizer_seconds,backend_tokenizer_count=count)
    if response['stream_complete'] and response['prompt_tokens']!=count:response['error']='usage_prompt_parity_failed'
    return response

def main():
    client=Client('http://127.0.0.1:8080',None,60)
    status,raw,_=client.get('/v1/models');models=json.loads(raw).get('data',[])
    matches=[m for m in models if m.get('id')==MODEL]
    if status!=200 or len(matches)!=1 or matches[0].get('max_model_len')!=204800:raise ValueError('runtime_identity_or_capacity')
    for line in sys.stdin.buffer:
        if len(line)>MAX_REQUEST+4096:raise ValueError('envelope_size')
        envelope=json.loads(line)
        try:result=capture(envelope)
        except Exception as exc:result=dict(case_id=envelope['case_id'],pre_generation_error=str(exc)[:200],stream_complete=False)
        print(wire(result).decode('utf-8'),flush=True)
        if result.get('pre_generation_error') or result.get('error'):return 3
    return 0

if __name__=='__main__':raise SystemExit(main())
