"""Admission, value-free JSON/SSE errors, and four-lane cleanup regressions."""
from concurrent.futures import Future
import copy
import json
import threading

import pytest
from fastapi.testclient import TestClient

from api_protocol import OutputFramer, normalize_request
from api_server import Job, MODEL, create_app, public_error
from engine.tool_calls import parse_tool_calls
from tool_schema import ToolCallError, ToolSchemaError, validate_tool_schema
from test_batched_mtp_api import setup_api
from test_continuous_mtp import greedy_gold


SCHEMA = dict(type='object', properties=dict(command=dict(type='string'),
    args=dict(type='array', items=dict(type='string')), timeoutMs=dict(type='number')),
    required=['command'], additionalProperties=False)
TOOLS = [dict(type='function', function=dict(name='search', parameters=SCHEMA))]
SECRET = 'PRIVATE_ARGUMENT_VALUE_84ce15'


def body(**kwargs):
    return dict(model=MODEL, messages=[dict(role='user', content='A')], temperature=0,
                max_tokens=32, **kwargs)


def call(parameters='', name='search'):
    return f'<tool_call><function={name}>{parameters}</function></tool_call>'


@pytest.mark.parametrize('schema', [dict(command=dict(type='string', required=True)),
    dict(type='object', properties=dict(command=dict(type='string', required=True))),
    dict(type='object', required=True), dict(type='object', properties=[]),
    dict(type='object', properties=dict(args=dict(type='array', items=[]))),
    dict(type='object', properties=dict(x={'$ref':'https://example.invalid/schema'})),
    dict(type='object', properties=dict(x={'$ref':'#/$defs/missing'})),
    dict(type='object', properties=dict(x={'$ref':'#'})),
    dict(type='object', properties=dict(x={'$dynamicRef':'#x'})), [], True, 'object'])
@pytest.mark.parametrize('stream', [False, True])
def test_malformed_schema_is_http400_before_admission(schema, stream):
    class UnusedRuntime:
        ready=True
        def submit(self, options):
            pytest.fail('Invalid schema reached scheduler/tokenizer/GPU')
    tool=copy.deepcopy(TOOLS);tool[0]['function']['parameters']=schema
    with TestClient(create_app(UnusedRuntime(), 'test')) as client:
        r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer test'},
                      json=body(tools=tool,stream=stream))
    assert r.status_code == 400
    e=r.json()['error']
    assert e['code']=='invalid_tool_schema' and e['type']=='invalid_request_error'
    assert e['tool']=='search' and e['param'].startswith('tools[0].function.parameters')
    assert 'search' in e['message'] and 'Retry' not in e['message']
    assert 'text/event-stream' not in r.headers['content-type']


def test_valid_schema_nested_values_optional_help_and_legacy_no_args():
    help_schema=copy.deepcopy(SCHEMA);help_schema.pop('required')
    validate_tool_schema(help_schema,tool='search_help',param='tools[1].function.parameters')
    assert parse_tool_calls(call('', 'search_help'), [dict(type='function',function=dict(
        name='search_help',parameters=help_schema))])[0]['function']['arguments']=={}
    xml=call('<parameter=command>\nhelp\n</parameter><parameter=args>["--x","你好"]</parameter>')
    args=parse_tool_calls(xml,TOOLS)[0]['function']['arguments']
    assert args==dict(command='help',args=['--x','你好'])
    for schema in ({},dict(type='object',properties={})):
        tool=[dict(type='function',function=dict(name='search',parameters=schema))]
        normalize_request(body(tools=tool),MODEL)
        assert parse_tool_calls(call(),tool)[0]['function']['arguments']=={}
    nested=dict(type='object',properties=dict(config={'$ref':'#/$defs/config'}),
        **{'$defs':dict(config=dict(type='object',properties=dict(enabled=dict(type='boolean')),
                                   required=['enabled'],additionalProperties=False))})
    validate_tool_schema(nested,tool='search',param='parameters')
    t=[dict(type='function',function=dict(name='search',parameters=nested))]
    assert parse_tool_calls(call('<parameter=config>{"enabled":true}</parameter>'),t)
    nested['properties']['command']={'$ref':'#/$defs/command'}
    nested['$defs']['command']=dict(type='string',enum=['help'])
    assert parse_tool_calls(call('<parameter=command>help</parameter>'),t)[0]['function']['arguments']=={'command':'help'}


@pytest.mark.parametrize('xml,path,reason', [
    (call(f'<parameter=unknown>{SECRET}</parameter>'),'arguments.unknown','Unknown tool argument'),
    (call(),'arguments.command','Missing required tool argument'),
    (call(f'<parameter=timeoutMs>"{SECRET}"</parameter>'),'arguments.timeoutMs','wrong type'),
    (call(f'<parameter=args>["ok",{{"private":"{SECRET}"}}]</parameter>'),'arguments.args[1]','wrong type'),
    (call(f'<parameter=args>{SECRET}</parameter>'),'arguments.args','valid JSON'),
    (call('<parameter=timeoutMs>1e400</parameter>'),'arguments.timeoutMs','Nonfinite'),
    (call('<parameter=command>help</parameter><parameter=command>other</parameter>'),
     'arguments.command','Duplicate'),
])
def test_tool_errors_name_tool_and_path_without_argument_values(xml,path,reason):
    with pytest.raises(ToolCallError) as caught:
        parse_tool_calls(xml,TOOLS)
    e=public_error(caught.value)
    assert e['code']=='invalid_tool_call' and e['param']==path and e['tool']=='search'
    assert reason in e['message'] and SECRET not in json.dumps(e)


def test_incomplete_final_call_has_typed_error():
    with pytest.raises(ToolCallError,match='incomplete tool call'):
        OutputFramer(TOOLS).feed('<tool_call><function=search>',final=True)


@pytest.mark.parametrize('stream',[False,True])
def test_generated_tool_error_json502_or_sse_has_same_structured_fields(stream):
    error=ToolCallError('Unknown tool argument',tool='search',param='arguments.unknown')
    class FailedRuntime:
        ready=True
        def submit(self, options):
            j=Job(options);j.started.set();j.future=Future();j.future.set_exception(error)
            j.events.put(('error',public_error(error)))
            return j
        def cancel_job(self,job):pass
        def expire_queued(self,job):return False
    with TestClient(create_app(FailedRuntime(),'test')) as client:
        r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer test'},
                      json=body(tools=TOOLS,stream=stream))
    if stream:
        assert r.status_code==200 and r.text.endswith('data: [DONE]\n\n')
        events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: {')]
        errors=[e['error'] for e in events if 'error' in e]
        assert len(errors)==1
        e=errors[0]
        assert not any(c.get('finish_reason') for event in events for c in event.get('choices',[]))
    else:
        assert r.status_code==502
        e=r.json()['error']
    assert e==public_error(error)


def test_four_batched_lanes_tool_failure_cancel_and_replacement_release_private_leases(setup_api):
    f,make=setup_api;r=make(108);owner=f.hold_owner(r)
    options=[f.options(name,24,True) for name in 'ABCD']
    options[0]['tools']=TOOLS
    first=greedy_gold(f.ids['A'],24)[0]
    r.tokenizer.decode=lambda ids,**kw: call(f'<parameter=unknown>{SECRET}</parameter>') \
        if ids and ids[0]==first else ','.join(map(str,ids))
    jobs=[r.submit(o) for o in options];f.prepared(jobs)
    fifth=r.submit(f.options('E',24,True));f.prepared([fifth])
    entered=threading.Event();release=f.gate();verify=r.batch_runner.batch_target.verify;once=[]
    def held(ids,states):
        outputs=verify(ids,states)
        if len(ids)==4 and not once:
            once.append(True);entered.set()
            if not release.wait(5):raise RuntimeError('Regression batch gate timeout')
        return outputs
    r.batch_runner.batch_target.verify=held
    owner.set();assert entered.wait(3)
    assert r.context_pool.snapshot()['reserved_tokens']==108
    assert r.cancel_job(jobs[1])
    assert r.context_pool.snapshot()['reserved_tokens']==108
    release.set()
    with pytest.raises(ToolCallError):jobs[0].future.result(5)
    assert jobs[1].future.result(5) is None
    for name,job in zip('CDE',[jobs[2],jobs[3],fifth]):
        out=job.future.result(5)
        assert out['choices'][0]['message']['content']==','.join(map(str,greedy_gold(f.ids[name],24)))
        assert SECRET not in json.dumps(out)
    errors=[v for kind,v in list(jobs[0].events.queue) if kind=='error']
    assert len(errors)==1 and errors[0]['code']=='invalid_tool_call'
    assert r.ready and not r.fatal and r.pending==0
    pool=r.context_pool.snapshot()
    assert pool['reserved_tokens']==0 and pool['reservation_count']==0
    assert pool['peak_reserved_tokens']<=108
    audits=[json.loads(line) for line in r.audit.read_text().splitlines()]
    failed=next(a for a in audits if a['id']==jobs[0].id)
    assert failed['error']==dict(code='invalid_tool_call',tool='search',param='arguments.unknown')
    assert SECRET not in r.audit.read_text()
    r.tokenizer.decode=lambda ids,**kw: ','.join(map(str,ids))
    assert r.submit(f.options('A',5,False)).future.result(5)
