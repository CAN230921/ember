"""Real scheduler/API contracts with CPU batch-model ownership fixtures."""
from types import SimpleNamespace
import json
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api_server import Runtime,create_app
import test_four_context_api as api_fixtures
from test_batched_mtp import runner,prefilled,finish_all
from test_continuous_mtp import greedy_gold


@pytest.fixture
def setup_api():
    fixture=api_fixtures.FourContextApiTests(methodName='runTest');fixture.setUp()
    def make(total=1048576):
        runtime=Runtime(mode='continuous_mtp',max_active=4,total_context=total,mtp_batch=True)
        fixture.runtimes.append(runtime);runtime.ready=True;runtime.eos={255}
        runtime.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:','.join(map(str,ids)))
        runtime.batch_runner=runner();return runtime
    yield fixture,make
    fixture.tearDown()


def test_four_json_lanes_are_batched_and_report_truthful_capabilities(setup_api):
    f,make=setup_api;r=make();owner=f.hold_owner(r)
    jobs=[r.submit(f.options(name,32,False)) for name in 'ABCD'];f.prepared(jobs);owner.set()
    outputs=[job.future.result(5) for job in jobs]
    for name,out in zip('ABCD',outputs):
        assert out['choices'][0]['message']['content']==','.join(map(str,greedy_gold(f.ids[name],32)))
        assert out['usage']['total_tokens']==35
    with TestClient(create_app(r,f.key)) as client:
        assert client.get('/v1/scheduler').status_code==401
        s=client.get('/v1/scheduler',headers=f.headers).json()
    assert s['batched_verification'] and s['mtp_batch'] and s['scheduling_policy']=='decode_priority'
    assert not s['decode_together'] and s['context_pool']['reserved_tokens']==0
    assert any(len(call)==4 for call in r.batch_runner.batch_target.calls)
    assert any(e['event']=='mtp_batch_decode' and len(e['request_ids'])==4 for e in s['events'])
    audit=[json.loads(line) for line in r.audit.read_text().splitlines()]
    assert any(a['speculative']['batched_verification'] for a in audit)


@pytest.mark.parametrize('cancelled',[0,1,2,3])
def test_cancel_during_shared_gpu_step_keeps_lease_until_owner_cleanup_and_admits_replacement(setup_api,cancelled):
    f,make=setup_api;r=make(108);owner=f.hold_owner(r)
    jobs=[r.submit(f.options(name,24,False)) for name in 'ABCD'];f.prepared(jobs)
    fifth=r.submit(f.options('E',24,False));f.prepared([fifth])
    entered=threading.Event();release=f.gate();verify=r.batch_runner.batch_target.verify;once=[]
    def held(ids,states):
        outputs=verify(ids,states)
        if len(ids)==4 and not once:
            once.append(True);entered.set()
            if not release.wait(5):raise RuntimeError('Batch owner gate timeout')
        return outputs
    r.batch_runner.batch_target.verify=held;owner.set();assert entered.wait(3)
    assert r.context_pool.snapshot()['reserved_tokens']==108
    assert r.cancel_job(jobs[cancelled])
    assert r.context_pool.snapshot()['reserved_tokens']==108
    assert r.pending==5
    release.set()
    outputs=[job.future.result(5) for job in jobs]+[fifth.future.result(5)]
    for i,(name,out) in enumerate(zip('ABCDE',outputs)):
        if i==cancelled:assert out is None
        else:assert out['choices'][0]['message']['content']==','.join(map(str,greedy_gold(f.ids[name],24)))
    assert r.pending==0 and r.context_pool.snapshot()['reserved_tokens']==0
    assert r.context_pool.snapshot()['peak_reserved_tokens']<=108


def test_variable_context_contracts_remain_within_shared_pool(setup_api):
    f,make=setup_api;r=make(60);owner=f.hold_owner(r);budgets=[24,40,15,10]
    jobs=[r.submit(f.options(name,cap,False)) for name,cap in zip('ABCD',budgets)]
    f.prepared(jobs);owner.set()
    for name,cap,job in zip('ABCD',budgets,jobs):
        out=job.future.result(5)
        assert out['choices'][0]['message']['content']==','.join(map(str,greedy_gold(f.ids[name],cap)))
    pool=r.context_pool.snapshot();assert pool['reserved_tokens']==0 and pool['peak_reserved_tokens']<=60


def test_each_reasoning_plan_controls_only_its_own_generation(setup_api):
    f,make=setup_api;r=make();prompts=[f.ids[name] for name in 'ABCD']
    sessions=prefilled(r.batch_runner,prompts,[20]*4)
    def bad():raise ValueError('Invalid private reasoning plan')
    plans=[lambda:(211,None),lambda:(None,1),lambda:(None,2),bad]
    requests=[SimpleNamespace(generation=s,reasoning=SimpleNamespace(plan=p)) for s,p in zip(sessions,plans)]
    sizes=[len(s.generated) for s in sessions];errors=r._decode_ready_batch(requests)
    assert errors[:3]==[None]*3 and isinstance(errors[3],ValueError)
    assert sessions[0].generated[1]==211
    assert [len(s.generated)-n for s,n in zip(sessions,sizes)]==[1,1,2,0]
    finish_all(r.batch_runner,sessions)
    for i in (1,2,3):assert sessions[i].generated==greedy_gold(prompts[i],20)


def test_shared_model_failure_closes_all_affected_states_and_returns_each_lease(setup_api):
    f,make=setup_api;r=make();owner=f.hold_owner(r)
    jobs=[r.submit(f.options(name,32,False)) for name in 'ABCD'];f.prepared(jobs)
    def fail(*args):raise RuntimeError('CUDA injected batch failure')
    r.batch_runner.batch_target.verify=fail;owner.set()
    for job in jobs:
        with pytest.raises(RuntimeError):job.future.result(5)
    assert r.fatal and not r.ready and r.pending==0
    assert r.context_pool.snapshot()['reserved_tokens']==0
    assert not r.scheduler.snapshot()['active_ids']


@pytest.mark.parametrize('mode',['continuous_greedy','serial_mtp'])
def test_batch_flag_cannot_enter_other_execution_modes(mode):
    with pytest.raises(ValueError,match='continuous_mtp'):Runtime(mode=mode,mtp_batch=True)


def test_batched_path_remains_explicitly_disabled_by_default(setup_api):
    f,_=setup_api
    with patch.dict('os.environ',SPARK_MTP_BATCH='0'):
        r=Runtime(mode='continuous_mtp',max_active=4);f.runtimes.append(r)
    assert not r.mtp_batch and not r.scheduler.mtp_batch


def test_four_sse_streams_emit_before_completion_and_finish_with_private_usage(setup_api):
    f,make=setup_api;r=make();owner=f.hold_owner(r);entered=threading.Event();release=f.gate()
    verify=r.batch_runner.batch_target.verify;once=[]
    def held(ids,states):
        result=verify(ids,states)
        if len(ids)==4 and not once:
            once.append(True);entered.set()
            if not release.wait(5):raise RuntimeError('Shared SSE gate timeout')
        return result
    r.batch_runner.batch_target.verify=held;app=create_app(r,f.key)
    async def scenario():
        bodies=[[] for _ in range(4)];content=[asyncio.Event() for _ in range(4)]
        async def request(index):
            sent=False;disconnected=asyncio.Event()
            body=json.dumps(dict(model='qwen3.8-flash-next-gb10',messages=[dict(role='user',content='ABCD'[index])],
                                 max_tokens=32,temperature=0,stream=True)).encode()
            async def receive():
                nonlocal sent
                if not sent:sent=True;return dict(type='http.request',body=body,more_body=False)
                await disconnected.wait();return dict(type='http.disconnect')
            async def send(message):
                if message['type']=='http.response.body':
                    raw=message.get('body',b'').decode();bodies[index].append(raw)
                    for line in raw.splitlines():
                        if line.startswith('data: {') and any(c.get('delta',{}).get('content')
                            for c in json.loads(line[6:]).get('choices',[])):content[index].set()
            scope=dict(type='http',asgi=dict(version='3.0',spec_version='2.4'),method='POST',scheme='http',
                path='/v1/chat/completions',raw_path=b'/v1/chat/completions',query_string=b'',root_path='',http_version='1.1',
                headers=[(b'content-type',b'application/json'),(b'authorization',('Bearer '+f.key).encode())],
                server=('localhost',80),client=('test',2000+index))
            await app(scope,receive,send)
        tasks=[asyncio.create_task(request(i)) for i in range(4)]
        for _ in range(100):
            if r.pending==4:break
            await asyncio.sleep(.01)
        assert r.pending==4
        preparations=[s.preparation for s in r.scheduler._waiting]
        await asyncio.gather(*(asyncio.to_thread(p.result,3) for p in preparations))
        owner.set();assert await asyncio.to_thread(entered.wait,3)
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in content)),3)
            assert all(not t.done() for t in tasks)
        finally:release.set()
        await asyncio.wait_for(asyncio.gather(*tasks),5)
        ids=[]
        for i,parts in enumerate(bodies):
            raw=''.join(parts);chunks=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            ids.append({c['id'] for c in chunks});assert len(ids[-1])==1
            usage=[c['usage'] for c in chunks if 'usage' in c]
            assert len(usage)==1 and usage[0]['total_tokens']==35
            text=''.join(c.get('delta',{}).get('content','') for chunk in chunks for c in chunk.get('choices',[]))
            assert text==','.join(map(str,greedy_gold(f.ids['ABCD'[i]],32)))
            assert raw.endswith('data: [DONE]\n\n')
        assert len(set.union(*ids))==4
    asyncio.run(scenario());assert r.context_pool.snapshot()['reserved_tokens']==0
