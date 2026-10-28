"""Four private MTP state machines and shared admission via the real API.

CPU toy models establish accounting and protocol behavior only. Real model
quality, device memory, and four simultaneous 262K SSE streams remain GPU gates.
"""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from api_server import Runtime,MODEL,normalize_request,create_app
from test_continuous_mtp import make_runner,greedy_gold
from test_context_limits import PositionTarget,NoDraft
from engine.continuous_mtp import ContinuousMTPRunner


class FourContextApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.runtimes=[];self.gates=[]
        self.env=patch.dict(os.environ,SPARK_API_AUDIT=str(Path(self.tmp.name)/'audit.jsonl'),
            SPARK_MTP_POLICY='acceptance',SPARK_MAX_CONTEXT='262144',SPARK_MAX_OUTPUT='0')
        self.env.start()
        self.ids={name:[index+1]*3 for index,name in enumerate('ABCDE')}
        self.tokenize=patch('api_server.chat_token_ids',side_effect=lambda tok,messages,*args:
            list(self.ids[messages[0]['content']]))
        self.tokenize.start()
        self.key='sk-four-test-'+32*'f';self.headers={'Authorization':'Bearer '+self.key}

    def tearDown(self):
        for gate in self.gates:gate.set()
        for runtime in self.runtimes:runtime.close()
        self.tokenize.stop();self.env.stop();self.tmp.cleanup()

    def runtime(self,total=1048576):
        runtime=Runtime(mode='continuous_mtp',max_active=4,max_queued=1,total_context=total)
        self.runtimes.append(runtime)
        runtime.ready=True;runtime.eos={255}
        runtime.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:','.join(map(str,ids)))
        runtime.batch_runner=make_runner(runtime.prefix_cache)
        return runtime

    def options(self,name,budget=12,stream=True):
        return normalize_request(dict(model=MODEL,messages=[dict(role='user',content=name)],
            max_tokens=budget,temperature=0,stream=stream),MODEL,0)

    def gate(self):
        release=threading.Event();self.gates.append(release)
        return release

    def hold_owner(self,runtime):
        entered=threading.Event();release=self.gate()
        def hold():
            entered.set()
            if not release.wait(10):raise RuntimeError('Owner gate timeout')
        runtime.executor.submit(hold);self.assertTrue(entered.wait(2))
        return release

    def prepared(self,jobs):
        for job in jobs:job.preparation.result(3)

    def test_four_mtp_outputs_cancel_replacement_and_cleanup_owns_reservation(self):
        runtime=self.runtime();owner=self.hold_owner(runtime)
        entered=threading.Event();release=self.gate();once=[]
        original=runtime.batch_runner.decode
        def decode(sessions,**kwargs):
            active=runtime.scheduler._active
            if len(active)==4 and all(slot.session.generation.generated for slot in active) and not once:
                once.append(True);entered.set()
                if not release.wait(10):raise RuntimeError('Four MTP gate timeout')
            return original(sessions,**kwargs)
        runtime.batch_runner.decode=decode
        closed=[];close=runtime.batch_runner.close
        def checked_close(generation):
            self.assertGreater(runtime.context_pool.snapshot()['reservation_count'],0)
            closed.append(id(generation));close(generation)
        runtime.batch_runner.close=checked_close
        jobs=[runtime.submit(self.options(name)) for name in 'ABCD']
        self.prepared(jobs);queued=runtime.submit(self.options('E',5))
        with self.assertRaises(HTTPException) as caught:runtime.submit(self.options('E'))
        self.assertEqual(caught.exception.status_code,429)
        owner.set();self.assertTrue(entered.wait(3))
        self.assertEqual(set(runtime.scheduler.snapshot()['active_ids']),{job.id for job in jobs})
        self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],60)
        for job in jobs:
            with job.events.mutex:
                self.assertTrue(any(kind=='delta' and value.get('content') for kind,value in job.events.queue))
        runtime.cancel_job(jobs[1]);release.set()
        self.assertIsNone(jobs[1].future.result(3))
        for name,job,budget in [('A',jobs[0],12),('C',jobs[2],12),('D',jobs[3],12),('E',queued,5)]:
            result=job.future.result(3)
            self.assertEqual(result['choices'][0]['message']['content'],
                ','.join(map(str,greedy_gold(self.ids[name],budget))))
            self.assertEqual(result['usage']['total_tokens'],3+budget)
        self.assertEqual(len(closed),5)
        self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)
        self.assertEqual(runtime.pending,0)
        self.assertFalse(any(e['event']=='batch_decode' for e in runtime.scheduler.snapshot()['events']))

    def test_mixed_lengths_admission_and_authenticated_pool_metadata(self):
        runtime=self.runtime(40);owner=self.hold_owner(runtime)
        self.ids.update(A=[1]*12,B=[2]*20,C=[3]*2,D=[4]*3)
        jobs=[runtime.submit(self.options(name,5)) for name in 'ABCD']
        self.prepared(jobs)
        entered=threading.Event();release=self.gate();prefill=runtime.batch_runner.prefill;once=[]
        def hold(session):
            if not once:
                once.append(True);entered.set()
                if not release.wait(10):raise RuntimeError('Mixed admission gate timeout')
            prefill(session)
        runtime.batch_runner.prefill=hold;owner.set();self.assertTrue(entered.wait(3))
        with TestClient(create_app(runtime,self.key)) as client:
            self.assertEqual(client.get('/v1/scheduler').status_code,401)
            status=client.get('/v1/scheduler',headers=self.headers).json()
        self.assertEqual(status['max_active'],4);self.assertEqual(status['aggregate_context_budget'],40)
        self.assertEqual(status['context_window_per_request'],262144)
        self.assertEqual(status['admission_waiting'],1);self.assertEqual(status['queued'],1)
        self.assertEqual(status['context_pool']['reserved_tokens'],32)
        self.assertEqual(status['context_pool']['available_tokens'],8)
        self.assertEqual(set(status['active_ids']),{jobs[i].id for i in (0,2,3)})
        release.set()
        for job in jobs:self.assertIsNotNone(job.future.result(3))
        self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)

    def test_oversize_shared_contract_is_http_400_and_omission_preserves_window(self):
        runtime=self.runtime(100)
        with TestClient(create_app(runtime,self.key)) as client:
            for budget in (98,None):
                body=dict(model=MODEL,messages=[dict(role='user',content='A')],temperature=0)
                if budget is not None:body['max_tokens']=budget
                response=client.post('/v1/chat/completions',headers=self.headers,json=body)
                self.assertEqual(response.status_code,400,response.text)
            response=client.post('/v1/chat/completions',headers=self.headers,json=dict(
                model=MODEL,messages=[dict(role='user',content='A')],temperature=0,max_tokens=10))
            self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(runtime.pending,0);self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)

    def test_cancel_during_cpu_tokenization_has_no_gpu_state_or_reservation(self):
        runtime=self.runtime();entered=threading.Event();release=self.gate()
        def slow(tok,messages,*args):
            name=messages[0]['content']
            if name=='B':
                entered.set()
                if not release.wait(10):raise RuntimeError('Tokenization gate timeout')
            return list(self.ids[name])
        with patch('api_server.chat_token_ids',side_effect=slow):
            b=runtime.submit(self.options('B'));self.assertTrue(entered.wait(2))
            a=runtime.submit(self.options('A',5))
            self.assertIsNotNone(a.future.result(3))
            self.assertFalse(b.executing.is_set())
            self.assertTrue(runtime.cancel_job(b));self.assertTrue(b.future.cancelled())
            self.assertEqual(b.options,{});self.assertEqual(runtime.pending,0)
            self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)
            release.set()
        self.assertIsNotNone(runtime.submit(self.options('C',5)).future.result(3))

    def test_context_wait_timeout_releases_queue_without_resizing_survivors(self):
        runtime=self.runtime(25);owner=self.hold_owner(runtime)
        entered=threading.Event();release=self.gate();prefill=runtime.batch_runner.prefill
        def hold(session):
            entered.set()
            if not release.wait(10):raise RuntimeError('Quota wait gate timeout')
            prefill(session)
        runtime.batch_runner.prefill=hold
        a=runtime.submit(self.options('A',20));self.prepared([a]);owner.set()
        self.assertTrue(entered.wait(3))
        runtime.queue_timeout=.02
        with TestClient(create_app(runtime,self.key)) as client:
            for stream in (False,True):
                response=client.post('/v1/chat/completions',headers=self.headers,json=dict(
                    model=MODEL,messages=[dict(role='user',content='B')],max_tokens=5,
                    temperature=0,stream=stream))
                self.assertEqual(response.status_code,503,response.text)
                self.assertEqual(response.headers['retry-after'],'1')
                self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],23)
        self.assertEqual(runtime.pending,1);release.set();self.assertIsNotNone(a.future.result(3))
        self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)

    def test_factory_error_releases_prepared_contract_and_recovers(self):
        runtime=self.runtime();factory=runtime.scheduler.factory
        runtime.scheduler.factory=lambda job:(_ for _ in ()).throw(ValueError('Factory failed'))
        with self.assertRaisesRegex(ValueError,'Factory failed'):
            runtime.submit(self.options('A',5)).future.result(3)
        self.assertEqual(runtime.pending,0);self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)
        runtime.scheduler.factory=factory
        self.assertIsNotNone(runtime.submit(self.options('B',5)).future.result(3))

    def test_failed_state_cleanup_makes_engine_unavailable(self):
        runtime=self.runtime()
        runtime.batch_runner.close=lambda session:(_ for _ in ()).throw(RuntimeError('Cleanup failed'))
        with self.assertRaisesRegex(RuntimeError,'Cleanup failed'):
            runtime.submit(self.options('A',5)).future.result(3)
        self.assertTrue(runtime.fatal);self.assertFalse(runtime.ready)
        with self.assertRaises(HTTPException) as caught:runtime.submit(self.options('B',5))
        self.assertEqual(caught.exception.status_code,503)

    def test_audit_preparation_failure_cleans_state_and_returns_quota(self):
        runtime=self.runtime();factory=runtime.scheduler.factory;closed=[]
        close=runtime.batch_runner.close
        def record(session):close(session);closed.append(session.closed)
        runtime.batch_runner.close=record
        def session(job):
            result=factory(job);start=result.start
            def broken():
                start()
                result.reasoning.snapshot=lambda:(_ for _ in ()).throw(ValueError('Audit failed'))
            result.start=broken
            return result
        runtime.scheduler.factory=session
        with self.assertRaisesRegex(ValueError,'Audit failed'):
            runtime.submit(self.options('A',5)).future.result(3)
        self.assertEqual(closed,[True]);self.assertEqual(runtime.pending,0)
        self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)
        runtime.scheduler.factory=factory
        self.assertIsNotNone(runtime.submit(self.options('B',5)).future.result(3))

    def test_four_asgi_sse_bodies_overlap_with_private_ids_and_usage(self):
        runtime=self.runtime();owner=self.hold_owner(runtime)
        entered=threading.Event();release=self.gate();decode=runtime.batch_runner.decode;once=[]
        def hold(sessions,**kwargs):
            active=runtime.scheduler._active
            if len(active)==4 and all(slot.session.generation.generated for slot in active) and not once:
                once.append(True);entered.set()
                if not release.wait(10):raise RuntimeError('Four SSE gate timeout')
            decode(sessions,**kwargs)
        runtime.batch_runner.decode=hold;app=create_app(runtime,self.key)
        async def scenario():
            bodies=[[] for _ in range(4)];content=[asyncio.Event() for _ in range(4)]
            async def request(index):
                sent=False;disconnected=asyncio.Event()
                body=json.dumps(dict(model=MODEL,messages=[dict(role='user',content='ABCD'[index])],
                    max_tokens=12,temperature=0,stream=True)).encode()
                async def receive():
                    nonlocal sent
                    if not sent:
                        sent=True;return dict(type='http.request',body=body,more_body=False)
                    await disconnected.wait()
                    return dict(type='http.disconnect')
                async def send(message):
                    if message['type']=='http.response.body':
                        raw=message.get('body',b'').decode();bodies[index].append(raw)
                        for line in raw.splitlines():
                            if line.startswith('data: {'):
                                chunk=json.loads(line[6:])
                                if any(c.get('delta',{}).get('content') for c in chunk.get('choices',[])):
                                    content[index].set()
                scope=dict(type='http',asgi=dict(version='3.0',spec_version='2.4'),method='POST',
                    scheme='http',path='/v1/chat/completions',raw_path=b'/v1/chat/completions',
                    query_string=b'',root_path='',http_version='1.1',headers=[(b'content-type',b'application/json'),
                    (b'authorization',('Bearer '+self.key).encode())],server=('localhost',80),client=('test',1000+index))
                await app(scope,receive,send)
            tasks=[asyncio.create_task(request(i)) for i in range(4)]
            # Preparation can finish in any order; hold the owner until all
            # handlers submitted and CPU tokenization completed.
            for _ in range(100):
                if runtime.pending==4:break
                await asyncio.sleep(.01)
            self.assertEqual(runtime.pending,4)
            preparations=[slot.preparation for slot in runtime.scheduler._waiting]
            await asyncio.gather(*(asyncio.to_thread(future.result,3) for future in preparations))
            owner.set();self.assertTrue(await asyncio.to_thread(entered.wait,3))
            try:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in content)),3)
                self.assertTrue(all(not task.done() for task in tasks))
            finally:release.set()
            await asyncio.wait_for(asyncio.gather(*tasks),4)
            ids=[]
            for index,parts in enumerate(bodies):
                raw=''.join(parts)
                chunks=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                request_ids={chunk['id'] for chunk in chunks};self.assertEqual(len(request_ids),1);ids.extend(request_ids)
                usage=[chunk['usage'] for chunk in chunks if 'usage' in chunk]
                self.assertEqual(len(usage),1);self.assertEqual(usage[0]['total_tokens'],15)
                text=''.join(c.get('delta',{}).get('content','') for chunk in chunks for c in chunk.get('choices',[]))
                self.assertEqual(text,','.join(map(str,greedy_gold(self.ids['ABCD'[index]],12))))
                self.assertTrue(raw.endswith('data: [DONE]\n\n'))
            self.assertEqual(len(set(ids)),4)
        asyncio.run(scenario())

    def test_cpu_oracle_four_complete_262144_contracts_are_not_truncated(self):
        runtime=self.runtime();owner=self.hold_owner(runtime)
        target=PositionTarget()
        runtime.batch_runner=ContinuousMTPRunner(target,NoDraft(),draft_tokens=1,low_sync=True)
        for marker,name in enumerate('ABCD',1):self.ids[name]=[marker]*262142
        jobs=[runtime.submit(self.options(name,2)) for name in 'ABCD']
        self.prepared(jobs)
        reserved=[];prefill=runtime.batch_runner.prefill
        def check(session):
            reserved.append(runtime.context_pool.snapshot()['reserved_tokens'])
            prefill(session)
        runtime.batch_runner.prefill=check;owner.set()
        for job in jobs:
            result=job.future.result(15)
            self.assertEqual(result['usage']['prompt_tokens'],262142)
            self.assertEqual(result['usage']['completion_tokens'],2)
            self.assertEqual(result['usage']['total_tokens'],262144)
        self.assertEqual(reserved[0],1048576)
        self.assertEqual({marker for marker,*rest in target.calls},{1,2,3,4})
        for marker in range(1,5):
            self.assertEqual(max(end for m,start,end,chunk in target.calls if m==marker),262143)
        self.assertEqual(runtime.context_pool.snapshot()['reserved_tokens'],0)

    def test_runner_accepts_four_separate_mtp_rounds_and_keeps_full_outputs(self):
        runner=make_runner();sessions=[runner.start(self.ids[name],12,{255},2048,cache=False) for name in 'ABCD']
        try:
            for session in sessions:runner.prefill(session)
            self.assertEqual(sum(session.phase=='decode' for session in sessions),4)
            with self.assertRaises(ValueError):runner.decode(sessions+[sessions[0]])
            runner.decode(sessions)
            while any(not session.done for session in sessions):
                runner.decode([session for session in sessions if not session.done])
            for name,session in zip('ABCD',sessions):
                self.assertEqual(session.generated,greedy_gold(self.ids[name],12))
                self.assertGreater(session.stats['proposed'],0)
        finally:
            for session in sessions:runner.close(session)


if __name__=='__main__':unittest.main()
