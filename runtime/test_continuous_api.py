"""Real scheduler/API with a CPU toy model, deterministic thread/ASGI gates."""
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
from api_server import Runtime,create_app,normalize_request,MODEL,MAX_CONTEXT
from test_continuous_engine import make_runner


class ContinuousApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        with patch.dict(os.environ,SPARK_API_AUDIT=str(Path(self.tmp.name)/'audit.jsonl')):
            self.runtime=Runtime(mode='continuous_greedy')
        self.runtime.ready=True;self.runtime.eos={63}
        self.runtime.tokenizer=SimpleNamespace(decode=lambda ids,**kw:','.join(map(str,ids)))
        self.runtime.batch_runner=make_runner(self.runtime.prefix_cache)
        self.ids={'A':[1,2,3],'B':[9,8],'C':[4,5,6,7],
                  'warm':[3]*2050,'cold':[4]*2050,'full':[5]*(MAX_CONTEXT-8)}
        self.tokenize=patch('api_server.chat_token_ids',side_effect=lambda tokenizer,messages,*args:
                            list(self.ids[messages[0]['content']]))
        self.tokenize.start()
        self.key='sk-test-'+32*'a';self.headers={'Authorization':'Bearer '+self.key}
        self.gates=[]

    def tearDown(self):
        for gate in self.gates:gate.set()
        self.runtime.close();self.tokenize.stop();self.tmp.cleanup()

    def options(self,name,stream=False,budget=8):
        return normalize_request(dict(model=MODEL,messages=[dict(role='user',content=name)],
            temperature=0,max_tokens=budget,stream=stream),MODEL,4096)

    def hold_owner(self):
        entered=threading.Event();release=threading.Event();self.gates.append(release)
        def work():
            entered.set()
            if not release.wait(10):raise RuntimeError('CPU test owner gate timed out')
        self.runtime.executor.submit(work);self.assertTrue(entered.wait(2))
        return release

    def hold_batch(self):
        entered=threading.Event();release=threading.Event();self.gates.append(release)
        original=self.runtime.batch_runner.decode
        once=[]
        def decode(sessions):
            if len(sessions)==2 and not once:
                once.append(True);entered.set()
                if not release.wait(10):raise RuntimeError('CPU test batch gate timed out')
            original(sessions)
        self.runtime.batch_runner.decode=decode
        return entered,release

    def test_both_active_emit_and_cancel_a_leaves_b_then_c_reuses_slot(self):
        owner=self.hold_owner();entered,release=self.hold_batch()
        a=self.runtime.submit(self.options('A',True));b=self.runtime.submit(self.options('B',True))
        c=self.runtime.submit(self.options('C',True,budget=3));owner.set()
        self.assertTrue(entered.wait(2))
        self.assertEqual(set(self.runtime.scheduler.snapshot()['active_ids']),{a.id,b.id})
        for job in (a,b):
            self.assertFalse(job.future.done())
            with job.events.mutex:self.assertTrue(any(k=='delta' and v.get('content') for k,v in job.events.queue))
        self.assertTrue(self.runtime.cancel_job(a));release.set()
        self.assertIsNone(a.future.result(3))
        rb=b.future.result(3);rc=c.future.result(3)
        self.assertEqual(rb['id'],b.id);self.assertEqual(rc['id'],c.id)
        gold=self.runtime.submit(self.options('B')).future.result(3)
        self.assertEqual(rb['choices'],gold['choices'])
        pairs=[e['request_ids'] for e in self.runtime.scheduler.snapshot()['events'] if e['event']=='batch_decode']
        self.assertIn([a.id,b.id],pairs);self.assertIn([b.id,c.id],pairs)
        self.assertEqual(self.runtime.pending,0)
        self.assertEqual(a.options,{})

    def test_full_queue_timeout_for_both_http_modes_and_immediate_payload_release(self):
        owner=self.hold_owner();entered,release=self.hold_batch()
        a=self.runtime.submit(self.options('A'));b=self.runtime.submit(self.options('B'));owner.set()
        self.assertTrue(entered.wait(2))
        queued=self.runtime.submit(self.options('C'))
        with self.assertRaises(HTTPException) as caught:self.runtime.submit(self.options('C'))
        self.assertEqual(caught.exception.status_code,429)
        self.assertEqual(caught.exception.headers['Retry-After'],'1')
        self.runtime.cancel_job(queued)
        self.assertTrue(queued.future.cancelled());self.assertEqual(queued.options,{})
        self.assertEqual(self.runtime.pending,2)
        self.runtime.queue_timeout=.02
        with TestClient(create_app(self.runtime,self.key)) as client:
            for stream in (False,True):
                response=client.post('/v1/chat/completions',headers=self.headers,json=dict(
                    model=MODEL,messages=[dict(role='user',content='C')],temperature=0,stream=stream))
                self.assertEqual(response.status_code,503,response.text)
                self.assertEqual(response.headers['retry-after'],'1')
                self.assertEqual(self.runtime.pending,2)
        release.set();a.future.result(3);b.future.result(3)

    def test_cached_usage_is_per_request_in_response_and_unconditional_sse_usage(self):
        self.runtime.submit(self.options('warm',budget=2)).future.result(3)
        owner=self.hold_owner()
        a=self.runtime.submit(self.options('warm',True,budget=2))
        b=self.runtime.submit(self.options('cold',True,budget=2));owner.set()
        for job,cached in ((a,2048),(b,0)):
            response=job.future.result(3)
            self.assertEqual(response['usage']['prompt_tokens_details']['cached_tokens'],cached)
            events=[]
            while not job.events.empty():events.append(job.events.get_nowait())
            done=[v for k,v in events if k=='done']
            self.assertEqual(len(done),1)
            self.assertEqual(done[0]['usage']['prompt_tokens_details']['cached_tokens'],cached)
        with TestClient(create_app(self.runtime,self.key)) as client:
            self.assertEqual(client.get('/v1/scheduler').status_code,401)
            status=client.get('/v1/scheduler',headers=self.headers).json()
        self.assertEqual(status['execution_mode'],'continuous_greedy')
        self.assertFalse(status['mtp_enabled']);self.assertEqual(status['max_active'],2)

    def test_each_request_has_full_context_budget_and_rejects_one_token_over(self):
        owner=self.hold_owner()
        a=self.runtime.submit(self.options('full'));b=self.runtime.submit(self.options('full'));owner.set()
        for job in (a,b):
            response=job.future.result(4)
            self.assertEqual(response['usage']['prompt_tokens'],MAX_CONTEXT-8)
            self.assertEqual(response['usage']['completion_tokens'],8)
        with TestClient(create_app(self.runtime,self.key)) as client:
            response=client.post('/v1/chat/completions',headers=self.headers,json=dict(model=MODEL,
                messages=[dict(role='user',content='full')],max_tokens=9,temperature=0))
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.runtime.pending,0)

    def test_asgi_sse_bodies_overlap_and_keep_ids_and_usage_separate(self):
        self._asgi_overlap(False)

    def test_asgi_disconnect_cancels_only_one_of_two_active_requests(self):
        self._asgi_overlap(True)

    def _asgi_overlap(self,cancel_first):
        # TestClient buffers response.text. Explicit ASGI send observes body
        # chunks while the owner thread is still blocked in a shared step.
        owner=self.hold_owner();entered,release=self.hold_batch()
        app=create_app(self.runtime,self.key)
        async def scenario():
            contents=[asyncio.Event(),asyncio.Event()];messages=[[],[]]
            disconnected=[asyncio.Event(),asyncio.Event()]
            async def request(index,name):
                sent=False
                body=json.dumps(dict(model=MODEL,messages=[dict(role='user',content=name)],
                    temperature=0,max_tokens=8,stream=True,stream_options=dict(include_usage=False))).encode()
                async def receive():
                    nonlocal sent
                    if not sent:
                        sent=True;return dict(type='http.request',body=body,more_body=False)
                    await disconnected[index].wait()
                    return dict(type='http.disconnect')
                async def send(message):
                    messages[index].append(message)
                    if message['type']=='http.response.body':
                        for line in message.get('body',b'').decode().splitlines():
                            if line.startswith('data: {'):
                                chunk=json.loads(line[6:])
                                if any(c.get('delta',{}).get('content') for c in chunk.get('choices',[])):
                                    contents[index].set()
                scope=dict(type='http',asgi=dict(version='3.0',spec_version='2.4'),method='POST',
                    scheme='http',path='/v1/chat/completions',raw_path=b'/v1/chat/completions',
                    query_string=b'',root_path='',http_version='1.1',
                    headers=[(b'content-type',b'application/json'),(b'authorization',('Bearer '+self.key).encode())],
                    server=('localhost',80),client=('test',1000+index))
                await app(scope,receive,send)
            tasks=[asyncio.create_task(request(i,n)) for i,n in enumerate(('A','B'))]
            # Both ASGI handlers enqueue before releasing the model thread.
            await asyncio.sleep(0)
            owner.set()
            self.assertTrue(await asyncio.to_thread(entered.wait,2))
            try:
                await asyncio.wait_for(asyncio.gather(*(e.wait() for e in contents)),2)
                self.assertTrue(all(not t.done() for t in tasks))
                if cancel_first:
                    disconnected[0].set()
                    await asyncio.wait_for(tasks[0],2)
                    self.assertFalse(tasks[1].done())
            finally:release.set()
            await asyncio.wait_for(asyncio.gather(*tasks),3)
            ids=[]
            for index,output in enumerate(messages):
                raw=b''.join(m.get('body',b'') for m in output if m['type']=='http.response.body').decode()
                chunks=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                request_ids={chunk['id'] for chunk in chunks}
                self.assertEqual(len(request_ids),1);ids.extend(request_ids)
                usage=[c['usage'] for c in chunks if 'usage' in c]
                if cancel_first and index==0:
                    self.assertEqual(usage,[])
                    self.assertFalse(raw.endswith('data: [DONE]\n\n'))
                else:
                    self.assertEqual(len(usage),1)
                    self.assertEqual(usage[0]['prompt_tokens_details']['cached_tokens'],0)
                    self.assertTrue(raw.endswith('data: [DONE]\n\n'))
            self.assertNotEqual(ids[0],ids[1])
            if cancel_first:
                cancels=[e['request_ids'] for e in self.runtime.scheduler.snapshot()['events'] if e['event']=='cancel']
                self.assertIn([ids[0]],cancels);self.assertNotIn([ids[1]],cancels)
        asyncio.run(scenario())

    def test_factory_failure_releases_api_capacity_and_next_request_runs(self):
        factory=self.runtime.scheduler.factory
        def fail(job):raise ValueError('CPU factory failure')
        self.runtime.scheduler.factory=fail
        job=self.runtime.submit(self.options('A'))
        with self.assertRaisesRegex(ValueError,'CPU factory'):job.future.result(2)
        self.assertTrue(job.started.is_set());self.assertEqual(job.options,{})
        self.assertEqual(self.runtime.pending,0)
        self.runtime.scheduler.factory=factory
        self.assertEqual(self.runtime.submit(self.options('B')).future.result(2)['usage']['completion_tokens'],8)

    def test_prefill_cancel_retains_only_complete_checkpoint_for_survivor(self):
        owner=self.hold_owner();original=self.runtime.batch_runner.prefill
        jobs={};cancelled=[]
        def prefill(session):
            original(session)
            if not cancelled:
                cancelled.append(True);self.runtime.cancel_job(jobs['A'])
        self.runtime.batch_runner.prefill=prefill
        jobs['A']=self.runtime.submit(self.options('warm'))
        b=self.runtime.submit(self.options('warm'));owner.set()
        self.assertIsNone(jobs['A'].future.result(3))
        response=b.future.result(3)
        # B checked out before A's first completed chunk was published. It
        # must report zero rather than claim a cache hit discovered later.
        self.assertEqual(response['usage']['prompt_tokens_details']['cached_tokens'],0)
        audit=[json.loads(line) for line in self.runtime.audit.read_text().splitlines()]
        cancelled_row=next(row for row in audit if row['id']==jobs['A'].id)
        self.assertEqual(cancelled_row['completed_prefill_tokens'],2048)
        self.assertEqual(cancelled_row['checkpoint_tokens'],2048)
        self.assertEqual(cancelled_row['output_tokens'],0)
        warm=self.runtime.submit(self.options('warm')).future.result(3)
        self.assertEqual(warm['usage']['prompt_tokens_details']['cached_tokens'],2048)
        self.assertEqual(response['choices'],warm['choices'])
        self.runtime.executor.submit(self.runtime.prefix_cache.clear).result(2)
        cold=self.runtime.submit(self.options('warm')).future.result(3)
        self.assertEqual(response['choices'],cold['choices'])

    def test_first_token_eos_frees_slot_while_survivors_continue(self):
        self.runtime.eos={15}  # Toy A's first argmax; B's first is 26.
        owner=self.hold_owner()
        a=self.runtime.submit(self.options('A'));b=self.runtime.submit(self.options('B'))
        c=self.runtime.submit(self.options('C'));owner.set()
        response=a.future.result(3)
        self.assertEqual(response['usage']['completion_tokens'],1)
        self.assertEqual(response['choices'][0]['finish_reason'],'stop')
        self.assertGreater(b.future.result(3)['usage']['completion_tokens'],1)
        c.future.result(3)
        pairs=[e['request_ids'] for e in self.runtime.scheduler.snapshot()['events'] if e['event']=='batch_decode']
        self.assertIn([b.id,c.id],pairs)


if __name__=='__main__':unittest.main()
