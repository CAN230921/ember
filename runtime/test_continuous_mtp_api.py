"""Real API/scheduler wiring with CPU MTP state machines, not model quality."""
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
from api_server import Runtime,create_app,normalize_request,MODEL
from engine.speculative_policy import CostAwareDraftDepth
from test_continuous_mtp import make_runner,greedy_gold


class ContinuousMTPApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,SPARK_API_AUDIT=str(Path(self.tmp.name)/'audit.jsonl'),
                            SPARK_MTP_POLICY='acceptance')
        self.env.start()
        self.runtime=Runtime(mode='continuous_mtp')
        self.runtime.ready=True;self.runtime.eos={255}
        self.runtime.tokenizer=SimpleNamespace(decode=lambda ids,**kw:','.join(map(str,ids)))
        self.runtime.batch_runner=make_runner(self.runtime.prefix_cache)
        self.ids={'A':[1,2,3],'B':[9,8],'C':[4,5,6,7],'long':[3]*2050}
        self.tokenize=patch('api_server.chat_token_ids',side_effect=lambda tok,messages,*args:
                            list(self.ids[messages[0]['content']]))
        self.tokenize.start();self.gates=[]
        self.key='sk-test-'+32*'b';self.headers={'Authorization':'Bearer '+self.key}

    def tearDown(self):
        for gate in self.gates:gate.set()
        self.runtime.close();self.tokenize.stop();self.env.stop();self.tmp.cleanup()

    def options(self,name,stream=False,budget=12):
        return normalize_request(dict(model=MODEL,messages=[dict(role='user',content=name)],
            temperature=0,max_tokens=budget,stream=stream),MODEL,4096)

    def hold_owner(self):
        entered=threading.Event();release=threading.Event();self.gates.append(release)
        def hold():
            entered.set()
            if not release.wait(10):raise RuntimeError('CPU owner gate timeout')
        self.runtime.executor.submit(hold);self.assertTrue(entered.wait(2))
        return release

    def hold_both_ready(self):
        entered=threading.Event();release=threading.Event();self.gates.append(release)
        original=self.runtime.batch_runner.decode;once=[]
        def decode(sessions):
            active=self.runtime.scheduler._active
            if len(active)==2 and all(s.session.generation.generated for s in active) and not once:
                once.append(True);entered.set()
                if not release.wait(10):raise RuntimeError('CPU MTP gate timeout')
            original(sessions)
        self.runtime.batch_runner.decode=decode
        return entered,release

    def test_two_mtp_streams_cancel_and_replacement_keep_private_outputs(self):
        owner=self.hold_owner();entered,release=self.hold_both_ready()
        a=self.runtime.submit(self.options('A',True));b=self.runtime.submit(self.options('B',True))
        c=self.runtime.submit(self.options('C',True,budget=5));owner.set()
        self.assertTrue(entered.wait(3))
        self.assertEqual(set(self.runtime.scheduler.snapshot()['active_ids']),{a.id,b.id})
        for job in (a,b):
            with job.events.mutex:
                self.assertTrue(any(kind=='delta' and value.get('content') for kind,value in job.events.queue))
        self.runtime.cancel_job(a);release.set()
        self.assertIsNone(a.future.result(3))
        for name,job,budget in (('B',b,12),('C',c,5)):
            result=job.future.result(3)
            self.assertEqual(result['id'],job.id)
            self.assertEqual(result['choices'][0]['message']['content'],
                             ','.join(map(str,greedy_gold(self.ids[name],budget))))
        trace=self.runtime.scheduler.snapshot()['events']
        self.assertTrue(any(e['event']=='mtp_decode' and len(e['active_ids'])==2 for e in trace))
        self.assertFalse(any(e['event']=='batch_decode' for e in trace))
        self.assertEqual(self.runtime.pending,0)
        rows=[json.loads(line) for line in self.runtime.audit.read_text().splitlines()]
        completed=[row for row in rows if row['status']=='completed']
        self.assertTrue(all(row['speculative']['enabled'] for row in completed))
        self.assertTrue(all(row['speculative']['proposed']>0 for row in completed))
        self.assertTrue(all(not row['speculative']['batched_verification'] for row in completed))

    def test_http_metadata_stream_and_positive_mtp_cache_reuse(self):
        with TestClient(create_app(self.runtime,self.key)) as client:
            status=client.get('/v1/scheduler',headers=self.headers).json()
            self.assertEqual(status['execution_mode'],'continuous_mtp')
            self.assertTrue(status['mtp_enabled']);self.assertTrue(status['experimental'])
            self.assertFalse(status['decode_together']);self.assertFalse(status['batched_verification'])
            body=dict(model=MODEL,messages=[dict(role='user',content='long')],max_tokens=8)
            cold=client.post('/v1/chat/completions',headers=self.headers,json=body)
            warm=client.post('/v1/chat/completions',headers=self.headers,json=body)
            self.assertEqual((cold.status_code,warm.status_code),(200,200))
            self.assertEqual(cold.json()['usage']['prompt_tokens_details']['cached_tokens'],0)
            self.assertEqual(warm.json()['usage']['prompt_tokens_details']['cached_tokens'],2048)
            self.assertEqual(cold.json()['choices'],warm.json()['choices'])
            response=client.post('/v1/chat/completions',headers=self.headers,json={**body,'stream':True})
            self.assertEqual(response.status_code,200)
            events=[line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual(events[-1],'[DONE]')
            chunks=[json.loads(line) for line in events[:-1]]
            usages=[c['usage'] for c in chunks if c.get('usage')]
            self.assertEqual(usages[-1]['prompt_tokens_details']['cached_tokens'],2048)

    def test_queued_timeout_and_overload_preserve_both_active_mtp_requests(self):
        owner=self.hold_owner();entered,release=self.hold_both_ready()
        a=self.runtime.submit(self.options('A'));b=self.runtime.submit(self.options('B'));owner.set()
        self.assertTrue(entered.wait(3))
        queued=self.runtime.submit(self.options('C'))
        with self.assertRaises(HTTPException) as caught:self.runtime.submit(self.options('C'))
        self.assertEqual(caught.exception.status_code,429)
        self.runtime.cancel_job(queued)
        self.assertTrue(queued.future.cancelled());self.assertEqual(queued.options,{})
        self.runtime.queue_timeout=.01
        with TestClient(create_app(self.runtime,self.key)) as client:
            for stream in (False,True):
                response=client.post('/v1/chat/completions',headers=self.headers,
                    json=dict(model=MODEL,messages=[dict(role='user',content='C')],max_tokens=8,stream=stream))
                self.assertEqual(response.status_code,503)
                self.assertEqual(response.headers['retry-after'],'1')
        self.assertFalse(a.future.done());self.assertFalse(b.future.done())
        release.set();a.future.result(3);b.future.result(3)
        self.assertEqual(self.runtime.pending,0)

    def test_cancel_inside_paired_prefill_reports_completed_checkpoint(self):
        owner=self.hold_owner();jobs=[];cancelled=[]
        original=self.runtime.batch_runner.draft
        def draft(ids,hidden,state=None,**kwargs):
            result=original(ids,hidden,state,**kwargs)
            if kwargs.get('prefill_chunk') and len(ids)==2048 and not cancelled:
                cancelled.append(True);self.runtime.cancel_job(jobs[0])
            return result
        self.runtime.batch_runner.draft=draft
        job=self.runtime.submit(self.options('long'));jobs.append(job);owner.set()
        self.assertIsNone(job.future.result(3))
        row=json.loads(self.runtime.audit.read_text().splitlines()[-1])
        self.assertEqual(row['status'],'cancelled')
        self.assertEqual(row['phase'],'prefill')
        self.assertEqual(row['completed_prefill_tokens'],2048)
        self.assertEqual(row['checkpoint_tokens'],2048)
        self.assertGreater(row['cache_retained_bytes'],0)
        recovered=self.runtime.submit(self.options('long')).future.result(3)
        self.assertEqual(recovered['usage']['prompt_tokens_details']['cached_tokens'],2048)

    def test_cost_policy_is_opt_in_and_does_not_change_default_mode(self):
        default=Runtime()
        try:self.assertEqual(default.mode,'serial_mtp')
        finally:default.close()
        with patch.dict(os.environ,SPARK_MTP_POLICY='cost'):
            with self.assertRaisesRegex(ValueError,'Cost policy requires'):Runtime(mode='serial_mtp')
            candidate=Runtime(mode='continuous_mtp')
            try:self.assertEqual(candidate.mtp_policy,'cost')
            finally:candidate.close()
        self.runtime.batch_runner=make_runner(self.runtime.prefix_cache,policy_factory=CostAwareDraftDepth)
        response=self.runtime.submit(self.options('A')).future.result(3)
        self.assertEqual(response['choices'][0]['message']['content'],
                         ','.join(map(str,greedy_gold(self.ids['A'],12))))


if __name__=='__main__':unittest.main()
