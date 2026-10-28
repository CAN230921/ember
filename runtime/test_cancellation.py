"""Cancellation must not publish half a target/draft checkpoint or keep a queue slot."""
import json
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api_server import create_app,QueueDeadlineExceeded,MODEL
from engine.prefix_cache import PrefixCache
from engine.request_control import GenerationCancelled
from engine.speculative import generate_mtp
from test_api_runtime import RuntimeTests
from test_speculative_oracle import Target,DepthTwoDraft


class QueueCancellationTests(RuntimeTests):
    def test_cancel_queued_releases_capacity_before_worker_finishes(self):
        runtime=self.runtime;runtime.ready=True
        entered=threading.Event();release=threading.Event();calls=[]
        def work(job):
            calls.append(job.id);job.executing.set();entered.set()
            try:release.wait(5)
            finally:runtime._release(job)
        with patch.object(runtime,'run',work):
            first=runtime.submit(self.options)
            try:
                self.assertTrue(entered.wait(1))
                second=runtime.submit(self.options)
                self.assertEqual(runtime.pending,2)
                self.assertTrue(runtime.cancel_job(second))
                self.assertTrue(second.future.cancelled())
                self.assertEqual(runtime.pending,1)
                self.assertFalse(runtime.cancel_job(second))
                third=runtime.submit(self.options)
                self.assertEqual(runtime.pending,2)
                runtime.cancel_job(third)
                self.assertEqual(calls,[first.id])
                self.assertEqual(runtime.pending,1)
                records=[json.loads(line) for line in runtime.audit.read_text().splitlines()]
                self.assertEqual([row['status'] for row in records],['cancelled_queued']*2)
                self.assertTrue(all(row['input_tokens']==0 for row in records))
            finally:release.set();first.future.result(2)
        self.assertEqual(runtime.pending,0)

    def test_queue_deadline_returns_503_for_both_http_modes(self):
        runtime=self.runtime;runtime.ready=True;runtime.queue_timeout=.02
        entered=threading.Event();release=threading.Event()
        def work(job):
            job.executing.set();entered.set()
            try:release.wait(5)
            finally:runtime._release(job)
        key='sk-test-'+32*'a'
        with patch.object(runtime,'run',work),TestClient(create_app(runtime,key)) as client:
            first=runtime.submit(self.options)
            try:
                self.assertTrue(entered.wait(1))
                first.queue_deadline=time.perf_counter()-1
                self.assertFalse(runtime.expire_queued(first))
                for stream in (False,True):
                    response=client.post('/v1/chat/completions',headers={'Authorization':'Bearer '+key},
                        json=dict(model=MODEL,messages=[dict(role='user',content='hello')],temperature=0,stream=stream))
                    self.assertEqual(response.status_code,503,response.text)
                    self.assertEqual(response.headers['retry-after'],'1')
                    self.assertEqual(runtime.pending,1)
                records=[json.loads(line) for line in runtime.audit.read_text().splitlines()]
                self.assertEqual([row['status'] for row in records],['queue_timeout']*2)
            finally:release.set();first.future.result(2)

    def test_worker_rechecks_deadline_before_tokenization(self):
        from api_server import Job
        job=Job(self.options);job.queue_deadline=time.perf_counter()-1;self.runtime.pending=1
        with patch('api_server.chat_token_ids') as tokenize:
            self.assertIsNone(self.runtime.run(job))
            tokenize.assert_not_called()
        self.assertIsInstance(job.start_error,QueueDeadlineExceeded)
        self.assertEqual(self.runtime.pending,0)
        self.assertEqual(json.loads(self.runtime.audit.read_text())['status'],'queue_timeout')


class PrefillCancellationTests(unittest.TestCase):
    def test_each_boundary_retains_only_complete_pairs_and_resumes_exactly(self):
        boundaries=[('prefill_start',0,0),('prefill_target',64,0),('prefill_draft',64,0),
            ('prefill_checkpoint',64,64),('prefill_target',128,64),('prefill_draft',128,64),
            ('prefill_checkpoint',128,128),('first_token',195,192),
            ('draft_prefill_complete',195,192),('decode',195,192),('decode_target',195,192)]
        for depth in (1,2,'adaptive'):
            for phase,completed,retained in boundaries:
                with self.subTest(depth=depth,phase=phase,completed=completed):
                    target=Target();target.prompt=[7,8]*97+[7]
                    draft=DepthTwoDraft(target,'accept');cache=PrefixCache();cancel=threading.Event()
                    def progress(current,count,checkpoint):
                        if (current,count)==(phase,completed):cancel.set()
                    with self.assertRaises(GenerationCancelled) as caught:
                        generate_mtp(target,draft,target.prompt,12,{19},64,low_sync=True,
                            prefix_cache=cache,draft_tokens=depth,should_cancel=cancel.is_set,on_progress=progress)
                    self.assertEqual(caught.exception.phase,phase)
                    self.assertEqual(caught.exception.checkpoint_tokens,retained)
                    self.assertEqual(cache.entries[-1].end if cache.entries else 0,retained)
                    for entry in cache.entries:
                        self.assertEqual(entry.target.tokens,target.prompt[:entry.end])
                        self.assertEqual(entry.draft.length,entry.end)
                        self.assertFalse(entry.draft.temporary)
                    row,state=generate_mtp(target,DepthTwoDraft(target,'accept'),target.prompt,12,{19},64,
                        low_sync=True,prefix_cache=cache,draft_tokens=depth)
                    self.assertEqual(row['generated_ids'],target.gold)
                    self.assertEqual(state.tokens,target.prompt+target.gold[:-1])
                    self.assertEqual(row['prefix_cache']['reused_tokens'],retained)

    def test_first_token_callback_cancel_preserves_complete_prompt_cache(self):
        target=Target();target.prompt=[7,8]*65+[7];cache=PrefixCache();cancel=threading.Event();seen=[]
        def callback(ids):seen.append(ids);cancel.set()
        with self.assertRaises(GenerationCancelled) as caught:
            generate_mtp(target,DepthTwoDraft(target,'accept'),target.prompt,12,{19},64,
                prefix_cache=cache,draft_tokens='adaptive',on_tokens=callback,should_cancel=cancel.is_set)
        self.assertEqual(caught.exception.phase,'draft_prefill_start')
        self.assertEqual(seen,[(11,)])
        self.assertEqual(cache.entries[-1].end,128)


if __name__=='__main__':unittest.main()
