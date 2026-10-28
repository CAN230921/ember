"""Queue/visible-output telemetry, cache usage, and cancellation recovery."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api_server import Job,Runtime,normalize_request,MODEL


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        with patch.dict(os.environ,SPARK_API_AUDIT=str(Path(self.tmp.name)/'requests.jsonl')):
            self.runtime=Runtime()
        self.runtime.audit=Path(self.tmp.name)/'requests.jsonl'
        self.runtime.eos={9};self.runtime.tokenizer=SimpleNamespace(decode=lambda *a,**kw:'42')
        self.options=normalize_request(dict(model=MODEL,messages=[dict(role='user',content='计算')],
            temperature=0,max_tokens=8,stream=True),MODEL,4096)

    def tearDown(self):
        self.runtime.executor.shutdown(wait=True);self.tmp.cleanup()

    def test_queue_and_cached_usage_are_separate_from_engine_time(self):
        job=Job(self.options);job.received-=.25;self.runtime.pending=1
        row=dict(generated_ids=[1],output_tokens=1,finish_reason='stop',first_token_seconds=.02,
            decode_tokens_per_second=None,request_seconds=.04,speculative={},prefill_chunks=[],
            target_prefill_seconds=.01,draft_prefill_before_first_seconds=0.,draft_prefill_after_first_seconds=0.,
            prefix_cache=dict(reused_tokens=64,uncached_tokens=36,checkpoint_seconds=.001,retained_bytes=1024))
        def generate(ids,budget,callback,**controls):
            callback((1,));return row,None
        with patch('api_server.chat_token_ids',return_value=list(range(100))),patch.object(self.runtime,'generate',generate):
            response=self.runtime.run(job)
        audit=json.loads(self.runtime.audit.read_text())
        self.assertEqual(response['usage']['prompt_tokens_details']['cached_tokens'],64)
        self.assertGreaterEqual(audit['queue_seconds'],.25)
        self.assertGreaterEqual(audit['first_output_seconds'],audit['queue_seconds'])
        self.assertGreaterEqual(audit['server_seconds'],audit['first_output_seconds'])
        self.assertEqual(audit['request_seconds'],.04)
        self.assertEqual(audit['status'],'completed');self.assertEqual(self.runtime.pending,0)

    def test_cancelled_generation_releases_slot_and_records_partial_tokens(self):
        job=Job(self.options);self.runtime.pending=1
        def generate(ids,budget,callback,**controls):
            callback((1,));job.cancel.set();callback((1,2))
        with patch('api_server.chat_token_ids',return_value=[3,4]),patch.object(self.runtime,'generate',generate):
            self.assertIsNone(self.runtime.run(job))
        audit=json.loads(self.runtime.audit.read_text())
        self.assertEqual(audit['status'],'cancelled');self.assertEqual(audit['output_tokens'],2)
        self.assertEqual(self.runtime.pending,0);self.assertTrue(job.started.is_set())


if __name__=='__main__':unittest.main()
