"""Output admission and EOS contracts with CPU doubles, not GPU capacity."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api_protocol import RequestError, normalize_request
from api_server import MODEL, Runtime, create_app
from test_continuous_mtp import make_runner, greedy_gold


class OutputBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ)
        self.env.start()
        for name in ('SPARK_MAX_OUTPUT','SPARK_MAX_CONTEXT','SPARK_EXECUTION_MODE'):
            os.environ.pop(name,None)
        os.environ['SPARK_API_AUDIT']=str(Path(self.tmp.name)/'requests.jsonl')
        os.environ['SPARK_MTP_POLICY']='acceptance'
        self.runtimes=[]
        self.key='sk-output-budget-'+32*'d'
        self.headers={'Authorization':'Bearer '+self.key}
        self.body=dict(model=MODEL,messages=[dict(role='user',content='answer')],temperature=0)

    def tearDown(self):
        for runtime in self.runtimes:runtime.close()
        self.env.stop();self.tmp.cleanup()

    def runtime(self,mode='continuous_mtp'):
        runtime=Runtime(mode=mode);self.runtimes.append(runtime)
        return runtime

    def serving_runtime(self):
        runtime=self.runtime();runtime.ready=True
        runtime.tokenizer=SimpleNamespace(decode=lambda ids,**kw:','.join(map(str,ids)))
        runtime.batch_runner=make_runner(runtime.prefix_cache)
        # EOS is the independently computed first output. Large permitted
        # budgets must still stop at EOS, rather than generating a full window.
        runtime.eos={greedy_gold([1,2,3],1)[0]}
        self.budgets=[]
        original=runtime.batch_runner.start
        def start(ids,budget,*args,**kwargs):
            self.budgets.append(budget)
            return original(ids,budget,*args,**kwargs)
        runtime.batch_runner.start=start
        return runtime

    def test_normalizer_preserves_legacy_default_and_null_rejection(self):
        self.assertEqual(normalize_request(self.body,MODEL)['max_tokens'],1024)
        for field in ('max_tokens','max_completion_tokens'):
            with self.subTest(field=field),self.assertRaises(RequestError):
                normalize_request(dict(self.body,**{field:None}),MODEL)
        with self.assertRaises(RequestError):
            normalize_request(dict(self.body,max_tokens=4097),MODEL)

    def test_uncapped_normalization_retains_unresolved_budget_and_alias_precedence(self):
        self.assertIsNone(normalize_request(self.body,MODEL,0)['max_tokens'])
        for field in ('max_tokens','max_completion_tokens'):
            with self.subTest(field=field):
                self.assertIsNone(normalize_request(dict(self.body,**{field:None}),MODEL,0)['max_tokens'])
                self.assertEqual(normalize_request(dict(self.body,**{field:5000}),MODEL,0)['max_tokens'],5000)
        self.assertEqual(normalize_request(dict(self.body,max_tokens=5000,max_completion_tokens=6000),MODEL,0)['max_tokens'],6000)
        self.assertIsNone(normalize_request(dict(self.body,max_tokens=5000,max_completion_tokens=None),MODEL,0)['max_tokens'])
        for budget in (True,False,0,-1,1.5,'5000'):
            with self.subTest(budget=budget),self.assertRaises(RequestError):
                normalize_request(dict(self.body,max_tokens=budget),MODEL,0)

    def test_runtime_defaults_overrides_and_metadata_are_instance_local(self):
        candidate=self.runtime()
        self.assertEqual((candidate.max_output,candidate.output_budget_policy),(0,'remaining_context'))
        for mode in ('serial_mtp','continuous_greedy'):
            legacy=self.runtime(mode)
            self.assertEqual((legacy.max_output,legacy.output_budget_policy),(4096,'fixed_limit'))
        with patch.dict(os.environ,SPARK_MAX_OUTPUT='7000'):
            capped=self.runtime()
        with patch.dict(os.environ,SPARK_MAX_OUTPUT='0'):
            uncapped_legacy=self.runtime('serial_mtp')
        for runtime,limit,policy in ((candidate,None,'remaining_context'),
                                    (capped,7000,'fixed_limit'),
                                    (uncapped_legacy,None,'remaining_context')):
            with TestClient(create_app(runtime,self.key)) as client:
                model=client.get('/v1/models',headers=self.headers).json()['data'][0]
                scheduler=client.get('/v1/scheduler',headers=self.headers).json()
            for metadata in (model,scheduler):
                self.assertEqual(metadata['max_output_tokens'],limit)
                self.assertEqual(metadata['output_budget_policy'],policy)
        for value in ('-1','1.5','true',''):
            with self.subTest(value=value),patch.dict(os.environ,SPARK_MAX_OUTPUT=value),self.assertRaises(ValueError):
                Runtime(mode='continuous_mtp')

    def test_budget_resolution_checks_context_without_clamping(self):
        runtime=self.runtime()
        self.assertEqual(runtime.resolve_output_budget([1,2,3],None),262141)
        self.assertEqual(runtime.resolve_output_budget([1,2,3],262141),262141)
        for budget in (262142,True,False,0,-1,1.5):
            with self.subTest(budget=budget),self.assertRaises(RequestError):
                runtime.resolve_output_budget([1,2,3],budget)
        with self.assertRaises(RequestError):runtime.resolve_output_budget([1]*262144,None)
        with patch.dict(os.environ,SPARK_MAX_OUTPUT='5000'):
            capped=self.runtime()
        self.assertEqual(capped.resolve_output_budget([1],5000),5000)
        for budget in (None,5001):
            with self.subTest(budget=budget),self.assertRaises(RequestError):
                capped.resolve_output_budget([1],budget)

    def test_nonstream_and_sse_resolve_omitted_null_and_explicit_large_budgets(self):
        runtime=self.serving_runtime()
        cases=[({},262141),({'max_tokens':None},262141),
               ({'max_completion_tokens':None},262141),({'max_tokens':5000},5000),
               ({'max_completion_tokens':6000},6000)]
        with patch('api_server.chat_token_ids',return_value=[1,2,3]),TestClient(create_app(runtime,self.key)) as client:
            for stream in (False,True):
                for fields,expected in cases:
                    with self.subTest(stream=stream,fields=fields):
                        response=client.post('/v1/chat/completions',headers=self.headers,
                            json={**self.body,**fields,'stream':stream})
                        self.assertEqual(response.status_code,200,response.text)
                        self.assertEqual(self.budgets[-1],expected)
                        if stream:
                            events=[json.loads(line[6:]) for line in response.text.splitlines()
                                    if line.startswith('data: ') and line!='data: [DONE]']
                            finish=[e['choices'][0]['finish_reason'] for e in events if e.get('choices') and e['choices'][0]['finish_reason']]
                            usage=[e['usage'] for e in events if e.get('usage')][-1]
                            self.assertEqual(finish,['stop'])
                        else:
                            self.assertEqual(response.json()['choices'][0]['finish_reason'],'stop')
                            usage=response.json()['usage']
                        self.assertEqual(usage['completion_tokens'],1)
                        audit=json.loads(runtime.audit.read_text().splitlines()[-1])
                        self.assertEqual(audit['output_budget'],expected)
                        self.assertEqual(audit['output_budget_policy'],'remaining_context')

    def test_explicit_overflow_is_http_400_for_both_transports(self):
        runtime=self.serving_runtime()
        with patch('api_server.chat_token_ids',return_value=[1,2,3]),TestClient(create_app(runtime,self.key)) as client:
            for stream in (False,True):
                for field in ('max_tokens','max_completion_tokens'):
                    with self.subTest(stream=stream,field=field):
                        response=client.post('/v1/chat/completions',headers=self.headers,
                            json={**self.body,field:262142,'stream':stream})
                        self.assertEqual(response.status_code,400,response.text)
                        self.assertIn('per-request context',response.json()['error']['message'])
        self.assertEqual(self.budgets,[])

    def test_direct_generate_uses_same_remaining_context_and_validation(self):
        runtime=self.serving_runtime()
        for supplied,expected in ((None,262141),(5000,5000)):
            row,state=runtime.generate([1,2,3],supplied,cache=False)
            self.assertEqual(self.budgets[-1],expected)
            self.assertEqual((row['output_tokens'],row['finish_reason']),(1,'stop'))
        for budget in (262142,True,-1):
            with self.subTest(budget=budget),self.assertRaises(RequestError):
                runtime.generate([1,2,3],budget)

    def test_finite_cap_keeps_default_and_rejects_large_or_null_at_http_boundary(self):
        with patch.dict(os.environ,SPARK_MAX_OUTPUT='4096'):
            runtime=self.serving_runtime()
        with patch('api_server.chat_token_ids',return_value=[1,2,3]),TestClient(create_app(runtime,self.key)) as client:
            for stream in (False,True):
                response=client.post('/v1/chat/completions',headers=self.headers,json=dict(self.body,stream=stream))
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(self.budgets[-1],1024)
                for budget in (4097,None):
                    response=client.post('/v1/chat/completions',headers=self.headers,
                        json=dict(self.body,stream=stream,max_tokens=budget))
                    self.assertEqual(response.status_code,400,response.text)


if __name__=='__main__':unittest.main()
