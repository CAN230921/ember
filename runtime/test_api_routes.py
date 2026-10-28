"""CPU-only HTTP contract checks; real GPU requests are verified separately."""
from concurrent.futures import Future
import json
import unittest
from fastapi.testclient import TestClient
from api_server import create_app,Job,MODEL,normalize_request


class FakeRuntime:
    ready=True
    def cancel_job(self,job):
        pass
    def expire_queued(self,job):
        return False
    def submit(self,options):
        job=Job(options); job.started.set(); job.future=Future()
        response=dict(id=job.id,object='chat.completion',created=job.created,model=MODEL,
            choices=[dict(index=0,message=dict(role='assistant',content='442'),finish_reason='stop')],
            usage=dict(prompt_tokens=8,completion_tokens=2,total_tokens=10,
                prompt_tokens_details=dict(cached_tokens=4)))
        job.events.put(('delta',dict(content='44'))); job.events.put(('delta',dict(content='2')))
        job.events.put(('done',dict(finish_reason='stop',usage=response['usage'])))
        job.future.set_result(response)
        return job


class RoutesTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(create_app(FakeRuntime(),'sk-test-'+32*'a'))
        self.headers=dict(Authorization='Bearer sk-test-'+32*'a')
        self.body=dict(model=MODEL,messages=[dict(role='user',content='计算')],temperature=0)

    def test_auth_and_model_discovery(self):
        self.assertEqual(self.client.get('/v1/models').status_code,401)
        self.assertEqual(self.client.get('/v1/models',headers=self.headers).json()['data'][0]['id'],MODEL)

    def test_invalid_sampling_is_json_400(self):
        response=self.client.post('/v1/chat/completions',headers=self.headers,json=dict(self.body,temperature=.5))
        self.assertEqual(response.status_code,400)
        self.assertEqual(response.json()['error']['type'],'invalid_request_error')

    def test_nonstream_and_sse_completion_with_usage(self):
        response=self.client.post('/v1/chat/completions',headers=self.headers,json=self.body)
        self.assertEqual(response.json()['choices'][0]['message']['content'],'442')
        response=self.client.post('/v1/chat/completions',headers=self.headers,
            json=dict(self.body,stream=True,stream_options=dict(include_usage=True)))
        self.assertEqual(response.status_code,200)
        self.assertIn('text/event-stream',response.headers['content-type'])
        self.assertIn('"content":"44"',response.text)
        self.assertIn('"completion_tokens":2',response.text)
        self.assertTrue(response.text.endswith('data: [DONE]\n\n'))

    def test_sse_always_reports_cache_without_opt_in(self):
        for options in (None,dict(include_usage=False)):
            body=dict(self.body,stream=True)
            if options is not None:body['stream_options']=options
            response=self.client.post('/v1/chat/completions',headers=self.headers,json=body)
            chunks=[json.loads(line[6:]) for line in response.text.splitlines()
                if line.startswith('data: ') and line!='data: [DONE]']
            usage=[value['usage'] for value in chunks if value.get('usage')]
            self.assertEqual(len(usage),1)
            self.assertEqual(usage[0]['prompt_tokens_details']['cached_tokens'],4)


if __name__=='__main__':unittest.main()
