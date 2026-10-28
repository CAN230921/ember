"""Per-request context contracts with real API/scheduler and bounded CPU fakes.

These tests do not allocate model KV, run CUDA, or establish 262K capacity.
The long-prompt oracle retains counters and one chunk of hidden rows, rather
than copying or re-hashing the complete prefix at every prefill boundary.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from fastapi.testclient import TestClient

from api_server import MODEL, Runtime, create_app
from engine.continuous_mtp import ContinuousMTPRunner
from engine.mtp import MTPDraft


LONG_CONTEXT = 262144
LEGACY_CONTEXT = 32768
CHUNK = 2048


def logits_for(token):
    logits = torch.full((1, 64), -100.)
    logits[0, token] = 100.
    return logits


class PositionTarget:
    """O(total input tokens), with no full-prefix state or model allocation."""
    def __init__(self):
        self.embedding = torch.empty(0)
        self.calls = []

    def __call__(self, ids, state=None, return_hidden=False,
                 prefill_chunk=False, owned_graph_cache=False):
        start = state.length if state is not None else 0
        marker = state.marker if state is not None else int(ids[0])
        end = start + len(ids)
        if prefill_chunk and len(ids) > CHUNK:
            raise AssertionError('Prefill bypassed the chunk boundary')
        self.calls.append((marker, start, end, prefill_chunk))
        updated = SimpleNamespace(length=end, marker=marker)
        result = logits_for(10 + marker), updated
        if return_hidden:
            hidden = torch.stack((torch.arange(start, end), torch.tensor(ids)), dim=1)
            return (*result, SimpleNamespace(multi=hidden))
        return result


class NoDraft:
    def __call__(self, *args, **kwargs):
        raise AssertionError('A two-token output budget should not invoke MTP')


class ShiftedPositionDraft:
    """Assert every target position and shifted prompt ID across 32K/262K."""
    def __init__(self, prompt, first_token):
        self.prompt = prompt
        self.first_token = first_token
        self.calls = []

    def __call__(self, ids, hidden, state=None, prefill_chunk=False):
        if not prefill_chunk:
            raise AssertionError('This oracle only exercises long paired prefill')
        start = state.length if state is not None else 0
        end = start + len(ids)
        torch.testing.assert_close(hidden[:, 0], torch.arange(start, end))
        torch.testing.assert_close(hidden[:, 1], torch.tensor(self.prompt[start:end]))
        expected = self.prompt[start + 1:end + 1]
        if end == len(self.prompt):
            expected = expected + [self.first_token]
        if list(ids) != expected:
            raise AssertionError('MTP lookahead IDs or absolute positions were truncated')
        self.calls.append((start, end, int(hidden[0, 0]), int(hidden[-1, 0])))
        return SimpleNamespace(logits=logits_for(self.first_token),
            state=SimpleNamespace(length=end), multi=hidden)


class ContextLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ)
        self.env.start()
        for key in ('SPARK_MAX_CONTEXT', 'SPARK_MAX_ACTIVE', 'SPARK_MAX_QUEUED',
                    'SPARK_EXECUTION_MODE'):
            os.environ.pop(key, None)
        os.environ['SPARK_MTP_POLICY'] = 'acceptance'
        os.environ['SPARK_API_AUDIT'] = str(Path(self.tmp.name) / 'requests.jsonl')
        self.runtimes, self.gates = [], []
        self.lengths = {}
        self.tokenize = patch('api_server.chat_token_ids', side_effect=self.token_ids)
        self.tokenize.start()
        self.key = 'sk-context-test-' + 32 * 'c'
        self.headers = {'Authorization': 'Bearer ' + self.key}

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        for runtime in self.runtimes:
            runtime.close()
        self.tokenize.stop()
        self.env.stop()
        self.tmp.cleanup()

    def token_ids(self, tokenizer, messages, *args):
        count, marker = self.lengths[messages[0]['content']]
        return [marker] * count

    def runtime(self, mode='continuous_mtp'):
        runtime = Runtime(mode=mode)
        self.runtimes.append(runtime)
        runtime.ready, runtime.eos = True, {63}
        runtime.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ','.join(map(str, ids)))
        if mode == 'continuous_mtp':
            runtime.batch_runner = ContinuousMTPRunner(PositionTarget(), NoDraft(),
                                                       draft_tokens=1, low_sync=True)
        return runtime

    @staticmethod
    def body(name, *, stream=False):
        return dict(model=MODEL, messages=[dict(role='user', content=name)],
                    temperature=0, max_tokens=2, stream=stream)

    def test_mode_defaults_and_model_metadata_are_instance_local(self):
        for mode, expected in (('serial_mtp', LEGACY_CONTEXT),
                               ('continuous_greedy', LEGACY_CONTEXT),
                               ('continuous_mtp', LONG_CONTEXT)):
            with self.subTest(mode=mode):
                runtime = self.runtime(mode)
                self.assertEqual(runtime.max_context, expected)
                with TestClient(create_app(runtime, self.key)) as client:
                    model = client.get('/v1/models', headers=self.headers).json()['data'][0]
                    scheduler = client.get('/v1/scheduler', headers=self.headers).json()
                self.assertEqual(model['context_window'], expected)
                self.assertEqual(scheduler['context_window_per_request'], expected)
                self.assertEqual(scheduler['aggregate_context_budget'], expected * runtime.max_active)

    def test_environment_override_is_per_instance_and_requires_positive_integer(self):
        original = self.runtime()
        with patch.dict(os.environ, SPARK_MAX_CONTEXT='65536'):
            for mode in ('serial_mtp', 'continuous_greedy', 'continuous_mtp'):
                with self.subTest(mode=mode):
                    runtime = self.runtime(mode)
                    self.assertEqual(runtime.max_context, 65536)
                    with TestClient(create_app(runtime, self.key)) as client:
                        self.assertEqual(client.get('/v1/models', headers=self.headers)
                                         .json()['data'][0]['context_window'], 65536)
            self.assertEqual(original.max_context, LONG_CONTEXT)
            with TestClient(create_app(original, self.key)) as client:
                self.assertEqual(client.get('/v1/models', headers=self.headers)
                                 .json()['data'][0]['context_window'], LONG_CONTEXT)
        for value in ('0', '-1', '1.5', 'invalid', ''):
            with self.subTest(value=value), patch.dict(os.environ, SPARK_MAX_CONTEXT=value):
                with self.assertRaises(ValueError):
                    Runtime(mode='continuous_mtp')

    def test_checkpoint_config_must_cover_this_instance_context(self):
        runtime = self.runtime()
        runtime.validate_context_config({'max_position_embeddings': LONG_CONTEXT})
        runtime.validate_context_config({'max_position_embeddings': LONG_CONTEXT + 1})
        for context in (LEGACY_CONTEXT, LONG_CONTEXT - 1):
            with self.subTest(context=context), self.assertRaises(ValueError):
                runtime.validate_context_config({'max_position_embeddings': context})
        legacy = self.runtime('serial_mtp')
        legacy.validate_context_config({'max_position_embeddings': LEGACY_CONTEXT})
        with patch.dict(os.environ, SPARK_MAX_CONTEXT='524288'):
            larger = self.runtime()
        with self.assertRaises(ValueError):
            larger.validate_context_config({'max_position_embeddings': LONG_CONTEXT})

    def test_environment_override_also_controls_real_http_admission(self):
        with patch.dict(os.environ, SPARK_MAX_CONTEXT='4096'):
            runtime = self.runtime()
        self.lengths.update(exact_override=(4094, 1), over_override=(4095, 1))
        with TestClient(create_app(runtime, self.key)) as client:
            accepted = client.post('/v1/chat/completions', headers=self.headers,
                                   json=self.body('exact_override'))
            rejected = client.post('/v1/chat/completions', headers=self.headers,
                                   json=self.body('over_override'))
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json()['usage']['total_tokens'], 4096)
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertIn('4096', rejected.json()['error']['message'])

    def test_real_http_accepts_exact_262144_and_rejects_one_more_before_sse(self):
        runtime = self.runtime()
        self.lengths.update(exact=(LONG_CONTEXT - 2, 1), over=(LONG_CONTEXT - 1, 1))
        with TestClient(create_app(runtime, self.key)) as client:
            for stream in (False, True):
                with self.subTest(stream=stream):
                    accepted = client.post('/v1/chat/completions', headers=self.headers,
                                           json=self.body('exact', stream=stream))
                    self.assertEqual(accepted.status_code, 200, accepted.text)
                    if stream:
                        events = [line[6:] for line in accepted.text.splitlines()
                                  if line.startswith('data: ')]
                        self.assertEqual(events[-1], '[DONE]')
                        chunks = [json.loads(value) for value in events[:-1]]
                        usage = [value['usage'] for value in chunks if value.get('usage')][-1]
                        self.assertTrue(any(choice.get('finish_reason') == 'length'
                                            for value in chunks for choice in value.get('choices', [])))
                    else:
                        usage = accepted.json()['usage']
                        self.assertEqual(accepted.json()['choices'][0]['finish_reason'], 'length')
                    self.assertEqual(usage['prompt_tokens'], LONG_CONTEXT - 2)
                    self.assertEqual(usage['completion_tokens'], 2)
                    self.assertEqual(usage['total_tokens'], LONG_CONTEXT)
                    calls = len(runtime.batch_runner.engine.calls)
                    rejected = client.post('/v1/chat/completions', headers=self.headers,
                                           json=self.body('over', stream=stream))
                    self.assertEqual(rejected.status_code, 400, rejected.text)
                    self.assertEqual(rejected.headers['content-type'], 'application/json')
                    self.assertIn(str(LONG_CONTEXT), rejected.json()['error']['message'])
                    self.assertEqual(len(runtime.batch_runner.engine.calls), calls)
        self.assertEqual(runtime.pending, 0)

    def test_old_modes_reject_above_32768_without_running_a_model(self):
        self.lengths['legacy_over'] = (LEGACY_CONTEXT - 1, 1)
        for mode in ('serial_mtp', 'continuous_greedy'):
            with self.subTest(mode=mode):
                runtime = self.runtime(mode)
                with TestClient(create_app(runtime, self.key)) as client:
                    response = client.post('/v1/chat/completions', headers=self.headers,
                                           json=self.body('legacy_over'))
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(str(LEGACY_CONTEXT), response.json()['error']['message'])

    def test_two_active_http_requests_each_own_the_full_context_budget(self):
        runtime = self.runtime()
        self.lengths.update(A=(LONG_CONTEXT - 2, 1), B=(LONG_CONTEXT - 2, 2))
        owner_entered, owner_release = threading.Event(), threading.Event()
        both_submitted, both_prefilled, prefill_release = (threading.Event() for _ in range(3))
        self.gates.extend((owner_release, prefill_release))
        jobs, jobs_lock, held = [], threading.Lock(), []

        def hold_owner():
            owner_entered.set()
            if not owner_release.wait(10):
                raise RuntimeError('CPU owner barrier timed out')

        submit, prefill = runtime.submit, runtime.batch_runner.prefill

        def record_submit(options):
            job = submit(options)
            with jobs_lock:
                jobs.append(job)
                if len(jobs) == 2:
                    both_submitted.set()
            return job

        def hold_after_both_prefills(session):
            prefill(session)
            active = runtime.scheduler._active
            if (not held and len(active) == 2 and
                    all(slot.session.generation.position > 0 for slot in active)):
                held.append(True)
                both_prefilled.set()
                if not prefill_release.wait(10):
                    raise RuntimeError('CPU prefill barrier timed out')

        runtime.submit = record_submit
        runtime.batch_runner.prefill = hold_after_both_prefills
        runtime.executor.submit(hold_owner)
        self.assertTrue(owner_entered.wait(2))
        app = create_app(runtime, self.key)
        with TestClient(app) as a_client, TestClient(app) as b_client, ThreadPoolExecutor(2) as pool:
            a = pool.submit(a_client.post, '/v1/chat/completions', headers=self.headers, json=self.body('A'))
            b = pool.submit(b_client.post, '/v1/chat/completions', headers=self.headers, json=self.body('B'))
            try:
                self.assertTrue(both_submitted.wait(3))
                owner_release.set()
                self.assertTrue(both_prefilled.wait(3))
                snapshot = runtime.scheduler.snapshot()
                self.assertEqual(set(snapshot['active_ids']), {job.id for job in jobs})
                self.assertEqual(snapshot['queued'], 0)
                self.assertTrue(all(not job.future.done() for job in jobs))
                self.assertEqual([job.input_tokens for job in jobs], [LONG_CONTEXT - 2] * 2)
                self.assertEqual(sum(job.input_tokens + job.options['max_tokens'] for job in jobs),
                                 2 * LONG_CONTEXT)
            finally:
                owner_release.set()
                prefill_release.set()
            responses = (a.result(5), b.result(5))
        for response, expected in zip(responses, ('11,11', '12,12')):
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['usage']['total_tokens'], LONG_CONTEXT)
            self.assertEqual(response.json()['choices'][0]['message']['content'], expected)
        self.assertEqual(runtime.pending, 0)

    def test_paired_prefill_reaches_262141_without_resetting_positions_at_32k(self):
        prompt = list(range(1, LONG_CONTEXT - 2))
        target, draft = PositionTarget(), ShiftedPositionDraft(prompt, first_token=11)
        runner = ContinuousMTPRunner(target, draft, draft_tokens=1, low_sync=True)
        session = runner.start(prompt, 3, {63}, CHUNK, cache=False)
        try:
            while session.phase == 'prefill':
                runner.prefill(session)
            self.assertEqual(session.position, LONG_CONTEXT - 3)
            self.assertEqual(session.state.length, len(prompt))
            self.assertEqual(session.draft_state.length, len(prompt))
            self.assertEqual(session.generated, [11])
            self.assertEqual(len(target.calls), 128)
            self.assertEqual(len(draft.calls), 128)
            self.assertEqual([row[1] for row in target.calls], list(range(0, len(prompt), CHUNK)))
            self.assertEqual([row[0] for row in draft.calls], list(range(0, len(prompt), CHUNK)))
            self.assertEqual(draft.calls[-1][3], len(prompt) - 1)
            self.assertGreater(draft.calls[-1][3], LEGACY_CONTEXT)
        finally:
            runner.close(session)

    def test_real_mtp_forward_checks_limit_before_fusion_and_keeps_absolute_positions(self):
        for start in (0, LONG_CONTEXT - 1):
            with self.subTest(start=start):
                # Invoke the production forward method, without loading any
                # weights. The replacements isolate its guard and positions.
                draft = MTPDraft.__new__(MTPDraft)
                torch.nn.Module.__init__(draft)
                draft.config = {'max_position_embeddings': LONG_CONTEXT}
                draft.fuse_inputs = Mock(side_effect=lambda ids, hidden: torch.empty(len(ids), 1))
                draft.attn_hc = SimpleNamespace(mix=lambda hidden, preserve: (hidden, hidden, None))
                draft.attention = Mock(side_effect=lambda mixed, positions, state, **kwargs:
                    (mixed, SimpleNamespace(length=kwargs['position_start'] + len(mixed))))
                combine = lambda residual, block, injection, preserve: (residual, block, None)
                draft.mlp_hc = SimpleNamespace(combine_and_mix=combine)
                draft.final_mixer = SimpleNamespace(combine_and_mix=combine)
                draft.moe = lambda mixed: mixed
                draft.head = lambda hidden: logits_for(11)
                state = None if start == 0 else SimpleNamespace(length=start)
                rows = LONG_CONTEXT - start

                result = draft(range(rows), object(), state, prefill_chunk=start == 0)
                draft.fuse_inputs.assert_called_once()
                draft.attention.assert_called_once()
                args, kwargs = draft.attention.call_args
                self.assertIs(args[2], state)
                self.assertEqual(kwargs['position_start'], start)
                torch.testing.assert_close(args[1], torch.arange(start, LONG_CONTEXT))
                self.assertEqual(result.state.length, LONG_CONTEXT)
                if state is not None:
                    self.assertEqual(state.length, start)

                draft.fuse_inputs.reset_mock()
                draft.attention.reset_mock()
                with self.assertRaisesRegex(ValueError, 'MTP context exceeds'):
                    draft(range(rows + 1), object(), state, prefill_chunk=start == 0)
                draft.fuse_inputs.assert_not_called()
                draft.attention.assert_not_called()


if __name__ == '__main__':
    unittest.main()
