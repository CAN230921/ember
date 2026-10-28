"""CPU canonical-MTP and API oracles for committed semantic stops."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from api_server import ContinuousRequest, Job, MODEL, Runtime, create_app, normalize_request
from test_api_protocol import CALL, TOOLS
from test_continuous_mtp import greedy_gold, history_hash, make_runner
from test_reasoning_mtp import autoregressive_oracle


def assert_canonical(test, session, prompt):
    target = list(prompt)+session.generated[:-1]
    test.assertEqual(session.state.tokens, target)
    test.assertEqual(int(session.state.bank[0]), history_hash(target))
    test.assertIsNone(session.state.prefixes)
    if session.draft_state is not None:
        shifted = list(prompt[1:])+session.generated
        test.assertEqual(session.draft_state.shifted, shifted)
        test.assertEqual(session.draft_state.length, len(shifted))
        test.assertFalse(session.draft_state.temporary)
        test.assertEqual(session.draft_state.last_hash, history_hash(list(prompt[:1])+shifted))
        test.assertEqual(int(session.draft_state.bank[0]), history_hash(list(prompt[:1])+shifted))


class SemanticMTPStopTests(unittest.TestCase):
    def ready(self, runner, prompt, budget=18, eos=(255,)):
        session = runner.start(prompt, budget, eos, 64)
        while session.phase == 'prefill':
            runner.prefill(session)
        return session

    def test_semantic_stop_preserves_canonical_committed_target_draft_and_statistics(self):
        for mode in ('accept', 'first_reject', 'second_reject', 'alternate'):
            for reason in ('protocol_end_marker', 'reasoning_reopened'):
                with self.subTest(mode=mode, reason=reason):
                    runner = make_runner(mode=mode, draft_tokens=2)
                    prompt = [3, 5, 8]
                    session = self.ready(runner, prompt)
                    runner.decode([session])
                    assert_canonical(self, session, prompt)
                    generated, stats = list(session.generated), dict(session.stats)
                    state, draft = session.state, session.draft_state
                    multi, proposal = session.draft_multi.clone(), session.proposal
                    calls = (len(runner.engine.calls), len(runner.engine.verify_calls), len(runner.draft.calls))
                    self.assertTrue(runner.stop(session, reason))
                    self.assertEqual(session.generated, greedy_gold(prompt, len(generated)))
                    self.assertEqual(session.generated, generated)
                    self.assertEqual(session.stats, stats)
                    self.assertIs(session.state, state)
                    self.assertIs(session.draft_state, draft)
                    torch.testing.assert_close(session.draft_multi, multi)
                    self.assertEqual(session.proposal, proposal)
                    assert_canonical(self, session, prompt)
                    result = session.result(0)
                    self.assertEqual((result['finish_reason'], result['stop_reason']), ('stop', reason))
                    self.assertEqual(result['output_tokens'], len(generated))
                    self.assertEqual((session.done, session.phase), (True, 'done'))
                    self.assertFalse(runner.stop(session, reason))
                    with self.assertRaises(ValueError):
                        runner.decode([session])
                    self.assertEqual(calls, (len(runner.engine.calls), len(runner.engine.verify_calls), len(runner.draft.calls)))
                    runner.close(session)

    def test_semantic_stop_does_not_override_output_budget_or_eos(self):
        for budget, eos, finish in ((1, (255,), 'length'), (18, (184,), 'stop')):
            with self.subTest(budget=budget, eos=eos):
                runner = make_runner()
                session = self.ready(runner, [3, 5, 8], budget, eos)
                self.assertTrue(session.done)
                self.assertFalse(runner.stop(session, 'protocol_end_marker'))
                self.assertIsNone(session.stop_reason)
                self.assertEqual(session.result(0)['finish_reason'], finish)
                runner.close(session)

    def test_invalid_or_uncommitted_stop_never_changes_history(self):
        runner = make_runner()
        session = runner.start([3, 5, 8], 18, {255}, 64)
        with self.assertRaises(ValueError):
            runner.stop(session, 'protocol_end_marker')
        runner.prefill(session)
        generated = list(session.generated)
        with self.assertRaises(ValueError):
            runner.stop(session, 'arbitrary text')
        self.assertEqual(session.generated, generated)
        self.assertFalse(session.done)
        runner.close(session)
        with self.assertRaises(ValueError):
            runner.stop(session, 'protocol_end_marker')


class MappingTokenizer:
    def __init__(self, prompts, pieces, *, starts=False, closing=(), opening=()):
        self.prompts = {tuple(prompt) for prompt in prompts}
        self.pieces, self.starts = pieces, starts
        self.closing, self.opening = list(closing), list(opening)

    def encode(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False)
        return list({'</think>': self.closing, '<think>': self.opening}[text])

    def decode(self, ids, **kwargs):
        if tuple(ids) in self.prompts:
            return '<|im_start|>assistant\n'+('<think>\n' if self.starts else '')
        return ''.join(self.pieces.get(token, f't{token}') for token in ids)


class SemanticMTPApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, SPARK_API_AUDIT=str(Path(self.temp.name)/'audit.jsonl'),
                              SPARK_MTP_POLICY='acceptance')
        self.env.start()
        self.runtime = Runtime(mode='continuous_mtp')
        self.runtime.ready = True
        self.runtime.batch_runner = make_runner(self.runtime.prefix_cache, draft_tokens=2)
        self.runtime.eos = {255}
        self.prompt = [3, 5, 8]
        self.sequence = greedy_gold(self.prompt, 18)
        self.assertEqual(len(set(self.sequence)), len(self.sequence))
        self.tokenize = patch('api_server.request_token_ids', return_value=list(self.prompt))
        self.tokenize.start()
        self.key = 'fixture-'+32*'t'
        self.headers = {'Authorization': 'Bearer '+self.key}

    def tearDown(self):
        self.runtime.close()
        self.tokenize.stop()
        self.env.stop()
        self.temp.cleanup()

    def configure(self, pieces, *, starts=False, closing=(), opening=()):
        self.runtime.tokenizer = MappingTokenizer([self.prompt], dict(zip(self.sequence, pieces)),
            starts=starts, closing=closing, opening=opening)

    def body(self, **options):
        return dict(model=MODEL, messages=[dict(role='user', content='fixture')], max_tokens=18, **options)

    def rows(self):
        return [json.loads(line) for line in self.runtime.audit.read_text().splitlines()]

    def make_request(self, **options):
        normalized = normalize_request(self.body(**options), MODEL, self.runtime.max_output)
        job = Job(normalized)
        self.runtime.pending += 1
        request = ContinuousRequest(self.runtime, job)
        request.start()
        return request

    def run_steps(self, request):
        while not request.done:
            if request.phase == 'prefill':
                request.prefill()
            else:
                request.decode()
            request.after_step()

    def test_single_protocol_token_stops_json_sse_without_next_decode(self):
        self.configure(['42', '<|im_start|>', 'discarded visible bytes', 'more bytes'])
        with TestClient(create_app(self.runtime, self.key)) as client:
            for stream in (False, True):
                before = len(self.runtime.batch_runner.engine.verify_calls)
                response = client.post('/v1/chat/completions', headers=self.headers, json=self.body(stream=stream))
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(self.runtime.batch_runner.engine.verify_calls)-before, 1)
                if stream:
                    events = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
                    self.assertEqual(events[-1], '[DONE]')
                    chunks = [json.loads(value) for value in events[:-1]]
                    choices = [choice for chunk in chunks for choice in chunk['choices']]
                    self.assertEqual(''.join(choice['delta'].get('content', '') for choice in choices), '42')
                    self.assertEqual([choice['finish_reason'] for choice in choices if choice.get('finish_reason')], ['stop'])
                    usage = next(chunk['usage'] for chunk in chunks if chunk.get('usage'))
                else:
                    data = response.json()
                    self.assertEqual(data['choices'][0]['message']['content'], '42')
                    self.assertEqual(data['choices'][0]['finish_reason'], 'stop')
                    usage = data['usage']
                # The accepted verify round already committed three outputs;
                # the semantic stop must neither invent EOS nor truncate them.
                self.assertEqual(usage['completion_tokens'], 4)
                audit = self.rows()[-1]
                self.assertEqual(audit['output_tokens'], 4)
                self.assertEqual(audit['stop_reason'], 'protocol_end_marker')
                self.assertEqual(audit['phase'], 'done')
                self.assertEqual(audit['status'], 'completed')

    def test_split_control_tag_waits_for_completion_then_prevents_following_round(self):
        self.configure(['42', '<|im_', 'st', 'ar', 't|>', 'discarded', 'discarded'])
        request = self.make_request()
        request.prefill(); request.after_step()
        request.decode(); request.after_step()
        self.assertFalse(request.done)
        self.assertEqual(request.framer.content, '42')
        self.assertIsNone(request.generation.stop_reason)
        request.decode(); request.after_step()
        self.assertTrue(request.done)
        self.assertEqual(len(self.runtime.batch_runner.engine.verify_calls), 2)
        self.assertEqual(request.framer.content, '42')
        assert_canonical(self, request.generation, self.prompt)
        response = request.finish()
        self.assertEqual(response['usage']['completion_tokens'], 7)
        self.assertEqual(response['choices'][0]['finish_reason'], 'stop')

    def test_all_protocol_end_markers_stop_at_completed_framing_boundary(self):
        for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'):
            with self.subTest(marker=marker):
                self.configure(['42', marker, 'discarded', 'discarded'])
                request = self.make_request()
                self.run_steps(request)
                self.assertEqual(request.generation.stop_reason, 'protocol_end_marker')
                response = request.finish()
                self.assertEqual(response['choices'][0]['message']['content'], '42')

    def test_stop_one_request_does_not_change_other_request_canonical_state(self):
        other_prompt = [8, 2]
        other_sequence = greedy_gold(other_prompt, 8)
        self.configure(['42', '<|im_start|>', 'discarded', 'discarded'])
        self.runtime.tokenizer.prompts.add(tuple(other_prompt))
        self.runtime.tokenizer.pieces.update({token: 'B' for token in other_sequence})
        a = self.make_request()
        with patch('api_server.request_token_ids', return_value=other_prompt):
            b = self.make_request()
        for request in (a, b):
            request.prefill(); request.after_step()
        a.decode(); a.after_step()
        self.assertTrue(a.done)
        self.assertFalse(b.done)
        assert_canonical(self, a.generation, self.prompt)
        assert_canonical(self, b.generation, other_prompt)
        response_a = a.finish()
        self.run_steps(b)
        assert_canonical(self, b.generation, other_prompt)
        self.assertEqual(b.generation.generated, greedy_gold(other_prompt, 18))
        response_b = b.finish()
        self.assertEqual(response_a['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(response_b['choices'][0]['finish_reason'], 'length')
        self.assertIsNone(self.rows()[-1]['stop_reason'])
        self.assertEqual(self.runtime.pending, 0)

    def test_reopened_reasoning_stops_and_usage_counts_all_committed_tokens(self):
        closing, opening = [self.sequence[1]], [self.sequence[3]]
        self.configure(['算', '</think>', '42', '<think>', '不应继续', '不应继续', '不应继续'],
                       starts=True, closing=closing, opening=opening)
        request = self.make_request(enable_thinking=True)
        self.run_steps(request)
        self.assertEqual(request.generation.stop_reason, 'reasoning_reopened')
        self.assertTrue(request.reasoning.reasoning_reopened)
        self.assertEqual(request.reasoning.used, 2)
        self.assertEqual(len(self.runtime.batch_runner.engine.verify_calls), 1)
        assert_canonical(self, request.generation, self.prompt)
        response = request.finish()
        self.assertEqual(response['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(response['choices'][0]['message']['content'], '42')
        self.assertEqual(response['usage']['completion_tokens'], 4)
        self.assertEqual(response['usage']['completion_tokens_details']['reasoning_tokens'], 2)
        self.assertEqual(self.rows()[-1]['stop_reason'], 'reasoning_reopened')

    def test_tool_close_does_not_trigger_semantic_end_stop(self):
        self.configure([CALL, ' ', '后续', '正文'])
        request = self.make_request(tools=TOOLS)
        request.prefill(); request.after_step()
        self.assertEqual(request.framer.mode, 'content')
        self.assertEqual(len(request.framer.calls), 1)
        self.assertFalse(request.done)
        self.assertIsNone(request.generation.stop_reason)
        self.run_steps(request)
        self.assertEqual(request.generation.generated, self.sequence)
        self.assertIsNone(request.generation.stop_reason)
        response = request.finish()
        self.assertEqual(response['choices'][0]['finish_reason'], 'tool_calls')

    def test_forced_budget_close_then_reopen_stops_without_recounting_allowance(self):
        self.sequence = autoregressive_oracle(self.prompt, 18, {2: 240, 3: 241})
        self.assertEqual(len(set(self.sequence)), len(self.sequence))
        self.configure(['核', '算', '</thi', 'nk>', '42', '<think>', '已提交但不得再续'],
                       starts=True, closing=[240, 241], opening=[self.sequence[5]])
        request = self.make_request(enable_thinking=True, thinking_budget=2)
        self.run_steps(request)
        self.assertEqual(request.generation.stop_reason, 'reasoning_reopened')
        self.assertEqual(request.generation.generated, self.sequence[:7])
        assert_canonical(self, request.generation, self.prompt)
        self.assertEqual(request.reasoning.used, 2)
        self.assertEqual(request.reasoning.forced, 2)
        self.assertEqual(request.reasoning.intercepted_reasoning_tokens, 1)
        response = request.finish()
        self.assertEqual(response['choices'][0]['message']['content'], '42')
        self.assertEqual(response['usage']['completion_tokens'], 7)
        self.assertEqual(response['usage']['completion_tokens_details']['reasoning_tokens'], 2)
        audit = self.rows()[-1]
        self.assertEqual(audit['stop_reason'], 'reasoning_reopened')
        self.assertEqual(audit['reasoning']['intercepted_reasoning_tokens'], 1)
        self.assertEqual(audit['speculative']['forced_control_tokens'], 2)

    def test_ordinary_marker_names_and_long_content_do_not_stop(self):
        self.configure(['im_start', '不是', '控制标记', 'think', '普通正文']+['长答案']*13)
        request = self.make_request()
        self.run_steps(request)
        self.assertIsNone(request.generation.stop_reason)
        self.assertEqual(request.generation.generated, self.sequence)
        response = request.finish()
        self.assertEqual(response['choices'][0]['finish_reason'], 'length')
        self.assertEqual(response['usage']['completion_tokens'], 18)

    def test_budget_wins_when_control_marker_is_last_allowed_token(self):
        self.configure(['42', '<|im_start|>'])
        body = self.body()
        body['max_tokens'] = 2
        options = normalize_request(body, MODEL, self.runtime.max_output)
        job = Job(options); self.runtime.pending += 1
        request = ContinuousRequest(self.runtime, job); request.start()
        self.run_steps(request)
        self.assertIsNone(request.generation.stop_reason)
        response = request.finish()
        self.assertEqual(response['choices'][0]['finish_reason'], 'length')
        self.assertEqual(response['usage']['completion_tokens'], 2)

    def test_non_mtp_after_step_does_not_call_unsupported_stop(self):
        self.configure(['42', '<|im_start|>', 'discarded', 'discarded'])
        request = self.make_request()
        request.prefill(); request.after_step()
        request.decode()
        self.runtime.mode = 'continuous_greedy'
        with patch.object(self.runtime.batch_runner, 'stop') as stop:
            request.after_step()
            stop.assert_not_called()
        self.assertFalse(request.done)
        self.runtime.mode = 'continuous_mtp'
        self.runtime.batch_runner.close(request.generation)
        request.job.released = True
        self.runtime.pending = 0


if __name__ == '__main__':
    unittest.main()
