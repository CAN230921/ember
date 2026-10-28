"""CPU output/usage integration with real tokenizer IDs; no model inference."""
import os
import threading
from types import SimpleNamespace
import unittest

from api_protocol import OutputFramer
from api_server import ContinuousRequest, Job, Runtime
from engine.request_control import GenerationCancelled
from output_decoder import IncrementalOutputDecoder
from reasoning_control import ReasoningControl
from test_api_protocol import CALL, TOOLS


@unittest.skipUnless(os.environ.get('SPARK_TEST_TOKENIZER_DIR'), 'Requires local tokenizer fixture')
class RealTokenizerOutputApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(os.environ['SPARK_TEST_TOKENIZER_DIR'],
                                                     local_files_only=True)

    def request(self, *, reasoning=False, tools=()):
        audits, stops, closed, released = [], [], [], []
        job = Job(dict(max_tokens=512, stream=True, tools=list(tools)))
        job.input_tokens = 3
        runner = SimpleNamespace(close=lambda session: closed.append(session))
        def stop(session, reason):
            stops.append(reason)
            session.done = True
            return True
        runner.stop = stop
        runtime = SimpleNamespace(tokenizer=self.tokenizer, eos={self.tokenizer.eos_token_id},
            mode='continuous_mtp', output_budget_policy='remaining_context',
            batch_runner=runner, prefix_cache=SimpleNamespace(bytes=0),
            append_audit=audits.append, lock=threading.RLock(), pending=1,
            _discard_preparation=lambda value: released.append(value),
            fatal=False, ready=True)
        # Exercise the real idempotent release contract, including job.released.
        runtime._release = lambda value: Runtime._release(runtime, value)
        request = ContinuousRequest(runtime, job)
        request.framer = OutputFramer(list(tools), lambda delta: job.events.put(('delta', delta)),
                                      starts_in_reasoning=reasoning)
        request.output_decoder = IncrementalOutputDecoder(self.tokenizer, runtime.eos)
        self.assertTrue(request.output_decoder.incremental)
        closing = self.tokenizer.encode('</think>', add_special_tokens=False) if reasoning else []
        opening = self.tokenizer.encode('<think>', add_special_tokens=False) if reasoning else []
        request.reasoning = ReasoningControl(reasoning, None, None, closing, opening_ids=opening)
        session = SimpleNamespace(generated=[], done=False, phase='decode', progress=lambda: None)
        session.result = lambda retained: dict(generated_ids=list(session.generated),
            output_tokens=len(session.generated), finish_reason='stop' if session.done else 'length',
            prefix_cache=dict(reused_tokens=0))
        request.generation = session
        return request, session, audits, stops, closed, released

    def test_interleaved_unicode_reasoning_and_tool_requests_keep_private_text(self):
        samples = [('中英𠮷🙂分析</think>答案�尾', True, ()), (CALL, False, TOOLS)]
        requests = [self.request(reasoning=r, tools=t) for _, r, t in samples]
        sequences = [self.tokenizer.encode(text, add_special_tokens=False) for text, _, _ in samples]
        for start in range(0, max(map(len, sequences)), 3):
            for (request, session, *_), ids in zip(requests, sequences):
                session.generated.extend(ids[start:start+3])
                request.after_step()
        for (text, reasoning, tools), (request, session, audits, stops, closed, released) in zip(samples, requests):
            session.done = True
            response = request.finish()
            expected = OutputFramer(list(tools), starts_in_reasoning=reasoning)
            expected.feed(self.tokenizer.decode(session.generated, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False), final=True)
            actual = response['choices'][0]['message']
            # Tool-call IDs are intentionally generated per framer.
            if actual.get('tool_calls'):
                self.assertEqual(actual['tool_calls'][0]['function'], expected.calls[0]['function'])
            else:
                self.assertEqual(actual, expected.message())
            self.assertEqual(response['usage']['completion_tokens'], len(session.generated))
            self.assertEqual(len(closed), 1)
            self.assertEqual(len(released), 1)
            self.assertIsNone(request.output_decoder)
            self.assertEqual(audits[0]['status'], 'completed')

    def test_protocol_marker_before_incomplete_unicode_stops_in_same_round(self):
        request, session, audits, stops, closed, released = self.request()
        marker = self.tokenizer.encode('<|im_start|>', add_special_tokens=False)
        incomplete = self.tokenizer.encode('𠮷', add_special_tokens=False)[:1]
        session.generated.extend(marker+incomplete)
        request.after_step()
        self.assertEqual(stops, ['protocol_end_marker'])
        self.assertTrue(session.done)
        response = request.finish()
        self.assertEqual(response['choices'][0]['message']['content'], '')
        self.assertEqual(response['usage']['completion_tokens'], len(marker+incomplete))

    def test_final_committed_but_unfed_suffix_is_flushed_once(self):
        request, session, audits, stops, closed, released = self.request()
        ids = self.tokenizer.encode('完整答案𠮷�', add_special_tokens=False)
        session.generated.extend(ids[:-2])
        request.after_step()
        session.generated.extend(ids[-2:])
        session.done = True
        response = request.finish()
        self.assertEqual(response['choices'][0]['message']['content'],
                         self.tokenizer.decode(ids, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False))
        self.assertEqual(response['usage']['completion_tokens'], len(ids))

    def test_cancel_after_commit_does_not_emit_unfed_tokens_or_leak_decoder(self):
        request, session, audits, stops, closed, released = self.request()
        session.generated.extend(self.tokenizer.encode('已提交', add_special_tokens=False))
        request.job.cancel.set()
        self.assertIsNone(request.finish(error=GenerationCancelled('decode')))
        self.assertEqual(audits[0]['output_tokens'], len(session.generated))
        self.assertEqual(audits[0]['status'], 'cancelled')
        self.assertFalse(any(kind == 'delta' for kind, _ in list(request.job.events.queue)))
        self.assertEqual((len(closed), len(released)), (1, 1))
        self.assertIsNone(request.output_decoder)

    def test_incomplete_tool_error_still_releases_state_and_decoder(self):
        request, session, audits, stops, closed, released = self.request(tools=TOOLS)
        session.generated.extend(self.tokenizer.encode('<tool_call><function=add_numbers>',
                                                       add_special_tokens=False))
        request.after_step()
        session.done = True
        with self.assertRaisesRegex(ValueError, 'incomplete tool call'):
            request.finish()
        self.assertEqual(audits[0]['status'], 'error')
        self.assertEqual((len(closed), len(released)), (1, 1))
        self.assertIsNone(request.output_decoder)


if __name__ == '__main__':
    unittest.main()
