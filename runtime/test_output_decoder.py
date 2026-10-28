"""CPU-only output tests; optional real tokenizer uses an explicit local path.

Run real UTF-8/protocol coverage without model inference or network access:
SPARK_TEST_TOKENIZER_DIR=/path/to/tokenizer \\
  python -m unittest test_output_decoder -v

Set SPARK_TEST_DECODER_BENCHMARK=1 too for synthetic local CPU timing. These
numbers measure text decoding only, never end-to-end model throughput.

Tested with tokenizers 0.22.2 and transformers 5.15.1. No GPU, model weights, or remote request is required.
"""
import json
import os
import random
import statistics
import sys
import time
import types
import unittest
from copy import copy
from pathlib import Path
from unittest.mock import patch

from output_decoder import IncrementalOutputDecoder


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, ids, **options):
        self.calls.append((list(ids), options))
        return ''.join({0: '', 1: 'a', 2: '中', 3: '\ufffd', 4: '</think>'}[i] for i in ids)


class DecoderFallbackTests(unittest.TestCase):
    def test_class_name_or_backend_duck_typing_cannot_enable_streaming(self):
        candidates = [
            type('Qwen2Tokenizer', (FakeTokenizer,), dict(is_fast=True,
                backend_tokenizer=object())),
            type('Qwen2Tokenizer', (FakeTokenizer,), dict(is_fast=True,
                __module__='transformers.models.qwen2.tokenization_qwen2',
                backend_tokenizer=object())),
            type('Qwen3_5Tokenizer', (FakeTokenizer,), dict(is_fast=False,
                __module__='transformers.models.qwen3_5.tokenization_qwen3_5',
                backend_tokenizer=object())),
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                decoder = IncrementalOutputDecoder(candidate(), ())
                self.assertFalse(decoder.incremental)
                self.assertEqual(decoder.feed([1, 2]), 'a中')
                self.assertEqual(decoder.finish([1, 2]), 'a中')

    def test_absent_stream_dependency_retains_full_decode_fallback(self):
        qwen_type = type('Qwen3_5Tokenizer', (FakeTokenizer,), dict(
            __module__='transformers.models.qwen3_5.tokenization_qwen3_5', is_fast=True))
        unavailable = types.ModuleType('tokenizers.decoders')
        with patch.dict(sys.modules, {'tokenizers.decoders': unavailable}):
            decoder = IncrementalOutputDecoder(qwen_type(), ())
        self.assertFalse(decoder.incremental)
        self.assertEqual(decoder.feed([1, 2]), 'a中')
        self.assertEqual(decoder.finish([1, 2]), 'a中')

    def test_fake_fallback_keeps_old_prefix_and_eos_behavior(self):
        tokenizer = FakeTokenizer()
        decoder = IncrementalOutputDecoder(tokenizer, {0})
        self.assertFalse(decoder.incremental)
        self.assertEqual(decoder.feed([1, 0]), 'a')
        self.assertEqual(decoder.feed([2, 3]), 'a中')
        self.assertEqual(decoder.feed([4]), 'a中\ufffd</think>')
        self.assertEqual(decoder.finish([1, 0, 2, 3, 4, 3]), 'a中\ufffd</think>\ufffd')
        self.assertEqual([call[0] for call in tokenizer.calls],
            [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 3]])
        self.assertTrue(all(call[1] == dict(skip_special_tokens=False,
            clean_up_tokenization_spaces=False) for call in tokenizer.calls))

    def test_final_decode_once_and_new_terminal_ids(self):
        tokenizer = FakeTokenizer()
        decoder = IncrementalOutputDecoder(tokenizer, ())
        self.assertEqual(decoder.feed([]), '')
        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(decoder.feed([1]), 'a')
        self.assertEqual(decoder.finish([1, 2]), 'a中')
        count = len(tokenizer.calls)
        self.assertEqual(decoder.finish([1, 2]), 'a中')
        self.assertEqual(len(tokenizer.calls), count)
        with self.assertRaises(RuntimeError):decoder.feed([1])
        with self.assertRaises(ValueError):decoder.finish([1])

    def test_history_changes_are_explicit_failures(self):
        decoder = IncrementalOutputDecoder(FakeTokenizer(), ())
        decoder.feed([1])
        with self.assertRaises(ValueError):decoder.finish([2])
        with self.assertRaises(RuntimeError):decoder.feed([1])

    def test_nonmonotonic_final_text_does_not_replay(self):
        tokenizer = FakeTokenizer()
        decoder = IncrementalOutputDecoder(tokenizer, ())
        decoder.feed([1])
        tokenizer.decode = lambda ids, **options: 'changed'
        with self.assertRaisesRegex(RuntimeError, 'cannot be safely replayed'):
            decoder.finish([1, 2])
        with self.assertRaises(RuntimeError):decoder.finish([1, 2])

    def test_incremental_calls_individual_ids_and_never_replays_on_error(self):
        class Stream:
            def __init__(self):self.calls = []
            def step(self, backend, token):
                self.calls.append(token)
                if token == 2:raise ValueError('synthetic decoder failure')
                return 'a'
        stream = Stream()
        tokenizer = FakeTokenizer()
        with patch('output_decoder._validated_stream', return_value=(stream, object())):
            decoder = IncrementalOutputDecoder(tokenizer, {0})
        self.assertEqual(decoder.feed([1, 0]), 'a')
        with self.assertRaisesRegex(RuntimeError, 'cannot be safely replayed'):
            decoder.feed([2])
        self.assertEqual(stream.calls, [1, 2])
        self.assertEqual(tokenizer.calls, [])
        with self.assertRaises(RuntimeError):decoder.finish([1, 0, 2])

    def test_two_request_fallback_state_is_independent(self):
        tokenizer = FakeTokenizer()
        a, b = (IncrementalOutputDecoder(tokenizer, ()) for _ in range(2))
        self.assertEqual(a.feed([1]), 'a')
        self.assertEqual(b.feed([2]), '中')
        self.assertEqual(a.feed([2]), 'a中')
        self.assertEqual(b.finish([2, 1]), '中a')
        self.assertEqual(a.finish([1, 2]), 'a中')


@unittest.skipUnless(os.environ.get('SPARK_TEST_TOKENIZER_DIR'), 'Real local tokenizer path not supplied')
class RealTokenizerDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from engine.tokenization import generation_eos_ids
        cls.tokenizer = AutoTokenizer.from_pretrained(os.environ['SPARK_TEST_TOKENIZER_DIR'],
            local_files_only=True, trust_remote_code=False)
        path = Path(os.environ['SPARK_TEST_TOKENIZER_DIR'])
        config = json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
        cls.eos = set(generation_eos_ids(cls.tokenizer, config.get('text_config', config), path))

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def raw(self, ids):
        return self.tokenizer.decode([i for i in ids if i not in self.eos],
            skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def test_validated_current_backend_is_incremental(self):
        self.assertTrue(IncrementalOutputDecoder(self.tokenizer, self.eos).incremental)

    def test_supported_class_with_non_bytelevel_backend_falls_back(self):
        from tokenizers import Tokenizer
        from tokenizers.decoders import WordPiece
        tokenizer = copy(self.tokenizer)
        tokenizer._tokenizer = Tokenizer.from_str(self.tokenizer.backend_tokenizer.to_str())
        tokenizer.backend_tokenizer.decoder = WordPiece()
        decoder = IncrementalOutputDecoder(tokenizer, self.eos)
        self.assertFalse(decoder.incremental)
        ids = self.ids('中文 synthetic token sequence')
        expected = tokenizer.decode(ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        self.assertEqual(decoder.feed(ids), expected.rstrip('\ufffd'))
        self.assertEqual(decoder.finish(ids), expected)

    def test_supported_backend_step_error_cannot_replay_or_final_decode(self):
        class Stream:
            def __init__(self, skip_special_tokens):
                if skip_special_tokens:raise AssertionError('Control markers must stay visible')
            def step(self, backend, token):
                if token == 2:raise ValueError('Synthetic incremental failure')
                return 'a'
        with patch('tokenizers.decoders.DecodeStream', Stream):
            decoder = IncrementalOutputDecoder(self.tokenizer, self.eos)
        self.assertTrue(decoder.incremental)
        with patch.object(self.tokenizer, 'decode', side_effect=AssertionError('No replay')) as complete:
            self.assertEqual(decoder.feed([1]), 'a')
            with self.assertRaises(RuntimeError):decoder.feed([2])
            with self.assertRaises(RuntimeError):decoder.finish([1, 2])
            complete.assert_not_called()

    def test_deterministic_random_legal_id_sequences(self):
        rng = random.Random(20261005)
        vocabulary_ids = sorted(set(self.tokenizer.get_vocab().values()))
        special_ids = sorted(set(self.tokenizer.get_added_vocab().values()))
        for case in range(128):
            ids = rng.choices(vocabulary_ids, k=30)
            if special_ids:
                ids[case % 30] = special_ids[case % len(special_ids)]
            if self.eos and case % 8 == 0:
                ids[(case+1) % 30] = sorted(self.eos)[case % len(self.eos)]
            stride = 1 + case % 3
            with self.subTest(case=case, stride=stride):
                decoder = IncrementalOutputDecoder(self.tokenizer, self.eos)
                for start in range(0, len(ids), stride):
                    raw = decoder.feed(ids[start:start+stride])
                    self.assertTrue(self.raw(ids[:start+stride]).startswith(raw))
                self.assertEqual(decoder.finish(ids), self.raw(ids))

    def test_unicode_protocol_and_terminal_replacement_strides(self):
        samples = [
            '中文测试：𠮷𠮶 🧑🏽‍💻 👨‍👩‍👧‍👦 🙂 \n',
            '<think>分析中文🙂</think>\n答案<|im_end|>',
            ' \nhello  world\t\n\\n <tool_call>{"name":"foo","arguments":{"x":"中文🙂"}}</tool_call>',
            'literal <tool_ then <tool_call> and </think>; \ufffd end',
            'ends with byte chars: 𠮷🙂\ufffd',
            '<|im_start|>assistant<think>跨边界标记</think>answer<|endoftext|>',
            'e\u0301 café   multiple\r\nspaces \u200d👩\u200d💻',
        ]
        for text in samples:
            ids = self.ids(text)
            for stride in (1, 2, 3):
                with self.subTest(text=text, stride=stride):
                    decoder = IncrementalOutputDecoder(self.tokenizer, self.eos)
                    for start in range(0, len(ids), stride):
                        raw = decoder.feed(ids[start:start+stride])
                        self.assertTrue(self.raw(ids[:start+stride]).startswith(raw))
                    self.assertEqual(decoder.finish(ids), self.raw(ids))

    def test_complete_marker_is_visible_before_same_round_incomplete_utf8(self):
        tail = self.ids('𠮷')[:1]
        self.assertTrue(self.raw(tail).endswith('\ufffd'))
        for marker in ('<think>', '</think>', '<tool_call>', '</tool_call>', '<|im_start|>'):
            with self.subTest(marker=marker):
                ids = self.ids(marker) + tail
                decoder = IncrementalOutputDecoder(self.tokenizer, self.eos)
                self.assertIn(marker, decoder.feed(ids))
                self.assertEqual(decoder.finish(ids), self.raw(ids))
        # Test an end marker that the API is not filtering as a generation EOS.
        ids = self.ids('<|im_end|>') + tail
        decoder = IncrementalOutputDecoder(self.tokenizer, ())
        self.assertIn('<|im_end|>', decoder.feed(ids))
        self.assertEqual(decoder.finish(ids), self.tokenizer.decode(ids,
            skip_special_tokens=False, clean_up_tokenization_spaces=False))

    def test_eos_filter_and_two_requests_do_not_share_utf8_buffers(self):
        a_ids, b_ids = self.ids('𠮷𠮶🙂'), self.ids('另一路<think>中文</think>answer')
        a, b = (IncrementalOutputDecoder(self.tokenizer, self.eos) for _ in range(2))
        for position in range(max(len(a_ids), len(b_ids))):
            if position < len(a_ids):a.feed(a_ids[position:position+1])
            if position < len(b_ids):b.feed(b_ids[position:position+1])
        a_ids = a_ids + list(self.eos)
        a.feed(list(self.eos))
        self.assertEqual(a.finish(a_ids), self.raw(a_ids))
        self.assertEqual(b.finish(b_ids), self.raw(b_ids))

    @unittest.skipUnless(os.environ.get('SPARK_TEST_DECODER_BENCHMARK'), 'Optional local CPU measurement')
    def test_synthetic_decode_measurement(self):
        seed = self.ids('人工测试文本。 CPU benchmark only: answer = {"ok": true};\n')
        rows = []
        for n in (32768, 65536):
            ids = (seed * ((n+len(seed)-1)//len(seed)))[:n]
            decoder = IncrementalOutputDecoder(self.tokenizer, self.eos)
            begin = time.perf_counter()
            for start in range(0, n-2, 2):decoder.feed(ids[start:start+2])
            prior_ms = (time.perf_counter()-begin)*1000
            begin = time.perf_counter();decoder.feed(ids[-2:])
            increment_ms = (time.perf_counter()-begin)*1000
            old = []
            for _ in range(5):
                begin = time.perf_counter();self.raw(ids).rstrip('\ufffd')
                old.append((time.perf_counter()-begin)*1000)
            self.assertEqual(decoder.finish(ids), self.raw(ids))
            rows.append(dict(tokens=n,full_prefix_last_step_ms=statistics.median(old),
                increment_last_two_tokens_ms=increment_ms,increment_prior_total_ms=prior_ms))
        print('\nSYNTHETIC_LOCAL_CPU_ONLY '+json.dumps(rows))


if __name__ == '__main__':unittest.main()
