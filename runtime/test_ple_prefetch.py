"""PLE page-cache prefetch: byte-exact lookup, exact page coverage, hint plumbing.

CPU-only with a synthetic checkpoint and a recorded posix_fadvise. This proves
no value changes and the intended read set; it proves nothing about NVMe or
end-to-end prefill speed (measure on the GPU host).
"""
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import torch

from engine.ops import ple_lookup
from engine.ops.ple import PLERef, hint_next_prefill_chunk
from engine.ops.ple_lookup import DiskPleTable, NGramHasher

PREFIX = 'model.layers.2.ple_embedding.ngram_embedding'
WIDTH, SHARD_ROWS, LAST_ROWS = 32, 1000, 700


def write_checkpoint(root):
    rng = np.random.default_rng(7)
    tables, weight_map = [], {}
    for index, rows in enumerate((SHARD_ROWS, LAST_ROWS)):
        data = rng.integers(0, 256, size=(rows, WIDTH), dtype=np.uint8)
        name = f'{PREFIX}.shard_{index}.weight'
        header = json.dumps({name: dict(dtype='F8_E4M3', shape=[rows, WIDTH],
                                        data_offsets=[0, rows*WIDTH])}).encode()
        header += b' ' * (-len(header) % 8 + 5)   # nonzero, unaligned data start
        filename = f'shard{index}.safetensors'
        (root / filename).write_bytes(struct.pack('<Q', len(header)) + header + data.tobytes())
        weight_map[name] = filename
        tables.append(data)
    (root / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=weight_map)))
    return np.concatenate(tables)


class Recorder:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, fd, offset, length, advice):
        with self.lock:
            self.calls.append((os.fstat(fd).st_ino, offset, length, advice))


class TableCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.expected = write_checkpoint(self.root)
        self.addCleanup(self.dir.cleanup)
        # Not defined on macOS; the Linux value is irrelevant to the recorded checks.
        constant = mock.patch('os.POSIX_FADV_WILLNEED', getattr(os, 'POSIX_FADV_WILLNEED', 3), create=True)
        constant.start()
        self.addCleanup(constant.stop)

    def table(self, **env):
        with mock.patch.dict(os.environ, env):
            return DiskPleTable(self.root, PREFIX, cache_rows=0)

    def test_gather_is_byte_exact_in_both_modes(self):
        ids = np.array([[3, 1500, 3], [999, 1000, 1699]])
        for env in ({}, {'ENGINE_PLE_IO': 'legacy'}):
            rec = Recorder()
            with mock.patch('os.posix_fadvise', rec, create=True):
                table = self.table(**env)
                rows = table.gather(ids)
            self.assertEqual(rows.shape, (2, 3, WIDTH))
            np.testing.assert_array_equal(rows, self.expected[ids])
            self.assertEqual(table.io_enabled, not env)
            self.assertEqual(bool(rec.calls), not env)

    def test_willneed_covers_exact_pages_and_coalesces(self):
        rec = Recorder()
        with mock.patch('os.posix_fadvise', rec, create=True):
            table = self.table()
            rows = [0, 1, 2, 500, 999, 1000, 1699, 5000, -1]   # last two invalid
            table.page = 64                                    # small pages to exercise merging
            calls = table.willneed(rows)
        self.assertEqual(calls, len(rec.calls))
        self.assertTrue(all(advice == os.POSIX_FADV_WILLNEED for *_, advice in rec.calls))
        for shard, local in ((0, [0, 1, 2, 500, 999]), (1, [0, 699])):
            inode = os.stat(table.paths[shard]).st_ino
            ranges = [(o, o+n) for i, o, n, _ in rec.calls if i == inode]
            for row in local:
                begin = table.offsets[shard] + row*WIDTH
                for byte in (begin, begin+WIDTH-1):
                    self.assertTrue(any(lo <= byte < hi for lo, hi in ranges), (shard, row))
            self.assertTrue(all(lo % 64 == 0 and hi % 64 == 0 for lo, hi in ranges))
            self.assertLess(sum(hi-lo for lo, hi in ranges), 64*(len(local)*2+2))  # nothing bulk
        self.assertEqual(table.willneed([]), 0)
        self.assertEqual(table.willneed([-4, 10**9]), 0)

    def test_legacy_mode_issues_no_advice(self):
        rec = Recorder()
        with mock.patch('os.posix_fadvise', rec, create=True):
            table = self.table(ENGINE_PLE_IO='legacy')
            self.assertEqual(table.willneed([1, 2, 3]), 0)
            self.assertFalse(table.submit(lambda: [1]))
        self.assertEqual(rec.calls, [])

    def test_background_worker_runs_bounds_queue_and_swallows_errors(self):
        rec = Recorder()
        with mock.patch('os.posix_fadvise', rec, create=True):
            table = self.table()
            gate, done = threading.Event(), threading.Event()
            def blocker():
                gate.wait(5)
                return [10]
            self.assertTrue(table.submit(blocker))          # taken by worker
            while table._jobs.qsize():                      # wait until worker took it
                done.wait(0.01)
            self.assertTrue(table.submit(lambda: [11]))
            self.assertTrue(table.submit(lambda: 1/0))      # queue now full (2)
            self.assertFalse(table.submit(lambda: [12]))    # dropped, never blocks the engine
            gate.set()
            for _ in range(500):
                if table.io_stats['async_errors'] and rec.calls and table._jobs.qsize() == 0:
                    break
                done.wait(0.01)
        self.assertEqual(table.io_stats['async_dropped'], 1)
        self.assertEqual(table.io_stats['async_errors'], 1)
        self.assertGreaterEqual(len(rec.calls), 2)


class StubPle:
    def __init__(self, hasher):
        self.hasher, self.hints = hasher, []

    def hint_next_chunk(self, ids, context, expected_rows):
        self.hints.append((list(ids), list(context), expected_rows))


class Layer:
    def __init__(self, ple):
        self.ple = ple


class HintCase(unittest.TestCase):
    def setUp(self):
        sizes = np.array([101, 103, 107, 109])
        self.hasher = NGramHasher([3, 5, 7], sizes, np.array([0, 101, 204, 311]), eos_token_id=7)

    def test_helper_context_matches_engine_rule_and_skips_non_ple(self):
        ids = list(range(20, 60))
        ple = StubPle(self.hasher)
        engine = type('E', (), {})()
        engine.layers = [Layer(None), Layer(ple)]
        hint_next_prefill_chunk(engine, ids, 0, 16, 16)
        hint_next_prefill_chunk(engine, ids, 16, 32, 16)
        hint_next_prefill_chunk(engine, ids, 32, 40, 16)    # final chunk: nothing follows
        hint_next_prefill_chunk(engine, ids, 24, 40, 16)    # end == len(ids)
        hint_next_prefill_chunk(object(), ids, 0, 16, 16)   # engines without layers
        self.assertEqual(ple.hints, [(ids[16:32], ids[14:16], 16), (ids[32:40], ids[30:32], 16)])

    def forward_with(self, hint, current_ids):
        ple = PLERef.__new__(PLERef)
        torch.nn.Module.__init__(ple)
        submitted = []
        ple.hasher, ple._hint = self.hasher, None
        ple.table = type('T', (), dict(submit=lambda s, fn: submitted.append(fn) or True,
                                       lookup=lambda s, ids, device: np.zeros(1)))()
        ple.core = lambda hidden, embeddings, state=None, return_prefixes=False: ('core',)
        if hint:
            ple.hint_next_chunk(*hint)
        context = np.asarray([[7, 7]])
        result = ple.forward(torch.zeros(1), np.asarray(current_ids), [0, len(current_ids)], context)
        return ple, submitted, result

    def test_forward_prefetches_exactly_the_rows_the_next_forward_will_hash(self):
        ids = [int(x) for x in np.random.default_rng(3).integers(0, 90, size=48)]
        chunk, end = 16, 16
        expected = self.hasher(ids, [0, len(ids)], np.asarray([[7, 7]]))[end:end+chunk]
        context = ([7]*2 + ids[:end])[-2:]
        ple, submitted, result = self.forward_with((ids[end:end+chunk], context, chunk), ids[:chunk])
        self.assertEqual(result, ('core',))
        self.assertEqual(len(submitted), 1)
        np.testing.assert_array_equal(submitted[0](), expected)
        self.assertIsNone(ple._hint)

    def test_mismatched_or_missing_hint_is_cleared_and_ignored(self):
        ids = list(range(10, 26))
        ple, submitted, _ = self.forward_with((ids, [7, 7], 99), ids)
        self.assertEqual(submitted, [])
        self.assertIsNone(ple._hint)
        _, submitted, _ = self.forward_with(None, ids)
        self.assertEqual(submitted, [])


if __name__ == '__main__':
    unittest.main()
