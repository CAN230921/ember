"""Host metadata identity/capture policy with CPU doubles; no CUDA proof."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


def host_module():
    """Load the real wrapper with inert Triton decorators, never a GPU backend."""
    triton = ModuleType('triton')
    language = ModuleType('triton.language')
    language.constexpr = object()
    triton.jit = lambda fn=None, **kwargs: fn if fn is not None else lambda value: value
    triton.language = language
    path = Path(__file__).parent/'engine/ops/gdn_triton.py'
    spec = importlib.util.spec_from_file_location('engine.ops._gdn_metadata_host_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'triton': triton, 'triton.language': language}):
        spec.loader.exec_module(module)
    return module


class FakeCuda:
    def __init__(self):
        self.index = 0
        self.capturing = False
        self.events = []

    def current_device(self):
        self.events.append(('current_device', self.index))
        return self.index

    @contextmanager
    def device(self, device):
        self.events.append(('device', str(device)))
        yield

    def is_current_stream_capturing(self):
        self.events.append(('capture_check',))
        return self.capturing


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = host_module()

    def test_cpu_reuse_has_original_values_dtype_and_never_calls_cuda(self):
        cuda = FakeCuda()
        cache = self.module.PrefillCuSeqLensCache(cuda=cuda)
        first = cache.get(2048, 'cpu')
        self.assertIs(cache.get(2048, torch.device('cpu')), first)
        self.assertEqual(first.tolist(), [0, 2048])
        self.assertEqual(first.dtype, torch.int32)
        self.assertEqual(first.device, torch.device('cpu'))
        self.assertEqual(cuda.events, [])

    def make_cuda_cache(self, max_entries=32):
        cuda, calls = FakeCuda(), []
        def cpu_factory(values, *, device, dtype):
            calls.append((tuple(values), str(device), dtype))
            return torch.tensor(values, dtype=dtype)
        cache = self.module.PrefillCuSeqLensCache(max_entries,
                    tensor_factory=cpu_factory, cuda=cuda)
        return cache, cuda, calls

    def test_complete_device_identity_and_tail_shape_are_isolated(self):
        cache, cuda, calls = self.make_cuda_cache()
        zero = cache.get(2048, 'cuda:0')
        self.assertIs(cache.get(2048, 'cuda'), zero)
        one = cache.get(2048, 'cuda:1')
        self.assertIsNot(one, zero)
        cuda.index = 1
        self.assertIs(cache.get(2048, 'cuda'), one)
        tail = cache.get(137, 'cuda:1')
        self.assertIsNot(tail, one)
        self.assertEqual(tail.tolist(), [0, 137])
        cpu = cache.get(2048, 'cpu')
        self.assertIsNot(cpu, zero)
        self.assertEqual([(values, device) for values, device, _ in calls],
                         [((0,2048), 'cuda:0'), ((0,2048), 'cuda:1'),
                          ((0,137), 'cuda:1'), ((0,2048), 'cpu')])

    def test_lru_is_bounded_and_eviction_only_drops_cache_reference(self):
        cache, _, calls = self.make_cuda_cache(max_entries=2)
        first = cache.get(2048, 'cuda:0')
        tail = cache.get(64, 'cuda:0')
        self.assertIs(cache.get(2048, 'cuda:0'), first)
        cache.get(65, 'cuda:0')
        self.assertEqual(len(cache._entries), 2)
        self.assertIs(cache.get(2048, 'cuda:0'), first)
        self.assertIsNot(cache.get(64, 'cuda:0'), tail)
        self.assertEqual(len(cache._entries), 2)
        self.assertEqual(len(calls), 4)
        # The evicted metadata may still be owned by FLA or a pending kernel.
        self.assertEqual(tail.tolist(), [0, 64])

    def test_capture_hit_does_not_mutate_lru_or_allocate(self):
        cache, cuda, calls = self.make_cuda_cache(max_entries=2)
        first = cache.get(2048, 'cuda:0')
        cache.get(137, 'cuda:0')
        order = list(cache._entries)
        cuda.capturing = True
        self.assertIs(cache.get(2048, 'cuda:0'), first)
        self.assertEqual(list(cache._entries), order)
        self.assertEqual(len(calls), 2)

    def test_capture_miss_requires_eager_warmup_without_mutation(self):
        cache, cuda, calls = self.make_cuda_cache()
        cache.get(2048, 'cuda:0')
        before = list(cache._entries)
        cuda.capturing = True
        with self.assertRaisesRegex(RuntimeError, 'Warm the complete GDN prefill metadata path'):
            cache.get(137, 'cuda:0')
        self.assertEqual(list(cache._entries), before)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
