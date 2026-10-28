"""CPU ownership/lifetime and host-dispatch tests; no CUDA numerical claim."""
import importlib.util
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch
import torch

from engine.ops.kv_cache import PagedKV
from engine.ops.qsa_workspace import QSAWorkspace


def inputs(rows=1, width=2051, length=33):
    return (torch.zeros(rows, 24, 256, dtype=torch.bfloat16),
            torch.zeros(length, 2, 288, dtype=torch.uint8),
            torch.full((rows, width), -1, dtype=torch.int32))


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.owner = object()
        self.workspace = QSAWorkspace(self.owner)
        self.query, self.packed, self.indices = inputs()

    def buffers(self, **kwargs):
        return self.workspace.buffers(self.query, self.indices, splits=64,
                                      owner=kwargs.get('owner', self.owner))

    def test_request_isolation_and_stable_storage(self):
        first = self.buffers()
        first[0].fill_(7)
        with patch('torch.empty', side_effect=AssertionError('reuse allocated')):
            again = self.buffers()
        self.assertTrue(all(a is b for a, b in zip(first, again)))
        self.assertTrue(torch.equal(again[0], torch.full_like(again[0], 7)))
        other_owner = object()
        other = QSAWorkspace(other_owner).buffers(self.query, self.indices, splits=64, owner=other_owner)
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(first, other)))
        with self.assertRaisesRegex(ValueError, 'different request'):
            self.buffers(owner=other_owner)
        with self.assertRaisesRegex(ValueError, 'different request'):
            self.buffers(owner=None)

    def test_context_length_is_not_a_storage_key(self):
        first = self.buffers()
        for length in (1, 17, 33, 256, 257, 8193):
            packed = torch.empty(length, 2, 288, dtype=torch.uint8)
            for history in (packed, PagedKV.from_dense(packed)):
                self.workspace.validate_inputs(self.query, history, self.indices, owner=self.owner)
                self.assertTrue(all(a is b for a, b in zip(first, self.buffers())))

    def test_fixed_shape_never_resizes_or_accumulates(self):
        first = self.buffers()
        cases = [(inputs(rows=2)[0], self.indices, 64),
                 (self.query, inputs(width=1024)[2], 64),
                 (self.query, self.indices, 32)]
        for query, indices, splits in cases:
            with self.subTest(splits=splits, rows=query.shape[0], width=indices.shape[1]):
                with self.assertRaises(ValueError):
                    self.workspace.buffers(query, indices, splits=splits, owner=self.owner)
        self.assertTrue(all(a is b for a, b in zip(first, self.buffers())))
        self.assertEqual(self.workspace.nbytes, sum(t.numel()*t.element_size() for t in first))

    def test_budget_checked_before_allocating_and_exact_limit_allowed(self):
        size = 24 * (256*2 + 64*(256*4+4))
        small = QSAWorkspace(self.owner, max_bytes=size-1)
        with patch('torch.empty', side_effect=AssertionError('over-budget allocation')):
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                small.buffers(self.query, self.indices, splits=64, owner=self.owner)
        self.assertIsNone(small.shape)
        self.assertEqual(small.nbytes, 0)
        exact = QSAWorkspace(self.owner, max_bytes=size)
        exact.buffers(self.query, self.indices, splits=64, owner=self.owner)
        self.assertEqual(exact.nbytes, size)
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                QSAWorkspace(self.owner, max_bytes=limit)

    def test_partial_allocation_failure_does_not_bind_workspace(self):
        empty = torch.empty
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('allocation failed')
            return empty(*args, **kwargs)
        with patch('torch.empty', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'allocation failed'):
                self.buffers()
        self.assertIsNone(self.workspace.shape)
        self.assertEqual(self.workspace.nbytes, 0)
        self.buffers()

    def test_reject_bad_query_index_metadata(self):
        cases = [(self.query.float(), self.indices),
                 (self.query[:, :23], self.indices),
                 (self.query.transpose(1, 2).contiguous().transpose(1, 2), self.indices),
                 (self.query[:0], self.indices[:0]),
                 (self.query, self.indices.float()),
                 (self.query, self.indices[:, :0]),
                 (self.query, torch.zeros(2, 2051, dtype=torch.int32)),
                 (self.query, torch.zeros(1, 4102, dtype=torch.int32)[:, ::2]),
                 (self.query, torch.empty(1, 2051, dtype=torch.int32, device='meta'))]
        for query, indices in cases:
            with self.subTest(shape=query.shape, index_shape=indices.shape, dtype=indices.dtype):
                with self.assertRaises(ValueError):
                    self.workspace.buffers(query, indices, splits=64, owner=self.owner)
        for splits in (0, 3, 128, True, 1.0):
            with self.assertRaises(ValueError):
                self.workspace.buffers(self.query, self.indices, splits=splits, owner=self.owner)
        self.workspace.buffers(self.query, self.indices.long(), splits=64, owner=self.owner)

    def test_reject_bad_kv_metadata(self):
        packed = self.packed
        histories = [packed.float(), packed[:0], packed[:, :1],
                     packed.transpose(0, 1).contiguous().transpose(0, 1),
                     torch.empty(packed.shape, dtype=packed.dtype, device='meta')]
        invalid_table = PagedKV.from_dense(packed)
        invalid_table.block_table = invalid_table.block_table.int()
        invalid_page = PagedKV.from_dense(packed)
        invalid_page._pages = (invalid_page._pages[0][:-1],)
        histories += [invalid_table, invalid_page]
        for history in histories:
            with self.assertRaises(ValueError):
                self.workspace.validate_inputs(self.query, history, self.indices, owner=self.owner)

    def test_borrowed_output_must_not_be_reused_as_input(self):
        output, _, _ = self.buffers()
        with self.assertRaisesRegex(ValueError, 'alias an input'):
            self.workspace.buffers(output, self.indices, splits=64, owner=self.owner)

    def test_kv_must_not_alias_partial_storage(self):
        _, partial, _ = self.buffers()
        raw = partial.view(torch.uint8).flatten()
        count = raw.numel() // (2*288)
        history = raw[:count*2*288].view(count, 2, 288)
        with self.assertRaisesRegex(ValueError, 'KV history/table must not alias'):
            self.workspace.validate_inputs(self.query, history, self.indices, owner=self.owner)

    def test_buffers_cannot_be_replaced_or_reshaped(self):
        output, _, _ = self.buffers()
        replacement = torch.empty_like(output)
        output.data = replacement
        with self.assertRaisesRegex(ValueError, 'storage was replaced'):
            self.buffers()
        owner = object()
        workspace = QSAWorkspace(owner)
        _, partial, _ = workspace.buffers(self.query, self.indices, splits=64, owner=owner)
        partial.resize_(1)
        with self.assertRaisesRegex(ValueError, 'shape/layout'):
            workspace.buffers(self.query, self.indices, splits=64, owner=owner)

    def test_buffer_cannot_retain_an_unbudgeted_larger_storage(self):
        output, _, _ = self.buffers()
        output.data = torch.empty((2, *output.shape), dtype=output.dtype)[0]
        with self.assertRaisesRegex(ValueError, 'storage capacity was changed'):
            self.buffers()

    def test_close_is_owner_checked_and_terminal(self):
        self.buffers()
        with self.assertRaises(ValueError):
            self.workspace.close(owner=object())
        self.workspace.close(owner=self.owner)
        self.assertEqual(self.workspace.nbytes, 0)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.buffers()


def host_dispatch_module():
    """Load the real wrapper with inert Triton decorators, never a GPU backend."""
    triton = ModuleType('triton')
    language = ModuleType('triton.language')
    language.constexpr = object()
    triton.jit = lambda fn=None, **kwargs: fn if fn is not None else lambda value: value
    triton.cdiv = lambda x, y: (x+y-1)//y
    triton.next_power_of_2 = lambda x: 1 << (x-1).bit_length()
    triton.language = language
    path = Path(__file__).parent/'engine/ops/qsa_triton.py'
    spec = importlib.util.spec_from_file_location('engine.ops._qsa_workspace_host_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'triton': triton, 'triton.language': language}):
        spec.loader.exec_module(module)
    return module


class LaunchRecorder:
    def __init__(self):
        self.calls = []
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


class DispatchTests(unittest.TestCase):
    def test_buffer_selection_keeps_existing_kernel_profile_and_arguments(self):
        module = host_dispatch_module()
        module._partial = LaunchRecorder()
        module._merge = LaunchRecorder()
        for rows, width, length in ((1,2051,1), (2,2051,31), (3,2051,257),
                                    (16,2051,2049), (32,2051,33), (256,2051,8193),
                                    (2048,2051,2051)):
            query, packed, indices = inputs(rows, width, length)
            for history in (packed, PagedKV.from_dense(packed)):
                with self.subTest(rows=rows, length=length, paged=isinstance(history, PagedKV)):
                    module._partial.calls.clear(); module._merge.calls.clear()
                    legacy = module.sparse_attention(query, history, indices)
                    old_partial = module._partial.calls[-1]
                    old_merge = list(module._merge.calls)
                    owner = object(); workspace = QSAWorkspace(owner)
                    explicit = module.sparse_attention(query, history, indices,
                        workspace=workspace, workspace_owner=owner)
                    new_partial = module._partial.calls[-1]
                    self.assertEqual(old_partial[0], new_partial[0])
                    self.assertEqual(old_partial[2], new_partial[2])
                    self.assertTrue(all(a is b for a, b in zip(old_partial[1][:4], new_partial[1][:4])))
                    self.assertEqual(old_partial[1][7:], new_partial[1][7:])
                    self.assertEqual(len(module._merge.calls), len(old_merge)*2)
                    if old_merge:
                        self.assertEqual(old_merge[-1][0], module._merge.calls[-1][0])
                        self.assertEqual(old_merge[-1][1][3:], module._merge.calls[-1][1][3:])
                        self.assertEqual(old_merge[-1][2], module._merge.calls[-1][2])
                    self.assertEqual(legacy.shape, explicit.shape)
                    again = module.sparse_attention(query, history, indices,
                        workspace=workspace, workspace_owner=owner)
                    self.assertIs(explicit, again)
                    self.assertNotEqual(legacy.data_ptr(), explicit.data_ptr())

    def test_predecoded_dispatch_retains_staging_and_profile(self):
        module = host_dispatch_module()
        module._partial = LaunchRecorder(); module._merge = LaunchRecorder()
        predecoded = ModuleType('engine.ops.qsa_prefill_sm121')
        predecoded._partial_predecoded = LaunchRecorder()
        staged = []
        def unpack(history):
            staged.append(history)
            return torch.empty(history.shape[0], 2, 512, dtype=torch.float16)
        predecoded.unpack_history = unpack
        with patch.dict('sys.modules', {'engine.ops.qsa_prefill_sm121': predecoded}):
            for rows, paged in ((1,False), (32,False), (256,False), (2048,False), (2048,True)):
                query, packed, indices = inputs(rows, length=257)
                history = PagedKV.from_dense(packed) if paged else packed
                expected_predecoded = rows >= 32 and (paged or rows <= 256)
                recorder = predecoded._partial_predecoded if expected_predecoded else module._partial
                before = len(staged)
                module.sparse_attention(query, history, indices, prefill_mode='predecoded')
                legacy = recorder.calls[-1]
                owner = object(); workspace = QSAWorkspace(owner)
                module.sparse_attention(query, history, indices, prefill_mode='predecoded',
                                        workspace=workspace, workspace_owner=owner)
                explicit = recorder.calls[-1]
                self.assertEqual(len(staged)-before, 2 if expected_predecoded else 0)
                self.assertEqual(legacy[0], explicit[0])
                self.assertEqual(legacy[2], explicit[2])
                scalar_start = 6 if expected_predecoded else 7
                self.assertEqual(legacy[1][scalar_start:], explicit[1][scalar_start:])

    def test_bad_owner_fails_before_launch(self):
        module = host_dispatch_module()
        module._partial = LaunchRecorder()
        query, packed, indices = inputs()
        owner = object(); workspace = QSAWorkspace(owner)
        for kwargs in ({'workspace': workspace}, {'workspace': workspace, 'workspace_owner': object()},
                       {'workspace_owner': owner}, {'workspace': object(), 'workspace_owner': owner}):
            with self.assertRaises((TypeError, ValueError)):
                module.sparse_attention(query, packed, indices, **kwargs)
        self.assertEqual(module._partial.calls, [])


if __name__ == '__main__':
    unittest.main()
