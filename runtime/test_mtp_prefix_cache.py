"""CPU cache-policy and storage-ownership checks; no GPU speed claims."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from engine.continuous_mtp import ContinuousMTPPrefixCache
from engine.ops.kv_cache import PagedKV
from engine.snapshot_cache import snapshot_requirements
from test_continuous_mtp import finish, make_runner


def states(ids, end, count=4):
    return (SimpleNamespace(tokens=list(ids[:end]), bank=torch.arange(count, dtype=torch.float32)),
            SimpleNamespace(length=end, bank=torch.full((count,), float(ids[end]))))


def publish(cache, ids, end=64, count=4, chunk_size=64, owner=None):
    target, draft = states(ids, end, count)
    return cache.publish(ids, end, target, draft, chunk_size, owner)


class MTPPrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.a = [3, 5]*97
        self.b = [9, 2]*97
        self.c = [7, 1]*97

    def test_alternating_requests_keep_both_deep_branches(self):
        cache = ContinuousMTPPrefixCache(65536)
        runner = make_runner(cache)
        reused = []
        for ids in (self.a, self.b, self.a, self.b):
            session = finish(runner, ids, 7)
            reused.append(session.reused)
            runner.close(session)
        self.assertEqual(reused, [0, 0, 192, 192])
        self.assertEqual(cache.snapshot()['publish'], 6)
        self.assertIsNone(cache.owner)

    def test_interleaved_writers_both_publish_and_close_retains_both(self):
        cache = ContinuousMTPPrefixCache(65536)
        runner = make_runner(cache)
        a = runner.start(self.a, 7, {255}, 64)
        b = runner.start(self.b, 7, {255}, 64)
        while a.phase == 'prefill' or b.phase == 'prefill':
            for session in (a, b):
                if session.phase == 'prefill':
                    runner.prefill(session)
        self.assertEqual((a.checkpoint_tokens, b.checkpoint_tokens), (192, 192))
        runner.close(a)
        self.assertEqual(cache.checkout(self.b, 64)[2], 192)
        runner.close(b)
        self.assertEqual(cache.checkout(self.a, 64)[2], 192)
        self.assertEqual(cache.checkout(self.b, 64)[2], 192)

    def test_longest_match_searches_other_branches_and_min_end(self):
        cache = ContinuousMTPPrefixCache(65536)
        publish(cache, self.a, 64)
        publish(cache, self.b, 128)
        publish(cache, self.a, 192)
        self.assertEqual(cache.checkout(self.a, 64)[2], 192)
        self.assertEqual(cache.checkout(self.b, 64)[2], 128)
        self.assertEqual(cache.checkout(self.a, 64, min_end=64)[2], 192)
        with patch('engine.continuous_mtp.clone_state') as clone:
            self.assertEqual(cache.checkout(self.a, 64, min_end=192), (None, None, 0, []))
            clone.assert_not_called()
        with self.assertRaises(ValueError):
            cache.checkout(self.a, 64, min_end=-1)

    def test_lookahead_is_part_of_key(self):
        cache = ContinuousMTPPrefixCache(65536)
        alternate = list(self.a)
        alternate[64] += 1
        publish(cache, self.a)
        self.assertEqual(cache.checkout(alternate, 64)[2], 0)
        publish(cache, alternate)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(cache.checkout(self.a, 64)[1].bank[0].item(), self.a[64])
        self.assertEqual(cache.checkout(alternate, 64)[1].bank[0].item(), alternate[64])

    def test_chunk_namespace_and_complete_boundary_validation(self):
        cache = ContinuousMTPPrefixCache(65536)
        publish(cache, self.a, 128, chunk_size=64)
        publish(cache, self.a, 128, chunk_size=128)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(cache.checkout(self.a, 64)[2], 128)
        self.assertEqual(cache.checkout(self.a, 128)[2], 128)
        self.assertEqual(cache.checkout(self.a, 256)[2], 0)
        target, draft = states(self.a, 64)
        for end, chunk in ((64, 0), (64, True), (63, 64), (64, 32), (len(self.a), 64)):
            with self.subTest(end=end, chunk=chunk), self.assertRaises(ValueError):
                cache.publish(self.a, end, target, draft, chunk, object())
        draft.length = 63
        with self.assertRaises(ValueError):
            cache.publish(self.a, 64, target, draft, 64, object())
        draft.length = 64
        target.tokens[-1] += 1
        with self.assertRaises(ValueError):
            cache.publish(self.a, 64, target, draft, 64, object())

    def test_duplicate_does_not_clone_touch_or_change_generation(self):
        cache = ContinuousMTPPrefixCache(65536)
        publish(cache, self.a)
        cache.checkout(self.a, 64)
        clock, generation = cache.entries[0].last_used, cache.generation
        with patch('engine.continuous_mtp.clone_state') as clone:
            self.assertTrue(publish(cache, self.a, owner=object()))
            clone.assert_not_called()
        self.assertEqual((cache.entries[0].last_used, cache.generation), (clock, generation))
        self.assertEqual(cache.snapshot()['duplicate'], 1)
        cache.release(object())
        self.assertEqual(cache.generation, generation)

    def test_checkout_copies_mutable_target_and_draft(self):
        cache = ContinuousMTPPrefixCache(65536)
        target, draft = states(self.a, 64)
        cache.publish(self.a, 64, target, draft, 64, object())
        target.bank.fill_(-1)
        draft.bank.fill_(-1)
        first, first_draft, _, _ = cache.checkout(self.a, 64)
        second, second_draft, _, _ = cache.checkout(self.a, 64)
        first.bank.fill_(99)
        first_draft.bank.fill_(99)
        torch.testing.assert_close(second.bank, torch.arange(4, dtype=torch.float32))
        self.assertEqual(second_draft.bank[0].item(), self.a[64])
        self.assertEqual(cache.entries[0].target.bank[0].item(), 0)
        self.assertEqual(cache.entries[0].draft.bank[0].item(), self.a[64])

    def test_shared_pages_are_counted_once_tables_are_private_and_append_cow(self):
        packed = PagedKV.from_dense(torch.zeros(64, 2, 288, dtype=torch.uint8), 128)
        page_bytes = packed._pages[0].untyped_storage().nbytes()
        cache = ContinuousMTPPrefixCache(page_bytes+24)
        for ids in (self.a, self.b):
            target = SimpleNamespace(tokens=list(ids[:64]), packed=packed)
            draft = SimpleNamespace(length=64, bank=torch.ones(1))
            self.assertTrue(cache.publish(ids, 64, target, draft, 64, object()))
        self.assertEqual(cache.bytes, page_bytes+24)
        self.assertEqual(cache.snapshot()['eviction'], 0)
        first, _, _, _ = cache.checkout(self.a, 64)
        second, _, _, _ = cache.checkout(self.b, 64)
        self.assertEqual(first.packed._pages[0].data_ptr(), second.packed._pages[0].data_ptr())
        self.assertNotEqual(first.packed.block_table.data_ptr(), second.packed.block_table.data_ptr())
        before = second.packed.to_dense().clone()
        appended = first.packed.append(torch.ones(1, 2, 288, dtype=torch.uint8))
        self.assertEqual((appended.length, second.packed.length), (65, 64))
        self.assertNotEqual(appended._pages[0].data_ptr(), second.packed._pages[0].data_ptr())
        torch.testing.assert_close(second.packed.to_dense(), before)
        self.assertLessEqual(cache.bytes, cache.max_bytes)

    def test_reservation_tracks_tensor_objects_and_actual_clone_storage(self):
        base = torch.arange(128, dtype=torch.float32)
        first, second = base[:4], base[4:8]
        self.assertEqual(snapshot_requirements([first, first], None)[1], 16)
        self.assertEqual(snapshot_requirements([first, second], None)[1], 32)
        # Separate target/draft clone calls must not share a mutable copy.
        self.assertEqual(snapshot_requirements(first, first)[1], 32)
        views = [base[:4], base[::3], base.reshape(16, 8).T, base.reshape(16, 8).diagonal()]
        for view in views:
            with self.subTest(shape=view.shape, stride=view.stride()):
                self.assertEqual(snapshot_requirements(view, None)[1], view.clone().untyped_storage().nbytes())

    def test_budget_skip_preserves_old_shallow_heads_without_partial_eviction(self):
        cache = ContinuousMTPPrefixCache(64)
        self.assertTrue(publish(cache, self.a, count=4))
        self.assertTrue(publish(cache, self.b, count=4))
        before, generation = list(cache.entries), cache.generation
        self.assertFalse(publish(cache, self.a, 128, count=8))
        self.assertEqual([id(item) for item in cache.entries], [id(item) for item in before])
        self.assertEqual(cache.generation, generation)
        self.assertEqual(cache.checkout(self.a, 64)[2], 64)
        self.assertEqual(cache.checkout(self.b, 64)[2], 64)
        self.assertEqual(cache.snapshot()['budget'], 1)
        self.assertEqual(cache.snapshot()['eviction'], 0)

    def test_third_branch_can_replace_old_protection_and_hit_refreshes_head(self):
        cache = ContinuousMTPPrefixCache(64, max_entries=2)
        publish(cache, self.a)
        publish(cache, self.b)
        cache.checkout(self.a, 64)
        self.assertTrue(publish(cache, self.c))
        self.assertEqual(cache.checkout(self.a, 64)[2], 64)
        self.assertEqual(cache.checkout(self.b, 64)[2], 0)
        self.assertEqual(cache.checkout(self.c, 64)[2], 64)
        self.assertEqual(cache.snapshot()['eviction'], 1)
        self.assertEqual(cache.bytes, 64)

    def test_only_actual_hit_updates_lru_ancestor_does_not_pin_deeper_head(self):
        cache = ContinuousMTPPrefixCache(65536, max_entries=3)
        publish(cache, self.a, 64)
        publish(cache, self.a, 128)
        publish(cache, self.b, 64)
        cache.checkout(self.a[:65], 64)
        self.assertTrue(publish(cache, self.c))
        self.assertEqual(cache.checkout(self.a, 64)[2], 64)
        self.assertEqual(cache.checkout(self.b, 64)[2], 64)
        self.assertEqual(cache.checkout(self.c, 64)[2], 64)
        self.assertEqual(len(cache.entries), 3)

    def test_oversize_does_not_clone_or_erase_valid_branch(self):
        cache = ContinuousMTPPrefixCache(32)
        publish(cache, self.a)
        generation = cache.generation
        with patch('engine.continuous_mtp.clone_state') as clone:
            self.assertFalse(publish(cache, self.b, count=16))
            clone.assert_not_called()
        self.assertEqual(cache.checkout(self.a, 64)[2], 64)
        self.assertEqual(cache.generation, generation)
        self.assertEqual(cache.snapshot()['oversize'], 1)

    def test_clone_failure_keeps_valid_pool_and_evicted_tensors_really_release(self):
        cache = ContinuousMTPPrefixCache(65536, max_entries=2)
        publish(cache, self.a)
        publish(cache, self.b)
        generation = cache.generation
        old_tensor = weakref.ref(cache.entries[0].target.bank)
        with patch('engine.continuous_mtp.clone_state', side_effect=RuntimeError('copy failed')):
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                publish(cache, self.c)
        self.assertIsNone(old_tensor())
        self.assertEqual(cache.checkout(self.b, 64)[2], 64)
        self.assertEqual(cache.checkout(self.c, 64)[2], 0)
        self.assertEqual(cache.generation, generation+1)
        self.assertEqual(cache.snapshot()['entries'], 1)
        self.assertEqual(cache.snapshot()['protected_heads'], 1)
        self.assertEqual(cache.snapshot()['clone_failure'], 1)
        self.assertEqual(cache.bytes, 32)

    def test_clone_failure_without_eviction_preserves_entries_and_heads(self):
        cache = ContinuousMTPPrefixCache(65536)
        publish(cache, self.a)
        generation = cache.generation
        with patch('engine.continuous_mtp.clone_state', side_effect=RuntimeError('copy failed')):
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                publish(cache, self.b)
        self.assertEqual(cache.generation, generation)
        self.assertEqual(cache.snapshot()['protected_heads'], 1)
        self.assertEqual(cache.checkout(self.a, 64)[2], 64)

    def test_clear_and_metrics_are_bounded_and_generation_changes_only_on_mutation(self):
        cache = ContinuousMTPPrefixCache(65536, max_entries=2)
        self.assertEqual(cache.max_bytes, 65536)
        self.assertEqual(ContinuousMTPPrefixCache().max_bytes, 1024**3)
        publish(cache, self.a)
        count_keys = set(cache.snapshot())
        for ids in (self.b, self.c, self.a)*20:
            publish(cache, ids)
        self.assertLessEqual(len(cache.entries), 2)
        self.assertLessEqual(len(cache._heads), 2)
        self.assertEqual(set(cache.snapshot()), count_keys)
        generation = cache.generation
        cache.clear()
        self.assertEqual(cache.generation, generation+1)
        self.assertEqual((cache.bytes, len(cache.entries), len(cache._heads)), (0, 0, 0))
        cache.clear()
        self.assertEqual(cache.generation, generation+1)


if __name__ == '__main__':
    unittest.main()
