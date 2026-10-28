"""CPU ownership/state-machine oracles; no CUDA arithmetic or speed claims."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from engine.continuous_mtp import ContinuousMTPPrefixCache, ContinuousMTPRunner
from engine.decode_graph import DecodeGraphBlock
from engine.decoder import DecoderState
from engine.ops.gdn import GDNState
from engine.prefix_cache import clone_state
from engine.request_control import GenerationCancelled
from engine.speculative import generate_mtp


def history_hash(ids, initial=7):
    value = initial
    for token in ids:
        value = (31*value + int(token)) % 1000003
    return value


def next_token(value):
    return 1 + value % 200


def greedy_gold(ids, budget, eos=()):
    """Independent, tensor-free autoregressive reference."""
    value, out = history_hash(ids), []
    for _ in range(budget):
        token = next_token(value)
        out.append(token)
        if token in eos:
            break
        value = history_hash([token], value)
    return out


class ToyTarget:
    """Shared Graph-like banks overwrite on every call; state encodes history."""
    def __init__(self):
        self.embedding = torch.empty(0)
        self.calls, self.verify_calls = [], []
        self.bank = torch.zeros(1, dtype=torch.int64)
        self.hidden_bank = torch.zeros(1, 3, dtype=torch.int64)

    def _hidden(self, tokens, start):
        rows = len(tokens) - start
        if len(self.hidden_bank) < rows:
            self.hidden_bank = torch.empty(rows, 3, dtype=torch.int64)
        value = history_hash(tokens[:start])
        for offset, token in enumerate(tokens[start:]):
            value = history_hash([token], value)
            self.hidden_bank[offset] = torch.tensor([start+offset, value, 0])
        return SimpleNamespace(multi=self.hidden_bank[:rows])

    @staticmethod
    def _logits(tokens):
        logits = torch.full((1, 256), -100.)
        logits[0, next_token(history_hash(tokens))] = 100.
        return logits

    @staticmethod
    def _check(state):
        if state is not None:
            assert int(state.bank[0]) == history_hash(state.tokens), 'Target Graph bank overwritten'

    def __call__(self, ids, state=None, return_hidden=False, prefill_chunk=False,
                 owned_graph_cache=False):
        self._check(state)
        old = [] if state is None else list(state.tokens)
        tokens = old + list(ids)
        self.calls.append(dict(ids=list(ids), prefill=prefill_chunk, owned=owned_graph_cache))
        self.bank.fill_(history_hash(tokens))
        bank = self.bank.clone() if prefill_chunk or owned_graph_cache else self.bank
        updated = SimpleNamespace(tokens=tokens, bank=bank, prefixes=None)
        result = self._logits(tokens), updated
        return (*result, self._hidden(tokens, len(old))) if return_hidden else result

    def verify_tokens(self, ids, state, borrowed_graph_cache=False):
        self._check(state)
        assert not borrowed_graph_cache, 'Cooperative sessions must own verification banks'
        self.verify_calls.append(list(ids))
        tokens, prefixes, logits = list(state.tokens), [], []
        for token in ids:
            tokens = tokens + [token]
            self.bank.fill_(history_hash(tokens))
            prefixes.append(SimpleNamespace(tokens=tokens, bank=self.bank.clone(), prefixes=None))
            logits.append(self._logits(tokens))
        updated = SimpleNamespace(tokens=tokens, bank=self.bank.clone(), prefixes=prefixes)
        return torch.cat(logits), updated, self._hidden(tokens, len(state.tokens))


class ToyDraft:
    """Checks shifted positions and hash continuity, including temporary futures."""
    def __init__(self, mode='accept'):
        self.mode = mode
        self.calls = []
        self.multi_bank = torch.empty(1, 3, dtype=torch.int64)

    def __call__(self, ids, hidden, state=None, prefill_chunk=False):
        start = 0 if state is None else state.length
        torch.testing.assert_close(hidden[:, 0], torch.arange(start, start+len(ids)))
        recursive = bool((hidden[:, 2] == 777).all())
        if state is not None:
            assert not state.temporary, 'Rejected future entered committed draft state'
            assert int(hidden[0, 1]) == state.last_hash, 'Wrong request/target hidden entered draft'
        assert recursive or bool((hidden[:, 2] == 0).all())
        if len(self.multi_bank) < len(ids):
            self.multi_bank = torch.empty(len(ids), 3, dtype=torch.int64)
        # Read all input before writing, just like an out-of-place native op.
        hashes = [history_hash([token], int(row[1])) for token, row in zip(ids, hidden)]
        for i, value in enumerate(hashes):
            self.multi_bank[i] = torch.tensor([start+i+1, value, 777])
        canonical = 0 if state is None else state.canonical
        wrong = (self.mode == 'first_reject' and not recursive or
                 self.mode == 'second_reject' and recursive or
                 self.mode == 'alternate' and (canonical + int(recursive)) % 3 == 1)
        token = next_token(hashes[-1])
        if wrong:
            token = 1 + token % 200
        logits = torch.full((1, 256), -100.)
        logits[0, token] = 100.
        updated = SimpleNamespace(length=start+len(ids), temporary=recursive,
            last_hash=hashes[-1], canonical=canonical+int(not recursive),
            shifted=([] if state is None else list(state.shifted))+list(ids),
            bank=torch.tensor([hashes[-1]]))
        self.calls.append(dict(start=start, ids=list(ids), recursive=recursive,
                               prefill=prefill_chunk, hidden=hidden.clone()))
        return SimpleNamespace(logits=logits, state=updated, multi=self.multi_bank[:len(ids)])


def make_runner(cache=None, draft_tokens='adaptive', policy_factory=None, mode='accept', low_sync=True):
    return ContinuousMTPRunner(ToyTarget(), ToyDraft(mode), cache, draft_tokens=draft_tokens,
                               policy_factory=policy_factory, low_sync=low_sync)


def finish(runner, ids, budget=12, eos=(255,), **kwargs):
    session = runner.start(ids, budget, eos, 64, **kwargs)
    while not session.done:
        if session.phase == 'prefill':
            runner.prefill(session)
        else:
            runner.decode([session])
        session.progress()
        session.emit()
    return session


class ContinuousMTPTests(unittest.TestCase):
    def assert_committed(self, session, prompt, expected):
        self.assertEqual(session.generated, expected)
        self.assertEqual(session.state.tokens, prompt+expected[:-1])
        self.assertEqual(int(session.state.bank[0]), history_hash(session.state.tokens))
        self.assertIsNone(session.state.prefixes)

    def test_all_acceptance_modes_depths_budgets_and_serial_oracle(self):
        for mode in ('accept', 'first_reject', 'second_reject', 'alternate'):
            for depth in (1, 2, 'adaptive'):
                for budget in range(1, 15):
                    for low_sync in (False, True):
                        with self.subTest(mode=mode, depth=depth, budget=budget, low_sync=low_sync):
                            prompt = [3, 5, 8]
                            runner = make_runner(draft_tokens=depth, mode=mode, low_sync=low_sync)
                            snapshots = []
                            session = finish(runner, prompt, budget, on_tokens=snapshots.append)
                            expected = greedy_gold(prompt, budget)
                            self.assert_committed(session, prompt, expected)
                            row, state = generate_mtp(ToyTarget(), ToyDraft(mode), prompt, budget,
                                {255}, 64, draft_tokens=depth, low_sync=low_sync)
                            self.assertEqual(row['generated_ids'], expected)
                            self.assertEqual(state.tokens, session.state.tokens)
                            self.assertEqual(snapshots[-1], tuple(expected))
                            self.assertTrue(all(a == b[:len(a)] for a, b in zip(snapshots, snapshots[1:])))
                            stats = session.result(0)['speculative']
                            self.assertEqual(stats['proposed'], stats['accepted']+stats['corrections']+stats['discarded'])
                            self.assertEqual(budget-1, stats['accepted']+stats['bonus_tokens']+stats['corrections']+stats['single_steps'])
                            self.assertEqual(stats['scheduling'], 'cooperative_round_robin')
                            self.assertFalse(stats['batched_verification'])
                            self.assertEqual(stats['policy']['strategy'],
                                             'acceptance' if depth == 'adaptive' else 'fixed')
                            if depth != 'adaptive':
                                self.assertEqual(stats['policy']['depth'], depth)
                            if budget < 3:
                                self.assertEqual(runner.draft.calls, [])

    def test_eos_first_proposal_second_proposal_bonus_and_correction(self):
        prompt = [3, 5, 8]
        sequence = greedy_gold(prompt, 8)
        self.assertEqual(len(set(sequence[:4])), 4)
        for index in range(4):
            for mode in ('accept', 'first_reject', 'second_reject'):
                with self.subTest(index=index, mode=mode):
                    runner = make_runner(mode=mode, draft_tokens=2)
                    session = finish(runner, prompt, 12, {sequence[index]})
                    self.assert_committed(session, prompt, sequence[:index+1])
                    self.assertEqual(session.result(0)['finish_reason'], 'stop')
                    if index == 0:
                        self.assertEqual(runner.draft.calls, [])

    def test_interleaved_prefill_rounds_and_slot_reuse_match_serial(self):
        runner = make_runner(mode='second_reject')
        prompts = ([3, 5]*66, [8, 2]*35, [11, 4]*33)
        a = runner.start(prompts[0], 14, {255}, 64)
        b = runner.start(prompts[1], 5, {255}, 64)
        runner.prefill(a); runner.prefill(b)
        self.assertEqual((a.position, b.position), (64, 64))
        runner.prefill(a); runner.prefill(b)
        self.assertEqual((a.phase, b.phase), ('prefill', 'decode'))
        runner.decode([b]); runner.prefill(a)
        # B's draft overwrites the shared draft output workspace before A resumes.
        a_multi = a.draft_multi.clone()
        runner.decode([b])
        torch.testing.assert_close(a.draft_multi, a_multi)
        runner.decode([a])
        while not b.done:
            runner.decode([b])
        self.assert_committed(b, prompts[1], greedy_gold(prompts[1], 5))
        runner.close(b)
        c = runner.start(prompts[2], 9, {255}, 64)
        while c.phase == 'prefill':
            runner.prefill(c)
            if not a.done:
                runner.decode([a])
        while not a.done or not c.done:
            ready = [s for s in (c, a) if not s.done]
            runner.decode(ready)
        for session, prompt, budget in ((a, prompts[0], 14), (c, prompts[2], 9)):
            self.assert_committed(session, prompt, greedy_gold(prompt, budget))
            torch.testing.assert_close(session.state.bank, finish(make_runner(mode='second_reject'), prompt, budget).state.bank)
        self.assertTrue(all(call['owned'] for call in runner.engine.calls))
        self.assertTrue(all(2 <= len(ids) <= 3 for ids in runner.engine.verify_calls))
        self.assertIsNone(b.state)
        self.assertIsNone(b.draft_multi)

    def test_actual_m1_m2_m3_graph_snapshot_paths_survive_other_requests(self):
        # Exercise the real ownership code with a CPU replay substitute. This
        # deliberately does not claim to capture or execute a CUDA Graph.
        for width in (1, 2, 3):
            with self.subTest(width=width):
                block = DecodeGraphBlock([SimpleNamespace(layer_type='linear_attention', ple=None)],
                                         return_prefixes=width > 1)
                block.pending = (torch.zeros(width, 1), None, None)
                prefixes = [DecoderState(GDNState(torch.zeros(1), torch.zeros(1))) for _ in range(width)]
                block.states = [DecoderState(GDNState(torch.zeros(1), torch.zeros(1)), prefixes=prefixes)]
                block.output = block.pending
                def replay():
                    conv = block.states[0].attention.conv.clone()
                    recurrent = block.states[0].attention.recurrent.clone()
                    for index, value in enumerate(block.pending[0]):
                        conv = conv + value
                        recurrent = recurrent + 2*value
                        prefixes[index].attention.conv.copy_(conv)
                        prefixes[index].attention.recurrent.copy_(recurrent)
                    block.states[0].attention.conv.copy_(conv)
                    block.states[0].attention.recurrent.copy_(recurrent)
                block.graph = SimpleNamespace(replay=replay)
                def states(value):
                    return [DecoderState(GDNState(torch.tensor([value]), torch.tensor([value])))]
                _, a = block((torch.ones(width, 1), None, None), states(100.), owned=width == 1)
                committed = a[0].prefixes[0] if width > 1 else a[0]
                _, b = block((torch.full((width, 1), 2.), None, None), states(200.), owned=width == 1)
                self.assertEqual(committed.attention.conv.item(), 101.)
                self.assertEqual(b[0].attention.conv.item(), 200.+2*width)
                self.assertNotEqual(committed.attention.conv.data_ptr(), block.states[0].attention.conv.data_ptr())
                _, resumed = block((torch.ones(width, 1), None, None), [committed], owned=width == 1)
                self.assertEqual(resumed[0].attention.conv.item(), 101.+width)
                self.assertEqual(b[0].attention.conv.item(), 200.+2*width)

    def test_final_m1_banks_and_warm_cache_small_budgets_stay_private(self):
        cache = ContinuousMTPPrefixCache(65536)
        runner = make_runner(cache)
        prompt = [3, 5]*65
        warm = finish(runner, prompt, 5)
        runner.close(warm)
        a = runner.start(prompt, 2, {255}, 64)
        b = runner.start([8, 2], 2, {255}, 64)
        before = len(runner.draft.calls)
        runner.prefill(a); runner.prefill(b)
        runner.decode([a]); runner.decode([b])
        self.assertEqual(len(runner.draft.calls), before)
        self.assert_committed(a, prompt, greedy_gold(prompt, 2))
        self.assert_committed(b, [8, 2], greedy_gold([8, 2], 2))
        self.assertIsNone(a.draft_state)
        self.assertIsNone(b.draft_state)

    def test_cache_pairs_lookahead_private_checkout_and_multiple_writers(self):
        cache = ContinuousMTPPrefixCache(65536)
        runner = make_runner(cache)
        prompt = [3, 5]*97
        warm = finish(runner, prompt, 7)
        runner.close(warm)
        self.assertEqual(cache.entries[-1].end, 192)
        a = runner.start(prompt, 7, {255}, 64)
        b = runner.start([9, 2]*97, 7, {255}, 64)
        self.assertEqual((a.reused, b.reused), (192, 0))
        target, draft, reused, _ = cache.checkout(prompt, 64)
        a.state.bank.add_(10)
        a.draft_state.bank.add_(10)
        self.assertNotEqual(a.state.bank.item(), target.bank.item())
        self.assertNotEqual(a.draft_state.bank.item(), draft.bank.item())
        self.assertEqual(cache.entries[-1].target.bank.item(), target.bank.item())
        self.assertEqual(cache.entries[-1].draft.bank.item(), draft.bank.item())
        # Restore only the test's intentional corruption, then resume checkout.
        a.state, a.draft_state = target, draft
        runner.prefill(a)
        runner.prefill(b)
        self.assertEqual(b.checkpoint_tokens, 64)
        c = runner.start([7, 1]*97, 7, {255}, 64)
        runner.prefill(c)
        self.assertEqual(c.checkpoint_tokens, 64)
        # Both cold branches publish while A retains its own live checkout.
        # Switching branches does not erase A's earlier completed checkpoint.
        self.assertEqual(cache.checkout(prompt, 64)[2], 192)
        self.assertEqual(cache.checkout([9, 2]*97, 64)[2], 64)
        self.assertEqual(cache.checkout([7, 1]*97, 64)[2], 64)
        runner.close(b)
        self.assertEqual(cache.checkout([9, 2]*97, 64)[2], 64)
        runner.prefill(c)
        self.assertEqual(c.checkpoint_tokens, 128)
        self.assertLessEqual(cache.bytes, cache.max_bytes)
        while not a.done:
            runner.decode([a])
        self.assert_committed(a, prompt, greedy_gold(prompt, 7))
        runner.close(c); runner.close(a)
        repeat = finish(runner, prompt, 7)
        self.assertEqual(repeat.reused, 192)
        self.assert_committed(repeat, prompt, greedy_gold(prompt, 7))
        runner.close(repeat)
        changed = list(prompt); changed[64] += 1
        isolated = ContinuousMTPPrefixCache(65536)
        isolated_runner = make_runner(isolated)
        warm = finish(isolated_runner, prompt, 3)
        isolated_runner.close(warm)
        self.assertEqual(isolated.checkout(changed, 64)[2], 0)
        self.assertEqual(isolated.checkout(prompt, 128)[2], 0)

    def test_cancel_after_pair_published_then_reuse_slot_and_cache(self):
        runner = make_runner(ContinuousMTPPrefixCache(65536))
        cancelled = [False]
        prompt = [3, 5]*97
        a = runner.start(prompt, 14, {255}, 64, should_cancel=lambda: cancelled[0])
        runner.prefill(a)
        self.assertEqual(a.checkpoint_tokens, 64)
        b = runner.start([4, 9], 8, {255}, 64)
        runner.prefill(b)
        cancelled[0] = True
        with self.assertRaises(GenerationCancelled) as caught:
            runner.prefill(a)
        self.assertEqual(caught.exception.checkpoint_tokens, 64)
        runner.close(a)
        self.assertIsNone(runner.cache.owner)
        c = runner.start(prompt, 9, {255}, 64)
        self.assertEqual(c.reused, 64)
        while not b.done or not c.done:
            if c.phase == 'prefill':
                runner.prefill(c)
            for session in (b, c):
                if session.phase == 'decode':
                    runner.decode([session])
        self.assert_committed(b, [4, 9], greedy_gold([4, 9], 8))
        self.assert_committed(c, prompt, greedy_gold(prompt, 9))
        with self.assertRaises(ValueError):
            runner.prefill(a)

    def test_cancel_at_completed_round_and_callbacks_are_external(self):
        runner = make_runner()
        flag = [False]
        def bad(*args):
            raise ValueError('callback failed')
        a = runner.start([3, 5], 10, {255}, 64, on_progress=bad, on_tokens=bad,
                         should_cancel=lambda: flag[0])
        b = runner.start([8, 2], 10, {255}, 64)
        runner.prefill(a); runner.prefill(b)
        runner.decode([a, b])
        self.assertGreater(len(a.generated), 1)
        self.assertGreater(len(b.generated), 1)
        with self.assertRaisesRegex(ValueError, 'callback'):
            a.progress()
        with self.assertRaisesRegex(ValueError, 'callback'):
            a.emit()
        original = runner.draft
        class CancelOnDraft:
            def __call__(self, *args, **kwargs):
                output = original(*args, **kwargs)
                flag[0] = True
                return output
        runner.draft = CancelOnDraft()
        with self.assertRaises(GenerationCancelled):
            runner.decode([a])
        self.assertEqual(a.state.tokens, a.ids+a.generated[:-1])
        runner.close(a)
        runner.draft = original
        while not b.done:
            runner.decode([b])
        self.assert_committed(b, [8, 2], greedy_gold([8, 2], 10))

    def test_adaptive_policy_private_and_cost_round_sync_protocol(self):
        runner = make_runner(mode='first_reject')
        a = runner.start([3, 5], 12, {255}, 64)
        b = runner.start([8, 2], 12, {255}, 64)
        self.assertIsNot(a.policy, b.policy)
        runner.prefill(a); runner.prefill(b)
        for _ in range(4):
            runner.decode([a])
        self.assertEqual(a.policy.depth(), 1)
        self.assertEqual(len(b.policy.recent), 0)
        class CostPolicy:
            def __init__(self):
                self.rows = []
            def depth(self):
                return 2
            def observe_round(self, first_accepted, **row):
                self.rows.append(dict(first_accepted=first_accepted, **row))
        cost = make_runner(policy_factory=CostPolicy)
        c = cost.start([3, 5], 9, {255}, 64)
        cost.prefill(c)
        with patch('engine.continuous_mtp._sync') as sync:
            cost.decode([c])
        self.assertEqual(sync.call_count, 2)
        row = c.policy.rows[0]
        self.assertEqual((row['depth'], row['accepted_tokens']), (2, 3))
        self.assertGreater(row['elapsed_seconds'], 0)
        from engine.speculative_policy import CostAwareDraftDepth
        result = finish(make_runner(policy_factory=CostAwareDraftDepth), [3, 5], 9).result(0)
        self.assertEqual(result['speculative']['policy']['strategy'], 'committed_tokens_per_second')
        self.assertGreater(result['speculative']['policy']['rounds'], 0)
        with self.assertRaises(ValueError):
            make_runner(draft_tokens=1, policy_factory=CostPolicy)

    def test_nonfinite_rejected_future_does_not_commit_and_other_session_finishes(self):
        runner = make_runner(mode='first_reject', draft_tokens=2)
        a = runner.start([3, 5], 9, {255}, 64)
        b = runner.start([8, 2], 9, {255}, 64)
        runner.prefill(a); runner.prefill(b)
        before = clone_state(a.state)
        verify = runner.engine.verify_tokens
        def nonfinite(*args, **kwargs):
            logits, state, hidden = verify(*args, **kwargs)
            logits[-1, 0] = float('nan')
            return logits, state, hidden
        with patch.object(runner.engine, 'verify_tokens', side_effect=nonfinite):
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
                runner.decode([a])
        self.assertEqual(a.state.tokens, before.tokens)
        torch.testing.assert_close(a.state.bank, before.bank)
        runner.close(a)
        while not b.done:
            runner.decode([b])
        self.assert_committed(b, [8, 2], greedy_gold([8, 2], 9))

    def test_small_cache_disables_oversized_snapshots_and_cache_opt_out(self):
        cache = ContinuousMTPPrefixCache(1)
        runner = make_runner(cache)
        session = finish(runner, [3, 5]*65, 7)
        self.assertEqual(cache.bytes, 0)
        self.assertEqual(session.checkpoint_tokens, 0)
        cache = ContinuousMTPPrefixCache(65536)
        runner = make_runner(cache)
        session = finish(runner, [3, 5]*65, 7, cache=False)
        self.assertEqual(cache.bytes, 0)
        self.assertEqual(session.reused, 0)


if __name__ == '__main__':
    unittest.main()
