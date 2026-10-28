"""Independent autoregressive and ownership oracles for batch MTP control.

Toy batch backends are contract fixtures, not native GPU arithmetic evidence.
"""
from types import SimpleNamespace
import threading
import pytest
import torch

from engine.batched_mtp import BatchedMTPRunner
from engine.request_control import GenerationCancelled
from test_continuous_mtp import ToyTarget, ToyDraft, greedy_gold, history_hash
from test_four_context_isolation import private_digest, mutable_storages, fingerprint


class ToyBatchTarget:
    def __init__(self, target):
        self.target = target; self.calls = []; self.after = None

    def verify(self, ids, states):
        self.calls.append([list(v) for v in ids]); outputs = []
        for tokens, state in zip(ids, states):
            logits, updated, hidden = self.target.verify_tokens(tokens, state)
            outputs.append((logits.clone(), updated, SimpleNamespace(multi=hidden.multi.clone())))
        if self.after:
            self.after()
        return outputs

    def snapshot(self):
        return dict(calls=len(self.calls))


class ToyBatchDraft:
    def __init__(self, draft):
        self.draft = draft; self.calls = 0

    def forward(self, ids, hidden, states):
        self.calls += 1; outputs = []
        for tokens, value, state in zip(ids, hidden, states):
            out = self.draft(tokens, value, state)
            outputs.append(SimpleNamespace(logits=out.logits.clone(), state=out.state, multi=out.multi.clone()))
        return outputs


def runner(mode='accept', depth='adaptive'):
    target, draft = ToyTarget(), ToyDraft(mode)
    batch_target = ToyBatchTarget(target); batch_draft = ToyBatchDraft(draft)
    return BatchedMTPRunner(target, draft, draft_tokens=depth,
                            batch_target=batch_target, batch_draft=batch_draft)


def prefilled(r, prompts, budgets, eos=None, flags=None):
    sessions = [r.start(ids, budget, {255} if eos is None else stop, 64,
                        should_cancel=None if flags is None else flag.is_set)
                for ids, budget, stop, flag in zip(prompts, budgets, eos or [{255}]*len(prompts),
                                                  flags or [None]*len(prompts))]
    for s in sessions:
        while s.phase == 'prefill':
            r.prefill(s)
    return sessions


def finish_all(r, sessions):
    while any(not s.done for s in sessions):
        live = [s for s in sessions if not s.done]
        assert r.decode_batch(live) == [None]*len(live)


@pytest.mark.parametrize('lanes', [2, 3, 4])
@pytest.mark.parametrize('mode', ['accept', 'first_reject', 'second_reject', 'alternate'])
@pytest.mark.parametrize('depth', [1, 2, 'adaptive'])
@pytest.mark.parametrize('budget', [2, 3, 4, 7, 16])
def test_batched_acceptance_rejection_eos_and_budget_match_independent_greedy(lanes, mode, depth, budget):
    r = runner(mode, depth)
    prompts = [[3+i, 11+i, 27+i] for i in range(lanes)]
    budgets = [budget]*lanes
    sessions = prefilled(r, prompts, budgets)
    try:
        finish_all(r, sessions)
        for s, ids, cap in zip(sessions, prompts, budgets):
            assert s.generated == greedy_gold(ids, cap)
            assert s.state.tokens == ids+s.generated[:-1]
            assert int(s.state.bank[0]) == history_hash(s.state.tokens)
            assert s.state.prefixes is None
            assert s.stats['proposed'] == s.stats['accepted']+s.stats['corrections']+s.stats['discarded']
        if budget >= 4:
            assert any(len(call) > 1 for call in r.batch_target.calls)
            assert any(s.result(0)['speculative']['batched_verification'] for s in sessions)
        storages = [mutable_storages((s.state, s.draft_state, s.draft_multi)) for s in sessions]
        assert all(not (a & b) for i, a in enumerate(storages) for b in storages[:i])
    finally:
        for s in sessions:
            r.close(s)


@pytest.mark.parametrize('mode', ['accept', 'first_reject', 'second_reject'])
@pytest.mark.parametrize('index', [1, 2, 3, 4])
def test_each_lane_stops_at_its_own_eos_prefix(mode, index):
    r = runner(mode, 2); prompts = [[3+i, 5, 8] for i in range(4)]
    gold = [greedy_gold(p, 12) for p in prompts]
    stops = [{v[index]} for v in gold]
    sessions = prefilled(r, prompts, [12]*4, stops)
    finish_all(r, sessions)
    for s, ids, eos in zip(sessions, prompts, stops):
        assert s.generated == greedy_gold(ids, 12, eos)
        assert s.result(0)['finish_reason'] == 'stop'


@pytest.mark.parametrize('cancelled', [0, 1, 2, 3, 'all'])
@pytest.mark.parametrize('when', ['before_batch', 'after_verification'])
def test_cancel_one_lane_keeps_peers_and_owned_state_unchanged(cancelled, when):
    r = runner(); flags = [threading.Event() for _ in range(4)]
    prompts = [[3+i, 5, 8] for i in range(4)]
    sessions = prefilled(r, prompts, [20]*4, flags=flags)
    indices = list(range(4)) if cancelled == 'all' else [cancelled]
    def state_digest(s):
        return fingerprint((s.state,s.draft_state,s.draft_multi,s.proposal,s.generated,s.closed))
    before = {i: state_digest(sessions[i]) for i in indices}
    def cancel():
        for i in indices:
            flags[i].set()
    if when == 'before_batch':
        cancel()
    else:
        r.batch_target.after = cancel
    errors = r.decode_batch(sessions)
    r.batch_target.after = None
    for i in range(4):
        assert isinstance(errors[i], GenerationCancelled) if i in indices else errors[i] is None
    for i in indices:
        # Disposable verification cannot replace a cancelled lane's history.
        assert state_digest(sessions[i]) == before[i]
        r.close(sessions[i])
    survivors = [s for i, s in enumerate(sessions) if i not in indices]
    finish_all(r, survivors)
    for i, s in enumerate(sessions):
        if i not in indices:
            assert s.generated == greedy_gold(prompts[i], 20)
            r.close(s)


def test_mixed_depths_and_per_lane_reasoning_limits_preserve_future_draft_state():
    r = runner(); prompts = [[3+i, 5, 8] for i in range(4)]
    sessions = prefilled(r, prompts, [20]*4)
    class FixedPolicy:
        def __init__(self, depth):self.value = depth
        def depth(self):return self.value
        def observe(self, first):pass
    for s, depth in zip(sessions,[1,1,2,2]):s.policy = FixedPolicy(depth)
    before = [len(s.generated) for s in sessions]
    assert r.decode_batch(sessions, emission_limits=[1, 2, 3, 1]) == [None]*4
    assert [len(s.generated)-n for s, n in zip(sessions, before)] == [1, 2, 3, 1]
    finish_all(r, sessions)
    for s, prompt in zip(sessions, prompts):assert s.generated == greedy_gold(prompt, 20)


@pytest.mark.parametrize('mode', ['accept','first_reject','second_reject','alternate'])
@pytest.mark.parametrize('depth', [1,2,'adaptive'])
def test_mixed_remaining_budgets_can_change_batch_shape_without_changing_outputs(mode,depth):
    r=runner(mode,depth);prompts=[[3+i,5,8] for i in range(4)];budgets=[3,5,8,11]
    sessions=prefilled(r,prompts,budgets);finish_all(r,sessions)
    for s,prompt,budget in zip(sessions,prompts,budgets):
        assert s.generated==greedy_gold(prompt,budget)
        assert s.state.tokens==prompt+s.generated[:-1]


def test_forced_token_is_private_and_follows_real_shifted_draft_positions():
    r = runner(); prompts = [[3+i, 5, 8] for i in range(4)]
    sessions = prefilled(r, prompts, [20]*4)
    first = sessions[1].generated[0]
    assert r.decode_batch(sessions, forced_tokens=[None, 211, None, None]) == [None]*4
    assert sessions[1].generated[:2] == [first, 211]
    finish_all(r, sessions)
    assert sessions[1].generated == [first, 211]+greedy_gold(prompts[1]+[first, 211], 18)
    for i in (0, 2, 3):assert sessions[i].generated == greedy_gold(prompts[i], 20)


def test_shared_cuda_failure_stops_other_shape_groups_without_committing_or_closing():
    r = runner(depth=1); sessions = prefilled(r, [[3+i, 5, 8] for i in range(4)], [20]*4)
    def fail(*args):raise RuntimeError('CUDA test failure')
    r.batch_target.verify = fail
    before = [private_digest(s) for s in sessions]
    errors = r.decode_batch(sessions)
    assert all(isinstance(e, RuntimeError) for e in errors)
    assert [private_digest(s) for s in sessions] == before
    assert not any(s.closed for s in sessions)


@pytest.mark.parametrize('bad', [0, -1, True])
def test_invalid_batch_controls_fail_before_any_model_work(bad):
    r = runner(); sessions = prefilled(r, [[3, 5, 8]]*2, [12]*2)
    before = [private_digest(s) for s in sessions]
    with pytest.raises(ValueError):r.decode_batch(sessions, emission_limits=[bad, None])
    assert [private_digest(s) for s in sessions] == before
