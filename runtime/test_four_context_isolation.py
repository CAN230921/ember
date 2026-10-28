"""Four cached private sessions and cancellation/accounting CPU regressions.

Toy-state and ASGI evidence only; no GPU capacity or real-network claims.
"""
import hashlib
import random
import threading
from types import SimpleNamespace

import pytest
import torch

from test_continuous_mtp import make_runner,finish,greedy_gold,history_hash
import test_four_context_api as api_fixtures
from engine.continuous_mtp import ContinuousMTPPrefixCache
from engine.request_control import GenerationCancelled


def fingerprint(value):
    def encode(value):
        if isinstance(value,torch.Tensor):
            return (str(value.dtype),tuple(value.shape),value.detach().contiguous().numpy().tobytes())
        if isinstance(value,SimpleNamespace):return encode(vars(value))
        if isinstance(value,dict):return tuple((key,encode(item)) for key,item in sorted(value.items()))
        if isinstance(value,(list,tuple)):return tuple(encode(item) for item in value)
        return value
    return hashlib.sha256(repr(encode(value)).encode()).hexdigest()


def private_digest(session):
    return fingerprint((session.state,session.draft_state,session.draft_multi,
        session.proposal,session.position,session.generated,session.stats,session.phase,session.closed))


def mutable_storages(value):
    if isinstance(value,torch.Tensor):return {value.untyped_storage().data_ptr()}
    if isinstance(value,SimpleNamespace):return mutable_storages(vars(value))
    if isinstance(value,dict):return set().union(*(mutable_storages(item) for item in value.values()))
    if isinstance(value,(list,tuple)):return set().union(*(mutable_storages(item) for item in value))
    return set()


@pytest.mark.parametrize('mode',['accept','first_reject','second_reject','alternate'])
@pytest.mark.parametrize('depth',[1,2,'adaptive'])
@pytest.mark.parametrize('cancel_phase',['prefill','after_verify'])
def test_four_cached_states_survive_peer_cancel_and_graph_bank_overwrite(mode,depth,cancel_phase):
    cache=ContinuousMTPPrefixCache(max_bytes=2**20)
    runner=make_runner(cache,draft_tokens=depth,mode=mode)
    shared=[1,2,3]*43
    warm=finish(runner,shared+[177]*64,budget=30,eos={255},cache=True)
    runner.close(warm)
    prompts=[shared+[marker]*64 for marker in range(21,25)]
    flags=[threading.Event() for _ in range(4)]
    sessions=[runner.start(ids,30,{255},64,cache=True,should_cancel=flag.is_set)
        for ids,flag in zip(prompts,flags)]
    cancelled=(0 if depth==1 else 1 if depth==2 else 3)
    try:
        assert all(session.reused==128 for session in sessions)
        storage_sets=[mutable_storages((s.state,s.draft_state)) for s in sessions]
        for a in range(4):
            for b in range(a):assert not (storage_sets[a]&storage_sets[b])
        live=list(range(4));order=random.Random(20261011)
        def step(index):
            others={i:private_digest(sessions[i]) for i in live if i!=index}
            session=sessions[index]
            if session.phase=='prefill':runner.prefill(session)
            else:runner.decode([session])
            assert all(private_digest(sessions[i])==digest for i,digest in others.items())
        if cancel_phase=='after_verify':
            while any(session.phase=='prefill' for session in sessions):
                for i in live:
                    if sessions[i].phase=='prefill':step(i)
            verify=runner.engine.verify_tokens
            def cancel_after_verify(*args,**kwargs):
                result=verify(*args,**kwargs)
                flags[cancelled].set()
                return result
            runner.engine.verify_tokens=cancel_after_verify
        else:flags[cancelled].set()
        before={i:private_digest(sessions[i]) for i in live if i!=cancelled}
        with pytest.raises(GenerationCancelled):
            if cancel_phase=='prefill':runner.prefill(sessions[cancelled])
            else:runner.decode([sessions[cancelled]])
        if cancel_phase=='after_verify':runner.engine.verify_tokens=verify
        runner.close(sessions[cancelled]);live.remove(cancelled)
        assert all(private_digest(sessions[i])==digest for i,digest in before.items())
        while any(not sessions[i].done for i in live):
            indices=list(live);order.shuffle(indices)
            for i in indices:
                if not sessions[i].done:step(i)
        for i in live:
            session=sessions[i]
            assert session.generated==greedy_gold(prompts[i],30)
            assert session.state.tokens==prompts[i]+session.generated[:-1]
            assert int(session.state.bank[0])==history_hash(session.state.tokens)
            assert session.state.prefixes is None
        assert cache.bytes<=cache.max_bytes
    finally:
        for session in sessions:
            if not session.closed:runner.close(session)


@pytest.mark.parametrize('cancelled',[0,1,2,3,'all'])
def test_api_cancellation_keeps_other_three_contracts_and_replacement_private(cancelled):
    fixture=api_fixtures.FourContextApiTests(methodName='runTest');fixture.setUp()
    try:
        runtime=fixture.runtime(61);owner=fixture.hold_owner(runtime)
        entered=threading.Event();release=fixture.gate();once=[]
        decode=runtime.batch_runner.decode
        def gate(sessions,**kwargs):
            active=runtime.scheduler._active
            if len(active)==4 and all(slot.session.generation.generated for slot in active) and not once:
                once.append(True);entered.set();assert release.wait(10)
            return decode(sessions,**kwargs)
        runtime.batch_runner.decode=gate
        budgets=[12,17,11,9]
        jobs=[runtime.submit(fixture.options(name,budget)) for name,budget in zip('ABCD',budgets)]
        fixture.prepared(jobs)
        replacement=runtime.submit(fixture.options('E',5)) if cancelled!='all' else None
        owner.set();assert entered.wait(3)
        assert runtime.context_pool.snapshot()['reserved_tokens']==61
        closed=[];close=runtime.batch_runner.close
        def release_after_state(generation):
            assert runtime.context_pool.snapshot()['reserved_tokens']>0
            close(generation);closed.append(generation.closed)
        runtime.batch_runner.close=release_after_state
        cancelled_indices=list(range(4)) if cancelled=='all' else [cancelled]
        for i in cancelled_indices:runtime.cancel_job(jobs[i])
        # Cancellation requests alone cannot give back still-owned state.
        assert runtime.context_pool.snapshot()['reserved_tokens']==61
        release.set()
        for i,job in enumerate(jobs):
            result=job.future.result(3)
            if i in cancelled_indices:assert result is None
            else:
                assert result['choices'][0]['message']['content']==','.join(map(str,greedy_gold(fixture.ids['ABCD'[i]],budgets[i])))
                assert result['usage']['total_tokens']==3+budgets[i]
        if replacement is not None:
            result=replacement.future.result(3)
            assert result['choices'][0]['message']['content']==','.join(map(str,greedy_gold(fixture.ids['E'],5)))
        assert all(closed)
        assert runtime.pending==0
        assert runtime.context_pool.snapshot()['reserved_tokens']==0
        assert runtime.context_pool.snapshot()['peak_reserved_tokens']==61
        assert runtime.submit(fixture.options('A',5)).future.result(3) is not None
    finally:fixture.tearDown()

