"""Deterministic submission ownership and four-ready decode regressions.

Barriers force the reviewed interleaving on real CPU preparation/model-owner
threads. No production service, device model, or network access is used.
"""
import inspect
import sys
import threading

import pytest

import test_four_context_api as api_fixtures
from continuous_scheduler import ContinuousScheduler
from test_continuous_scheduler import ManualExecutor,TimedSession,FakeClock,make_job
from test_continuous_mtp import greedy_gold


@pytest.fixture
def api_fixture():
    fixture=api_fixtures.FourContextApiTests(methodName='runTest')
    fixture.setUp()
    try:yield fixture
    finally:fixture.tearDown()


def hold_first_prefill(runtime,fixture):
    entered=threading.Event();release=fixture.gate();once=[]
    original=runtime.batch_runner.prefill
    def prefill(session):
        if not once:
            once.append(True);entered.set()
            assert release.wait(10), 'Owner prefill gate timed out'
        return original(session)
    runtime.batch_runner.prefill=prefill
    return entered,release


def check_private_contracts(runtime):
    """Check leases against all live private states at checkout and cleanup."""
    live={};starts=[]
    start=runtime.batch_runner.start;close=runtime.batch_runner.close
    def start_checked(ids,budget,*args,**kwargs):
        session=start(ids,budget,*args,**kwargs)
        live[id(session)]=len(ids)+budget
        starts.append(id(session))
        assert sum(live.values())<=runtime.context_pool.snapshot()['reserved_tokens']<=runtime.total_context
        return session
    def close_checked(session):
        assert sum(live.values())<=runtime.context_pool.snapshot()['reserved_tokens']
        close(session)
        assert session.closed
        live.pop(id(session))
    runtime.batch_runner.start=start_checked;runtime.batch_runner.close=close_checked
    return live,starts


@pytest.mark.parametrize('cancel_first',[False,True])
def test_fast_preparation_callback_registration_keeps_active_lease(api_fixture,cancel_first):
    fixture=api_fixture;runtime=fixture.runtime(13)
    owner=fixture.hold_owner(runtime)
    entered,release=hold_first_prefill(runtime,fixture)
    live,starts=check_private_contracts(runtime)
    jobs=[];submitted=[];errors=[]
    prepare=runtime.scheduler.prepare
    def fast(job):
        jobs.append(job)
        future=prepare(job)
        future.result(3)  # Real, completed preparation before owner admission.
        return future
    runtime.scheduler.prepare=fast
    paused=threading.Event();resume=fixture.gate()
    lines,line_start=inspect.getsourcelines(ContinuousScheduler.submit)
    line=line_start+next(i for i,text in enumerate(lines)
        if '.add_done_callback(lambda done:self.wake())' in text)
    code=ContinuousScheduler.submit.__code__
    def trace(frame,event,arg):
        if event=='line' and frame.f_code is code and frame.f_lineno==line:
            paused.set();assert resume.wait(10), 'Submission gate timed out'
        return trace
    def submit():
        sys.settrace(trace)
        try:submitted.append(runtime.submit(fixture.options('A',10)))
        except BaseException as error:errors.append(error)
        finally:sys.settrace(None)
    thread=threading.Thread(target=submit,name='submission-regression')
    thread.start()
    try:
        assert paused.wait(3)
        owner.set();assert entered.wait(3)
        assert runtime.scheduler._active[0].preparation is None
        assert runtime.context_pool.snapshot()['reserved_tokens']==13
        resume.set();thread.join(3)
        assert not thread.is_alive()
        assert not errors, errors
        assert submitted==jobs
        first=submitted[0]
        assert first.future.running() and not first.released and not first.cancel.is_set()
        assert runtime.pending==1
        second=runtime.submit(fixture.options('B',10))
        fixture.prepared([second])
        assert runtime.context_pool.snapshot()['reserved_tokens']==13
        assert runtime.pending==2 and not second.executing.is_set()
        if cancel_first:
            runtime.cancel_job(first)
            assert runtime.context_pool.snapshot()['reserved_tokens']==13
        release.set()
        first_result=first.future.result(3)
        if cancel_first:assert first_result is None
        else:assert first_result['choices'][0]['message']['content']==','.join(map(str,greedy_gold(fixture.ids['A'],10)))
        assert second.future.result(3)['usage']['total_tokens']==13
        assert len(starts)==2 and not live
        assert runtime.pending==0
        assert runtime.context_pool.snapshot()['reserved_tokens']==0
        assert runtime.context_pool.snapshot()['peak_reserved_tokens']==13
    finally:
        resume.set();owner.set();release.set();thread.join(3)


@pytest.mark.parametrize('admitted',[False,True])
def test_post_enqueue_submit_failure_cancels_without_releasing_live_state(api_fixture,admitted):
    fixture=api_fixture;runtime=fixture.runtime(13)
    owner=fixture.hold_owner(runtime) if not admitted else None
    entered,release=hold_first_prefill(runtime,fixture)
    live,starts=check_private_contracts(runtime)
    jobs=[];submit=runtime.scheduler.submit
    def fail_after_enqueue(job,**kwargs):
        jobs.append(job)
        submit(job,**kwargs)
        if admitted:assert entered.wait(3)
        raise ValueError('Submission failed after enqueue')
    runtime.scheduler.submit=fail_after_enqueue
    with pytest.raises(ValueError,match='Submission failed after enqueue'):
        runtime.submit(fixture.options('A',10))
    runtime.scheduler.submit=submit
    job=jobs[0]
    assert job.cancel.is_set()
    if admitted:
        assert not job.released and runtime.pending==1
        assert job.future.running() and len(live)==1
        assert runtime.context_pool.snapshot()['reserved_tokens']==13
        replacement=runtime.submit(fixture.options('B',10))
        fixture.prepared([replacement])
        assert runtime.context_pool.snapshot()['reserved_tokens']==13
        assert not replacement.executing.is_set()
        release.set();assert job.future.result(3) is None
    else:
        assert job.released and runtime.pending==0 and not job.executing.is_set()
        assert job.future.cancelled() and not live
        assert runtime.scheduler.snapshot()['queued']==0
        assert runtime.context_pool.snapshot()['reserved_tokens']==0
        replacement=runtime.submit(fixture.options('B',10))
        release.set();owner.set()
    assert replacement.future.result(3)['usage']['total_tokens']==13
    assert not live and runtime.pending==0
    assert len(starts)==(2 if admitted else 1)
    assert runtime.context_pool.snapshot()['reserved_tokens']==0
    assert runtime.context_pool.snapshot()['peak_reserved_tokens']==13
    assert runtime.submit(fixture.options('C',10)).future.result(3)['usage']['total_tokens']==13


def ready_scheduler(max_active=4,hook=lambda *args:None,fatal=lambda:False):
    executor=ManualExecutor();clock=FakeClock();events=[];operations=[]
    scheduler=ContinuousScheduler(executor,
        lambda job:TimedSession(job,events,hook,clock,operations),
        lambda sessions:None,max_active=max_active,max_queued=1,
        decode_together=False,clock=clock,fatal=fatal)
    return scheduler,executor,operations


def ready_job(name,budget=8):
    job=make_job(name,budget=budget);job.ready=True
    return job


def decode_rounds(scheduler):
    rounds=[]
    for event in scheduler.snapshot()['events']:
        if event['event']=='decode_round':rounds.append([])
        if event['event']=='mtp_decode':rounds[-1].extend(event['request_ids'])
    return [row for row in rounds if row]


@pytest.mark.parametrize('count',[3,4])
def test_all_ready_decode_round_leaders_rotate(count):
    scheduler,executor,_=ready_scheduler(count)
    names=list('ABCD'[:count])
    futures=[scheduler.submit(ready_job(name)) for name in names]
    executor.run()
    rounds=decode_rounds(scheduler)
    assert [row[0] for row in rounds]==[names[i%count] for i in range(7)]
    assert all(set(row)==set(names) for row in rounds)
    assert [future.result() for future in futures]==[[(name,i) for i in range(8)] for name in names]


@pytest.mark.parametrize('name',list('ABCD'))
@pytest.mark.parametrize('departure',['cancel','finish'])
def test_departed_decode_leader_advances_to_surviving_successor(name,departure):
    turn='ABCD'.index(name)
    def hook(event,session):
        if departure=='cancel' and event=='output' and session.job.id==name and len(session.tokens)==turn+2:
            session.job.cancel.set()
    scheduler,executor,_=ready_scheduler(hook=hook)
    jobs=[ready_job(n,turn+2 if n==name and departure=='finish' else 10) for n in 'ABCD']
    futures=[scheduler.submit(job) for job in jobs]
    replacement=scheduler.submit(ready_job('E',5))
    executor.run()
    leaders=[row[0] for row in decode_rounds(scheduler)]
    assert leaders[:turn+1]==list('ABCD'[:turn+1])
    assert leaders[turn+1]=='ABCD'[(turn+1)%4]
    for job,future in zip(jobs,futures):
        if job.id==name and departure=='cancel':assert future.result() is None
        else:assert future.result()==[(job.id,i) for i in range(job.budget)]
    assert replacement.result()==[('E',i) for i in range(5)]
    assert not scheduler.snapshot()['active_ids'] and scheduler.snapshot()['queued']==0


def test_four_ready_fatal_after_leader_skips_all_other_device_steps():
    broken=[]
    def hook(event,session):
        if event=='output' and session.job.id=='A':broken.append(True)
    scheduler,executor,operations=ready_scheduler(hook=hook,fatal=lambda:bool(broken))
    futures=[scheduler.submit(ready_job(name)) for name in 'ABCDE']
    executor.run()
    for future in futures:
        with pytest.raises(RuntimeError,match='unavailable'):future.result()
    assert [name for event,name,_ in operations if event=='decode']==['A']
    assert not scheduler.snapshot()['active_ids'] and scheduler.snapshot()['queued']==0
