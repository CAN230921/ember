"""Deterministic scheduler control tests; no model or accelerator is loaded."""
from concurrent.futures import Future
import gc
import threading
from types import SimpleNamespace
import unittest
import weakref

from continuous_scheduler import ContinuousScheduler, SchedulerFull
from engine.request_control import GenerationCancelled


class ManualExecutor:
    def __init__(self):self.tasks=[]
    def submit(self,fn):
        future=Future();self.tasks.append((future,fn));return future
    def run(self):
        while self.tasks:
            future,fn=self.tasks.pop(0)
            if future.set_running_or_notify_cancel():
                try:future.set_result(fn())
                except BaseException as error:future.set_exception(error)
                future.result()


def make_job(name,chunks=1,budget=8):
    return SimpleNamespace(id=name,cancel=threading.Event(),phase='queued',
        completed_prefill_tokens=0,checkpoint_tokens=0,chunks=chunks,budget=budget)


class FakeSession:
    def __init__(self,job,events,hook):
        self.job=job;self.events=events;self.hook=hook
        self.phase='prefill';self.done=False;self.tokens=[];self.prefills=0
    def note(self,event):
        self.events.append((event,self.job.id,len(self.tokens),threading.get_ident()))
        self.hook(event,self)
    def start(self):self.note('start')
    def prefill(self):
        self.prefills+=1;self.job.completed_prefill_tokens=self.prefills*2048
        self.note('prefill')
        if self.prefills==self.job.chunks:self.emit()
    def emit(self):
        self.tokens.append((self.job.id,len(self.tokens)))
        self.done=len(self.tokens)>=self.job.budget
        self.phase='done' if self.done else 'decode'
        self.job.phase=self.phase
    def decode(self):self.emit()
    def after_step(self):self.note('output')
    def finish(self,error=None):
        self.note('cancel' if isinstance(error,GenerationCancelled) else 'error' if error else 'finish')
        if error and not isinstance(error,GenerationCancelled):raise error
        return None if error else list(self.tokens)


class FakeClock:
    def __init__(self):self.now=50.
    def __call__(self):return self.now
    def advance(self,seconds):self.now+=seconds


class TimedSession(FakeSession):
    def __init__(self,job,events,hook,clock,operations):
        super().__init__(job,events,hook)
        self.clock=clock;self.operations=operations
    def start(self):
        super().start()
        if getattr(self.job,'ready',False):self.emit()
    def prefill(self):
        self.operations.append(('prefill',self.job.id,self.clock()))
        self.clock.advance(getattr(self.job,'prefill_seconds',.1))
        super().prefill()
    def decode(self):
        self.operations.append(('decode',self.job.id,self.clock()))
        self.clock.advance(getattr(self.job,'decode_seconds',.1))
        super().decode()


class PrioritySchedulerTests(unittest.TestCase):
    def setUp(self):
        self.clock=FakeClock();self.executor=ManualExecutor()
        self.events=[];self.operations=[];self.hook=lambda *a:None
        def factory(job):
            return TimedSession(job,self.events,lambda *a:self.hook(*a),self.clock,self.operations)
        def batch(sessions):
            self.events.append(('batch',tuple(s.job.id for s in sessions),0,threading.get_ident()))
            for session in sessions:session.decode()
        self.factory=factory;self.batch=batch
        self.scheduler=ContinuousScheduler(self.executor,factory,batch,
            decode_together=False,decode_rounds_before_prefill=2,
            max_prefill_wait_seconds=100.,trace_limit=256,clock=self.clock)

    def ready_job(self,name,budget=8,seconds=.1):
        job=make_job(name,budget=budget);job.ready=True;job.decode_seconds=seconds
        return job

    def test_ready_decode_runs_first_and_round_quota_bounds_prefill_progress(self):
        a=self.ready_job('A',budget=20)
        b=make_job('B',chunks=3,budget=1);b.prefill_seconds=.2
        af=self.scheduler.submit(a);bf=self.scheduler.submit(b)
        self.executor.run()
        self.assertEqual([(e[0],e[1]) for e in self.operations[:3]],
                         [('decode','A'),('decode','A'),('prefill','B')])
        preceding=0;prefill_rounds=[]
        for event,name,_ in self.operations:
            if event=='decode':preceding+=1
            elif name=='B':prefill_rounds.append(preceding);preceding=0
        self.assertEqual(prefill_rounds,[2,2,2])
        self.assertEqual(af.result(),[('A',i) for i in range(20)])
        self.assertEqual(bf.result(),[('B',0)])
        snapshot=self.scheduler.snapshot()
        self.assertEqual(snapshot['scheduling_policy'],'decode_priority')
        self.assertEqual(snapshot['timing']['prefill_chunks'],3)
        self.assertAlmostEqual(snapshot['timing']['prefill_seconds'],.6)
        self.assertAlmostEqual(snapshot['timing']['decode_seconds'],1.9)
        self.assertAlmostEqual(snapshot['timing']['max_prefill_wait_seconds'],.2)
        self.assertTrue(any(e.get('reason')=='decode_round_limit' for e in snapshot['events']))
        self.assertEqual(len({e[3] for e in self.events}),1)

    def test_monotonic_deadline_runs_prefill_before_large_round_quota(self):
        self.scheduler.decode_rounds_before_prefill=32
        self.scheduler.max_prefill_wait_seconds=.5
        a=self.ready_job('A',budget=12,seconds=.3)
        b=make_job('B',chunks=2,budget=1)
        self.scheduler.submit(a);bf=self.scheduler.submit(b);self.executor.run()
        prefill=[e for e in self.scheduler.snapshot()['events'] if e['event']=='prefill']
        self.assertEqual([e['reason'] for e in prefill],['prefill_deadline']*2)
        for event in prefill:
            self.assertGreaterEqual(event['wait_seconds'],.5)
            # Deadline is checked at complete model boundaries, not mid-round.
            self.assertLess(event['wait_seconds'],.5+.3+1e-9)
            self.assertEqual(event['decode_rounds_since_prefill'],2)
        self.assertEqual(bf.result(),[('B',0)])
        self.assertAlmostEqual(self.scheduler.snapshot()['timing']['max_prefill_wait_seconds'],.6)

    def test_atomic_step_is_not_preempted_when_deadline_expires(self):
        self.scheduler.max_prefill_wait_seconds=.5
        a=self.ready_job('A',budget=4,seconds=2.)
        b=make_job('B',budget=1)
        self.scheduler.submit(a);self.scheduler.submit(b);self.executor.run()
        self.assertEqual([(e[0],e[1]) for e in self.operations[:2]],
                         [('decode','A'),('prefill','B')])
        prefill=next(e for e in self.scheduler.snapshot()['events'] if e['event']=='prefill')
        self.assertEqual(prefill['reason'],'prefill_deadline')
        self.assertEqual(prefill['wait_seconds'],2.)

    def test_two_prefill_requests_rotate_without_waiting_for_decode(self):
        futures=[self.scheduler.submit(make_job(name,chunks=3,budget=1)) for name in ('A','B')]
        self.executor.run()
        self.assertEqual([e[1] for e in self.operations],['A','B','A','B','A','B'])
        self.assertTrue(all(e[0]=='prefill' for e in self.operations))
        self.assertTrue(all(len(f.result())==1 for f in futures))
        self.assertEqual(self.scheduler.snapshot()['timing']['decode_rounds'],0)

    def test_cancel_after_first_request_skips_next_and_admits_replacement(self):
        a=self.ready_job('A');b=self.ready_job('B');c=make_job('C',budget=2)
        futures=[self.scheduler.submit(job) for job in (a,b,c)]
        def hook(event,session):
            if event=='output' and session.job.id=='A' and len(session.tokens)==2:b.cancel.set()
        self.hook=hook;self.executor.run()
        self.assertIsNone(futures[1].result())
        self.assertFalse(any(e[1]=='B' for e in self.operations))
        self.assertEqual(futures[0].result(),[('A',i) for i in range(8)])
        self.assertEqual(futures[2].result(),[('C',0),('C',1)])
        ends=[e[1] for e in self.events if e[0]=='finish']
        self.assertEqual(ends,['C','A'])
        self.assertEqual(sum(e[0]=='cancel' and e[1]=='B' for e in self.events),1)

    def test_fatal_after_first_request_cleans_both_and_queued_request(self):
        broken=[];self.scheduler.fatal=lambda:bool(broken)
        futures=[self.scheduler.submit(job) for job in
                 (self.ready_job('A'),self.ready_job('B'),make_job('C'))]
        def hook(event,session):
            if event=='output' and session.job.id=='A' and len(session.tokens)==2:broken.append(True)
        self.hook=hook;self.executor.run()
        for future in futures:
            with self.assertRaisesRegex(RuntimeError,'unavailable'):future.result()
        self.assertEqual([(e[0],e[1]) for e in self.operations],[('decode','A')])
        self.assertEqual(self.scheduler.snapshot()['active_ids'],[])
        self.assertEqual(self.scheduler.snapshot()['queued'],0)

    def test_late_prefill_admission_gets_fresh_decode_quota(self):
        futures={}
        def hook(event,session):
            if event=='output' and session.job.id=='A' and len(session.tokens)==4:
                futures['B']=self.scheduler.submit(make_job('B',budget=1))
        self.hook=hook
        futures['A']=self.scheduler.submit(self.ready_job('A',budget=10))
        self.executor.run()
        first_prefill=next(i for i,e in enumerate(self.operations) if e[0]=='prefill')
        # Three rounds before admission do not consume B's pending-prefill quota.
        self.assertEqual(first_prefill,5)
        self.assertEqual(futures['B'].result(),[('B',0)])

    def test_greedy_keeps_prefill_first_and_two_request_batch(self):
        self.scheduler.decode_together=True
        a=self.scheduler.submit(self.ready_job('A'))
        b=self.scheduler.submit(make_job('B',chunks=2,budget=4))
        self.executor.run()
        self.assertEqual(self.operations[0][:2],('prefill','B'))
        self.assertIn(('A','B'),[e[1] for e in self.events if e[0]=='batch'])
        self.assertEqual(a.result(),[('A',i) for i in range(8)])
        self.assertEqual(b.result(),[('B',i) for i in range(4)])
        self.assertEqual(self.scheduler.snapshot()['scheduling_policy'],'prefill_then_decode')

    def test_timing_and_trace_are_bounded_and_never_copy_payload(self):
        self.scheduler._trace=__import__('collections').deque(maxlen=12)
        a=self.ready_job('A',budget=100);a.prompt='private prompt';a.key='private key'
        b=make_job('B',chunks=4,budget=1)
        self.scheduler.submit(a);self.scheduler.submit(b);self.executor.run()
        snapshot=self.scheduler.snapshot()
        self.assertEqual(len(snapshot['events']),12)
        self.assertEqual(set(snapshot['timing']),{'decode_rounds','prefill_chunks',
            'max_decode_wait_seconds','max_prefill_wait_seconds','decode_seconds','prefill_seconds'})
        self.assertGreater(snapshot['timing']['max_decode_wait_seconds'],0)
        self.assertNotIn('private prompt',str(snapshot));self.assertNotIn('private key',str(snapshot))
        self.assertTrue(all(50. <= e['time'] <= self.clock() for e in snapshot['events']))

    def test_policy_parameters_are_explicit_and_validated(self):
        default=ContinuousScheduler(self.executor,self.factory,self.batch).snapshot()
        self.assertEqual(default['decode_rounds_before_prefill'],1)
        self.assertEqual(default['prefill_wait_limit_seconds'],1.)
        for rounds in (0,33,True,1.5):
            with self.assertRaises(ValueError):
                ContinuousScheduler(self.executor,self.factory,self.batch,decode_rounds_before_prefill=rounds)
        for limit in (0,-1,True,float('nan'),float('inf'),'1'):
            with self.assertRaises(ValueError):
                ContinuousScheduler(self.executor,self.factory,self.batch,max_prefill_wait_seconds=limit)
        with self.assertRaises(ValueError):
            ContinuousScheduler(self.executor,self.factory,self.batch,clock=None)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.executor=ManualExecutor();self.events=[];self.hook=lambda *a:None
        def factory(job):return FakeSession(job,self.events,lambda *a:self.hook(*a))
        def batch(sessions):
            self.events.append(('batch',tuple(s.job.id for s in sessions),0,threading.get_ident()))
            for session in sessions:session.decode()
        self.scheduler=ContinuousScheduler(self.executor,factory,batch,trace_limit=32)

    def test_cooperative_rounds_are_isolated_and_do_not_claim_batch(self):
        self.scheduler.decode_together=False
        a=make_job('A',budget=6);b=make_job('B',budget=7);c=make_job('C',budget=3)
        futures=[self.scheduler.submit(job) for job in (a,b,c)]
        def hook(event,session):
            if event=='output' and session.job.id=='A' and len(session.tokens)==3:
                raise ValueError('A framing failed')
        self.hook=hook
        self.executor.run()
        with self.assertRaisesRegex(ValueError,'A framing'):futures[0].result()
        self.assertEqual(futures[1].result(),[('B',i) for i in range(7)])
        self.assertEqual(futures[2].result(),[('C',i) for i in range(3)])
        trace=self.scheduler.snapshot()
        self.assertFalse(trace['decode_together'])
        self.assertTrue(any(e['event']=='mtp_decode' for e in trace['events']))
        self.assertFalse(any(e['event']=='batch_decode' for e in trace['events']))
        self.assertFalse(any(e[0]=='batch' for e in self.events))

    def test_cooperative_rechecks_device_failure_before_next_request(self):
        self.scheduler.decode_together=False
        broken=[]
        self.scheduler.fatal=lambda:bool(broken)
        a=make_job('A',budget=6);b=make_job('B',budget=7)
        futures=[self.scheduler.submit(job) for job in (a,b)]
        def hook(event,session):
            if event=='output' and session.job.id=='A' and len(session.tokens)==3:
                broken.append(True)
        self.hook=hook
        self.executor.run()
        for future in futures:
            with self.assertRaisesRegex(RuntimeError,'unavailable'):future.result()
        # B prefills before A's second round, but must not decode after the
        # failure in A; a GPU error makes all shared engine work unsafe.
        self.assertFalse(any(e['event']=='mtp_decode' and e['request_ids']==['B']
                             for e in self.scheduler.snapshot()['events']))

    def test_dynamic_one_two_one_two_with_eos_and_slot_reuse(self):
        futures={}
        def hook(event,session):
            if event=='output' and session.job.id=='A' and len(session.tokens)==2:
                futures['B']=self.scheduler.submit(make_job('B',budget=2))
            if event=='finish' and session.job.id=='B':
                futures['C']=self.scheduler.submit(make_job('C',budget=3))
        self.hook=hook
        futures['A']=self.scheduler.submit(make_job('A',budget=8))
        self.executor.run()
        self.assertEqual(len(futures['A'].result()),8)
        self.assertEqual(len(futures['B'].result()),2)
        self.assertEqual(len(futures['C'].result()),3)
        pairs=[e[1] for e in self.events if e[0]=='batch']
        self.assertIn(('A','B'),pairs);self.assertIn(('A','C'),pairs)
        ends=[e[1] for e in self.events if e[0]=='finish']
        self.assertEqual(ends,['B','C','A'])
        self.assertEqual(len({e[3] for e in self.events}),1)
        self.assertFalse(self.scheduler.snapshot()['active_ids'])

    def test_active_cancel_does_not_change_survivor_and_reuses_slot(self):
        a=make_job('A');b=make_job('B',budget=7);c=make_job('C',budget=2)
        futures=[self.scheduler.submit(j) for j in (a,b,c)]
        def hook(event,session):
            if event=='output' and session.job.id=='B' and len(session.tokens)==2:a.cancel.set()
        self.hook=hook;self.executor.run()
        self.assertIsNone(futures[0].result())
        self.assertEqual(futures[1].result(),[('B',i) for i in range(7)])
        self.assertEqual(futures[2].result(),[('C',i) for i in range(2)])
        self.assertEqual(sum(e[0]=='cancel' and e[1]=='A' for e in self.events),1)
        self.assertIn(('B','C'),[e[1] for e in self.events if e[0]=='batch'])

    def test_long_prefill_yields_to_short_and_decode_every_turn(self):
        a=make_job('A',chunks=8,budget=3);b=make_job('B',budget=5)
        seen=[]
        def hook(event,session):
            if event=='output' and session.job.id=='B' and session.tokens:
                seen.append(a.completed_prefill_tokens)
        self.hook=hook
        self.scheduler.submit(a);self.scheduler.submit(b);self.executor.run()
        self.assertLess(seen[0],8*2048)
        self.assertEqual(seen[0],2048)
        self.assertLessEqual(max(y-x for x,y in zip(seen,seen[1:])),2048)

    def test_capacity_cancelled_payload_and_bounded_trace(self):
        a=self.scheduler.submit(make_job('A'));b=self.scheduler.submit(make_job('B'))
        refs=[]
        class Payload:pass
        for index in range(100):
            job=make_job(str(index));job.payload=Payload();refs.append(weakref.ref(job.payload))
            future=self.scheduler.submit(job)
            with self.assertRaises(SchedulerFull):self.scheduler.submit(make_job('full'))
            self.assertTrue(future.cancel())
            del job,future
        gc.collect();self.assertTrue(all(ref() is None for ref in refs))
        self.assertLessEqual(len(self.scheduler.snapshot()['events']),32)
        self.executor.run();self.assertTrue(a.done() and b.done())

    def test_factory_failure_and_top_level_failure_recover(self):
        original=self.scheduler.factory
        def factory(job):
            if job.id=='bad':raise ValueError('construction failed')
            return original(job)
        self.scheduler.factory=factory
        bad=self.scheduler.submit(make_job('bad'));good=self.scheduler.submit(make_job('good'))
        self.executor.run()
        with self.assertRaisesRegex(ValueError,'construction'):bad.result()
        self.assertEqual(len(good.result()),8)
        # An unexpected bookkeeping exception must terminate all admitted work.
        check=self.scheduler._check_active
        self.scheduler._check_active=lambda:(_ for _ in ()).throw(RuntimeError('pump fault'))
        failed=self.scheduler.submit(make_job('failure'));self.executor.run()
        with self.assertRaisesRegex(RuntimeError,'pump fault'):failed.result()
        self.scheduler._check_active=check
        recovered=self.scheduler.submit(make_job('recovered'));self.executor.run()
        self.assertEqual(len(recovered.result()),8)

    def test_output_error_is_per_request_after_shared_commit(self):
        def hook(event,session):
            if event=='output' and session.job.id=='A' and len(session.tokens)==3:
                raise ValueError('invalid tool output')
        self.hook=hook
        a=self.scheduler.submit(make_job('A'));b=self.scheduler.submit(make_job('B'))
        self.executor.run()
        with self.assertRaisesRegex(ValueError,'invalid tool'):a.result()
        self.assertEqual(b.result(),[('B',i) for i in range(8)])

    def test_shutdown_detaches_queue_before_future_callbacks(self):
        first=make_job('A');second=make_job('B');third=make_job('C')
        futures=[self.scheduler.submit(j) for j in (first,second,third)]
        observed=[]
        # Re-enter pump exactly between shutdown's queue detach and cancel.
        cancel=futures[0].cancel
        def reentrant_cancel():
            self.executor.run()
            observed.append(self.scheduler.snapshot())
            return cancel()
        futures[0].cancel=reentrant_cancel
        self.scheduler.shutdown()
        self.assertTrue(all(j.cancel.is_set() for j in (first,second,third)))
        self.assertTrue(all(f.cancelled() for f in futures))
        self.assertEqual(self.events,[])
        self.assertEqual(observed[0]['queued'],0)
        with self.assertRaises(RuntimeError):self.scheduler.submit(make_job('late'))

    def test_exception_causes_do_not_retain_payload_and_cycles_terminate(self):
        refs=[]
        class Payload:pass
        def failing_factory(job):
            payload=Payload();refs.append(weakref.ref(payload))
            try:raise ValueError('inner')
            except ValueError as cause:raise RuntimeError('outer') from cause
        self.scheduler.factory=failing_factory
        future=self.scheduler.submit(make_job('bad'));self.executor.run()
        gc.collect();self.assertIsNone(refs[0]())
        self.assertIsInstance(future.exception(),RuntimeError)
        from continuous_scheduler import strip_error_frames
        a=ValueError('a');b=RuntimeError('b');a.__cause__=b;b.__context__=a
        strip_error_frames(a)
        self.assertIsNone(a.__cause__);self.assertIsNone(b.__context__)


if __name__=='__main__':unittest.main()
