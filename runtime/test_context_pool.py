"""Admission, fairness and bounded preparation; no GPU capacity claims."""
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import unittest
import weakref

from context_pool import ContextPool
from preparation_executor import PreparationExecutor,PreparationFull
from continuous_scheduler import ContinuousScheduler
from test_continuous_scheduler import ManualExecutor,FakeSession,make_job


class ContextPoolTests(unittest.TestCase):
    def test_four_full_contracts_and_no_overcommit(self):
        pool=ContextPool(1048576,262144)
        for name in 'ABCD':
            self.assertTrue(pool.try_reserve(name,262080,64))
        self.assertFalse(pool.try_reserve('E',1,1))
        self.assertEqual(pool.snapshot()['reserved_tokens'],1048576)
        self.assertTrue(pool.release('B'))
        self.assertFalse(pool.release('B'))
        self.assertTrue(pool.try_reserve('E',10,20))
        self.assertEqual(pool.snapshot()['available_tokens'],262114)
        with self.assertRaises(ValueError):pool.try_reserve('E',10,21)

    def test_atomic_concurrent_reservations_never_exceed_pool(self):
        pool=ContextPool(100,100)
        gate=threading.Barrier(16)
        def reserve(i):
            gate.wait()
            return pool.try_reserve(str(i),9,1)
        with ThreadPoolExecutor(max_workers=16) as executor:
            results=list(executor.map(reserve,range(16)))
        self.assertEqual(sum(results),10)
        self.assertEqual(pool.snapshot()['reserved_tokens'],100)

    def test_invalid_and_permanently_oversize_contracts(self):
        pool=ContextPool(100,262144)
        for args in ((1,0),(-1,1),(True,1),(1,True),(100,1),(262144,1)):
            with self.subTest(args=args),self.assertRaises(ValueError):pool.try_reserve('A',*args)
        for total,request in ((0,1),(1,0),(True,1),(1,1.5)):
            with self.assertRaises(ValueError):ContextPool(total,request)
        self.assertEqual(pool.snapshot()['reservation_count'],0)


class FourSlotSchedulingTests(unittest.TestCase):
    def make_scheduler(self,**kwargs):
        self.executor=ManualExecutor();self.events=[]
        def factory(job):return FakeSession(job,self.events,lambda *args:None)
        self.scheduler=ContinuousScheduler(self.executor,factory,
            lambda sessions:self.fail('MTP requests must verify separately'),
            max_active=4,max_queued=2,decode_together=False,**kwargs)
        return self.scheduler

    def test_prefill_round_robin_all_four_without_starvation(self):
        scheduler=self.make_scheduler()
        futures=[scheduler.submit(make_job(name,chunks=4,budget=5)) for name in 'ABCD']
        self.executor.run()
        prefills=[name for event,name,*rest in self.events if event=='prefill']
        self.assertEqual(prefills,list('ABCD')*4)
        for name,future in zip('ABCD',futures):
            self.assertEqual(future.result(),[(name,i) for i in range(5)])

    def test_oldest_fitting_request_bypasses_long_contract_and_returns_quota(self):
        pool=ContextPool(40,262144)
        def admit(job):return pool.try_reserve(job.id,job.demand-1,1)
        scheduler=self.make_scheduler(admit=admit,discard=lambda job:pool.release(job.id))
        factory=scheduler.factory
        def session(job):
            result=factory(job);finish=result.finish
            def close(error=None):
                try:return finish(error)
                finally:pool.release(job.id)
            result.finish=close
            return result
        scheduler.factory=session
        jobs=[make_job(name,chunks=2,budget=8) for name in 'ABCD']
        for job,demand in zip(jobs,(17,25,7,8)):job.demand=demand
        futures=[scheduler.submit(job) for job in jobs]
        self.executor.run()
        admissions=[e['request_ids'][0] for e in scheduler.snapshot()['events'] if e['event']=='admit']
        self.assertEqual(admissions,list('ACDB'))
        self.assertTrue(all(f.result() for f in futures))
        self.assertEqual(pool.snapshot()['reserved_tokens'],0)

    def test_incomplete_preparation_parks_and_wakes_without_spinning(self):
        from concurrent.futures import Future
        prepared={name:Future() for name in 'AB'}
        scheduler=self.make_scheduler(prepare=lambda job:prepared[job.id])
        a=scheduler.submit(make_job('A'));b=scheduler.submit(make_job('B'))
        self.executor.run()
        self.assertFalse(a.done());self.assertFalse(b.done());self.assertEqual(self.executor.tasks,[])
        prepared['B'].set_result(None);self.executor.run()
        self.assertTrue(b.done());self.assertFalse(a.done())
        prepared['A'].set_result(None);self.executor.run();self.assertTrue(a.done())

    def test_backfill_is_bounded_so_short_arrivals_cannot_starve_long_request(self):
        pool=ContextPool(40,262144)
        scheduler=self.make_scheduler(admit=lambda job:pool.try_reserve(job.id,job.demand-1,1))
        def job(name,demand,budget):
            result=make_job(name,budget=budget);result.demand=demand;return result
        pending=[];factory=scheduler.factory
        def session(request):
            result=factory(request);finish=result.finish
            def close(error=None):
                value=finish(error);pool.release(request.id)
                if request.id.startswith('S'):
                    index=int(request.id[1:])
                    if index<5:pending.append(scheduler.submit(job(f'S{index+1}',7,1)))
                return value
            result.finish=close
            return result
        scheduler.factory=session
        pending.extend([scheduler.submit(job('A',17,40)),scheduler.submit(job('B',25,2)),
            scheduler.submit(job('S1',7,1))])
        self.executor.run()
        self.assertTrue(all(future.done() for future in pending))
        order=[event['request_ids'][0] for event in scheduler.snapshot()['events'] if event['event']=='admit']
        self.assertLess(order.index('B'),order.index('S5'))
        self.assertEqual(pool.snapshot()['reserved_tokens'],0)

    def test_preparation_wakeup_during_admission_is_not_lost(self):
        from concurrent.futures import Future
        preparation=Future();scheduler=self.make_scheduler(prepare=lambda job:preparation)
        future=scheduler.submit(make_job('A'))
        admit=scheduler._admit
        def race():
            admit()
            if not preparation.done():preparation.set_result(None)
        scheduler._admit=race
        self.executor.run();self.assertTrue(future.done())

    def test_greedy_batch_cannot_accidentally_enable_four(self):
        with self.assertRaises(ValueError):
            ContinuousScheduler(ManualExecutor(),None,None,max_active=4,decode_together=True)


class PreparationExecutorTests(unittest.TestCase):
    def test_bounded_even_when_running_preparation_was_cancelled(self):
        executor=PreparationExecutor(2,3);release=threading.Event();entered=threading.Barrier(3)
        def hold():
            entered.wait()
            if not release.wait(3):raise RuntimeError('Preparation gate timeout')
        try:
            a=executor.submit(hold);b=executor.submit(hold);entered.wait(timeout=2)
            self.assertFalse(a.cancel());self.assertFalse(b.cancel())
            queued=executor.submit(lambda:3)
            with self.assertRaises(PreparationFull):executor.submit(lambda:4)
            self.assertTrue(queued.cancel())
            replacement=executor.submit(lambda:5)
            with self.assertRaises(PreparationFull):executor.submit(lambda:6)
            release.set();a.result(3);b.result(3);self.assertEqual(replacement.result(3),5)
        finally:release.set();executor.shutdown(wait=True,cancel_futures=True)

    def test_cancelled_queue_drops_payload_immediately(self):
        executor=PreparationExecutor(1,2);release=threading.Event();entered=threading.Event()
        class Payload:pass
        payload=Payload();reference=weakref.ref(payload)
        def hold():entered.set();release.wait(3)
        try:
            running=executor.submit(hold);self.assertTrue(entered.wait(2))
            queued=executor.submit(lambda value=payload:value)
            del payload
            self.assertTrue(queued.cancel());gc.collect();self.assertIsNone(reference())
            release.set();running.result(3)
        finally:release.set();executor.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':unittest.main()
