"""Regression: cancelled requests must not retain large payloads behind a long job."""
import gc
import threading
import unittest
import weakref
from serial_executor import SerialExecutor


class Payload:pass


class SerialExecutorTests(unittest.TestCase):
    def test_cancelled_queue_payloads_are_freed_while_first_request_runs(self):
        pool=SerialExecutor();blocked=threading.Event();entered=threading.Event();refs=[]
        def hold():entered.set();blocked.wait(5)
        first=pool.submit(hold)
        try:
            self.assertTrue(entered.wait(1))
            for _ in range(100):
                value=Payload();refs.append(weakref.ref(value))
                future=pool.submit(lambda item:None,value);self.assertTrue(future.cancel())
            del value,future
            gc.collect();self.assertEqual(sum(ref() is not None for ref in refs),0)
            self.assertFalse(first.done())
        finally:blocked.set();pool.shutdown()

    def test_fifo_same_thread_exception_and_recovery(self):
        pool=SerialExecutor();order=[]
        def work(i):order.append(i);return threading.get_ident()
        try:
            futures=[pool.submit(work,i) for i in range(20)]
            identities={f.result(2) for f in futures}
            self.assertEqual(len(identities),1);self.assertEqual(order,list(range(20)))
            failure=pool.submit(lambda:1/0)
            with self.assertRaises(ZeroDivisionError):failure.result(2)
            self.assertIn(pool.submit(work,20).result(2),identities)
        finally:pool.shutdown()
        with self.assertRaises(RuntimeError):pool.submit(work,21)

    def test_running_job_cannot_be_cancelled_by_future(self):
        pool=SerialExecutor();entered=threading.Event();release=threading.Event()
        def work():entered.set();release.wait(5);return 42
        future=pool.submit(work)
        try:
            self.assertTrue(entered.wait(1));self.assertFalse(future.cancel())
        finally:release.set();pool.shutdown()
        self.assertEqual(future.result(),42)


if __name__=='__main__':unittest.main()
