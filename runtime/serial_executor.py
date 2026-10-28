"""One model thread; cancelling pending work removes its payload immediately."""
from collections import deque
from concurrent.futures import Future
import threading


class SerialExecutor:
    def __init__(self,thread_name='gb10-engine'):
        self._condition=threading.Condition()
        self._waiting=deque();self._closed=False
        self._thread=threading.Thread(target=self._worker,name=thread_name,daemon=False)
        self._thread.start()

    def submit(self,fn,*args,**kwargs):
        future=Future()
        def remove_cancelled(done):
            if done.cancelled():
                with self._condition:
                    self._waiting=deque(item for item in self._waiting if item[0] is not done)
        future.add_done_callback(remove_cancelled)
        with self._condition:
            if self._closed:raise RuntimeError('Cannot schedule work after shutdown')
            self._waiting.append((future,fn,args,kwargs))
            self._condition.notify()
        return future

    def _worker(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._waiting or self._closed)
                if not self._waiting:return
                future,fn,args,kwargs=self._waiting.popleft()
            try:
                if future.set_running_or_notify_cancel():
                    try:future.set_result(fn(*args,**kwargs))
                    except BaseException as error:future.set_exception(error)
            finally:
                # Do not retain a completed or cancelled payload while idle.
                del future,fn,args,kwargs

    def shutdown(self,wait=True,cancel_futures=False):
        with self._condition:
            self._closed=True
            pending=[item[0] for item in self._waiting] if cancel_futures else []
            self._condition.notify_all()
        # Callbacks can acquire Runtime.lock; keep them outside our lock.
        for future in pending:future.cancel()
        if wait:self._thread.join()
