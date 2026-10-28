"""Bounded CPU-only preparation with immediate removal of queued payloads."""
from collections import deque
from concurrent.futures import Future
import threading


class PreparationFull(Exception):
    pass


class PreparationExecutor:
    def __init__(self, workers, capacity):
        if not 1 <= workers <= 2 or capacity < workers:
            raise ValueError('Expected one/two workers and a bounded capacity')
        self._condition = threading.Condition()
        self._waiting = deque()
        self._running = 0
        self._capacity = capacity
        self._closed = False
        self._threads = [threading.Thread(target=self._worker,
            name=f'gb10-tokenize-{i}',daemon=False) for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, fn):
        future = Future()
        def remove(done):
            if done.cancelled():
                with self._condition:
                    self._waiting = deque(item for item in self._waiting if item[0] is not done)
        future.add_done_callback(remove)
        with self._condition:
            if self._closed:
                raise RuntimeError('Preparation executor is closed')
            if self._running + len(self._waiting) >= self._capacity:
                raise PreparationFull('CPU request preparation is full. Retry shortly.')
            self._waiting.append((future,fn))
            self._condition.notify()
        return future

    def _worker(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._waiting or self._closed)
                if not self._waiting:
                    return
                future,fn = self._waiting.popleft()
                self._running += 1
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        value = fn()
                    except BaseException as error:
                        # Preparation errors must not retain a tokenized prompt
                        # through their execution frames while queued for GPU.
                        from continuous_scheduler import strip_error_frames
                        future.set_exception(strip_error_frames(error))
                    else:
                        future.set_result(value)
                        del value
            finally:
                with self._condition:
                    self._running -= 1
                del future,fn

    def shutdown(self, wait=True, cancel_futures=False):
        with self._condition:
            self._closed = True
            waiting = [item[0] for item in self._waiting] if cancel_futures else []
            self._condition.notify_all()
        for future in waiting:
            future.cancel()
        if wait:
            for thread in self._threads:
                thread.join()
