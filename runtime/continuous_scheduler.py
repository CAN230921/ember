"""Bounded cooperative request scheduling on the existing model-owner thread.

Sessions implement start(), prefill(), decode(), after_step(), finish(error),
phase and done.
Only this pump calls them. Greedy preserves one prefill chunk followed by a
decode step for every ready session. Cooperative MTP prioritizes ready decode,
with a bounded number of complete rounds between prefill chunks and a monotonic
prefill waiting deadline. Atomic model steps are never interrupted. Greedy can
batch two ready sessions; MTP defaults to independent verification, with an
explicit opt-in batch callback that retains decode priority and per-lane errors.
No model work is submitted from the HTTP threads.
"""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import threading
import time
import math

from engine.request_control import GenerationCancelled


class SchedulerFull(Exception):
    pass


def strip_error_frames(error):
    """Drop execution frames throughout exception chains, including cycles."""
    stack=[error];seen=set()
    while stack:
        current=stack.pop()
        if current is None or id(current) in seen:continue
        seen.add(id(current))
        stack.extend((current.__cause__,current.__context__))
        stack.extend(getattr(current,'exceptions',()))
        current.__traceback__=None
        current.__cause__=None
        current.__context__=None
    return error


@dataclass
class Slot:
    job: object
    future: Future
    session: object = None
    prefill_wait_since: object = None
    decode_ready_since: object = None
    preparation: object = None
    admission_wait: bool = False
    admission_bypasses: int = 0


class ContinuousScheduler:
    def __init__(self, executor, factory, decode_batch, *, max_active=2,
                 max_queued=1, trace_limit=256, fatal=lambda: False, on_start_error=None,
                 decode_together=True, decode_rounds_before_prefill=1,
                 max_prefill_wait_seconds=1., clock=time.monotonic,
                 prepare=None, admit=None, discard=None, mtp_batch=False):
        if type(mtp_batch) is not bool or mtp_batch and decode_together:
            raise ValueError('MTP batching requires decode-priority scheduling')
        limit = 2 if decode_together else 4
        if type(max_active) is not int or not 1 <= max_active <= limit or not 0 <= max_queued <= 32:
            raise ValueError(f'Expected 1..{limit} active slots and 0..32 queued slots')
        if not 1 <= trace_limit <= 4096:
            raise ValueError('Expected a bounded scheduling trace')
        if type(decode_rounds_before_prefill) is not int or not 1 <= decode_rounds_before_prefill <= 32:
            raise ValueError('Expected 1..32 decode rounds between prefill chunks')
        if (isinstance(max_prefill_wait_seconds,bool) or
                not isinstance(max_prefill_wait_seconds,(int,float)) or
                not math.isfinite(max_prefill_wait_seconds) or max_prefill_wait_seconds <= 0):
            raise ValueError('Expected a finite positive prefill waiting deadline')
        if not callable(clock):
            raise ValueError('Expected a monotonic clock')
        self.executor = executor
        self.factory = factory
        self.decode_batch = decode_batch
        self.decode_together = decode_together
        self.mtp_batch = mtp_batch
        self.decode_rounds_before_prefill = decode_rounds_before_prefill
        self.max_prefill_wait_seconds = float(max_prefill_wait_seconds)
        self._clock = clock
        self.max_active = max_active
        self.max_queued = max_queued
        self.fatal = fatal
        self.on_start_error = on_start_error
        self.prepare = prepare
        self.admit = admit
        self.discard = discard
        self._lock = threading.Lock()
        self._waiting = deque()
        self._active = []
        self._trace = deque(maxlen=trace_limit)
        self._sequence = 0
        self._scheduled = False
        self._wake_version = 0
        self._closed = False
        self._last_prefill = None
        self._decode_successors = ()
        self._decode_rounds_since_prefill = 0
        self._timing = dict(decode_rounds=0, prefill_chunks=0,
            max_decode_wait_seconds=0., max_prefill_wait_seconds=0.,
            decode_seconds=0., prefill_seconds=0.)

    def _record(self, event, slots=(), **metadata):
        # All callers hold _lock. Only IDs and scheduling metadata are retained.
        self._sequence += 1
        self._trace.append(dict(sequence=self._sequence, time=self._clock(),
            event=event, request_ids=[s.job.id for s in slots],
            active_ids=[s.job.id for s in self._active], queued=len(self._waiting), **metadata))

    def snapshot(self):
        with self._lock:
            return dict(max_active=self.max_active, max_queued=self.max_queued,
                decode_together=self.decode_together,
                mtp_batch=self.mtp_batch,
                scheduling_policy='prefill_then_decode' if self.decode_together else 'decode_priority',
                decode_rounds_before_prefill=self.decode_rounds_before_prefill,
                prefill_wait_limit_seconds=self.max_prefill_wait_seconds,
                decode_rounds_since_prefill=self._decode_rounds_since_prefill,
                timing=dict(self._timing),
                active_ids=[s.job.id for s in self._active], queued=len(self._waiting),
                admission_waiting=sum(s.admission_wait for s in self._waiting),
                preparing=sum(s.preparation is not None and not s.preparation.done() for s in self._waiting),
                events=[dict(e, request_ids=list(e['request_ids']),
                             active_ids=list(e['active_ids'])) for e in self._trace])

    def submit(self, job, *, future=None):
        # Runtime keeps this handle before publication so a later submission
        # failure can cancel queued work or request owner cleanup of live work.
        future = Future() if future is None else future
        slot = Slot(job, future)
        def remove_cancelled(done):
            if done.cancelled():
                with self._lock:
                    self._waiting = deque(s for s in self._waiting if s.future is not done)
                    self._record('cancel_queued', [slot])
                if slot.preparation is not None:
                    slot.preparation.cancel()
                if self.discard is not None:
                    self.discard(slot.job)
                # A cancelled Future must not retain the prompt via its callback.
                slot.job = None
                slot.preparation = None
                self.wake()
        future.add_done_callback(remove_cancelled)
        with self._lock:
            if self._closed:
                raise RuntimeError('Scheduler is closed')
            if len(self._waiting) + len(self._active) >= self.max_active + self.max_queued:
                raise SchedulerFull('Active and queued request slots are full')
            self._waiting.append(slot)
            try:
                if self.prepare is not None:
                    slot.preparation = self.prepare(job)
            except BaseException:
                self._waiting.remove(slot)
                raise
            # The owner may clear the slot as soon as the pump is published.
            # Keep a stable reference, but register outside this lock: an
            # already-complete Future invokes wake() synchronously.
            preparation = slot.preparation
            self._record('enqueue', [slot])
            if not self._scheduled:
                self._scheduled = True
                try:
                    self.executor.submit(self._pump)
                except BaseException:
                    self._scheduled = False
                    self._waiting.remove(slot)
                    if slot.preparation is not None:
                        slot.preparation.cancel()
                    if self.discard is not None:
                        self.discard(job)
                    raise
        if preparation is not None:
            preparation.add_done_callback(lambda done:self.wake())
        return future

    def wake(self):
        """Preparation/cancellation may wake a parked owner without polling."""
        with self._lock:
            self._wake_version += 1
            if self._closed or self._scheduled or not self._waiting:
                return
            self._scheduled = True
            try:
                self.executor.submit(self._pump)
            except BaseException:
                self._scheduled = False
                raise

    def _finish(self, slot, error=None):
        # Remove before callbacks so a completing request releases admission
        # immediately; session cleanup itself still runs on the owner thread.
        with self._lock:
            if slot not in self._active:
                return
            self._active.remove(slot)
            self._record('cancel' if isinstance(error, GenerationCancelled)
                         else 'error' if error else 'complete', [slot])
        try:
            if slot.session is not None:
                result = slot.session.finish(error)
            elif self.on_start_error is not None:
                result = self.on_start_error(slot.job, error)
            else:
                raise error
        except BaseException as failure:
            # Tracebacks may contain a whole GPU request. Never keep those in
            # completed Futures while an HTTP caller still holds the Future.
            slot.future.set_exception(strip_error_frames(failure))
        else:
            slot.future.set_result(result)
        finally:
            slot.session = None
            slot.job = None
            slot.preparation = None

    def _check_active(self):
        for slot in list(self._active):
            if slot.job.cancel.is_set():
                self._finish(slot, GenerationCancelled(slot.job.phase,
                    slot.job.completed_prefill_tokens, slot.job.checkpoint_tokens))
            elif self.fatal():
                self._finish(slot, RuntimeError('Engine is unavailable'))
            elif slot.session.done:
                self._finish(slot)

    def _admit(self):
        while True:
            error = None
            with self._lock:
                if self._closed or len(self._active) >= self.max_active or not self._waiting:
                    return
                slot = None
                blocked = []
                for candidate in self._waiting:
                    if candidate.future.cancelled():
                        continue
                    if candidate.preparation is not None and not candidate.preparation.done():
                        continue
                    try:
                        if candidate.preparation is not None:
                            candidate.preparation.result()
                        allowed = self.admit(candidate.job) if self.admit is not None else True
                    except Exception as failure:
                        error = failure
                        allowed = True
                    if not allowed:
                        if not candidate.admission_wait:
                            candidate.admission_wait = True
                            self._record('context_wait', [candidate])
                        # The oldest fitting request progresses; a large prompt
                        # cannot block short requests behind it.
                        # Bound backfilling so a steady stream of short jobs
                        # cannot indefinitely starve the oldest large contract.
                        if candidate.admission_bypasses >= self.max_active:
                            break
                        blocked.append(candidate)
                        continue
                    slot = candidate
                    for waiting in blocked:
                        waiting.admission_bypasses += 1
                    break
                if slot is None:
                    return
                self._waiting.remove(slot)
                if not slot.future.set_running_or_notify_cancel():
                    if self.discard is not None:
                        self.discard(slot.job)
                    continue
                self._active.append(slot)
                slot.prefill_wait_since = self._clock()
                self._record('admit', [slot])
            # Factory must only construct bookkeeping; start does tokenization
            # and checkout so even admission failures go through finish().
            try:
                if error is not None:
                    raise error
                slot.session = self.factory(slot.job)
                if self.fatal():
                    raise RuntimeError('Engine is unavailable')
                slot.session.start()
            except Exception as error:
                self._finish(slot, error)
            finally:
                slot.preparation = None

    def _pump(self):
        try:
            self._run_pump()
        except BaseException as error:
            # Bookkeeping/factory failures must not strand running Futures or
            # leave _scheduled stuck True. Fail this turn and allow recovery.
            with self._lock:
                while self._waiting:
                    slot = self._waiting.popleft()
                    if slot.future.set_running_or_notify_cancel():
                        self._active.append(slot)
                affected = list(self._active)
                self._record('pump_error', affected)
            for slot in affected:
                self._finish(slot, error)
            with self._lock:
                self._scheduled = False
                if self._waiting:
                    self._scheduled = True
                    self.executor.submit(self._pump)

    def _refresh_phase_times(self):
        now = self._clock()
        for slot in self._active:
            if slot.session.phase == 'prefill':
                if slot.prefill_wait_since is None:
                    slot.prefill_wait_since = now
                slot.decode_ready_since = None
            elif slot.session.phase == 'decode':
                slot.prefill_wait_since = None
                if slot.decode_ready_since is None:
                    slot.decode_ready_since = now

    def _prefill_slot(self, prefill, reason):
        candidates = prefill
        if reason == 'prefill_deadline':
            oldest = min(s.prefill_wait_since for s in prefill)
            candidates = [s for s in prefill if s.prefill_wait_since == oldest]
        return self._rotate(candidates, self._last_prefill)[0]

    def _rotate(self, slots, previous):
        # Rotate after the last request in admission order, including when
        # that request is now decoding or has just left the active list.
        order = [s.job.id for s in self._active]
        if previous not in order:
            return list(slots)
        pivot = order.index(previous)
        return sorted(slots, key=lambda s:(order.index(s.job.id)-pivot-1) % len(order))

    def _run_prefill(self, slot, reason):
        now = self._clock()
        wait = max(0., now-slot.prefill_wait_since)
        self._last_prefill = slot.job.id
        with self._lock:
            self._timing['max_prefill_wait_seconds'] = max(self._timing['max_prefill_wait_seconds'],wait)
            self._record('prefill', [slot], reason=reason, wait_seconds=wait,
                decode_rounds_since_prefill=self._decode_rounds_since_prefill)
        error = None
        try:
            slot.session.prefill()
            slot.session.after_step()
        except Exception as failure:
            error = failure
        elapsed = max(0., self._clock()-now)
        with self._lock:
            self._timing['prefill_chunks'] += 1
            self._timing['prefill_seconds'] += elapsed
            self._decode_rounds_since_prefill = 0
            self._record('prefill_done', [slot], seconds=elapsed,
                status='error' if error else 'completed')
        slot.prefill_wait_since = self._clock() if slot.session.phase == 'prefill' else None
        if error is not None:
            self._finish(slot,error)

    def _decode_wait(self, slot):
        return max(0.,self._clock()-slot.decode_ready_since)

    def _decode_order(self, ready):
        if self.max_active <= 2:
            return ready  # Keep the single/dual-request order.
        order = list(ready)
        # Prefer the previous leader's first surviving ready successor. IDs
        # survive slot cleanup, so finishing/cancelling a leader or successor
        # cannot reset rotation to the oldest admission.
        for request_id in self._decode_successors:
            for index, slot in enumerate(order):
                if slot.job.id == request_id:
                    return order[index:] + order[:index]
        return order

    def _run_decode_round(self, ready, *, prefill_waiting=True):
        # Keep per-request failure and cancellation boundaries in cooperative
        # MTP. Rechecking before B prevents a device failure in A from running B.
        begin = self._clock()
        ids = [s.job.id for s in ready]
        waits = {s.job.id:self._decode_wait(s) for s in ready}
        with self._lock:
            self._record('decode_round',ready,reason='decode_ready',
                decode_rounds_since_prefill=self._decode_rounds_since_prefill,
                wait_seconds=max(waits.values()))
        if self.mtp_batch:
            self._check_active()
            order = [s for s in self._decode_order(ready) if s in self._active]
            if order:
                self._decode_successors = tuple(s.job.id for s in order[1:]+order[:1])
                with self._lock:
                    self._timing['max_decode_wait_seconds'] = max(self._timing['max_decode_wait_seconds'],max(waits.values()))
                    self._record('mtp_batch_decode',order,wait_seconds=max(waits.values()))
                try:
                    errors = self.decode_batch([s.session for s in order])
                    if not isinstance(errors,(list,tuple)) or len(errors)!=len(order):
                        raise RuntimeError('MTP batch callback must return one error status per request')
                    if any(error is not None and not isinstance(error,Exception) for error in errors):
                        raise RuntimeError('Invalid MTP batch error status')
                except Exception as error:
                    errors = [error]*len(order)
                for slot,error in zip(order,errors):
                    # State commits finish before framing. A cancellation or
                    # framing error closes only its own private request/lease.
                    if error is None and self.fatal():
                        error = RuntimeError('Engine is unavailable after a shared step')
                    if error is None:
                        try:
                            slot.session.after_step()
                        except Exception as failure:
                            error = failure
                    with self._lock:
                        self._record('mtp_batch_decode_done',[slot],seconds=max(0.,self._clock()-begin),
                            status='error' if error else 'completed')
                    slot.decode_ready_since = self._clock()
                    if error is not None:
                        self._finish(slot,error)
        elif not self.decode_together:
            order = self._decode_order(ready)
            ids = [s.job.id for s in order]
            leader_recorded = False
            for index, slot in enumerate(order):
                self._check_active()
                if slot not in self._active:
                    continue
                if not leader_recorded:
                    self._decode_successors = tuple(ids[index+1:] + ids[:index+1])
                    leader_recorded = True
                started = self._clock()
                wait = self._decode_wait(slot)
                with self._lock:
                    self._timing['max_decode_wait_seconds'] = max(self._timing['max_decode_wait_seconds'],wait)
                    self._record('mtp_decode',[slot],wait_seconds=wait)
                error = None
                try:
                    slot.session.decode()
                    slot.session.after_step()
                except Exception as failure:
                    error = failure
                with self._lock:
                    self._record('mtp_decode_done',[slot],seconds=max(0.,self._clock()-started),
                        status='error' if error else 'completed')
                slot.decode_ready_since = self._clock()
                if error is not None:
                    self._finish(slot,error)
        else:
            with self._lock:
                self._timing['max_decode_wait_seconds'] = max(
                    self._timing['max_decode_wait_seconds'],max(waits.values()))
                self._record('batch_decode' if len(ready)==2 else 'decode',ready,
                    wait_seconds=max(waits.values()))
            try:
                if len(ready)==2:
                    self.decode_batch([s.session for s in ready])
                else:
                    ready[0].session.decode()
            except Exception as error:
                # A failed shared step has no independent commit point.
                for slot in ready:
                    self._finish(slot,error)
            else:
                for slot in ready:
                    try:
                        slot.session.after_step()
                    except Exception as error:
                        self._finish(slot,error)
                    else:
                        slot.decode_ready_since = self._clock()
        elapsed = max(0.,self._clock()-begin)
        with self._lock:
            self._decode_rounds_since_prefill = self._decode_rounds_since_prefill+1 if prefill_waiting else 0
            self._timing['decode_rounds'] += 1
            self._timing['decode_seconds'] += elapsed
            # Request cleanup may have detached job references in this round.
            self._record('decode_round_done',seconds=elapsed,round_request_ids=tuple(ids))

    def _run_pump(self):
        while True:
            with self._lock:
                wake_version = self._wake_version
            self._check_active()
            self._admit()
            self._check_active()
            with self._lock:
                if not self._active:
                    if self._waiting and self._wake_version != wake_version:
                        continue
                    self._scheduled = False
                    return
            prefill = [s for s in self._active if s.session.phase == 'prefill']
            self._refresh_phase_times()
            ready = [s for s in self._active if s.session.phase == 'decode']
            if not self.decode_together:
                reason = None
                if prefill:
                    oldest_wait = max(self._clock()-s.prefill_wait_since for s in prefill)
                    if not ready:
                        reason = 'no_decode_ready'
                    elif oldest_wait >= self.max_prefill_wait_seconds:
                        reason = 'prefill_deadline'
                    elif self._decode_rounds_since_prefill >= self.decode_rounds_before_prefill:
                        reason = 'decode_round_limit'
                if reason is not None:
                    self._run_prefill(self._prefill_slot(prefill,reason),reason)
                elif ready:
                    self._run_decode_round(ready,prefill_waiting=bool(prefill))
                self._check_active()
                continue
            if prefill:
                self._run_prefill(self._prefill_slot(prefill,'greedy_compatibility'),'greedy_compatibility')
            self._check_active()
            self._refresh_phase_times()
            ready = [s for s in self._active if s.session.phase == 'decode']
            if ready:
                self._run_decode_round(ready)
            self._check_active()

    def shutdown(self):
        with self._lock:
            self._closed = True
            waiting = [s.future for s in self._waiting]
            for slot in self._waiting:
                slot.job.cancel.set()
            self._waiting.clear()
            for slot in self._active:
                slot.job.cancel.set()
        for future in waiting:
            future.cancel()
