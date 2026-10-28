"""Resumable ordinary greedy sessions for one/two independent requests.

This mode intentionally does not run MTP. The existing serial MTP runner stays
available as a separate startup mode; no suspended draft is ever resumed after
a batch step. Every yield owns persistent state and has consumed all logits.
"""
import time
import torch

from .batch import BatchEngine
from .ops.greedy_select import GreedySelector
from .speculative import _sync


class GreedySession:
    def __init__(self, ids, budget, eos, chunk_size, on_tokens, on_progress):
        if not ids or budget < 1 or chunk_size <= 0 or chunk_size % 64:
            raise ValueError('Expected a nonempty prompt, positive budget and aligned prefill chunks')
        self.ids = list(ids)
        self.budget = budget
        self.eos = set(eos)
        self.chunk_size = chunk_size
        self.on_tokens = on_tokens
        self.on_progress = on_progress
        self.state = None
        self.selector = GreedySelector()
        self.position = self.reused = self.checkpoint_tokens = 0
        self.generated = []
        self.phase = 'prefill'
        self.done = False
        self.begin = time.perf_counter()
        self.first_seconds = None
        self.chunks = []
        self.cache_seconds = 0.
        self.batch_steps = self.single_steps = 0
        self.cache_owner = object()

    def progress(self):
        if self.on_progress:
            self.on_progress(self.phase, self.position, self.checkpoint_tokens)

    def accept(self, token, state):
        self.state = state
        self.generated.append(int(token))
        if self.first_seconds is None:
            self.first_seconds = time.perf_counter() - self.begin
        self.done = token in self.eos or len(self.generated) >= self.budget
        self.phase = 'done' if self.done else 'decode'

    def emit(self):
        if self.on_tokens:
            self.on_tokens(tuple(self.generated))

    def result(self, cache_bytes):
        if not self.done:
            raise RuntimeError('Cannot finish an incomplete greedy request')
        elapsed = time.perf_counter() - self.begin
        decode_seconds = elapsed - self.first_seconds
        return dict(generated_ids=list(self.generated), output_tokens=len(self.generated),
            finish_reason='stop' if self.generated[-1] in self.eos else 'length',
            first_token_seconds=self.first_seconds, request_seconds=elapsed,
            decode_tokens_per_second=(len(self.generated)-1)/decode_seconds
                if len(self.generated)>1 and decode_seconds>0 else None,
            speculative=dict(enabled=False, strategy='continuous_greedy',
                             single_steps=self.single_steps, batch_steps=self.batch_steps),
            prefill_chunks=list(self.chunks),
            target_prefill_seconds=sum(c['target_seconds'] for c in self.chunks),
            draft_prefill_before_first_seconds=0., draft_prefill_after_first_seconds=0.,
            prefix_cache=dict(reused_tokens=self.reused, uncached_tokens=len(self.ids)-self.reused,
                checkpoint_seconds=self.cache_seconds, retained_bytes=cache_bytes,
                namespace='greedy_target'))

    def close(self):
        self.state = None
        self.ids = []
        self.selector = None
        self.on_tokens = self.on_progress = None


class GreedyBatchRunner:
    def __init__(self, engine, cache, *, graph_decode=True):
        self.engine = engine
        self.cache = cache
        self.batch = BatchEngine(engine, graph_decode=graph_decode)
        self.batch_selector = GreedySelector()

    @torch.no_grad()
    def start(self, ids, budget, eos, chunk_size, on_tokens=None, on_progress=None, cache=True):
        session = GreedySession(ids, budget, eos, chunk_size, on_tokens, on_progress)
        session.cache_enabled = cache
        begin = time.perf_counter()
        if self.cache is not None and cache:
            session.state, session.reused = self.cache.checkout(ids, chunk_size)
            session.position = session.checkpoint_tokens = session.reused
        _sync(self.engine)
        session.cache_seconds += time.perf_counter()-begin
        session.progress()
        return session

    @torch.no_grad()
    def prefill(self, session):
        if session.phase != 'prefill':
            raise ValueError('Request is not awaiting prefill')
        start = session.position
        end = min(start+session.chunk_size, len(session.ids))
        begin = time.perf_counter()
        logits, state = self.engine(session.ids[start:end], session.state,
                                    prefill_chunk=True, owned_graph_cache=True)
        _sync(self.engine)
        session.state, session.position = state, end
        session.chunks.append(dict(start=start, end=end,
            target_seconds=time.perf_counter()-begin, draft_seconds=0.))
        if end < len(session.ids):
            begin = time.perf_counter()
            if self.cache is not None and session.cache_enabled and self.cache.checkpoint(
                    session.ids, end, state, session.chunk_size, owner=session.cache_owner):
                session.checkpoint_tokens = end
            _sync(self.engine)
            session.cache_seconds += time.perf_counter()-begin
            session.progress()
        else:
            session.accept(session.selector(logits)[0], state)

    @torch.no_grad()
    def decode(self, sessions):
        if not 1 <= len(sessions) <= 2 or any(s.phase != 'decode' for s in sessions):
            raise ValueError('Expected one/two decode-ready independent sessions')
        if len(sessions) == 2:
            logits, states = self.batch.decode([s.generated[-1] for s in sessions],
                                               [s.state for s in sessions])
            tokens = self.batch_selector(logits)
            for session in sessions:
                session.batch_steps += 1
        else:
            session = sessions[0]
            logits, state = self.engine([session.generated[-1]], session.state, owned_graph_cache=True)
            states = [state]
            tokens = session.selector(logits)
            session.single_steps += 1
        # Consume both logits and commit both private states before callbacks.
        # Batch graph snapshots own banks; M1 explicitly requests owned banks.
        for session, token, state in zip(sessions, tokens, states):
            session.accept(token, state)

    def close(self, session):
        try:
            _sync(self.engine)
        finally:
            if self.cache is not None:self.cache.release(session.cache_owner)
            session.close()
