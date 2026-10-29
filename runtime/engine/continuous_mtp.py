"""Resumable learned MTP on one model-owner thread.

Each decode turn finishes one request's complete verify/commit/draft round.
Two ready requests run cooperatively; they never share a verification batch.
Arithmetic and acceptance rules are those of speculative.generate_mtp.
"""
import time

import torch

from .ops.greedy_select import GreedySelector
from .draft_head import select_draft
from .prefix_cache import PrefixCache, clone_state
from .snapshot_cache import SnapshotCheckpoint, reserved_bytes, snapshot_requirements, storage_map
from .prefill_memory import PrefillMemoryTrimmer, memory_summary, record_memory_check
from .request_control import GenerationCancelled
from .speculative import _sync, _token
from .speculative_policy import AdaptiveDraftDepth
from .ops.ple import hint_next_prefill_chunk
from .dflash_draft import LAYERS as DFLASH_LAYERS


class ContinuousMTPPrefixCache(PrefixCache):
    """One bounded multi-branch pool on the model-owner thread.

    Checkouts own mutable tensors; only immutable PagedKV pages are shared.
    Two recent branch heads receive soft retention, so growing one branch
    cannot remove the other branch's only useful checkpoint. MTP keys include
    the lookahead token and the prefill chunk size remains a namespace.
    """
    def __init__(self, max_bytes=1024**3, *, max_entries=128):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Expected a nonnegative checkpoint byte budget')
        if type(max_entries) is not int or max_entries < 2:
            raise ValueError('Expected room for at least two checkpoint entries')
        super().__init__(max_bytes)
        self.max_entries = max_entries
        self.owner = None
        self.generation = self._clock = 0
        self._heads = []
        self._counts = dict(hit=0, miss=0, duplicate=0, publish=0, eviction=0,
                            oversize=0, budget=0, clone_failure=0)

    @property
    def bytes(self):
        union = {}
        for entry in self.entries:
            union.update(entry.storages)
        return sum(union.values())

    def snapshot(self):
        return dict(self._counts, bytes=self.bytes, entries=len(self.entries),
                    max_bytes=self.max_bytes, max_entries=self.max_entries,
                    protected_heads=len(self._heads), generation=self.generation)

    def checkout(self, ids, chunk_size, *, min_end=0):
        if type(min_end) is not int or min_end < 0:
            raise ValueError('Expected a nonnegative minimum checkpoint boundary')
        keys = {}
        matched = []
        for entry in self.entries:
            if entry.chunk_size != chunk_size or not min_end < entry.end < len(ids):
                continue
            if entry.end not in keys:
                keys[entry.end] = tuple(ids[:entry.end+1])
            if keys[entry.end] == entry.key:
                matched.append(entry)
        if not matched:
            self._counts['miss'] += 1
            return None, None, 0, []
        matched.sort(key=lambda entry: entry.end)
        entry = matched[-1]
        target, draft = clone_state(entry.target), clone_state(entry.draft)
        self._counts['hit'] += 1
        self._clock += 1
        entry.last_used = self._clock
        # An ancestor hit does not falsely refresh a deeper branch's head.
        if any(head is entry for head in self._heads):
            self._heads = [head for head in self._heads if head is not entry] + [entry]
        return target, draft, entry.end, matched

    def _branch_plan(self, key, end, chunk_size):
        for head in self._heads:
            if head.chunk_size == chunk_size and head.key == key[:head.end+1]:
                return {id(item) for item in self._heads if item is not head}, id(head), False
        if any(head.chunk_size == chunk_size and key == head.key[:end+1]
               for head in self._heads):
            # Backfilling an ancestor neither displaces nor refreshes a head.
            return {id(head) for head in self._heads}, None, False
        # A third branch may replace the oldest soft protection. It must still
        # retain the most recent other branch and obey the hard total budget.
        return {id(head) for head in self._heads[-1:]}, None, True

    def _evict(self, indices):
        if not indices:
            return
        removed = set(indices)
        self.entries = [entry for index, entry in enumerate(self.entries) if index not in removed]
        retained = {id(entry) for entry in self.entries}
        self._heads = [head for head in self._heads if id(head) in retained]
        self._counts['eviction'] += len(removed)
        self.generation += 1

    def publish(self, ids, end, target, draft, chunk_size, owner):
        if (type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 64 or
                type(end) is not int or not 0 < end < len(ids) or end % chunk_size or
                target.tokens != list(ids[:end]) or draft is None or draft.length != end):
            raise ValueError('Expected a complete target/MTP chunk with known lookahead')
        key = tuple(ids[:end+1])
        if any(entry.chunk_size == chunk_size and entry.key == key for entry in self.entries):
            self._counts['duplicate'] += 1
            return True

        shared, copied = snapshot_requirements(target, draft)
        if reserved_bytes([], shared, copied) > self.max_bytes:
            self._counts['oversize'] += 1
            return False
        protected, replaced_head, new_branch = self._branch_plan(key, end, chunk_size)
        # Published snapshots are immutable. Their address/size maps contain
        # no tensor references and avoid rescanning every long token list.
        storages = [entry.storages for entry in self.entries]
        kept = set(range(len(self.entries)))
        evicted = []
        candidates = sorted((index for index, entry in enumerate(self.entries)
                             if id(entry) not in protected),
                            key=lambda index: self.entries[index].last_used)
        while (len(kept)+1 > self.max_entries or
               reserved_bytes((storages[index] for index in kept), shared, copied) > self.max_bytes):
            if not candidates:
                self._counts['budget'] += 1
                return False
            index = candidates.pop(0)
            kept.remove(index)
            evicted.append(index)
        # Plans contain only indices/storage addresses. Remove all cache/head
        # references before cloning so evicted allocations really can release.
        self._evict(evicted)
        target_copy = draft_copy = None
        try:
            target_copy = clone_state(target)
            draft_copy = clone_state(draft)
            new_storages = storage_map((target_copy, draft_copy))
            if reserved_bytes((entry.storages for entry in self.entries), new_storages, 0) > self.max_bytes:
                self._counts['budget'] += 1
                return False
        except Exception:
            self._counts['clone_failure'] += 1
            target_copy = draft_copy = None
            raise
        self._clock += 1
        checkpoint = SnapshotCheckpoint(key, end, target_copy, draft_copy, chunk_size,
                                        self._clock, new_storages)
        self.entries.append(checkpoint)
        if replaced_head is not None:
            self._heads = [head for head in self._heads if id(head) != replaced_head] + [checkpoint]
        elif new_branch:
            self._heads = (self._heads + [checkpoint])[-2:]
        self.chunk_size = chunk_size
        self._counts['publish'] += 1
        self.generation += 1
        return True

    def clear(self):
        changed = bool(self.entries)
        self.entries = []
        self._heads = []
        self.chunk_size = None
        if changed:
            self.generation += 1

    def release(self, owner):
        # No request owns the published pool. Closing a request never drops a
        # completed checkpoint that another request can safely check out.
        return None


class MTPSession:
    def __init__(self, ids, budget, eos, chunk_size, on_tokens, on_progress,
                 should_cancel, draft_tokens, policy, low_sync):
        if (not ids or not isinstance(budget, int) or budget < 1 or
                not isinstance(chunk_size, int) or chunk_size <= 0 or chunk_size % 64):
            raise ValueError('Expected a nonempty prompt, positive budget and aligned prefill chunks')
        self.ids = list(ids)
        self.budget, self.eos, self.chunk_size = budget, set(eos), chunk_size
        self.on_tokens, self.on_progress, self.should_cancel = on_tokens, on_progress, should_cancel
        self.draft_tokens, self.policy, self.low_sync = draft_tokens, policy, low_sync
        self.selector = GreedySelector() if low_sync else None
        self.state = self.draft_state = self.draft_multi = self.proposal = None
        self.position = self.reused = self.checkpoint_tokens = 0
        self.generated, self.chunks, self.iterations = [], [], []
        self.stats = dict(proposed=0, accepted=0, corrections=0, bonus_tokens=0,
                          single_steps=0, discarded=0, recursive_drafts=0,
                          forced_control_tokens=0)
        self.phase, self.done, self.closed = 'prefill', False, False
        self.stop_reason = None
        self.begin, self.first_seconds = time.perf_counter(), None
        self.cache_seconds = self.draft_prefill_seconds = 0.
        self.cache_owner = object()
        self.cache_enabled = True
        self.prefill_memory = None

    def select(self, logits):
        return self.selector(logits) if self.selector is not None else [
            _token(logits, row) for row in range(len(logits))]

    def progress(self):
        if self.on_progress:
            self.on_progress(self.phase, self.position, self.checkpoint_tokens)

    def emit(self):
        if self.on_tokens:
            self.on_tokens(tuple(self.generated))

    def accept(self, tokens, state):
        self.state = state
        self.generated.extend(int(token) for token in tokens)
        self.done = self.generated[-1] in self.eos or len(self.generated) >= self.budget
        self.phase = 'done' if self.done else 'decode'

    def result(self, cache_bytes):
        if not self.done:
            raise RuntimeError('Cannot finish an incomplete MTP request')
        elapsed = time.perf_counter() - self.begin
        decode_seconds = elapsed - self.first_seconds
        stats = dict(self.stats, enabled=True, strategy='continuous_mtp',
                     scheduling='decode_priority_batch' if self.stats.get('batch_rounds',0) else 'cooperative_round_robin',
                     batched_verification=bool(self.stats.get('batch_verifications',0)),
                     acceptance_rate=self.stats['accepted']/self.stats['proposed']
                         if self.stats['proposed'] else None)
        snapshot = getattr(self.policy, 'snapshot', None)
        stats['policy'] = (snapshot() if snapshot is not None else
                           dict(strategy='acceptance', depth=self.policy.depth())
                           if self.policy is not None else
                           dict(strategy='fixed', depth=self.draft_tokens))
        return dict(generated_ids=list(self.generated), output_tokens=len(self.generated),
            finish_reason='stop' if self.stop_reason is not None or self.generated[-1] in self.eos else 'length',
            stop_reason=self.stop_reason,
            first_token_seconds=self.first_seconds, request_seconds=elapsed,
            decode_tokens_per_second=(len(self.generated)-1)/decode_seconds
                if len(self.generated)>1 and decode_seconds>0 else None,
            active_decode_seconds=sum(row['seconds'] for row in self.iterations),
            speculative=stats, iterations=list(self.iterations), prefill_chunks=list(self.chunks),
            prefill_chunk_size=self.chunk_size,
            target_prefill_seconds=sum(row['target_seconds'] for row in self.chunks),
            draft_prefill_seconds=self.draft_prefill_seconds,
            # The first output is exposed after the final paired prefill turn.
            draft_prefill_before_first_seconds=self.draft_prefill_seconds,
            draft_prefill_after_first_seconds=0., low_sync=self.low_sync,
            borrowed_graph_cache=False, draft_tokens=self.draft_tokens,
            prefill_memory=dict(self.prefill_memory) if self.prefill_memory is not None else None,
            prefix_cache=dict(reused_tokens=self.reused, uncached_tokens=len(self.ids)-self.reused,
                checkpoint_seconds=self.cache_seconds, retained_bytes=cache_bytes,
                namespace='mtp_target_draft'))

    def close(self):
        self.closed = True
        self.state = self.draft_state = self.draft_multi = self.proposal = None
        self.ids = []
        self.selector = self.policy = None
        self.on_tokens = self.on_progress = self.should_cancel = None


class ContinuousMTPRunner:
    session_type=MTPSession
    def __init__(self, engine, draft, cache=None, *, draft_tokens='adaptive',
                 policy_factory=None, low_sync=True):
        if draft_tokens not in (1, 2, 'adaptive'):
            raise ValueError('Expected one/two drafts or adaptive depth')
        if policy_factory is not None and draft_tokens != 'adaptive':
            raise ValueError('A draft policy requires adaptive depth')
        if cache is not None and not isinstance(cache, ContinuousMTPPrefixCache):
            raise TypeError('Continuous MTP requires ContinuousMTPPrefixCache')
        self.engine, self.draft, self.cache = engine, draft, cache
        self.prefill_allocator = PrefillMemoryTrimmer(engine.embedding.device)
        self.draft_tokens, self.policy_factory, self.low_sync = draft_tokens, policy_factory, low_sync

    def _cancel(self, session):
        if session.should_cancel is not None and session.should_cancel():
            _sync(self.engine)
            raise GenerationCancelled(session.phase, session.position, session.checkpoint_tokens)

    def _wait(self):
        if not self.low_sync:
            _sync(self.engine)

    @torch.no_grad()
    def start(self, ids, budget, eos, chunk_size, on_tokens=None, on_progress=None,
              cache=True, should_cancel=None):
        policy = ((self.policy_factory or AdaptiveDraftDepth)()
                  if self.draft_tokens == 'adaptive' else None)
        session = self.session_type(ids, budget, eos, chunk_size, on_tokens, on_progress,
                             should_cancel, self.draft_tokens, policy, self.low_sync)
        session.prefill_memory = memory_summary(self.prefill_allocator.enabled,self.prefill_allocator.threshold_bytes)
        session.cache_enabled = cache
        self._cancel(session)
        begin = time.perf_counter()
        if self.cache is not None and cache:
            # Only checkout state belongs to this request; discard cache-entry
            # references immediately, including entries from an earlier owner.
            target, draft, reused, _ = self.cache.checkout(ids, chunk_size)
            session.state, session.draft_state, session.reused = target, draft, reused
            session.position = session.checkpoint_tokens = reused
        _sync(self.engine)
        session.cache_seconds += time.perf_counter() - begin
        self._cancel(session)
        return session

    def _remember_draft(self, session, output):
        session.proposal = select_draft(output, session.select)[0]
        # MTP's QSA forward returns a new state and uses immutable/COW KV.
        # No MTP CUDA Graph owns that state. Only the recursive draft's last
        # HC row survives a turn; cloning detaches it from any output workspace.
        session.draft_state = output.state
        session.draft_multi = output.multi[-1:].clone()

    def stop(self, session, reason):
        """Finish at a committed API boundary without changing model history."""
        if session.closed or not session.generated or session.phase == 'prefill':
            raise ValueError('Expected an open request with committed output')
        if reason not in ('protocol_end_marker', 'reasoning_reopened'):
            raise ValueError('Unknown semantic generation stop reason')
        if session.done:
            return False
        session.stop_reason = reason
        session.done, session.phase = True, 'done'
        return True

    @torch.no_grad()
    def prefill(self, session):
        if session.closed or session.phase != 'prefill':
            raise ValueError('Request is not awaiting prefill')
        self._cancel(session)
        # The previous turn's local activations have left scope. Keep the
        # existing dense KV and all live request/Graph tensors; only trim
        # oversized, unused allocator reservations before the next chunk.
        record_memory_check(session.prefill_memory,self.prefill_allocator.before_prefill())
        start = session.position
        end = min(start + session.chunk_size, len(session.ids))
        begin = time.perf_counter()
        hint_next_prefill_chunk(self.engine, session.ids, start, end, session.chunk_size)
        dflash = getattr(self, 'dflash', None); taps = None
        if dflash is not None and getattr(session, 'dflash_ctx', 'unset') == 'unset':
            session.dflash_ctx = dflash.new_context(start)
        if dflash is not None and session.dflash_ctx is not None:
            taps = {}
            def observer(index, value, _taps=taps, _tap=dflash.tap):
                if isinstance(index, int) and index in DFLASH_LAYERS: _taps[index] = _tap(*value)
        if taps is None:
            logits, state, hidden = self.engine(session.ids[start:end], session.state,
                return_hidden=True, prefill_chunk=True, owned_graph_cache=True)
        else:
            logits, state, hidden = self.engine(session.ids[start:end], session.state, observer,
                return_hidden=True, prefill_chunk=True, owned_graph_cache=True)
        if taps is not None:
            if session.dflash_ctx.length != start:
                session.dflash_ctx = None
            else:
                dflash.extend(session.dflash_ctx, torch.cat([taps[i] for i in DFLASH_LAYERS], -1))
        _sync(self.engine)
        target_seconds = time.perf_counter() - begin
        final = end == len(session.ids)
        token = session.select(logits)[0] if final else None
        draft_seconds = 0.
        if session.budget >= 3 and (not final or token not in session.eos):
            shifted = (session.ids[start+1:end] + [token] if final
                       else session.ids[start+1:end+1])
            begin = time.perf_counter()
            output = self.draft(shifted, hidden.multi, session.draft_state, prefill_chunk=True)
            if final:
                self._remember_draft(session, output)
            else:
                session.draft_state = output.state
            _sync(self.engine)
            draft_seconds = time.perf_counter() - begin
        session.state, session.position = state, end
        session.chunks.append(dict(start=start, end=end, target_seconds=target_seconds,
                                   draft_seconds=draft_seconds))
        session.draft_prefill_seconds += draft_seconds
        if not final and session.budget >= 3 and self.cache is not None and session.cache_enabled:
            begin = time.perf_counter()
            if self.cache.publish(session.ids, end, state, session.draft_state,
                                  session.chunk_size, session.cache_owner):
                session.checkpoint_tokens = end
            _sync(self.engine)
            session.cache_seconds += time.perf_counter() - begin
        if final:
            session.accept([token], state)
            session.first_seconds = time.perf_counter() - session.begin
            if session.done or session.budget - len(session.generated) < 2:
                session.draft_state = session.draft_multi = session.proposal = None
        self._cancel(session)

    @torch.no_grad()
    def decode(self, sessions, *, emission_limit=None):
        if (not 1 <= len(sessions) <= 4 or len({id(s) for s in sessions}) != len(sessions)
                or any(s.closed or s.phase != 'decode' for s in sessions)):
            raise ValueError('Expected one to four distinct decode-ready MTP sessions')
        if emission_limit is not None and (type(emission_limit) is not int or emission_limit < 1):
            raise ValueError('Expected a positive per-round emission limit')
        # Callbacks live in the scheduler's per-request after_step boundary.
        # A scheduler needing independent error handling calls decode([s])
        # for each request, because these are separate rounds, never a B2 step.
        for session in sessions:
            self._decode_one(session, emission_limit)

    def _one_token(self, session, forced=None):
        """Consume the known token and commit the next token in both models.

        Forced tokens are explicit reasoning-budget control tokens, counted in
        the ordinary output/context budget. Draft state still follows the same
        shifted target HC rows; no disposable proposal is ever committed.
        """
        known = session.generated[-1]
        logits, state, hidden = self.engine([known], session.state,
            return_hidden=True, owned_graph_cache=True)
        token = session.select(logits)[0] if forced is None else forced
        session.accept([token], state)
        if not session.done and session.budget-len(session.generated) >= 2:
            output = self.draft([token], hidden.multi, session.draft_state)
            self._remember_draft(session, output)
        else:
            session.draft_state = session.draft_multi = session.proposal = None
        key = 'single_steps' if forced is None else 'forced_control_tokens'
        session.stats[key] += 1

    @torch.no_grad()
    def force_next(self, session, token):
        if session.closed or session.phase != 'decode' or type(token) is not int or token < 0 or token in session.eos:
            raise ValueError('Expected a decode-ready request and a non-EOS control token')
        self._cancel(session)
        self._wait()
        begin = time.perf_counter()
        self._one_token(session, token)
        self._wait()
        session.iterations.append(dict(mode='reasoning_control', emitted=1,
            seconds=time.perf_counter()-begin))
        self._cancel(session)

    def _decode_one(self, session, emission_limit=None):
        self._cancel(session)
        known, remaining = session.generated[-1], session.budget - len(session.generated)
        if emission_limit is not None:
            remaining = min(remaining, emission_limit)
        measured_policy = session.policy is not None and hasattr(session.policy, 'observe_round')
        if measured_policy:
            _sync(self.engine)
        else:
            self._wait()
        begin = time.perf_counter()
        if remaining == 1:
            self._one_token(session)
            self._wait()
            session.iterations.append(dict(mode='single', emitted=1, seconds=time.perf_counter()-begin))
            self._cancel(session)
            return
        if session.proposal is None or session.draft_multi is None:
            raise RuntimeError('Missing learned MTP proposal')
        proposals = [session.proposal]
        depth = session.policy.depth() if session.policy is not None else session.draft_tokens
        if depth not in (1, 2):
            raise ValueError('Draft policy must choose one or two drafts')
        if depth == 2 and remaining >= 3 and proposals[0] not in session.eos:
            # This branch is disposable. QSA forward must preserve the original
            # committed draft state, exactly as in the serial MTP runner.
            future = self.draft([proposals[0]], session.draft_multi, session.draft_state)
            proposals.append(select_draft(future, session.select)[0])
            session.stats['recursive_drafts'] += 1
            del future
        verify_ids = [known] + proposals
        # The default verifier clones all Graph prefix banks. Borrowed caches
        # are never allowed to survive a cooperative request switch.
        target, target_state, hidden = self.engine.verify_tokens(verify_ids, session.state)
        predictions = session.select(target)  # checks every row, including rejected futures
        first_accepted = predictions[0] == proposals[0]
        session.stats['proposed'] += len(proposals)
        emitted, accepted = [], True
        for row, proposed in enumerate(proposals):
            if predictions[row] != proposed:
                emitted.append(predictions[row])
                session.stats['corrections'] += 1
                accepted = False
                break
            emitted.append(proposed)
            session.stats['accepted'] += 1
            if proposed in session.eos:
                break
        session.stats['discarded'] += len(proposals) - len(emitted)
        if accepted and emitted[-1] not in session.eos:
            emitted.append(predictions[len(proposals)])
            session.stats['bonus_tokens'] += 1
        committed = len(emitted)
        # Select even the final prefix to avoid retaining the other speculative
        # states in EngineState.prefixes. The last output remains unconsumed.
        state = target_state.prefixes[committed-1]
        session.accept(emitted, state)
        if not session.done and session.budget - len(session.generated) >= 2:
            output = self.draft(emitted, hidden.multi[:committed], session.draft_state)
            self._remember_draft(session, output)
        else:
            session.draft_state = session.draft_multi = session.proposal = None
        if measured_policy:
            _sync(self.engine)
        else:
            self._wait()
        elapsed = time.perf_counter() - begin
        if session.policy is not None:
            observer = getattr(session.policy, 'observe_round', None)
            if observer is not None:
                observer(first_accepted, depth=len(proposals), accepted_tokens=committed,
                         elapsed_seconds=elapsed)
            else:
                session.policy.observe(first_accepted)
        session.iterations.append(dict(mode='accept' if accepted else 'reject', emitted=committed,
            verification_rows=len(verify_ids), first_proposal_accepted=first_accepted, seconds=elapsed))
        self._cancel(session)

    def close(self, session):
        try:
            _sync(self.engine)
        finally:
            if self.cache is not None:
                self.cache.release(session.cache_owner)
            session.close()
