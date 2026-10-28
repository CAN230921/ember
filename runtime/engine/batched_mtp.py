"""Opt-in continuous MTP: batch proposals, verification and committed drafts.

Each lane chooses its own accepted prefix, EOS, budget and control token.
Errors/cancellation are returned per lane to the model-owner scheduler; the
runner never closes a peer or returns its context lease from an HTTP thread.
"""
import os
import time
import torch

from .batch_mtp_model import BatchMTPModel, BatchMTPDraft, materialize_views, LAZY_CLONE
from .continuous_mtp import ContinuousMTPRunner
from .draft_head import select_draft
from .limits import VERIFY_ROWS
from .speculative_policy import DeepDraftDepth, DraftSourceChooser
from .dflash_draft import DFlashDraft, LAYERS as DFLASH_LAYERS
from .model import ForwardHidden

PAD_RAGGED = os.environ.get('SPARK_MTP_PAD', '0') == '1'
PROMPT_LOOKUP = os.environ.get('SPARK_PROMPT_LOOKUP', '0') == '1'
# Opt-in deeper single-lane speculation: up to VERIFY_ROWS-1 chained MTP drafts per round.
DRAFT_ROW_COST = float(os.environ.get('SPARK_DRAFT_ROW_COST', '0.2'))
# Opt-in DFlash2 block drafter for single-lane decode (draft-only; the target verifies every token).
DFLASH_DIR = os.environ.get('SPARK_DFLASH_DIR')
DFLASH_BLOCK = int(os.environ.get('SPARK_DFLASH_BLOCK', '5'))
DFLASH_PAUSE_MTP = os.environ.get('SPARK_DFLASH_PAUSE_MTP', '1') == '1'
DFLASH_COMPACT_HEAD = os.environ.get('SPARK_DFLASH_COMPACT_HEAD', '1') == '1'
# Measured 8-row verification adds ~11-14% of a one-row step per extra row (2026-10-21), below the MTP chain's cost.
DFLASH_ROW_COST = float(os.environ.get('SPARK_DFLASH_ROW_COST', '0.12'))
# Multi-lane DFlash (opt-in): lanes keep tapping features in batch verification and may verify a DFlash block
# of up to DFLASH_BATCH_ROWS rows next to 1-3-row MTP lanes (one ragged target pass, no padding to the deep lane).
# Bit-exact, but ~5.5 GB more peak memory with four lanes (2026-10-21), so off by default.
DFLASH_BATCH = os.environ.get('SPARK_DFLASH_BATCH', '0') == '1'
DFLASH_BATCH_ROWS = int(os.environ.get('SPARK_DFLASH_BATCH_ROWS', str(VERIFY_ROWS)))
# Chooser costs relative to a multi-lane step (measured 2026-10-21: 2/4 lanes x 3 rows 68/99 ms, each extra
# deep-lane row +3.6/+7.4 ms, NVFP4 DFlash proposal ~3 ms); a lane's base share of the step is 1/lanes.
BATCH_ROW_COST = float(os.environ.get('SPARK_CHOOSER_BATCH_ROW_COST', '0.065'))
BATCH_DFLASH_COST = float(os.environ.get('SPARK_CHOOSER_BATCH_DFLASH_COST', '0.04'))
LOOKUP_MAX = 4  # prompt-lookup proposal length (trimmed to the lane's verification width)


class PromptLookup:
    """Incremental 3-gram index over a lane's committed tokens (prompt + output).

    Proposes the two tokens that followed the most recent earlier occurrence of the last
    three tokens. Proposals only steer verification; the target model checks every token.
    EMAs of first-proposal hits for lookup and MTP decide which source a lane uses.
    """
    def __init__(self):
        self.index = {}; self.upto = 2; self.ema = dict(lookup=0.6, mtp=0.5)

    def propose(self, tokens):
        # Index keys whose next token exists: key ends at j, next token at j+1 <= len-1.
        for j in range(self.upto, len(tokens)-1):
            self.index[(tokens[j-2], tokens[j-1], tokens[j])] = j+1
        self.upto = max(self.upto, len(tokens)-1)
        if len(tokens) < 4:
            return []
        start = self.index.get((tokens[-3], tokens[-2], tokens[-1]))
        return [] if start is None else list(tokens[start:start+LOOKUP_MAX])

    def observe(self, source, hit):
        self.ema[source] = 0.9*self.ema[source]+0.1*float(hit)


def trim_hidden(hidden, rows):
    # Drop padded verification rows; keep whichever per-row fields the output carries.
    sampled = getattr(hidden, 'sampled', None)
    return ForwardHidden(hidden.multi[:rows], None if sampled is None else sampled[:rows])


def device_failure(error):
    return isinstance(error, RuntimeError) and any(s in str(error).lower() for s in ('cuda', 'nonfinite'))


class BatchedMTPRunner(ContinuousMTPRunner):
    def __init__(self, *args, batch_target=None, batch_draft=None, **kwargs):
        if kwargs.get('policy_factory') is not None:
            raise ValueError('Batched MTP currently requires the acceptance policy')
        super().__init__(*args, **kwargs)
        self.batch_target = batch_target if batch_target is not None else BatchMTPModel(self.engine)
        self.batch_draft = batch_draft if batch_draft is not None else BatchMTPDraft(self.draft)
        self.dflash = (DFlashDraft(DFLASH_DIR, self.engine, DFLASH_BLOCK,
                                   head=getattr(self.draft, 'head', None) if DFLASH_COMPACT_HEAD else None)
                       if DFLASH_DIR else None)

    def snapshot(self):
        return dict(enabled=True, target=self.batch_target.snapshot(), draft_calls=self.batch_draft.calls,
                    scheduling='decode_priority_batch', mixed_prefill_decode=False)

    def _mtp_catch_up(self, session):
        backlog = session.mtp_backlog; session.mtp_backlog = None
        tokens = [t for emitted, _ in backlog for t in emitted]
        multi = torch.cat([rows for _, rows in backlog])
        # One prefill-mode draft pass over every row committed while DFlash drafted (draft-only arithmetic).
        output = self.draft(tokens, multi, session.draft_state, prefill_chunk=True)
        self._remember_draft(session, output)
        session.stats['mtp_catch_up_rows'] = session.stats.get('mtp_catch_up_rows', 0)+len(tokens)

    def _draft_many(self, jobs):
        if len(jobs) == 1:
            ids, multi, state = jobs[0]
            return [self.draft(ids, multi, state)]
        return self.batch_draft.forward([j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs])

    @torch.no_grad()
    def decode(self, sessions, *, emission_limit=None):
        if len(sessions) == 1:
            return super().decode(sessions, emission_limit=emission_limit)
        errors = self.decode_batch(sessions, emission_limits=[emission_limit]*len(sessions))
        if any(errors):
            raise next(error for error in errors if error is not None)

    @torch.no_grad()
    def decode_batch(self, sessions, *, emission_limits=None, forced_tokens=None):
        count = len(sessions)
        if (not 1 <= count <= 4 or len({id(s) for s in sessions}) != count or
                any(s.closed or s.phase != 'decode' for s in sessions)):
            raise ValueError('Expected one to four distinct decode-ready MTP sessions')
        limits = [None]*count if emission_limits is None else list(emission_limits)
        forced = [None]*count if forced_tokens is None else list(forced_tokens)
        if len(limits) != count or len(forced) != count:
            raise ValueError('Control lane counts differ')
        if any(v is not None and (type(v) is not int or v < 1) for v in limits):
            raise ValueError('Expected positive per-lane emission limits')
        if any(t is not None and (type(t) is not int or t < 0 or t in s.eos)
               for s, t in zip(sessions, forced)):
            raise ValueError('Expected non-EOS per-lane control tokens')
        errors = [None]*count; plans = {}; recursive = []; taps = {}; catch_up_seconds = {}
        if count > 1 and not DFLASH_BATCH:
            for s in sessions:
                s.dflash_ctx = None   # batch verification would not tap features; this context would go stale
        self._wait(); begin = time.perf_counter()
        def fail(indices, error):
            for i in indices:
                errors[i] = error
            if device_failure(error):
                for i in plans:
                    if errors[i] is None:
                        errors[i] = error
                return True
            return False
        for i, session in enumerate(sessions):
            try:
                self._cancel(session)
                remaining = session.budget-len(session.generated)
                if limits[i] is not None:
                    remaining = min(remaining, limits[i])
                single = remaining == 1 or forced[i] is not None
                deep = count == 1 and VERIFY_ROWS > 3 and not single
                ctx = getattr(session, 'dflash_ctx', None); source = None
                dflash_lane = deep or (count > 1 and DFLASH_BATCH and VERIFY_ROWS > 3 and not single)
                if dflash_lane and self.dflash is not None and ctx is not None and ctx.length == len(session.state.tokens):
                    chooser = session.__dict__.get('draft_chooser')
                    if chooser is None:
                        chooser = session.draft_chooser = DraftSourceChooser()
                    source = chooser.choose()
                if source != 'dflash' and session.__dict__.get('mtp_backlog') and not single:
                    rows_behind = sum(len(emitted) for emitted, _ in session.mtp_backlog)
                    t_catch = time.perf_counter(); self._mtp_catch_up(session); self._wait()
                    catch_up_seconds[i] = time.perf_counter()-t_catch
                    if session.__dict__.get('draft_chooser') is not None:
                        session.draft_chooser.charge_catch_up(rows_behind)
                proposals = [] if single or source == 'dflash' else [session.proposal]
                if not single and source != 'dflash' and (session.proposal is None or session.draft_multi is None):
                    raise RuntimeError('Missing learned MTP proposal')
                if deep:
                    if getattr(session, 'deep_policy', None) is None:
                        session.deep_policy = DeepDraftDepth(VERIFY_ROWS-1, row_cost=DRAFT_ROW_COST)
                    depth = session.deep_policy.depth()
                else:
                    depth = 1 if single else (session.policy.depth() if session.policy is not None else session.draft_tokens)
                    if depth not in (1, 2):
                        raise ValueError('Draft policy must choose one or two drafts')
                plans[i] = dict(single=single, proposals=proposals, ids=[session.generated[-1]], batch_size=1)
                lookup = []
                if PROMPT_LOOKUP and not single and source != 'dflash':
                    if getattr(session, 'prompt_lookup', None) is None:
                        session.prompt_lookup = PromptLookup()
                    lookup = session.prompt_lookup.propose(session.state.tokens)
                    plans[i].update(mtp_first=proposals[0], lookup_first=lookup[0] if lookup else None)
                    pl = session.prompt_lookup
                    if lookup and (lookup[0] == proposals[0] or pl.ema['lookup'] > pl.ema['mtp']):
                        limit = VERIFY_ROWS-1 if count == 1 else 2
                        take = []
                        for token in lookup[:limit]:
                            if len(take)+2 > remaining: break
                            take.append(token)
                            if token in session.eos: break
                        take = take or lookup[:1]
                        plans[i]['proposals'] = take; plans[i]['source'] = 'lookup'
                        session.stats['lookup_steps'] = session.stats.get('lookup_steps', 0)+1
                        continue
                if source == 'dflash':
                    # Positional adaptive depth (as the MTP chain): verify only the prefix worth its rows.
                    dpol = session.__dict__.get('dflash_depth')
                    if dpol is None:
                        dpol = session.dflash_depth = DeepDraftDepth(min(DFLASH_BLOCK-1, VERIFY_ROWS-1), row_cost=DFLASH_ROW_COST, warmup=2)
                    # The first rounds verify the whole block so the source choice sees what DFlash can reach;
                    # later probes use the learned positional depth, so poor content probes cheaply.
                    limit = dpol.max_depth if dpol.rounds < 2 else dpol.depth()
                    if count > 1: limit = min(limit, DFLASH_BATCH_ROWS-1)
                    plans[i]['dflash_depth'] = limit
                    take = []
                    for token in self.dflash.propose(ctx, session.generated[-1])[:limit]:
                        if len(take)+2 > remaining: break
                        take.append(token)
                        if token in session.eos: break
                    plans[i]['proposals'] = take; plans[i]['source'] = 'dflash'
                    session.stats['dflash_steps'] = session.stats.get('dflash_steps', 0)+1
                    continue
                if deep:
                    # Chain disposable MTP drafts; the committed draft state is never mutated.
                    multi, dstate, last = session.draft_multi, session.draft_state, proposals[0]
                    while len(proposals) < depth and remaining >= len(proposals)+2 and last not in session.eos:
                        deep_head = getattr(self.draft, 'deep_head', None)
                        future = (self.draft([last], multi, dstate) if deep_head is None
                                  else self.draft([last], multi, dstate, head=deep_head))
                        last = select_draft(future, session.select)[0]
                        proposals.append(last); session.stats['recursive_drafts'] += 1
                        multi, dstate = future.multi[-1:], future.state
                    plans[i]['depth'] = depth
                elif depth == 2 and remaining >= 3 and proposals[0] not in session.eos:
                    recursive.append(i)
            except Exception as error:
                errors[i] = error
        if recursive:
            try:
                futures = self._draft_many([([sessions[i].proposal], sessions[i].draft_multi, sessions[i].draft_state)
                                            for i in recursive])
                if len(futures) != len(recursive):
                    raise RuntimeError('Recursive draft result lane counts differ')
                for i, output in zip(recursive, futures):
                    plans[i]['proposals'].append(select_draft(output, sessions[i].select)[0])
                    sessions[i].stats['recursive_drafts'] += 1
            except Exception as error:
                if fail(recursive, error):
                    return errors
        groups = {}
        for i, plan in plans.items():
            if errors[i] is not None:
                continue
            try:
                self._cancel(sessions[i])
            except Exception as error:
                errors[i] = error; continue
            plan['ids'] += plan['proposals']
            groups.setdefault(len(plan['ids']), []).append(i)
        draft_jobs = []; finished = []
        padded = {}
        ragged = len(groups) > 1 and any(plans[i].get('source') == 'dflash' for members in groups.values() for i in members)
        if ragged:
            # A DFlash lane next to MTP lanes: one ragged pass (lanes keep their own row counts).
            groups = {max(groups): sorted(i for members in groups.values() for i in members)}
        elif PAD_RAGGED and len(groups) > 1:
            # One target pass for every lane: shorter lanes repeat their last id up to the
            # longest lane. Causal rows ahead of the padding are unaffected, commits only
            # read prefixes within each lane's real length, and padded rows are dropped.
            rows = max(groups); indices = sorted(i for members in groups.values() for i in members)
            padded = {i: len(plans[i]['ids']) for i in indices}
            groups = {rows: indices}
        for rows, indices in groups.items():
            try:
                if len(indices) > 1:
                    ids = [plans[i]['ids'] if ragged else plans[i]['ids']+[plans[i]['ids'][-1]]*(rows-len(plans[i]['ids']))
                           for i in indices]
                    observer = None
                    if self.dflash is not None and any(getattr(sessions[i], 'dflash_ctx', None) is not None for i in indices):
                        group_taps = {}
                        def observer(index, value, _taps=group_taps, _tap=self.dflash.tap):
                            if isinstance(index, int) and index in DFLASH_LAYERS: _taps[index] = _tap(*value)
                    outputs = self.batch_target.verify(ids, [sessions[i].state for i in indices],
                                                       *(() if observer is None else (observer,)))
                    if observer is not None:
                        offset = 0
                        for i, lane_ids in zip(indices, ids):
                            if getattr(sessions[i], 'dflash_ctx', None) is not None:
                                taps[i] = {l: group_taps[l][offset:offset+len(lane_ids)] for l in DFLASH_LAYERS}
                            offset += len(lane_ids)
                    if padded:
                        outputs = [(logits[:padded[i]], state, trim_hidden(hidden, padded[i]))
                                   for i, (logits, state, hidden) in zip(indices, outputs)]
                else:
                    i = indices[0]; session = sessions[i]; observer = None
                    if self.dflash is not None and getattr(session, 'dflash_ctx', None) is not None:
                        taps[i] = lane_taps = {}
                        def observer(index, value, _taps=lane_taps, _tap=self.dflash.tap):
                            if isinstance(index, int) and index in DFLASH_LAYERS: _taps[index] = _tap(*value)
                    extra = () if observer is None else (observer,)
                    if plans[i]['single']:
                        outputs = [self.engine(plans[i]['ids'], session.state, *extra, return_hidden=True, owned_graph_cache=True)]
                    else:
                        outputs = [self.engine.verify_tokens(plans[i]['ids'], session.state, *extra,
                                                               **({'lazy_prefixes': True} if LAZY_CLONE else {}))]
                if len(outputs) != len(indices):
                    raise RuntimeError('Verification result lane counts differ')
                predictions = [sessions[i].select(output[0]) for i, output in zip(indices, outputs)]
                if any(len(value) != len(plans[i]['ids']) for i, value in zip(indices, predictions)):
                    raise RuntimeError('Verification prediction row counts differ')
            except Exception as error:
                if fail(indices, error):
                    return errors
                continue
            for i, prediction, output in zip(indices, predictions, outputs):
                session, plan = sessions[i], plans[i]
                try:
                    self._cancel(session)
                    _, state, hidden = output
                    first_accepted = None; accepted = True
                    if plan['single']:
                        emitted = [prediction[0] if forced[i] is None else forced[i]]
                        if len(indices) > 1:
                            state = state.prefixes[0]
                        key = 'single_steps' if forced[i] is None else 'forced_control_tokens'
                        session.stats[key] += 1
                    else:
                        proposals = plan['proposals']; first_accepted = prediction[0] == proposals[0]
                        if 'mtp_first' in plan:
                            pl = session.prompt_lookup
                            pl.observe('mtp', plan['mtp_first'] == prediction[0])
                            if plan['lookup_first'] is not None:
                                pl.observe('lookup', plan['lookup_first'] == prediction[0])
                        session.stats['proposed'] += len(proposals); emitted = []; matched = 0
                        for row, proposed in enumerate(proposals):
                            session.stats[f'pos{row}_tried'] = session.stats.get(f'pos{row}_tried', 0)+1
                            if prediction[row] != proposed:
                                emitted.append(prediction[row]); session.stats['corrections'] += 1
                                accepted = False; break
                            session.stats[f'pos{row}_hit'] = session.stats.get(f'pos{row}_hit', 0)+1
                            emitted.append(proposed); session.stats['accepted'] += 1; matched += 1
                            if proposed in session.eos:
                                break
                        session.stats['discarded'] += len(proposals)-len(emitted)
                        if accepted and emitted[-1] not in session.eos:
                            emitted.append(prediction[len(proposals)]); session.stats['bonus_tokens'] += 1
                        state = state.prefixes[len(emitted)-1]
                    session.accept(emitted, materialize_views(state))
                    if i in taps and getattr(session, 'dflash_ctx', None) is not None:
                        self.dflash.push(session.dflash_ctx, torch.cat([taps[i][l][:len(emitted)] for l in DFLASH_LAYERS], -1))
                    plan.update(batch_size=len(indices), rows=len(plan['ids']), emitted=len(emitted),
                                first_accepted=first_accepted, accepted=accepted,
                                matched=None if plan['single'] else matched)
                    if len(indices) > 1:
                        session.stats['batch_rounds'] = session.stats.get('batch_rounds', 0)+1
                        if not plan['single']:
                            session.stats['batch_verifications'] = session.stats.get('batch_verifications', 0)+1
                    chooser = session.__dict__.get('draft_chooser')
                    if (not session.done and session.budget-len(session.generated) >= 2 and plan.get('source') == 'dflash'
                            and DFLASH_PAUSE_MTP and chooser is not None and chooser.preferred() == 'dflash'):
                        # MTP stays behind while DFlash drafts; its committed rows are replayed on the next MTP step.
                        session.mtp_backlog = (session.__dict__.get('mtp_backlog') or [])+[(emitted, hidden.multi[:len(emitted)].clone())]
                        session.proposal = None
                    elif not session.done and session.budget-len(session.generated) >= 2:
                        draft_jobs.append((i, (emitted, hidden.multi[:len(emitted)].clone(), session.draft_state)))
                    else:
                        session.draft_state = session.draft_multi = session.proposal = None
                    finished.append(i)
                except Exception as error:
                    if fail([i], error):
                        return errors
        if draft_jobs:
            indices = [i for i, _ in draft_jobs]
            try:
                outputs = self._draft_many([job for _, job in draft_jobs])
                if len(outputs) != len(indices):
                    raise RuntimeError('Committed draft result lane counts differ')
                for i, output in zip(indices, outputs):
                    self._remember_draft(sessions[i], output)
            except Exception as error:
                fail(indices, error)
        self._wait(); elapsed = time.perf_counter()-begin
        for i in finished:
            session, plan = sessions[i], plans[i]
            if not plan['single'] and session.policy is not None and plan.get('source') not in ('lookup', 'dflash'):
                session.policy.observe(plan['first_accepted'])
            chooser = session.__dict__.get('draft_chooser')
            if chooser is not None and not plan['single'] and plan.get('source') in (None, 'dflash'):
                # MTP catch-up is a cost of the DFlash steps that left MTP behind, not of this MTP step.
                caught = catch_up_seconds.get(i, 0.0)
                chooser.observe(plan.get('source') or 'mtp', plan['emitted'], rows=plan['rows'], drafts=len(plan['proposals']),
                                **({} if plan['batch_size'] == 1 else dict(row_cost=BATCH_ROW_COST, dflash_cost=BATCH_DFLASH_COST,
                                                                           base=1/plan['batch_size'])))
                if caught: chooser.charge('dflash', caught)
                if chooser.hopeless() and not session.__dict__.get('mtp_backlog'):
                    # Stop tapping features and probing for the rest of this request.
                    session.dflash_ctx = None; session.stats['dflash_disabled'] = 1
            if 'dflash_depth' in plan and session.__dict__.get('dflash_depth') is not None:
                session.dflash_depth.observe_counts(plan['matched'], not plan['accepted'], len(plan['proposals']))
            if 'depth' in plan and getattr(session, 'deep_policy', None) is not None:
                session.deep_policy.observe_counts(plan['matched'], not plan['accepted'], len(plan['proposals']))
            session.iterations.append(dict(mode=('reasoning_control' if forced[i] is not None else 'single')
                if plan['single'] else ('accept' if plan['accepted'] else 'reject'), emitted=plan['emitted'],
                verification_rows=plan['rows'], first_proposal_accepted=plan['first_accepted'],
                batch_size=plan['batch_size'], shared_round_seconds=elapsed, seconds=elapsed))
            try:
                self._cancel(session)
            except Exception as error:
                errors[i] = errors[i] or error
        return errors
