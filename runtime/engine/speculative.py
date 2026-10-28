"""Greedy learned MTP with one/two drafts and explicit target commit points."""
import time
import torch
from .request_control import GenerationCancelled
from .draft_head import select_draft
from .ops.ple import hint_next_prefill_chunk


def _sync(engine):
    if engine.embedding.is_cuda:torch.cuda.synchronize(engine.embedding.device)


def _token(logits,row=0):
    if not bool(torch.isfinite(logits).all()):raise RuntimeError('Nonfinite speculative logits')
    return int(logits[row].argmax())


@torch.no_grad()
def generate_mtp(engine,draft,prompt_ids,max_tokens,eos,prefill_chunk_size=None,low_sync=False,borrowed_graph_cache=False,on_tokens=None,prefix_cache=None,draft_tokens=1,should_cancel=None,on_progress=None):
    if max_tokens<1 or not prompt_ids:raise ValueError('Expected a prompt and positive token budget')
    if draft_tokens not in (1,2,'adaptive'):raise ValueError('Expected one/two drafts or adaptive depth')
    policy=None
    if draft_tokens=='adaptive':
        from .speculative_policy import AdaptiveDraftDepth
        policy=AdaptiveDraftDepth()
    generated=[]; iterations=[]
    stats=dict(proposed=0,accepted=0,corrections=0,bonus_tokens=0,single_steps=0,discarded=0,recursive_drafts=0)
    if low_sync:
        from .ops.greedy_select import GreedySelector
        selector=GreedySelector()
        single=lambda logits:selector(logits)[0]
    else:single=_token
    draft_single=lambda output:select_draft(output,lambda logits:[single(logits)])[0]
    wait=lambda:None if low_sync else _sync(engine)
    _sync(engine); request_begin=time.perf_counter(); first_begin=request_begin
    draft_state=None; draft_output=None; proposal=None; draft_prefill_before_first=0.; draft_prompt_start=0
    chunks=[]; reused=0; cache_seconds=0.; entries=[]
    completed_prefill=0
    def cancellation_point(phase):
        if on_progress is not None:
            on_progress(phase,completed_prefill,entries[-1].end if entries else 0)
        if should_cancel is not None and should_cancel():
            # A following request may reuse graph buffers. Drain all work
            # before unwinding, and publish only completed target+MTP pairs.
            _sync(engine)
            if prefix_cache is not None and entries:
                prefix_cache.commit(entries,prefill_chunk_size)
            raise GenerationCancelled(phase,completed_prefill,entries[-1].end if entries else 0)
    cancellation_point('prefill_start')
    if prefill_chunk_size is None:
        logits,state,hidden=engine(prompt_ids,return_hidden=True)
        completed_prefill=len(prompt_ids)
        cancellation_point('prefill_target')
    else:
        if not isinstance(prefill_chunk_size,int) or prefill_chunk_size<=0 or prefill_chunk_size%64:
            raise ValueError('Prefill chunk size must be a positive multiple of 64')
        state=None
        if prefix_cache is not None:
            begin=time.perf_counter()
            state,draft_state,reused,entries=prefix_cache.checkout(prompt_ids,prefill_chunk_size)
            _sync(engine); cache_seconds+=time.perf_counter()-begin
            completed_prefill=reused
            cancellation_point('prefill_checkout')
        for start in range(reused,len(prompt_ids),prefill_chunk_size):
            cancellation_point('prefill_target_start')
            end=min(start+prefill_chunk_size,len(prompt_ids))
            begin=time.perf_counter()
            hint_next_prefill_chunk(engine,prompt_ids,start,end,prefill_chunk_size)
            logits,state,hidden=engine(prompt_ids[start:end],state,return_hidden=True,prefill_chunk=True)
            _sync(engine)
            completed_prefill=end
            chunks.append(dict(start=start,end=end,target_seconds=time.perf_counter()-begin,draft_seconds=0.))
            draft_prompt_start=start
            cancellation_point('prefill_target')
            if end<len(prompt_ids) and max_tokens>=3:
                # The next prompt token is already known. Feed this chunk to
                # MTP now and release its large HC activations before the next
                # target chunk; only the final sampled token is deferred.
                _sync(engine); begin=time.perf_counter()
                output=draft(list(prompt_ids[start+1:end+1]),hidden.multi,draft_state,prefill_chunk=True)
                draft_state=output.state
                _sync(engine); seconds=time.perf_counter()-begin
                draft_prefill_before_first+=seconds; chunks[-1]['draft_seconds']=seconds
                cancellation_point('prefill_draft')
                if prefix_cache is not None:
                    begin=time.perf_counter()
                    prefix_cache.checkpoint(entries,prompt_ids,end,state,draft_state)
                    _sync(engine); cache_seconds+=time.perf_counter()-begin
                del output
                cancellation_point('prefill_checkpoint')
        if prefix_cache is not None:prefix_cache.commit(entries,prefill_chunk_size)
    cancellation_point('first_token')
    token=single(logits); generated.append(token)
    wait(); first_seconds=time.perf_counter()-first_begin
    if on_tokens is not None:on_tokens(tuple(generated))
    draft_prefill_after_first=0.
    if token not in eos and max_tokens-len(generated)>=2:
        cancellation_point('draft_prefill_start')
        begin=time.perf_counter()
        # Target hidden at p is paired with the token at p+1; positions do
        # not shift. The final prefill row is paired with the sampled token.
        shifted=list(prompt_ids[draft_prompt_start+1:])+[token]
        output=(draft(shifted,hidden.multi,draft_state,prefill_chunk=True)
                if prefill_chunk_size is not None else draft(shifted,hidden.multi))
        proposal=draft_single(output); draft_state=output.state; draft_output=output
        wait(); draft_prefill_after_first=time.perf_counter()-begin
        if chunks:chunks[-1]['draft_seconds']=draft_prefill_after_first
        cancellation_point('draft_prefill_complete')
    while len(generated)<max_tokens and generated[-1] not in eos:
        cancellation_point('decode')
        known=generated[-1]; remaining=max_tokens-len(generated)
        wait(); begin=time.perf_counter()
        if remaining==1:
            logits,state=engine([known],state)
            cancellation_point('decode_target')
            generated.append(single(logits)); stats['single_steps']+=1
            wait()
            iterations.append(dict(mode='single',emitted=1,seconds=time.perf_counter()-begin))
            if on_tokens is not None:on_tokens(tuple(generated))
            break
        if proposal is None:raise RuntimeError('Missing learned MTP proposal')
        proposals=[proposal]
        depth=policy.depth() if policy is not None else draft_tokens
        if depth==2 and remaining>=3 and proposal not in eos:
            # This future is temporary. After target verification, advance
            # the original committed draft cache with true target features.
            future=draft([proposal],draft_output.multi[-1:],draft_state)
            proposals.append(draft_single(future))
            stats['recursive_drafts']+=1
            del future
            cancellation_point('decode_draft')
        verify_ids=[known]+proposals
        if borrowed_graph_cache:
            target,target_state,hidden=engine.verify_tokens(verify_ids,state,borrowed_graph_cache=True)
        else:target,target_state,hidden=engine.verify_tokens(verify_ids,state)
        cancellation_point('decode_target')
        predictions=selector(target) if low_sync else [_token(target,row) for row in range(len(verify_ids))]
        if policy is not None:policy.observe(predictions[0]==proposals[0])
        stats['proposed']+=len(proposals)
        emitted=[]; accepted=True
        for row,proposed in enumerate(proposals):
            if predictions[row]!=proposed:
                emitted.append(predictions[row]); stats['corrections']+=1
                accepted=False
                break
            emitted.append(proposed); stats['accepted']+=1
            if proposed in eos:break
        tested=len(emitted)
        stats['discarded']+=len(proposals)-tested
        if accepted and emitted[-1] not in eos:
            emitted.append(predictions[len(proposals)]); stats['bonus_tokens']+=1
        generated.extend(emitted)
        # The last emitted token (including EOS) is always unconsumed.
        # The number of emitted tokens equals the number of committed rows.
        committed=len(emitted)
        state=target_state if committed==len(verify_ids) else target_state.prefixes[committed-1]
        if on_tokens is not None:on_tokens(tuple(generated))
        if len(generated)<max_tokens and generated[-1] not in eos and max_tokens-len(generated)>=2:
            # Draft cache advances only with committed target rows. Rejected
            # speculative hidden/state never enters the draft history.
            output=draft(emitted,hidden.multi[:committed],draft_state)
            proposal=draft_single(output); draft_state=output.state; draft_output=output
        wait()
        iterations.append(dict(mode='accept' if accepted else 'reject',emitted=committed,
                               verification_rows=len(verify_ids),
                               first_proposal_accepted=predictions[0]==proposals[0],
                               seconds=time.perf_counter()-begin))
    decode_seconds=draft_prefill_after_first+sum(x['seconds'] for x in iterations)
    stats['acceptance_rate']=stats['accepted']/stats['proposed'] if stats['proposed'] else None
    return dict(generated_ids=generated,finish_reason='stop' if generated[-1] in eos else 'length',
                output_tokens=len(generated),first_token_seconds=first_seconds,
                draft_prefill_seconds=draft_prefill_before_first+draft_prefill_after_first,
                draft_prefill_before_first_seconds=draft_prefill_before_first,
                draft_prefill_after_first_seconds=draft_prefill_after_first,
                prefill_chunk_size=prefill_chunk_size,iterations=iterations,
                prefill_chunks=chunks,prefix_cache=dict(reused_tokens=reused,
                    uncached_tokens=len(prompt_ids)-reused,checkpoint_seconds=cache_seconds,
                    retained_bytes=prefix_cache.bytes if prefix_cache is not None else 0),
                target_prefill_seconds=sum(item['target_seconds'] for item in chunks),
                low_sync=low_sync,borrowed_graph_cache=borrowed_graph_cache,draft_tokens=draft_tokens,
                decode_tokens_per_second=(len(generated)-1)/decode_seconds if decode_seconds else None,
                request_seconds=time.perf_counter()-request_begin,speculative=stats),state
