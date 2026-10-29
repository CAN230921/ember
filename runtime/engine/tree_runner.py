"""Opt-in tree sessions on the existing owner-thread scheduler and cache.

Prefill keeps paired MTP cache checkpoints. Decode uses private target/DFlash
states only; lanes execute cooperatively and never mix their tree histories.
"""
import os
import time

import torch

from .candidate_tree import CandidateTree
from .continuous_mtp import ContinuousMTPRunner,MTPSession
from .dflash_draft import DFlashDraft,LAYERS
from .prefix_cache import clone_state


class TreeSession(MTPSession):
    def result(self,cache_bytes):
        result=super().result(cache_bytes)
        result['draft_tokens']=None
        result['speculative'].update(strategy='tree',scheduling='cooperative_round_robin',
            batched_verification=False,policy=dict(strategy=self.tree_policy,max_nodes=8,max_leaves=4),
            tree_consumption_rate=self.stats.get('tree_consumed_nodes',0)/self.stats['tree_nodes']
                if self.stats.get('tree_nodes') else None)
        return result

    def close(self):
        self.dflash_ctx=None
        super().close()


class TreeMTPRunner(ContinuousMTPRunner):
    session_type=TreeSession

    def __init__(self,*args,dflash=None,tree_policy='best_first',tree_kernel=True,**kwargs):
        if tree_policy not in ('beam','best_first'):raise ValueError('Unknown tree proposal policy')
        if type(tree_kernel) is not bool:raise ValueError('Expected a boolean tree kernel option')
        super().__init__(*args,**kwargs)
        self.tree_policy=tree_policy;self.engine.tree_gdn_kernel=tree_kernel
        self.engine.tree_chain_fastpath=True
        if dflash is None:
            directory=os.environ.get('SPARK_DFLASH_DIR')
            if not directory:raise ValueError('Tree speculation requires DFlash2')
            head=getattr(self.draft,'head',None) if os.environ.get('SPARK_DFLASH_COMPACT_HEAD','1')=='1' else None
            dflash=DFlashDraft(directory,self.engine,int(os.environ.get('SPARK_DFLASH_BLOCK','8')),head=head)
        self.dflash=dflash

    def snapshot(self):
        return dict(enabled=True,strategy='tree',policy=self.tree_policy,max_nodes=8,max_leaves=4,
            scheduling='cooperative_round_robin',batched_verification=False,mixed_prefill_decode=False)

    def start(self,*args,**kwargs):
        session=super().start(*args,**kwargs);session.tree_policy=self.tree_policy
        session.stats.update(tree_rounds=0,tree_nodes=0,tree_consumed_nodes=0)
        return session

    def prefill(self,session):
        super().prefill(session)
        if session.phase!='prefill':
            # Cached prefill snapshots own their paired MTP states. This
            # request never reenters MTP decode, so no stale fallback exists.
            session.draft_state=session.draft_multi=session.proposal=None

    def _decode_one(self,session,emission_limit=None):
        self._step(session,emission_limit)

    def force_next(self,session,token):
        error=self.decode_batch([session],forced_tokens=[token])[0]
        if error is not None:raise error

    def _step(self,session,emission_limit=None,forced=None):
        self._cancel(session);self._wait();begin=time.perf_counter()
        remaining=session.budget-len(session.generated)
        if emission_limit is not None:remaining=min(remaining,emission_limit)
        ctx=getattr(session,'dflash_ctx',None)
        if ctx is None or ctx.length!=len(session.state.tokens):raise RuntimeError('Tree draft history is misaligned')
        taps={}
        def observe(index,value):
            if isinstance(index,int) and index in LAYERS:taps[index]=self.dflash.tap(*value)
        known=session.generated[-1];tree=None
        if forced is not None:
            # Ordinary forward can mutate its input container. The forced
            # branch stays disposable until the cancellation check below.
            _,state=self.engine([known],clone_state(session.state),observer=observe,owned_graph_cache=True)
            nodes=(0,);emitted=(forced,);reason='forced'
        else:
            tree=(CandidateTree((known,),(-1,)) if remaining==1 else
                  self.dflash.propose_tree(ctx,known,node_budget=8,policy=self.tree_policy))
            tx=self.engine.verify_tree(tree,session.state,observe)
            try:
                selected=tx.select(remaining,session.eos)
                self._cancel(session)
                state=tx.commit(selected)
                nodes=selected.consumed_nodes;emitted=selected.emitted;reason=selected.reason
            finally:
                if tx.status=='open':tx.discard()
        indices=torch.tensor(nodes,device=self.engine.embedding.device,dtype=torch.long)
        features=torch.cat([taps[i].index_select(0,indices) for i in LAYERS],-1)
        self._cancel(session)
        self.dflash.push(ctx,features);session.accept(emitted,state)
        if ctx.length!=len(state.tokens):raise RuntimeError('Committed tree draft history is misaligned')
        session.stats['tree_rounds']+=1
        session.stats['tree_nodes']+=len(tree.tokens) if tree is not None else 1
        session.stats['tree_consumed_nodes']+=len(nodes)
        if forced is not None:session.stats['forced_control_tokens']+=1
        elif len(tree.tokens)==1:session.stats['single_steps']+=1
        else:
            edges=set(zip(tree.parents[1:],tree.tokens[1:]))
            accepted=sum((node,token) in edges for node,token in zip(nodes,emitted))
            corrections=int(reason=='miss');proposed=len(tree.tokens)-1
            session.stats['proposed']+=proposed;session.stats['accepted']+=accepted
            session.stats['corrections']+=corrections;session.stats['discarded']+=proposed-accepted-corrections
            session.stats['bonus_tokens']+=int(reason=='leaf')
        self._wait()
        session.iterations.append(dict(mode='reasoning_control' if forced is not None else 'tree',
            emitted=len(emitted),verification_rows=len(tree.tokens) if tree is not None else 1,
            tree_depth=max(map(len,tree.paths()))-1 if tree is not None else 0,
            tree_reason=reason,batch_size=1,seconds=time.perf_counter()-begin))
        self._cancel(session)

    @torch.no_grad()
    def decode_batch(self,sessions,*,emission_limits=None,forced_tokens=None):
        count=len(sessions)
        if (not 1<=count<=4 or len({id(s) for s in sessions})!=count or
                any(s.closed or s.phase!='decode' for s in sessions)):
            raise ValueError('Expected one to four distinct decode-ready tree sessions')
        limits=[None]*count if emission_limits is None else list(emission_limits)
        forced=[None]*count if forced_tokens is None else list(forced_tokens)
        if len(limits)!=count or len(forced)!=count:raise ValueError('Control lane counts differ')
        if any(v is not None and (type(v) is not int or v<1) for v in limits):raise ValueError('Expected positive emission limits')
        if any(t is not None and (type(t) is not int or t<0 or t in s.eos) for s,t in zip(sessions,forced)):
            raise ValueError('Expected non-EOS control tokens')
        errors=[None]*count
        for index,session in enumerate(sessions):
            try:self._step(session,limits[index],forced[index])
            except Exception as error:
                errors[index]=error
                if isinstance(error,RuntimeError) and any(s in str(error).lower() for s in ('cuda','nonfinite')):
                    return [e or error for e in errors]
        return errors
