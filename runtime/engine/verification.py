"""Explicit two-phase linear verification; never activated by import.

Large GDN recurrence banks are replaced by a short-lived verification tape.
QSA and PLE keep the existing immutable per-prefix states. CUDA Graph tapes
are borrowed only until commit/discard or the next model call; no borrowed
tensor is published as a committed GDN state.
"""
import numpy as np
import torch

from .decoder import DecoderState
from .decode_graph import DecodeGraphBlock
from .limits import VERIFY_WIDTHS
from .model import EngineState, ForwardHidden
from .ops.gdn_deferred import DeferredGDN


class DeferredGraphBlock(DecodeGraphBlock):
    def _run(self):
        value=self.pending
        self.tapes=[];self.intermediates=[]
        for layer,state in zip(self.layers,self.states):
            value,updated=layer(value[0],None,value[1],value[2],state=state,
                prefill=False,return_prefixes=True,defer_gdn=True)
            self.tapes.append(updated.attention)
            self.intermediates.append(value)
        return value

    def forward(self,pending,states):
        if self.graph is None:self._capture(pending,states)
        else:self._copy_inputs(pending,states)
        self.graph.replay();self.replays+=1
        updated=[]
        for source,captured in zip(states,self.tapes):
            tape=DeferredGDN(captured.output,captured.model,source.attention,
                captured.conv_prefixes,captured.convolved,captured.ba)
            updated.append(DecoderState(tape,prefixes=[DecoderState(tape) for _ in range(tape.rows)]))
        return self.output,updated

    def __call__(self,pending,states):
        return self.forward(pending,states)


class LinearVerification:
    def __init__(self,engine,ids,base):
        self.engine=engine;self.ids=tuple(ids);self.base=base
        self.base_tokens=tuple(base.tokens);self.layers=[]
        self.logits=self.hidden=None;self.status='open'

    @property
    def retained_gdn_bytes(self):
        return sum(layer.attention.retained_bytes for layer in self.layers
                   if isinstance(layer.attention,DeferredGDN))

    def _validate(self,count):
        if self.status!='open' or getattr(self.engine,'_pending_verification',None) is not self:
            raise RuntimeError('Verification is closed or superseded')
        if type(count) is not int or not 1<=count<=len(self.ids):
            raise ValueError('Expected a nonempty consumed input prefix')
        if tuple(self.base.tokens)!=self.base_tokens:
            raise RuntimeError('Verification base token history changed')
        for layer in self.layers:
            if isinstance(layer.attention,DeferredGDN):layer.attention.validate(count)

    def commit(self,count):
        self._validate(count)
        try:
            tapes=[layer.attention for layer in self.layers if isinstance(layer.attention,DeferredGDN)]
            if tapes:
                from .ops.gdn_commit import commit_many
                committed=iter(commit_many(tapes,count))
            else:committed=iter(())
            layers=[]
            for layer in self.layers:
                prefix=layer.prefixes[count-1]
                attention=(next(committed) if isinstance(layer.attention,DeferredGDN)
                           else prefix.attention)
                layers.append(DecoderState(attention,prefix.ple))
            result=EngineState(layers,list(self.base_tokens)+list(self.ids[:count]))
            self.status='committed'
            return result
        except BaseException:
            self.status='failed'
            raise
        finally:
            self._release()

    def discard(self):
        if self.status!='open':raise RuntimeError('Verification is already closed')
        self.status='discarded';self._release()

    def _release(self):
        for layer in self.layers:
            if isinstance(layer.attention,DeferredGDN) and layer.attention._status=='open':
                layer.attention.discard()
        if getattr(self.engine,'_pending_verification',None) is self:
            self.engine._pending_verification=None
        self.layers=[];self.base=None


@torch.no_grad()
def verify_linear(engine,input_ids,state,observer=None):
    ids=list(input_ids)
    if (len(ids) not in VERIFY_WIDTHS or not state.tokens or
            len(state.layers)!=len(engine.layers) or
            any(type(t) is not int or not 0<=t<engine.config['vocab_size'] for t in ids)):
        raise ValueError('Expected prefilled state and checkpoint verification tokens')
    start=len(state.tokens)
    if start+len(ids)>engine.config['max_position_embeddings']:
        raise ValueError('Verification exceeds context limit')
    engine.discard_verification()
    tx=LinearVerification(engine,ids,state);engine._pending_verification=tx
    try:
        device=engine.embedding.device
        token_ids=torch.tensor(ids,dtype=torch.long,device=device)
        positions=torch.arange(start,start+len(ids),device=device)
        preceding=engine.config['ngram_size']-1
        context=([engine.config['eos_token_id']]*preceding+state.tokens)[-preceding:] if preceding else []
        context=np.asarray([context],dtype=np.int64);ids_cpu=np.asarray(ids,dtype=np.int64)
        pending=(engine.embedding[token_ids].repeat(1,engine.config['hc_count']),None,None)
        index=0
        while index<len(engine.layers):
            if engine.graph_decode and engine.graph_verify and index in engine._graph_ranges:
                end=engine._graph_ranges[index]
                graphs=engine.__dict__.setdefault('_deferred_verify_graphs',{})
                pools=engine.__dict__.setdefault('_deferred_verify_pools',{})
                key=(index,len(ids))
                if key not in graphs:
                    if len(ids) not in pools:pools[len(ids)]=torch.cuda.graph_pool_handle()
                    graphs[key]=DeferredGraphBlock(engine.layers[index:end],pools[len(ids)],True)
                captured=graphs[key]
                pending,updated=captured(pending,state.layers[index:end])
                tx.layers.extend(updated)
                if observer is not None:
                    for offset,value in enumerate(captured.intermediates):observer(index+offset,value)
                index=end
            else:
                layer=engine.layers[index]
                pending,updated=layer(pending[0],positions,pending[1],pending[2],ids_cpu,context,
                    state.layers[index],prefill=False,position_start=start,return_prefixes=True,
                    defer_gdn=layer.layer_type=='linear_attention')
                tx.layers.append(updated)
                if observer is not None:observer(index,pending)
                index+=1
        multi,sampled,_=engine.final_mixer.combine_and_mix(*pending,True)
        if observer is not None:observer('final',(multi,sampled,None))
        tx.logits=engine.logits(sampled);tx.hidden=ForwardHidden(multi,sampled)
        return tx
    except BaseException:
        tx.discard()
        raise
