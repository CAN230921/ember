"""Qwen3.8 Flash Next's learned one-layer MTP, independent of vLLM.

Inputs are the target's pre-final-mixer HC streams and token IDs shifted one
position left. Positions remain target-aligned, as in the pinned proposer.
Embedding is shared; an optional draft-only packed vocabulary head may be supplied.
"""
from dataclasses import dataclass
import os
import torch
from torch import nn
from torch.nn import functional as F

from .ops.hyperconnection import GatedResidualRef
from .ops.mtp_moe import MTPMoE
from .ops.qsa_attention import QSARef,gemma_norm
from .graph_capture import graph as capture_graph

# Opt-in: decode-step drafts replay their dense parts as CUDA graphs (see _DraftGraph).
DRAFT_GRAPH=os.environ.get('SPARK_DRAFT_GRAPH','0')=='1'
DRAFT_GRAPH_ROWS=5
DRAFT_GRAPH_WARMUP=2


@dataclass
class MTPOutput:
    logits: torch.Tensor
    state: object
    multi: torch.Tensor
    sampled: torch.Tensor
    vocab_ids: object = None


class MTPDraft(nn.Module):
    def __init__(self,store,config,embedding,head,expert_backend='flashinfer',top_k=None,kv_backend='contiguous'):
        super().__init__()
        self.config=config; self.head=head
        self.register_buffer('embedding',embedding,persistent=False)
        for name in ('fc_embedding','fc_hidden','pre_fc_norm_embedding','pre_fc_norm_hidden'):
            self.register_buffer(name,store.get('mtp.'+name+'.weight'))
        count=config['hc_count']; eps=config['rms_norm_eps']
        base='mtp.layers.0'
        self.attn_hc=GatedResidualRef.from_weight_store(store,base+'.attn_hyper_connection',count,eps,backend='triton')
        self.mlp_hc=GatedResidualRef.from_weight_store(store,base+'.mlp_hyper_connection',count,eps,backend='triton')
        self.attention=QSARef(store,base+'.self_attn',config,'sm121-native','triton',kv_backend)
        self.moe=MTPMoE(store,base+'.mlp',config,expert_backend,top_k)
        self.final_mixer=GatedResidualRef.from_weight_store(store,'mtp.hyper_connection_mixer',count,eps,False,'triton')
        self.schema_names=sorted(name for name in store.weight_map if name.startswith('mtp.'))
        if len(self.schema_names)!=31:raise ValueError('Expected the verified 31-tensor MTP checkpoint')

    def fuse_inputs(self,input_ids,multi_hidden):
        return self.fuse_ids(torch.as_tensor(input_ids,device=self.embedding.device,dtype=torch.long),multi_hidden)

    def fuse_ids(self,ids,multi_hidden):
        rows=ids.numel(); hidden=self.config['hidden_size']; count=self.config['hc_count']
        if ids.ndim!=1 or multi_hidden.shape!=(rows,count*hidden) or multi_hidden.dtype!=torch.bfloat16:
            raise ValueError('MTP requires shifted token IDs and target HC rows of matching length')
        embedding=gemma_norm(self.embedding[ids],self.pre_fc_norm_embedding,self.config['rms_norm_eps'])
        embedding=F.linear(embedding,self.fc_embedding)
        normalized=gemma_norm(multi_hidden,self.pre_fc_norm_hidden,self.config['rms_norm_eps'])
        projected=F.linear(normalized.reshape(rows*count,hidden),self.fc_hidden).reshape(rows,count,hidden)
        return (projected+embedding[:,None]).flatten(1)

    def forward(self,input_ids,multi_hidden,state=None,observer=None,prefill_chunk=False,head=None):
        start=0 if state is None else state.length
        if start+len(input_ids)>self.config['max_position_embeddings']:
            raise ValueError('MTP context exceeds actual checkpoint override')
        if (DRAFT_GRAPH and state is not None and not prefill_chunk and observer is None
                and 1<=len(input_ids)<=DRAFT_GRAPH_ROWS and multi_hidden.is_cuda):
            head=head or self.head
            graphs=self.__dict__.setdefault('_draft_graphs',{})
            key=(len(input_ids),id(head))
            graph=graphs.get(key)
            if graph is None:graph=graphs[key]=_DraftGraph(self,len(input_ids),head)
            return graph(input_ids,multi_hidden,state,start)
        preserve_rows=state is not None and not prefill_chunk
        hidden=self.fuse_inputs(input_ids,multi_hidden)
        if observer is not None:observer('fused_inputs',hidden)
        positions=torch.arange(start,start+hidden.shape[0],device=hidden.device)
        residual,mixed,injection=self.attn_hc.mix(hidden,preserve_rows)
        attention,state=self.attention(mixed,positions,state,position_start=start,prefill_chunk=prefill_chunk)
        if observer is not None:observer('attention',attention)
        residual,mixed,injection=self.mlp_hc.combine_and_mix(residual,attention,injection,preserve_rows)
        if observer is not None:observer('moe_input',mixed)
        block=self.moe(mixed)
        multi,sampled,_=self.final_mixer.combine_and_mix(residual,block,injection,preserve_rows)
        # An optional draft-only head (e.g. a smaller vocabulary for deep recursive steps).
        head=head or self.head
        logits=head(sampled[-1:].contiguous())
        return MTPOutput(logits,state,multi,sampled,getattr(head,'vocab_ids',None))


class _DraftGraph:
    """One decode-step MTP draft shape (rows, head) as two CUDA graphs around the eager QSA attention.

    pre  = fuse_ids + attention HC mix; post = MLP HC combine/mix, MoE, final mixer and head. These
    are the very modules and kernels of MTPDraft.forward on static buffers (preserve_rows=True, as in
    every stateful decode call); attention appends KV, so it runs eagerly between the replays. The
    first DRAFT_GRAPH_WARMUP calls run eagerly (Triton/cuBLAS warm-up), then both graphs are captured.
    Returned tensors are clones: later replays overwrite the static outputs.
    """
    def __init__(self,draft,rows,head):
        self.draft,self.rows,self.head=draft,rows,head
        self.calls=0; self.replays=0; self.pre=self.post=None

    def _pre(self):
        return self.draft.attn_hc.mix(self.draft.fuse_ids(self.ids,self.multi_in),True)

    def _post(self):
        d=self.draft; residual,_,injection=self.pre_out
        residual,mixed,injection=d.mlp_hc.combine_and_mix(residual,self.att_in,injection,True)
        block=d.moe(mixed)
        multi,sampled,_=d.final_mixer.combine_and_mix(residual,block,injection,True)
        return multi,sampled,self.head(sampled[-1:].contiguous())

    def _eager(self,input_ids,multi_hidden,state,start):
        d=self.draft
        hidden=d.fuse_inputs(input_ids,multi_hidden)
        positions=torch.arange(start,start+hidden.shape[0],device=hidden.device)
        residual,mixed,injection=d.attn_hc.mix(hidden,True)
        attention,state=d.attention(mixed,positions,state,position_start=start,prefill_chunk=False)
        residual,mixed,injection=d.mlp_hc.combine_and_mix(residual,attention,injection,True)
        block=d.moe(mixed)
        multi,sampled,_=d.final_mixer.combine_and_mix(residual,block,injection,True)
        logits=self.head(sampled[-1:].contiguous())
        return MTPOutput(logits,state,multi,sampled,getattr(self.head,'vocab_ids',None))

    def __call__(self,input_ids,multi_hidden,state,start):
        self.calls+=1
        if self.pre is None and self.calls<=DRAFT_GRAPH_WARMUP:
            return self._eager(input_ids,multi_hidden,state,start)
        d=self.draft
        if self.pre is None:
            self.ids=torch.zeros(self.rows,dtype=torch.long,device=multi_hidden.device)
            self.multi_in=torch.empty_like(multi_hidden)
            self.pool=torch.cuda.graph_pool_handle()
        self.ids.copy_(torch.as_tensor(input_ids,dtype=torch.long),non_blocking=False)
        self.multi_in.copy_(multi_hidden)
        if self.pre is None:
            torch.cuda.synchronize()
            self.pre=torch.cuda.CUDAGraph()
            with capture_graph(self.pre,pool=self.pool):self.pre_out=self._pre()
        self.pre.replay()
        positions=torch.arange(start,start+self.rows,device=multi_hidden.device)
        attention,state=d.attention(self.pre_out[1],positions,state,position_start=start,prefill_chunk=False)
        if self.post is None:
            self.att_in=torch.empty_like(attention); self.att_in.copy_(attention)
            torch.cuda.synchronize()
            self.post=torch.cuda.CUDAGraph()
            with capture_graph(self.post,pool=self.pool):self.post_out=self._post()
        else:
            self.att_in.copy_(attention)
        self.post.replay(); self.replays+=1
        multi,sampled,logits=(t.clone() for t in self.post_out)
        return MTPOutput(logits,state,multi,sampled,getattr(self.head,'vocab_ids',None))
