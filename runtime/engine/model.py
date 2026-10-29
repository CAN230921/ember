"""Single-sequence 48-layer greedy engine for the mixed GB10 checkpoint.

Runtime uses torch/FlashInfer plus this repository, never a vLLM model runner.
GDN prefill uses vendored FLA; ordinary decode uses specialized Triton kernels.
PLE remains on disk; compressed experts can reside, and GDN decode can use
CUDA Graphs. Two-token verification retains both cache commit points; the
learned MTP module/loop live in mtp.py and speculative.py. Independent two-row
requests live in batch.py. Text sparse-context smoke reaches 262016 inputs;
long MTP is checked through 32778 inputs. Multimodal and public serving remain
outside this engine's current acceptance scope.
"""
from dataclasses import dataclass,field
import numpy as np
import torch
from .limits import VERIFY_WIDTHS
from torch import nn
from .decoder import DecoderLayerRef
from .ops.hyperconnection import GatedResidualRef
from .ops.m0_refs import gptq_int8_dequant
from .ops.ple import hint_next_prefill_chunk


@dataclass
class EngineState:
    layers: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    prefixes: object = None


@dataclass
class ForwardHidden:
    # Main-model multi-stream values before the final HC mixer projection.
    # MTP consumes these values with shifted token IDs at unchanged positions.
    multi: torch.Tensor
    sampled: torch.Tensor


class GreedyEngineRef(nn.Module):
    def __init__(self,store,config,activation_backend='sm121-cutlass',expert_cache_size=4,hc_backend='triton',gdn_backend='kernels',expert_backend='torch',head_backend='torch',resident_experts=False,qsa_backend='torch',router_backend='torch',moe_finalize='fused',graph_decode=False,hc_projection_backend='torch',graph_verify=False,kv_backend='contiguous'):
        super().__init__()
        self.config=config; self.store=store
        if graph_decode and not (resident_experts and activation_backend=='sm121-native'
                and expert_backend=='flashinfer' and router_backend=='native'
                and moe_finalize=='deterministic' and gdn_backend=='kernels' and hc_backend=='triton'):
            raise ValueError('Graph decode requires the resident native deterministic kernel configuration')
        self.graph_decode=graph_decode
        if graph_verify and not graph_decode:raise ValueError('Verification graphs require graph_decode')
        self.graph_verify=graph_verify
        self._decode_graphs={}; self._graph_pool=None; self._graph_ranges={}
        self._verify_graphs={}; self._verify_graph_pools={}
        if head_backend not in ('torch','sm121'):raise ValueError('Unknown output head backend')
        self.head_backend=head_backend
        self.register_buffer('embedding',store.get('model.language_model.embed_tokens.weight'))
        self.layers=nn.ModuleList()
        for index in range(config['num_hidden_layers']):
            self.layers.append(DecoderLayerRef(store,config,index,activation_backend,hc_backend,gdn_backend,expert_backend,qsa_backend,router_backend,moe_finalize,hc_projection_backend,kv_backend))
            self.layers[-1].moe.cache_size=expert_cache_size
            if resident_experts:
                self.layers[-1].moe.preload_experts()
            print('ENGINE_LAYER_LOADED',index,flush=True)
        self.final_mixer=GatedResidualRef.from_weight_store(store,
            'model.language_model.hyper_connection_mixer',config['hc_count'],config['rms_norm_eps'],False,hc_backend,hc_projection_backend)
        self.head_name='lm_head'
        if self.head_name+'.qweight' not in store.weight_map:
            names=[name[:-8] for name in store.weight_map if name.endswith('lm_head.qweight')]
            if len(names)!=1:raise ValueError('Expected one GPTQ output head')
            self.head_name=names[0]
        for name in ('qweight','qzeros','scales'):
            self.register_buffer('head_'+name,store.get(self.head_name+'.'+name))
        # Capture contiguous GDN runs. QSA selection and disk PLE remain
        # explicit boundaries; their CPU work must finish before replay.
        index=0
        while index<len(self.layers):
            layer=self.layers[index]
            if layer.layer_type!='linear_attention' or layer.ple is not None:
                index+=1; continue
            end=index+1
            while end<len(self.layers) and self.layers[end].layer_type=='linear_attention' and self.layers[end].ple is None:
                end+=1
            self._graph_ranges[index]=end
            index=end

    def graph_stats(self):
        return dict(enabled=self.graph_decode,blocks=len(self._decode_graphs),
                    layers=sum(len(block.layers) for block in self._decode_graphs.values()),
                    replays={str(start):block.replays for start,block in self._decode_graphs.items()},
                    verify_enabled=self.graph_verify,verify_blocks=len(self._verify_graphs),
                    verify_layers=sum(len(block.layers) for block in self._verify_graphs.values()),
                    verify_replays={str(key):block.replays for key,block in self._verify_graphs.items()})

    def logits(self,hidden,chunk=4096):
        if self.head_backend=='sm121':
            from .ops.int8_head import packed_head
            return packed_head(hidden,self.head_qweight,self.head_scales).float()
        output=torch.empty(hidden.shape[0],self.config['vocab_size'],dtype=hidden.dtype,device=hidden.device)
        for start in range(0,output.shape[1],chunk):
            end=min(start+chunk,output.shape[1])
            weight=gptq_int8_dequant(self.head_qweight[:,start:end],self.head_qzeros[:,start//4:end//4],self.head_scales[:,start:end])
            output[:,start:end]=(hidden.float()@weight).to(hidden.dtype)
        return output.float()

    def forward(self,input_ids,state=None,observer=None,return_hidden=False,return_prefixes=False,prefill_chunk=False,borrowed_graph_cache=False,owned_graph_cache=False,lazy_prefixes=False):
        self.discard_verification()
        ids=[int(token) for token in input_ids]
        if not ids:raise ValueError('Expected at least one token')
        if any(token<0 or token>=self.config['vocab_size'] for token in ids):raise ValueError('Token is outside checkpoint vocabulary')
        state=state or EngineState([None]*len(self.layers),[])
        if len(state.layers)!=len(self.layers):raise ValueError('Invalid layer state count')
        if return_prefixes and (not state.tokens or len(ids) not in VERIFY_WIDTHS):
            raise ValueError('Prefix capture requires one to three decode tokens')
        if return_prefixes and prefill_chunk:
            raise ValueError('Prefill chunks cannot capture speculative prefixes')
        if borrowed_graph_cache and not (return_prefixes and self.graph_verify):
            raise ValueError('Borrowed verification caches require verification graphs')
        if borrowed_graph_cache and owned_graph_cache:
            raise ValueError('Graph caches cannot be both borrowed and owned')
        if state.tokens and len(ids)!=1 and not return_prefixes and not prefill_chunk:
            raise ValueError('Multi-token decode requires prefix capture')
        if return_prefixes:
            # The verification branch owns its layer list. Rejecting a draft
            # must not modify the caller's request or graph-backed caches.
            state=EngineState(list(state.layers),list(state.tokens))
        start=len(state.tokens)
        if start+len(ids)>self.config['max_position_embeddings']:raise ValueError('Context exceeds actual checkpoint override')
        device=self.embedding.device
        tokens=torch.tensor(ids,dtype=torch.long,device=device)
        positions=torch.arange(start,start+len(ids),device=device)
        context=[self.config['eos_token_id']]*(self.config['ngram_size']-1)
        preceding=self.config['ngram_size']-1
        context=(context+state.tokens[-preceding:])[-preceding:] if preceding else []
        context=np.asarray([context],dtype=np.int64)
        ids_cpu=np.asarray(ids,dtype=np.int64)
        hidden=self.embedding[tokens].repeat(1,self.config['hc_count'])
        block=injection=None
        index=0
        while index<len(self.layers):
            if start>0 and not prefill_chunk and self.graph_decode and (not return_prefixes or self.graph_verify) and index in self._graph_ranges:
                from .decode_graph import DecodeGraphBlock
                end=self._graph_ranges[index]
                if return_prefixes:
                    key=(index,len(ids))
                    from .graph_capture import shared_pool
                    if borrowed_graph_cache and shared_pool() is not None:raise ValueError('Borrowed graph caches cannot use the shared graph pool')
                    if len(ids) not in self._verify_graph_pools:self._verify_graph_pools[len(ids)]=shared_pool() or torch.cuda.graph_pool_handle()
                    if key not in self._verify_graphs:
                        self._verify_graphs[key]=DecodeGraphBlock(self.layers[index:end],self._verify_graph_pools[len(ids)],True)
                    captured=self._verify_graphs[key]
                else:
                    if index not in self._decode_graphs:
                        from .graph_capture import shared_pool
                        if self._graph_pool is None:self._graph_pool=shared_pool() or torch.cuda.graph_pool_handle()
                        self._decode_graphs[index]=DecodeGraphBlock(self.layers[index:end],self._graph_pool)
                    captured=self._decode_graphs[index]
                (hidden,block,injection),updated=captured((hidden,block,injection),state.layers[index:end],borrowed_graph_cache,owned_graph_cache,lazy_prefixes)
                state.layers[index:end]=updated
                if observer is not None:
                    for offset,value in enumerate(captured.intermediates):observer(index+offset,value)
                index=end
            else:
                layer=self.layers[index]
                (hidden,block,injection),state.layers[index]=layer(hidden,positions,block,injection,
                    ids_cpu,context,state.layers[index],prefill=start==0 or prefill_chunk,position_start=start,
                    return_prefixes=return_prefixes)
                if observer is not None:observer(index,(hidden,block,injection))
                index+=1
        multi,sampled,_=self.final_mixer.combine_and_mix(hidden,block,injection,return_prefixes)
        if observer is not None:observer('final',(multi,sampled,None))
        state.tokens.extend(ids)
        if return_prefixes:
            state.prefixes=[EngineState([layer.prefixes[row] for layer in state.layers],state.tokens[:start+row+1])
                            for row in range(len(ids))]
        result=(self.logits(sampled if return_prefixes else sampled[-1:]),state)
        return (*result,ForwardHidden(multi,sampled)) if return_hidden else result

    def prefill(self,input_ids,chunk_size=2048,return_hidden=False,progress=None):
        """Bound activations while preserving prefill recurrence across chunks.

        GDN uses its chunk-prefill arithmetic for every continuation. PLE's
        preceding n-gram context and QSA logical positions continue normally.
        Decode graphs are never used to process a prefill chunk.
        """
        if not isinstance(chunk_size,int) or chunk_size<=0 or chunk_size%64:
            raise ValueError('Prefill chunk size must be a positive multiple of 64')
        ids=list(input_ids)
        if not ids:raise ValueError('Expected a nonempty prompt')
        state=None; multi=[]; sampled=[]
        for start in range(0,len(ids),chunk_size):
            hint_next_prefill_chunk(self,ids,start,min(start+chunk_size,len(ids)),chunk_size)
            result=self(ids[start:start+chunk_size],state,return_hidden=return_hidden,prefill_chunk=True)
            logits,state=result[:2]
            if return_hidden:
                multi.append(result[2].multi); sampled.append(result[2].sampled)
            if progress is not None:progress(start,min(start+chunk_size,len(ids)),len(ids))
        result=(logits,state)
        return (*result,ForwardHidden(torch.cat(multi),torch.cat(sampled))) if return_hidden else result

    def verify_tokens(self,input_ids,state,observer=None,borrowed_graph_cache=False,lazy_prefixes=False):
        """Verify one to three sequential tokens and retain each commit point."""
        # lazy_prefixes: graph prefix banks come back as replay-buffer views (GDNState.graph_view); the caller
        # must materialize the committed prefix (batch_mtp_model.materialize_views) before any graph replays again.
        return self(input_ids,state,observer,return_hidden=True,return_prefixes=True,borrowed_graph_cache=borrowed_graph_cache,
            lazy_prefixes=lazy_prefixes)

    def discard_verification(self):
        pending=getattr(self,'_pending_verification',None)
        if pending is not None:
            pending.discard()

    def verify_deferred(self,input_ids,state,observer=None):
        """Explicit experimental API; commit/discard before another model call.

        A subsequent model call invalidates any uncommitted transaction. Only
        consumed input prefixes are committed; the last predicted token stays
        unconsumed, as in verify_tokens. Existing serving paths are unchanged.
        """
        from .verification import verify_linear
        return verify_linear(self,input_ids,state,observer)

    def verify_tree_reference(self,tree,state):
        """Experimental correctness oracle: expands paths, recomputes ancestors."""
        from .tree_verification import verify_tree_reference
        return verify_tree_reference(self,tree,state)

    def verify_tree(self,tree,state,observer=None):
        """Opt-in eager unique-node tree prototype; existing serving is unchanged."""
        from .tree_model import verify_tree
        return verify_tree(self,tree,state,observer)

    @torch.no_grad()
    def generate(self,prompt_ids,max_tokens=8):
        state=None; ids=prompt_ids; generated=[]
        for step in range(max_tokens):
            logits,state=self(ids,state)
            token=int(logits[0].argmax())
            generated.append(token); ids=[token]
        return generated
