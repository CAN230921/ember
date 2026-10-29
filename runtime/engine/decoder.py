"""Independent decoder layer with pending HC residual and optional PLE.

Tensor math does not import vLLM. CPU token IDs/context stay on the engine
side for PLE; a single sequence owns separate attention and PLE histories.
"""
from dataclasses import dataclass
from torch import nn
from .ops.hyperconnection import GatedResidualRef,hc_combine
from .ops.gdn_attention import GDNRef
from .ops.moe import MoERef
from .ops.ple import PLERef


@dataclass
class DecoderState:
    attention: object = None
    ple: object = None
    prefixes: object = None


class DecoderLayerRef(nn.Module):
    def __init__(self,store,config,layer_index,activation_backend='torch',hc_backend='torch',gdn_backend='torch',expert_backend='torch',qsa_backend='torch',router_backend='torch',moe_finalize='fused',hc_projection_backend='torch',kv_backend='contiguous'):
        super().__init__()
        self.index=layer_index
        self.layer_type=config['layer_types'][layer_index]
        self.count=config['hc_count']
        prefix=f'model.language_model.layers.{layer_index}'
        eps=config['rms_norm_eps']
        self.attn_hc=GatedResidualRef.from_weight_store(store,prefix+'.attn_hyper_connection',self.count,eps,backend=hc_backend,projection_backend=hc_projection_backend)
        self.mlp_hc=GatedResidualRef.from_weight_store(store,prefix+'.mlp_hyper_connection',self.count,eps,backend=hc_backend,projection_backend=hc_projection_backend)
        if self.layer_type=='linear_attention':
            self.attention=GDNRef.from_weight_store(store,prefix+'.linear_attn',eps,
                config.get('output_gate_type','silu'),activation_backend,gdn_backend)
        elif self.layer_type=='full_attention':
            from .ops.qsa_attention import QSARef
            self.attention=QSARef(store,prefix+'.self_attn',config,activation_backend,qsa_backend,kv_backend)
        else:raise ValueError('Unsupported layer type')
        self.ple=PLERef(store,prefix+'.ple',config) if layer_index+1 in config['ple_layer_ids'] else None
        self.moe=MoERef(store,prefix+'.mlp',config['num_experts'],config['num_experts_per_tok'],
            config.get('norm_topk_prob',True),activation_backend=activation_backend,expert_backend=expert_backend,router_backend=router_backend,moe_finalize=moe_finalize)

    def forward(self,hidden_states,positions,prev_block_output=None,prev_injection=None,
                input_ids_cpu=None,context_cpu=None,state=None,prefill=False,position_start=None,return_prefixes=False,
                defer_gdn=False):
        if defer_gdn and (prefill or not return_prefixes):
            raise ValueError('Deferred GDN requires decode verification')
        state=state or DecoderState()
        if (prev_block_output is None)!=(prev_injection is None):
            raise ValueError('Pending block output/injection must arrive together')
        ple_state=state.ple
        ple_prefixes=None
        if self.ple is not None:
            if prev_block_output is not None:
                hidden_states=self.attn_hc.combine(hidden_states,prev_block_output,prev_injection)
                prev_block_output=prev_injection=None
            if input_ids_cpu is None or context_cpu is None:
                raise ValueError('PLE needs CPU token IDs and preceding n-gram context')
            ple_result=self.ple(hidden_states,input_ids_cpu,[0,hidden_states.shape[0]],context_cpu,ple_state,return_prefixes)
            enhancement,ple_state=ple_result[:2]
            if return_prefixes:ple_prefixes=ple_result[2]
            hidden_states=hidden_states+enhancement
        if prev_block_output is None:
            residual,mixed,injection=self.attn_hc.mix(hidden_states,return_prefixes)
        else:
            residual,mixed,injection=self.attn_hc.combine_and_mix(hidden_states,prev_block_output,prev_injection,return_prefixes)
        if self.layer_type=='linear_attention':
            if defer_gdn:
                from .ops.gdn_deferred import verify_gdn
                qkvz,ba=self.attention.project(mixed,preserve_rows=True)
                tape=verify_gdn(self.attention,qkvz,ba,state.attention)
                attention_result=(self.attention.out_linear(tape.output),tape,[tape]*len(mixed))
            else:
                attention_result=self.attention(mixed,state.attention,prefill,return_prefixes)
        else:
            attention_result=self.attention(mixed,positions,state.attention,position_start=position_start,return_prefixes=return_prefixes,prefill_chunk=prefill)
        attention,attention_state=attention_result[:2]
        residual,mixed,injection=self.mlp_hc.combine_and_mix(residual,attention,injection,return_prefixes)
        updated=DecoderState(attention_state,ple_state)
        if return_prefixes:
            updated.prefixes=[DecoderState(att,ple_prefixes[row] if ple_prefixes is not None else state.ple)
                              for row,att in enumerate(attention_result[2])]
        moe=self.moe(mixed,preserve_rows=True) if return_prefixes else self.moe(mixed)
        return (residual,moe,injection),updated
