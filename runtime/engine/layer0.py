"""Independent first GDN decoder layer for this exact checkpoint.

The layer returns the pending HC residual tuple used by the next layer. This
is not yet a complete language model, tokenizer, scheduler or serving API.
"""
from torch import nn
from .ops.hyperconnection import GatedResidualRef
from .ops.gdn_attention import GDNRef
from .ops.moe import MoERef


class FirstLayerRef(nn.Module):
    def __init__(self, store, config,activation_backend='torch'):
        super().__init__()
        if config['layer_types'][0] != 'linear_attention':
            raise ValueError('Only the first GDN layer is currently supported')
        prefix = 'model.language_model.layers.0'
        eps, count = config['rms_norm_eps'], config['hc_count']
        self.attn_hc = GatedResidualRef.from_weight_store(store,
            prefix+'.attn_hyper_connection',count,eps)
        self.attention = GDNRef.from_weight_store(store,prefix+'.linear_attn',
            eps,config.get('output_gate_type','silu'),activation_backend)
        self.mlp_hc = GatedResidualRef.from_weight_store(store,
            prefix+'.mlp_hyper_connection',count,eps)
        self.moe = MoERef(store,prefix+'.mlp',config['num_experts'],
            config['num_experts_per_tok'],config.get('norm_topk_prob',True),activation_backend=activation_backend)

    def forward(self, hidden_states, state=None, prefill=False):
        residual, mixed, injection = self.attn_hc.mix(hidden_states)
        attention, state = self.attention(mixed,state,prefill)
        residual, mixed, injection = self.mlp_hc.combine_and_mix(residual,attention,injection)
        return (residual,self.moe(mixed),injection),state
