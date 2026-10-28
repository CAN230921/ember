"""Single-sequence Qwen GDN attention bring-up with real mixed weights.

Prefill preserves the incumbent chunk algorithm's BF16 intermediate storage.
Decode supports CUDA fused-norm math and Triton rounded-beta math.
This torch runtime is independent of vLLM and deliberately unoptimized.
"""
import torch
from ..limits import VERIFY_WIDTHS, per_row
from torch import nn
from torch.nn import functional as F
from .gdn import GDNState, causal_conv_step, post_conv_prep, recurrent_step
from .fp8 import linear as fp8_linear
from .gdn_prefill import chunk_prefill


class GDNRef(nn.Module):
    def __init__(self, weights, eps=1e-6, gate_activation='silu',activation_backend='torch',core_backend='torch'):
        super().__init__()
        for name,tensor in weights.items():
            self.register_buffer(name,tensor)
        self.eps,self.gate_activation=eps,gate_activation
        self.activation_backend=activation_backend
        if core_backend not in ('torch','kernels'):raise ValueError('Unknown GDN core backend')
        self.core_backend=core_backend
        self.key_heads,self.value_heads,self.key_dim,self.value_dim=16,48,128,128
        if self.conv.shape != (10240,4) or self.qkvz.shape != (16384,2560):
            raise ValueError('Only the verified GB10 checkpoint GDN shapes are supported')

    @classmethod
    def from_weight_store(cls,store,prefix,eps=1e-6,gate_activation='silu',activation_backend='torch',core_backend='torch'):
        def get(suffix): return store.get(prefix+'.'+suffix)
        weights={
            'qkvz':torch.cat((get('in_proj_qkv.weight'),get('in_proj_z.weight'))),
            'qkvz_scale':torch.cat((get('in_proj_qkv.weight_scale_inv'),get('in_proj_z.weight_scale_inv'))),
            'ba':torch.cat((get('in_proj_b.weight'),get('in_proj_a.weight'))),
            'conv':get('conv1d.weight').reshape(10240,4),
            'A_log':get('A_log'),'dt_bias':get('dt_bias'),'norm':get('norm.weight'),
            'out':get('out_proj.weight'),'out_scale':get('out_proj.weight_scale_inv'),
        }
        return cls(weights,eps,gate_activation,activation_backend,core_backend)

    def project(self,x,preserve_rows=False):
        # M=2 selects a different cuBLAS BA projection and changes a near-zero
        # BF16 value on real inputs. Keep the tiny gate projection's ordinary
        # per-token arithmetic while sharing the large FP8 weight traversal.
        from .row_gemv import per_row_linear
        ba=per_row_linear(x,self.ba) if per_row(x.shape[0]) and preserve_rows else F.linear(x,self.ba)
        return self.qkvz_linear(x),ba

    def qkvz_linear(self,x):
        from . import nvfp4_linear
        lin=getattr(self,'nvfp4',{}).get('qkvz')
        if lin is not None and 'qkvz' in nvfp4_linear.ACTIVE:return lin(x)
        return fp8_linear(x,self.qkvz,self.qkvz_scale,self.activation_backend)

    def out_linear(self,x):
        from . import nvfp4_linear
        lin=getattr(self,'nvfp4',{}).get('out')
        if lin is not None and 'out' in nvfp4_linear.ACTIVE:return lin(x)
        return fp8_linear(x,self.out,self.out_scale,self.activation_backend)

    def run_core(self,qkvz,ba,state=None,prefill=False,fused_cuda=True,return_prefixes=False):
        if self.core_backend=='kernels':
            from .gdn_triton import run_core
            return run_core(self,qkvz,ba,state,prefill,return_prefixes)
        # Prefill is one packed sequence; decode is one token of that sequence.
        tokens=qkvz.shape[0]
        if not prefill and tokens not in VERIFY_WIDTHS:
            raise ValueError('This attention slice supports one to three decode tokens')
        if return_prefixes and prefill:
            raise ValueError('Prefix capture requires decode')
        if state is None:
            state=GDNState.zeros(1,device=qkvz.device,dtype=qkvz.dtype,state_dtype=torch.bfloat16)
        if prefill:
            # Preserve the incumbent chunk's BF16 WY and score boundaries.
            convolved=[]
            hist=state.conv
            for token in range(tokens):
                conv,hist=causal_conv_step(qkvz[token:token+1,:10240],hist,self.conv)
                convolved.append(conv)
            b,a=ba.split((48,48),dim=-1)
            q,k,v,g,beta=post_conv_prep(torch.cat(convolved),a,b,self.A_log,self.dt_bias,prefill=True)
            core,updated=chunk_prefill(q,k,v,g,beta,state.recurrent)
            gate=qkvz[:,10240:].float().reshape(tokens,48,128)
            if self.gate_activation in ('silu','swish'):gate=F.silu(gate)
            elif self.gate_activation=='sigmoid':gate=torch.sigmoid(gate)
            else:raise ValueError('Unsupported GDN output gate')
            core=core.float()
            norm=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+self.eps)*self.norm.float()
            return (norm*gate).to(qkvz.dtype).reshape(tokens,6144),GDNState(hist,updated)
        normalized=[]; prefixes=[]
        for token in range(tokens):
            qkv,z=qkvz[token:token+1].split((10240,6144),dim=-1)
            b,a=ba[token:token+1].split((48,48),dim=-1)
            conv,hist=causal_conv_step(qkv,state.conv,self.conv)
            q,k,v,g,beta=post_conv_prep(conv,a,b,self.A_log,self.dt_bias,
                prefill=prefill,decode_beta_round=not fused_cuda)
            core,new_state=recurrent_step(q,k,v,g,beta,state.recurrent,
                output_dtype=torch.float32 if fused_cuda and not prefill else None)
            gate=z.float().reshape(1,48,128)
            if self.gate_activation in ('silu','swish'):
                gate=F.silu(gate)
            elif self.gate_activation=='sigmoid':
                gate=torch.sigmoid(gate)
            else:
                raise ValueError('Unsupported GDN output gate')
            core=core.float()
            norm=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+self.eps)*self.norm.float()
            normalized.append((norm*gate).to(qkvz.dtype).reshape(1,6144))
            state=GDNState(hist,new_state)
            if return_prefixes:prefixes.append(state)
        result=(torch.cat(normalized),state)
        return (*result,prefixes) if return_prefixes else result

    def forward(self,x,state=None,prefill=False,return_prefixes=False):
        projected,ba=self.project(x,preserve_rows=not prefill)
        core=self.run_core(projected,ba,state,prefill,return_prefixes=return_prefixes)
        result=(self.out_linear(core[0]),core[1])
        return (*result,core[2]) if return_prefixes else result
