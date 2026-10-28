"""Dedicated BF16 MTP expert bank, shared expert and native router.

The checkpoint's fused bank stores gate then up; FlashInfer stores up then
gate. No target-bank quantization or activation calibration applies here.
"""
import os

import torch
from torch import nn
from torch.nn import functional as F


DRAFT_FP8 = os.environ.get('SPARK_DRAFT_FP8', '0') == '1'
# Draft-only NVFP4 bank (half of FP8: ~1.17 GB less); takes precedence over SPARK_DRAFT_FP8.
DRAFT_FP4 = os.environ.get('SPARK_DRAFT_FP4', '0') == '1'


class MTPMoE(nn.Module):
    def __init__(self,store,prefix,config,backend='flashinfer',top_k=None):
        super().__init__()
        if backend not in ('flashinfer','torch'):raise ValueError('Unknown BF16 MTP MoE backend')
        self.backend=backend
        self.top_k=config['num_experts_per_tok'] if top_k is None else top_k
        self.renormalize=config.get('norm_topk_prob',True)
        self.num_experts=config['num_experts']
        self.intermediate=config['moe_intermediate_size']
        if not 1<=self.top_k<=self.num_experts:raise ValueError('Invalid draft expert top-k')
        get=lambda suffix:store.get(prefix+'.'+suffix)
        self.register_buffer('router',get('gate.weight'))
        self.register_buffer('shared_gate',get('shared_expert_gate.weight'))
        self.register_buffer('shared_up',torch.cat([get('shared_expert.'+p+'.weight') for p in ('gate_proj','up_proj')]))
        self.register_buffer('shared_down',get('shared_expert.down_proj.weight'))
        bank=get('experts.gate_up_proj')
        if bank.shape!=(self.num_experts,2*self.intermediate,config['hidden_size']) or bank.dtype!=torch.bfloat16:
            raise ValueError('Unexpected checkpoint MTP gate/up bank')
        gate,up=bank.chunk(2,1)
        self.register_buffer('w13',torch.cat((up,gate),1))
        self.register_buffer('w2',get('experts.down_proj'))
        if self.w2.shape!=(self.num_experts,config['hidden_size'],self.intermediate) or self.w2.dtype!=torch.bfloat16:
            raise ValueError('Unexpected checkpoint MTP down bank')
        self.fp4=DRAFT_FP4 and self.w13.is_cuda
        self.fp8=DRAFT_FP8 and self.w13.is_cuda and not self.fp4
        if self.fp4:
            from .mtp_moe_fp4 import quantize_bank as quantize_fp4
            q13,s13,g13=quantize_fp4(self.w13); q2,s2,g2=quantize_fp4(self.w2)
            self.w13=torch.empty(0,device=q13.device,dtype=torch.bfloat16); self.w2=torch.empty(0,device=q13.device,dtype=torch.bfloat16)
            for name,value in (('w13_fp4',q13),('s13',s13),('g13',g13),('w2_fp4',q2),('s2',s2),('g2',g2)):
                self.register_buffer(name,value)
            torch.cuda.empty_cache()
        if self.fp8:
            # Draft-only: FP8 block-128 bank on tensor cores; the BF16 bank is released.
            from .mtp_moe_fp8 import quantize_bank
            q13,s13=quantize_bank(self.w13); q2,s2=quantize_bank(self.w2)
            self.w13=torch.empty(0,device=q13.device,dtype=torch.bfloat16); self.w2=torch.empty(0,device=q13.device,dtype=torch.bfloat16)
            self.register_buffer('w13_fp8',q13); self.register_buffer('s13',s13)
            self.register_buffer('w2_fp8',q2); self.register_buffer('s2',s2)
            torch.cuda.empty_cache()

    def route(self,x):
        from .native_kernels import topk_softmax
        logits=F.linear(x,self.router)
        weights,ids=topk_softmax(logits,self.top_k,self.renormalize)
        return weights,ids,logits

    def routed(self,x,weights,ids,backend=None):
        if getattr(self,'fp4',False):
            from .mtp_moe_fp4 import routed as fp4_routed
            return fp4_routed(x,weights,ids,self.w13_fp4,self.s13,self.g13,self.w2_fp4,self.s2,self.g2)
        if getattr(self,'fp8',False):
            from .mtp_moe_fp8 import routed as fp8_routed
            return fp8_routed(x,weights,ids,self.w13_fp8,self.s13,self.w2_fp8,self.s2)
        backend=backend or self.backend
        if backend=='flashinfer':
            from flashinfer.autotuner import autotune
            from flashinfer.fused_moe import cutlass_fused_moe
            output=torch.empty_like(x)
            with autotune(False,skip_ops={'trtllm::fused_moe::gemm1','trtllm::fused_moe::gemm2'}):
                cutlass_fused_moe(input=x.contiguous(),token_selected_experts=ids,
                    token_final_scales=weights,fc1_expert_weights=self.w13,fc2_expert_weights=self.w2,
                    output_dtype=x.dtype,output=output,quant_scales=[],use_fused_finalize=False)
            return output
        output=torch.zeros_like(x,dtype=torch.float32)
        # This independent mathematical oracle shares the stored bank only.
        # It uses ordinary BF16 projections and FP32 deterministic accumulation.
        for expert in ids.unique().tolist():
            rows,slots=torch.where(ids==expert)
            up,gate=F.linear(x[rows],self.w13[expert]).chunk(2,-1)
            activated=(F.silu(gate.float())*up.float()).to(x.dtype)
            projected=F.linear(activated,self.w2[expert])
            output.index_add_(0,rows,projected.float()*weights[rows,slots,None])
        return output.to(x.dtype)

    def shared(self,x):
        gate,up=F.linear(x,self.shared_up).chunk(2,-1)
        activation=F.silu(gate.float()).to(x.dtype)*up
        return F.linear(activation,self.shared_down)*torch.sigmoid(F.linear(x,self.shared_gate))

    def forward(self,x,return_details=False):
        weights,ids,logits=self.route(x)
        routed=self.routed(x,weights,ids); shared=self.shared(x)
        output=routed+shared
        return (output,dict(weights=weights,ids=ids,router_logits=logits,routed=routed,shared=shared)) if return_details else output
