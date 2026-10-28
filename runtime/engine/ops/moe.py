"""Single-GPU correctness bring-up for this checkpoint's mixed-precision MoE.

No vLLM runtime imports. The torch expert path dequantizes NVFP4 weights;
the flashinfer path keeps selected experts packed and calls a standalone
FP4 fused kernel. Both paths retain bounded expert caches.
"""
from collections import OrderedDict
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from .fp8 import linear as fp8_linear
from .m0_refs import nvfp4_dequant
from ..limits import per_row
import os
SHARED_GATE_BATCH = os.environ.get('SPARK_SHARED_GATE_BATCH', '0') == '1'


def quantize_dequantize_nvfp4(x, input_scale):
    from flashinfer import fp4_quantize
    if x.ndim != 2 or x.shape[1] % 16 or x.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 rows divisible by 16")
    # The FI MoE path uses inverse global input scales. Keeping scales in
    # linear order permits the same mathematical dequantizer as weights.
    packed, scales = fp4_quantize(x.contiguous(), input_scale.reciprocal().reshape(1),
        sf_vec_size=16, is_sf_swizzled_layout=False)
    scales = scales.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
    return nvfp4_dequant(packed.view(torch.uint8), scales, input_scale)


class MoERef(nn.Module):
    def __init__(self, store, prefix, num_experts=512, top_k=5,
                 renormalize=True, cache_size=16,activation_backend='torch',expert_backend='torch',router_backend='torch',moe_finalize='fused'):
        super().__init__()
        self.store, self.prefix = store, prefix
        self.num_experts = num_experts
        self.top_k, self.renormalize, self.cache_size = top_k, renormalize, cache_size
        self.activation_backend=activation_backend
        if router_backend not in ('torch','native'):raise ValueError('Unknown router backend')
        self.router_backend=router_backend
        if moe_finalize not in ('fused','deterministic'):raise ValueError('Unknown expert finalize mode')
        self.moe_finalize=moe_finalize
        if expert_backend not in ('torch','flashinfer'):raise ValueError('Unknown MoE expert backend')
        self.expert_backend=expert_backend
        self._packed_options={}
        if expert_backend=='flashinfer' and moe_finalize=='deterministic':
            import inspect
            from flashinfer.fused_moe import cutlass_fused_moe
            if 'use_fused_finalize' not in inspect.signature(cutlass_fused_moe).parameters:
                raise RuntimeError('Installed FlashInfer lacks deterministic finalize; '
                                   'do not silently substitute atomic finalize')
            self._packed_options['use_fused_finalize']=False
        self.register_buffer('router', store.get(prefix+'.gate.weight'))
        self.register_buffer('shared_gate', store.get(prefix+'.shared_expert_gate.weight'))
        shared = prefix+'.shared_expert.'
        self.register_buffer('shared_up', torch.cat([store.get(shared+p+'.weight')
            for p in ('gate_proj','up_proj')]))
        self.register_buffer('shared_up_scale', torch.cat([store.get(shared+p+'.weight_scale_inv')
            for p in ('gate_proj','up_proj')]))
        self.register_buffer('shared_down', store.get(shared+'down_proj.weight'))
        self.register_buffer('shared_down_scale', store.get(shared+'down_proj.weight_scale_inv'))
        # Read scalar calibration values on CPU; no full expert bank is loaded.
        from engine.weights import WeightStore
        cpu_store = WeightStore(store.model_dir, 'cpu')
        a1, a2 = [], []
        for expert in range(num_experts):
            base = prefix+f'.experts.{expert}.'
            a1.extend(cpu_store.get(base+p+'.input_scale').item() for p in ('gate_proj','up_proj'))
            a2.append(cpu_store.get(base+'down_proj.input_scale').item())
        self.register_buffer('a1_scale', torch.tensor(max(a1), device=self.router.device))
        self.register_buffer('a2_scale', torch.tensor(max(a2), device=self.router.device))
        # Static calibration: match the real call's per-expert vectors and
        # avoid two reciprocal kernels on every resident-bank forward.
        self.register_buffer('a1_inverse', self.a1_scale.reciprocal(),persistent=False)
        self.register_buffer('a2_inverse', self.a2_scale.reciprocal(),persistent=False)
        self.register_buffer('a1_inverse_experts',self.a1_inverse.expand(num_experts).clone(),persistent=False)
        self.register_buffer('a2_inverse_experts',self.a2_inverse.expand(num_experts).clone(),persistent=False)
        self._expert_cache = OrderedDict()
        self._packed_cache = OrderedDict()
        self.resident_experts = False

    def route(self, x,preserve_rows=False):
        from .row_gemv import per_row_linear
        logits = (per_row_linear(x,self.router)
                  if per_row(x.shape[0]) and preserve_rows else F.linear(x,self.router))
        if self.router_backend=='native':
            from .native_kernels import topk_softmax
            weights,ids=topk_softmax(logits,self.top_k,self.renormalize)
            return weights,ids,logits
        probabilities = logits.float().softmax(-1)
        weights, ids = probabilities.topk(self.top_k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        return weights, ids, logits

    def expert_weights(self, expert):
        if expert in self._expert_cache:
            self._expert_cache.move_to_end(expert)
            return self._expert_cache[expert]
        base = self.prefix+f'.experts.{expert}.'
        decoded = []
        for proj in ('gate_proj','up_proj','down_proj'):
            p = base+proj
            decoded.append(nvfp4_dequant(self.store.get(p+'.weight'),
                self.store.get(p+'.weight_scale'), self.store.get(p+'.weight_scale_2')))
        pair = (torch.cat(decoded[:2]), decoded[2])
        self._expert_cache[expert] = pair
        if len(self._expert_cache) > self.cache_size:
            self._expert_cache.popitem(last=False)
        return pair

    def shared(self, x,preserve_rows=False):
        gate, up = fp8_linear(x, self.shared_up, self.shared_up_scale,self.activation_backend).chunk(2,-1)
        # SiluAndMul narrows SiLU to BF16 before the BF16 multiply. Keeping
        # the product in FP32 moves FP8 midpoints in the following projection.
        activated = F.silu(gate.float()).to(x.dtype) * up
        out = fp8_linear(activated, self.shared_down, self.shared_down_scale,self.activation_backend)
        shared_gate=(torch.cat([F.linear(row,self.shared_gate) for row in x.split(1)])
                     if per_row(x.shape[0]) and preserve_rows and not SHARED_GATE_BATCH else F.linear(x,self.shared_gate))
        return out * torch.sigmoid(shared_gate)

    def packed_expert(self,expert):
        if expert in self._packed_cache:
            self._packed_cache.move_to_end(expert)
            return self._packed_cache[expert]
        from flashinfer import nvfp4_block_scale_interleave
        base=self.prefix+f'.experts.{expert}.'
        # FlashInfer SwiGLU stores up before gate. The original checkpoint
        # loader uses gate's global weight scale for this merged projection.
        w13=torch.cat([self.store.get(base+p+'.weight') for p in ('up_proj','gate_proj')])
        sf13=torch.cat([self.store.get(base+p+'.weight_scale') for p in ('up_proj','gate_proj')])
        w2=self.store.get(base+'down_proj.weight')
        sf2=self.store.get(base+'down_proj.weight_scale')
        sf13=nvfp4_block_scale_interleave(sf13.view(torch.uint8)).reshape_as(sf13).view(torch.int32)
        sf2=nvfp4_block_scale_interleave(sf2.view(torch.uint8)).reshape_as(sf2).view(torch.int32)
        alpha1=self.store.get(base+'gate_proj.weight_scale_2').float()*self.a1_scale
        alpha2=self.store.get(base+'down_proj.weight_scale_2').float()*self.a2_scale
        result=(w13.view(torch.int64),w2.view(torch.int64),sf13,sf2,alpha1,alpha2)
        self._packed_cache[expert]=result
        if len(self._packed_cache)>self.cache_size:self._packed_cache.popitem(last=False)
        return result

    def preload_experts(self):
        """Keep the compressed bank once; never dequantize the whole bank."""
        if self.expert_backend != 'flashinfer':
            raise ValueError('Resident experts require the packed FlashInfer backend')
        if self.resident_experts:
            return
        tensors = [self.packed_expert(expert) for expert in range(self.num_experts)]
        for name, parts in zip(('w13', 'w2', 'sf13', 'sf2', 'alpha1', 'alpha2'), zip(*tensors)):
            self.register_buffer('resident_' + name, torch.stack(parts), persistent=False)
        self._packed_cache.clear()
        self.resident_experts = True

    def packed_forward(self,x,weights,ids,active_experts=None,activation_quant=None):
        from flashinfer import fp4_quantize
        from flashinfer.fused_moe import cutlass_fused_moe
        if self.resident_experts and active_experts is None:
            w13,w2,sf13,sf2,alpha1,alpha2 = [getattr(self, 'resident_' + name)
                for name in ('w13', 'w2', 'sf13', 'sf2', 'alpha1', 'alpha2')]
            selected = ids.to(torch.int32)
            inverse1,inverse2=self.a1_inverse_experts,self.a2_inverse_experts
        else:
            if active_experts is None:
                active,inverse=ids.unique(sorted=True,return_inverse=True)
            else:
                active=torch.as_tensor(active_experts,device=ids.device,dtype=ids.dtype)
                inverse=torch.searchsorted(active,ids)
            tensors=[self.packed_expert(expert) for expert in active.tolist()]
            w13,w2,sf13,sf2,alpha1,alpha2=[torch.stack(part) for part in zip(*tensors)]
            selected = inverse.reshape_as(ids).to(torch.int32)
            inverse1=self.a1_inverse.expand(w13.shape[0]).contiguous()
            inverse2=self.a2_inverse.expand(w13.shape[0]).contiguous()
        q,input_sf=(fp4_quantize(x.contiguous(),self.a1_inverse.reshape(1))
            if activation_quant is None else activation_quant)
        routed=torch.empty_like(x)
        context=nullcontext()
        if self.moe_finalize=='deterministic':
            from flashinfer.autotuner import autotune
            # The pinned image's cache keys omit finalize mode. Its fused
            # profiles can be invalid for non-fused runners; use the vendor
            # heuristic consistently in the original reference and engine.
            context=autotune(False,skip_ops={'trtllm::fused_moe::gemm1','trtllm::fused_moe::gemm2'})
        with context:
            cutlass_fused_moe(input=q,token_selected_experts=selected,
                token_final_scales=weights,fc1_expert_weights=w13,fc2_expert_weights=w2,
                output_dtype=x.dtype,output=routed,input_sf=input_sf,
                quant_scales=[inverse1,sf13,alpha1.flatten(),
                    inverse2,sf2,alpha2.flatten()],**self._packed_options)
        return routed

    def forward(self, x, return_details=False,preserve_rows=False):
        weights, ids, logits = self.route(x,preserve_rows)
        if self.expert_backend=='flashinfer':
            routed=self.packed_forward(x,weights,ids)
            if (per_row(x.shape[0]) and not return_details and self.resident_experts
                    and self.activation_backend == 'sm121-native'):
                from .shared_epilogue import activation, finalize
                gate_up = fp8_linear(x, self.shared_up, self.shared_up_scale, self.activation_backend)
                activated = activation(gate_up)
                shared = fp8_linear(activated, self.shared_down, self.shared_down_scale, self.activation_backend)
                shared_gate = (torch.cat([F.linear(row, self.shared_gate) for row in x.split(1)])
                    if per_row(x.shape[0]) and preserve_rows and not SHARED_GATE_BATCH else F.linear(x, self.shared_gate))
                return finalize(routed, shared, shared_gate)
            shared=self.shared(x,preserve_rows)
            result=(routed+shared).to(x.dtype)
            if return_details:return result,dict(weights=weights,ids=ids,logits=logits,routed=routed,shared=shared)
            return result
        decoded_x = quantize_dequantize_nvfp4(x, self.a1_scale)
        routed = torch.zeros_like(x, dtype=torch.float32)
        # Gather each selected expert's rows. Keep the bank bounded, independent
        # of how many different experts the prompt chooses.
        for expert in ids.unique().tolist():
            token, slot = torch.where(ids == expert)
            w13, w2 = self.expert_weights(expert)
            gate, up = (decoded_x[token] @ w13.T).to(x.dtype).chunk(2,-1)
            activation = (F.silu(gate.float()) * up.float()).to(x.dtype)
            quantized = quantize_dequantize_nvfp4(activation, self.a2_scale)
            expert_out = (quantized @ w2.T).to(x.dtype)
            routed.index_add_(0, token, expert_out.float() * weights[token,slot,None])
        shared = self.shared(x,preserve_rows)
        result = (routed.to(x.dtype) + shared).to(x.dtype)
        if return_details:
            return result, {'weights': weights, 'ids': ids, 'logits': logits,
                            'routed': routed.to(x.dtype), 'shared': shared}
        return result
