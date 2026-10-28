"""Independent torch bring-up for GB10 Qwen HyperConnection.

Implements the math and BF16 boundaries in the reference NVIDIA hc.py.
This is a correctness implementation; no performance claim is implied.
Runtime imports only torch, not vLLM. Layout is [tokens, streams * hidden].
"""
import os

import torch

from ..limits import per_row
from torch import nn
from torch.nn import functional as F


def grouped_gemma_rmsnorm(x, weight, eps, hc_count):
    if x.ndim != 2 or x.shape[-1] % hc_count:
        raise ValueError("Expected [tokens, streams * hidden]")
    hidden = x.shape[-1] // hc_count
    if weight.numel() not in (hidden, x.shape[-1]):
        raise ValueError("Invalid grouped norm affine shape")
    grouped = x.float().reshape(x.shape[0], hc_count, hidden)
    norm = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    affine = weight.float().reshape(1, -1, hidden)
    return (norm + norm * affine).reshape_as(x).to(x.dtype)


def hc_silu(x, hc_count):
    return F.silu(x.float() / hc_count).to(x.dtype)


def hc_gate_mix(x, gate, hc_count):
    if gate.shape != x.shape or x.shape[-1] % hc_count:
        raise ValueError("Invalid gate layout")
    hidden = x.shape[-1] // hc_count
    values = (x.float() * torch.sigmoid(gate.float())).reshape(x.shape[0], hc_count, hidden)
    # Match the reference's sequential stream accumulation.
    acc = torch.zeros_like(values[:, 0])
    for stream in range(hc_count):
        acc = acc + values[:, stream]
    return (acc / hc_count).to(x.dtype)


def hc_combine(residual, block_output, injection, hc_count):
    tokens, dim = residual.shape
    hidden = dim // hc_count
    if dim % hc_count or block_output.shape != (tokens, hidden) or injection.shape != (tokens, hc_count):
        raise ValueError("Invalid residual injection layout")
    gain = 2.0 * torch.sigmoid(injection.float() / hc_count)
    combined = residual.float().reshape(tokens, hc_count, hidden) + block_output.float()[:, None, :] * gain[:, :, None]
    return combined.reshape_as(residual).to(residual.dtype)


def hc_combine_norm(residual, block_output, injection, weight, eps, hc_count):
    combined = hc_combine(residual, block_output, injection, hc_count)
    # The fused reference rounds combined to BF16 before normalization.
    return combined, grouped_gemma_rmsnorm(combined, weight, eps, hc_count)


class GatedResidualRef(nn.Module):
    @classmethod
    def from_weight_store(cls, store, prefix, hc_count=4, eps=1e-6, use_combine=True,backend='torch',projection_backend='torch'):
        norm = store.get(prefix + ".hc_norm.weight")
        down = store.get(prefix + ".input_mix_weight_down.weight")
        up = store.get(prefix + ".input_mix_weight_up.weight")
        if use_combine:
            injection = store.get(prefix + ".block_inject_weight.weight")
            padding = (-(down.shape[0] + hc_count)) % 16
            down = torch.cat((down, injection, down.new_zeros(padding, down.shape[1])), dim=0)
        return cls(norm, down, up, hc_count, eps,use_combine,backend,projection_backend)

    def __init__(self, norm_weight, down_weight, up_weight, hc_count=4, eps=1e-6, use_combine=True,backend='torch',projection_backend='torch'):
        super().__init__()
        self.register_buffer("norm_weight", norm_weight)
        self.register_buffer("down_weight", down_weight)
        self.register_buffer("up_weight", up_weight)
        self.hc_count, self.eps, self.use_combine = hc_count, eps, use_combine
        if backend not in ('torch','triton'):raise ValueError('Unknown HC backend')
        self.backend=backend
        if projection_backend not in ('torch','sm121','sm121-two'):raise ValueError('Unknown HC projection backend')
        self.projection_backend=projection_backend
        self.fused = os.environ.get('SPARK_HC_FUSED','0')=='1'
        self.rank = up_weight.shape[1]
        dim = up_weight.shape[0]
        if down_weight.ndim != 2 or down_weight.shape[1] != dim or dim % hc_count:
            raise ValueError("Invalid projection shape")
        if down_weight.shape[0] < self.rank + (hc_count if use_combine else 0):
            raise ValueError("Missing injection rows")

    def fused_rows(self, rows):
        # Opt-in: one weight traversal for every row (all lanes), silu/up/gate_mix fused.
        # The Triton projection reproduces the cuBLAS M=1 reduction, so the plain
        # per-row ('torch') draft layout is covered as well as 'sm121-two'.
        return (self.fused and self.backend=='triton' and self.projection_backend in ('torch','sm121-two')
                and 1<=rows<=16 and self.rank==320 and self.down_weight.shape[1]==10240)

    def _mix_normalized(self, residual, xn, preserve_rows=False):
        if self.fused_rows(xn.shape[0]):
            from .hc_projection import linear,up_gate_mix
            projected = linear(xn,self.down_weight,max_rows=16)
            injection = projected[:, self.rank:self.rank + self.hc_count] if self.use_combine else None
            return residual,up_gate_mix(projected[:, :self.rank],self.up_weight,xn,self.hc_count),injection
        projected = self.project(xn,self.down_weight,preserve_rows)
        lowrank = projected[:, :self.rank]
        injection = projected[:, self.rank:self.rank + self.hc_count] if self.use_combine else None
        if self.backend=='triton':
            from . import hc_triton
            gate=self.project(hc_triton.silu(lowrank,self.hc_count),self.up_weight,preserve_rows)
            return residual,hc_triton.gate_mix(xn,gate,self.hc_count),injection
        gate = self.project(hc_silu(lowrank, self.hc_count), self.up_weight,preserve_rows)
        return residual, hc_gate_mix(xn, gate, self.hc_count), injection

    def project(self,x,weight,preserve_rows=False):
        if self.projection_backend=='sm121-two' and x.shape[0] in (2,3) and preserve_rows:
            from .hc_projection import linear
            return linear(x,weight)
        if self.projection_backend=='sm121' and (x.shape[0]==1 or x.shape[0] in (2,3) and preserve_rows):
            from .hc_projection import linear
            return torch.cat([linear(row,weight) for row in x.split(1)]) if x.shape[0]>1 else linear(x,weight)
        if per_row(x.shape[0]) and preserve_rows:
            # Preserve the ordinary M=1 split-K/BF16 partial sums. cuBLAS
            # switches arithmetic for M=2 and the difference compounds through
            # the 48 HC layers, even when both attention cores are exact.
            return torch.cat([F.linear(row,weight) for row in x.split(1)])
        return F.linear(x,weight)

    def mix(self, residual, preserve_rows=False):
        if self.backend=='triton':
            from . import hc_triton
            return self._mix_normalized(residual,hc_triton.norm(residual,self.norm_weight,self.eps,self.hc_count),preserve_rows)
        xn = grouped_gemma_rmsnorm(residual, self.norm_weight, self.eps, self.hc_count)
        return self._mix_normalized(residual, xn,preserve_rows)

    def combine_and_mix(self, residual, block_output, injection,preserve_rows=False):
        if self.backend=='triton':
            from . import hc_triton
            combined,xn=hc_triton.combine_norm(residual,block_output,injection,self.norm_weight,self.eps,self.hc_count)
            return self._mix_normalized(combined,xn,preserve_rows)
        combined, xn = hc_combine_norm(residual, block_output, injection, self.norm_weight, self.eps, self.hc_count)
        return self._mix_normalized(combined, xn,preserve_rows)

    def combine(self,residual,block_output,injection):
        if self.backend=='triton':
            from . import hc_triton
            return hc_triton.combine(residual,block_output,injection,self.hc_count)
        return hc_combine(residual,block_output,injection,self.hc_count)
