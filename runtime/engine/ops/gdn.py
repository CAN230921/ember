"""Independent correctness runtime for non-interleaved Qwen GDN.

State layout: convolution [batch, channels, kernel-1], recurrent
[batch, value_heads, value_dim, key_dim]. Uses torch only. CUDA performance
kernels, speculative rollback and serving allocation remain separate work.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class GDNState:
    conv: torch.Tensor
    recurrent: torch.Tensor

    @classmethod
    def zeros(cls, batch, key_heads=16, value_heads=48, key_dim=128, value_dim=128,
              kernel=4, device="cpu", dtype=torch.bfloat16, state_dtype=torch.float32):
        channels = 2 * key_heads * key_dim + value_heads * value_dim
        return cls(torch.zeros(batch, channels, kernel-1, device=device, dtype=dtype),
                   torch.zeros(batch, value_heads, value_dim, key_dim, device=device, dtype=state_dtype))


def causal_conv_step(x, conv_state, weight, bias=None):
    weight = weight.reshape(weight.shape[0], -1)
    if x.ndim != 2 or conv_state.shape != (*x.shape, weight.shape[1]-1):
        raise ValueError("Invalid single-token convolution state")
    window = torch.cat((conv_state, x.unsqueeze(-1)), dim=-1)
    output = (window.float() * weight.float()[None]).sum(-1)
    if bias is not None:
        output = output + bias.float()
    return F.silu(output).to(x.dtype), window[:, :, 1:].contiguous()


def post_conv_prep(qkv, a, b, A_log, dt_bias, key_heads=16, key_dim=128, value_dim=128,
                   prefill=False, decode_beta_round=True):
    tokens = qkv.shape[0]
    value_heads = A_log.numel()
    qdim, vdim = key_heads*key_dim, value_heads*value_dim
    if qkv.shape != (tokens, 2*qdim+vdim) or a.shape != (tokens,value_heads) or b.shape != a.shape:
        raise ValueError("Invalid QKV/gate shape")
    q, k, v = qkv.split((qdim,qdim,vdim),dim=-1)
    q, k = q.float().reshape(tokens,key_heads,key_dim), k.float().reshape(tokens,key_heads,key_dim)
    q = q * torch.rsqrt(q.square().sum(-1,keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1,keepdim=True) + 1e-6)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float(), threshold=20)
    beta = torch.sigmoid(b.float())
    if prefill:
        # The prefill prep kernel stores normalized Q/K in activation dtype,
        # but retains beta in FP32. Packed decode keeps Q/K in FP32 while
        # rounding beta to the dtype of b before updating recurrent state.
        q, k = q.to(qkv.dtype), k.to(qkv.dtype)
    elif decode_beta_round:
        beta = beta.to(b.dtype).float()
    return q, k, v.reshape(tokens,value_heads,value_dim), g, beta


def recurrent_step(q, k, v, g, beta, state, output_dtype=None):
    batch, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[1:]
    if value_heads % key_heads or state.shape != (batch,value_heads,value_dim,key_dim):
        raise ValueError("Invalid recurrent state/head mapping")
    repeat = value_heads // key_heads
    q = q.float().repeat_interleave(repeat,dim=1) * key_dim**-0.5
    k = k.float().repeat_interleave(repeat,dim=1)
    decayed = state.float() * g.float().exp()[:, :, None, None]
    delta = (v.float() - (decayed * k[:, :, None, :]).sum(-1)) * beta.float()[:, :, None]
    updated = decayed + delta[:, :, :, None] * k[:, :, None, :]
    output = (updated * q[:, :, None, :]).sum(-1).to(output_dtype or v.dtype)
    return output, updated.to(state.dtype)


def gdn_core_step(qkv, a, b, A_log, dt_bias, conv_weight, state, key_heads=16,
                  key_dim=128, value_dim=128, conv_bias=None, prefill=False):
    convolved, conv = causal_conv_step(qkv, state.conv, conv_weight, conv_bias)
    q,k,v,g,beta = post_conv_prep(convolved,a,b,A_log,dt_bias,key_heads,key_dim,value_dim,prefill)
    output,recurrent = recurrent_step(q,k,v,g,beta,state.recurrent)
    return output, GDNState(conv,recurrent)
