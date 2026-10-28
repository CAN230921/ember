"""Torch WY chunk prefill with the incumbent BF16 intermediate boundaries.

The sequential recurrence is mathematically equivalent, but this quantized
model amplifies its different rounding through FP8/NVFP4 downstream layers.
This implementation keeps chunk coefficients, W/U and scores in BF16 where
the Triton/FLA reference stores them. Single sequence, chunk size 64.
"""
import torch


def chunk_prefill(q, k, v, g, beta, state, chunk_size=64):
    if q.ndim != 3 or q.dtype != torch.bfloat16 or k.shape != q.shape:
        raise ValueError('Expected BF16 [tokens, key_heads, K] Q/K')
    tokens, key_heads, key_dim = q.shape
    value_heads = v.shape[1]
    if value_heads % key_heads or state.shape != (1,value_heads,v.shape[2],key_dim):
        raise ValueError('Invalid single-sequence prefill layout')
    repeat=value_heads//key_heads
    q=q.repeat_interleave(repeat,dim=1).transpose(0,1)
    k=k.repeat_interleave(repeat,dim=1).transpose(0,1)
    v=v.transpose(0,1)
    g,beta=g.transpose(0,1).float(),beta.transpose(0,1).float()
    h=state[0].float()
    outputs=[]
    for begin in range(0,tokens,chunk_size):
        end=min(tokens,begin+chunk_size)
        qc,kc,vc=q[:,begin:end],k[:,begin:end],v[:,begin:end]
        gates=g[:,begin:end].cumsum(-1)
        betac=beta[:,begin:end]
        length=end-begin
        lower=torch.ones(length,length,device=q.device,dtype=torch.bool).tril()
        decay=(gates[:,:,None]-gates[:,None,:]).clamp_max(0).exp()
        # beta-scaled K is FP32 in the NVIDIA KKT kernel.
        coefficients=((kc.float()*betac[:,:,None]) @ kc.float().transpose(-1,-2)) * decay
        coefficients=coefficients.tril(-1)
        identity=torch.eye(length,device=q.device).expand(value_heads,-1,-1)
        inverse=torch.linalg.solve_triangular(identity+coefficients,identity,upper=False).to(q.dtype)
        weighted_k=(kc.float()*betac[:,:,None]*gates.exp()[:,:,None]).to(q.dtype)
        weighted_v=(vc.float()*betac[:,:,None]).to(q.dtype)
        w=(inverse.float() @ weighted_k.float()).to(q.dtype)
        u=(inverse.float() @ weighted_v.float()).to(q.dtype)
        # The chunk history is stored as BF16 for the state-to-output GEMMs.
        stored_h=h.to(q.dtype).float()
        residual_v=u.float()-w.float() @ stored_h.transpose(-1,-2)
        v_new=residual_v.to(q.dtype)
        scores=(qc.float() @ kc.float().transpose(-1,-2))*decay
        scores=torch.where(lower,scores,0).to(q.dtype)
        output=(qc.float() @ stored_h.transpose(-1,-2))*gates.exp()[:,:,None]
        output=(output+scores.float() @ v_new.float())*key_dim**-.5
        outputs.append(output.to(q.dtype).transpose(0,1))
        tail_decay=(gates[:,-1,None]-gates).exp()
        updated_v=(residual_v*tail_decay[:,:,None]).to(q.dtype)
        h=h*gates[:,-1].exp()[:,None,None]+updated_v.float().transpose(-1,-2) @ kc.float()
    return torch.cat(outputs),h.unsqueeze(0).to(state.dtype)
