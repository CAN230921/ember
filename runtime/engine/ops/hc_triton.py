"""Independent HC GPU primitives preserving the checkpoint's FMA boundaries.

Equations and BF16 boundaries are based on NVIDIA's Apache-2.0 HC reference.
No vLLM imports or registration. Projection GEMMs remain separate for now.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _norm(X,W,Y,SX:tl.constexpr,SY:tl.constexpr,H:tl.constexpr,C:tl.constexpr,
          SHARED:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0)%C; row=tl.program_id(0)//C
    d=tl.arange(0,B); off=group*H+d
    x=tl.load(X+row*SX+off,d<H,0).to(tl.float32)
    w=tl.load(W+tl.where(SHARED,d,off),d<H,0).to(tl.float32)
    y=x*tl.rsqrt(tl.sum(x*x)/H+EPS)
    y=tl.fma(y,w,y)
    tl.store(Y+row*SY+off,y,d<H)


@tr.jit
def _combine_norm(R,O,I,W,COMBINED,NORM,SR:tl.constexpr,SO:tl.constexpr,SI:tl.constexpr,
        H:tl.constexpr,C:tl.constexpr,SHARED:tl.constexpr,EPS:tl.constexpr,T:tl.constexpr,B:tl.constexpr,CP:tl.constexpr):
    row=tl.program_id(0); group=tl.program_id(1)
    d=tl.arange(0,T)[:,None]*B+tl.arange(0,B)[None,:]
    r=tl.load(R+row*SR+group*H+d,d<H,0).to(tl.float32)
    o=tl.load(O+row*SO+d,d<H,0).to(tl.float32)
    # Reduce a broadcast vector as in the original materialization boundary.
    c=tl.arange(0,CP)
    logits=tl.load(I+row*SI+c,c<C,0).to(tl.float32)
    gains=2*tl.sigmoid(logits/C)
    gain=tl.sum(tl.where(c==group,gains,0))
    combined=tl.fma(o,gain,r).to(COMBINED.dtype.element_ty)
    tl.store(COMBINED+row*(H*C)+group*H+d,combined,d<H)
    x=combined.to(tl.float32)
    square=tl.sum(tl.sum(x*x,1),0)
    y=x*tl.rsqrt(square/H+EPS)
    w=tl.load(W+tl.where(SHARED,d,group*H+d),d<H,0).to(tl.float32)
    y=tl.fma(y,w,y)
    tl.store(NORM+row*(H*C)+group*H+d,y,d<H)


@tr.jit
def _combine(R,O,I,Y,SR:tl.constexpr,SO:tl.constexpr,SI:tl.constexpr,H:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0); group=tl.program_id(1); tile=tl.program_id(2)
    d=tile*B+tl.arange(0,B)
    r=tl.load(R+row*SR+group*H+d,d<H,0).to(tl.float32)
    o=tl.load(O+row*SO+d,d<H,0).to(tl.float32)
    gain=2*tl.sigmoid(tl.load(I+row*SI+group).to(tl.float32)/C)
    tl.store(Y+row*(H*C)+group*H+d,tl.fma(o,gain,r),d<H)


@tr.jit
def _silu(X,Y,SX:tl.constexpr,D:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0); d=tl.arange(0,B)
    x=tl.load(X+row*SX+d,d<D,0).to(tl.float32)/C
    tl.store(Y+row*D+d,x*tl.sigmoid(x),d<D)


@tr.jit
def _gate_mix(X,G,Y,SX:tl.constexpr,SG:tl.constexpr,H:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B)
    acc=tl.full((B,),0,tl.float32)
    for c in tl.static_range(C):
        x=tl.load(X+row*SX+c*H+d,d<H,0).to(tl.float32)
        g=tl.load(G+row*SG+c*H+d,d<H,0).to(tl.float32)
        acc=tl.fma(tl.sigmoid(g),x,acc)
    tl.store(Y+row*H+d,acc/C,d<H)


def norm(x,weight,eps,count):
    y=torch.empty_like(x); hidden=x.shape[1]//count
    _norm[(x.shape[0]*count,)](x,weight,y,x.stride(0),y.stride(0),hidden,count,
        weight.numel()==hidden,eps,tr.next_power_of_2(hidden))
    return y


def combine_norm(residual,output,injection,weight,eps,count):
    combined=torch.empty_like(residual); normalized=torch.empty_like(residual)
    hidden=residual.shape[1]//count
    _combine_norm[(residual.shape[0],count)](residual,output,injection,weight,combined,normalized,
        residual.stride(0),output.stride(0),injection.stride(0),hidden,count,weight.numel()==hidden,eps,
        tr.next_power_of_2(tr.cdiv(hidden,512)),512,tr.next_power_of_2(count))
    return combined,normalized


def combine(residual,output,injection,count):
    result=torch.empty_like(residual); hidden=residual.shape[1]//count
    _combine[(residual.shape[0],count,tr.cdiv(hidden,256))](residual,output,injection,result,
        residual.stride(0),output.stride(0),injection.stride(0),hidden,count,256)
    return result


def silu(x,count):
    y=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    _silu[(x.shape[0],)](x,y,x.stride(0),x.shape[1],count,tr.next_power_of_2(x.shape[1]))
    return y


def gate_mix(x,gate,count):
    hidden=x.shape[1]//count
    y=torch.empty(x.shape[0],hidden,device=x.device,dtype=x.dtype)
    _gate_mix[(x.shape[0],tr.cdiv(hidden,256))](x,gate,y,x.stride(0),gate.stride(0),hidden,count,256)
    return y
