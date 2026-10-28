"""BF16 HC projections for one to three tokens of the fixed checkpoint.

The down projection retains the observed five 2048-element split-K groups.
Single-token replacement gave no whole-engine gain. The two-token option
shares each weight traversal while retaining ordinary BF16 split-K storage.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _partial(X,W,P,K:tl.constexpr,N:tl.constexpr,SPLITS:tl.constexpr,
             CHUNK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,M:tl.constexpr=1):
    tile,split=tl.program_id(0),tl.program_id(1)
    m=tl.arange(0,16)
    n=tile*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((16,BN),tl.float32)
    for block in range(tr.cdiv(CHUNK,BK)):
        offset=split*CHUNK+block*BK+k
        a=tl.load(X+m[:,None]*K+offset[None,:],(m[:,None]<M)&(offset[None,:]<K),0)
        b=tl.load(W+n[None,:]*K+offset[:,None],(n[None,:]<N)&(offset[:,None]<K),0)
        acc=tl.dot(a,b,acc=acc)
    # Store only the single useful row. BF16 partial materialization is an
    # explicit option because the native down kernel uses a split reducer.
    if M==1:
        output=tl.sum(tl.where(m[:,None]==0,acc,0.),0)
        tl.store(P+split*N+n,output,n<N)
    else:
        tl.store(P+(split*M+m[:,None])*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


@tr.jit
def _merge(P,Y,N:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr,BN:tl.constexpr,M:tl.constexpr=1):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    row=tl.program_id(1)
    split=tl.arange(0,BS)
    values=tl.load(P+(split[:,None]*M+row)*N+n[None,:],(split[:,None]<SPLITS)&(n[None,:]<N),0).to(tl.float32)
    tl.store(Y+row*N+n,tl.sum(values,0),n<N)


@tr.jit
def _up_gate_mix(L,W,X,Y,SL:tl.constexpr,SX:tl.constexpr,K:tl.constexpr,H:tl.constexpr,C:tl.constexpr,
                 BH:tl.constexpr,BK:tl.constexpr,M:tl.constexpr):
    # Fuses hc_silu -> up projection -> gate_mix. Each stage keeps the unfused
    # Triton path's BF16 storage boundaries (silu output, gate) and the same
    # 16-row tl.dot K order, so every row equals the M=2/3 unfused rows.
    tile=tl.program_id(0)
    m=tl.arange(0,16)
    h=tile*BH+tl.arange(0,BH)
    k=tl.arange(0,BK)
    acc=tl.zeros((16,BH),tl.float32)
    for c in tl.static_range(C):
        n=c*H+h
        gate=tl.zeros((16,BH),tl.float32)
        for block in range(tr.cdiv(K,BK)):
            offset=block*BK+k
            raw=tl.load(L+m[:,None]*SL+offset[None,:],(m[:,None]<M)&(offset[None,:]<K),0).to(tl.float32)/C
            a=(raw*tl.sigmoid(raw)).to(tl.bfloat16)
            b=tl.load(W+n[None,:]*K+offset[:,None],(h[None,:]<H)&(offset[:,None]<K),0)
            gate=tl.dot(a,b,acc=gate)
        g=gate.to(tl.bfloat16).to(tl.float32)
        x=tl.load(X+m[:,None]*SX+n[None,:],(m[:,None]<M)&(h[None,:]<H),0).to(tl.float32)
        acc=tl.fma(tl.sigmoid(g),x,acc)
    tl.store(Y+m[:,None]*H+h[None,:],acc/C,(m[:,None]<M)&(h[None,:]<H))


def up_gate_mix(lowrank,weight,x,count,*,tile=32,warps=2):
    """silu(lowrank/count) @ weight.T, then sigmoid-gated stream mean of x, in one launch."""
    rows,k=lowrank.shape; n=weight.shape[0]; hidden=n//count
    if not 1<=rows<=16 or lowrank.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16 or x.dtype!=torch.bfloat16:
        raise ValueError('Fused HC gate mix requires one to sixteen BF16 rows')
    if (k,n)!=(320,10240) or weight.shape[1]!=k or x.shape!=(rows,n) or n%count:
        raise ValueError('Unverified fused HC gate-mix shape')
    if lowrank.stride(1)!=1 or not weight.is_contiguous() or x.stride(1)!=1:
        raise ValueError('Fused HC gate mix requires unit inner strides')
    y=torch.empty(rows,hidden,device=x.device,dtype=x.dtype)
    _up_gate_mix[(tr.cdiv(hidden,tile),)](lowrank,weight,x,y,lowrank.stride(0),x.stride(0),k,hidden,count,
                                         tile,128,rows,num_warps=warps,num_stages=2)
    return y


def linear(x,weight,*,tile=32,warps=4,round_partials=True,max_rows=3):
    if not 1<=x.shape[0]<=max_rows or max_rows>16 or x.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16:
        raise ValueError('HC projection requires one to max_rows (<=16) BF16 tokens')
    k=weight.shape[1]; n=weight.shape[0]
    if x.shape[1]!=k or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('HC prototype requires contiguous checkpoint rows')
    if k==10240 and n in (320,336):splits,chunk=5,2048
    elif k==320 and n==10240:splits,chunk=1,320
    else:raise ValueError('Unverified HC projection shape')
    if tile not in (16,32,64) or warps not in (4,8):raise ValueError('Unverified launch profile')
    rows=x.shape[0]
    result=torch.empty(rows,n,device=x.device,dtype=x.dtype)
    if splits==1:
        partial=result
    else:
        partial=torch.empty(splits,rows,n,device=x.device,dtype=x.dtype if round_partials else torch.float32)
    _partial[(tr.cdiv(n,tile),splits)](x,weight,partial,k,n,splits,chunk,tile,128,rows,
                                    num_warps=warps,num_stages=2)
    if splits>1:
        _merge[(tr.cdiv(n,128),rows)](partial,result,n,splits,tr.next_power_of_2(splits),128,rows,
                                 num_warps=4)
    return result
