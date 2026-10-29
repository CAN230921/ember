"""Small-tree GDN with the incumbent BF16 arithmetic and parent caches.

One convolution, recurrence and normalization launch per layer. The recurrence
expression/order follows gdn_triton._decode; every parent cache is materialized
as BF16. This retains per-node states and is not an FP32 triangular solve.
"""
import torch
import triton as tr
import triton.language as tl

from .gdn import GDNState
from .gdn_triton import _gated_norm


@tr.jit
def _conv_tree(X,S,W,P,Y,BANK,SX:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    d=tl.program_id(0)*B+tl.arange(0,B)
    w0=tl.load(W+d*4).to(tl.float32)
    w1=tl.load(W+d*4+1).to(tl.float32)
    w2=tl.load(W+d*4+2).to(tl.float32)
    w3=tl.load(W+d*4+3).to(tl.float32)
    for row in range(T):
        parent=tl.load(P+row)
        source=tl.where(parent<0,S+d*3,BANK+parent*10240*3+d*3)
        first=tl.load(source).to(tl.float32)
        second=tl.load(source+1).to(tl.float32)
        third=tl.load(source+2).to(tl.float32)
        current=tl.load(X+row*SX+d).to(tl.float32)
        value=(first*w0).to(Y.dtype.element_ty).to(tl.float32)
        value+=(second*w1).to(Y.dtype.element_ty).to(tl.float32)
        value+=(third*w2).to(Y.dtype.element_ty).to(tl.float32)
        value+=(current*w3).to(Y.dtype.element_ty).to(tl.float32)
        activated=value/(1+tl.exp(-value))
        tl.store(Y+row*10240+d,activated)
        tl.store(BANK+row*10240*3+d*3,second)
        tl.store(BANK+row*10240*3+d*3+1,third)
        tl.store(BANK+row*10240*3+d*3+2,current)


@tr.jit
def _decode_tree(X,A,B,ALOG,BIAS,S,P,BANK,O,T:tl.constexpr,
                 SA:tl.constexpr,SB:tl.constexpr):
    head=tl.program_id(1); key_head=head//3
    kdim=tl.arange(0,128); vdim=tl.program_id(0)*32+tl.arange(0,32)
    alog=tl.load(ALOG+head).to(tl.float32)
    bias=tl.load(BIAS+head).to(tl.float32)
    offset=head*16384+vdim[:,None]*128+kdim[None,:]
    for row in range(T):
        parent=tl.load(P+row)
        source=tl.where(parent<0,S+offset,BANK+parent*48*16384+offset)
        updated=tl.load(source).to(tl.float32)
        q=tl.load(X+row*10240+key_head*128+kdim).to(tl.float32)
        k=tl.load(X+row*10240+2048+key_head*128+kdim).to(tl.float32)
        v=tl.load(X+row*10240+4096+head*128+vdim).to(tl.float32)
        q=q/tl.sqrt(tl.sum(q*q)+1e-6)
        k=k/tl.sqrt(tl.sum(k*k)+1e-6)
        q=q*128**-.5
        a=tl.load(A+row*SA+head).to(tl.float32)
        b=tl.load(B+row*SB+head).to(tl.float32)
        x=a+bias
        softplus=tl.where(x<=20,tl.log(1+tl.exp(x)),x)
        g=-tl.exp(alog)*softplus
        beta=tl.sigmoid(b).to(B.dtype.element_ty).to(tl.float32)
        h=updated*tl.exp(g)
        delta=(v-tl.sum(h*k[None,:],1))*beta
        updated=h+delta[:,None]*k[None,:]
        core=tl.sum(updated*q[None,:],1)
        tl.store(O+row*6144+head*128+vdim,core)
        tl.store(BANK+row*48*16384+offset,updated)


def run_tree(model,qkvz,ba,state,tree,parents=None):
    from ..candidate_tree import CandidateTree
    if (not isinstance(tree,CandidateTree) or not qkvz.is_cuda or
            qkvz.shape!=(len(tree.tokens),16384) or ba.shape!=(len(tree.tokens),96) or
            any(x.dtype!=torch.bfloat16 or x.device!=qkvz.device for x in
                (qkvz,ba,state.conv,state.recurrent)) or not qkvz.is_contiguous() or
            state.conv.shape!=(1,10240,3) or state.recurrent.shape!=(1,48,128,128) or
            not state.conv.is_contiguous() or not state.recurrent.is_contiguous()):
        raise ValueError('Expected current-checkpoint CUDA BF16 tree inputs and private state')
    if parents is None:parents=torch.tensor(tree.parents,device=qkvz.device,dtype=torch.int32)
    if (parents.shape!=(len(tree.tokens),) or parents.device!=qkvz.device or
            parents.dtype!=torch.int32 or not parents.is_contiguous()):
        raise ValueError('Expected tree parent indices')
    rows=len(tree.tokens)
    convolved=qkvz.new_empty(rows,10240)
    conv=qkvz.new_empty(rows,*state.conv.shape)
    recurrent=qkvz.new_empty(rows,*state.recurrent.shape)
    core=qkvz.new_empty(rows,6144);output=torch.empty_like(core)
    _conv_tree[(40,)](qkvz,state.conv,model.conv,parents,convolved,conv,qkvz.stride(0),rows,256)
    b,a=ba.split((48,48),-1)
    _decode_tree[(4,48)](convolved,a,b,model.A_log,model.dt_bias,state.recurrent,parents,
        recurrent,core,rows,a.stride(0),b.stride(0),num_warps=1,num_stages=3)
    z=qkvz[:,10240:]
    _gated_norm[(rows*48,)](core,z,model.norm,output,z.stride(0),model.eps,
        model.gate_activation=='sigmoid',rows*48,1,num_warps=1)
    states=[]
    for node in range(rows):
        value=GDNState(conv[node],recurrent[node]);value.tree_view=True;states.append(value)
    return output,states
