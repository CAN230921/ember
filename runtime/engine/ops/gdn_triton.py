"""Specialized single-sequence GDN conv/prep/decode/gated-norm kernels.

Prefill uses vendored FLA arithmetic. Decode preserves the ordinary packed
decode's BF16 beta and core-output boundaries; MTP fused-norm arithmetic is
different and cannot substitute for this checkpoint's non-speculative trace.
"""
import torch
from ..limits import VERIFY_WIDTHS
import triton as tr
import triton.language as tl
from collections import OrderedDict
from .gdn import GDNState


class PrefillCuSeqLensCache:
    """Bounded, read-only single-sequence metadata shared by prefill layers.

    FLA caches derived chunk indices by tensor identity, so constructing a new
    [0, rows] CUDA tensor in every layer causes repeated metadata D2H waits.
    FLA only reads this tensor; no request state or numerical input is shared.
    The production prefill path is eager. A future captured caller must warm
    the complete FLA prefill path first, including its derived metadata cache.
    Capture hits do not mutate this LRU; capture misses fail before allocation.
    """
    def __init__(self, max_entries=32, *, tensor_factory=None, cuda=None):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('Expected a positive metadata cache capacity')
        self.max_entries = max_entries
        self.tensor_factory = torch.tensor if tensor_factory is None else tensor_factory
        self.cuda = torch.cuda if cuda is None else cuda
        self._entries = OrderedDict()

    def _get(self, rows, device, capturing):
        key = (device.type, device.index, rows)
        if key in self._entries:
            if not capturing:
                self._entries.move_to_end(key)
            return self._entries[key]
        if capturing:
            raise RuntimeError('Warm the complete GDN prefill metadata path before CUDA Graph capture')
        value = self.tensor_factory([0, rows], device=device, dtype=torch.int32)
        self._entries[key] = value
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return value

    def get(self, rows, device):
        if type(rows) is not int or rows < 1:
            raise ValueError('Expected a positive prefill sequence length')
        device = torch.device(device)
        if device.type != 'cuda':
            return self._get(rows, device, False)
        if device.index is None:
            device = torch.device('cuda', self.cuda.current_device())
        with self.cuda.device(device):
            return self._get(rows, device, self.cuda.is_current_stream_capturing())


_prefill_cu_seqlens = PrefillCuSeqLensCache()


@tr.jit
def _conv(X,S,W,Y,UPDATED,SX:tl.constexpr,D:tl.constexpr,T:tl.constexpr,B:tl.constexpr,
          ROUNDED_PRODUCTS:tl.constexpr=True,PREFIX=None,CAPTURE_PREFIX:tl.constexpr=False):
    d=tl.program_id(0)*B+tl.arange(0,B)
    first=tl.load(S+d*3,d<D,0).to(tl.float32)
    second=tl.load(S+d*3+1,d<D,0).to(tl.float32)
    third=tl.load(S+d*3+2,d<D,0).to(tl.float32)
    w0=tl.load(W+d*4,d<D,0).to(tl.float32)
    w1=tl.load(W+d*4+1,d<D,0).to(tl.float32)
    w2=tl.load(W+d*4+2,d<D,0).to(tl.float32)
    w3=tl.load(W+d*4+3,d<D,0).to(tl.float32)
    for row in range(T):
        current=tl.load(X+row*SX+d,d<D,0).to(tl.float32)
        if ROUNDED_PRODUCTS:
            # The checkpoint's original causal conv rounds each BF16 product
            # before FP32 accumulation. Fusing these products changes the model.
            value=(first*w0).to(Y.dtype.element_ty).to(tl.float32)
            value+=(second*w1).to(Y.dtype.element_ty).to(tl.float32)
            value+=(third*w2).to(Y.dtype.element_ty).to(tl.float32)
            value+=(current*w3).to(Y.dtype.element_ty).to(tl.float32)
        else:
            value=first*w0
            value=tl.fma(second,w1,value)
            value=tl.fma(third,w2,value)
            value=tl.fma(current,w3,value)
        activated=value/(1+tl.exp(-value))
        tl.store(Y+row*D+d,activated,d<D)
        first=second; second=third; third=current
        if CAPTURE_PREFIX:
            tl.store(PREFIX+row*D*3+d*3,first,d<D)
            tl.store(PREFIX+row*D*3+d*3+1,second,d<D)
            tl.store(PREFIX+row*D*3+d*3+2,third,d<D)
    tl.store(UPDATED+d*3,first,d<D)
    tl.store(UPDATED+d*3+1,second,d<D)
    tl.store(UPDATED+d*3+2,third,d<D)


@tr.jit
def _prep(X,A,B,ALOG,BIAS,Q,K,V,G,BETA,SA:tl.constexpr,SB:tl.constexpr,T:tl.constexpr,BT:tl.constexpr):
    rows=tl.program_id(0)*BT+tl.arange(0,BT)
    head=tl.program_id(1); d=tl.arange(0,128)
    if head<16:
        q=tl.load(X+rows[:,None]*10240+head*128+d[None,:],rows[:,None]<T,0).to(tl.float32)
        k=tl.load(X+rows[:,None]*10240+2048+head*128+d[None,:],rows[:,None]<T,0).to(tl.float32)
        q=q*(1/tl.sqrt(tl.sum(q*q,1)+1e-6))[:,None]
        k=k*(1/tl.sqrt(tl.sum(k*k,1)+1e-6))[:,None]
        tl.store(Q+rows[:,None]*2048+head*128+d[None,:],q,rows[:,None]<T)
        tl.store(K+rows[:,None]*2048+head*128+d[None,:],k,rows[:,None]<T)
    else:
        hv=head-16
        v=tl.load(X+rows[:,None]*10240+4096+hv*128+d[None,:],rows[:,None]<T,0)
        tl.store(V+rows[:,None]*6144+hv*128+d[None,:],v,rows[:,None]<T)
        a=tl.load(A+rows*SA+hv,rows<T,0).to(tl.float32)
        b=tl.load(B+rows*SB+hv,rows<T,0).to(tl.float32)
        alog=tl.load(ALOG+hv).to(tl.float32)
        bias=tl.load(BIAS+hv).to(tl.float32)
        x=a+bias
        softplus=tl.where(x>0,x+tl.log(1+tl.exp(-x)),tl.log(1+tl.exp(x)))
        softplus=tl.where(x<=20,softplus,x)
        tl.store(G+rows*48+hv,-tl.exp(alog)*softplus,rows<T)
        tl.store(BETA+rows*48+hv,tl.sigmoid(b),rows<T)


@tr.jit
def _gated_norm(X,Z,W,Y,SZ:tl.constexpr,EPS:tl.constexpr,SIGMOID:tl.constexpr,
                M:tl.constexpr,BM:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM); d=tl.arange(0,128)
    x=tl.load(X+rows[:,None]*128+d[None,:],rows[:,None]<M,0).to(tl.float32)
    z=tl.load(Z+(rows//48)[:,None]*SZ+(rows%48)[:,None]*128+d[None,:],rows[:,None]<M,0).to(tl.float32)
    weight=tl.load(W+d).to(tl.float32)
    normalized=x*tl.rsqrt(tl.sum(x*x,1)/128+EPS)[:,None]*weight[None,:]
    gate=tl.sigmoid(z)
    if not SIGMOID:gate=z*gate
    tl.store(Y+rows[:,None]*128+d[None,:],normalized*gate,rows[:,None]<M)


@tr.jit
def _decode(X,A,B,ALOG,BIAS,S,UPDATED,O,T:tl.constexpr=1,
            SA:tl.constexpr=48,SB:tl.constexpr=48,PREFIX=None,
            CAPTURE_PREFIX:tl.constexpr=False,STORE_FINAL:tl.constexpr=True):
    head=tl.program_id(1); key_head=head//3
    kdim=tl.arange(0,128); vdim=tl.program_id(0)*32+tl.arange(0,32)
    alog=tl.load(ALOG+head).to(tl.float32)
    bias=tl.load(BIAS+head).to(tl.float32)
    offset=head*16384+vdim[:,None]*128+kdim[None,:]
    updated=tl.load(S+offset).to(tl.float32)
    for row in range(T):
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
        if CAPTURE_PREFIX:
            tl.store(PREFIX+row*48*16384+offset,updated)
        # An ordinary decode writes the recurrent cache to BF16 between
        # tokens. The next row must observe exactly that materialization.
        if row+1<T:
            updated=updated.to(S.dtype.element_ty).to(tl.float32)
    if STORE_FINAL:
        tl.store(UPDATED+offset,updated)


def run_core(model,qkvz,ba,state,prefill,return_prefixes=False):
    rows=qkvz.shape[0]
    if return_prefixes and (prefill or rows not in VERIFY_WIDTHS):
        raise ValueError('Prefix capture requires one to three decode tokens')
    if state is None:state=GDNState.zeros(1,device=qkvz.device,dtype=qkvz.dtype,state_dtype=torch.bfloat16)
    convolved=torch.empty(rows,10240,device=qkvz.device,dtype=qkvz.dtype)
    prefix_conv=torch.empty((rows,*state.conv.shape),device=state.conv.device,dtype=state.conv.dtype) if return_prefixes else None
    updated_conv=prefix_conv[-1] if return_prefixes else torch.empty_like(state.conv)
    _conv[(tr.cdiv(10240,256),)](qkvz,state.conv,model.conv,convolved,updated_conv,qkvz.stride(0),10240,rows,256,True,
        prefix_conv,return_prefixes)
    b,a=ba.split((48,48),-1)
    z=qkvz[:,10240:]
    normalized=torch.empty(rows,6144,device=qkvz.device,dtype=qkvz.dtype)
    if prefill:
        from .gdn_prefill_profile_sm121 import configure
        configure(qkvz.device.index)
        q=torch.empty(rows,16,128,device=qkvz.device,dtype=qkvz.dtype); k=torch.empty_like(q)
        v=torch.empty(rows,48,128,device=qkvz.device,dtype=qkvz.dtype)
        g=torch.empty(rows,48,device=qkvz.device); beta=torch.empty_like(g)
        _prep[(tr.cdiv(rows,16),64)](convolved,a,b,model.A_log,model.dt_bias,q,k,v,g,beta,a.stride(0),b.stride(0),rows,16,
            num_warps=4,num_stages=2)
        from engine.vendor.fla.chunk import chunk_gated_delta_rule
        # Match the reference's packed, variable-length prefill contract,
        # including the explicit single sequence's start/end metadata.
        cu_seqlens=_prefill_cu_seqlens.get(rows,qkvz.device)
        core,recurrent=chunk_gated_delta_rule(q[None],k[None],v[None],g[None],beta[None],
            initial_state=state.recurrent,output_final_state=True,cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False)
        # FLA returns FP32 final state; the reference writes it into its BF16
        # cache before the next token. Preserve that write boundary here.
        recurrent=recurrent.to(state.recurrent.dtype)
    else:
        if rows not in VERIFY_WIDTHS:raise ValueError('GDN kernels support the configured decode verification widths')
        core=torch.empty_like(normalized)
        prefix_recurrent=torch.empty((rows,*state.recurrent.shape),device=state.recurrent.device,dtype=state.recurrent.dtype) if return_prefixes else None
        recurrent=prefix_recurrent[-1] if return_prefixes else torch.empty_like(state.recurrent)
        _decode[(4,48)](convolved,a,b,model.A_log,model.dt_bias,state.recurrent,recurrent,core,
            rows,a.stride(0),b.stride(0),prefix_recurrent,return_prefixes,
            num_warps=1,num_stages=3)
    norm_rows=rows*48
    sm_count=torch.cuda.get_device_properties(qkvz.device).multi_processor_count
    # Decode must retain the ordinary single-row reduction layout. With
    # three tokens, the occupancy heuristic packs two value heads together
    # and shifts a BF16 midpoint on a real layer-44 input.
    block_rows=min(4,tr.next_power_of_2(tr.cdiv(norm_rows,2*sm_count))) if prefill else 1
    _gated_norm[(tr.cdiv(norm_rows,block_rows),)](core,z,model.norm,normalized,z.stride(0),model.eps,
        model.gate_activation=='sigmoid',norm_rows,block_rows,num_warps=1)
    result=(normalized,GDNState(updated_conv,recurrent))
    if return_prefixes:
        return (*result,[GDNState(prefix_conv[row],prefix_recurrent[row]) for row in range(rows)])
    return result
