"""Independent QSA runtime and correctness reference for this checkpoint.

No vLLM imports. Uses FlashInfer only for standalone NVFP4 cache quantization.
Torch math remains available as a diagnostic backend. The accepted Triton path
uses sparse selection, hardware NVFP4 attention and immutable paged KV; each
request owns its QSA history while the batch engine shares large projections.
"""
import math
from dataclasses import dataclass
import torch
from ..limits import VERIFY_ROWS, VERIFY_WIDTHS, per_row
from torch import nn
from torch.nn import functional as F
from .fp8 import linear as fp8_linear
from .m0_refs import nvfp4_dequant
from .kv_cache import append_kv,materialize_kv


def gemma_norm(x,weight,eps):
    value=x.float()
    return (value*torch.rsqrt(value.square().mean(-1,keepdim=True)+eps)*(1+weight.float())).to(x.dtype)


class TextRoPE:
    """NeoX partial RoPE, including incumbent MRoPE YaRN frequency semantics.

    MRoPE's constructor expands the original range by four BEFORE calculating
    YaRN correction frequencies. Cache values are BF16 in this BF16 model.
    Only text (identical temporal/height/width positions) is supported here.
    """
    def __init__(self,parameters,head_dim,device):
        self.dim=int(head_dim*parameters.get('partial_rotary_factor',1))
        self.parameters=parameters
        self.device=device
        base=parameters.get('rope_theta',10000)
        exponent=torch.arange(0,self.dim,2,dtype=torch.float32,device=device)/self.dim
        frequency=base**exponent
        kind=parameters.get('rope_type','default')
        self.mscale=1.
        if kind=='yarn':
            factor=parameters['factor']
            maximum=parameters['original_max_position_embeddings']
            if 'mrope_section' in parameters:maximum*=4
            def boundary(rotations):
                return self.dim*math.log(maximum/(rotations*2*math.pi))/(2*math.log(base))
            low,high=boundary(parameters.get('beta_fast',32)),boundary(parameters.get('beta_slow',1))
            if parameters.get('truncate',True):low,high=math.floor(low),math.ceil(high)
            low,high=max(low,0),min(high,self.dim-1)
            if low==high:high+=.001
            ramp=((torch.arange(self.dim//2,device=device,dtype=torch.float32)-low)/(high-low)).clamp(0,1)
            mask=(1-ramp)*parameters.get('extrapolation_factor',1)
            self.inv_freq=(1/(factor*frequency))*(1-mask)+(1/frequency)*mask
            self.mscale=(1+.1*math.log(factor) if factor>1 else 1)*parameters.get('attn_factor',1)
        elif kind=='default':self.inv_freq=1/frequency
        else:raise ValueError('Only checkpoint default/YaRN RoPE supported')

    def canonical(self,positions):
        if positions.ndim==2:
            if positions.shape[0]!=3 or not torch.equal(positions,positions[0:1].expand_as(positions)):
                raise ValueError('Text runtime requires identical MRoPE axes')
            positions=positions[0]
        if positions.ndim!=1:raise ValueError('Expected text positions [T] or [3,T]')
        return positions

    def cache_rows(self,positions):
        positions=self.canonical(positions)
        frequency=torch.outer(positions.float(),self.inv_freq)
        return torch.cat((frequency.cos()*self.mscale,frequency.sin()*self.mscale),-1).bfloat16()

    def __call__(self,x,positions):
        cache=self.cache_rows(positions).float()
        cos,sin=cache.chunk(2,-1)
        cos,sin=cos[:,None,:],sin[:,None,:]
        first,second=x[...,:self.dim].float().chunk(2,-1)
        rotated=torch.cat((first*cos-second*sin,second*cos+first*sin),-1).to(x.dtype)
        return torch.cat((rotated,x[...,self.dim:]),-1)


@dataclass
class QSAState:
    packed: object = None  # [tokens, KV heads, K bytes/V bytes/K scales/V scales]
    raw_tail: object = None
    tail_positions: object = None
    compressed: object = None

    @property
    def length(self):return 0 if self.packed is None else self.packed.shape[0]


def select_tokens(queries,compressed,positions,ratio=4,budget=2048,positions_cpu=None,fast_prefix=True):
    """Sum of positive per-head scores; select blocks plus current open tail.

    CUDA sparse selection preserves the pinned deterministic top-k's order.
    CPU sorting is a mathematical reference and has a different tie/order policy.
    """
    if budget%ratio:raise ValueError('Budget must contain complete groups')
    host_positions=positions.tolist() if positions_cpu is None else positions_cpu
    maximum=max(host_positions,default=-1)
    if fast_prefix and queries.is_cuda and ratio==4 and budget==2048 and maximum<2051:
        if (maximum+1)//ratio>compressed.shape[0]:raise ValueError('Missing compressed history')
        from .qsa_indexer import full_prefix
        return full_prefix(positions)
    if queries.is_cuda and ratio==4 and budget==2048 and maximum>=2051:
        from .qsa_indexer import select
        return select(queries,compressed,positions)
    output=torch.full((queries.shape[0],budget+ratio-1),-1,dtype=torch.int32,device=queries.device)
    for row,position in enumerate(host_positions):
        visible=(position+1)//ratio
        if visible>compressed.shape[0]:raise ValueError('Missing compressed history')
        if visible<=budget//ratio:
            # The incumbent top-k fast path emits IDs in logical order when
            # the entire visible prefix fits, irrespective of score ranking.
            ids=torch.arange(visible*ratio,device=queries.device)
        elif visible:
            score=torch.einsum('hd,gd->hg',queries[row].float(),compressed[:visible,0].float())
            score=score.relu().sum(0)/math.sqrt(queries.shape[-1])
            chosen=torch.argsort(score,descending=True,stable=True)[:budget//ratio]
            ids=(chosen[:,None]*ratio+torch.arange(ratio,device=queries.device)).flatten()
        tail=torch.arange(visible*ratio,position+1,device=queries.device)
        ids=torch.cat((ids,tail))
        output[row,:ids.numel()]=ids.to(torch.int32)
    return output


def pack_kv(keys,values):
    from flashinfer.quantization.fp4_quantization import nvfp4_kv_quantize
    shape=keys.shape
    scale=torch.ones(1,device=keys.device,dtype=torch.float32)
    k,ks=nvfp4_kv_quantize(keys.contiguous().reshape(-1,shape[-1]),scale)
    v,vs=nvfp4_kv_quantize(values.contiguous().reshape(-1,shape[-1]),scale)
    return torch.cat((k,v,ks,vs),-1).reshape(*shape[:-1],-1)


def unpack_kv(packed,dim):
    shape=packed.shape[:-1]
    flat=packed.reshape(-1,packed.shape[-1])
    k,v,ks,vs=flat.split((dim//2,dim//2,dim//16,dim//16),-1)
    scale=torch.ones((),device=packed.device)
    k=nvfp4_dequant(k,ks,scale).half().reshape(*shape,dim)
    v=nvfp4_dequant(v,vs,scale).half().reshape(*shape,dim)
    return k,v


def sparse_attention(query,packed,indices):
    """Match FP16 operands, unnormalized probability rounding and split merge."""
    packed=materialize_kv(packed)
    rows,heads,dim=query.shape
    kv_heads=packed.shape[1]
    group=heads//kv_heads
    base=rows*kv_heads
    if base<=4:block,target=16,64
    elif base<32:block,target=16,32
    elif base<=256:block,target=64,8
    elif base<=512:block,target=64,4
    else:block,target=64,1
    tiles=(indices.shape[-1]+block-1)//block
    splits=min(1<<(tiles.bit_length()-1),target)
    keys,values=unpack_kv(packed,dim)
    output=torch.empty_like(query)
    for row in range(rows):
        for head in range(kv_heads):
            q=query[row,head*group:(head+1)*group].half().float()
            partial,lse=[],[]
            for split in range(splits):
                maximum=torch.full((group,),-1e20,device=query.device)
                total=torch.zeros(group,device=query.device)
                accumulator=torch.zeros(group,dim,device=query.device)
                for tile in range(split*tiles//splits,(split+1)*tiles//splits):
                    ids=indices[row,tile*block:(tile+1)*block].long()
                    ids=ids[ids>=0]
                    if not ids.numel():continue
                    score=(q@keys[ids,head].float().T)*(dim**-.5*math.log2(math.e))
                    next_max=torch.maximum(maximum,score.max(-1).values)
                    alpha=torch.exp2(maximum-next_max)
                    probability=torch.exp2(score-next_max[:,None])
                    accumulator=accumulator*alpha[:,None]+probability.half().float()@values[ids,head].float()
                    total=total*alpha+probability.sum(-1)
                    maximum=next_max
                partial.append(accumulator/total.clamp_min(1e-20)[:,None])
                lse.append(torch.where(total>0,maximum+torch.log2(total.clamp_min(1e-20)),-torch.inf))
            lse=torch.stack(lse)
            weights=torch.exp2(lse-lse.max(0).values)
            weights=weights/weights.sum(0)
            merged=(torch.stack(partial)*weights[:,:,None]).sum(0)
            output[row,head*group:(head+1)*group]=merged.to(query.dtype)
    return output


class QSARef(nn.Module):
    def __init__(self,store,prefix,config,activation_backend='torch',attention_backend='torch',kv_backend='contiguous'):
        super().__init__()
        self.config=config; self.activation_backend=activation_backend
        if attention_backend not in ('torch', 'triton'):
            raise ValueError('Unknown QSA attention backend')
        self.attention_backend=attention_backend
        if kv_backend not in ('contiguous','paged','auto'):raise ValueError('Unknown KV storage backend')
        self.kv_backend=kv_backend
        self.heads=config['num_attention_heads']; self.kv_heads=config['num_key_value_heads']
        self.dim=config['head_dim']; self.eps=config['rms_norm_eps']
        def get(suffix):return store.get(prefix+'.'+suffix)
        # Original checkpoint stores separate Q+gate, K and V projections.
        quantized=prefix+'.q_proj.weight_scale_inv' in store.weight_map
        weights={'qkv':torch.cat((get('q_proj.weight'),get('k_proj.weight'),get('v_proj.weight'))),
            'qkv_scale':torch.cat((get('q_proj.weight_scale_inv'),get('k_proj.weight_scale_inv'),get('v_proj.weight_scale_inv'))) if quantized else None,
            'q_norm':get('q_norm.weight'),'k_norm':get('k_norm.weight'),
            'out':get('o_proj.weight'),'out_scale':get('o_proj.weight_scale_inv') if quantized else None,
            'index_qk':get('indexer.index_qk_proj.weight'),
            'index_qnorm':get('indexer.q_layernorm.weight'),'index_knorm':get('indexer.k_layernorm.weight')}
        for name,tensor in weights.items():self.register_buffer(name,tensor)
        self.rope=TextRoPE(config['rope_parameters'],self.dim,self.qkv.device)

    def project(self,x,positions):
        projected=F.linear(x,self.qkv) if self.qkv_scale is None else fp8_linear(x,self.qkv,self.qkv_scale,self.activation_backend)
        return self.project_core(projected,positions)

    def project_core(self,projected,positions):
        if self.attention_backend=='triton':
            from .qsa_triton import project_qkv
            return project_qkv(projected,self.q_norm,self.k_norm,self.rope.cache_rows(positions),self.eps)
        qgate,k,v=projected.split((self.heads*2*self.dim,self.kv_heads*self.dim,self.kv_heads*self.dim),-1)
        q,gate=qgate.reshape(-1,self.heads,2*self.dim).chunk(2,-1)
        q=self.rope(gemma_norm(q,self.q_norm,self.eps),positions)
        k=self.rope(gemma_norm(k.reshape(-1,self.kv_heads,self.dim),self.k_norm,self.eps),positions)
        return q,k,v.reshape(-1,self.kv_heads,self.dim),gate.reshape(projected.shape[0],-1)

    def index(self,x,positions,state,position_start=None,return_prefixes=False,prefill_chunk=False):
        c=self.config; heads=c['indexer_n_heads']; dim=c['indexer_head_dim']; ratio=c['indexer_compress_ratio']
        # The BF16 indexer chooses a different GEMM for two rows. Its tiny
        # decode projection stays rowwise so compressed-history bytes remain
        # identical to ordinary decoding, including a rejected draft prefix.
        from .row_linear import per_row_linear
        projected=(per_row_linear(x,self.index_qk)
                   if per_row(x.shape[0]) and state.length>0 and not prefill_chunk else F.linear(x,self.index_qk))
        q,raw=projected.split((heads*dim,dim),-1)
        q=q.reshape(-1,heads,dim)
        if position_start is None or (position_start+x.shape[0])//ratio>c['indexer_budget']//ratio:
            if self.attention_backend=='triton':
                from .qsa_indexer import norm_rope
                q=norm_rope(q,self.index_qnorm,self.rope.cache_rows(positions),self.eps,
                            self.rope.parameters.get('mrope_section'))
            else:q=self.rope(gemma_norm(q,self.index_qnorm,self.eps),positions)
        # The all-prefix selection branch never reads query values. Retain
        # the identical joint projection and compressed-key history, while
        # avoiding unused query norm/RoPE launches for these short requests.
        raw=raw.reshape(-1,1,dim)
        positions=self.rope.canonical(positions)
        if state.raw_tail is not None:
            raw=torch.cat((state.raw_tail,raw)); positions_all=torch.cat((state.tail_positions,positions))
        else:positions_all=positions
        complete=raw.shape[0]//ratio
        compressed=state.compressed
        if complete:
            pooled_positions=positions_all[:complete*ratio:ratio]
            if self.attention_backend=='triton':
                from .qsa_indexer import compress
                pooled=compress(raw[:complete*ratio],self.index_knorm,self.rope.cache_rows(pooled_positions),self.eps,
                                self.rope.parameters.get('mrope_section'))
            else:
                pooled=raw[:complete*ratio].float().reshape(complete,ratio,1,dim).mean(1).to(x.dtype)
                pooled=self.rope(gemma_norm(pooled,self.index_knorm,self.eps),pooled_positions)
            compressed=pooled if compressed is None else torch.cat((compressed,pooled))
        if compressed is None:compressed=raw.new_empty(0,1,dim)
        host_positions=range(position_start,position_start+x.shape[0]) if position_start is not None else None
        ids=select_tokens(q,compressed,positions,ratio,c['indexer_budget'],host_positions,
                          getattr(self,'fast_prefix',True))
        final=QSAState(state.packed,raw[complete*ratio:].contiguous(),positions_all[complete*ratio:].contiguous(),compressed)
        if return_prefixes:
            prefixes=[]
            old_groups=state.length//ratio
            old_tail=state.length%ratio
            for rows in range(1,x.shape[0]+1):
                length=state.length+rows
                consumed=(length//ratio-old_groups)*ratio
                end=old_tail+rows
                prefixes.append(QSAState(state.packed,raw[consumed:end].contiguous(),
                    positions_all[consumed:end].contiguous(),compressed[:length//ratio]))
            return ids,final,prefixes
        return ids,final

    def forward(self,x,positions,state=None,position_start=None,return_prefixes=False,prefill_chunk=False):
        state=state or QSAState()
        if return_prefixes and x.shape[0] not in VERIFY_WIDTHS:
            raise ValueError('Prefix capture requires one to three tokens')
        previous_length=state.length
        positions=self.rope.canonical(positions)
        if position_start is None:
            expected=torch.arange(state.length,state.length+x.shape[0],device=positions.device)
            if not torch.equal(positions,expected):raise ValueError('QSA requires contiguous text positions for a single sequence')
        elif not isinstance(position_start,int) or position_start!=state.length or positions.shape!=(x.shape[0],):
            raise ValueError('Invalid engine-owned contiguous text position metadata')
        # The engine constructs positions from this same CPU start. External
        # tensor-only callers retain the explicit GPU value check above.
        q,k,v,gate=self.project(x,positions)
        indexed=self.index(x,positions,state,position_start,return_prefixes,prefill_chunk)
        indices,state=indexed[:2]
        packed=pack_kv(k,v)
        state.packed=append_kv(state.packed,packed,self.kv_backend,
            prefill_chunk or previous_length==0 or x.shape[0]>VERIFY_ROWS)
        if self.attention_backend == 'triton':
            from .qsa_triton import sparse_attention as kernel_attention
            core=kernel_attention(q.contiguous(),state.packed,indices,
                prefill_mode=getattr(self,'prefill_mode','packed'),verify=return_prefixes)
        else:
            core=sparse_attention(q,state.packed,indices)
        gated=core.flatten(1)*torch.sigmoid(gate)
        output=F.linear(gated,self.out) if self.out_scale is None else fp8_linear(gated,self.out,self.out_scale,self.activation_backend)
        result=(output,state)
        if return_prefixes:
            for rows,prefix in enumerate(indexed[2],1):
                prefix.packed=state.packed[:previous_length+rows]
            return (*result,indexed[2])
        return result
